#!/usr/bin/env python3
description="""
Take a vcf file with ancestral allele information (e.g. from 1000 genomes phase 1, such as
ALL.chr22.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz
and convert it to the huge samples _times_ sites array used in tsinfer. Store this
together with the variant positions and the sample names in hdf5 output files, one per
chromosome.

The file can be run 

This script deals with samples with different ploidies (assuming the data is phased) by
adding '#a', '#b' etc to the sample names. For haploid samples, the '#x' suffix is removed.

Since we require complete data, we need to prune out mising values. At the moment we do
this by pruning samples with missing values, but we could equally do so by pruning sites
instead, or perhaps optimally remove both sites and samples (see 
https://stackoverflow.com/questions/48355644/optimally-remove-rows-or-columns-in-numpy-to-remove-missing-data)
"""

"""
NB - a slightly odd observation: all the alleles for which the ancestral state is not present in the dataset are SNPs

read using
with h5py.File(filename, 'r') as f:
    print(f['data']['variants'][()])
"""
import collections
import argparse
import string
import sys
import os


import numpy as np
import zarr
import bsddb3
import pysam
import tsinfer

UNKNOWN_ALLELE = -1

parser = argparse.ArgumentParser(description=description)
parser.add_argument('infile', 
                    help='a vcf or vcf.gz file')
parser.add_argument('outfile',
                    help='the output file prefix: data will be saved in hdf5 format as PREFIXchr.hdf5')
parser.add_argument('-r','--recombination-rate', type=float, default=1.0,
                    help="The genome-wide recombination rate")
parser.add_argument('-m','--genetic-map-pattern',
                    help="If you know that recombination rate varies," \
        " specify the filename for the genetic maps in HapMap format." \
        " Any {} characters will be replaced with the choromosome number from the vcf file e.g." \
        "genetic_map_GRCh37_chr{}.txt, and this will replace any --recombination-rate specified previously")
parser.add_argument('--only_use_n_variants', '-n', type=int, default=None,
                    help='For testing purposes, only use the first n variants')

args = parser.parse_args()

vcf_in = pysam.VariantFile(args.infile)

def process_variant(rec, rows, max_ploidy):
    """
    Return the true position of this variant, or None if we wish to omit
    Also fills out the site_data array.
    If the first returned item is None, we have failed to read useful data
    """
    #restrict to cases where ancestral state contains only letters ATCG
    # i.e. where the ancestral state is certain (see ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/supporting/ancestral_alignments/human_ancestor_GRCh37_e59.README
    if "AA" in rec.info and all(letter in "ATCG" for letter in rec.info["AA"]):
        #only use biallelic variants 
        if len(rec.alleles) == 2:
            if rec.info["AA"] not in rec.alleles:
                print("Ancestral state {} not in allele list ({}) for position {}".format(\
                    rec.info["AA"], rec.alleles, rec.pos))
                pass
            else:
                omit=False
                #check for duplicate positions, often caused e.g. by C/CAT as opposed to -/AT
                #if the first x letters are the same, we have an intermediate position, e.g.
                #if C is at position 123, we can place the indel AT at position 123.5
                allele_start = 0
                for i in range(min([len(a) for a in rec.alleles])):
                    if len(set([a[i] for a in rec.alleles])) > 1:
                        #alleles differ here
                        break
                    allele_start += 1
                if allele_start != 0:
                    pos = rec.pos+allele_start
                    if len(set([len(a) for a in rec.alleles])) == 1:
                        #all alleles are the same length, => this is not an indel
                        print("The variants at {} share sequence, but are not an indel: {}".format({rec.id:rec.pos}, rec.alleles))
                    else:
                        pos-=0.5
                    if allele_start > 1:
                        print("The variants at {} share more than one starting letter: {}".format({rec.id:rec.pos}, rec.alleles))
                        
                    #print("Starting allele at an incremented position ({} not {}) for {} (alleles:{})".format(
                    #    pos, rec.pos, rec.id, rec.alleles))
                else:
                    allele_start=0
                    pos = rec.pos

                site_data = UNKNOWN_ALLELE * np.ones((len(rows),), dtype="i1")
                for label, samp in rec.samples.items():
                    if len(samp.alleles) == 2:
                        print("2 alleles on chr {} sample {} at pos {}".format(rec.chrom, label, rec.pos))
                    for i in range(min(max_ploidy, len(samp.alleles))):
                        #print("allele {} (pos {}, sample {}, ancestral state {}, alleles {})".format( \
                        #    rec.alleles[i], rec.pos, label+suffix, rec.info["AA"], rec.alleles))
                        if samp.alleles[i] not in rec.alleles:
                            continue
                        suffix = string.ascii_lowercase[i]
                        site_data[rows[label+'#'+suffix]] = samp.alleles[i][allele_start:]!=rec.info["AA"][allele_start:]
                return site_data, rec.chrom, pos
    return None, rec.chrom, rec.pos

allele_count = {}
rows, row = {}, 0
max_ploidy = 2
suffixes = string.ascii_lowercase[:max_ploidy] #use 'a' and 'b' for ploidy suffix

for sample_name in vcf_in.header.samples:
    #assume each sample can have a maximum of 2 variants (maternal & paternal)
    #i.e. max is diploid. For haploid samples, we just use the suffix 'a'
    #and hack to remove it afterwards
    for suffix in suffixes:
        rows[sample_name+'#'+suffix]=row
        row+=1
extend_amount = 10000 #numpy storage array will be extended by this much at a time
chromosomes = {} #most data stored here
output_freq_variants = 1e3 #output status after multiples of this many variants read


for j, variant in enumerate(vcf_in.fetch()):
    if j==args.only_use_n_variants:
        break
    #keep track of allele numbers
    allele_count[len(variant.alleles)] = allele_count.get(len(variant.alleles),0) + 1

    locus_data, c, position = process_variant(variant, rows, max_ploidy)
    #have we stored this chromosome yet
    if c not in chromosomes:
        chromosomes[c] = {
            'position':collections.OrderedDict(),
            'previous_position': -1,
            'sites_by_samples': UNKNOWN_ALLELE * np.ones((extend_amount, len(rows)), dtype="i1")}
    #check if we now need more storage
    if chromosomes[c]['sites_by_samples'].shape[0] <= len(chromosomes[c]['position']):
        chromosomes[c]['sites_by_samples'] = np.append(chromosomes[c]['sites_by_samples'],
            UNKNOWN_ALLELE * np.ones((extend_amount, len(rows)), dtype="i1"), axis=0)
    if locus_data is not None:
        if position < chromosomes[c]['previous_position']:
            print("A position is out of order. We require a vcf file with all positions in strictly increasing order")
        elif position in chromosomes[c]['position']:
            print("Trying to store more than one set of variants at position {} of chr {}. ".format(
                position, c))
            print("Previous was {}, this is {}.".format(
                chromosomes[c]['position'][position], {variant.id:variant.alleles}))
            chromosomes[c]['position'].popitem()
            print("Deleting first one")
        else:
            chromosomes[c]['sites_by_samples'][len(chromosomes[c]['position']),:]=locus_data
            chromosomes[c]['position'][position]={variant.id: variant.alleles}
            chromosomes[c]['previous_position']=position
    
    if j % output_freq_variants == 0:
        print("{} variants read ({} with ancestral alleles saved). Base position {} Mb chr {} (alleles per site: {})".format(
            j+1, len(chromosomes[c]['position']), variant.pos/1e6, c, 
            [(k, allele_count[k]) for k in sorted(allele_count.keys())]), 
            flush=True)

#Finished reading

for c, dat in chromosomes.items():
    print("Processing chromosome {}".format(c))

    print("Doing some internal tidying:")
    #check for samples with entirely missing data (e.g. if haploid)
    use_samples = np.ones((dat['sites_by_samples'].shape[1],), np.bool)
    for colnum in range(dat['sites_by_samples'].shape[1]):
        if np.all(dat['sites_by_samples'][:,colnum] == UNKNOWN_ALLELE):
            use_samples[colnum]=False
    sites_by_samples = dat['sites_by_samples'][:,use_samples]
    reduced_rows = [name for i,name in enumerate(sorted(rows, key=rows.get)) if use_samples[i]]
    #simplify the names for haploids (remove e.g. terminal 'a' from the key 'XXXXa'
    # if there is not an XXXXb in the list)
    reduced_rows = [n if any((n[:-1]+suffix) in reduced_rows for suffix in suffixes if suffix != n[-1]) else n[:-2] for n in reduced_rows]
    print(" (removed {}/{} unused sample slots - if this is haploid, half should be removed)".format(
        sum(~use_samples), use_samples.shape[0]))

    #remove overly extended parts of the sites_by_samples array (all UNKNOWN_ALLELE for a site)
    use_sites = np.zeros((sites_by_samples.shape[0],), np.bool)
    for rownum in range(len(dat['position'])):
        if np.any(sites_by_samples[rownum] != UNKNOWN_ALLELE):
            use_sites[rownum] = True
    sites_by_samples = sites_by_samples[use_sites,:]
    print(" (pruned excess array allocation (now at {} sites)".format(sites_by_samples.shape[0]))
    
    #check for samples with partial missing data
    use_samples = np.ones((sites_by_samples.shape[1],), np.bool)
    for colnum in range(sites_by_samples.shape[1]):
        if np.any(sites_by_samples[:,colnum] == UNKNOWN_ALLELE):
            use_samples[colnum]=False
    sites_by_samples = sites_by_samples[:,use_samples]
    reduced_rows = [name for i,name in enumerate(reduced_rows) if use_samples[i]]
    if sum(~use_samples):
        print("Removed {}/{} samples which have incomplete data".format(
            sum(~use_samples), use_samples.shape[0]))

    recombination_rates = np.zeros_like(len(dat['position'])) + args.recombination_rate
    if args.genetic_map_pattern is None:
        print("Using consant recombination rate of {}".format(args.recombination_rate))
    else:
        if "{}" in args.genetic_map_pattern:
            genetic_map_file = args.genetic_map_pattern.format(c)
        else:
            genetic_map_file = args.genetic_map_pattern
        try:
            genetic_map = msprime.RecombinationMap.read_hapmap(genetic_map_file)
            print("Using recombination map {}".format(genetic_map_file))
            map_length = genetic_map.get_length()
            last_physical_pos = 0
            last_genetic_pos = 0
            for i, physical_pos in enumerate(dat['position'].keys()):
                #if physical_pos >= map_length:
                #    break
                genetic_pos = genetic_map.physical_to_genetic(physical_pos)
                physical_dist = physical_pos - last_physical_pos
                genetic_dist = genetic_pos - last_genetic_pos
                scaled_recomb_rate = 0
                if genetic_dist > 0:
                    scaled_recomb_rate = physical_dist / genetic_dist
                recombination_rates[i]=scaled_recomb_rate
        except FileNotFoundError:
            print("Genetic map file {} not found, defaulting to constant recombination rate of {}".format(
                genetic_map_file, args.recombination_rate))


    output_file = args.outfile + str(c) + ".tsinf"
    if os.path.exists(output_file):
        os.unlink(output_file)
    input_hdf5 = zarr.DBMStore(output_file, open=bsddb3.btopen)
    root = zarr.group(store=input_hdf5, overwrite=True)
    tsinfer.InputFile.build(
        root, 
        genotypes=sites_by_samples,
        position=list(dat['position'].keys()),
        recombination_rate=recombination_rates)
    #sample_names=[s.encode() for s in reduced_rows]
    input_hdf5.close()
    print("Saved {} biallelic loci for {} samples into {}".format(len(dat['position']), len(reduced_rows), output_file))


"""
Then do something like
    input_hdf5 = zarr.DBMStore(output_file, open=bsddb3.btopen)
    input_root = zarr.group(store=input_hdf5)
    ancestors_root = zarr.group()
    tsinfer.build_ancestors(
        input_root, ancestors_root, method=method, chunk_size=16, compress=False)
    ancestors_ts = tsinfer.match_ancestors(
        input_root, ancestors_root, method=method, path_compression=path_compression)
    full_inferred_ts = tsinfer.match_samples(
        input_root, ancestors_ts, method=method, path_compression=path_compression,
        simplify=simplify)
    full_inferred_ts.dump(output_file + '.ts')
"""