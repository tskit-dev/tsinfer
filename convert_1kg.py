"""
Example script for importing 1000 Genomes data.
"""
import argparse
import subprocess
import multiprocessing
import os
import shutil
import sys

import numpy as np
import msprime
import tsinfer
import attr
import cyvcf2
import tqdm


@attr.s()
class Site(object):
    position = attr.ib(None)
    alleles = attr.ib(None)
    genotypes = attr.ib(None)
    metadata = attr.ib({})


def filter_duplicates(vcf):
    """
    Returns the list of variants from this specified VCF with duplicate sites filtered
    out. If any site appears more than once, throw all variants away.
    """
    # TODO this had not been tested properly.
    row = next(vcf, None)
    bad_pos = -1
    for next_row in vcf:
        if bad_pos == -1 and next_row.POS != row.POS:
            yield row
        else:
            if bad_pos == -1:
                bad_pos = row.POS
            elif bad_pos != next_row.POS:
                bad_pos = -1
        row = next_row
    if row is not None and bad_pos != -1:
        yield row

def variants(vcf_path, show_progress=False):

    output = subprocess.check_output(["bcftools", "index", "--nrecords", vcf_path])
    num_rows = int(output)
    progress = tqdm.tqdm(total=num_rows, disable=not show_progress)

    vcf = cyvcf2.VCF(vcf_path)

    num_diploids = len(vcf.samples)
    num_samples = 2 * num_diploids
    j = 0
    for row in filter_duplicates(vcf):
        progress.update()
        ancestral_state = None
        try:
            aa = row.INFO["AA"]
            # Format = AA|REF|ALT|IndelType
            splits = aa.split("|")
            if len(splits) == 4 and len(splits[0]) == 1:
                base = splits[0].upper()
                if base in "ACTG":
                    ancestral_state = base
        except KeyError:
            pass
        if row.num_called == num_diploids and ancestral_state is not None:
            a = np.zeros(num_samples, dtype=np.uint8)
            all_alleles = set([ancestral_state])
            # Fill in a with genotypes.
            bases = np.array(row.gt_bases)
            for j in range(num_diploids):
                alleles = bases[j].split("|")
                for allele in alleles:
                    all_alleles.add(allele)
                a[2 * j] = alleles[0] != ancestral_state
                a[2 * j + 1] = alleles[1] != ancestral_state
            if len(all_alleles) == 2:
                all_alleles.remove(ancestral_state)
                alleles = [ancestral_state, all_alleles.pop()]
                metadata = {"ID": row.ID, "INFO": dict(row.INFO)}
                yield Site(
                    position=row.POS, alleles=alleles, genotypes=a, metadata=metadata)

    vcf.close()

def convert(vcf_file, output_file, max_variants=None, show_progress=False):

    if max_variants is None:
        max_variants = 2**32  # Arbitrary, but > defined max for VCF

    sample_data = tsinfer.SampleData.initialise(path=output_file, num_flush_threads=2)
    vcf = cyvcf2.VCF(vcf_file)
    for sample in vcf.samples:
        metadata = {"name": sample}
        sample_data.add_sample(metadata)
        sample_data.add_sample(metadata)
    vcf.close()

    for index, site in enumerate(variants(vcf_file, show_progress)):
        sample_data.add_site(site.position, site.alleles, site.genotypes, site.metadata)
        if index == max_variants:
            break
    sample_data.finalise(command=sys.argv[0], parameters=sys.argv[1:])


def worker(t):
    vcf, output, max_variants = t
    print("Converting", vcf)
    convert(vcf, output, max_variants)


def main():
    parser = argparse.ArgumentParser(
        description="Script to convert VCF files into tsinfer input.")
    parser.add_argument(
        "vcf_pattern", help="The input VCF files file pattern.")
    parser.add_argument(
        "output_file_pattern", help="The tsinfer output file pattern to write to.")
    parser.add_argument(
        "-n", "--max-variants", default=None, type=int,
        help="Keep only the first n variants")
    parser.add_argument(
        "--start", default=1, type=int, help="The first autosome")
    parser.add_argument(
        "--stop", default=22, type=int, help="The last autosome")
    parser.add_argument(
        "-P", "--processes", default=10, type=int, help="The number of worker processes")
    parser.add_argument(
        "-p", "--progress", action="store_true")

    args = parser.parse_args()
    # TODO Fix this up and make it an optional argument.
    chromosomes = list(range(args.start, args.stop + 1))

    # Build the file lists for the 22 autosomes.
    vcf_files = [args.vcf_pattern.format(j) for j in chromosomes]
    output_files = [args.output_file_pattern.format(j) for j in chromosomes]
    max_variants = [args.max_variants for _ in chromosomes]

    for vcf_file in vcf_files:
        if not os.path.exists(vcf_file):
            raise ValueError("{} does not exist".format(vcf_file))

#     work = reversed(list(zip(
#         vcf_files, genetic_map_files, output_files, max_variants)))
#     with multiprocessing.Pool(args.processes) as pool:
#         pool.map(worker, work)
#     # for t in work:
#     #     worker(t)

    convert(
        vcf_files[0], output_files[0], args.max_variants, show_progress=args.progress)


if __name__ == "__main__":
    main()
