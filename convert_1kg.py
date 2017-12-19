"""
Example script for importing 1000 Genomes data using the HapMap
genetic map.
"""
import argparse
import subprocess
import multiprocessing
import os

import numpy as np
import msprime
import tsinfer
import attr
import cyvcf2
import tqdm

import zarr
import bsddb3

@attr.s()
class Variant(object):
    position = attr.ib(None)
    genotypes = attr.ib(None)


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
            if row.is_snp and len(row.ALT) == 1:
                # Fill in a with genotypes.
                bases = row.gt_bases
                for j in range(num_diploids):
                    a[2 * j] = bases[j][0] != ancestral_state
                    a[2 * j + 1] = bases[j][2] != ancestral_state
                yield Variant(position=row.POS, genotypes=a)
    vcf.close()

def convert(vcf_file, genetic_map_file, output_file, max_variants=None, show_progress=False):
    genetic_map = msprime.RecombinationMap.read_hapmap(genetic_map_file)
    map_length = genetic_map.get_length()

    if max_variants is None:
        max_variants = 2**32  # Arbitrary, but > defined max for VCF

    last_physical_pos = 0
    last_genetic_pos = 0
    positions = []
    genotypes = []
    recombination_rates = []
    for index, variant in enumerate(variants(vcf_file, show_progress)):
        physical_pos = variant.position
        if index >= max_variants or physical_pos >= map_length:
            break
        genetic_pos = genetic_map.physical_to_genetic(variant.position)
        physical_dist = physical_pos - last_physical_pos
        genetic_dist = genetic_pos - last_genetic_pos
        scaled_recomb_rate = 0
        if genetic_dist > 0:
            scaled_recomb_rate = physical_dist / genetic_dist
        recombination_rates.append(scaled_recomb_rate)
        genotypes.append(variant.genotypes)
        positions.append(physical_pos)
        last_physical_pos = physical_pos
        last_genetic_pos = genetic_pos

    G = np.array(genotypes, dtype=np.uint8)

    # This is crud, need to abstract this away from the user.
    if os.path.exists(output_file):
        os.unlink(output_file)
    input_hdf5 = zarr.DBMStore(output_file, open=bsddb3.btopen)
    root = zarr.group(store=input_hdf5, overwrite=True)
    tsinfer.InputFile.build(
        root, genotypes=G, position=positions, recombination_rate=recombination_rates)
    input_hdf5.close()
    print("Wrote", output_file)


def worker(t):
    vcf, genetic_map, output, max_variants = t
    print("Converting", vcf)
    convert(vcf, genetic_map, output, max_variants)


def main():
    parser = argparse.ArgumentParser(
        description="Script to convert VCF files into tsinfer input.")
    parser.add_argument(
        "vcf_pattern", help="The input VCF files file pattern.")
    parser.add_argument(
        "genetic_map_pattern", help="The input genetic maps in HapMap format file pattern")
    parser.add_argument(
        "output_file_pattern", help="The tsinfer output file pattern to write to.")
    parser.add_argument(
        "-n", "--max-variants", default=None, type=int,
        help="Keep only the first n variants")
    parser.add_argument(
        "--start", default=1, type=int, help="The first autosome")
    parser.add_argument(
        "--stop", default=22, type=int, help="The last autosome")

    args = parser.parse_args()
    chromosomes = list(range(args.start, args.stop + 1))

    # Build the file lists for the 22 autosomes.
    vcf_files = [args.vcf_pattern.format(j) for j in chromosomes]
    genetic_map_files = [args.genetic_map_pattern.format(j) for j in chromosomes]
    output_files = [args.output_file_pattern.format(j) for j in chromosomes]
    max_variants = [args.max_variants for _ in chromosomes]

    for vcf_file in vcf_files:
        if not os.path.exists(vcf_file):
            raise ValueError("{} does not exist".format(vcf_file))

    for genetic_map_file in genetic_map_files:
        if not os.path.exists(genetic_map_file):
            raise ValueError("{} does not exist".format(genetic_map_file))

    work = reversed(list(zip(vcf_files, genetic_map_files, output_files, max_variants)))
    with multiprocessing.Pool(10) as pool:
        pool.map(worker, work)
    # for t in work:
    #     worker(t)



if __name__ == "__main__":
    main()
