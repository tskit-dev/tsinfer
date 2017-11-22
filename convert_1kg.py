"""
Example script for importing 1000 Genomes data using the HapMap
genetic map.
"""
import argparse
import subprocess
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


def variants(vcf_path):

    output = subprocess.check_output(["bcftools", "index", "--nrecords", vcf_path])
    num_rows = int(output)
    progress = tqdm.tqdm(total=num_rows)

    vcf = cyvcf2.VCF(vcf_path)

    num_diploids = len(vcf.samples)
    num_samples = 2 * num_diploids
    j = 0
    for row in vcf:
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


def main():
    parser = argparse.ArgumentParser(
        description="Script to convert a VCF + genetic map to tsinfer input.")
    parser.add_argument(
        "vcf", help="The input VCF file.")
    parser.add_argument(
        "genetic_map", help="The input genetic map in HapMap format..")
    parser.add_argument(
        "output_file", help="The tsinfer output file to write to.")
    parser.add_argument(
        "-n", "--max-variants", default=None, type=int,
        help="Keep only the first n variants")

    args = parser.parse_args()
    genetic_map = msprime.RecombinationMap.read_hapmap(args.genetic_map)
    map_length = genetic_map.get_length()

    max_variants = 2**32  # Arbitrary, but > defined max for VCF
    if args.max_variants is not None:
        max_variants = args.max_variants

    last_physical_pos = 0
    last_genetic_pos = 0
    positions = []
    genotypes = []
    recombination_rates = []
    for index, variant in enumerate(variants(args.vcf)):
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
    output = args.output_file
    if os.path.exists(output):
        os.unlink(output)
    input_hdf5 = zarr.DBMStore(output, open=bsddb3.btopen)
    root = zarr.group(store=input_hdf5, overwrite=True)
    tsinfer.InputFile.build(
        root, genotypes=G, position=positions, recombination_rate=recombination_rates)
    input_hdf5.close()


if __name__ == "__main__":
    main()
