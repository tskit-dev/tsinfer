"""
Simple script to convert input data into HDF5 format so that
we can feed it into the C development CLI.
"""

import numpy as np
import tsinfer
import h5py
import sys


def main(infile, outfile):
    sample_data = tsinfer.SampleData.load(infile)
    print(sample_data)
    shape = (sample_data.num_inference_sites, sample_data.num_samples)
    G = np.empty(shape, dtype=np.uint8)
    for j, (_, genotypes) in enumerate(sample_data.genotypes(inference_sites=True)):
        G[j] = genotypes
    with h5py.File(outfile, "w") as root:
        root["haplotypes"] = G.T


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
