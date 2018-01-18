"""
As of Jan 9th 2018 the tsinfer.match_samples stage either stalls (method='P') or encounters an assertion failure (method='C':

Assertion failed: (max_L > 0), function ancestor_matcher_renormalise_likelihoods, file lib/ancestor_matcher.c, line 394.
)
"""
import os
import sys
import random

import numpy as np
import zarr

# use the local copy of msprime in preference to the global one
import msprime
import tsinfer

l=100000000
recombination_rate = rho = 0.000000000001
recombination_map = msprime.RecombinationMap.uniform_map(l, rho, l)
rng1 = random.Random(808131929)
sim_seed = rng1.randint(1, 2**31)
ts = msprime.simulate(
    4, 5000, recombination_map=recombination_map, mutation_rate=0.00000001,
    random_seed=sim_seed, model="smc_prime")

positions = pos = np.array([v.position for v in ts.variants()])
S = np.zeros((ts.sample_size, ts.num_mutations), dtype="u1")
for variant in ts.variants():
    S[:,variant.index] = variant.genotypes

G = S.astype(np.uint8).T

#Create the ancestors
input_root = zarr.group()
tsinfer.InputFile.build(
    input_root, genotypes=G,
    # genotype_qualities=tsinfer.proba_to_phred(error_probability),
    position=positions,
    recombination_rate=rho, sequence_length=ts.sequence_length,
    compress=False)
ancestors_root = zarr.group()


#tsinfer.extract_ancestors(ts, ancestors_root)
tsinfer.build_simulated_ancestors(input_root, ancestors_root, ts)

ancestors_ts = tsinfer.match_ancestors(input_root, ancestors_root)
assert ancestors_ts.sequence_length == ts.num_sites
inferred_ts = tsinfer.match_samples(
    input_root, ancestors_ts, method="C",
    simplify=False)

print("inferred num_edges = ", inferred_ts.num_edges)
