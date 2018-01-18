"""
With rng1 seeded using random.Random(10) we infer 2 extra trees which we shouldn't do
"""
import os
import sys
import random
import collections

import numpy as np
import zarr

# use the local copy of msprime in preference to the global one
sys.path.insert(1,os.path.join(sys.path[0],'..','..','msprime'))
sys.path.insert(1,os.path.join(sys.path[0],'..','..','tsinfer'))
import msprime
import tsinfer

l=6000000
recombination_rate = rho = 5E-11
recombination_map = msprime.RecombinationMap.uniform_map(l, rho, l)
rng1 = random.Random(1181207362)
sim_seed = rng1.randint(1, 2**31)
ts = msprime.simulate(
    4, 5000, recombination_map=recombination_map, mutation_rate=5E-09,
    random_seed=sim_seed, model="smc_prime")

#remove singletons
sites = msprime.SiteTable()
mutations = msprime.MutationTable()
for variant in ts.variants():
    if np.sum(variant.genotypes) > 1:
        site_id = sites.add_row(
            position=variant.site.position,
            ancestral_state=variant.site.ancestral_state)
        for mutation in variant.site.mutations:
            assert mutation.parent == -1  # No back mutations
            mutations.add_row(
                site=site_id, node=mutation.node, derived_state=mutation.derived_state)

tables = ts.dump_tables()
ts = msprime.load_tables(
    nodes=tables.nodes, edges=tables.edges, sites=sites, mutations=mutations)

nodes = collections.defaultdict(list)
for m in ts.mutations():
    nodes[m.node].append(m.position)
#print(nodes)
print("numbers of mutations above each node: ", {k: len(nodes[k]) for k in sorted(nodes.keys())})
for t in ts.trees():
    print(t.get_interval())
    print(t.draw(format="unicode"))

positions = pos = np.array([v.position for v in ts.variants()])
S = np.zeros((ts.sample_size, ts.num_mutations), dtype="u1")
for variant in ts.variants():
    S[:,variant.index] = variant.genotypes

G = S.astype(np.uint8).T

G = np.array([
    [0,1,1,1,1,0],
    [1,1,0,0,1,1],
    [0,0,1,0,0,0],
    [1,1,1,1,1,1]], dtype=np.uint8).T
positions = pos = list(range(0,6))
    
#Create the ancestors
input_root = zarr.group()
tsinfer.InputFile.build(
    input_root, genotypes=G,
    # genotype_qualities=tsinfer.proba_to_phred(error_probability),
    position=positions,
    recombination_rate=rho, sequence_length=ts.sequence_length,
    compress=False)
ancestors_root = zarr.group()


tsinfer.build_ancestors(input_root, ancestors_root, method="P", chunk_size=16, compress=False) 
#tsinfer.build_simulated_ancestors(input_root, ancestors_root, ts, guess_unknown=False)

A = ancestors_root["ancestors/haplotypes"][:]

print("ANCESTORS")
# print(A.astype(np.int8))
for j, a in enumerate(A):
    s = "".join(str(x) if x < 255 else "*" for x in a)
    print(j, "\t", s)

print("samples")
for j, s in enumerate(ts.haplotypes()):
    print(A.shape[0] + j, "\t", s)

ancestors_ts = tsinfer.match_ancestors(input_root, ancestors_root)
#assert ancestors_ts.sequence_length == ts.num_sites

print("========")
print("INFERRED ANCESTRAL PATHS")
print(ancestors_ts.tables.edges)


inferred_ts = tsinfer.match_samples(
    input_root, ancestors_ts, method="C",
    simplify=False) 

print("num_edges: original: {}; inferred: {}".format(ts.num_edges,inferred_ts.num_edges))

for t in inferred_ts.trees():
    print(t.get_interval())
    print(t.draw(format="unicode"))

