import os
import msprime
import tqdm
import sys
sys.path.insert(0, os.path.abspath('..'))
import tsinfer

if True:
    ts = msprime.simulate(
        sample_size=10000, Ne=10**4, recombination_rate=1e-8, mutation_rate=1e-8,
        length=10*10**6, random_seed=42)
    ts.dump("simulation-source.trees")
    print("Simulation done:", ts.num_trees, "trees and", ts.num_sites)

    with tsinfer.SampleData(
            sequence_length=ts.sequence_length, path="simulation.samples",
            num_flush_threads=2) as samples:
        for var in tqdm.tqdm(ts.variants(), total=ts.num_sites):
            samples.add_site(var.site.position, var.genotypes, var.alleles)

else:
    source = msprime.load("simulation-source.trees")
    inferred = msprime.load("simulation.trees")

    subset = range(0, 6)
    source_subset = source.simplify(subset)
    inferred_subset = inferred.simplify(subset)

    tree = source_subset.first()
    print("True tree: interval=", tree.interval)
    print(tree.draw(format="unicode"))

    tree = inferred_subset.first()
    print("Inferred tree: interval=", tree.interval)
    print(tree.draw(format="unicode"))
