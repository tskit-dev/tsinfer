import os
import msprime
import tqdm
import sys
sys.path.insert(0, os.path.abspath('..'))
import tsinfer

if False:
    ts = msprime.simulate(
        sample_size=10000, Ne=10**4, recombination_rate=1e-8, mutation_rate=1e-8,
        length=10*10**6, random_seed=42)
    ts.dump("simulation-source.trees")
    print("Simulation done:", ts.num_trees, "trees and", ts.num_sites)

    progress = tqdm.tqdm(total=ts.num_sites)
    sample_data = tsinfer.SampleData.initialise(
        num_samples=ts.num_samples, sequence_length=ts.sequence_length,
        path="simulation.samples", num_flush_threads=2)
    for var in ts.variants():
        sample_data.add_site(var.site.position, var.alleles, var.genotypes)
        progress.update()
    progress.close()
    sample_data.finalise()
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
