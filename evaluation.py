"""
Script for statistically evaluating various aspects of tsinfer performance.
"""

import numpy as np
import dendropy

import tsinfer
import msprime

def infer_from_simulation(ts, recombination_rate, sample_error=0):
    samples = np.zeros((ts.num_samples, ts.num_sites), dtype=np.int8)
    for variant in ts.variants():
        samples[:, variant.index] = variant.genotypes
    positions = [mut.position for mut in ts.mutations()]
    return tsinfer.infer(
        samples=samples, positions=positions, sequence_length=ts.sequence_length,
        recombination_rate=recombination_rate, sample_error=sample_error)


def get_mean_rf_distance(ts1, ts2):
    """
    Returns the mean distance between the trees in the specified tree sequences.
    """
    assert ts1.sample_size == ts2.sample_size
    assert ts1.sequence_length == ts2.sequence_length
    trees1 = []
    intervals1 = []
    trees2 = []
    intervals2 = []
    tns = dendropy.TaxonNamespace()
    for t in ts1.trees():
        dt = dendropy.Tree.get(data=t.newick(), schema="newick", taxon_namespace=tns)
        trees1.append(dt)
        intervals1.append(t.interval)
    assert len(trees1) == ts1.num_trees
    for t in ts2.trees():
        dt = dendropy.Tree.get(data=t.newick(), schema="newick", taxon_namespace=tns)
        trees2.append(dt)
        intervals2.append(t.interval)
    assert len(trees2) == ts2.num_trees
    j1 = 0
    j2 = 0
    total_distance = 0
    total_metric = 0
    # I haven't tested this algorithm thoroughly, so there might be corner cases
    # not handled correctly. However, the total_distance assert below should
    # catch the problem if it occurs.
    while j1 < len(trees1) and j2 < len(trees2):
        # Each iteration of this loop considers one overlapping interval and
        # increments the counters.
        l1, r1 = intervals1[j1]
        l2, r2 = intervals2[j2]
        l = max(l1, l2)
        r = min(r1, r2)
        rf_distance = dendropy.calculate.treecompare.symmetric_difference(
                trees1[j1], trees2[j2])
        total_metric += rf_distance * (r - l)
        total_distance += r - l
        if r1 <= r2:
            j1 += 1
        if r1 >= r2:
            j2 += 1
    # assert total_distance, ts1.sequence_length)
    return total_metric / total_distance


def check_basic_performance():

    num_samples = 10
    MB = 10**6
    seed = 12234
    ts_source = msprime.simulate(
        num_samples, length=1*MB, Ne=10**4, recombination_rate=1e-8, mutation_rate=1e-8,
        random_seed=seed)
    print("sim: n = ",
            ts_source.num_samples, ", m =", ts_source.num_sites,
            "num_trees = ", ts_source.num_trees)
    for exponent in range(1, 10):
        recombination_rate = 10**(-exponent)
        ts_inferred = infer_from_simulation(
            ts_source, recombination_rate=recombination_rate, sample_error=1e-15)
        rf = get_mean_rf_distance(ts_source, ts_inferred)
        print("recombination_rate = ", recombination_rate)
        print("mean rf = ", rf)
        print("num_trees = ", ts_inferred.num_trees / ts_source.num_trees)
        print("num_edges= ", ts_inferred.num_edges/ ts_source.num_edges)

    # print(ts_source.num_trees, ts_inferred.num_trees)
    # trees_source = ts_source.trees()
    # trees_inferred = ts_inferred.trees()
    # tree_source = next(trees_source)
    # tree_inferred = next(trees_inferred)

    # print(tree_source.draw(format="unicode"))
    # print(tree_inferred.draw(format="unicode"))



if __name__ == "__main__":
    check_basic_performance()
