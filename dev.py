
import random
import os
import h5py
import zarr
import sys
import pandas as pd
import daiquiri
#import bsddb3
import time
import scipy
import pickle
import collections
import itertools
import tqdm
import shutil
import pprint
import numpy as np
import json

import matplotlib as mp
# Force matplotlib to not use any Xwindows backend.
mp.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tsinfer
import msprime



def plot_breakpoints(ts, map_file, output_file):
    # Read in the recombination map using the read_hapmap engine,
    recomb_map = msprime.RecombinationMap.read_hapmap(map_file)

    # Now we get the positions and rates from the recombination
    # map and plot these using 500 bins.
    positions = np.array(recomb_map.get_positions()[1:])
    rates = np.array(recomb_map.get_rates()[1:])
    num_bins = 500
    v, bin_edges, _ = scipy.stats.binned_statistic(
        positions, rates, bins=num_bins)
    x = bin_edges[:-1][np.logical_not(np.isnan(v))]
    y = v[np.logical_not(np.isnan(v))]
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(x, y, color="blue", label="Recombination rate")
    ax1.set_ylabel("Recombination rate")
    ax1.set_xlabel("Chromosome position")

    # Now plot the density of breakpoints along the chromosome
    breakpoints = np.array(list(ts.breakpoints()))
    ax2 = ax1.twinx()
    v, bin_edges = np.histogram(breakpoints, num_bins, density=True)
    ax2.plot(bin_edges[:-1], v, color="green", label="Breakpoint density")
    ax2.set_ylabel("Breakpoint density")
    ax2.set_xlim(1.5e7, 5.3e7)
    plt.legend()
    fig.savefig(output_file)


def make_errors(v, p):
    """
    For each sample an error occurs with probability p. Errors are generated by
    sampling values from the stationary distribution, that is, if we have an
    allele frequency of f, a 1 is emitted with probability f and a
    0 with probability 1 - f. Thus, there is a possibility that an 'error'
    will in fact result in the same value.
    """
    w = np.copy(v)
    if p > 0:
        m = v.shape[0]
        frequency = np.sum(v) / m
        # Randomly choose samples with probability p
        samples = np.where(np.random.random(m) < p)[0]
        # Generate observations from the stationary distribution.
        errors = (np.random.random(samples.shape[0]) < frequency).astype(int)
        w[samples] = errors
    return w


def generate_samples(ts, error_p):
    """
    Returns samples with a bits flipped with a specified probability.

    Rejects any variants that result in a fixed column.
    """
    S = np.zeros((ts.sample_size, ts.num_mutations), dtype=np.int8)
    for variant in ts.variants():
        done = False
        # Reject any columns that have no 1s or no zeros
        while not done:
            S[:, variant.index] = make_errors(variant.genotypes, error_p)
            s = np.sum(S[:, variant.index])
            done = 0 < s < ts.sample_size
    return S.T


def tsinfer_dev(
        n, L, seed, num_threads=1, recombination_rate=1e-8,
        error_rate=0, engine="C", log_level="WARNING",
        debug=True, progress=False, path_compression=True):

    np.random.seed(seed)
    random.seed(seed)
    L_megabases = int(L * 10**6)

    # daiquiri.setup(level=log_level)

    ts = msprime.simulate(
            n, Ne=10**4, length=L_megabases,
            recombination_rate=recombination_rate, mutation_rate=1e-8,
            random_seed=seed)
    if debug:
        print("num_sites = ", ts.num_sites)
    assert ts.num_sites > 0

    time_map = collections.defaultdict(list)
    j = 0
    for tree in ts.trees():
        for site in tree.sites():
            assert len(site.mutations) == 1
            node = site.mutations[0].node
            if tree.num_samples(node) > 1:
                time_map[ts.node(node).time].append(j)
                j += 1
    times = [None for _ in range(j)]
    for j, time in enumerate(sorted(time_map.keys()), start=2):
        # print(j, time, time_map[time])
        for site in time_map[time]:
            times[site] = j

    sample_data = tsinfer.SampleData.from_tree_sequence(ts)

    # ancestor_data = tsinfer.generate_ancestors(
    #     sample_data, engine=engine, num_threads=num_threads)

    print(times)
    ancestor_data = tsinfer.AncestorData(sample_data)
    generator = tsinfer.AncestorsGenerator(
        sample_data, ancestor_data, tsinfer.DummyProgressMonitor(),
        engine=engine, num_threads=num_threads)
    generator.add_sites(times)
    generator.run()

    print(ancestor_data)
    ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data, engine=engine)
    # output_ts = tsinfer.match_samples(subset_samples, ancestors_ts, engine=engine)

    output_ts = tsinfer.match_samples(sample_data, ancestors_ts, engine=engine)
#     # dump_provenance(output_ts)
    G1 = ts.genotype_matrix()
    G2 = output_ts.genotype_matrix()
    assert np.array_equal(G1, G2)



def dump_provenance(ts):
    print("dump provenance")
    for p in ts.provenances():
        print("-" * 50)
        print(p.timestamp)
        pprint.pprint(json.loads(p.record))


def build_profile_inputs(n, num_megabases):
    L = num_megabases * 10**6
    input_file = "tmp__NOBACKUP__/profile-n={}-m={}.input.trees".format(
            n, num_megabases)
    if os.path.exists(input_file):
        ts = msprime.load(input_file)
    else:
        ts = msprime.simulate(
            n, length=L, Ne=10**4, recombination_rate=1e-8, mutation_rate=1e-8,
            random_seed=10)
        print("Ran simulation: n = ", n, " num_sites = ", ts.num_sites,
                "num_trees =", ts.num_trees)
        ts.dump(input_file)
    filename = "tmp__NOBACKUP__/profile-n={}-m={}.samples".format(n, num_megabases)
    if os.path.exists(filename):
        os.unlink(filename)
    # daiquiri.setup(level="DEBUG")
    with tsinfer.SampleData(
            sequence_length=ts.sequence_length, path=filename,
            num_flush_threads=4) as sample_data:
        # progress_monitor = tqdm.tqdm(total=ts.num_samples)
        # for j in range(ts.num_samples):
        #     sample_data.add_sample(metadata={"name": "sample_{}".format(j)})
        #     progress_monitor.update()
        # progress_monitor.close()
        progress_monitor = tqdm.tqdm(total=ts.num_sites)
        for variant in ts.variants():
            sample_data.add_site(variant.site.position, variant.genotypes)
            progress_monitor.update()
        progress_monitor.close()

    print(sample_data)

#     filename = "tmp__NOBACKUP__/profile-n={}_m={}.ancestors".format(n, num_megabases)
#     if os.path.exists(filename):
#         os.unlink(filename)
#     ancestor_data = tsinfer.AncestorData.initialise(sample_data, filename=filename)
#     tsinfer.build_ancestors(sample_data, ancestor_data, progress=True)
#     ancestor_data.finalise()

def copy_1kg():
    source = "tmp__NOBACKUP__/1kg_chr22.samples"
    sample_data = tsinfer.SampleData.load(source)
    copy = sample_data.copy("tmp__NOBACKUP__/1kg_chr22_copy.samples")
    copy.finalise()
    print(sample_data)
    print("copy = ")
    print(copy)

def tutorial_samples():
    import tqdm
    import msprime
    import tsinfer

    ts = msprime.simulate(
        sample_size=10000, Ne=10**4, recombination_rate=1e-8,
        mutation_rate=1e-8, length=10*10**6, random_seed=42)
    ts.dump("tmp__NOBACKUP__/simulation-source.trees")
    print("simulation done:", ts.num_trees, "trees and", ts.num_sites,  "sites")

    progress = tqdm.tqdm(total=ts.num_sites)
    with tsinfer.SampleData(
            path="tmp__NOBACKUP__/simulation.samples",
            sequence_length=ts.sequence_length,
            num_flush_threads=2) as sample_data:
        for var in ts.variants():
            sample_data.add_site(var.site.position, var.genotypes, var.alleles)
            progress.update()
    progress.close()


def subset_sites(ts, position):
    """
    Return a copy of the specified tree sequence with sites reduced to those
    with positions in the specified list.
    """
    tables = ts.dump_tables()
    lookup = frozenset(position)
    tables.sites.clear()
    tables.mutations.clear()
    for site in ts.sites():
        if site.position in lookup:
            site_id = tables.sites.add_row(
                site.position, ancestral_state=site.ancestral_state,
                metadata=site.metadata)
            for mutation in site.mutations:
                tables.mutations.add_row(
                    site_id, node=mutation.node, parent=mutation.parent,
                    derived_state=mutation.derived_state,
                    metadata=mutation.metadata)
    return tables.tree_sequence()

def minimise(ts):
    tables = ts.dump_tables()

    out_map = {}
    in_map = {}
    first_site = 0
    for (_, edges_out, edges_in), tree in zip(ts.edge_diffs(), ts.trees()):
        for edge in edges_out:
            out_map[edge.child] = edge
        for edge in edges_in:
            in_map[edge.child] = edge
        if tree.num_sites > 0:
            sites = list(tree.sites())
            if first_site:
                x = 0
                first_site = False
            else:
                x = sites[0].position
            print("X = ", x)
            for edge in out_map.values():
                print("FLUSH", edge)
            for edge in in_map.values():
                print("INSER", edge)

            # # Flush the edge buffer.
            # for left, parent, child in edge_buffer:
            #     tables.edges.add_row(left, x, parent, child)
            # # Add edges for each node in the tree.
            # edge_buffer.clear()
            # for root in tree.roots:
            #     for u in tree.nodes(root):
            #         if u != root:
            #             edge_buffer.append((x, tree.parent(u), u))

    # position = np.hstack([[0], tables.sites.position, [ts.sequence_length]])
    # position = tables.sites.position
    # edges = []
    # print(position)
    # tables.edges.clear()
    # for edge in ts.edges():
    #     left = np.searchsorted(position, edge.left)
    #     right = np.searchsorted(position, edge.right)

    #     print(edge, left, right)
    #     # if right - left > 1:
    #         # print("KEEP:", edge, left, right)
    #         # tables.edges.add_row(
    #         #     position[left], position[right], edge.parent, edge.child)
    #         # print("added", tables.edges[-1])
    #     # else:
    #         # print("SKIP:", edge, left, right)

    # ts = tables.tree_sequence()
    # for tree in ts.trees():
    #     print("TREE:", tree.interval)
    #     print(tree.draw(format="unicode"))





def minimise_dev():
    ts = msprime.simulate(5, mutation_rate=1, recombination_rate=2, random_seed=3)
    # ts = msprime.load(sys.argv[1])

    position = ts.tables.sites.position[::2]
    subset_ts = subset_sites(ts, position)
    print("Got subset")

    ts_new = tsinfer.minimise(subset_ts)
    for tree in ts_new.trees():
        print("TREE:", tree.interval)
        print(tree.draw(format="unicode"))
    # print(ts_new.tables)
    print("done")
    other = minimise(subset_ts)


def run_build():

    sample_data = tsinfer.load(sys.argv[1])
    ad = tsinfer.generate_ancestors(sample_data)
    print(ad)


if __name__ == "__main__":

    # run_build()

    # np.set_printoptions(linewidth=20000)
    # np.set_printoptions(threshold=20000000)

    # tutorial_samples()

    # build_profile_inputs(10, 10)
    # build_profile_inputs(100, 10)
    # build_profile_inputs(1000, 100)
    # build_profile_inputs(10**4, 100)
    # build_profile_inputs(10**5, 100)

    # for j in range(1, 100):
    #     tsinfer_dev(15, 0.5, seed=j, num_threads=0, engine="P", recombination_rate=1e-8)
    # copy_1kg()
    tsinfer_dev(36, 0.3, seed=4, num_threads=0, engine="P", recombination_rate=1e-8)

    # minimise_dev()

#     for seed in range(1, 10000):
#         print(seed)
#         # tsinfer_dev(40, 2.5, seed=seed, num_threads=1, genotype_quality=1e-3, engine="C")
