import numpy as np
import random
import os
import h5py
import zarr
import sys
import pandas as pd
import daiquiri
import bsddb3
import time
import scipy
import pickle

import matplotlib as mp
# Force matplotlib to not use any Xwindows backend.
mp.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tsinfer
import msprime



def plot_breakpoints(ts, map_file, output_file):
    # Read in the recombination map using the read_hapmap method,
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


def check_infer(
        n, L, seed, num_threads=1, recombination_rate=1e-8,
        genotype_quality=0, method="C", log_level="WARNING",
        debug=True, progress=False):

    np.random.seed(seed)
    random.seed(seed)
    L_megabases = int(L * 10**6)

    daiquiri.setup(level=log_level)
    ts = msprime.simulate(
            n, Ne=10**4, length=L_megabases,
            recombination_rate=1e-8, mutation_rate=1e-8,
            random_seed=seed)
    if debug:
        print("num_sites = ", ts.num_sites)
    assert ts.num_sites > 0
    positions = np.array([site.position for site in ts.sites()])
    V = ts.genotype_matrix()
    # print(V)
    G = generate_samples(ts, genotype_quality)
    # print(S.T)
    # print(np.where(S.T != V))
    recombination_rate = np.zeros_like(positions) + recombination_rate

    inferred_ts = tsinfer.infer(
        G, positions, ts.sequence_length, recombination_rate,
        sample_error=genotype_quality, method="method", num_threads=num_threads,
        progress=progress)

    assert np.array_equal(G, inferred_ts.genotype_matrix())

def tsinfer_dev(
        n, L, seed, num_threads=1, recombination_rate=1e-8,
        genotype_quality=0, method="C", log_level="WARNING",
        debug=True, progress=False):

    np.random.seed(seed)
    random.seed(seed)
    L_megabases = int(L * 10**6)

    daiquiri.setup(level=log_level)

    ts = msprime.simulate(
            n, Ne=10**4, length=L_megabases,
            recombination_rate=1e-8, mutation_rate=1e-8,
            random_seed=seed)
    if debug:
        print("num_sites = ", ts.num_sites)
    assert ts.num_sites > 0
    positions = np.array([site.position for site in ts.sites()])
    V = ts.genotype_matrix()
    # print(V)
    G = generate_samples(ts, genotype_quality)
    # print(S.T)
    # print(np.where(S.T != V))
    recombination_rate = np.zeros_like(positions) + recombination_rate

    input_root = zarr.group()
    tsinfer.InputFile.build(
        input_root, genotypes=G,
        # genotype_qualities=tsinfer.proba_to_phred(error_probability),
        position=positions,
        recombination_rate=recombination_rate, sequence_length=ts.sequence_length,
        compress=False)
    ancestors_root = zarr.group()

    #TMP changed method to C here for make the sets of ancestors comparable.
    tsinfer.build_ancestors(
        input_root, ancestors_root, method="C", chunk_size=16, compress=False)

    ancestors_ts = tsinfer.match_ancestors(
        input_root, ancestors_root, method=method, num_threads=num_threads,
        )
        # output_path="tmp__NOBACKUP__/bad_tb.tsancts", output_interval=0.1)
        # output_path=None, traceback_file_pattern="tmp__NOBACKUP__/traceback_{}.pkl")
    assert ancestors_ts.sequence_length == ts.num_sites

    A = ancestors_root["ancestors/haplotypes"][:]
    A[A == 255] = 0
    for v in ancestors_ts.variants():
        assert np.array_equal(v.genotypes, A[:, v.index])

    inferred_ts = tsinfer.match_samples(
        input_root, ancestors_ts, method=method,
        genotype_quality=genotype_quality, num_threads=num_threads,
        simplify=False) #, traceback_file_pattern="tmp__NOBACKUP__/traceback_{}.pkl")

    print("num_edges = ", inferred_ts.num_edges)

    # with open("tmp__NOBACKUP__/traceback_59.pkl", 'rb') as f:
    #     d = pickle.load(f)
    #     tb = d["traceback"]
    #     for j, row in enumerate(tb):
    #         print(j)
    #         for k, v in row.items():
    #             print("\t", k, "\t{:.14f}".format(v))

    # print(inferred_ts.tables)
    # for t in inferred_ts.trees():
    #     # print(t.draw(format="unicode"))
    #     sites = list(t.sites())
    #     name = "t_{}_{}.svg".format(sites[0].index, sites[-1].index + 1)
    #     t.draw(name, width=800, height=600)

    flags = inferred_ts.tables.nodes.flags
    samples = np.where(flags == 1)[0][-n:]
    inferred_ts, node_map = inferred_ts.simplify(samples.astype(np.int32), map_nodes=True)

#     print("SIMPLIFIED")
#     node_labels = {node_map[k]: str(k) for k in range(node_map.shape[0])}
#     for t in inferred_ts.trees():
#         print(t.draw(format="unicode", node_label_text=node_labels))

    assert inferred_ts.num_samples == ts.num_samples
    assert inferred_ts.num_sites == ts.num_sites
    assert inferred_ts.sequence_length == ts.sequence_length
    assert np.array_equal(G, inferred_ts.genotype_matrix())
    # for v1, v2 in zip(ts.variants(), inferred_ts.variants()):
    #     assert np.array_equal(v1.genotypes, v2.genotypes)
    #     assert v1.position == v2.position

def compress(filename, output_file):
    ts = msprime.load(filename)
    ts.dump(output_file, zlib_compression=True)


def analyse_file(filename):
    before = time.process_time()
    ts = msprime.load(filename)
    duration = time.process_time() - before
    print("loaded in {:.2f} seconds".format(duration))
    print("num_trees = ", ts.num_trees)
    print("size = {:.2f}MiB".format(os.path.getsize(filename) / 1024**2))

    plot_breakpoints(ts, "data/hapmap/genetic_map_GRCh37_chr22.txt",
        "chr22_breakpoints.png")

    before = time.process_time()
    j = 0
    for t in ts.trees():
        j += 1
        # if j == ts.num_trees / 2:
        #     t.draw(path="chr22_tree.svg")
    assert j == ts.num_trees
    duration = time.process_time() - before
    print("Iterated over trees in {:.2f} seconds".format(duration))



    num_children = []
    for j, e in enumerate(ts.edgesets()):
        # print(e.left, e.right, e.parent, ts.time(e.parent), e.children, sep="\t")
        num_children.append(len(e.children))

    num_children = np.array(num_children)

    print("total edges= ", ts.num_edges)
    print("non binary     = ", np.sum(num_children > 2))
    print("max children   = ", np.max(num_children))
    print("mean children  = ", np.mean(num_children))
    print("median children= ", np.median(num_children))

    plt.clf()
    sns.distplot(num_children)
    plt.savefig("chr22_num_children.png")





    # for l, r_out, r_in in ts.diffs():
    #     print(l, len(r_out), len(r_in), sep="\t")

    # for t in ts.trees():
    #     t.draw(
    #         "tree_{}.svg".format(t.index), 4000, 4000, show_internal_node_labels=False,
    #         show_leaf_node_labels=False)
    #     if t.index == 10:
    #         break



def debug_no_recombination():

    method = "P"
    num_samples = 3
    seed = 8
    ts_source = msprime.simulate(num_samples, random_seed=seed, mutation_rate=5)
    print("sim = ", num_samples, ts_source.num_sites, seed)
    nodes = set()
    for site in ts_source.sites():
        for mutation in site.mutations:
            nodes.add(mutation.node)
    assert nodes == set(range(ts_source.num_nodes - 1))


    # ts_inferred = infer_from_simulation(ts_source, method="P")
    # for t in ts_source.trees():
    #     print(t.draw(format="unicode"))
    # print(ts_inferred.num_trees)
    # for t in ts_inferred.trees():
    #     print(t.draw(format="unicode"))
    # assert ts_inferred.num_trees == 1

    input_root = zarr.group()
    tsinfer.InputFile.build(
        input_root, genotypes=ts_source.genotype_matrix(),
        recombination_rate=1,
        sequence_length=ts_source.num_sites,
        compress=False)
    ancestors_root = zarr.group()

    tsinfer.build_ancestors(
        input_root, ancestors_root, method=method, compress=False)

    ancestors_ts = tsinfer.match_ancestors(input_root, ancestors_root, method=method)
    assert ancestors_ts.sequence_length == ts_source.num_sites

    # for t in ancestors_ts.trees():
    #     print(t.draw(format="unicode"))
    # print(ancestors_ts.tables)

    print("Ancestors")
    A = ancestors_root["ancestors/haplotypes"][:]
    print(A)

    print("Samples")
    print(ts_source.genotype_matrix().T)
    A[A == 255] = 0
    for v in ancestors_ts.variants():
        assert np.array_equal(v.genotypes, A[:, v.index])

    inferred_ts = tsinfer.match_samples(input_root, ancestors_ts, method=method,
            simplify=False)

#     for t in inferred_ts.trees():
#         print(t.draw(format="unicode"))

#     print(inferred_ts.tables)

    print("num_edges = ", inferred_ts.num_edges)
    print("num_trees = ", inferred_ts.num_trees)



def build_profile_inputs(n, num_megabases):
    L = num_megabases * 10**6
    ts = msprime.simulate(
        n, length=L, Ne=10**4, recombination_rate=1e-8, mutation_rate=1e-8,
        random_seed=10)
    print("Ran simulation: n = ", n, " num_sites = ", ts.num_sites,
            "num_trees =", ts.num_trees)
    input_file = "tmp__NOBACKUP__/large-input-source-n={}-m={}.hdf5".format(
            n, num_megabases)
    ts.dump(input_file)
    V = ts.genotype_matrix()
    # V = np.zeros((ts.num_sites, ts.sample_size), dtype=np.uint8)
    # for v in ts.variants():
    #     V[v.index:] = v.genotypes
    print("Built variant matrix: {:.2f} MiB".format(V.nbytes / (1024 * 1024)))
    positions = np.array([site.position for site in ts.sites()])
    recombination_rate = np.zeros(ts.num_sites) + 1e-8
    input_file = "tmp__NOBACKUP__/profile-n={}_m={}_dbm.tsinf".format(n, num_megabases)
    # with h5py.File(input_file, "w") as input_hdf5:
    # with zarr.ZipStore(input_file) as input_hdf5:
    # input_hdf5 = zarr.DirectoryStore(input_file)
    if os.path.exists(input_file):
        os.unlink(input_file)
    input_hdf5 = zarr.DBMStore(input_file, open=bsddb3.btopen)
    # input_hdf5 = zarr.ZipStore(input_file)
    root = zarr.group(store=input_hdf5, overwrite=True)
    tsinfer.InputFile.build(
        root, genotypes=V, position=positions,
        recombination_rate=recombination_rate, sequence_length=ts.sequence_length)
    input_hdf5.close()
    print("Wrote", input_file)


def build_1kg_sim():
    n = 5008
    chrom = "22"
    infile = "data/hapmap/genetic_map_GRCh37_chr{}.txt".format(chrom)
    recomb_map = msprime.RecombinationMap.read_hapmap(infile)

    # ts = msprime.simulate(
    #     sample_size=n, Ne=10**4, recombination_map=recomb_map,
    #     mutation_rate=5*1e-8)

    # print("simulated chr{} with {} sites".format(chrom, ts.num_sites))

    prefix = "tmp__NOBACKUP__/sim1kg_chr{}".format(chrom)
    outfile = prefix + ".source.ts"
    # ts.dump(outfile)
    ts = msprime.load(outfile)

    V = ts.genotype_matrix()
    print("Built variant matrix: {:.2f} MiB".format(V.nbytes / (1024 * 1024)))
    positions = np.array([site.position for site in ts.sites()])
    recombination_rates = np.zeros_like(positions)
    last_physical_pos = 0
    last_genetic_pos = 0
    for site in ts.sites():
        physical_pos = site.position
        genetic_pos = recomb_map.physical_to_genetic(physical_pos)
        physical_dist = physical_pos - last_physical_pos
        genetic_dist = genetic_pos - last_genetic_pos
        scaled_recomb_rate = 0
        if genetic_dist > 0:
            scaled_recomb_rate = physical_dist / genetic_dist
        recombination_rates[site.index] = scaled_recomb_rate
        last_physical_pos = physical_pos
        last_genetic_pos = genetic_pos

    input_file = prefix + ".tsinf"
    if os.path.exists(input_file):
        os.unlink(input_file)
    input_hdf5 = zarr.DBMStore(input_file, open=bsddb3.btopen)
    root = zarr.group(store=input_hdf5, overwrite=True)
    tsinfer.InputFile.build(
        root, genotypes=V, position=positions,
        recombination_rate=recombination_rates, sequence_length=ts.sequence_length)
    input_hdf5.close()
    print("Wrote", input_file)




def large_profile(input_file, output_file, num_threads=2, log_level="DEBUG"):
    hdf5 = h5py.File(input_file, "r")
    tsp = tsinfer.infer(
        samples=hdf5["samples/haplotypes"][:],
        positions=hdf5["sites/position"][:],
        recombination_rate=hdf5["sites/recombination_rate"][:],
        sequence_length=hdf5.attrs["sequence_length"],
        num_threads=num_threads, log_level=log_level, progress=True)
    tsp.dump(output_file)

    # print(tsp.tables)
    # for t in tsp.trees():
    #     print("tree", t.index)
    #     print(t.draw(format="unicode"))

def save_ancestor_ts(
        n, L, seed, num_threads=1, recombination_rate=1e-8,
        resolve_shared_recombinations=False,
        progress=False, error_rate=0, method="C", log_level="WARNING"):
    L_megabases = int(L * 10**6)
    ts = msprime.simulate(
            n, Ne=10**4, length=L_megabases,
            recombination_rate=1e-8, mutation_rate=1e-8,
            random_seed=seed)
    print("num_sites = ", ts.num_sites)
    positions = np.array([site.position for site in ts.sites()])
    S = generate_samples(ts, 0)
    recombination_rate = np.zeros_like(positions) + recombination_rate

    # make_input_hdf5("ancestor_example.hdf5", S, positions, recombination_rate,
    #         ts.sequence_length)

    manager = tsinfer.InferenceManager(
        S, positions, ts.sequence_length, recombination_rate,
        num_threads=num_threads, method=method, progress=progress, log_level=log_level,
        resolve_shared_recombinations=resolve_shared_recombinations)
        # ancestor_traceback_file_pattern="tmp__NOBACKUP__/tracebacks/tb_{}.pkl")

    manager.initialise()
    manager.process_ancestors()
    ts_new = manager.get_tree_sequence()

    A = manager.ancestors()
    # Need to reset the unknown values to be zeros.
    A[A == -1] = 0
    B = np.zeros((manager.num_ancestors, manager.num_sites), dtype=np.int8)
    for v in ts_new.variants():
        B[:, v.index] = v.genotypes
    assert np.array_equal(A, B)
    print(ts_new.tables)
    # ts.dump("tmp__NOBACKUP__/ancestor_ts-{}.hdf5".format(ts.num_sites))
    for t in ts_new.trees():
        print(t.interval)
        print(t.draw(format="unicode"))
    new_nodes = [j for j, node in enumerate(ts_new.nodes()) if node.flags == 0]
    print(new_nodes)
    for e in ts_new.edges():
        if e.child in new_nodes or e.parent in new_nodes:
            print("{:.0f}\t{:.0f}".format(e.left, e.right), e.parent, e.child, sep="\t")

    nodes = ts_new.tables.nodes
    nodes.set_columns(flags=np.ones_like(nodes.flags), time=nodes.time)
    print(nodes)
    t = ts_new.tables
    tsp = msprime.load_tables(
            nodes=nodes, edges=t.edges,  sites=t.sites, mutations=t.mutations)
    print(tsp.tables)
    for j, h in enumerate(tsp.haplotypes()):
        print(j, "\t",h)



def examine_ancestor_ts(filename):
    ts = msprime.load(filename)
    print("num_sites = ", ts.num_sites)
    print("num_trees = ", ts.num_trees)
    print("num_edges = ", ts.num_edges)

    for (left, right), edges_in, edges_out in ts.edge_diffs():
        print("NEW TREE: {:.2f}".format(right - left), len(edges_in), len(edges_out), sep="\t")
        print("OUT")
        for e in edges_out:
            print("\t", e.parent, e.child)
        print("IN")
        for e in edges_in:
            print("\t", e.parent, e.child)

    # zero_edges = 0
    # edges = msprime.EdgeTable()
    # for e in ts.edges():
    #     if e.parent == 0:
    #         zero_edges += 1
    #     else:
    #         edges.add_row(e.left, e.right, e.parent, e.child)
    # print("zero_edges = ", zero_edges, zero_edges / ts.num_edges)
    # t = ts.tables
    # t.edges = edges
    # ts = msprime.load_tables(**t.asdict())
    # print("num_sites = ", ts.num_sites)
    # print("num_trees = ", ts.num_trees)
    # print("num_edges = ", ts.num_edges)

    # for t in ts.trees():
    #     print("Tree:", t.interval)
    #     print(t.draw(format="unicode"))
    #     print("=" * 200)

    # import pickle
    # j = 960
    # filename = "tmp__NOBACKUP__/tracebacks/tb_{}.pkl".format(j)
    # with open(filename, "rb") as f:
    #     debug = pickle.load(f)

    # tracebacks = debug["traceback"]
    # # print("focal = ", debug["focal_sites"])
    # del debug["traceback"]
    # print("debug:", debug)
    # a = debug["ancestor"]
    # lengths = [len(t) for t in tracebacks]
    # import matplotlib as mp
    # # Force matplotlib to not use any Xwindows backend.
    # mp.use('Agg')
    # import matplotlib.pyplot as plt

    # plt.clf()
    # plt.plot(lengths)
    # plt.savefig("tracebacks_{}.png".format(j))

#     start = 0
#     for j, t in enumerate(tracebacks[start:]):
#         print("TB", j, len(t))
#         for k, v in t.items():
#             print("\t", k, "\t", v)

#     site_id = 0
#     for t in ts.trees():
#         for site in t.sites():
#             L = tracebacks[site_id]
#             site_id += 1
#         # print("TREE")
#             print(L)
#             # for x1 in L.values():
#             #     for x2 in L.values():
#             #         print("\t", x1, x2, x1 == x2, sep="\t")
#             print("SITE = ", site_id)
#             print("root children = ", len(t.children(t.root)))
#             for u, v in L.items():
#                 path = []
#                 while u != msprime.NULL_NODE:
#                     path.append(u)
#                     u = t.parent(u)
#                     # if u in L and L[u] == v:
#                     #     print("ERROR", u)
#                 print(v, path)
#             print()
#             node_labels = {u: "{}:{:.2G}".format(u, L[u]) for u in L.keys()}
#             if site_id == 694:
#                 print(t.draw(format="unicode", node_label_text=node_labels))


    # for j, L in enumerate(tracebacks):
    #     print(j, L)
        # if len(L) > max_len:
        #     max_len = len(L)
        #     max_index = j
    # # print(j, "\t", L)
    # print("max len = ", max_len)
    # for k, v in tracebacks[max_index].items():
        # print(k, "\t", v)

def verify(file1, file2):
    ts1 = msprime.load(file1)
    ts2 = msprime.load(file2)
    assert ts1.num_samples == ts2.num_samples
    assert ts1.num_sites == ts2.num_sites

    for v1, v2 in zip(ts1.variants(), ts2.variants()):
        assert v1.position == v2.position
        assert np.array_equal(v1.genotypes, v2.genotypes)

def lookat(filename):

    # for j in range(1, 1000):
    #     tbfile = "tmp__NOBACKUP__/traceback_{}.pkl".format(j)
    #     with open(tbfile, "rb") as f:
    #         debug = pickle.load(f)
    #         tb = debug["traceback"]
    #         for j, row in enumerate(tb):
    #             # print(j, row)
    #             if 596 in row:
    #                 print(j)
    #                 for k, v in row.items():
    #                     print("\t", k, "\t{:.14f}".format(v))

    ts = msprime.load(filename)
    print(ts.num_edges, ts.num_trees)

    for t in ts.trees():
        for root in t.roots:
            if root != 0:
                if len(list(t.nodes(root))) != 1:
                    print("ERROR at ", root, "in tree ", t.index, t.interval)

        # print(t.draw(format="unicode"))


    sys.exit(0)


if __name__ == "__main__":

    np.set_printoptions(linewidth=20000)
    np.set_printoptions(threshold=20000000)

    # lookat("tmp__NOBACKUP__/bad_tb.tsancts")

    # build_1kg_sim()

    # compress(sys.argv[1], sys.argv[2])
    # analyse_file(sys.argv[1])

    # verify(sys.argv[1], sys.argv[2])

    # build_profile_inputs(10, 1)

#     build_profile_inputs(1000, 10)
#     build_profile_inputs(1000, 100)
#     build_profile_inputs(10**4, 100)
#     build_profile_inputs(10**5, 100)

    # build_profile_inputs(100)

    # debug_no_recombination()

    # large_profile(sys.argv[1], "{}.inferred.hdf5".format(sys.argv[1]),
    #         num_threads=40, log_level="DEBUG")

    # save_ancestor_ts(100, 10, 1, recombination_rate=1, num_threads=2)
    # examine_ancestor_ts(sys.argv[1])

    # save_ancestor_ts(15, 0.03, 7, recombination_rate=1, method="P",
    #         resolve_shared_recombinations=False)

    # tsinfer_dev(10, 0.1, seed=6, num_threads=0,
    #         genotype_quality=0.0, method="P", log_level="WARNING")

    # tsinfer_dev(40, 0.2, seed=84, num_threads=0, method="C",
    #         genotype_quality=0.001)

    for seed in range(1, 10000):
    # for seed in [4]:
        print(seed)
        # check_infer(20, 0.2, seed=seed, genotype_quality=0.0, num_threads=0, method="P")
        # tsinfer_dev(40, 2.5, seed=seed, num_threads=1, genotype_quality=1e-3, method="C")

        tsinfer_dev(10, 0.2, seed=seed, genotype_quality=0.0, num_threads=0, method="P")
    #     # tsinfer_dev(30, 1.5, seed=seed, num_threads=2, genotype_quality=0.01, method="C")

    # tsinfer_dev(60, 1000, num_threads=5, seed=1, error_rate=0.1, method="C",
    #         log_level="INFO", progress=True)
    # for seed in range(1, 1000):
    #     print(seed)
    #     tsinfer_dev(36, 10, seed=seed, error_rate=0.1, method="python")
