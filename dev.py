

import subprocess
import os
import numpy as np
import itertools
import multiprocessing
import pandas as pd
import random
import statistics
import sys
import attr
import collections
import concurrent.futures
import time
import random
import threading
import math
import resource
# import profilehooks

import humanize
import h5py
import tqdm
import psutil

import tsinfer
import _tsinfer
import msprime



@attr.s
class LinkedSegment(object):
    """
    A mapping of a half-open interval to a specific value in a linked list.
    Lists of segments must not contain overlapping intervals.
    """
    start = attr.ib(default=None)
    end = attr.ib(default=None)
    value = attr.ib(default=None)
    next = attr.ib(default=None)

def load_ancestors_dev(filename):

    with h5py.File(filename, "r") as f:
        sites = f["sites"]
        segments = f["segments"]
        ancestors = f["ancestors"]
        store = tsinfer.AncestorStore(
            position=sites["position"],
            site=segments["site"],
            start=segments["start"],
            end=segments["end"],
            state=segments["state"],
            focal_site=ancestors["focal_site"][:])
        samples = f["samples"]
        S = samples["haplotypes"][()]

    mutation_rate = 0.01
    # print(store.num_sites, store.num_ancestors)
    ancestor_ids = list(range(1, store.num_ancestors))
    # matcher = _tsinfer.AncestorMatcher(store, 0.01)
    matcher = tsinfer.AncestorMatcher(store, 1e-8)
    ts_builder = tsinfer.TreeSequenceBuilder(
            store.num_sites, S.shape[0], store.num_ancestors)

    # traceback = _tsinfer.Traceback(store, 2**10)
    traceback = tsinfer.Traceback(store)
    h = np.zeros(store.num_sites, dtype=np.int8)
    P = np.zeros(store.num_sites, dtype=np.int32)
    M = np.zeros(store.num_sites, dtype=np.uint32)

    # store.print_state()

    for ancestor_id in ancestor_ids:
        start_site, focal_site, end_site = store.get_ancestor(ancestor_id, h)
        # print(start_site, focal_site, end_site)
        # a = "".join(str(x) if x != -1 else '*' for x in h)
        # print(ancestor_id, "\t", a)
        best_match = matcher.best_path(
            ancestor_id, h, start_site, end_site, focal_site, 0, traceback)
        # print("best_match = ", best_match)
        traceback.run_build_ts(
            h, start_site, end_site, best_match, ts_builder, ancestor_id)
        # print("traceback", traceback)
        traceback.reset()
        # print()
        # assert num_mutations == 1

    # print("MATCHING SAMPLES")

    for j, h in enumerate(S):
        sample_id = j + store.num_ancestors
        # a = "".join(str(x) if x != -1 else '*' for x in h)
        # print(sample_id, "\t", a)
        best_match = matcher.best_path(
            store.num_ancestors, h, 0, store.num_sites, -1, mutation_rate, traceback)
        # print("best_match = ", best_match)
        num_mutations = traceback.run_build_ts(
            h, 0, store.num_sites, best_match, ts_builder, sample_id)
        # print("traceback", traceback)
        traceback.reset()
        # print()
        # assert num_mutations == 1

    # ts_builder.print_state()
    tsp = ts_builder.finalise()
    recurrent_mutations = 0
    back_mutations = 0
    for site in tsp.sites():
        # print(site)
        assert site.ancestral_state == '0'
        recurrent_mutations += (len(site.mutations) > 1)
        for mut in site.mutations:
            back_mutations += mut.derived_state == '0'
    print("recurrent muts    :", recurrent_mutations)
    print("back      muts    :", back_mutations)
    Sp = np.zeros((tsp.sample_size, tsp.num_sites), dtype="i1")
    for variant in tsp.variants():
        Sp[:, variant.index] = variant.genotypes
    assert np.all(Sp == S)
    # print("S = ")
    # print(S)
    # print("~Sp = ")
    # print(Sp)
    # print(Sp == S)
    tsp = tsp.simplify()
    # Need to compare on the haplotypes here because we might have a
    # ancestral state of 1 after simplify.
    H = list(tsp.haplotypes())
    for j in range(S.shape[0]):
        assert "".join(map(str, S[j])) == H[j]

    # for variant in tsp.variants():
    #     Sp[:, variant.index] = variant.genotypes
    # assert np.all(Sp == S)
    # tsp.dump("back-mutations-error.hdf5")



def compress_ancestors(filename):

    with h5py.File(filename, "r") as f:
        sites = f["sites"]
        segments = f["segments"]
        store = _tsinfer.AncestorStore(
            num_sites=sites["position"].shape[0],
            site=segments["site"],
            start=segments["start"],
            end=segments["end"],
            state=segments["state"])
        samples = f["samples"]
        S = samples["haplotypes"][()]

    # print(store.num_sites, store.num_ancestors)
    print(store)
    for l in range(store.num_sites):
        print(l, ":", store.get_site(l))

    A = np.zeros((store.num_ancestors, store.num_sites), dtype=np.int8) - 1
    for l in range(store.num_sites):
        for start, end, value in store.get_site(l):
            A[start:end, l] = value
    for a in A:
        a = "".join(str(x) if x != -1 else '*' for x in a)
        print(a)



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
            S[:,variant.index] = make_errors(variant.genotypes, error_p)
            s = np.sum(S[:, variant.index])
            done = 0 < s < ts.sample_size
    return S


def sort_ancestor_slice(A, p, start, end, sort_order, depth=0):
    n = end - start
    if n > 1:
        # print("  " * depth, "Sort Ancestor slice:", start, ":", end, sep="")
        # print("  " * depth, "p = ", p, sep="")
        m = A.shape[1]
        max_segment_breaks = 0
        sort_site = -1
        for l in range(m):
            col = A[p[start:end], l]
            segment_breaks = 0
            for j in range(n):
                if j < n - 1:
                    if sort_order == 0:
                        if col[j] > col[j + 1]:
                            segment_breaks += 1
                    else:
                        if col[j] <= col[j + 1]:
                            segment_breaks += 1
            # if segment_breaks > 1:
            #     print("col = ", col, "segment_breaks = ", segment_breaks)
            if segment_breaks > max_segment_breaks:
                max_segment_breaks = segment_breaks
                sort_site = l
        if max_segment_breaks > 1:
            # print("sort_site = ", sort_site)
            # # print("A = ", A[p[start: end], sort_site])
            sorting = np.argsort(A[p[start: end], sort_site])
            if sort_order == 1:
                sorting = sorting[::-1]
            # print("sorting = ", sorting, sorting.dtype)
            # print(p)
            # print(p[sorting])
            p[start: end] = p[start + sorting]
            # print("after", p)
            assert np.all(np.unique(p) == np.arange(p.shape[0]))
            # recursively sort within these partitions.
            for j in range(start, end - 1):
                # print(depth * "  ", "testing:", j, A[p[j], sort_site],  A[p[j + 1], sort_site])
                if A[p[j], sort_site] != A[p[j + 1], sort_site]:
                    sort_ancestor_slice(A, p, start, j + 1, sort_order, depth + 1)
                    start = j + 1
                    # print(depth * " ", "start = ", start)


def sort_ancestors(A, p, sort_order):
    """
    Sorts the specified array of ancestors to maximise the effectiveness
    of the run length encoding.
    """
    n, m = A.shape
    p[:] = np.arange(n, dtype=int)
    # print("BEFORE")
    # for j in range(n):
    #     a = "".join(str(x) if x != -1 else '*' for x in A[p[j],:])
    #     print(j, "\t", a)
    sort_ancestor_slice(A, p, 0, n, sort_order, 0)
    # print("AFTER")
    # for j in range(n):
    #     a = "".join(str(x) if x != -1 else '*' for x in A[p[j], :])
    #     print(j, "\t", a)


def build_ancestors(n, L, seed, filename):

    ts = msprime.simulate(
        n, length=L, recombination_rate=1e-8, mutation_rate=1e-8,
        Ne=10**4, random_seed=seed)
    # print("num_sites = ", ts.num_sites)
    # print("simulation done, num_sites = ", ts.num_sites)

    ts.dump(filename.split(".")[0] + ".ts.hdf5")

    position = [site.position for site in ts.sites()]

    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    np.random.seed(seed)
    # S = generate_samples(ts, 0.01)

    builder = _tsinfer.AncestorBuilder(S, position)
    store_builder = _tsinfer.AncestorStoreBuilder(
        builder.num_sites, builder.num_sites * 8192)
    num_threads = 20

    def build_frequency_class(work):
        frequency, focal_sites = work
        num_ancestors = len(focal_sites)
        A = np.zeros((num_ancestors, builder.num_sites), dtype=np.int8)
        p = np.zeros(num_ancestors, dtype=np.uint32)
        # print("frequency:", frequency, "sites = ", focal_sites)
        for j, focal_site in enumerate(focal_sites):
            builder.make_ancestor(focal_site, A[j, :])
        _tsinfer.sort_ancestors(A, p)
        # p = np.arange(num_ancestors, dtype=np.uint32)
        return frequency, focal_sites, A, p

    # TODO make these numpy arrays
    ancestor_focal_site = [-1]
    ancestor_frequency = [0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for result in executor.map(build_frequency_class, builder.get_frequency_classes()):
            frequency, focal_sites, A, p = result
            for index in p:
                # print("adding:", A[index, :], frequency, focal_sites[index])
                # TODO add in the focal site informatino for matching on
                # ancestors.
                store_builder.add(A[index, :])
                ancestor_focal_site.append(focal_sites[index])
                ancestor_frequency.append(frequency)

    N = store_builder.total_segments
    site = np.zeros(N, dtype=np.uint32)
    start = np.zeros(N, dtype=np.int32)
    end = np.zeros(N, dtype=np.int32)
    state = np.zeros(N, dtype=np.int8)
    store_builder.dump_segments(site, start, end, state)
    with h5py.File(filename, "w") as f:
        g = f.create_group("segments")
        g.create_dataset("site", data=site)
        g.create_dataset("start", data=start)
        g.create_dataset("end", data=end)
        g.create_dataset("state", data=state)
        g = f.create_group("sites")
        g.create_dataset("position", data=position)
        g = f.create_group("samples")
        g.create_dataset("haplotypes", data=S)
        g = f.create_group("ancestors")
        g.create_dataset("frequency", data=ancestor_frequency, dtype=np.int32)
        g.create_dataset("focal_site", data=ancestor_focal_site, dtype=np.uint32)

    store = _tsinfer.AncestorStore(
        position=position, focal_site=ancestor_focal_site,
        site=site, start=start, end=end, state=state)

    print("num sites        :", store.num_sites)
    print("num ancestors    :", store.num_ancestors)
    print("max_segments     :", store.max_num_site_segments)
    print("mean_segments    :", store.total_segments / store.num_sites)
    print("Memory           :", humanize.naturalsize(store.total_memory))
    print("Uncompressed     :", humanize.naturalsize(store.num_ancestors * store.num_sites))
    print("Sample memory    :", humanize.naturalsize(S.nbytes))

def load_ancestors(filename, show_progress=True, num_threads=1):

    with h5py.File(filename, "r") as f:
        sites = f["sites"]
        segments = f["segments"]
        ancestors = f["ancestors"]
        store = _tsinfer.AncestorStore(
            position=sites["position"],
            focal_site=ancestors["focal_site"],
            site=segments["site"],
            start=segments["start"],
            end=segments["end"],
            state=segments["state"])
        samples = f["samples"]
        S = samples["haplotypes"][()]

    print("pid              :", os.getpid())
    print("num sites        :", store.num_sites)
    print("num ancestors    :", store.num_ancestors)
    print("max_segments     :", store.max_num_site_segments)
    print("mean_segments    :", store.total_segments / store.num_sites)
    print("Memory           :", humanize.naturalsize(store.total_memory))
    print("Uncompressed     :", humanize.naturalsize(store.num_ancestors * store.num_sites))

    num_samples = S.shape[0]
    tree_sequence_builder = _tsinfer.TreeSequenceBuilder(store, num_samples);
    recombination_rate = 1e-8
    error_rate = 1e-20
    method = "C"
    tsinfer.match_ancestors(
        store, recombination_rate, tree_sequence_builder, method=method,
        num_threads=num_threads)
    tsinfer.match_samples(
        store, S, recombination_rate, error_rate, tree_sequence_builder,
        method=method, num_threads=num_threads)
    ts = tsinfer.finalise_tree_sequence(num_samples, store, tree_sequence_builder)

    ts = ts.simplify()
    ts.dump(filename.split(".")[0] + ".inferred_ts.hdf5")

    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print("RAM              :", humanize.naturalsize(memory))

    for j, h in enumerate(ts.haplotypes()):
        assert "".join(map(str, S[j])) == h

    # duration_cpu = time.clock() - before_cpu
    # duration_wall = time.time() - before_wall
    # print("Copying CPU time :", humanize.naturaldelta(duration_cpu))
    # print("Copying wall time:", humanize.naturaldelta(duration_wall))

def new_segments(n, L, seed):

    np.set_printoptions(linewidth=2000)
    np.set_printoptions(threshold=20000)
    np.random.seed(seed)
    random.seed(seed)

    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    if ts.num_sites == 0:
        print("zero sites; skipping")
        return
    positions = np.array([site.position for site in ts.sites()])
    # S = generate_samples(ts, 0.01)
    S = generate_samples(ts, 0)

    tsp = tsinfer.infer(S, positions, 1e-6, 1e-6, num_threads=20, method="C")

    Sp = np.zeros((tsp.sample_size, tsp.num_sites), dtype="i1")
    for variant in tsp.variants():
        Sp[:, variant.index] = variant.genotypes
    # print(S)
    # print()
    # print(Sp)
    assert np.all(Sp == S)

    # for t in tsp.trees():
    #     print(t.interval, t)
    # for site in tsp.sites():
    #     if len(site.mutations) > 1:
    #         print("Recurrent mutation")

    ts_simplified = tsp.simplify()
    # Need to compare on the haplotypes here because we might have a
    # ancestral state of 1 after simplify.
    H = list(tsp.haplotypes())
    for j in range(S.shape[0]):
        assert "".join(map(str, S[j])) == H[j]



def export_samples(n, L, seed):

    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    print("num_sites = ", ts.num_sites)
    with open("tmp__NOBACKUP__/samples.txt", "w") as out:
        for variant in ts.variants():
            print(variant.position, "".join(map(str, variant.genotypes)), sep="\t", file=out)




def compare_timings(n, L, seed):
    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    if ts.num_sites == 0:
        print("zero sites; skipping")
        return
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes

    total_matching_time_new = 0
    # Inline the code here so we can time it.
    samples = S
    num_samples, num_sites = samples.shape
    builder = tsinfer.AncestorBuilder(samples)
    matcher = _tsinfer.AncestorMatcher(num_sites)
    num_ancestors = builder.num_ancestors
    # tree_sequence_builder = TreeSequenceBuilder(num_samples, num_ancestors, num_sites)
    tree_sequence_builder = tsinfer.TreeSequenceBuilder(num_samples, num_ancestors, num_sites)

    A = np.zeros(num_sites, dtype=np.int8)
    P = np.zeros(num_sites, dtype=np.int32)
    M = np.zeros(num_sites, dtype=np.uint32)
    for j, A in enumerate(builder.build_all_ancestors()):
        before = time.clock()
        num_mutations = matcher.best_path(A, P, M, 0.01, 1e-200)
        total_matching_time_new += time.clock() - before
        # print(A)
        # print(P)
        # print("num_mutations = ", num_mutations, M[:num_mutations])
        assert num_mutations == 1
        # assert M[0] == focal_site
        matcher.add(A)
        tree_sequence_builder.add_path(j + 1, P, A, M[:num_mutations])
    # tree_sequence_builder.print_state()

    for j in range(num_samples):
        before = time.clock()
        num_mutations = matcher.best_path(samples[j], P, M, 0.01, 1e-200)
        total_matching_time_new += time.clock() - before
        u = num_ancestors + j + 1
        tree_sequence_builder.add_path(u, P, samples[j], M[:num_mutations])
    # tree_sequence_builder.print_state()
    tsp = tree_sequence_builder.finalise()

    Sp = np.zeros((tsp.sample_size, tsp.num_sites), dtype="i1")
    for variant in tsp.variants():
        Sp[:, variant.index] = variant.genotypes
    assert np.all(Sp == S)

    S = S.astype(np.uint8)
    panel = tsinfer.ReferencePanel(
        S, [site.position for site in ts.sites()], ts.sequence_length,
        rho=0.001, algorithm="c")
    before = time.clock()
    P, mutations = panel.infer_paths(num_workers=1)
    total_matching_time_old = time.clock() - before
    ts_new = panel.convert_records(P, mutations)
    Sp = np.zeros((ts_new.sample_size, ts_new.num_sites), dtype="u1")
    for variant in ts_new.variants():
        Sp[:, variant.index] = variant.genotypes
    assert np.all(Sp == S)
    print(n, L, total_matching_time_old, total_matching_time_new, sep="\t")

def ancestor_gap_density(n, L, seed):
    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    if ts.num_sites == 0:
        print("zero sites; skipping")
        return
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes

    samples = S
    num_samples, num_sites = samples.shape
    builder = tsinfer.AncestorBuilder(samples)
    # builder.print_state()
    matcher = tsinfer.AncestorMatcher(num_sites)
    num_ancestors = builder.num_ancestors

    # A = np.zeros(num_sites, dtype=np.int8)
    P = np.zeros(num_sites, dtype=np.int32)
    M = np.zeros(num_sites, dtype=np.uint32)
    for A in builder.build_all_ancestors():
        matcher.add(A)

#     for j in range(builder.num_ancestors):
#         focal_site = builder.site_order[j]
#         builder.build(j, A)
#         matcher.add(A)
    # matcher.print_state()
#     builder.print_state()
#     builder.print_all_ancestors()

    total_segments = np.zeros(ts.num_sites)
    total_blank_segments = np.zeros(ts.num_sites)
    total_blank_segments_distance = 0

    for l in range(matcher.num_sites):
        seg = matcher.sites_head[l]
        while seg is not None:
            # print(seg.start, seg.end, seg.value)
            total_segments[l] += 1
            if seg.value == -1:
                total_blank_segments[l] += 1
                total_blank_segments_distance += seg.end - seg.start
            seg = seg.next

    return {
        "n": n, "L": L,
        "num_sites":matcher.num_sites,

        "num_ancestors": matcher.num_ancestors,
        "mean_total_segments": np.mean(total_segments),
        "mean_blank_segments": np.mean(total_blank_segments),
        "total_blank_fraction": total_blank_segments_distance / (num_sites * num_ancestors)
    }


if __name__ == "__main__":

    np.set_printoptions(linewidth=20000)
    np.set_printoptions(threshold=200000)

    for j in range(1, 100000):
        print(j)
        new_segments(200, 100, j)

    # new_segments(4, 2, 5)
    # # new_segments(40, 20, 304)

    # export_samples(10, 100, 304)

    # n = 10
    # for L in np.linspace(100, 1000, 10):
    #     compare_timings(n, L, 1)

    # d = ancestor_gap_density(20, 40, 1)

    # rows = []
    # n = 10
    # for L in np.linspace(10, 5000, 20):
    #     d = ancestor_gap_density(n, L, 2)
    #     rows.append(d)
    #     df = pd.DataFrame(rows)
    #     print(df)
    #     df.to_csv("gap-analysis.csv")

    # n = 10
    # for j in np.arange(101, 200, 10):
    #     print("n                :", n)
    #     print("L                :", j, "Mb")
    #     filename = "tmp__NOBACKUP__/n={}_L={}.hdf5".format(n, j)
    #     build_ancestors(n, j * 10**6, 1, filename)
    #     # if not os.path.exists(filename):
    #     #     break
    #     # load_ancestors(filename, num_threads=40)
    #     print()

    # for j in range(1, 10000):
    # # for j in [4]:
    #     print(j, file=sys.stderr)
    #     filename = "tmp__NOBACKUP__/tmp-3.hdf5"
    #     build_ancestors(20, 0.2 * 10**6, j, filename)
    #     load_ancestors(filename)
        # load_ancestors_dev(filename)

#     filename = "tmp__NOBACKUP__/tmp2.hdf5"
#     build_ancestors(10, 0.1 * 10**6, 3, filename)
#     # compress_ancestors(filename)
#     load_ancestors_dev(filename)

    # load_ancestors("tmp__NOBACKUP__/n=10_L=11.hdf5", num_threads=40)
    # load_ancestors("tmp__NOBACKUP__/n=10_L=121.hdf5")

    # for j in range(1, 100000):
    #     build_ancestors(10, 10, j)
    #     if j % 1000 == 0:
    #         print(j)
