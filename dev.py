

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
import operator
import heapq
import sortedcontainers
# import profilehooks

import humanize
import h5py
import tqdm
import psutil
import svgwrite
import colour

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

# script_path = __file__ if "__file__" in locals() else "./dummy.py"
# sys.path.insert(1,os.path.join(os.path.dirname(os.path.abspath(script_path)),'..','msprime')) # use the local copy of msprime in preference to the global one
# sys.path.insert(1,os.path.join(os.path.dirname(os.path.abspath(script_path)),'..','tsinfer')) # use the local copy of tsinfer in preference to the global one


import tsinfer
import _tsinfer
import msprime
import _msprime



def draw_segments(segments, L):
    print("   ", "-" * L)
    for seg in segments:
        s = "{:<4d}".format(seg.value)
        s += " " * (seg.start - 0)
        s += "=" * (seg.end - seg.start)
        print(s)
    print("   ", "-" * L)



def build_ancestors_dev(n, L, seed):

    ts = msprime.simulate(
        n, length=L, recombination_rate=1e-8, mutation_rate=1e-8,
        Ne=10**4, random_seed=seed)
    position = np.array([site.position for site in ts.sites()])
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes

    print(S)
    num_sites = S.shape[1]
    frequency = S.sum(axis=0)
    print(frequency)
    bins = collections.defaultdict(list)
    for k in range(num_sites):
        bins[frequency[k]].append(k)
    print(bins)
    del bins[1]
    for freq in reversed(list(bins.keys())):
        print(freq, "->", bins[freq])
        # Group sites within a frequency class into identical sets.
        classes = collections.defaultdict(list)
        for site in bins[freq]:
            classes[tuple(S[:, site])].append(site)
        for k, v in classes.items():
            print("\t", k, "->", v)


    # builder = _tsinfer.AncestorBuilder(S, position)
    store = tsinfer.build_ancestors(S, position, method="C")
    print("n = ", n, "num_sites = ", ts.num_sites)

    A = np.zeros((store.num_ancestors, store.num_sites), dtype=np.int8)
    for j in range(store.num_ancestors):
        start, end, num_older_ancestors, focal_sites = store.get_ancestor(j, A[j, :])
        print(num_older_ancestors, "\t", A[j], "\t", focal_sites)

    # n0 = 0
    # n1 = 0
    # num_segments = np.zeros(store.num_sites, dtype=int)
    # for l in range(store.num_sites):
    #     segments = store.get_site(l)
    #     num_segments[l] = len(segments)

        # print(l, store.get_site(l))
    #     for start, end, state in store.get_site(l):
    #         if state == 0:
    #             n0 += (end - start)
    #         elif state == 1:
    #             n1 += (end - start)
    # print(num_segments)
    # pyplot.plot(num_segments)

    # pyplot.hist(num_segments, 50)
    # pyplot.savefig("tmp__NOBACKUP__/num_segments_n_{}s_{}.pdf".format(n, ts.num_sites))


    # total = store.num_ancestors * store.num_sites
    # nm1 = total - n0 - n1

    # print("n      ", n)
    # print("sites  ", ts.num_sites)
    # # print("zero   ", n0 / total)
    # # print("one    ", n1 / total)
    # # print("null   ", nm1 / total)


    # p = np.zeros(store.num_ancestors, dtype=np.uint32)
    # _tsinfer.sort_ancestors(A, p)

    # print("p = ", p)
    # for j in range(store.num_ancestors):
    #     a = A[p[j]]
    #     s = "".join(str(x) if x != -1 else '*' for x in a)
    #     print(s)


def examine_ancestors():
    A = np.load("tmp__NOBACKUP__/10k-ancestors.npy")
    print(A.shape)
    for j in range(A.shape[0]):
        a = A[j,-500:]
        s = "".join(str(x) if x != -1 else '*' for x in a)
        print(s)

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
        start_site, end_site, num_older_ancestors, focal_sites = store.get_ancestor(ancestor_id, h)
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

def new_segments(n, L, seed, num_threads=10, method="C", log_level="WARNING"):

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
    # S = generate_samples(ts, 0.1)
    S = generate_samples(ts, 0)

    tsp = tsinfer.infer(S, positions, L, 1e-9, 1e-200,
            num_threads=num_threads, method=method, log_level=log_level)
    new_positions = np.array([site.position for site in tsp.sites()])
    assert np.all(new_positions == positions)

    Sp = np.zeros((tsp.sample_size, tsp.num_sites), dtype="i1")
    for variant in tsp.variants():
        Sp[:, variant.index] = variant.genotypes
    #print(S,np.sum(S))
    #print()
    #print(Sp,np.sum(Sp))
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

    num_edges = sum(len(e.children) for e in ts_simplified.edgesets())
    print(ts.num_edgesets, ts_simplified.num_edgesets, num_edges, num_edges / ts_simplified.num_edgesets)



def test_ancestor_store(n, L, seed, num_threads=10, method="C"):

    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    if ts.num_sites == 0:
        print("zero sites; skipping")
        return
    positions = np.array([site.position for site in ts.sites()])
    samples = generate_samples(ts, 0)

    num_samples, num_sites = samples.shape
    if method == "C":
        builder = _tsinfer.AncestorBuilder(samples, positions)
        gigabyte = 2**30
        store_builder = _tsinfer.AncestorStoreBuilder(builder.num_sites, gigabyte)
    else:
        builder = tsinfer.AncestorBuilder(samples, positions)
        store_builder = tsinfer.AncestorStoreBuilder(builder.num_sites)

    frequency_classes = builder.get_frequency_classes()
    total_ancestors = 1
    num_focal_sites = 0
    for age, ancestor_focal_sites in frequency_classes:
        assert len(ancestor_focal_sites) > 0
        total_ancestors += len(ancestor_focal_sites)
        for focal_sites in ancestor_focal_sites:
            assert len(focal_sites) > 0
            num_focal_sites += len(focal_sites)
    print("num_sites = ", ts.num_sites, "total ancestors = ", total_ancestors)

    ancestor_age = np.zeros(total_ancestors, dtype=np.uint32)
    focal_site_ancestor = np.zeros(num_focal_sites, dtype=np.int32)
    focal_site = np.zeros(num_focal_sites, dtype=np.uint32)

    row = 0
    ancestor_id = 1
    A = np.zeros((total_ancestors, num_sites), dtype=np.int8)
    for age, ancestor_focal_sites in frequency_classes:
        num_ancestors = len(ancestor_focal_sites)
        for focal_sites in ancestor_focal_sites:
            builder.make_ancestor(focal_sites, A[ancestor_id, :])
            store_builder.add(A[ancestor_id, :])
            for site in focal_sites:
                focal_site_ancestor[row] = ancestor_id
                focal_site[row] = site
                row += 1
            ancestor_age[ancestor_id] = age
            ancestor_id += 1
    assert row == num_focal_sites
    assert ancestor_id == total_ancestors


    N = store_builder.total_segments
    site = np.zeros(N, dtype=np.uint32)
    start = np.zeros(N, dtype=np.int32)
    end = np.zeros(N, dtype=np.int32)
    store_builder.dump_segments(site, start, end)
    # assert np.max(end) == total_ancestors
    # assert np.min(start) == 0

    if method == "C":
        store = _tsinfer.AncestorStore(
            position=positions, site=site, start=start, end=end,
            ancestor_age=ancestor_age, focal_site_ancestor=focal_site_ancestor,
            focal_site=focal_site)
    else:
        store = tsinfer.AncestorStore(
            position=positions, site=site, start=start, end=end,
            ancestor_age=ancestor_age, focal_site_ancestor=focal_site_ancestor,
            focal_site=focal_site)

    # Now decode these ancestors into a local array.
    B = np.zeros((total_ancestors, num_sites), dtype=np.int8)
    ancestor_ids = list(range(total_ancestors))

    def store_decoder_worker(thread_index):
        chunk_size = int(math.ceil(len(ancestor_ids) / num_threads))
        start = thread_index * chunk_size
        for j in ancestor_ids[start: start + chunk_size]:
            store.get_ancestor(j, B[j, :])

    if num_threads > 1:
        threads = [
            threading.Thread(target=store_decoder_worker, args=(j,))
            for j in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        store_decoder_worker(0)
    # print(A)
    # print(B)
    assert np.all(A == B)



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

def site_set_stats(n, L, seed):
    ts = msprime.simulate(
        n, length=L, recombination_rate=1e-8, mutation_rate=1e-8,
        Ne=10**4, random_seed=seed)
    # print("n          = ", n)
    # print("L          = ", L / 10**6, "Mb")
    # print("sites      = ", ts.num_sites)

    positions = [site.position for site in ts.sites()]
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes

    num_samples, num_sites = S.shape
    builder = _tsinfer.AncestorBuilder(S, positions)
    store_builder = _tsinfer.AncestorStoreBuilder(
            builder.num_sites, 8192 * builder.num_sites)
    frequency_classes = builder.get_frequency_classes()
    num_ancestors = 1 + sum(len(sites) for _, sites in frequency_classes)
    # print("ancestors  = ", num_ancestors)

    A = np.zeros((num_ancestors, num_sites), dtype=np.int8)
    j = 1
    for frequency, focal_sites in builder.get_frequency_classes():
        for focal_site in focal_sites:
            assert np.sum(S[:, focal_site]) == frequency

            builder.make_ancestor(focal_site, A[j, :])
            j += 1
    sparsity = np.sum(A == -1) / (num_ancestors * num_sites)
    # print("sparsity   = ", sparsity)
    last_zeros = set()
    last_ones = set()
    last_zeros = set(np.where(A[:, 0] == 0)[0])
    last_ones = set(np.where(A[:, 0] == 1)[0])
    total_diffs = np.zeros(num_sites)
    for l in range(1, num_sites):
        zeros = set(np.where(A[:, l] == 0)[0])
        ones = set(np.where(A[:, l] == 1)[0])
        print("l = ", l)
        print("zeros = ", len(zeros), "::\t", zeros)
        print("ones  = ", len(ones), "::\t", ones)
        zero_zero_diff = zeros ^ last_zeros
        zero_one_diff = ones ^ last_zeros
        if len(zero_one_diff) < len(zero_zero_diff):
            # print("Swapping", len(zero_one_diff), len(zero_zero_diff))
            zeros = set(np.where(A[:, l] == 1)[0])
            ones = set(np.where(A[:, l] == 0)[0])
        zeros_diff = zeros ^ last_zeros
        ones_diff = ones ^ last_ones

        total_diffs[l] = len(zeros_diff) + len(ones_diff)
        print("total diffs = ", total_diffs[l])
        last_zeros = zeros
        last_ones = ones
    # print("mean diffs = ", np.mean(total_diffs))
    return {
        "n": n,
        "L": L,
        "sites": ts.num_sites,
        "ancestors": num_ancestors,
        "sparsity": sparsity,
        "mean_diffs": np.mean(total_diffs)
    }

def rle(inarray):
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.array(inarray)                  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return {'length':z, 'start':p, 'value':ia[i]}

class Visualiser(object):

    def __init__(self, width, num_sites, font_size=12):
        self.drawing = svgwrite.Drawing(width=width, debug=True)
        self.num_sites = num_sites
        self.scale = width / (num_sites + 1)
        self.font_size = font_size
        # We need this to make sure the coordinate system works properly.
        # Must be a better way to do it!
        self.drawing.add(self.drawing.rect(
            (0, 0), (width, self.scale),
            stroke="white", fill="white"))
        if font_size is not None:
            self.text_offset = self.font_size / 2 # TODO improve this!
            self.labels = self.drawing.add(
                self.drawing.g(font_size=font_size, text_anchor="middle"))
        stroke = "lightgray"
        self.focal_boxes = self.drawing.add(
                self.drawing.g(fill="red", stroke=stroke))
        self.one_boxes = self.drawing.add(
                self.drawing.g(fill="salmon", stroke=stroke))
        self.zero_boxes = self.drawing.add(
                self.drawing.g(fill="blue", stroke=stroke))
        self.missing_boxes = self.drawing.add(
                self.drawing.g(fill="white", stroke=stroke))
        self.current_row = 0
        self.label_map = {}

    def add_site_coordinates(self):
        if self.font_size is not None:
            for k in range(self.num_sites):
                coord = (
                    (k + 1) * self.scale + self.scale / 2,
                    self.current_row + self.scale / 2)
                self.labels.add(self.drawing.text(str(k), coord, dy=[self.text_offset]))
        self.current_row += 1

    def add_row(self, a, row_label=None, focal=[]):
        j = self.current_row
        self.label_map[row_label] = j
        if self.font_size is not None and row_label is not None:
            coord = (self.scale / 2, j * self.scale + self.scale / 2)
            self.labels.add(
                self.drawing.text(str(row_label), coord, dy=[self.text_offset]))
        for k in range(self.num_sites):
            corner = ((k + 1) * self.scale, j * self.scale)
            if a[k] == 0:
                self.zero_boxes.add(self.drawing.rect(corner, (self.scale, self.scale)))
            elif a[k] == -1:
                self.missing_boxes.add(self.drawing.rect(corner, (self.scale, self.scale)))
            else:
                if k in focal:
                    self.focal_boxes.add(self.drawing.rect(corner, (self.scale, self.scale)))
                else:
                    self.one_boxes.add(self.drawing.rect(corner, (self.scale, self.scale)))
        self.current_row += 1

    def add_intensity_row(self, d, row_label=None):
        j = self.current_row
        if self.font_size is not None and row_label is not None:
            coord = (self.scale / 2, j * self.scale + self.scale / 2)
            self.labels.add(
                self.drawing.text(str(row_label), coord, dy=[self.text_offset]))
        bottom = colour.Color("white")
        colours = list(bottom.range_to(colour.Color("black"), 10))
        bins = np.linspace(0, 1, 10)
        a = np.digitize(d, bins) - 1
        for k in range(self.num_sites):
            corner = ((k + 1) * self.scale, j * self.scale)
            self.drawing.add(self.drawing.rect(
                corner, (self.scale, self.scale), stroke="lightgrey",
                fill=colours[a[k]]))
        self.current_row += 1

    def add_separator(self):
        self.current_row += 1

    def show_path(self, label, path, fade_recents=True, copy_groups=None):
        """
        param copy_groups is only useful if there is the same ancestor on multiple rows
        """
        #highlight (darken) the cells we copied from
        for k in range(self.num_sites):
            rows = [path[k]] if copy_groups is None else copy_groups[path[k]]
            for r in rows:
                j = self.label_map[r]
                corner = ((k + 1) * self.scale, j * self.scale)
                self.drawing.add(self.drawing.rect(
                    corner, (self.scale, self.scale), stroke="lightgrey",
                    fill="black", opacity=0.6))

        if not fade_recents and label in self.label_map:
            # (slightly) highlight the current line
            row = self.label_map[label]
            corner = self.scale, row * self.scale
            self.drawing.add(self.drawing.rect(
                corner, ((self.scale * self.num_sites), self.scale),
                stroke="black", fill_opacity=0.2))

        elif fade_recents and label + 1 in self.label_map:
            #  fade out the more recent stuff
            row = self.label_map[label + 1]
            corner = self.scale, row * self.scale
            height = (self.current_row - row) * self.scale
            self.drawing.add(self.drawing.rect(
                corner, ((self.scale * self.num_sites), height),
                fill="white", opacity=0.8))

    def fade_row_false(self, bool_arr, label):
        row = self.label_map[label]
        runlengths = rle(bool_arr)
        height = self.scale
        for i in np.where(~runlengths['value'])[0]:
            corner = self.scale * (runlengths['start'][i]+1), row * self.scale
            self.drawing.add(self.drawing.rect(
                corner, ((self.scale * runlengths['length'][i]), height),
                fill="white", opacity=0.8))

    def save(self, filename):
        self.drawing.saveas(filename)


def visualise_ancestors(n, L, seed):

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
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    frequency = np.sum(S, axis=0)

    store = tsinfer.build_ancestors(S, positions, method="P")
    visualiser = Visualiser(800, store.num_sites, font_size=16)
    visualiser.add_site_coordinates()
    visualiser.add_row(a, 0)
    a = np.zeros(store.num_sites, dtype=np.int8)
    last_frequency = 0
    for j in range(1, store.num_ancestors):
        start, end, num_older_ancestors, focal_sites= store.get_ancestor(j, a)
        if any(frequency[focal] != last_frequency for focal in focal_sites):
            visualiser.add_separator()
            last_frequency = frequency[focal]
        visualiser.add_row(a, j)

    visualiser.add_separator()
    j = store.num_ancestors
    for a in S:
        visualiser.add_row(a, j)
        j += 1

    visualiser.save("tmp.svg")


def draw_copying_density(ts, width, breaks):
    """
    Visualises the copying density of the specified tree sequence.
    """
    num_ancestors = ts.num_nodes - ts.sample_size
    D = np.zeros((num_ancestors, ts.num_sites), dtype=int)
    for e in ts.edgesets():
        # FIXME!!! This will break when we start outputting positions correctly!
        left = int(e.left)
        right = int(e.right)
        D[e.parent, left:right] = len(e.children)
    # Rescale D into 0/1
    D = D / np.max(D)

    visualiser = Visualiser(width, ts.num_sites, font_size=16)
    visualiser.add_site_coordinates()
    for j in range(num_ancestors):
        if j in breaks:
            visualiser.add_separator()
        visualiser.add_intensity_row(D[j], j)

    visualiser.save("intensity.svg")



def visualise_copying(n, L, seed):

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
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    frequency = np.sum(S, axis=0)

    store = tsinfer.build_ancestors(S, positions, method="C")
    inferred_ts = tsinfer.infer(S, positions, L, 1e-6, 1e-200)

    last_frequency = 0
    breaks = set()
    a = np.zeros(store.num_sites, dtype=np.int8)
    for j in range(1, store.num_ancestors):
        start, end, num_older_ancestors, focal_sites = store.get_ancestor(j, a)
        if frequency[focal_sites[0]] != last_frequency:
            last_frequency = frequency[focal_sites[0]]
            breaks.add(j)

    visualiser = Visualiser(800, store.num_sites, font_size=16)
    visualiser.add_site_coordinates()
    a = np.zeros(store.num_sites, dtype=np.int8)
    visualiser.add_row(a, 0)
    for j in range(1, store.num_ancestors):
        if j in breaks:
            visualiser.add_separator()
        start, end, num_older_ancestors, focal_sites = store.get_ancestor(j, a)
        visualiser.add_row(a, j)

    N = store.num_ancestors + S.shape[0]
    P = np.zeros((N, store.num_sites), dtype=int) - 1
    site_index = {}
    for site in inferred_ts.sites():
        site_index[site.position] = site.index
    site_index[inferred_ts.sequence_length] = inferred_ts.num_sites
    site_index[0] = 0

    for e in inferred_ts.edgesets():
        for c in e.children:
            left = site_index[e.left]
            right = site_index[e.right]
            assert left < right
            P[c, left:right] = e.parent

#     visualiser.save("tmp.svg")

#     draw_copying_density(inferred_ts, 800, breaks)
    #make a map of which are actually used in the inferred ts
    locations = np.array([s.position for s in inferred_ts.sites()])
    used = np.zeros((store.num_ancestors, len(locations)),dtype=np.bool)
    #to do - this is a hack to get a row number for a node
    #it may not continue to work
    max_time = max([int(n.time) for n in inferred_ts.nodes()])
    rows_for_nodes = [max_time-int(n.time) for n in inferred_ts.nodes()]
    for es in inferred_ts.edgesets():
        used_variants = np.logical_and(es.left<=locations, locations<es.right)
        used[rows_for_nodes[es.parent], used_variants]=True

    for k in range(1, store.num_ancestors):
        #one file for each copied ancestor
        focal2row = {}
        visualiser = Visualiser(800, store.num_sites, font_size=9)
        big_visualiser = Visualiser(800, store.num_sites, font_size=9)

        visualiser.add_site_coordinates()
        big_visualiser.add_site_coordinates()

        a = np.zeros(store.num_sites, dtype=np.int8)

        visualiser.add_row(a, 0)
        big_visualiser.add_row(a, 0)

        for j in range(1, store.num_ancestors):
            if j in breaks:
                visualiser.add_separator()
                big_visualiser.add_separator()
            start, end, num_older_ancestors, focal_sites = store.get_ancestor(j, a)
            for f in focal_sites:
                focal2row[f]=j
                big_visualiser.add_row(a, j, focal_sites)
                big_visualiser.fade_row_false(used[j], j)
            visualiser.add_row(a, j, focal_sites)
            visualiser.fade_row_false(used[j], j)
        #highlight the path
        visualiser.show_path(k, P[k], False)
        big_visualiser.show_path(k, P[k], False)

        #add samples
        visualiser.add_separator()
        big_visualiser.add_separator()
        for j in range(S.shape[0]):
            visualiser.add_row(S[j],None,np.where(np.sum(S,0)==1)[0])
            big_visualiser.add_row(S[j],None,np.where(np.sum(S,0)==1)[0])
        print("Writing inferred ancestor copy plots", k)
        visualiser.save("tmp__NOBACKUP__/inferred_{}.svg".format(k))
        big_visualiser.save("tmp__NOBACKUP__/inferred_big_{}.svg".format(k))

    #visualize the true copying process, with real ancestral fragments
    #in the same order as in the inferred sequence.
    h, p = msprime_to_inference_matrices.make_ancestral_matrices(ts)
    freq_order, node_mutations = {}, {}
    for v in ts.variants():
        freq = np.sum(v.genotypes)
        for m in v.site.mutations:
            freq_order.setdefault(freq,[]).append({'node':m.node,'site':m.site, 'row':focal2row.get(m.site)})
            node_mutations.setdefault(m.node,[]).append(m.site)

    #for k,v in freq_order.items(): #print the list of ancestors output
    #    print(k,v)
    #    print()
    #exclude ancestors of singletons
    output_rows = [n for k in freq_order.keys() for n in freq_order[k] if k>1]
    output_rows.sort(key=operator.itemgetter('row'))
    freq_ordered_mutation_nodes = np.array([o['node'] for o in output_rows], dtype=np.int)
    #add the samples to the rows to keep
    keep_nodes = np.append(freq_ordered_mutation_nodes, range(ts.sample_size))
    H = h[keep_nodes,:]
    P, row_map = msprime_to_inference_matrices.relabel_copy_matrix(p, keep_nodes)
    groups = [row_map[n] for n in keep_nodes] + [[0]]
    visualiser = Visualiser(800, store.num_sites, font_size=9)
    visualiser.add_site_coordinates()
    a = np.zeros(store.num_sites, dtype=np.int8)
    visualiser.add_row(a, 0)
    visualiser.add_separator()
    row = 0
    for k in reversed(sorted(freq_order.keys())):
        if k>1:
            for j in freq_order[k]:
                visualiser.add_row(H[row,], keep_nodes[row], node_mutations[keep_nodes[row]])
                row += 1
            visualiser.add_separator()
    while row < H.shape[0]:
        visualiser.add_row(H[row,], keep_nodes[row], np.where(np.sum(H[-ts.sample_size:,],0)==1)[0])
        row += 1
    visualiser.save("tmp__NOBACKUP__/real.svg")

def run_large_infers():
    seed = 100
    n = 2000
    # n = 10
    for j in np.arange(10, 200, 10):
        print("n                :", n)
        print("L                :", j, "Mb")
        filename = "tmp__NOBACKUP__/n={}_L={}_original.hdf5".format(n, j)
        L = j * 10**6
        ts = msprime.simulate(
            n, length=L, recombination_rate=1e-8, mutation_rate=1e-8,
            Ne=10**4, random_seed=seed)
        print("sites            :", ts.num_sites)
        ts.dump(filename)
        positions = np.array([site.position for site in ts.sites()])
        S = generate_samples(ts, 0)
        before_cpu = time.clock()
        before_wall = time.time()
        ts_inferred = tsinfer.infer(
            S, positions, L, 1e-8, 1e-200, num_threads=40, method="C",
            show_progress=True, log_level="INFO")
        duration_cpu = time.clock() - before_cpu
        duration_wall = time.time() - before_wall
        filename = "tmp__NOBACKUP__/n={}_L={}_inferred.hdf5".format(n, j)
        ts_inferred.dump(filename)
        ts_simplified = ts_inferred.simplify()
        filename = "tmp__NOBACKUP__/n={}_L={}_simplified.hdf5".format(n, j)
        ts_simplified.dump(filename)
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        print("CPU time       :", humanize.naturaldelta(duration_cpu))
        print("wall time:     :", humanize.naturaldelta(duration_wall))
        print("RAM            :", humanize.naturalsize(memory))
        print()

def analyse_file(filename):
    ts = msprime.load(filename)

    num_children = np.zeros(ts.num_edgesets, dtype=np.int)
    for j, e in enumerate(ts.edgesets()):
        # print(e.left, e.right, e.parent, ts.time(e.parent), e.children, sep="\t")
        num_children[j] = len(e.children)

    print("total edgesets = ", ts.num_edgesets)
    print("non binary     = ", np.sum(num_children > 2))
    print("max children   = ", np.max(num_children))
    print("mean children  = ", np.mean(num_children))

    for l, r_out, r_in in ts.diffs():
        print(l, len(r_out), len(r_in), sep="\t")

    # for t in ts.trees():
    #     t.draw(
    #         "tree_{}.svg".format(t.index), 4000, 4000, show_internal_node_labels=False,
    #         show_leaf_node_labels=False)
    #     if t.index == 10:
    #         break


def draw_tree_for_position(site_index, ts):
    """
    useful for debugging
    """
    position = ts.site(site_index).position
    for t in ts.trees():
        l, r = r.interval
        if l <= position < r:
            t.draw("tmp__NOBACKUP__/tree_at_pos{}.svg".format(site_index))



def sort_encode(A):
    """
    Encodes the specified ancestor matrix site based run-length encoding.
    """
    N, m = A.shape
    initial_sorting = np.argsort(A[:,0], kind="mergesort")
    last_sorting = initial_sorting
    print(initial_sorting)

    for j in range(m):
        sorting = np.argsort(A[:, j], kind="mergesort")
        diff = sorting - last_sorting
        # print("j = ", j)
        # print("sort = ", sorting)
        # print("diff = ", diff)
        print(diff)
        last_sorting = sorting


def run_traceback(traceback):
    # print("==================")
    # traceback.print_state()
    V = traceback.best_segment
    T = traceback.site_head
    store = traceback.store
    m = store.num_sites
    # Choose the oldest value in the best_segment
    # print(V)
    parent = V[m - 1].start
    p = np.zeros(m, dtype=np.int32)
    # print("H = ", haplotype)
    # print("INITIAL parent = ", parent, "options = ", options)
    end = m
    for l in range(m - 1, 0, -1):
        switch = False
        u = T[l]
        while u is not None:
            if u.start <= parent < u.end:
                switch = True
                break
            if u.start > parent:
                break
            u = u.next
        if switch:
            # Complete a segment at this site
            assert l < end
            p[l:end] = parent
            # self.add_mapping(l, end, parent, child)
            end = l
            parent = V[l - 1].start
            # print("SWITCH @", l, "parent = ", parent, "options = ", options)
    # assert start_site < end
    # self.add_mapping(start_site, end, parent, child)
    # print("mapping", 0, m, parent)
    p[0:end] = parent
    # state = self.store.get_state(l, parent)
    # if state != haplotype[l]:
    #     self.add_mutation(child, l, haplotype[l])
    # print("AFTER")
    # self.print_state()
    return p

class CompressedStore(object):
    def __init__(self, num_sites, num_ancestors):
        self.num_sites = num_sites
        self.P = np.zeros((num_ancestors, num_sites), dtype=int)
        self.mutation_root = np.zeros(num_sites, dtype=int) - 1
        self.num_ancestors = 1

    def add(self, h, p, focal_sites):
        # print("Adding\t", h, p, focal_sites)
        self.P[self.num_ancestors] = p
        for site in focal_sites:
            self.mutation_root[site] = self.num_ancestors
        self.num_ancestors += 1

    def get_ancestor(self, j, a):
        a[:] = 0
        for l in range(self.num_sites):
            u = j
            while u != 0 and u != self.mutation_root[l]:
                u = self.P[u, l]
            if u == self.mutation_root[l]:
                a[l] = 1

    def print_state(self):
        print("Encoded state")
        for l in range(self.num_sites):
            print("site: ", l)
            print("\tP = ", self.P[:, l])
            print("\tr = ", self.mutation_root[l])



def ancestor_copy_ordering_dev(n, L, seed):

    ts = msprime.simulate(
        n, length=L, recombination_rate=1e-8, mutation_rate=1e-8,
        Ne=10**4, random_seed=seed)
    position = np.array([site.position for site in ts.sites()])
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    # print(S)
    print("Size= ", S.shape)

    store = tsinfer.build_ancestors(S, position, method="P")
    matcher = tsinfer.AncestorMatcher(store, 1e-8)
    traceback = tsinfer.Traceback(store)
    # print("Ancestors built")
    h = np.zeros(store.num_sites, dtype=np.int8)
    # for j in range(0, store.num_ancestors):
    #     _, _, num_older_ancestors, focal_sites = store.get_ancestor(j, h)
    #     print(j, ":\t", h)
    # print()
    print("num_ancestors = ", store.num_ancestors)

    # cstore = CompressedStore(store.num_sites, store.num_ancestors)
    P = np.zeros((store.num_ancestors, store.num_sites), dtype=int)
    for j in range(1, store.num_ancestors):
        _, _, num_older_ancestors, focal_sites = store.get_ancestor(j, h)
        # print(j, num_older_ancestors, focal_sites, h)
        matcher.best_path(
            num_ancestors=num_older_ancestors,
            haplotype=h, start_site=0, end_site=store.num_sites,
            focal_sites=focal_sites, error_rate=0, traceback=traceback)
        p = run_traceback(traceback)
        P[j,:] = p
        # print(j, ":\t", p)
        # cstore.add(h, p, focal_sites)
    # cstore.print_state()
    print()
    # P = cstore.P
    # print(P)
    diff_size = np.zeros(store.num_sites)
    for l in range(1, store.num_sites):
        diff = P[:,l] - P[:,l - 1]
        diff_size[l] = np.sum(diff != 0)
    # print(diff_size)
    print("Average diff size = ", np.mean(diff_size))
    # print("DONE")
    # a = np.zeros(store.num_sites, dtype=np.int8)
    # b = np.zeros(store.num_sites, dtype=np.int8)
    # for j in range(store.num_ancestors):
    #     cstore.get_ancestor(j, a)
    #     store.get_ancestor(j, b)
    #     print(a)
    #     print(b)
    #     print()
    #     # assert np.array_equal(a, b)
    # # print()
    # cstore.print_state()
    # last_permutation = cstore.permutation[0]
    # for l in range(store.num_sites):
    #     print(cstore.permutation[l] - last_permutation)
    #     last_permutation = cstore.permutation[l]


@attr.s
class Site(object):
    id = attr.ib(default=None)
    frequency = attr.ib(default=None)

@attr.s
class SiteTreeNode(object):
    sites = attr.ib(default=None)
    pattern = attr.ib(default=None)
    children = attr.ib(default=None)


class AncestorBuilder(object):
    """
    Builds inferred ancestors.
    """
    def __init__(self, S, positions):
        self.haplotypes = S
        self.num_samples = S.shape[0]
        self.num_sites = S.shape[1]
        self.sites = [None for j in range(self.num_sites)]
        self.sorted_sites = [None for j in range(self.num_sites)]
        for j in range(self.num_sites):
            self.sites[j] = Site(j, np.sum(S[:, j]))
            self.sorted_sites[j] = Site(j, np.sum(S[:, j]))
        self.sorted_sites.sort(key=lambda x: (-x.frequency, x.id))
        frequency_sites = collections.defaultdict(list)
        for site in self.sorted_sites:
            if site.frequency > 1:
                frequency_sites[site.frequency].append(site)
        # Group together identical sites within a frequency class
        self.frequency_classes = {}
        for frequency, sites in frequency_sites.items():
            patterns = collections.defaultdict(list)
            for site in sites:
                state = tuple(self.haplotypes[:, site.id])
                patterns[state].append(site.id)
            self.frequency_classes[frequency] = list(patterns.values())

    def get_frequency_classes(self):
        ret = []
        for frequency in reversed(sorted(self.frequency_classes.keys())):
            ret.append((frequency, self.frequency_classes[frequency]))
        return ret

    def build_site_tree(self):

        def print_tree(node, depth=0):
            print("    " * depth, node.pattern)
            for child in node.children:
                print_tree(child, depth + 1)


        def is_descendent(u, v):
            """
            Return true if v is a descencent of u.
            """
            return np.all(v[np.where(u == 0)] == 0)

        # print()
        root = SiteTreeNode(
            pattern=np.ones(self.num_samples, dtype=np.int8), children=list())

        for freq, sites_list in self.get_frequency_classes():
            for sites in sites_list:

                # print("TREE")
                # print_tree(root)

                pattern = self.haplotypes[:, sites[0]]
                # print("Placing", freq, pattern, sites)
                # print_tree(root)
                destination_node = None
                node = root
                while destination_node is None:
                    child_found = False
                    for child in node.children:
                        # print("\tcompare", child.pattern, pattern, is_descendent(
                        #     child.pattern, pattern))
                        if is_descendent(child.pattern, pattern):
                            node = child
                            child_found = True
                            break
                    if not child_found:
                        destination_node = node

                # print("Inserting into", destination_node.pattern)
                destination_node.children.append(
                    SiteTreeNode(pattern=pattern, sites=sites, children=list()))

        print("TREE")
        print_tree(root)
        # for j in range(self.num_sites):
        #     print(j, "\t", self.haplotypes[:, j])
        return root


    def build_ancestors(self):
        root = self.build_site_tree()
        freq_classes = self.get_frequency_classes()
        freq, sites = freq_classes[0]
        sites = sites[0]

        a = np.zeros(self.num_sites, dtype=np.int8)
        stack = list(root.children)
        while len(stack) > 0:
            node = stack.pop()
            for child in node.children:
                stack.append(child)
            focal_sites = node.sites
            # print("VISIT", node.pattern)

            # print("make ancestor", focal_sites)
            a[:] = 0
            focal_site = self.sites[focal_sites[0]]
            sites = range(focal_sites[-1] + 1, self.num_sites)
            d_right = self.__build_ancestor_sites(focal_site, sites, a)
            focal_site = self.sites[focal_sites[-1]]
            sites = range(focal_sites[0] - 1, -1, -1)
            d_left = self.__build_ancestor_sites(focal_site, sites, a)
            d_middle = set()
            for j in range(focal_sites[0], focal_sites[-1] + 1):
                if j in focal_sites:
                    a[j] = 1
                else:
                    s = self.__build_ancestor_sites(focal_site, [j], a)
                    d_middle.update(s)
            print(a)

            # print("Acnestor = ", a)
            # descendent_sites = d_left | d_middle | d_right
            # # print("descendes = ", descendent_sites)
            # freq_map = collections.defaultdict(list)
            # for l in descendent_sites:
            #     freq_map[self.sites[l].frequency].append(l)
            # # print(freq_map)
            # if len(freq_map) > 0:
            #     min_dist = self.num_sites
            #     for site in freq_map[max(freq_map.keys())]:
            #         dist = abs(focal_site.id - site)
            #         if dist < min_dist:
            #             best_site = site
            #             min_dist = dist

            #     # print("choose", best_site, min_dist)
            #     site_queue.append([best_site])


    def __build_ancestor_sites(self, focal_site, sites, a):
        S = self.haplotypes
        samples = set()
        descendents = set()
        for j in range(self.num_samples):
            if S[j, focal_site.id] == 1:
                samples.add(j)
        for l in sites:
            a[l] = 0
            if self.sites[l].frequency > focal_site.frequency:
                # print("\texamining:", self.sites[l])
                # print("\tsamples = ", samples)
                num_ones = 0
                num_zeros = 0
                for j in samples:
                    if S[j, l] == 1:
                        num_ones += 1
                    else:
                        num_zeros += 1
                # TODO choose a branch uniformly if we have equality.
                if num_ones >= num_zeros:
                    a[l] = 1
                    samples = set(j for j in samples if S[j, l] == 1)
                else:
                    samples = set(j for j in samples if S[j, l] == 0)
            else:
                u = S[:, focal_site.id]
                v = S[:, l]
                is_descendent = (
                    self.sites[l].frequency < focal_site.frequency and
                    np.all(v[np.where(u == 0)] == 0))
                # print("Is ", l, "a descendent site of ", focal_site)
                # print("\t", u)
                # print("\t", v)
                # print("\t", is_descendent)
                if is_descendent:
                    descendents.add(l)
            if len(samples) == 1:
                # print("BREAK")
                break
        return descendents

def ancestor_tree_dev(n, L, seed):

    ts = msprime.simulate(
        n, length=L, recombination_rate=1e-8, mutation_rate=1e-8,
        Ne=10**4, random_seed=seed)
    position = np.array([site.position for site in ts.sites()])
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    # print(S)

    b = AncestorBuilder(S, position)
    b.build_ancestors()


def run_segment_traceback(traceback):
    # print("==================")
    # traceback.print_state()
    V = traceback.best_segment
    T = traceback.site_head
    store = traceback.store
    m = store.num_sites
    # Choose the oldest value in the best_segment
    # print(V)
    parent = V[m - 1].start
    ret = []
    # print("H = ", haplotype)
    # print("INITIAL parent = ", parent, "options = ", options)
    end = m
    for l in range(m - 1, 0, -1):
        switch = False
        u = T[l]
        while u is not None:
            if u.start <= parent < u.end:
                switch = True
                break
            if u.start > parent:
                break
            u = u.next
        if switch:
            # Complete a segment at this site
            assert l < end
            ret.append((l, end, parent))
            # self.add_mapping(l, end, parent, child)
            end = l
            parent = V[l - 1].start
            # print("SWITCH @", l, "parent = ", parent, "options = ", options)
    # assert start_site < end
    # self.add_mapping(start_site, end, parent, child)
    # print("mapping", 0, m, parent)
    ret.append((0, end, parent))
    # state = self.store.get_state(l, parent)
    # if state != haplotype[l]:
    #     self.add_mutation(child, l, haplotype[l])
    # print("AFTER")
    # self.print_state()
    return ret


def tree_copy_process_dev(n, L, seed):

    ts = msprime.simulate(
        n, length=L, recombination_rate=1e-8, mutation_rate=1e-8,
        Ne=10**4, random_seed=seed)
    if ts.num_sites < 2:
        # Skip this
        return
    site_position = np.array([site.position for site in ts.sites()])
    edgeset_position = np.hstack([0, site_position, ts.sequence_length])
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    # print(S)
    print("Size= ", S.shape)

    store = tsinfer.build_ancestors(S, site_position, method="P")
    matcher = tsinfer.AncestorMatcher(store, 1e-8)
    traceback = tsinfer.Traceback(store)
    h = np.zeros(store.num_sites, dtype=np.int8)
    # For checking the output.
    A = np.zeros((store.num_ancestors, store.num_sites), dtype=np.int8)

    # This code checks that we can sucessfully represent the ancestors
    # using a tree sequence sequentially.

    nodes = msprime.NodeTable()
    edgesets = msprime.EdgesetTable()
    sites = msprime.SiteTable()
    mutations = msprime.MutationTable()
    nodes.add_row(flags=msprime.NODE_IS_SAMPLE, time=store.num_ancestors)

    for site in ts.sites():
        sites.add_row(position=site.index, ancestral_state='0')

    for ancestor_id in range(1, store.num_ancestors):
        start_site, end_site, num_older_ancestors, focal_sites = store.get_ancestor(
                ancestor_id, h)
        for site in focal_sites:
            assert h[site] == 1
        assert len(focal_sites) > 0
        matcher.best_path(
            num_ancestors=num_older_ancestors, haplotype=h,
            start_site=0, end_site=store.num_sites,
            focal_sites=focal_sites, error_rate=0, traceback=traceback)
        segments = run_segment_traceback(traceback)
        traceback.reset()
        for left, right, parent in reversed(segments):
            edgesets.add_row(
                left=left, right=right,
                # left=edgeset_position[left], right=edgeset_position[right],
                parent=parent, children=(ancestor_id,))
        nodes.add_row(
            flags=msprime.NODE_IS_SAMPLE,
            time=store.num_ancestors - num_older_ancestors)
        A[ancestor_id] = h
        # print("AFTER")
        # print(nodes)
        # print(edgesets)
        for focal_site in focal_sites:
            # Position == index for now.
            mutations.add_row(
                site=focal_site, node=ancestor_id, derived_state='1')
    # print("A = ")
    # print(A)

    assert nodes.num_rows == store.num_ancestors
    assert sites.num_rows == store.num_sites

    # print("BEFORE")
    # print(nodes)
    # print(edgesets)
    # print(sites)
    # print(mutations)

    msprime.sort_tables(
        nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)
    msprime.simplify_tables(filter_invariant_sites=False,
            samples=np.arange(store.num_ancestors, dtype=np.int32),
            nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)

    assert nodes.num_rows == store.num_ancestors
    assert sites.num_rows == store.num_sites

    # print("AFTER")
    # print(nodes)
    # print(edgesets)
    # print(sites)
    # print(mutations)
    new_ts = msprime.load_tables(
        nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)
    # for tree in ts.trees():
    #     print(tree)
    # print()
    for v  in new_ts.variants():
        pos = int(v.position)
        # print(pos)
        # print(pos)
        # print(v.genotypes)
        # print(A[:, pos])
        # print()
        # assert np.array_equal(v.genotypes, A[:, pos])
        if not np.array_equal(v.genotypes, A[:, pos]):
            print("Site differs", pos)
            print(A[:, pos])
            print(v.genotypes)
    # # print(A)
    # # print()
    # # print(B)
    # assert np.array_equal(A, B)
    new_ts.dump("ancestors_example.hdf5")

def rle(inarray):
    """
    run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values) """
    ia = np.array(inarray)                  # force numpy
    n = len(ia)
    assert n > 0
    y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)   # must include last element posi
    z = np.diff(np.append(-1, i))       # run lengths
    p = np.cumsum(np.append(0, z))[:-1] # positions
    for position, length, value in zip(p, z, ia[i]):
        yield position, position + length, value

@attr.s
class Edge(object):
    left = attr.ib(default=None)
    right = attr.ib(default=None)
    parent = attr.ib(default=None)
    child = attr.ib(default=None)


def is_descendant(pi, u, v):
    """
    Returns True if the specified node u is a descendent of node v. That is,
    v is on the path to root from u.
    """
    # print("IS_DESCENDENT(", u, v, ")")
    while u != v and u != msprime.NULL_NODE:
        # print("\t", u)
        u = pi[u]
    # print("END, ", u, v)
    return u == v


class TreeSequenceBuilder(object):

    def __init__(self):
        self.num_nodes = 0
        self.num_match_nodes = 0
        self.time = []
        self.sites = []
        self.mutations = {}
        self.edges = []
        self.pending_edges = []
        self.pending_mutations = []

    def print_state(self):
        print("TreeSequenceBuilder state")
        print("time = ", self.time)
        print("edges = ")
        for e in self.edges:
            print("\t", e)
        print("pending edgest = ")
        for e in self.pending_edges:
            print("\t", e)

    def add_node(self, time):
        self.time.append(time)
        self.num_nodes += 1
        return self.num_nodes - 1

    def add_site(self, position):
        self.sites.append(position)

    def add_mutation(self, site, node):
        # print("adding mutation", site, node)
        self.pending_mutations.append((site, node))

    def add_edge(self, left, right, parent, child):
        # print("adding edge", left, right, parent, child)
        self.pending_edges.append(Edge(left, right, parent, child))

    def get_sites(self, left, right):
        # TODO this would be different for non integer coordinates.
        return range(left, right)

    def best_path(self, h, p):

        # print("best_path", h)

        recombination_rate = 1e-8

        M = len(self.edges)
        I = self.insertion_order
        O = self.removal_order
        n = self.num_match_nodes
        m = len(self.sites)
        pi = np.zeros(n, dtype=int) - 1
        L = {u: 1.0 for u in range(n)}
        traceback = [None for _ in range(m)]
        edges = self.edges

        r = 1 - np.exp(-recombination_rate / n)
        recomb_proba = r / n
        no_recomb_proba = 1 - r + r / n

        j = 0
        k = 0
        while j < M:
            left = edges[I[j]].left
            while edges[O[k]].right == left:
                parent = edges[O[k]].parent
                child = edges[O[k]].child
                k = k + 1
                pi[child] = -1
                if child not in L:
                    # If the child does not already have a u value, traverse
                    # upwards until we find an L value for u
                    u = parent
                    while u not in L:
                        u = pi[u]
                    L[child] = L[u]
            right = edges[O[k]].right
            while j < M and edges[I[j]].left == left:
                parent = edges[I[j]].parent
                child = edges[I[j]].child
                # print("INSERT", parent, child)
                pi[child] = parent
                j += 1
                # Traverse upwards until we find the L value for the parent.
                u = parent
                while u not in L:
                    u = pi[u]
                # The child must have an L value. If it is the same as the parent
                # we can delete it.
                if L[child] == L[u]:
                    del L[child]

            # print("END OF TREE LOOP", left, right)
            # print("left = ", left)
            # print("right = ", right)
            # print(L)
            # print(pi)
            for site in self.get_sites(left, right):
                if site not in self.mutations:
                    traceback[site] = dict(L)
                    continue
                mutation_node = self.mutations[site]
                state = h[site]
                # print("Site ", site, "mutation = ", mutation_node, "state = ", state)

                # Insert an new L-value for the mutation node if needed.
                if mutation_node not in L:
                    u = mutation_node
                    while u not in L:
                        u = pi[u]
                    L[mutation_node] = L[u]
                traceback[site] = dict(L)

                # Update the likelihoods for this site.
                max_L = -1
                for v in L.keys():
                    x = L[v] * no_recomb_proba
                    assert x >= 0
                    y = recomb_proba
                    if x > y:
                        z = x
                    else:
                        z = y
                    if state == 1:
                        emission_p = int(is_descendant(pi, v, mutation_node))
                    else:
                        emission_p = int(not is_descendant(pi, v, mutation_node))
                    L[v] = z * emission_p
                    if L[v] > max_L:
                        max_L = L[v]
                assert max_L > 0

                # Normalise
                for v in L.keys():
                    L[v] /= max_L

                # Compress
                # TODO we probably don't need the second dict here and can just take
                # a copy of the keys.
                L_next = {}
                for u in L.keys():
                    if pi[u] != -1:
                        # Traverse upwards until we find another L value
                        v = pi[u]
                        while v not in L:
                            v = pi[v]
                        if L[u] != L[v]:
                            L_next[u] = L[u]
                    else:
                        L_next[u] = L[u]
                L = L_next

        u = [node for node, v in L.items() if v == 1.0][0]
        p[:] = -1
        p[m - 1] = u
        # Now go back through the trees.
        j = M - 1
        k = M - 1
        # print("TRACEBACK")
        I = self.removal_order
        O = self.insertion_order
        while j >= 0:
            right = edges[I[j]].right
            while edges[O[k]].left == right:
                pi[edges[O[k]].child] = -1
                k -= 1
            left = edges[O[k]].left
            while j >= 0 and edges[I[j]].right == right:
                pi[edges[I[j]].child] = edges[I[j]].parent
                j -= 1
            # print("left = ", left, "right = ", right)
            for l in range(right - 1, max(left - 1, 0), -1):
                u = p[l]
                L = traceback[l]
                v = u
                while v not in L:
                    v = pi[v]
                x = L[v]
                if x != 1.0:
                    u = [node for node, v in L.items() if v == 1.0][0]
                assert l > 0
                p[l - 1] = u
        # print(p)
        assert np.all(p >= 0)


    def update(self):
        # print("Update")
        # self.print_state()
        self.edges.extend(self.pending_edges)
        self.pending_edges = []
        for site, node in self.pending_mutations:
            self.mutations[site] = node
        self.pending_mutations = []

        # Build the indexes for tree generation.
        M = len(self.edges)
        self.insertion_order = sorted(
            range(M), key=lambda j: (
                self.edges[j].left, self.time[self.edges[j].parent]))
        self.removal_order = sorted(
            range(M), key=lambda j: (
                self.edges[j].right, -self.time[self.edges[j].parent]))
        self.num_match_nodes = self.num_nodes


        # O = sorted(range(M), key=lambda j: (r[j], -t[u[j]]))

    def finalise(self):
        self.update()
        nodes = msprime.NodeTable()
        for t in self.time:
            nodes.add_row(flags=1, time=t)
        edgesets = msprime.EdgesetTable()
        for e in self.edges:
            edgesets.add_row(e.left, e.right, e.parent, (e.child,))
        sites = msprime.SiteTable()
        for site in self.sites:
            sites.add_row(position=site, ancestral_state='0')
        mutations = msprime.MutationTable()
        for site, node in self.mutations.items():
            mutations.add_row(site=site, node=node, derived_state='1')

        msprime.sort_tables(nodes, edgesets, sites=sites, mutations=mutations)
        samples = np.arange(nodes.num_rows, dtype=np.int32)
        # print("simplify:")
        # print(samples)
        # print(nodes)
        # print(edgesets)
        msprime.simplify_tables(samples, nodes, edgesets)
        ts = msprime.load_tables(
            nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)
        return ts


def finalise_builder(tsb):
    nodes = msprime.NodeTable()
    flags = np.zeros(tsb.num_nodes, dtype=np.uint32)
    flags[:] = 1
    time = np.zeros(tsb.num_nodes, dtype=np.float64)
    tsb.dump_nodes(flags=flags, time=time)
    nodes.set_columns(flags=flags, time=time)

    edgesets = msprime.EdgesetTable()
    left = np.zeros(tsb.num_edges, dtype=np.float64)
    right = np.zeros(tsb.num_edges, dtype=np.float64)
    parent = np.zeros(tsb.num_edges, dtype=np.int32)
    child = np.zeros(tsb.num_edges, dtype=np.int32)
    tsb.dump_edges(left=left, right=right, parent=parent, child=child)
    edgesets.set_columns(
        left=left, right=right, parent=parent, children=child,
        children_length=np.ones(tsb.num_edges, dtype=np.uint32))

    sites = msprime.SiteTable()
    sites.set_columns(
        position=np.arange(tsb.num_sites),
        ancestral_state=np.zeros(tsb.num_sites, dtype=np.int8) + ord('0'),
        ancestral_state_length=np.ones(tsb.num_sites, dtype=np.uint32))
    mutations = msprime.MutationTable()
    site = np.zeros(tsb.num_mutations, dtype=np.int32)
    node = np.zeros(tsb.num_mutations, dtype=np.int32)
    derived_state = np.zeros(tsb.num_mutations, dtype=np.int8)
    tsb.dump_mutations(site=site, node=node, derived_state=derived_state)
    derived_state += ord('0')
    mutations.set_columns(
        site=site, node=node, derived_state=derived_state,
        derived_state_length=np.ones(tsb.num_mutations, dtype=np.uint32))

    # print(nodes)
    # print(edgesets)
    # print(sites)
    # print(mutations)

    msprime.sort_tables(nodes, edgesets, sites=sites, mutations=mutations)
    samples = np.arange(nodes.num_rows, dtype=np.int32)
    # print("simplify:")
    # print(samples)
    # print(nodes)
    # print(edgesets)
    msprime.simplify_tables(samples, nodes, edgesets)
    ts = msprime.load_tables(
        nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)
    return ts



def new_copy_process_dev(n, L, seed):

    ts = msprime.simulate(
        n, length=L, recombination_rate=1e-8, mutation_rate=1e-8,
        Ne=10**4, random_seed=seed)
    if ts.num_sites < 2:
        # Skip this
        return
    print("num sites=  ", ts.num_sites)
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    # print(S)

    site_position = np.array([site.position for site in ts.sites()])
    store = tsinfer.build_ancestors(S, site_position, method="C")
    h = np.zeros(store.num_sites, dtype=np.int8)
    p = np.zeros(store.num_sites, dtype=np.int32)
    L = store.num_sites

    print("n = ", S.shape[0], "num_sites = ", store.num_sites, "num_ancestors = ",
            store.num_ancestors)
    tsb = _tsinfer.TreeSequenceBuilder(store.num_sites, store.num_ancestors + 1,
            100 * store.num_ancestors)

    tsb.update(1, store.num_epochs - 1, [], [], [], [], [], [])

    # child = 1;
    # for (j = store.num_epochs - 2; j > 0; j--) {
    #     /* printf("STARTING EPOCH %d\n", (int) j); */
    #     ret = ancestor_store_get_epoch_ancestors(&store, j, epoch_ancestors,
    #             &num_epoch_ancestors);
    #     if (ret != 0) {
    #         fatal_error("error getting epoch ancestors");
    #     }

    epoch_ancestors = np.zeros(store.num_ancestors, dtype=np.int32)
    for epoch in range(store.num_epochs - 2, 0, -1):
        num_epoch_ancestors = store.get_epoch_ancestors(epoch, epoch_ancestors)
        e_left = []
        e_right = []
        e_parent = []
        e_child = []
        s_site = []
        s_node = []
        for node in map(int, epoch_ancestors[:num_epoch_ancestors]):
            _, _, _, focal_sites = store.get_ancestor(node, h)
            # print("node = ", node)
            # print("h = ", h)
            # print("focal_sites = ", focal_sites)
            for s in focal_sites:
                assert h[s] == 1
                h[s] = 0
                s_site.append(s)
                s_node.append(node)
            edges = tsb.find_path(node, h)
            for left, right, parent, child in zip(*edges):
                e_left.append(left)
                e_right.append(right)
                e_parent.append(parent)
                e_child.append(child)
        print("EPOCH", epoch, num_epoch_ancestors, node, store.num_ancestors)
        tsb.update(
            num_epoch_ancestors, epoch,
            e_left, e_right, e_parent, e_child,
            s_site, s_node)

    ts = finalise_builder(tsb)

    # For checking the output.
    A = np.zeros((store.num_ancestors, store.num_sites), dtype=np.int8)
    for j in range(store.num_ancestors):
        store.get_ancestor(j, h)
        A[j] = h

    B = np.zeros((ts.sample_size, ts.num_sites), dtype=np.int8)
    for v in ts.variants():
        B[:, v.index] = v.genotypes

    for ancestor_id in range(store.num_ancestors):
        node_id = ancestor_id
        # node_id = ancestor_node_map[ancestor_id]
        # print(ancestor_id, "->",  node_id)
        # print(A[ancestor_id])
        # print(B[node_id])
        if not np.array_equal(A[ancestor_id], B[node_id]):
            print("ERROR")
        assert np.array_equal(A[ancestor_id], B[node_id])
    # assert np.array_equal(A, B)
    # print(A)
    # print(B)
    # print("match_time = ", match_time)


def generate_trees(l, r, u, c, t):
    """
    Algorithm T. Sequentially visits all trees in the specified
    tree sequence.
    """
    # Calculate the index vectors
    M = len(l)
    I = sorted(range(M), key=lambda j: (l[j], t[u[j]]))
    O = sorted(range(M), key=lambda j: (r[j], -t[u[j]]))
    pi = [-1 for j in range(t.shape[0])]
    j = 0
    k = 0
    while j < M:
        x = l[I[j]]
        while r[O[k]] == x:
            h = O[k]
            for q in c[h]:
                pi[q] = -1
            k = k + 1
        while j < M and l[I[j]] == x:
            h = I[j]
            for q in c[h]:
                pi[q] = u[h]
            j += 1
        yield pi


if __name__ == "__main__":

    np.set_printoptions(linewidth=20000)
    np.set_printoptions(threshold=20000000)

    # for j in range(1, 100000):
    #     print(j)
    #     new_segments(20, 300, j)
        # new_segments(10, 30, j, num_threads=1, method="P")
        # test_ancestor_store(20, 30, j, method="P")
        # test_ancestor_store(1000, 5000, j, method="C")

    # test_ancestor_store(20, 30, 861, method="P")

    # new_segments(20, 100, 1, num_threads=1, method="C", log_level="INFO")
    # new_segments(20, 10, 1, num_threads=1, method="P")

    # export_samples(10, 100, 304)

    # run_large_infers()
    # analyse_file("tmp__NOBACKUP__/n=2000_L=10_original.hdf5")
    # analyse_file("tmp__NOBACKUP__/n=2000_L=10_simplified.hdf5")


    # visualise_copying(8, 4, 5)

    # build_ancestors_dev(10000, 10 * 10**6, 3)
    # build_ancestors_dev(10, 1 * 10**5, 3)
    # examine_ancestors()

    # for n in [10, 100, 1000, 10**4, 10**5]:
    # n = 1000
    # n = 1000
    # for j in range(1, 10):
    #     ancestor_copy_ordering_dev(n, j * 10**7, 2)
    #     print()
    # ancestor_tree_dev(100, 5 * 10**5, 1)
    # ancestor_copy_ordering_dev(100, 20 * 10**4, 2)
    # tree_copy_process_dev(25, 1 * 10**4, 9)
    # tree_copy_process_dev(10, 5 * 10**4, 5)
    # for j in range(10000):
    #     print(j)
    #     tree_copy_process_dev(50, 30 * 10**4, j + 2)

    new_copy_process_dev(10000, 1000 * 10**4, 1)
    # for x in range(1, 10):
    #     new_copy_process_dev(20, x * 20 * 10**4, 74)
    # for j in range(1, 10000):
    #     print(j)
    #     new_copy_process_dev(40, 100 * 10**4, j)
