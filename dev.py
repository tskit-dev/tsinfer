

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
import msprime_to_inference_matrices


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


def draw_tree_for_position(pos, ts):
    """
    useful for debugging
    """
    for t in ts.trees():
        if t.get_interval()[0]<list(ts.sites())[pos].position and t.get_interval()[1]>list(ts.sites())[pos].position:
            t.draw("tmp__NOBACKUP__/tree_at_pos{}.svg".format(pos))




if __name__ == "__main__":

    np.set_printoptions(linewidth=20000)
    np.set_printoptions(threshold=20000000)

    # for j in range(1, 100000):
    #     print(j)
    #     new_segments(20, 300, j)
    #     # new_segments(10, 30, j, num_threads=1, method="P")
    #     # test_ancestor_store(20, 30, j, method="P")
    #     # test_ancestor_store(1000, 5000, j, method="C")

    # test_ancestor_store(20, 30, 861, method="P")

    # new_segments(20, 10, 1, num_threads=1, method="C", log_level="INFO")
    # new_segments(8, 10, 1, num_threads=1, method="P")

    # export_samples(10, 100, 304)

    # run_large_infers()
    # analyse_file("tmp__NOBACKUP__/n=2000_L=10_original.hdf5")
    # analyse_file("tmp__NOBACKUP__/n=2000_L=10_simplified.hdf5")


    # visualise_copying(8, 4, 5)

    # build_ancestors_dev(10000, 10 * 10**6, 3)
    build_ancestors_dev(10, 1 * 10**5, 3)
    # examine_ancestors()
