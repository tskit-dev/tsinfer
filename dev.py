

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
import time
# import profilehooks

import tsinfer
import _tsinfer
import msprime



def build_ancestors(n, L, seed):

    ts = msprime.simulate(
        n, length=L, recombination_rate=5.5, mutation_rate=1, random_seed=seed)
    print("num_sites = ", ts.num_sites)

    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    for h in ts.haplotypes():
        print(h)
    print()

    # # print("OLD version")
    # H1 = np.zeros((builder.num_ancestors, builder.num_sites), dtype=int)
    # for j, A in enumerate(builder.build_all_ancestors()):
    #     H1[j, :] = A
    #     a = "".join(str(x) if x != -1 else '*' for x in A)
    #     print(a)
    # # print()
    # # print(H1)

    num_sites = ts.num_sites

    # H2 = np.zeros((builder.num_ancestors, builder.num_sites), dtype=int)
    # store = tsinfer.AncestorStore(num_sites)
    # a = np.zeros(num_sites, dtype=np.int8)
    store = _tsinfer.AncestorStore(num_sites)
    store.init_build(100)

    builder = tsinfer.AncestorBuilder(S)
    # for j, A in enumerate(builder.build_ancestors()):
    #     H2[j, :] = A
    for A in builder.build_ancestors():
        store.add(A)

    matcher = _tsinfer.AncestorMatcher(store, 0.01, 1e-200)
    print(store.num_sites)
    print(store.num_ancestors)
    P = np.zeros(store.num_sites, dtype=np.int32)
    M = np.zeros(store.num_sites, dtype=np.uint32)
    for h in S:
        print(h)
        num_mutations = matcher.best_path(store.num_ancestors, h, P, M)
        print("num_mutation = ", num_mutations)
        print(P)

    # np.savetxt(sys.stdout, (site, start, end, state), fmt='%i %i %i %i')
    # x = y = z = np.arange(0.0,5.0,1.0)
    # np.savetxt('test.out', (x,y,z))

        # a = "".join(str(x) if x != -1 else '*' for x in A)
        # print(a)

    # print(H2)
    # print(np.all(H1 == H2))
    # print(np.where(H1 != H2))
    # print(H1 == H2)


def new_segments(n, L, seed):

    np.set_printoptions(linewidth=2000)
    np.set_printoptions(threshold=20000)

    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    if ts.num_sites == 0:
        print("zero sites; skipping")
        return
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    # tsp = tsinfer.infer(S, 0.01, 1e-200, matcher_algorithm="python")
    tsp = tsinfer.infer(S, 0.01, 1e-200, matcher_algorithm="C")

    Sp = np.zeros((tsp.sample_size, tsp.num_sites), dtype="i1")
    for variant in tsp.variants():
        Sp[:, variant.index] = variant.genotypes
    assert np.all(Sp == S)
    # print(S)
    # print()
    # print(Sp)

    # for t in tsp.trees():
    #     print(t.interval, t)
    ts_simplified = tsp.simplify()
    # for h in ts_simplified.haplotypes():
    #     print(h)
    # for e in ts_simplified.edgesets():
    #     print(e.left, e.right, e.parent, e.children, sep="\t")
    # print()

    Sp = np.zeros((ts_simplified.sample_size, ts_simplified.num_sites), dtype="i1")
    for variant in ts_simplified.variants():
        Sp[:, variant.index] = variant.genotypes
    assert np.all(Sp == S)



def export_ancestors(n, L, seed):

    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    print("num_sites = ", ts.num_sites)
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    builder = tsinfer.AncestorBuilder(S)
    print("total ancestors = ", builder.num_ancestors)
    # matcher = tsinfer.AncestorMatcher(ts.num_sites)
    A = np.zeros((builder.num_ancestors, ts.num_sites), dtype=int)
    P = np.zeros((builder.num_ancestors, ts.num_sites), dtype=int)
    for j, a in enumerate(builder.build_all_ancestors()):
        # builder.build(j, A[j,:])
        A[j, :] = a
        if j % 100 == 0:
            print("done", j)
        # p = matcher.best_path(a, 0.01, 1e-200)
        # P[j,:] = p
        # matcher.add(a)
    # print("A = ")
    # print(A)
    # print("P = ")
    # print(P)
    np.savetxt("tmp__NOBACKUP__/ancestors.txt", A, fmt="%d", delimiter="\t")
    # np.savetxt("tmp__NOBACKUP__/path.txt", P, fmt="%d", delimiter="\t")

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
    # main()
    # example()
    # bug()
    # for n in [100, 1000, 10000, 20000, 10**5]:
    #     ts_ls(n)
    # ts_ls(20)
    # leaf_lists_dev()

    # for m in [40]:
    #     segment_algorithm(100, m)
        # print()
    # segment_stats()
    # for j in range(1, 100000):
    #     print(j)
    #     new_segments(10, 100, j)
    # new_segments(10, 100, 1)
    # new_segments(4, 4, 304)
    # export_ancestors(10, 500, 304)
    # export_samples(10, 10, 1)

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


    build_ancestors(5, 10, 2)
