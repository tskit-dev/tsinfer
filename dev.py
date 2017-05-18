
import tsinfer
import subprocess
import os
import numpy as np
import itertools
import multiprocessing
import pandas as pd
import random

import msprime

def bug():
    samples_file = "../treeseq-inference/data/raw__NOBACKUP__/metrics_by_mutation_rate/simulations/msprime-n10_Ne5000.0_l10000_rho0.000000025_mu0.000000237734-gs1865676553_ms1865676553err0.1.npy"
    pos_file = "../treeseq-inference/data/raw__NOBACKUP__/metrics_by_mutation_rate/simulations/msprime-n10_Ne5000.0_l10000_rho0.000000025_mu0.000000237734-gs1865676553_ms1865676553err0.1.pos.npy"
    length = 10000
    rho = 0.0005
    error_probability = 0.1

    S = np.load(samples_file)
    pos = np.load(pos_file)
    panel = tsinfer.ReferencePanel(
        S, pos, length, rho, ancestor_error=0, sample_error=error_probability)
    P, mutations = panel.infer_paths(1)
    ts_new = panel.convert_records(P, mutations)
    ts_simplified = ts_new.simplify()
    ts_simplified.dump(args.output)
    # Quickly verify that we get the sample output.
    Sp = np.zeros(S.shape)
    for j, h in enumerate(ts_simplified.haplotypes()):
        Sp[j] = np.fromstring(h, np.uint8) - ord('0')
    assert np.all(Sp == S)


def get_random_data_example(num_samples, num_sites):
    S = np.random.randint(2, size=(num_samples, num_sites)).astype(np.uint8)
    # Weed out any invariant sites
    for j in range(num_sites):
        if np.sum(S[:, j]) == 0:
            S[0, j] = 1
        elif np.sum(S[:, j]) == num_samples:
            S[0, j] = 0
    return S

def make_ancestors(S):
    n, m = S.shape
    frequency = np.sum(S, axis=0)
    num_ancestors = np.sum(frequency > 1)
    site_order = frequency.argsort(kind="mergesort")[::-1]
    N = n + num_ancestors + 1
    H = np.zeros((N, m), dtype=np.int8)

    for j in range(n):
        H[j] = S[j]
    mask = np.zeros(m, dtype=np.int8)
    for j in range(num_ancestors):
        site = site_order[j]
        mask[site] = 1
        # Find all samples that have a 1 at this site
        R = S[S[:,site] == 1]
        # Mask out mutations that haven't happened yet.
        M = np.logical_and(R, mask).astype(int)
        A = -1 * np.ones(m, dtype=int)
        A[site] = 1
        l = site - 1
        consistent_samples = {k: {(1, 1)} for k in range(R.shape[0])}
        while l >= 0 and len(consistent_samples) > 0:
            # print("l = ", l, consistent_samples)
            # Get the consensus among the consistent samples for this locus.
            # Only mutations older than this site are considered.
            s = 0
            for k in consistent_samples.keys():
                s += M[k, l]
            A[l] = int(s >= len(consistent_samples) / 2)
            # Now we have computed the ancestor, go through the samples and
            # update their four-gametes patterns with the ancestor. Any
            # samples inconsistent with the ancestor are dropped.
            dropped = []
            for k, patterns in consistent_samples.items():
                patterns.add((A[l], S[k, l]))
                if len(patterns) == 4:
                    dropped.append(k)
            for k in dropped:
                del consistent_samples[k]
            l -= 1
        l = site + 1
        consistent_samples = {k: {(1, 1)} for k in range(R.shape[0])}
        while l < m and len(consistent_samples) > 0:
            # print("l = ", l, consistent_samples)
            # Get the consensus among the consistent samples for this locus.
            s = 0
            for k in consistent_samples.keys():
                s += M[k, l]
            # print("s = ", s)
            A[l] = int(s >= len(consistent_samples) / 2)
            # Now we have computed the ancestor, go through the samples and
            # update their four-gametes patterns with the ancestor. Any
            # samples inconsistent with the ancestor are dropped.
            dropped = []
            for k, patterns in consistent_samples.items():
                patterns.add((A[l], S[k, l]))
                if len(patterns) == 4:
                    dropped.append(k)
            for k in dropped:
                del consistent_samples[k]
            l += 1
        H[N - j - 2] = A
    return H



def get_gap_density(n, length, seed):
    ts = msprime.simulate(
        sample_size=n, recombination_rate=1, mutation_rate=1,
        length=length, random_seed=seed)
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    H = make_ancestors(S)
    A = H[ts.sample_size:]
    # print(H)
    # print()
    # print(A)
    gaps = np.sum(A == -1, axis=0)
    # normalalise and take the mean
    return ts.num_sites, A.shape[0], np.mean(gaps / A.shape[0])

def gap_density_worker(work):
    return get_gap_density(*work)

def main():
    # for seed in range(10000):
    # print("seed = ", seed)
    # np.random.seed(seed)
    # S = get_random_data_example(100, 1000)
    rho = 1
    num_replicates = 80
    print("n", "L", "sites", "ancestors", "density", sep="\t")
    for n in [10, 50, 100, 1000]:
        for length in [10, 100, 1000]:
            # for j in range(num_replicates):
            #     s, g = get_gap_density(n, length, j + 1)
            #     gap_density[j] = g
            #     num_sites[j] = s
            with multiprocessing.Pool(processes=40) as pool:
                work = [(n, length, j + 1) for j in range(num_replicates)]
                results = pool.map(gap_density_worker, work)

            num_sites = np.zeros(num_replicates)
            num_ancestors = np.zeros(num_replicates)
            gap_density = np.zeros(num_replicates)
            for j, (s, a, g) in enumerate(results):
                num_sites[j] = s
                num_ancestors[j] = a
                gap_density[j] = g
            print(
                n, length, np.mean(num_sites), np.mean(num_ancestors),
                np.mean(gap_density), sep="\t")

    # for tree in ts.trees():
    #     print("tree: ", tree.interval, tree.parent_dict)

    # sites = [site.position for site in ts.sites()]
    # panel = tsinfer.ReferencePanel(
    #     S, sites, ts.sequence_length, rho=rho, algorithm="python")
    # P, mutations = panel.infer_paths(num_workers=1)

    # ts_new = panel.convert_records(P, mutations)

    # num_samples, num_sites = S.shape
    # print("num_sites = ", num_sites)
    # # sites = np.arange(num_sites)
    # panel = tsinfer.ReferencePanel(
    #     S, sites, num_sites, rho=rho, sample_error=0, ancestor_error=0,
    #     algorithm="c")
    # P, mutations = panel.infer_paths(num_workers=1)

    # illustrator = tsinfer.Illustrator(panel, P, mutations)

    # for j in range(panel.num_haplotypes):
    #     pdf_file = "tmp__NOBACKUP__/temp_{}.pdf".format(j)
    #     png_file = "tmp__NOBACKUP__/temp_{}.png".format(j)
    #     illustrator.run(j, pdf_file, H)
    #     subprocess.check_call("convert -geometry 3000 -density 600 {} {}".format(
    #         pdf_file, png_file), shell=True)

    #     print(png_file)
    #     os.unlink(pdf_file)

    # ts = panel.convert_records(P, mutations)
    # for e in ts.edgesets():
    #     print(e)
    # for s in ts.sites():
    #     print(s.position)
    #     for mutation in s.mutations:
    #         print("\t", mutation)
    # for h in ts.haplotypes():
    #     print(h)
    # for variant in ts.variants():
    #     assert np.all(variant.genotypes == S[:, variant.index])

#     ts_simplified = ts.simplify()

#     S2 = np.empty(S.shape, np.uint8)
#     for j, h in enumerate(ts_simplified.haplotypes()):
#         S2[j,:] = np.fromstring(h, np.uint8) - ord('0')
#     assert np.all(S == S2)

def verify_breaks(n, H, P):
    """
    Make sure that the trees defined by P do not include any matrix
    entries marked as rubbish in H.
    """
    N, m = H.shape

    for l in range(m):
        for j in range(n):
            u = j
            stack = []
            # Trace this leaf back up through the tree at this locus.
            while u != -1:
                stack.append(u)
                if H[u, l] == -1:
                    print("sample ", j, "locus ", l)
                    print("ERROR at ", u)
                    print("STACK = ", stack)
                assert H[u, l] != -1
                u = P[u, l]

def segments_intersection(A, B):
    """
    Returns an iterator over the intersection of the specified ordered lists of
    segments [(start, end, value), ...]. For each (start, end) intersection that
    we find, return the pair of values in order A, B.
    """
    # print("INTERSECT")
    # print("\t", A)
    # print("\t", B)
    A_iter = iter(A)
    B_iter = iter(B)
    A_head = next(A_iter)
    B_head = next(B_iter)
    while A_head is not None and B_head is not None:
        A_s, A_e, A_v = A_head
        B_s, B_e, B_v = B_head
        # print("A_head = ", A_head)
        # print("B_head = ", B_head)
        if A_s <= B_s:
            if A_e > B_s:
                yield max(A_s, B_s), min(A_e, B_e), A_v, B_v
        else:
            if B_e > A_s:
                yield max(A_s, B_s), min(A_e, B_e), A_v, B_v

        if A_e <= B_e:
            A_head = next(A_iter, None)
        if B_e <= A_e:
            B_head = next(B_iter, None)



def ls_copy(h, segments, N):
    """
    Runs the copying process on the specified haplotype for the specified
    reference panel expressed in run-length encoded segments.
    """
    rho = 1e-3
    mu = 1e-7
    m = len(segments)
    print("copy ", h)
    for segs in segments:
        print(segs)
    # The intial value of V is determinined only by the emission probabilities.
    p = rho / N
    q = 1 - rho + rho / N

    V = []
    for start, end, v in segments[0]:
        if v == h[0]:
            V.append((start, end, 1))
    for l in range(1, m):
        print("V = ", V)
        max_p = -1
        for seg in V:
            if seg[-1] >= max_p:
                V_max = seg
                max_p = V_max[-1]
        assert max_p > 0
        V = [[s, e, p / max_p] for (s, e, p) in V]
        # Merge edjacent equal values.
        Vp = [V[0]]
        for s, e, p in V[1:]:
            sp, ep, pp = Vp[-1]
            if sp == e and p == pp:
                print("Squashing", (sp, ep, pp), ":", s, e, p)
                Vp[-1][1] = e
            else:
                Vp.append((s, e, p))
        V = Vp
        print("locus ", l)
        print("V = ", V)
        print("V_max = ", V_max)
        print("A = ", segments[l])
        V_next = []
        for s, e, proba, v in segments_intersection(V, segments[l]):
            emission = 1 - mu
            if v != h[l]:
                emission = mu
            x = q * proba
            y = p # V_max is by 1 by normalisation
            V_next.append((s, e, emission * max(x, y)))
            print("\tINTER", v == h[l], s, e, proba)
        V = V_next


    # for l in range(m):
    #     print("locus", l)
    #     recomb_proba = len(segments[l]) * rho
    #     for s, e, v in segments[l]:
    #         print("\t", s, e, v)


def run_length_decode(segments, N):
    m = len(segments)
    A = -1 * np.ones((N, m), dtype=int)
    for k, col in enumerate(segments):
        for start, end, value in col:
            A[start:end, k] = value
    return A


def run_length_encode(H, n):
    print("Creating ancestors and rle for H")
    A = H[n:]
    S = H[:n]
    m = A.shape[1]
    print("S = \n", S, sep="")
    print("A = \n", A, sep="")
    segments = [[] for _ in range(m)]
    for l in range(m):
        s = 0
        v = A[s, l]
        for j in range(1, A.shape[0]):
            if A[j, l] != v:
                if v != -1:
                    segments[l].append((s, j, v))
                v = A[j, l]
                s = j
        if v != -1:
            segments[l].append((s, j + 1, v))
    # Sanity check -- we should get the same array back.
    Ap = run_length_decode(segments, A.shape[0])
    assert np.all(A == Ap)

    # for l in range(1, m):
    #     for x in segments_intersection(segments[l - 1], segments[l]):
    #         print("INTERSECT", x)

    ls_copy(S[0], segments, A.shape[0])

def example():
    np.set_printoptions(linewidth=100)
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999
    pd.options.display.width = 999

    rho = 3
    # for seed in range(1, 10000):
    for seed in [2]:
        ts = msprime.simulate(
            sample_size=15, recombination_rate=rho, mutation_rate=1,
            length=7, random_seed=seed)
        print("seed = ", seed)
        sites = [site.position for site in ts.sites()]
        S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
        for variant in ts.variants():
            S[:, variant.index] = variant.genotypes
        H = make_ancestors(S)
        # print(H.shape)
        # print(H)
        # run_length_encode(H, ts.sample_size)

        # df = pd.DataFrame(H)
        # print(df[ts.sample_size:])

        panel = tsinfer.ReferencePanel(
            S, sites, ts.sequence_length, rho=rho, sample_error=0, ancestor_error=0,
            algorithm="python", haplotypes=H)
        # print(panel.haplotypes)
        # index = H != -1
        # assert np.all(H[index] == panel.haplotypes[index])
        threader = tsinfer.inference.PythonThreader(panel)
        p = np.zeros(H.shape[1], dtype=np.uint32)
        threader.run(0, ts.sample_size, rho, 0, p)

        P, mutations = panel.infer_paths(num_workers=1)
        P = P.astype(np.int32)

        # new_ts = panel.convert_records(P, mutations)
        # Sp = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
        # for variant in new_ts.variants():
        #     Sp[:, variant.index] = variant.genotypes
        # assert np.all(Sp == S)
        # tss = new_ts.simplify()

        # # verify_breaks(ts.sample_size, H, P)

        illustrator = tsinfer.Illustrator(panel, P, mutations)
        # for j in range(panel.num_haplotypes):
        for j in [0]:
            # pdf_file = "tmp__NOBACKUP__/temp_{}.pdf".format(j)
            # png_file = "tmp__NOBACKUP__/temp_{}.png".format(j)
            pdf_file = "tmp__NOBACKUP__/temp_{}.pdf".format(seed)
            png_file = "tmp__NOBACKUP__/temp_{}.png".format(seed)
            illustrator.run(j, pdf_file, panel.haplotypes)
            subprocess.check_call("convert -geometry 3000 -density 600 {} {}".format(
                pdf_file, png_file), shell=True)
            print(png_file)
            os.unlink(pdf_file)

    # # print("p = ", p, "q  = ", q)
    # V = (H[:,0] == h[0]).astype(int)
    # T = np.zeros_like(H)
    # for l in range(1, m):
    #     V_next = np.zeros(n)
    #     # Find the maximum of V and normalise
    #     max_V_index = np.argmax(V)
    #     V = V / V[max_V_index]
    #     # print("l = ", l, "max_V_index = ", max_V_index)
    #     # print("h = ", h[l])
    #     # print("H = ", H[:,l])
    #     # print("V = ", V)
    #     for j in range(n):
    #         x = V[j] * qr
    #         y = V[max_V_index] * pr
    #         if x > y:
    #             T[j, l] = j
    #             z = x
    #         else:
    #             T[j, l] = max_V_index
    #             z = y
    #         if H[j][l] == h[l]:
    #             V_next[j] = z * qm
    #         else:
    #             V_next[j] = z * pm
    #     V = V_next
    # recomb_proba = r / n
    # no_recomb_proba = 1 - r + r / n
    # print(recomb_proba, no_recomb_proba)
    # V = [[left, right, int(state == h[0])] for left, right, state in R[0]]
    # T = [[] for _ in R]
    # for l in range(1, len(R)):
    #     # Find the maximum probabilty and normalise
    #     max_V_index = 0
    #     max_p = -1
    #     for j, seg in enumerate(V):
    #         if seg[-1] >= max_p:
    #             max_p = seg[-1]
    #             max_V_index = j
    #     for seg in V:
    #         seg[-1] /= max_p
    #     # print("max_index = ", max_V_index)
    #     # print("V = ", V)
    #     # print("R = ", R[l])
    #     Vp = []
    #     for left, right, v, state in segments_intersection(V, R[l]):
    #         x = v * no_recomb_proba
    #         assert V[max_V_index][-1] == 1.0
    #         y = V[max_V_index][-1] * recomb_proba
    #         if x > y:
    #             T[l].append([[left, right], [left, right]])
    #         else:
    #             T[l].append([[left, right], V[max_V_index][:2]])
    #         # print("\t", left, right, v, state, x, y, sep="\t")
    #         emission = max(x, y) * int(state == h[l])
    #         Vp.append([left, right, emission])
    #     # Compress the adjacent segments.
    #     # print("Vp = ", Vp)
    #     V = [Vp[0]]
    #     for left, right, v in Vp[1:]:
    #         if V[-1][-1] == v:
    #             # print("Sqaushing:", V[-1], left, right, v)
    #             assert V[-1][1] == left
    #             V[-1][1] = right
    #         else:
    #             V.append([left, right, v])
    # print("TRACEBACK")
    # print(V)
    # max_V_index = 0
    # max_p = -1
    # for j, seg in enumerate(V):
    #     if seg[-1] >= max_p:
    #         max_p = seg[-1]
    #         max_V_index = j
    # print("max V = ", V[max_V_index])
    # left, right = V[max_V_index][:2]
    # traceback = [[(left, right)]]
    # for segs in T[::-1]:
    #     print("T:", left, right)
    #     next_left, next_right = None, None
    #     # Find the segment that overlaps with left, right
    #     print("\t", segs)
    #     for current, prev in segs:
    #         if current[0] == left and current[1] == right:
    #             print("Found overlap!")
    #             next_left, next_right = current
    #             traceback = [[(left, right)]]

    #     left, right = next_left, next_right


def decode_traceback(E, n):
    """
    Decode the specified encoded traceback matrix into the standard integer
    matrix.
    """
    m = len(E)
    T = np.zeros((n, m), dtype=int)
    for l in range(1, m):
        T[:,l] = np.arange(n)
        for start, end, value in E[l]:
            T[start:end, l] = value
    return T

def run_traceback_encoded(E, n, starting_point):
    """
    Returns the array of haplotype indexes that the specified encoded traceback
    defines for the given startin point at locus m - 1.
    """
    m = len(E)
    P = np.zeros(m, dtype=int)
    P[-1] = starting_point
    for l in range(m - 1, 0, -1):
        v = None
        for start, end, value in E[l]:
            if start <= P[l] < end:
                v = value
                break
            if start > P[l]:
                break
        if v is None:
            v = P[l]
        P[l - 1] = v
    return P

def run_traceback(T, starting_point):
    n, m = T.shape
    P = np.zeros(m, dtype=int)
    P[-1] = starting_point
    for l in range(m - 2, -1, -1):
        P[l] = T[P[l + 1], l + 1]
    return P


def encode_traceback(T):
    """
    Encode the traceback matrix column-wise with segments (a, b, v) denoting
    a section of the column s.t. T[a:b, k] = v. All other values are assumed
    to be such that T[x, k] = x
    """
    n, m = T.shape
    E = [[] for j in range(m)]
    for l in range(1, m):
        col = T[:, l]
        seg = None
        for j in range(n):
            if col[j] != j:
                if seg is None:
                    seg = [j, None, col[j]]
                elif seg[-1] != col[j]:
                    seg[1] = j
                    E[l].append(seg)
                    seg = [j, None, col[j]]
            elif seg is not None:
                seg[1] = j
                E[l].append(seg)
                seg = None
        if seg is not None:
            seg[1] = n
            E[l].append(seg)
        # print(col)
        # print(E[l])
        # print()
    return E

def match_haplotype_encoded(R, n, h, rho, theta):
    m = len(R)
    r = 1 - np.exp(-rho / n)
    pr = r / n
    qr = 1 - r + r / n
    # pm = mutation; qm no mutation
    pm = 0.5 * theta / (n + theta)
    qm = n / (n + theta) + 0.5 * theta / (n + theta)

    V = [
        [left, right, qm if h[0] == state else pm] for left, right, state in R[0]]
    T = [[] for l in range(m)]
    print("V = ", V)
    for l in range(1, m):
        max_v = -1
        best_haplotype = -1
        for start, end, v in V:
            if v >= max_v:
                max_v = v
                best_haplotype = end - 1
        # Renormalise V
        for seg in V:
            seg[-1] /= max_v
        V_next = []
        print("R = ", R[l])
        print("V = ", V)
        for start, end, v, state in segments_intersection(V, R[l]):
            # print("\t", start, end, v, state)
            x = v * qr
            y = pr  # v for maximum is 1 by normalisation
            if x >= y:
                z = x
            else:
                z = y
                if len(T[l]) == 0:
                    T[l].append([start, end, best_haplotype])
                else:
                    if T[l][-1][1] == start:
                        T[l][-1][1] = end
                    else:
                        T[l].append([start, end, best_haplotype])
            if state == h[l]:
                V_next.append([start, end, z * qm])
            else:
                V_next.append([start, end, z * pm])
        # Compress the adjacent segments.
        # print("Vp = ", Vp)
        V = [V_next[0]]
        for start, end, v in V_next[1:]:
            if V[-1][-1] == v:
                # print("Sqaushing:", V[-1], left, right, v)
                assert V[-1][1] == start
                V[-1][1] = end
            else:
                V.append([start, end, v])
        # print("T = ", T[l])
    max_v = -1
    best_haplotype = -1
    for start, end, v in V:
        if v >= max_v:
            max_v = v
            best_haplotype = end - 1
    return T, best_haplotype



def match_haplotype_simple(H, h, rho, theta):
    n, m = H.shape
    r = 1 - np.exp(-rho / n)
    pr = r / n
    qr = 1 - r + r / n
    # pm = mutation; qm no mutation
    pm = 0.5 * theta / (n + theta)
    qm = n / (n + theta) + 0.5 * theta / (n + theta)
    # print("p = ", p, "q  = ", q)
    condition = H[:,0] == h[0]
    V = np.zeros(n)
    V[condition] = qm
    V[np.logical_not(condition)] = pm
    T = np.zeros_like(H)
    # print("V = ", V)
    for l in range(1, m):
        V_next = np.zeros(n)
        # Find the maximum of V and normalise
        max_V_index = np.argmax(V)
        V = V / V[max_V_index]
        # print("l = ", l, "max_V_index = ", max_V_index)
        # print("h = ", h[l])
        # print("H = ", H[:,l])
        # print("V = ", V)
        for j in range(n):
            x = V[j] * qr
            y = V[max_V_index] * pr
            if x >= y:
                T[j, l] = j
                z = x
            else:
                T[j, l] = max_V_index
                z = y
            if H[j][l] == h[l]:
                V_next[j] = z * qm
            else:
                V_next[j] = z * pm
        V = V_next
    return T, np.argmax(V)


def trees_leaf_lists(ts):
    """
    Returns an iterator over the trees and leaf lists of this speficied tree
    sequence.
    """
    trees = ts.trees()
    for pi, xi, head, tail in leaf_sets(ts, list(ts.samples())):
        t = next(trees)
        pi_prime = [t.parent(u) for u in range(ts.num_nodes)]
        assert pi == pi_prime
        yield t, head, tail

def leaf_list_partition(tree, head, tail, u):
    """
    Returns the partition of (h, t) pairs defining the partition of
    leaf list to the left, under and right of the specified node.
    If left and right are empty they are returned as None.
    """
    l_h = None
    l_t = None
    m_h = head[u]
    m_t = tail[u]
    r_h = None
    r_t = None
    root_h = head[tree.root]
    root_t = tail[tree.root]
    if m_h != root_h:
        l_h = root_h
        l_t = m_h.prev
    if m_t != root_t:
        r_h = m_t.next
        r_t = root_t
    l = l_h, l_t
    r = r_h, r_t
    assert m_h is not None
    assert m_t is not None
    return l, (m_h, m_t), r

def match_haplotype_ts(ts, h, rho, theta):
    # Abandoning this because the problem of finding the intersection of
    # the leaf lists in V and in the mutation partition is difficult. We
    # need some way of telling where we are in the leaf list, which isn't
    # possible with the current linked list approach. We need some more
    # sophisticated data structure that maintains order, and allows us to
    # move chunks of leaves around within the tree. There is most likely
    # a very fast implementation of L&S here, but it doesn't get us any
    # closer to the immediate goal.
    n = ts.sample_size
    m = ts.num_sites
    r = 1 - np.exp(-rho / n)
    pr = r / n
    qr = 1 - r + r / n
    # pm = mutation; qm no mutation
    pm = 0.5 * theta / (n + theta)
    qm = n / (n + theta) + 0.5 * theta / (n + theta)

    V = []
    iterator = trees_leaf_lists(ts)
    tree, head, tail = next(iterator)
    for site in tree.sites():
        for mutation in site.mutations:
            print(site.position, mutation.node)
            (l_h, l_t), (m_h, m_t), (r_h, r_t) = leaf_list_partition(
                    tree, head, tail, mutation.node)
            print("\t", leaf_list(l_h, l_t), leaf_list(m_h, m_t), leaf_list(r_h, r_t))
            if l_h is not None:
                V.append([l_h, l_t, pm])
            V.append([m_h, m_t, qm])
            if r_h is not None:
                V.append([r_h, r_t, pm])
            # l, m, r = leaf_list_intersection(tree, head, tail, V, mutation.node)

            # print("V = ", V)
        break



def ts_ls(n):
    """
    Experimental code to run L&S on a tree sequence.
    """

    np.set_printoptions(linewidth=200)
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999
    pd.options.display.width = 999
    random.seed(2)

    ts = msprime.simulate(
        n, length=10, recombination_rate=0, mutation_rate=1, random_seed=1)
    # print(ts.num_trees, ts.num_sites)
    t = next(ts.trees())
    # t.draw("tree.svg")
    order = np.array(list(t.leaves(t.root)), dtype=int)
    R = []
    V = np.zeros((ts.num_sites, ts.sample_size), dtype=int)
    for variant in ts.variants():
        col = variant.genotypes[order]
        # col = variant.genotypes
        V[variant.index] = col
        # Run length encode col
        segs = [[0, 1, col[0]]]
        for j in range(1, col.shape[0]):
            if col[j] == segs[-1][-1]:
                segs[-1][1] += 1
            else:
                segs.append([j, j + 1, col[j]])
        # print(col, segs)
        R.append(segs)
    # Make sure our encoding is correct
    Vp = np.zeros((ts.num_sites, ts.sample_size), dtype=int)
    for j, segs in enumerate(R):
        for l, r, v in segs:
            Vp[j, l:r] = v
    assert np.all(V == Vp)
    H = V.T
    print(pd.DataFrame(H))
    n = ts.sample_size
    h = np.copy(H[0])
    j = 0
    P = np.zeros(ts.num_sites, dtype=int)
    # for k in [8, 20, 60, 100, ts.num_sites - 10, ts.num_sites]:
    for k in [10, ts.num_sites]:
        haplotype = random.randint(0, n)
        h[j: k] = H[haplotype][j:k]
        P[j: k] = haplotype
        j = k

    # print("\n", h)
    print()
    # print(pd.DataFrame(h).T)
    # T, start1 = match_haplotype_simple(H, h, 0.01, 0.01)
    # P1 = run_traceback(T, start1)
    # print(T, start)
    # print("P = ", P)
    # print("Q = ", Q)
    E, start2 = match_haplotype_encoded(R, n, h, 0.01, 0.01)
    P2 = run_traceback_encoded(E, n, start2)
    # for row in E:
    #     print(row)
    # print(P2)
    # print("start = ", start2)
    # print(E, start)
    # print("Starts = ", start1, start2)
    # Tp = decode_traceback(E, n)

    # print("T = ")
    # print(pd.DataFrame(T))
    # print("Tp =")
    # print(pd.DataFrame(Tp))

    # print("P1 = ", P1)
    print("P  = ", P)
    print("P2 = ", P2)


class LeafListNode(object):
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

    def __str__(self):
        next = -1 if self.next is None else self.next.value
        return "{}->{}".format(self.value, next)

    def __repr__(self):
        return str(self.value)

def update_leaf_lists(u, pi, xi, head, tail):
    while u != -1:
        head[u] = None
        tail[u] = None
        for v in xi[u]:
            if head[v] is not None:
                if head[u] is None:
                    head[u] = head[v]
                    tail[u] = tail[v]
                else:
                    tail[u].next = head[v]
                    head[v].prev = tail[u]
                    tail[u] = tail[v]
        u = pi[u]


def leaf_sets(ts, S):
    """
    Sequentially visits all trees in the specified
    tree sequence and maintain the leaf sets for all leaves in
    specified set for each node.
    """
    l = [e.left for e in ts.edgesets()]
    r = [e.right for e in ts.edgesets()]
    u = [e.parent for e in ts.edgesets()]
    c = [e.children for e in ts.edgesets()]
    t = [r.time for r in ts.records()]
    # Calculate the index vectors
    M = ts.num_edgesets
    I = sorted(range(M), key=lambda j: (l[j], t[j]))
    O = sorted(range(M), key=lambda j: (r[j], -t[j]))
    pi = [-1 for j in range(max(u) + 1)]
    xi = [[] for j in range(max(u) + 1)]
    head = [None for j in range(max(u) + 1)]
    tail = [None for j in range(max(u) + 1)]
    for j in S:
        node = LeafListNode(j)
        head[j] = node
        tail[j] = node
    j = 0
    k = 0
    while j < M:
        x = l[I[j]]
        while r[O[k]] == x:
            h = O[k]
            xi[u[h]] = []
            for q in c[h]:
                pi[q] = -1
            update_leaf_lists(u[h], pi, xi, head, tail)
            k += 1
        while j < M and l[I[j]] == x:
            h = I[j]
            for q in c[h]:
                pi[q] = u[h]
            xi[u[h]] = c[h]
            update_leaf_lists(u[h], pi, xi, head, tail)
            j += 1
        yield pi, xi, head, tail


def leaf_list(head, tail, forward=True):
    if head is None:
        return []
    ret = []
    if forward:
        u = head
        while True:
            ret.append(u.value)
            if u == tail:
                break
            u = u.next
    else:
        u = tail
        while True:
            ret.append(u.value)
            if u == head:
                break
            u = u.prev
    return ret

def leaf_lists_dev():

    random.seed(1)
    ts = msprime.simulate(
        10, length=1, recombination_rate=0.1, mutation_rate=1, random_seed=1)

    V = np.zeros((ts.num_sites, ts.sample_size), dtype=int)
    for variant in ts.variants():
        V[variant.index, :] = variant.genotypes
    H = V.T
    print(pd.DataFrame(H))
    n = ts.sample_size
    h = np.copy(H[0])
    j = 0
    # for k in [8, 20, 60, 100, ts.num_sites - 10, ts.num_sites]:
    P = np.zeros(ts.num_sites, dtype=int)
    for k in [10, ts.num_sites]:
        haplotype = random.randint(0, n)
        h[j: k] = H[haplotype][j:k]
        P[j: k] = haplotype
        j = k
    print(h)
    match_haplotype_ts(ts, h, 0.01, 0.01)



    # trees = ts.trees(leaf_lists=True)
    # for pi, xi, head, tail in leaf_sets(ts, list(ts.samples())):
    #     print("NEW TREE")
    #     t = next(trees)
    #     pi_prime = [t.parent(u) for u in range(ts.num_nodes)]
    #     assert pi == pi_prime
    #     # print(pi)
    #     # print(pi_prime)
    #     # print(leaf_list(head[t.root], tail[t.root]))
    #     leaves = list(t.leaves(t.root))
    #     # print(leaves)
    #     assert leaf_list(head[t.root], tail[t.root]) == leaves
    #     assert leaf_list(head[t.root], tail[t.root], forward=False) == list(reversed(leaves))
    #     for site in t.sites():
    #         for mutation in site.mutations:
    #             u = mutation.node
    #             l_h = None
    #             l_t = None
    #             m_h = head[u]
    #             m_t = tail[u]
    #             r_h = None
    #             r_t = None
    #             root_h = head[t.root]
    #             root_t = tail[t.root]
    #             if m_h != root_h:
    #                 l_h = root_h
    #                 l_t = m_h.prev
    #             if m_t != root_t:
    #                 r_h = m_t.next
    #                 r_t = root_t
    #             l = []
    #             if l_h is not None:
    #                 l = leaf_list(l_h, l_t)
    #             r = []
    #             if r_h is not None:
    #                 r = leaf_list(r_h, r_t)
    #             m = leaf_list(m_h, m_t)
    #             assert leaves == l + m + r
    #             assert leaves == leaf_list(root_h, root_t)
    #             print(l, m, r)


    # for t in ts.trees():

if __name__ == "__main__":
    # main()
    # example()
    # bug()
    # for n in [100, 1000, 10000, 20000, 10**5]:
    #     ts_ls(n)
    # ts_ls(20)
    leaf_lists_dev()


