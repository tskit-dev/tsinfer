
import tsinfer
import subprocess
import os
import numpy as np
import itertools
import multiprocessing
import pandas as pd

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
        # print("A = ", A)
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


def example():
    np.set_printoptions(linewidth=100)
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999
    pd.options.display.width = 999

    rho = 3
    for seed in range(1, 10000):
    # for seed in [2]:
        ts = msprime.simulate(
            sample_size=10, recombination_rate=rho, mutation_rate=1,
            length=6, random_seed=seed)
        print("seed = ", seed)
        sites = [site.position for site in ts.sites()]
        S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
        for variant in ts.variants():
            S[:, variant.index] = variant.genotypes
        H = make_ancestors(S)
        print(H.shape)
        # df = pd.DataFrame(H)
        # print(df[ts.sample_size:])

        panel = tsinfer.ReferencePanel(
            S, sites, ts.sequence_length, rho=rho, sample_error=0, ancestor_error=0,
            algorithm="python", haplotypes=H)
        # print(panel.haplotypes)
        # index = H != -1
        # assert np.all(H[index] == panel.haplotypes[index])

        P, mutations = panel.infer_paths(num_workers=1)
        P = P.astype(np.int32)

        new_ts = panel.convert_records(P, mutations)
        Sp = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
        for variant in new_ts.variants():
            Sp[:, variant.index] = variant.genotypes
        assert np.all(Sp == S)
        tss = new_ts.simplify()

        # verify_breaks(ts.sample_size, H, P)

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

if __name__ == "__main__":
    # main()
    example()
    # bug()

