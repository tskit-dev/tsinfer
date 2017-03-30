
import tsinfer
import subprocess
import os
import numpy as np
import itertools

import msprime

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
    site_order = frequency.argsort()[::-1]
    N = n + num_ancestors + 1
    H = np.zeros((N, m), dtype=np.int8)

    for j in range(n):
        H[j] = S[j]

    for j in range(num_ancestors):
        site = site_order[j]
        # Find all samples that have a 1 at this site.
        R = S[S[:,site] == 1]
        # Get the consensus sequence among these ancestors.
        A = (np.sum(R, axis=0) >= R.shape[0] / 2).astype(int)
        left_bound = site - 1
        patterns = set([(1, 1)])
        while left_bound > 0:
            for k in range(R.shape[0]):
                patterns.add((A[left_bound], R[k, left_bound]))
            if len(patterns) == 4:
                break
            left_bound -= 1
        right_bound = site + 1
        patterns = set([(1, 1)])
        while right_bound < m:
            for k in range(R.shape[0]):
                patterns.add((A[right_bound], R[k, right_bound]))
            if len(patterns) == 4:
                break
            right_bound += 1
        A[0: left_bound] = -1
        A[right_bound:] = -1
        H[N - j - 2] = A
    return H


def get_gap_density(ts):
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    H = make_ancestors(S)
    A = H[ts.sample_size:]
    # print(H)
    # print()
    # print(A)
    gaps = np.sum(A == -1, axis=0)
    return gaps / A.shape[0]

def main():
    # for seed in range(10000):
    # print("seed = ", seed)
    # np.random.seed(seed)
    # S = get_random_data_example(100, 1000)
    rho = 1
    num_replicates = 100
    for n in [10, 50, 100]:
        for length in [10, 100, 1000]:
            num_sites = np.zeros(num_replicates)
            gap_density = np.zeros(num_replicates)
            replicates = msprime.simulate(
                sample_size=n, recombination_rate=rho, mutation_rate=1,
                length=length, num_replicates=num_replicates)
            for j, ts in enumerate(replicates):
                gap_density[j] = np.mean(get_gap_density(ts))
                num_sites[j] = ts.num_sites
            print(n, length, np.mean(num_sites), np.mean(gap_density))
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

main()
