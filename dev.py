
import tsinfer
import subprocess
import os
import numpy as np

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
    print("making ancestors for \n", S)
    frequency = np.sum(S, axis=0)
    num_ancestors = np.sum(frequency > 1)
    site_order = frequency.argsort()[::-1]
    N = n + num_ancestors + 1
    H = np.zeros((N, m), dtype=np.int8)

    for j in range(n):
        H[j] = S[j]

    for j in range(num_ancestors):
        site = site_order[j]
        row = N - j - 2
        H[row, site] = 1
        print(j, row, N - 2, "site = ", site, H[row])
        if j > 0:
            break_found = False
            for k in range(site - 1, -1, -1):
                older = H[row + 1: N - 1, k]
                print("\tback: examinging site", k, older)
                if not break_found:
                    values = np.unique(older)
                    print("vlalues = ", values)
                    if values[0] == -1:
                        values = values[1:]
                    if len(values) == 1:
                        H[row, k] = values[0]
                    else:
                        break_found = True
                if break_found:
                    H[row, k] = -1
            break_found = False
            for k in range(site + 1, m):
                older = H[row + 1: N - 1, k]
                print("\tfwd : examinging site", k, older)
                if not break_found:
                    values = np.unique(older)
                    print("values", values)
                    if values[0] == -1:
                        values = values[1:]
                    if len(values) == 1:
                        H[row, k] = values[0]
                    else:
                        break_found = True
                if break_found:
                    H[row, k] = -1
    print("H = ")
    print(H)


def main():
    # for seed in range(10000):
    # print("seed = ", seed)
    # np.random.seed(seed)
    # S = get_random_data_example(100, 1000)

    seed = 5
    rho = 10
    ts = msprime.simulate(
        sample_size=8, recombination_rate=rho, mutation_rate=1,
        length=1, random_seed=seed)
    # for tree in ts.trees():
    #     print("tree: ", tree.interval, tree.parent_dict)
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    H = make_ancestors(S)
    sites = [site.position for site in ts.sites()]
    panel = tsinfer.ReferencePanel(
        S, sites, ts.sequence_length, rho=rho, algorithm="python")
    # P, mutations = panel.infer_paths(num_workers=1)
    # ts_new = panel.convert_records(P, mutations)

    num_samples, num_sites = S.shape
    print("num_sites = ", num_sites)
    # sites = np.arange(num_sites)
    panel = tsinfer.ReferencePanel(
        S, sites, num_sites, rho=rho, sample_error=0, ancestor_error=0,
        algorithm="c")
    P, mutations = panel.infer_paths(num_workers=1)

#     illustrator = tsinfer.Illustrator(panel, P, mutations)
#     for j in range(panel.num_haplotypes):
#         pdf_file = "tmp__NOBACKUP__/temp_{}.pdf".format(j)
#         png_file = "tmp__NOBACKUP__/temp_{}.png".format(j)
#         illustrator.run(j, pdf_file)
#         subprocess.check_call("convert -geometry 3000 -density 600 {} {}".format(
#             pdf_file, png_file), shell=True)
#         print(png_file)
#         os.unlink(pdf_file)

    ts = panel.convert_records(P, mutations)
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

    ts_simplified = ts.simplify()

    S2 = np.empty(S.shape, np.uint8)
    for j, h in enumerate(ts_simplified.haplotypes()):
        S2[j,:] = np.fromstring(h, np.uint8) - ord('0')
    assert np.all(S == S2)

main()
