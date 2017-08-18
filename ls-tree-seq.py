"""
Implemenation of the Li and Stephens algorithm on a tree sequence.
sequence.

"""
import random
import sys


import numpy as np

import msprime

if sys.version_info[0] < 3:
    raise Exception("Python 3 you idiot!")

def best_path(h, H, recombination_rate):
    n, m = H.shape
    r = 1 - np.exp(-recombination_rate / n)
    recomb_proba = r / n
    no_recomb_proba = 1 - r + r / n
    L = np.ones(n)
    T = [set() for _ in range(m)]
    T_dest = np.zeros(m, dtype=int)

    for l in range(m):
        L_next = np.zeros(n)
        for j in range(n):
            x = L[j] * no_recomb_proba
            y = recomb_proba
            if x > y:
                z = x
            else:
                z = y
                T[l].add(j)
            emission_p = int(H[j, l] == h[l])
            L_next[j] = z * emission_p
        # Find the max and renormalise
        L = L_next
        j = np.argmax(L)
        T_dest[l] = j
        L /= L[j]
    P = np.zeros(m, dtype=int)
    P[m - 1] = T_dest[m - 1]
    for l in range(m - 1, 0, -1):
        j = P[l]
        if j in T[l]:
            assert l != 0
            j = T_dest[l - 1]
        P[l - 1] = j
    return P

def random_mosaic(H):
    n, m = H.shape
    h = np.zeros(m, dtype=int)
    for l in range(m):
        h[l] = H[random.randint(0, n - 1), l]
    return h

def copy_process_dev(n, L, seed):
    random.seed(seed)
    ts = msprime.simulate(
        n, length=L, mutation_rate=1, recombination_rate=1, random_seed=seed)
    m = ts.num_sites
    H = np.zeros((n, m), dtype=int)
    for v in ts.variants():
        H[:, v.index] = v.genotypes

    # print(H)
    for j in range(100):
        h = random_mosaic(H)
        p = best_path(h, H, 1e-8)
        hp = H[p, np.arange(m)]
        # print()
        # print(h)
        # print(hp)
        # print(p)
        assert np.array_equal(h, hp)


def main():
    np.set_printoptions(linewidth=2000)
    np.set_printoptions(threshold=20000)
    copy_process_dev(20, 100, 1)

if __name__ == "__main__":
    main()
