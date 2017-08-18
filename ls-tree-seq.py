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
        print(l, ":", L)
    P = np.zeros(m, dtype=int)
    P[m - 1] = T_dest[m - 1]
    for l in range(m - 1, 0, -1):
        j = P[l]
        if j in T[l]:
            assert l != 0
            j = T_dest[l - 1]
        P[l - 1] = j
    return P

def is_descendent(tree, u, v):
    """
    Returns True if the specified node u is a descendent of node v. That is,
    v is on the path to root from u.
    """
    while u != v and u != msprime.NULL_NODE:
        u = tree.parent(u)
    return u == v

def best_path_ts(h, ts, recombination_rate):
    n = ts.sample_size
    m = ts.num_sites
    r = 1 - np.exp(-recombination_rate / n)
    recomb_proba = r / n
    no_recomb_proba = 1 - r + r / n
    t1 = next(ts.trees())
    L = {u:1 for u in ts.samples()}
    T = [set() for _ in range(m)]
    T_dest = np.zeros(m, dtype=int)
    for t, diff in zip(ts.trees(), ts.diffs()):
        # print("At tree", t.parent_dict)
        # t.draw("t0.svg", width=800, height=800, mutation_locations=False)
        # print("diff = ", diff)
        for site in t.sites():
            L_next = {}
            l = site.index
            # print()
            # print("updating for site", l, "h = ", h[l])
            # print("L_start = ", L)
            u = site.mutations[0].node
            u_leaves = set(t.leaves(u))
            # print("\temission p for leaves below ", u, " = 1")
            # print("\tleaves = ", list(t.leaves(u)))
            for v in ts.samples():
                x = L[v] * no_recomb_proba
                y = recomb_proba
                if x > y:
                    z = x
                else:
                    z = y
                    T[l].add(v)
                if h[l] == 1:
                    emission_p = int(v in u_leaves)
                else:
                    emission_p = int(v not in u_leaves)
                L_next[v] = z * emission_p
            # Find max and normalise
            L = {}
            max_u = -1
            max_x = -1
            for u, x in L_next.items():
                if x > max_x:
                    max_x = x
                    max_u = u
            T_dest[l] = max_u
            for u in L_next.keys():
                L[u] = L_next[u] / max_x

            V = {}
            # Take all the U values an propagate them up the tree.
            for u in ts.samples():
                x = L[u]
                while u != msprime.NULL_NODE and u not in V:
                    V[u] = x
                    u = t.parent(u)
                if u != msprime.NULL_NODE and V[u] != x:
                    # Mark the path up to root as invalid
                    while u!= msprime.NULL_NODE:
                        V[u] = -1
                        u = t.parent(u)
            W = {}
            # Get the distinct roots from the sample in V
            for u in ts.samples():
                x = V[u]
                last_u = u
                while u != msprime.NULL_NODE and V[u] != -1:
                    last_u = u
                    u = t.parent(u)
                W[last_u] = x
            # Make sure that we get all the samples from the nodes in W
            samples = set()
            for u in W.keys():
                leaves = set(t.leaves(u))
                assert len(leaves & samples) == 0
                samples |= leaves
            assert samples == set(ts.samples())

            print("W", W)

            # print("\t", l,":", L)
            # print("L = ", L)
            # print("T = ", T[l])
            # print("T_dest", T_dest[l])

    # print(T)
    # print(T_dest)
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

    print(H)
    for j in range(1):
        h = random_mosaic(H)
        # print()
        # print(h)
        # p = best_path(h, H, 1e-8)
        p = best_path_ts(h, ts, 1e-8)

        hp = H[p, np.arange(m)]
        print()
        print(h)
        print(hp)
        print(p)
        assert np.array_equal(h, hp)


def main():
    np.set_printoptions(linewidth=2000)
    np.set_printoptions(threshold=20000)
    copy_process_dev(200, 100, 1)

if __name__ == "__main__":
    main()
