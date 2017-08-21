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
    # print("IS_DESCENDENT(", u, v, ")")
    while u != v and u != msprime.NULL_NODE:
        # print("\t", u)
        u = tree.parent(u)
    # print("END, ", u, v)
    return u == v

def check_sample_coverage(tree, nodes):
    """
    Ensures that all the samples from the specified tree are covered by the
    set of nodes with no overlap.
    """
    samples = set()
    for u in nodes:
        leaves = set(tree.leaves(u))
        assert len(leaves & samples) == 0
        samples |= leaves
    # NOTE: will not work for more general samples.
    assert samples == set(range(tree.sample_size))

def get_tree_likelihood(tree, state, site, L, recombination_rate, T, T_dest):
    # print("get tree likelihood", state, mutation_node, L)
    mutation_node = site.mutations[0].node
    n = tree.sample_size
    r = 1 - np.exp(-recombination_rate / n)
    recomb_proba = r / n
    no_recomb_proba = 1 - r + r / n

    L_next = {}
    for L_node, L_value in L.items():
        if is_descendent(tree, mutation_node, L_node):
            # print("Splitting for mutation_node = ", mutation_node, "L_ndoe = ", L_node)
            L_next[mutation_node] = L_value
            # Traverse upwards until we reach old L node, adding values
            # for the siblings off the path.
            u = mutation_node
            while u != L_node:
                v = tree.parent(u)
                # print("\t u = ", u, "v = ", v)
                for w in tree.children(v):
                    # print("\t\tw = ", w)
                    if w != u:
                        # print("\t\tset ", w, "->", L_value)
                        L_next[w] = L_value
                u = v
        else:
            L_next[L_node] = L_value
    # print("Updated L", L_next)
    # Update the likelihoods.
    # print("mutation node = ", mutation_node)
    max_L = -1
    for v in L_next.keys():
        x = L_next[v] * no_recomb_proba
        y = recomb_proba
        if x > y:
            z = x
        else:
            z = y
            T[site.index].add(v)
        # print("\tstate = ", state, "v = ", v, "is_descendent = ",
        #         is_descendent(tree, mutation_node, v))
        if state == 1:
            emission_p = int(is_descendent(tree, v, mutation_node))
        else:
            emission_p = int(not is_descendent(tree, v, mutation_node))
        # print("\tv = ", v, " z = ", z, "emission = ", emission_p)
        L_next[v] = z * emission_p
        if L_next[v] > max_L:
            max_L = L_next[v]
    # print(L_next)
    check_sample_coverage(tree, L_next.keys())
    assert max_L > 0
    # Normalise
    for v in L_next.keys():
        L_next[v] /= max_L

    # Coalesce equal values
    V = {}
    # Take all the L values an propagate them up the tree.
    for u in L_next.keys():
        x = L_next[u]
        while u != msprime.NULL_NODE and u not in V:
            V[u] = x
            u = tree.parent(u)
        if u != msprime.NULL_NODE and V[u] != x:
            # Mark the path up to root as invalid
            while u!= msprime.NULL_NODE:
                V[u] = -1
                u = tree.parent(u)
    W = {}
    # Get the distinct roots from L in V
    for u in L_next.keys():
        x = V[u]
        last_u = u
        while u != msprime.NULL_NODE and V[u] != -1:
            last_u = u
            u = tree.parent(u)
        if x != -1:
            W[last_u] = x

    # Find a node with W == 1 and register as the recombinant haplotype root.
    found = False
    for u, value in W.items():
        if value == 1.0:
            T_dest[site.index] = u
            found = True
            break
    assert found
    return W

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

    P = [-1 for _ in range(ts.num_nodes)]
    C = [None for _ in range(ts.num_nodes)]

    L_size = []

    T_tree = [set() for _ in range(m)]
    T_dest_tree = np.zeros(m, dtype=int)
    L_tree = {u: 1 for u in ts.samples()}
    for t, diff in zip(ts.trees(), ts.diffs()):
        # print("At tree", t.index, t.parent_dict)
        # print("L before = ", L_tree)
        t.draw("t{}.svg".format(t.index), width=800, height=800, mutation_locations=False)
        _, records_out, records_in = diff
        for parent, children, _ in records_out:
            for c in children:
                P[c] = -1
            C[parent] = None
            # print("\tout = ", parent, children)
            if parent in L_tree:
                x = L_tree.pop(parent)
                for c in children:
                    L_tree[c] = x
            else:
                # Traverse upwards until we find an L value
                u = parent
                while u != -1 and u not in L_tree:
                    u = P[u]
                if u != -1:
                    x = L_tree[u]
                    for c in children:
                        L_tree[c] = x
        # print("AFTER OUT:", L_tree)
        for parent, children, _ in records_in:
            # print("\tin = ", parent, children)
            C[parent] = children
            for c in children:
                P[c] = parent
            # Coalesce the L values for children if possible.
            L_children = []
            for c in children:
                if c in L_tree:
                    L_children.append(L_tree[c])
            if len(L_children) == len(children) and len(set(L_children)) == 1:
                L_tree[parent] = L_tree[children[0]]
                for c in children:
                    del L_tree[c]
            if len(L_children) > 0:
                # Need to check for conflicts with L values higher in the tree.
                u = P[parent]
                while u != msprime.NULL_NODE and u not in L_tree:
                    u = P[u]
                # print("Traversed upwards from ", parent, "to", u)
                if u != msprime.NULL_NODE:
                    # print("CONFLICT:", u, L_tree[u])
                    top = u
                    x = L_tree.pop(top)
                    u = parent
                    while u != top:
                        v = P[u]
                        for w in C[v]:
                            if w != u:
                                L_tree[w] = x
                        u = v
        # print("AFTER IN:", L_tree)
        P_dict = {u: P[u] for u in range(ts.num_nodes) if P[u] != -1}
        assert t.parent_dict == P_dict
        check_sample_coverage(t, L_tree.keys())
        # print("DONE")

        for site in t.sites():
            L_next = {}
            l = site.index
            # print()
            # print("updating for site", l, "h = ", h[l])
            # print("L_start = ", L)
            u = site.mutations[0].node
            mutation_node = u
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
            # print("L_next = ", L_next)
            for u in L_next.keys():
                L[u] = L_next[u] / max_x
            # print("L = ", L)

            # Compute the tree node liklihoods the long way around.
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

            L_tree = get_tree_likelihood(t, h[l], site, L_tree, recombination_rate,
                    T_tree, T_dest_tree)
            # print("W", W)
            # print("L", L_tree)
            check_sample_coverage(t, W.keys())
            check_sample_coverage(t, L_tree.keys())
            assert W == L_tree
            L_size.append(len(L_tree))

#             print(l,":", W)
#             # print("L = ", L)
#             print("\tmutation node = ", mutation_node)
#             print("\tT = ", T[l])
#             print("\tT_tree = ", T_tree[l])
#             print("\tT_dest", T_dest[l])
#             print("\tT_dest_tree", T_dest_tree[l])
    # print("mean L_size = ", np.mean(L_size))

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

    P_tree = np.zeros(m, dtype=int)
    trees = ts.trees()
    l = m - 1
    tree = next(trees)
    sites = list(ts.sites())
    while tree.interval[1] < sites[l].position:
        tree = next(trees)
    u = T_dest_tree[m - 1]
    while not tree.is_leaf(u):
        u = tree.children(u)[0]
    P_tree[m - 1] = u
    # for l in range(m):
    #     print(l, T_dest_tree[l], T_tree[l])
    for l in range(m - 1, 0, -1):
        trees = ts.trees()
        tree = next(trees)
        while tree.interval[1] < sites[l].position:
            tree = next(trees)
        left, right = tree.interval
        assert left <= sites[l].position < right
        j = P_tree[l]
        for u in T_tree[l]:
            if is_descendent(tree, j, u):
                assert l != 0
                # print("RECOMBINING at ", l, ":", j, u, T_dest_tree[l - 1])
                j = T_dest_tree[l - 1]
                break
        # Change j into a sample
        # P_tree[l - 1] = j
        u = j
        while not tree.is_leaf(u):
            u = tree.children(u)[0]
        P_tree[l - 1] = u
    # print("P_tree = ", P_tree)
    # Change these into samples
    # for tree in ts.trees():
    #     for site in tree.sites():
    #         u = P_tree[site.index]
    #         while not tree.is_leaf(u):
    #             u = tree.children(u)[0]
    #         # print("Mapping", P_tree[site.index], "->", u)
    #         P_tree[site.index] = u
    # print(P)
    return P_tree
    # return P



def random_mosaic(H):
    n, m = H.shape
    h = np.zeros(m, dtype=int)
    for l in range(m):
        h[l] = H[random.randint(0, n - 1), l]
    return h

def copy_process_dev(n, L, seed):
    random.seed(seed)
    ts = msprime.simulate(
        n, length=L, mutation_rate=1, recombination_rate=0, random_seed=seed)
    m = ts.num_sites
    H = np.zeros((n, m), dtype=int)
    for v in ts.variants():
        H[:, v.index] = v.genotypes

    # print(H)
    for j in range(10):
        h = random_mosaic(H)
        # h = np.hstack([H[0,:10], H[1,10:]])
        # print()
        # print(h)
        # p = best_path(h, H, 1e-8)
        p = best_path_ts(h, ts, 1e-8)

        # print("p = ", p)
        hp = H[p, np.arange(m)]
        # print()
        # print(h)
        # print(hp)
        assert np.array_equal(h, hp)


def main():
    np.set_printoptions(linewidth=2000)
    np.set_printoptions(threshold=20000)
    for j in range(1, 10000):
        print(j)
        copy_process_dev(30, 50, j)
    # copy_process_dev(20, 30, 4)


if __name__ == "__main__":
    main()
