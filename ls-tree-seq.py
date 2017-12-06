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

def map_sample(ts, site_id, node):
    """
    Maps the specified node for the specified site to a sample node.
    """
    trees = ts.trees()
    tree = next(trees)
    position = list(ts.sites())[site_id].position
    while tree.interval[1] < position:
        tree = next(trees)
    left, right = tree.interval
    assert left <= position < right
    u = node
    while not tree.is_leaf(u):
        u = tree.children(u)[0]
    node = ts.node(u)
    assert node.is_sample
    return u

def is_descendent_at_site(ts, site_id, u, v):
    trees = ts.trees()
    tree = next(trees)
    position = list(ts.sites())[site_id].position
    while tree.interval[1] < position:
        tree = next(trees)
    left, right = tree.interval
    assert left <= position < right
    return is_descendent(tree, u, v)


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
        # t.draw("t{}.svg".format(t.index), width=800, height=800, mutation_locations=False)
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
                # The children are now the roots of disconnected subtrees, and
                # need to be assigned L values. We set these by traversing up
                # the tree until we find the L value and then set this to the
                # children.
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
    P_tree[m - 1] = map_sample(ts, m - 1, T_dest[m - 1])

    # for l in range(m):
    #     print(l, T_dest_tree[l], T_tree[l])
    for l in range(m - 1, 0, -1):
        u = P_tree[l]
        for v in T_tree[l]:
            if is_descendent_at_site(ts, l, u, v):
                assert l != 0
                # print("RECOMBINING at ", l, ":", j, u, T_dest_tree[l - 1])
                u = map_sample(ts, l - 1, T_dest_tree[l - 1])
                break
        P_tree[l - 1] = u

    return P_tree
    # return P


class HaplotypeMatcher(object):

    def __init__(self, tree_sequence, recombination_rate, samples=None):
        self.tree_sequence = tree_sequence
        self.tree = None
        if samples is None:
            samples = list(self.tree_sequence.samples())
        self.samples = samples
        self.num_sites = tree_sequence.num_sites
        self.recombination_rate = recombination_rate
        # Map of tree nodes to likelihoods. We maintain the property that the
        # nodes in this map are non-overlapping; that is, for any u in the map,
        # there is no v that is an ancestor of u.
        self.likelihood = {}
        # We keep a local copy of the parent array to allow us maintain the
        # likelihood map between tree transitions.
        self.parent = np.zeros(self.tree_sequence.num_nodes, dtype=int) - 1
        # For each locus, store a set of nodes at which we must recombine during
        # traceback.
        self.traceback = [[] for _ in range(self.num_sites)]
        # If we recombine during traceback, this is the node we recombine to.
        self.recombination_dest = np.zeros(self.num_sites, dtype=int) - 1

    def reset(self):
        self.likelihood = {}
        for u in self.samples:
            self.likelihood[u] = 1.0
        self.parent[:] = -1
        self.traceback = [[] for _ in range(self.num_sites)]
        self.recombination_dest[:] = -1

    def print_state(self):
        print("HaplotypeMatcher state")
        print("likelihood:")
        for k, v in self.likelihood.items():
            print("\t", k, "->", v)
        print("tree = ", repr(self.tree))
        if self.tree is not None:
            print("\tindex = ", self.tree.index)
            print("\tnum_sites = ", len(list(self.tree.sites())))
            print("\tp = ", self.tree.parent_dict)
        print("Traceback:")
        for l in range(self.num_sites):
            print("\t", l, "\t", self.recombination_dest[l], "\t", self.traceback[l])

    def check_sample_coverage(self, nodes):
        """
        Ensures that all the samples from the specified tree are covered by the
        set of nodes with no overlap.
        """
        samples = set()
        for u in nodes:
            # TODO should be samples not leaves.
            leaves = set(self.tree.leaves(u))
            assert len(leaves & samples) == 0
            samples |= leaves
        assert samples == set(self.tree_sequence.samples())

    def check_state(self):
        # print("AFTER IN:", L_tree)
        ts = self.tree_sequence
        P_dict = {
            u: self.parent[u] for u in range(ts.num_nodes) if self.parent[u] != -1}
        assert self.tree.parent_dict == P_dict
        self.check_sample_coverage(self.likelihood.keys())
        # print("DONE")

    def update_tree_state(self, diff):
        """
        Update the state of the likelihood map to reflect the new tree. We use
        the diffs to efficiently migrate the likelihoods from nodes in the previous
        tree to the new tree.
        """
        _, records_out, records_in = diff
        for parent, children, _ in records_out:
            for c in children:
                self.parent[c] = msprime.NULL_NODE
            if parent in self.likelihood:
                # If we remove a node and it has an L value, then this L value is
                # mapped to its children.
                x = self.likelihood.pop(parent)
                for c in children:
                    self.likelihood[c] = x
            else:
                # The children are now the roots of disconnected subtrees, and
                # need to be assigned L values. We set these by traversing up
                # the tree until we find the L value and then set this to the
                # children.
                u = parent
                while u != -1 and u not in self.likelihood:
                    u = self.parent[u]
                # TODO It's not completely clear to me what's happening in the
                # case where u is -1. The logic of this section can be clarified
                # here I think as we should be setting values for the children
                # in all cases where they do not have L values already.
                if u != -1:
                    x = self.likelihood[u]
                    for c in children:
                        self.likelihood[c] = x
        # TODO we are not correctly coalescing all equal valued L values among
        # children here. Definitely need another pass at this algorithm to
        # make it more elegant and catch all the corner cases.
        for parent, children, _ in records_in:
            for c in children:
                self.parent[c] = parent
            # Coalesce the L values for children if possible.
            # TODO this is ugly and inefficient. Need a simpler approach.
            L_children = []
            for c in children:
                if c in self.likelihood:
                    L_children.append(self.likelihood[c])
            if len(L_children) == len(children) and len(set(L_children)) == 1:
                self.likelihood[parent] = self.likelihood[children[0]]
                for c in children:
                    del self.likelihood[c]
            if len(L_children) > 0:
                # Need to check for conflicts with L values higher in the tree.
                u = self.parent[parent]
                while u != msprime.NULL_NODE and u not in self.likelihood:
                    u = self.parent[u]
                if u != msprime.NULL_NODE:
                    top = u
                    x = self.likelihood.pop(top)
                    u = parent
                    while u != top:
                        v = self.parent[u]
                        for w in self.tree.children(v):
                            if w != u:
                                self.likelihood[w] = x
                        u = v
    def add_recombination_node(self, site_id, u):
        """
        Adds a recombination node for the specified site.
        """
        self.traceback[site_id].append(u)

    def choose_recombination_destination(self, site_id):
        """
        Given the state of the likelihoods, choose the destination for
        haplotypes recombining onto this site.
        """
        # Find a node with L == 1 and register as the recombinant haplotype root.
        found = False
        for u, value in self.likelihood.items():
            if value == 1.0:
                self.recombination_dest[site_id] = u
                found = True
                break
        assert found

    def coalesce_equal(self, L):
        """
        Coalesce L values into the minimal representation by propagating
        values up the tree and finding the roots of the subtrees sharing
        the same L value.
        """
        tree = self.tree
        # Coalesce equal values
        V = {}
        # Take all the L values an propagate them up the tree.
        for u in L.keys():
            x = L[u]
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
        for u in L.keys():
            x = V[u]
            last_u = u
            while u != msprime.NULL_NODE and V[u] != -1:
                last_u = u
                u = tree.parent(u)
            if x != -1:
                W[last_u] = x
        return W

    def update_site(self, site, state):
        """
        Updates the algorithm state for the specified site given the specified
        input state.
        """
        assert len(site.mutations) == 1
        assert site.ancestral_state == '0'
        mutation_node = site.mutations[0].node
        n = len(self.samples)
        r = 1 - np.exp(-self.recombination_rate / n)
        recomb_proba = r / n
        no_recomb_proba = 1 - r + r / n

        L = self.likelihood
        tree = self.tree
        # Update L to add nodes for the mutation node, splitting and removing
        # existing L nodes as necessary.
        L_next = {}
        for L_node, L_value in L.items():
            if is_descendent(tree, mutation_node, L_node):
                L_next[mutation_node] = L_value
                # Traverse upwards until we reach old L node, adding values
                # for the siblings off the path.
                u = mutation_node
                while u != L_node:
                    v = tree.parent(u)
                    for w in tree.children(v):
                        if w != u:
                            L_next[w] = L_value
                    u = v
            else:
                L_next[L_node] = L_value
        # Update the likelihoods for this site.
        max_L = -1
        for v in L_next.keys():
            x = L_next[v] * no_recomb_proba
            y = recomb_proba
            if x > y:
                z = x
            else:
                z = y
                self.add_recombination_node(site.index, v)
            if state == 1:
                emission_p = int(is_descendent(tree, v, mutation_node))
            else:
                emission_p = int(not is_descendent(tree, v, mutation_node))
            L_next[v] = z * emission_p
            if L_next[v] > max_L:
                max_L = L_next[v]
        assert max_L > 0

        # Normalise
        for v in L_next.keys():
            L_next[v] /= max_L
        self.likelihood = self.coalesce_equal(L_next)
        self.choose_recombination_destination(site.index)


    def map_sample(self, site_id, node):
        """
        Maps the specified node for the specified site to a sample node.
        """
        trees = self.tree_sequence.trees()
        tree = next(trees)
        position = list(self.tree_sequence.sites())[site_id].position
        while tree.interval[1] < position:
            tree = next(trees)
        left, right = tree.interval
        assert left <= position < right
        u = node
        while not tree.is_leaf(u):
            u = tree.children(u)[0]
        node = self.tree_sequence.node(u)
        assert node.is_sample
        return u

    def run_traceback(self):
        m = self.num_sites
        p = np.zeros(m, dtype=int)
        p[m - 1] = self.map_sample(m - 1, self.recombination_dest[m - 1])

        for l in range(m - 1, 0, -1):
            u = p[l]
            for v in self.traceback[l]:
                if is_descendent_at_site(self.tree_sequence, l, u, v):
                    assert l != 0
                    # print("RECOMBINING at ", l, ":", j, u, T_dest_tree[l - 1])
                    u = self.map_sample(l - 1, self.recombination_dest[l - 1])
                    break
            p[l - 1] = u
        return p

    def run(self, haplotype):
        self.reset()
        ts = self.tree_sequence
        # self.print_state()
        for tree, diff in zip(ts.trees(), ts.diffs()):
            self.tree = tree
            self.update_tree_state(diff)
            # self.tree.draw("t{}.svg".format(self.tree.index),
            #         width=800, height=800, mutation_locations=False)
            self.check_state()
            # self.print_state()
            for site in tree.sites():
                self.update_site(site, haplotype[site.index])
                self.check_state()
                # self.print_state()
        # self.print_state()
        return self.run_traceback()


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

    # matcher = HaplotypeMatcher(ts, recombination_rate=1e-8)
    # print(H)
    for j in range(1):
        h = random_mosaic(H)
        # h = np.hstack([H[0,:10], H[1,10:]])
        # print()
        # print(h)
        p = best_path(h, H, 1e-2)
        # p = best_path_ts(h, ts, 1e-8)

        # p = matcher.run(h)

        # print("p = ", p)
        hp = H[p, np.arange(m)]
        # print()
        # print(h)
        # print(hp)
        assert np.array_equal(h, hp)


def main():
    np.set_printoptions(linewidth=2000)
    np.set_printoptions(threshold=20000)
    # for j in range(1, 10000):
    #     print(j)
    #     copy_process_dev(200, 20, j)
    copy_process_dev(10, 10, 4)


if __name__ == "__main__":
    main()
