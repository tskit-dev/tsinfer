"""
Tests for the haplotype matching algorithm.
"""
import collections
import dataclasses
import io

import numpy as np
import pytest
import sortedcontainers
import tskit

import _tsinfer


@dataclasses.dataclass
class Edge:
    left: float = dataclasses.field(default=None)
    right: float = dataclasses.field(default=None)
    parent: int = dataclasses.field(default=None)
    child: int = dataclasses.field(default=None)


def add_vestigial_root(ts):
    """
    Adds the nodes and edges required by tsinfer to the specified tree sequence
    and returns it.
    """
    if not ts.discrete_genome:
        raise ValueError("Only discrete genome coords supported")

    base_tables = ts.dump_tables()
    tables = base_tables.copy()
    tables.nodes.clear()
    t = ts.max_root_time
    tables.nodes.add_row(time=t + 1)
    num_additonal_nodes = len(tables.nodes)
    tables.mutations.node += num_additonal_nodes
    tables.edges.child += num_additonal_nodes
    tables.edges.parent += num_additonal_nodes
    for node in base_tables.nodes:
        tables.nodes.append(node)
    for tree in ts.trees():
        root = tree.root + num_additonal_nodes
        tables.edges.add_row(
            tree.interval.left, tree.interval.right, parent=0, child=root
        )
    tables.edges.squash()
    tables.sort()
    return tables.tree_sequence()


def example_binary(n, L):
    tree = tskit.Tree.generate_balanced(n, span=L)
    return add_vestigial_root(tree.tree_sequence)


# Special values used to indicate compressed paths and nodes that are
# not present in the current tree.
COMPRESSED = -1
NONZERO_ROOT = -2


class TreeSequenceBuilder:
    # Temporary dummy implementation to get things working.
    def __init__(self, ts):
        self.time = ts.nodes_time
        self.num_nodes = ts.num_nodes
        self.num_match_nodes = ts.num_nodes
        self.num_sites = ts.num_sites
        self.left_index = sortedcontainers.SortedDict()
        self.right_index = sortedcontainers.SortedDict()

        for tsk_edge in ts.edges():
            edge = Edge(
                int(tsk_edge.left), int(tsk_edge.right), tsk_edge.parent, tsk_edge.child
            )
            self.left_index[(edge.left, self.time[edge.child], edge.child)] = edge
            self.right_index[(edge.right, -self.time[edge.child], edge.child)] = edge
        self.num_alleles = [var.num_alleles for var in ts.variants()]

        self.mutations = collections.defaultdict(list)
        for site in ts.sites():
            for mutation in site.mutations:
                # FIXME - should be allele index
                self.mutations[site.id].append((mutation.node, 1))


class AncestorMatcher:
    def __init__(
        self,
        ts,
        # recombination=None,
        # mismatch=None,
        # precision=None,
        extended_checks=False,
    ):
        self.tree_sequence_builder = TreeSequenceBuilder(ts)
        self.recombination = np.zeros(ts.num_sites) + 1e-9
        self.mismatch = np.zeros(ts.num_sites)
        # self.mismatch = mismatch
        # self.recombination = recombination
        # self.precision = precision
        self.precision = 14
        self.extended_checks = extended_checks
        self.num_sites = ts.num_sites
        self.parent = None
        self.left_child = None
        self.right_sib = None
        self.traceback = None
        self.max_likelihood_node = None
        self.likelihood = None
        self.likelihood_nodes = None
        self.allelic_state = None
        self.total_memory = 0

    def print_state(self):
        # TODO - don't crash when self.max_likelihood_node or self.traceback == None
        print("Ancestor matcher state")
        print("max_L_node\ttraceback")
        for site_index in range(self.num_sites):
            print(
                site_index,
                self.max_likelihood_node[site_index],
                self.traceback[site_index],
                sep="\t",
            )

    def is_root(self, u):
        return self.parent[u] == tskit.NULL

    def check_likelihoods(self):
        assert len(set(self.likelihood_nodes)) == len(self.likelihood_nodes)
        # Every value in L_nodes must be positive.
        for u in self.likelihood_nodes:
            assert self.likelihood[u] >= 0
        for u, v in enumerate(self.likelihood):
            # Every non-negative value in L should be in L_nodes
            if v >= 0:
                assert u in self.likelihood_nodes
            # Roots other than 0 should have v == -2
            if u != 0 and self.is_root(u) and self.left_child[u] == -1:
                # print("root: u = ", u, self.parent[u], self.left_child[u])
                assert v == -2

    def set_allelic_state(self, site):
        """
        Sets the allelic state array to reflect the mutations at this site.
        """
        # We know that 0 is always a root.
        # FIXME assuming for now that the ancestral state is always zero.
        self.allelic_state[0] = 0
        for node, state in self.tree_sequence_builder.mutations[site]:
            self.allelic_state[node] = state

    def unset_allelic_state(self, site):
        """
        Sets the allelic state values for this site back to null.
        """
        # We know that 0 is always a root.
        self.allelic_state[0] = -1
        for node, _ in self.tree_sequence_builder.mutations[site]:
            self.allelic_state[node] = -1
        assert np.all(self.allelic_state == -1)

    def update_site(self, site, haplotype_state):
        n = self.tree_sequence_builder.num_match_nodes
        rho = self.recombination[site]
        mu = self.mismatch[site]
        num_alleles = self.tree_sequence_builder.num_alleles[site]
        assert haplotype_state < num_alleles

        self.set_allelic_state(site)

        for node, _ in self.tree_sequence_builder.mutations[site]:
            # Insert an new L-value for the mutation node if needed.
            if self.likelihood[node] == COMPRESSED:
                u = node
                while self.likelihood[u] == COMPRESSED:
                    u = self.parent[u]
                self.likelihood[node] = self.likelihood[u]
                self.likelihood_nodes.append(node)

        max_L = -1
        max_L_node = -1
        for u in self.likelihood_nodes:
            # Get the allelic_state at u. TODO we can cache these states to
            # avoid some upward traversals.
            v = u
            while self.allelic_state[v] == -1:
                v = self.parent[v]
                assert v != -1

            p_last = self.likelihood[u]
            p_no_recomb = p_last * (1 - rho + rho / n)
            p_recomb = rho / n
            recombination_required = False
            if p_no_recomb > p_recomb:
                p_t = p_no_recomb
            else:
                p_t = p_recomb
                recombination_required = True
            self.traceback[site][u] = recombination_required
            p_e = mu
            if haplotype_state in (tskit.MISSING_DATA, self.allelic_state[v]):
                p_e = 1 - (num_alleles - 1) * mu
            self.likelihood[u] = p_t * p_e

            if self.likelihood[u] > max_L:
                max_L = self.likelihood[u]
                max_L_node = u

        if max_L == 0:
            if mu == 0:
                raise _tsinfer.MatchImpossible(
                    "Trying to match non-existent allele with zero mismatch rate"
                )
            elif mu == 1:
                raise _tsinfer.MatchImpossible(
                    "Match impossible: mismatch prob=1 & no haplotype with other allele"
                )
            elif rho == 0:
                raise _tsinfer.MatchImpossible(
                    "Matching failed with recombination=0, potentially due to "
                    "rounding issues. Try increasing the precision value"
                )
            raise AssertionError("Unexpected matching failure")

        for u in self.likelihood_nodes:
            x = self.likelihood[u] / max_L
            self.likelihood[u] = round(x, self.precision)

        self.max_likelihood_node[site] = max_L_node
        self.unset_allelic_state(site)
        self.compress_likelihoods()

    def compress_likelihoods(self):
        L_cache = np.zeros_like(self.likelihood) - 1
        cached_paths = []
        old_likelihood_nodes = list(self.likelihood_nodes)
        self.likelihood_nodes.clear()
        for u in old_likelihood_nodes:
            # We need to find the likelihood of the parent of u. If this is
            # the same as u, we can delete it.
            if not self.is_root(u):
                p = self.parent[u]
                cached_paths.append(p)
                v = p
                while self.likelihood[v] == -1 and L_cache[v] == -1:
                    v = self.parent[v]
                L_p = L_cache[v]
                if L_p == -1:
                    L_p = self.likelihood[v]
                # Fill in the L cache
                v = p
                while self.likelihood[v] == -1 and L_cache[v] == -1:
                    L_cache[v] = L_p
                    v = self.parent[v]

                if self.likelihood[u] == L_p:
                    # Delete u from the map
                    self.likelihood[u] = -1
            if self.likelihood[u] >= 0:
                self.likelihood_nodes.append(u)
        # Reset the L cache
        for u in cached_paths:
            v = u
            while v != -1 and L_cache[v] != -1:
                L_cache[v] = -1
                v = self.parent[v]
        assert np.all(L_cache == -1)

    def remove_edge(self, edge):
        p = edge.parent
        c = edge.child
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == tskit.NULL:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == tskit.NULL:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = tskit.NULL
        self.left_sib[c] = tskit.NULL
        self.right_sib[c] = tskit.NULL

    def insert_edge(self, edge):
        p = edge.parent
        c = edge.child
        self.parent[c] = p
        u = self.right_child[p]
        if u == tskit.NULL:
            self.left_child[p] = c
            self.left_sib[c] = tskit.NULL
            self.right_sib[c] = tskit.NULL
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = tskit.NULL
        self.right_child[p] = c

    def is_nonzero_root(self, u):
        return u != 0 and self.is_root(u) and self.left_child[u] == -1

    def find_path(self, h, start, end, match):
        Il = self.tree_sequence_builder.left_index
        Ir = self.tree_sequence_builder.right_index
        M = len(Il)
        n = self.tree_sequence_builder.num_nodes
        m = self.tree_sequence_builder.num_sites
        self.parent = np.zeros(n, dtype=int) - 1
        self.left_child = np.zeros(n, dtype=int) - 1
        self.right_child = np.zeros(n, dtype=int) - 1
        self.left_sib = np.zeros(n, dtype=int) - 1
        self.right_sib = np.zeros(n, dtype=int) - 1
        self.traceback = [{} for _ in range(m)]
        self.max_likelihood_node = np.zeros(m, dtype=int) - 1
        self.allelic_state = np.zeros(n, dtype=int) - 1

        self.likelihood = np.full(n, NONZERO_ROOT, dtype=float)
        self.likelihood_nodes = []
        L_cache = np.zeros_like(self.likelihood) - 1

        # print("MATCH: start=", start, "end = ", end, "h = ", h)
        j = 0
        k = 0
        left = 0
        pos = 0
        right = m
        if j < M and start < Il.peekitem(j)[1].left:
            right = Il.peekitem(j)[1].left
        while j < M and k < M and Il.peekitem(j)[1].left <= start:
            while Ir.peekitem(k)[1].right == pos:
                self.remove_edge(Ir.peekitem(k)[1])
                k += 1
            while j < M and Il.peekitem(j)[1].left == pos:
                self.insert_edge(Il.peekitem(j)[1])
                j += 1
            left = pos
            right = m
            if j < M:
                right = min(right, Il.peekitem(j)[1].left)
            if k < M:
                right = min(right, Ir.peekitem(k)[1].right)
            pos = right
        assert left < right

        for u in range(n):
            if not self.is_root(u):
                self.likelihood[u] = -1

        last_root = 0
        if self.left_child[0] != -1:
            last_root = self.left_child[0]
            assert self.right_sib[last_root] == -1
        self.likelihood_nodes.append(last_root)
        self.likelihood[last_root] = 1

        remove_start = k
        while left < end:
            # print("START OF TREE LOOP", left, right)
            # print("L:", {u: self.likelihood[u] for u in self.likelihood_nodes})
            assert left < right
            for site_index in range(remove_start, k):
                edge = Ir.peekitem(site_index)[1]
                for u in [edge.parent, edge.child]:
                    if self.is_nonzero_root(u):
                        self.likelihood[u] = NONZERO_ROOT
                        if u in self.likelihood_nodes:
                            self.likelihood_nodes.remove(u)
            root = 0
            if self.left_child[0] != -1:
                root = self.left_child[0]
                assert self.right_sib[root] == -1

            if root != last_root:
                if last_root == 0:
                    self.likelihood[last_root] = NONZERO_ROOT
                    self.likelihood_nodes.remove(last_root)
                if self.likelihood[root] == NONZERO_ROOT:
                    self.likelihood[root] = 0
                    self.likelihood_nodes.append(root)
                last_root = root

            if self.extended_checks:
                self.check_likelihoods()
            for site in range(max(left, start), min(right, end)):
                self.update_site(site, h[site])

            remove_start = k
            while k < M and Ir.peekitem(k)[1].right == right:
                edge = Ir.peekitem(k)[1]
                self.remove_edge(edge)
                k += 1
                if self.likelihood[edge.child] == -1:
                    # If the child has an L value, traverse upwards until we
                    # find the parent that carries it. To avoid repeated traversals
                    # along the same path we make a cache of the L values.
                    u = edge.parent
                    while self.likelihood[u] == -1 and L_cache[u] == -1:
                        u = self.parent[u]
                    L_child = L_cache[u]
                    if L_child == -1:
                        L_child = self.likelihood[u]
                    # Fill in the L_cache
                    u = edge.parent
                    while self.likelihood[u] == -1 and L_cache[u] == -1:
                        L_cache[u] = L_child
                        u = self.parent[u]
                    self.likelihood[edge.child] = L_child
                    self.likelihood_nodes.append(edge.child)
            # Clear the L cache
            for site_index in range(remove_start, k):
                edge = Ir.peekitem(site_index)[1]
                u = edge.parent
                while L_cache[u] != -1:
                    L_cache[u] = -1
                    u = self.parent[u]
            assert np.all(L_cache == -1)

            left = right
            while j < M and Il.peekitem(j)[1].left == left:
                edge = Il.peekitem(j)[1]
                self.insert_edge(edge)
                j += 1
                # There's no point in compressing the likelihood tree here as we'll be
                # doing it after we update the first site anyway.
                for u in [edge.parent, edge.child]:
                    if u != 0 and self.likelihood[u] == NONZERO_ROOT:
                        self.likelihood[u] = 0
                        self.likelihood_nodes.append(u)
            right = m
            if j < M:
                right = min(right, Il.peekitem(j)[1].left)
            if k < M:
                right = min(right, Ir.peekitem(k)[1].right)

        return self.run_traceback(start, end, match)

    def run_traceback(self, start, end, match):
        Il = self.tree_sequence_builder.left_index
        Ir = self.tree_sequence_builder.right_index
        M = len(Il)
        u = self.max_likelihood_node[end - 1]
        output_edge = Edge(right=end, parent=u)
        output_edges = [output_edge]
        recombination_required = (
            np.zeros(self.tree_sequence_builder.num_nodes, dtype=int) - 1
        )

        # Now go back through the trees.
        j = M - 1
        k = M - 1
        # Construct the matched haplotype
        match[:] = 0
        match[:start] = tskit.MISSING_DATA
        match[end:] = tskit.MISSING_DATA
        # Reset the tree.
        self.parent[:] = -1
        self.left_child[:] = -1
        self.right_child[:] = -1
        self.left_sib[:] = -1
        self.right_sib[:] = -1

        pos = self.tree_sequence_builder.num_sites
        while pos > start:
            # print("Top of loop: pos = ", pos)
            while k >= 0 and Il.peekitem(k)[1].left == pos:
                edge = Il.peekitem(k)[1]
                self.remove_edge(edge)
                k -= 1
            while j >= 0 and Ir.peekitem(j)[1].right == pos:
                edge = Ir.peekitem(j)[1]
                self.insert_edge(edge)
                j -= 1
            right = pos
            left = 0
            if k >= 0:
                left = max(left, Il.peekitem(k)[1].left)
            if j >= 0:
                left = max(left, Ir.peekitem(j)[1].right)
            pos = left

            assert left < right
            for site_index in range(min(right, end) - 1, max(left, start) - 1, -1):
                u = output_edge.parent
                self.set_allelic_state(site_index)
                v = u
                while self.allelic_state[v] == -1:
                    v = self.parent[v]
                match[site_index] = self.allelic_state[v]
                self.unset_allelic_state(site_index)

                for u, recombine in self.traceback[site_index].items():
                    # Mark the traceback nodes on the tree.
                    recombination_required[u] = recombine
                # Now traverse up the tree from the current node. The first marked node
                # we meet tells us whether we need to recombine.
                u = output_edge.parent
                while u != 0 and recombination_required[u] == -1:
                    u = self.parent[u]
                if recombination_required[u] and site_index > start:
                    output_edge.left = site_index
                    u = self.max_likelihood_node[site_index - 1]
                    output_edge = Edge(right=site_index, parent=u)
                    output_edges.append(output_edge)
                # Reset the nodes in the recombination tree.
                for u in self.traceback[site_index].keys():
                    recombination_required[u] = -1
        output_edge.left = start

        self.mean_traceback_size = sum(len(t) for t in self.traceback) / self.num_sites

        left = np.zeros(len(output_edges), dtype=np.uint32)
        right = np.zeros(len(output_edges), dtype=np.uint32)
        parent = np.zeros(len(output_edges), dtype=np.int32)
        for j, e in enumerate(output_edges):
            assert e.left >= start
            assert e.right <= end
            # TODO this does happen in the C code, so if it ever happends in a Python
            # instance we need to pop the last edge off the list. Or, see why we're
            # generating it in the first place.
            assert e.left < e.right
            left[j] = e.left
            right[j] = e.right
            parent[j] = e.parent

        return left, right, parent


class TestSingleBalancedTreeExample:
    # 4.00┊    0    ┊
    #     ┊    ┃    ┊
    # 3.00┊    7    ┊
    #     ┊  ┏━┻━┓  ┊
    # 2.00┊  5   6  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 1.00┊ 1 2 3 4 ┊
    #     0         4

    @staticmethod
    def ts():
        tables = example_binary(4, 4).dump_tables()
        # Add a site for each sample with a single mutation above that sample.
        for j in range(4):
            tables.sites.add_row(j, "0")
            tables.mutations.add_row(site=j, node=1 + j, derived_state="1")
        return tables.tree_sequence()

    @pytest.mark.parametrize("j", [0, 1, 2, 3])
    def test_match_sample(self, j):
        ts = self.ts()
        am = AncestorMatcher(ts)
        m = 4
        match = np.zeros(m, dtype=int)
        h = np.zeros(m)
        h[j] = 1
        left, right, parent = am.find_path(h, 0, m, match)
        assert list(left) == [0]
        assert list(right) == [m]
        assert list(parent) == [ts.samples()[j]]


class TestMultiTreeExample:
    # 1.84┊     0   ┊    0    ┊
    #     ┊     ┃   ┊    ┃    ┊
    # 0.84┊     8   ┊    8    ┊
    #     ┊   ┏━┻━┓ ┊  ┏━┻━┓  ┊
    # 0.42┊   ┃   ┃ ┊  7   ┃  ┊
    #     ┊   ┃   ┃ ┊ ┏┻┓  ┃  ┊
    # 0.05┊   6   ┃ ┊ ┃ ┃  ┃  ┊
    #     ┊ ┏━┻┓  ┃ ┊ ┃ ┃  ┃  ┊
    # 0.04┊ ┃  5  ┃ ┊ ┃ ┃  5  ┊
    #     ┊ ┃ ┏┻┓ ┃ ┊ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 1 2 3 4 ┊ 1 4 2 3 ┊
    #     0         2         4
    @staticmethod
    def ts():
        nodes = """\
        is_sample       time
        0       1.838075
        1       0.000000
        1       0.000000
        1       0.000000
        1       0.000000
        0       0.041304
        0       0.045967
        0       0.416719
        0       0.838075
        """
        edges = """\
        left    right   parent  child
        0.000000        4.000000       5       2
        0.000000        4.000000       5       3
        0.000000        2.000000       6       1
        0.000000        2.000000       6       5
        2.000000        4.000000       7       1
        2.000000        4.000000       7       4
        0.000000        2.000000       8       4
        2.000000        4.000000       8       5
        0.000000        2.000000       8       6
        2.000000        4.000000       8       7
        0.000000        4.000000       0       8
        """
        ts = tskit.load_text(
            nodes=io.StringIO(nodes), edges=io.StringIO(edges), strict=False
        )
        tables = ts.dump_tables()
        # Add a site for each sample with a single mutation above that sample.
        for j in range(4):
            tables.sites.add_row(j, "0")
            tables.mutations.add_row(site=j, node=1 + j, derived_state="1")
        return tables.tree_sequence()

    @pytest.mark.parametrize("j", [0, 1, 2, 3])
    def test_match_sample(self, j):
        ts = self.ts()
        am = AncestorMatcher(ts)
        m = 4
        match = np.zeros(m, dtype=int)
        h = np.zeros(m)
        h[j] = 1
        left, right, parent = am.find_path(h, 0, m, match)
        assert list(left) == [0]
        assert list(right) == [m]
        assert list(parent) == [ts.samples()[j]]
