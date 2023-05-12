"""
Tests for the haplotype matching algorithm.
"""
import collections
import dataclasses
import io
import pickle

import msprime
import numpy as np
import pytest
import tskit

import _tsinfer
import tsinfer
from tsinfer import matching


@dataclasses.dataclass
class Edge:
    left: int = dataclasses.field(default=None)
    right: int = dataclasses.field(default=None)
    parent: int = dataclasses.field(default=None)
    child: int = dataclasses.field(default=None)


# Special values used to indicate compressed paths and nodes that are
# not present in the current tree.


def convert_edge_list(edges, order):
    values = []
    for j in order:
        tsk_edge = edges[j]
        edge = Edge(
            int(tsk_edge.left), int(tsk_edge.right), tsk_edge.parent, tsk_edge.child
        )
        values.append(edge)
    return values


class MatcherIndexes:
    """
    The memory that can be shared between AncestorMatcher instances.
    """

    def __init__(self, in_tables):
        ts = matching.add_vestigial_root(in_tables.tree_sequence())
        tables = ts.dump_tables()

        self.sequence_length = tables.sequence_length
        self.num_nodes = len(tables.nodes)
        self.num_sites = len(tables.sites)

        # Store the edges in left and right order.
        self.left_index = convert_edge_list(
            tables.edges, tables.indexes.edge_insertion_order
        )
        self.right_index = convert_edge_list(
            tables.edges, tables.indexes.edge_removal_order
        )

        # TODO fixme
        self.num_alleles = np.zeros(self.num_sites, dtype=int) + 2
        self.sites_position = np.zeros(ts.num_sites + 1, dtype=np.uint32)
        self.sites_position[:-1] = tables.sites.position
        self.sites_position[-1] = tables.sequence_length
        self.mutations = collections.defaultdict(list)
        last_site = -1
        for mutation in tables.mutations:
            if last_site == mutation.site:
                raise ValueError("Only single mutations supported for now")
            # FIXME - should be allele index
            self.mutations[mutation.site].append((mutation.node, 1))
            last_site = mutation.site


COMPRESSED = -1
NONZERO_ROOT = -2


class AncestorMatcher:
    def __init__(
        self,
        matcher_indexes,
        recombination=None,
        mismatch=None,
        precision=None,
        extended_checks=False,
    ):
        self.matcher_indexes = matcher_indexes
        self.num_sites = matcher_indexes.num_sites
        self.num_nodes = matcher_indexes.num_nodes
        self.mismatch = mismatch
        self.recombination = recombination
        self.precision = 22
        self.extended_checks = extended_checks
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
        for node, state in self.matcher_indexes.mutations[site]:
            self.allelic_state[node] = state

    def unset_allelic_state(self, site):
        """
        Sets the allelic state values for this site back to null.
        """
        # We know that 0 is always a root.
        self.allelic_state[0] = -1
        for node, _ in self.matcher_indexes.mutations[site]:
            self.allelic_state[node] = -1
        assert np.all(self.allelic_state == -1)

    def update_site(self, site, haplotype_state):
        n = self.num_nodes
        rho = self.recombination[site]
        mu = self.mismatch[site]
        num_alleles = self.matcher_indexes.num_alleles[site]
        assert haplotype_state < num_alleles

        self.set_allelic_state(site)

        for node, _ in self.matcher_indexes.mutations[site]:
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

    def find_path(self, h):
        Il = self.matcher_indexes.left_index
        Ir = self.matcher_indexes.right_index
        sequence_length = self.matcher_indexes.sequence_length
        sites_position = self.matcher_indexes.sites_position
        M = len(Il)
        n = self.num_nodes
        m = self.num_sites
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

        start = 0
        while start < m and h[start] == tskit.MISSING_DATA:
            start += 1

        end = m - 1
        while end >= 0 and h[end] == tskit.MISSING_DATA:
            end -= 1
        end += 1

        # print("MATCH: start=", start, "end = ", end, "h = ", h)
        j = 0
        k = 0
        left = 0
        start_pos = sites_position[start]
        end_pos = sites_position[end]
        pos = 0
        right = sequence_length
        if j < M and start_pos < Il[j].left:
            right = Il[j].left
        while j < M and k < M and Il[j].left <= start_pos:
            while Ir[k].right == pos:
                self.remove_edge(Ir[k])
                k += 1
            while j < M and Il[j].left == pos:
                self.insert_edge(Il[j])
                j += 1
            left = pos
            right = sequence_length
            if j < M:
                right = min(right, Il[j].left)
            if k < M:
                right = min(right, Ir[k].right)
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

        current_site = 0
        while sites_position[current_site] < left:
            current_site += 1

        remove_start = k
        while left < end_pos:
            # print("START OF TREE LOOP", left, right)
            # print("L:", {u: self.likelihood[u] for u in self.likelihood_nodes})
            assert left < right
            for e in range(remove_start, k):
                edge = Ir[e]
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

            while left <= sites_position[current_site] < min(right, end_pos):
                self.update_site(current_site, h[current_site])
                current_site += 1

            remove_start = k
            while k < M and Ir[k].right == right:
                edge = Ir[k]
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
            for e in range(remove_start, k):
                edge = Ir[e]
                u = edge.parent
                while L_cache[u] != -1:
                    L_cache[u] = -1
                    u = self.parent[u]
            assert np.all(L_cache == -1)

            left = right
            while j < M and Il[j].left == left:
                edge = Il[j]
                self.insert_edge(edge)
                j += 1
                # There's no point in compressing the likelihood tree here as we'll be
                # doing it after we update the first site anyway.
                for u in [edge.parent, edge.child]:
                    if u != 0 and self.likelihood[u] == NONZERO_ROOT:
                        self.likelihood[u] = 0
                        self.likelihood_nodes.append(u)
            right = sequence_length
            if j < M:
                right = min(right, Il[j].left)
            if k < M:
                right = min(right, Ir[k].right)

        return self.run_traceback(start, end)

    def run_traceback(self, start, end):
        Il = self.matcher_indexes.left_index
        Ir = self.matcher_indexes.right_index
        L = self.matcher_indexes.sequence_length
        sites_position = self.matcher_indexes.sites_position
        M = len(Il)
        u = self.max_likelihood_node[end - 1]
        output_edge = Edge(right=end, parent=u)
        output_edges = [output_edge]
        recombination_required = np.zeros(self.num_nodes, dtype=int) - 1

        # Now go back through the trees.
        j = M - 1
        k = M - 1
        start_pos = sites_position[start]
        end_pos = sites_position[end]
        # Construct the matched haplotype
        match = np.zeros(self.num_sites, dtype=np.int8)
        match[:start] = tskit.MISSING_DATA
        match[end:] = tskit.MISSING_DATA
        # Reset the tree.
        self.parent[:] = -1
        self.left_child[:] = -1
        self.right_child[:] = -1
        self.left_sib[:] = -1
        self.right_sib[:] = -1

        pos = L
        site_index = self.num_sites - 1
        while pos > start_pos:
            # print("Top of loop: pos = ", pos)
            while k >= 0 and Il[k].left == pos:
                edge = Il[k]
                self.remove_edge(edge)
                k -= 1
            while j >= 0 and Ir[j].right == pos:
                edge = Ir[j]
                self.insert_edge(edge)
                j -= 1
            right = pos
            left = 0
            if k >= 0:
                left = max(left, Il[k].left)
            if j >= 0:
                left = max(left, Ir[j].right)
            pos = left

            assert left < right
            while left <= sites_position[site_index] < right:
                if start_pos <= sites_position[site_index] < end_pos:
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
                    # Now traverse up the tree from the current node. The first
                    # marked node we meet tells us whether we need to
                    # recombine.
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
                site_index -= 1

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
            left[j] = sites_position[e.left]
            right[j] = sites_position[e.right]
            parent[j] = e.parent

        # Convert the parent node IDs back to original values
        parent -= 1
        path = matching.Path(left[::-1], right[::-1], parent[::-1])
        return matching.Match(path, match)


def run_match(ts, h):
    h = h.astype(np.int8)
    assert len(h) == ts.num_sites
    recombination = np.zeros(ts.num_sites) + 1e-9
    mismatch = np.zeros(ts.num_sites)
    precision = 22
    matcher_indexes = MatcherIndexes(ts.tables)
    matcher = AncestorMatcher(
        matcher_indexes,
        recombination=recombination,
        mismatch=mismatch,
        precision=precision,
    )
    match_py = matcher.find_path(h)

    mi = tsinfer.MatcherIndexes(ts)
    am = tsinfer.AncestorMatcher2(
        mi, recombination=recombination, mismatch=mismatch, precision=precision
    )
    match_c = am.find_match(h)
    match_py.assert_equals(match_c)

    return match_py


class TestMatchClassUtils:
    def test_pickle(self):
        m1 = matching.Match(
            matching.Path(np.array([0]), np.array([1]), np.array([0])), np.array([0])
        )
        m2 = pickle.loads(pickle.dumps(m1))
        m1.assert_equals(m2)


# TODO the tests on these two classes are the same right now, should
# refactor.


def add_unique_sample_mutations(ts):
    """
    Adds a mutation for each of the samples at equally spaced locations
    along the genome.
    """
    tables = ts.dump_tables()
    L = int(ts.sequence_length)
    assert L % ts.num_samples == 0
    gap = L // ts.num_samples
    x = 0
    for u in ts.samples():
        site = tables.sites.add_row(position=x, ancestral_state="0")
        tables.mutations.add_row(site=site, derived_state="1", node=u)
        x += gap
    return tables.tree_sequence()


class TestSingleBalancedTreeExample:
    # 3.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 2.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 1.00┊ 0 1 2 3 ┊
    #     0         8

    @staticmethod
    def ts():
        return add_unique_sample_mutations(
            tskit.Tree.generate_balanced(4, span=8).tree_sequence
        )

    @pytest.mark.parametrize("j", [0, 1, 2, 3])
    def test_match_sample(self, j):
        ts = self.ts()
        h = np.zeros(4)
        h[j] = 1
        m = run_match(ts, h)
        assert list(m.path.left) == [0]
        assert list(m.path.right) == [ts.sequence_length]
        assert list(m.path.parent) == [ts.samples()[j]]
        np.testing.assert_array_equal(h, m.matched_haplotype)

    @pytest.mark.parametrize("j", [1, 2])
    def test_match_sample_missing_flanks(self, j):
        ts = self.ts()
        h = np.zeros(4)
        h[0] = -1
        h[-1] = -1
        h[j] = 1
        m = run_match(ts, h)
        assert list(m.path.left) == [2]
        assert list(m.path.right) == [6]
        assert list(m.path.parent) == [ts.samples()[j]]
        np.testing.assert_array_equal(h, m.matched_haplotype)

    def test_switch_each_sample(self):
        ts = self.ts()
        h = np.ones(4)
        m = run_match(ts, h)
        assert list(m.path.left) == [0, 2, 4, 6]
        assert list(m.path.right) == [2, 4, 6, 8]
        assert list(m.path.parent) == [0, 1, 2, 3]
        np.testing.assert_array_equal(h, m.matched_haplotype)

    def test_switch_each_sample_missing_flanks(self):
        ts = self.ts()
        h = np.ones(4)
        h[0] = -1
        h[-1] = -1
        m = run_match(ts, h)
        assert list(m.path.left) == [2, 4]
        assert list(m.path.right) == [4, 6]
        assert list(m.path.parent) == [1, 2]
        np.testing.assert_array_equal(h, m.matched_haplotype)


class TestMultiTreeExample:
    # 0.84┊     7   ┊    7    ┊
    #     ┊   ┏━┻━┓ ┊  ┏━┻━┓  ┊
    # 0.42┊   ┃   ┃ ┊  6   ┃  ┊
    #     ┊   ┃   ┃ ┊ ┏┻┓  ┃  ┊
    # 0.05┊   5   ┃ ┊ ┃ ┃  ┃  ┊
    #     ┊ ┏━┻┓  ┃ ┊ ┃ ┃  ┃  ┊
    # 0.04┊ ┃  4  ┃ ┊ ┃ ┃  4  ┊
    #     ┊ ┃ ┏┻┓ ┃ ┊ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊ 0 3 1 2 ┊
    #     0         2         4
    @staticmethod
    def ts():
        nodes = """\
        is_sample       time
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
        0.000000        4.000000       4       1
        0.000000        4.000000       4       2
        0.000000        2.000000       5       0
        0.000000        2.000000       5       4
        2.000000        4.000000       6       0
        2.000000        4.000000       6       3
        0.000000        2.000000       7       3
        2.000000        4.000000       7       4
        0.000000        2.000000       7       5
        2.000000        4.000000       7       6
        """
        ts = tskit.load_text(
            nodes=io.StringIO(nodes), edges=io.StringIO(edges), strict=False
        )
        return add_unique_sample_mutations(ts)

    @pytest.mark.parametrize("j", [0, 1, 2, 3])
    def test_match_sample(self, j):
        ts = self.ts()
        h = np.zeros(4)
        h[j] = 1
        m = run_match(self.ts(), h)
        assert list(m.path.left) == [0]
        assert list(m.path.right) == [4]
        assert list(m.path.parent) == [ts.samples()[j]]
        np.testing.assert_array_equal(h, m.matched_haplotype)

    def test_switch_each_sample(self):
        ts = self.ts()
        h = np.ones(4)
        m = run_match(ts, h)
        assert list(m.path.left) == [0, 1, 2, 3]
        assert list(m.path.right) == [1, 2, 3, 4]
        assert list(m.path.parent) == [0, 1, 2, 3]
        np.testing.assert_array_equal(h, m.matched_haplotype)

    def test_switch_each_sample_missing_flanks(self):
        ts = self.ts()
        h = np.ones(4)
        h[0] = -1
        h[-1] = -1
        m = run_match(ts, h)
        assert list(m.path.left) == [1, 2]
        assert list(m.path.right) == [2, 3]
        assert list(m.path.parent) == [1, 2]
        np.testing.assert_array_equal(h, m.matched_haplotype)


class TestSimulationExamples:
    def check_exact_sample_matches(self, ts):
        H = ts.genotype_matrix().T
        for u, h in zip(ts.samples(), H):
            m = run_match(ts, h)
            np.testing.assert_array_equal(h, m.matched_haplotype)
            assert list(m.path.left) == [0]
            assert list(m.path.right) == [ts.sequence_length]
            assert list(m.path.parent) == [u]

    def check_switch_all_samples(self, ts):
        h = np.ones(ts.num_sites, dtype=np.int8)
        m = run_match(ts, h)
        X = np.append(ts.sites_position, [ts.sequence_length])
        np.testing.assert_array_equal(h, m.matched_haplotype)
        np.testing.assert_array_equal(m.path.left, X[:-1])
        np.testing.assert_array_equal(m.path.right, X[1:])
        np.testing.assert_array_equal(m.path.parent, ts.samples())

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_single_tree_exact_match(self, n):
        ts = msprime.sim_ancestry(n, sequence_length=100, random_seed=2)
        ts = add_unique_sample_mutations(ts)
        self.check_exact_sample_matches(ts)

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_multiple_trees_exact_match(self, n):
        ts = msprime.sim_ancestry(
            n, sequence_length=20, recombination_rate=0.1, random_seed=2234
        )
        assert ts.num_trees > 1
        ts = add_unique_sample_mutations(ts)
        self.check_exact_sample_matches(ts)

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_single_tree_switch_all_samples(self, n):
        ts = msprime.sim_ancestry(n, sequence_length=100, random_seed=2345)
        ts = add_unique_sample_mutations(ts)
        self.check_switch_all_samples(ts)

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_multiple_trees_switch_all_sample(self, n):
        ts = msprime.sim_ancestry(
            n, sequence_length=20, recombination_rate=0.1, random_seed=12234
        )
        assert ts.num_trees > 1
        ts = add_unique_sample_mutations(ts)
        self.check_switch_all_samples(ts)
