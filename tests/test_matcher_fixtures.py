#
# Copyright (C) 2018-2026 University of Oxford
#
# This file is part of tsinfer.
#
# tsinfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tsinfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Matching fixtures: small tree sequences with known match results from the
current AncestorMatcher.  These serve as regression tests when swapping
to AncestorMatcher2 / MatcherIndexes.

Each fixture builds a tree sequence using make_root_ts + extend_ts (the
same path the real pipeline uses), then matches query haplotypes and
asserts the exact path segments and mutations.
"""

import numpy as np
import pytest

from tsinfer import matching, vcz
from tsinfer.grouping import MatchJob

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ArrayReader:
    """Serves rows of a numpy array as haplotypes."""

    def __init__(self, haps):
        self._haps = np.atleast_2d(np.asarray(haps, dtype=np.int8))
        self._idx = 0

    def read_haplotype(self, job):
        h = self._haps[self._idx]
        self._idx += 1
        return h


def _match(ts, positions, query, allele_mapper):
    """Run the current Matcher and return (path, mutations) as tuples."""
    matcher = matching.Matcher(ts, positions, allele_mapper=allele_mapper)
    results = list(matcher.match([None], _ArrayReader([query])))
    _, r = results[0]
    path = [(s.left, s.right, s.parent) for s in r.path]
    mutations = [(m.position, m.derived_state) for m in r.mutations]
    return path, mutations


def _add_node(ts, time, path_segments, mutations, allele_mapper):
    """Add a single ancestor node to *ts* via extend_ts."""
    job = MatchJob(
        haplotype_index=0,
        source="test",
        sample_id="s0",
        ploidy_index=0,
        time=time,
        start_position=0,
        end_position=100,
        group=0,
        node_flags=0,
    )
    result = matching.MatchResult(
        path=[matching.PathSegment(left, r, p) for left, r, p in path_segments],
        mutations=[matching.Mutation(pos, ds) for pos, ds in mutations],
    )
    return matching.extend_ts(
        ts, paired_results=[(job, result)], allele_mapper=allele_mapper
    )


# ---------------------------------------------------------------------------
# Fixtures — tree sequence builders
# ---------------------------------------------------------------------------


def _star_ts():
    """
    Star topology: virtual root (1) with two children.

    Tree [10, 100):

        0 (t=2.0)  ultimate root
        |
        1 (t=1.0)  virtual root
       / \\
      2   3
    (0.5) (0.3)

    Sites at positions [10, 20, 30].
    Node 2 carries mutation at site 1 (pos 20).
    Node 3 carries mutation at site 0 (pos 10).
    Haplotypes: node2=[0,1,0], node3=[1,0,0].
    """
    positions = np.array([10, 20, 30], dtype=np.int32)
    am = vcz.AlleleMapper(3, [["A", "T"]] * 3)
    intervals = np.array([[10, 100]], dtype=np.int32)
    ts = matching.make_root_ts(100.0, positions, intervals, allele_mapper=am)
    ts = _add_node(ts, 0.5, [(10.0, 100.0, 1)], [(20.0, 1)], am)
    ts = _add_node(ts, 0.3, [(10.0, 100.0, 1)], [(10.0, 1)], am)
    return ts, positions, am


def _binary_ts():
    """
    Binary tree: internal node with two leaf children.

    Tree [10, 100):

        0 (t=2.0)
        |
        1 (t=1.0)
        |
        2 (t=0.7)  internal, mutation at site 0
       / \\
      3   4
    (0.4) (0.2)
    mut@2 mut@3

    Sites at positions [10, 20, 30, 40].
    Haplotypes: node2=[1,0,0,0], node3=[1,0,1,0], node4=[1,0,0,1].
    """
    positions = np.array([10, 20, 30, 40], dtype=np.int32)
    am = vcz.AlleleMapper(4, [["A", "T"]] * 4)
    intervals = np.array([[10, 100]], dtype=np.int32)
    ts = matching.make_root_ts(100.0, positions, intervals, allele_mapper=am)
    ts = _add_node(ts, 0.7, [(10.0, 100.0, 1)], [(10.0, 1)], am)
    ts = _add_node(ts, 0.4, [(10.0, 100.0, 2)], [(30.0, 1)], am)
    ts = _add_node(ts, 0.2, [(10.0, 100.0, 2)], [(40.0, 1)], am)
    return ts, positions, am


def _two_tree_ts():
    """
    Two local trees with a breakpoint at position 30.

    Tree [10, 30):        Tree [30, 100):

        0                     0
        |                     |
        1                     1
       / \\                    |
      2   3                   2
                              |
                              3

    Sites at positions [10, 20, 30, 40].
    Node 2 (A, t=0.6): mutation at site 0. Haplotype [1,0,0,0].
    Node 3 (B, t=0.3): copies from virtual root in [10,30),
        from A in [30,100); mutation at site 2.
        Haplotype [1,0,1,0].
    """
    positions = np.array([10, 20, 30, 40], dtype=np.int32)
    am = vcz.AlleleMapper(4, [["A", "T"]] * 4)
    intervals = np.array([[10, 100]], dtype=np.int32)
    ts = matching.make_root_ts(100.0, positions, intervals, allele_mapper=am)
    ts = _add_node(ts, 0.6, [(10.0, 100.0, 1)], [(10.0, 1)], am)
    ts = _add_node(ts, 0.3, [(10.0, 30.0, 1), (30.0, 100.0, 2)], [(30.0, 1)], am)
    return ts, positions, am


def _deep_chain_ts():
    """
    Deep chain: root -> A -> B -> C.

    Tree [10, 100):

        0 (t=2.0)
        |
        1 (t=1.0)
        |
        2 (t=0.8) A, mutation at site 0
        |
        3 (t=0.5) B, mutation at site 1
        |
        4 (t=0.2) C, mutation at site 2

    Sites at positions [10, 20, 30, 40].
    Haplotypes: A=[1,0,0,0], B=[1,1,0,0], C=[1,1,1,0].
    """
    positions = np.array([10, 20, 30, 40], dtype=np.int32)
    am = vcz.AlleleMapper(4, [["A", "T"]] * 4)
    intervals = np.array([[10, 100]], dtype=np.int32)
    ts = matching.make_root_ts(100.0, positions, intervals, allele_mapper=am)
    ts = _add_node(ts, 0.8, [(10.0, 100.0, 1)], [(10.0, 1)], am)
    ts = _add_node(ts, 0.5, [(10.0, 100.0, 2)], [(20.0, 1)], am)
    ts = _add_node(ts, 0.2, [(10.0, 100.0, 3)], [(30.0, 1)], am)
    return ts, positions, am


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStarTree:
    """Match queries against a star-topology tree sequence."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ts, self.positions, self.am = _star_ts()

    def test_copy_node2(self):
        """Query [0,1,0] exactly matches node 2."""
        path, mutations = _match(self.ts, self.positions, [0, 1, 0], self.am)
        assert path == [(0.0, 100.0, 2)]
        assert mutations == []

    def test_copy_node3(self):
        """Query [1,0,0] exactly matches node 3."""
        path, mutations = _match(self.ts, self.positions, [1, 0, 0], self.am)
        assert path == [(0.0, 100.0, 3)]
        assert mutations == []

    def test_mixed(self):
        """Query [1,1,0] recombines between nodes 2 and 3."""
        path, mutations = _match(self.ts, self.positions, [1, 1, 0], self.am)
        assert path == [(20.0, 100.0, 2), (0.0, 20.0, 3)]
        assert mutations == []

    def test_ancestral(self):
        """Query [0,0,0] (all ancestral) copies from virtual root."""
        path, mutations = _match(self.ts, self.positions, [0, 0, 0], self.am)
        assert path == [(0.0, 100.0, 1)]
        assert mutations == []


class TestBinaryTree:
    """Match queries against a binary-topology tree sequence."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ts, self.positions, self.am = _binary_ts()

    def test_copy_leaf_node3(self):
        """Query [1,0,1,0] exactly matches node 3."""
        path, mutations = _match(self.ts, self.positions, [1, 0, 1, 0], self.am)
        assert path == [(0.0, 100.0, 3)]
        assert mutations == []

    def test_copy_leaf_node4(self):
        """Query [1,0,0,1] exactly matches node 4."""
        path, mutations = _match(self.ts, self.positions, [1, 0, 0, 1], self.am)
        assert path == [(0.0, 100.0, 4)]
        assert mutations == []

    def test_copy_internal(self):
        """Query [1,0,0,0] matches internal node 2."""
        path, mutations = _match(self.ts, self.positions, [1, 0, 0, 0], self.am)
        assert path == [(0.0, 100.0, 2)]
        assert mutations == []

    def test_mixed_leaves(self):
        """Query [0,0,1,1] recombines across nodes."""
        path, mutations = _match(self.ts, self.positions, [0, 0, 1, 1], self.am)
        assert path == [
            (40.0, 100.0, 4),
            (30.0, 40.0, 3),
            (0.0, 30.0, 1),
        ]
        assert mutations == []


class TestTwoTrees:
    """Match queries against a tree sequence with two local trees."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ts, self.positions, self.am = _two_tree_ts()

    def test_copy_A(self):
        """Query [1,0,0,0] copies entirely from A (node 2)."""
        path, mutations = _match(self.ts, self.positions, [1, 0, 0, 0], self.am)
        assert path == [(0.0, 100.0, 2)]
        assert mutations == []

    def test_copy_B_site2(self):
        """Query [0,0,1,0] copies from B (node 3)."""
        path, mutations = _match(self.ts, self.positions, [0, 0, 1, 0], self.am)
        assert path == [(0.0, 100.0, 3)]
        assert mutations == []

    def test_recombination(self):
        """Query [1,0,1,0] matches B's haplotype, requiring recombination."""
        path, mutations = _match(self.ts, self.positions, [1, 0, 1, 0], self.am)
        assert path == [(30.0, 100.0, 3), (0.0, 30.0, 2)]
        assert mutations == []

    def test_partial_missing(self):
        """Query [-1,-1,1,0]: only right half non-missing, copies from B."""
        path, mutations = _match(self.ts, self.positions, [-1, -1, 1, 0], self.am)
        assert path == [(30.0, 100.0, 3)]
        assert mutations == []


class TestDeepChain:
    """Match queries against a chain-topology tree sequence."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ts, self.positions, self.am = _deep_chain_ts()

    def test_copy_C(self):
        """Query [1,1,1,0] exactly matches deepest node C (4)."""
        path, mutations = _match(self.ts, self.positions, [1, 1, 1, 0], self.am)
        assert path == [(0.0, 100.0, 4)]
        assert mutations == []

    def test_copy_B(self):
        """Query [1,1,0,0] matches node B (3)."""
        path, mutations = _match(self.ts, self.positions, [1, 1, 0, 0], self.am)
        assert path == [(0.0, 100.0, 3)]
        assert mutations == []

    def test_copy_A(self):
        """Query [1,0,0,0] matches node A (2)."""
        path, mutations = _match(self.ts, self.positions, [1, 0, 0, 0], self.am)
        assert path == [(0.0, 100.0, 2)]
        assert mutations == []

    def test_novel_haplotype(self):
        """Query [0,0,0,1] doesn't match any node — copies virtual root
        with a mutation."""
        path, mutations = _match(self.ts, self.positions, [0, 0, 0, 1], self.am)
        assert path == [(0.0, 100.0, 1)]
        assert mutations == [(40.0, 1)]
