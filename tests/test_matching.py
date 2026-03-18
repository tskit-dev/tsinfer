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
Tests for tsinfer.matching: make_root_ts, Matcher, extend_ts.
"""

from __future__ import annotations

import numpy as np
import tskit

from tsinfer.matching import (
    Matcher,
    MatchResult,
    Mutation,
    PathSegment,
    extend_ts,
    make_root_ts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_ts(positions, sequence_length, node_times, haplotypes):
    """
    Build a minimal tree sequence with one node per haplotype, connected to a
    virtual root (node 0, time=1.0) via edges spanning all sites, with
    mutations where the haplotype has value 1.

    positions: (num_sites,) int array
    node_times: (n_nodes,) float array (excluding root)
    haplotypes: (n_nodes, num_sites) int8 array
    """
    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {}
    for pos in positions:
        tables.sites.add_row(position=float(pos), ancestral_state="0")
    # node 0 = virtual root
    tables.nodes.add_row(time=1.0, flags=0)
    for time, hap in zip(node_times, haplotypes):
        node_id = tables.nodes.add_row(time=float(time), flags=0)
        # single edge spanning full sequence
        tables.edges.add_row(
            left=float(positions[0]),
            right=float(sequence_length),
            parent=0,
            child=node_id,
        )
        # mutations where hap == 1
        for site_idx in np.where(np.asarray(hap) == 1)[0]:
            tables.mutations.add_row(site=int(site_idx), node=node_id, derived_state="1")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


class _ArrayReader:
    """Adapter that serves rows of a numpy array as read_haplotype(job)."""

    def __init__(self, haplotypes):
        self._haps = np.atleast_2d(np.asarray(haplotypes, dtype=np.int8))
        self._idx = 0

    def read_haplotype(self, job):
        h = self._haps[self._idx]
        self._idx += 1
        return h


def _jobs(n):
    """Return *n* opaque dummy jobs (the reader ignores them)."""
    return [None] * n


def _results_only(paired):
    """Extract MatchResult list from [(job, result), ...] pairs."""
    return [result for _, result in paired]


def _dummy_result():
    """A MatchResult with no edges or mutations."""
    return MatchResult(path=[], mutations=[])


# ---------------------------------------------------------------------------
# TestMakeRootTs
# ---------------------------------------------------------------------------


class TestMakeRootTs:
    def test_sequence_length(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.sequence_length == 1000.0

    def test_num_nodes_two(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.num_nodes == 2

    def test_sites_at_correct_positions(self):
        positions = np.array([5, 15, 25, 100], dtype=np.int32)
        intervals = np.array([[5, 101]], dtype=np.int32)
        ts = make_root_ts(200.0, positions, intervals)
        assert ts.num_sites == 4
        np.testing.assert_array_equal(ts.tables.sites.position, [5, 15, 25, 100])

    def test_sequence_intervals_in_metadata(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31], [100, 200]], dtype=np.int32)
        ts = make_root_ts(500.0, positions, intervals)
        meta = ts.metadata
        assert "sequence_intervals" in meta
        assert meta["sequence_intervals"] == [[10, 31], [100, 200]]

    def test_single_site(self):
        positions = np.array([42], dtype=np.int32)
        intervals = np.array([[42, 43]], dtype=np.int32)
        ts = make_root_ts(100.0, positions, intervals)
        assert ts.num_sites == 1
        assert ts.num_nodes == 2
        assert ts.num_edges == 1
        np.testing.assert_array_equal(ts.tables.sites.position, [42])

    def test_one_edge(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.num_edges == 1

    def test_no_mutations(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.num_mutations == 0

    def test_sequence_length_float(self):
        positions = np.array([1, 2, 3], dtype=np.int32)
        intervals = np.array([[1, 4]], dtype=np.int32)
        ts = make_root_ts(999.5, positions, intervals)
        assert ts.sequence_length == 999.5


# ---------------------------------------------------------------------------
# TestMatcher
# ---------------------------------------------------------------------------


class TestMatcher:
    """Tests for Matcher using a simple known tree sequence."""

    def setup_method(self):
        self.positions = np.array([10, 20, 30], dtype=np.int32)
        self.seq_len = 100.0
        self.num_sites = 3

    def _make_ts_with_root_only(self):
        """Empty tree sequence (just sites, no nodes)."""
        return make_root_ts(
            self.seq_len,
            self.positions,
            np.array([[10, 31]], dtype=np.int32),
        )

    def _make_ts_with_one_ancestor(self, hap):
        """Tree sequence with virtual root + one ancestor haplotype."""
        return _make_simple_ts(
            self.positions,
            self.seq_len,
            node_times=[0.5],
            haplotypes=[hap],
        )

    def test_match_identical_haplotype(self):
        """Matching an identical haplotype returns a MatchResult with path edges."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        assert len(results) == 1
        _, r = results[0]
        # Should have path segments (copy from some ancestor)
        assert len(r.path) > 0

    def test_match_with_mutation(self):
        """Haplotype that differs from ancestor gets a mutation."""
        ancestor_hap = np.array([0, 0, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(ancestor_hap)
        # Query haplotype has derived state at site 1
        query_hap = np.array([0, 1, 0], dtype=np.int8)
        matcher = Matcher(
            ts, self.positions, recombination_rate=1e-4, mismatch_ratio=1.0
        )
        results = list(matcher.match(_jobs(1), _ArrayReader([query_hap])))
        _, r = results[0]
        # There should be a mutation at position 20 (site index 1)
        mut_positions = [m.position for m in r.mutations]
        assert 20.0 in mut_positions

    def test_match_all_missing(self):
        """All-missing haplotype should still return a MatchResult."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        missing_hap = np.array([-1, -1, -1], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(1), _ArrayReader([missing_hap])))
        assert len(results) == 1

    def test_match_partial_missing(self):
        """Haplotype with some missing sites: active range shrinks."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        query_hap = np.array([-1, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(1), _ArrayReader([query_hap])))
        assert len(results) == 1
        _, r = results[0]
        # Missing site (position 10) should not appear in mutations
        mut_positions = [m.position for m in r.mutations]
        assert 10.0 not in mut_positions

    def test_match_returns_one_result_per_haplotype(self):
        """match() returns one MatchResult per input haplotype."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        haplotypes = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(3), _ArrayReader(haplotypes)))
        assert len(results) == 3

    def test_match_result_types(self):
        """MatchResult should contain PathSegment and Mutation objects."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        _, r = results[0]
        assert isinstance(r.path, list)
        assert isinstance(r.mutations, list)
        for seg in r.path:
            assert isinstance(seg, PathSegment)
            assert isinstance(seg.left, float)
            assert isinstance(seg.right, float)
            assert isinstance(seg.parent, int)

    def test_path_segments_have_valid_coordinates(self):
        """PathSegment left < right for every segment."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        _, r = results[0]
        for seg in r.path:
            assert seg.left < seg.right

    def test_mutations_have_valid_fields(self):
        """Mutation objects should have position and derived_state."""
        hap = np.array([0, 0, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        query = np.array([1, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(1), _ArrayReader([query])))
        _, r = results[0]
        assert len(r.mutations) > 0
        for m in r.mutations:
            assert isinstance(m, Mutation)
            assert isinstance(m.position, float)
            assert isinstance(m.derived_state, int)


# ---------------------------------------------------------------------------
# TestExtendTs
# ---------------------------------------------------------------------------


class TestExtendTs:
    def setup_method(self):
        self.positions = np.array([10, 20, 30], dtype=np.int32)
        self.seq_len = 100.0
        self.intervals = np.array([[10, 31]], dtype=np.int32)

    def _root_ts(self):
        return make_root_ts(self.seq_len, self.positions, self.intervals)

    def _simple_match_result(self, parent_id, num_sites):
        """A MatchResult that copies from parent_id across all sites."""
        left_pos = float(self.positions[0])
        right_pos = self.seq_len
        return MatchResult(
            path=[PathSegment(left=left_pos, right=right_pos, parent=parent_id)],
            mutations=[],
        )

    def test_node_count_increases(self):
        ts = self._root_ts()
        assert ts.num_nodes == 2  # ultimate root + virtual root

        # Add a node
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=[self._simple_match_result(1, len(self.positions))],
            node_metadata=[{}],
            create_individuals=np.array([False]),
            ploidy=1,
        )
        assert ts2.num_nodes == 3

    def test_edges_present_after_extend(self):
        ts = self._root_ts()
        # Add a node with edges (parent=1 is the virtual root)
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=[self._simple_match_result(1, len(self.positions))],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts2.num_edges > 0

    def test_mutations_present_after_extend(self):
        ts = self._root_ts()
        # Add node with a mutation at site 1 (parent=1 is virtual root)
        result_with_mut = MatchResult(
            path=[PathSegment(left=10.0, right=100.0, parent=1)],
            mutations=[Mutation(position=20.0, derived_state=1)],
        )
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=[result_with_mut],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts2.num_mutations > 0

    def test_individual_creation_ploidy2(self):
        ts = self._root_ts()
        # Two haplotypes should become one individual (parent=1 is virtual root)
        r1 = self._simple_match_result(1, len(self.positions))
        r2 = self._simple_match_result(1, len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.0, 0.0]),
            results=[r1, r2],
            node_metadata=[{}, {}],
            create_individuals=np.array([True, True]),
            ploidy=2,
        )
        assert ts2.num_individuals == 1
        ind = list(ts2.individuals())[0]
        assert len(ind.nodes) == 2

    def test_metadata_preserved(self):
        ts = self._root_ts()
        # Root TS already has metadata
        assert "sequence_intervals" in ts.metadata
        # Extend and check metadata survives
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=[self._simple_match_result(1, len(self.positions))],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        meta = ts2.metadata
        assert "sequence_intervals" in meta

    def test_no_individual_when_create_individuals_false(self):
        ts = self._root_ts()
        r1 = self._simple_match_result(1, len(self.positions))
        r2 = self._simple_match_result(1, len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.0, 0.0]),
            results=[r1, r2],
            node_metadata=[{}, {}],
            create_individuals=np.array([False, False]),
            ploidy=2,
        )
        assert ts2.num_individuals == 0

    def test_sites_preserved(self):
        ts = self._root_ts()
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=[self._simple_match_result(1, len(self.positions))],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts2.num_sites == len(self.positions)
        np.testing.assert_array_equal(ts2.tables.sites.position, self.positions)

    def test_multiple_individuals_ploidy2(self):
        ts = self._root_ts()
        # 4 haplotypes → 2 individuals (parent=1 is virtual root)
        results = [self._simple_match_result(1, len(self.positions)) for _ in range(4)]
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.0, 0.0, 0.0, 0.0]),
            results=results,
            node_metadata=[{}, {}, {}, {}],
            create_individuals=np.array([True, True, True, True]),
            ploidy=2,
        )
        assert ts2.num_individuals == 2


# ---------------------------------------------------------------------------
# TestMatcherExtendCycle
# ---------------------------------------------------------------------------


class TestMatcherExtendCycle:
    """Integration tests: make_root_ts → extend_ts (root) → Matcher → extend_ts."""

    def setup_method(self):
        self.positions = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        self.seq_len = 100.0
        self.intervals = np.array([[10, 51]], dtype=np.int32)
        self.num_sites = 5

    def _build_root_ts(self):
        """Create a root TS with ultimate root + virtual root (2 nodes)."""
        return make_root_ts(self.seq_len, self.positions, self.intervals)

    def test_node_time_correct_after_cycle(self):
        """Node times in the output TS should match the input times."""
        ts = self._build_root_ts()
        # Add an ancestor at time 0.5
        ancestor_hap = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(1), _ArrayReader([ancestor_hap])))
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=_results_only(results),
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        node_times = [n.time for n in ts2.nodes()]
        assert 1.0 in node_times
        assert 0.5 in node_times

    def test_matching_after_two_extends(self):
        """After adding ancestor, matching a similar haplotype gives a path."""
        ts = self._build_root_ts()
        ancestor_hap = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = list(matcher.match(_jobs(1), _ArrayReader([ancestor_hap])))
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=_results_only(results),
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )

        # Now match a sample against ts2
        sample_hap = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        matcher2 = Matcher(ts2, self.positions, recombination_rate=1e-4)
        results2 = list(matcher2.match(_jobs(1), _ArrayReader([sample_hap])))
        _, r = results2[0]
        # Should have at least one path edge
        assert len(r.path) > 0

    def test_num_nodes_accumulates(self):
        """Each extend_ts call should add the expected number of nodes."""
        ts = self._build_root_ts()
        assert ts.num_nodes == 2  # ultimate root + virtual root

        # Add ancestors one at a time at different time levels to avoid the
        # multi-child-of-root constraint in the C ancestor matcher.
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        hap1 = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        results1 = list(matcher.match(_jobs(1), _ArrayReader(hap1)))
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.7]),
            results=_results_only(results1),
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts2.num_nodes == 3  # 2 roots + ancestor1

        matcher2 = Matcher(ts2, self.positions, recombination_rate=1e-4)
        hap2 = np.array([[1, 0, 0, 1, 0]], dtype=np.int8)
        results2 = list(matcher2.match(_jobs(1), _ArrayReader(hap2)))
        ts3 = extend_ts(
            ts2,
            node_times=np.array([0.5]),
            results=_results_only(results2),
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts3.num_nodes == 4  # 2 roots + 2 ancestors

        matcher3 = Matcher(ts3, self.positions, recombination_rate=1e-4)
        sample_hap = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        results4 = list(matcher3.match(_jobs(1), _ArrayReader(sample_hap)))
        ts4 = extend_ts(
            ts3,
            node_times=np.array([0.0]),
            results=_results_only(results4),
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts4.num_nodes == 5

    def test_metadata_survives_multiple_cycles(self):
        """sequence_intervals metadata should survive multiple extend_ts calls."""
        ts = self._build_root_ts()
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        hap = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        results = list(matcher.match(_jobs(1), _ArrayReader(hap)))
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=_results_only(results),
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert "sequence_intervals" in ts2.metadata
        assert ts2.metadata["sequence_intervals"] == [[10, 51]]
