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
Tests for tsinfer.matching: make_root_ts, compute_groups, Matcher, extend_ts.
"""

from __future__ import annotations

import numpy as np
import tskit

import _tsinfer
from tsinfer.matching import (
    Matcher,
    MatchResult,
    compute_groups,
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

    positions: (n_sites,) int array
    node_times: (n_nodes,) float array (excluding root)
    haplotypes: (n_nodes, n_sites) int8 array
    """
    n_sites = len(positions)
    num_alleles = [2] * n_sites

    tsb = _tsinfer.TreeSequenceBuilder(num_alleles)

    # Add root node (time=1.0)
    root_id = tsb.add_node(1.0)
    assert root_id == 0

    # Add haplotype nodes and edges from root
    node_ids = []
    for time, hap in zip(node_times, haplotypes):
        node_id = tsb.add_node(float(time))
        node_ids.append(node_id)
        # Edge spanning all sites
        tsb.add_path(
            child=node_id,
            left=[0],
            right=[n_sites],
            parent=[root_id],
            compress=False,
        )
        # Mutations where hap == 1
        mut_sites = np.where(np.asarray(hap) == 1)[0].tolist()
        if mut_sites:
            tsb.add_mutations(
                node=node_id,
                site=mut_sites,
                derived_state=[1] * len(mut_sites),
            )

    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {}

    for pos in positions:
        tables.sites.add_row(position=float(pos), ancestral_state="0")

    flags_arr, times_arr = tsb.dump_nodes()
    for t, fl in zip(times_arr, flags_arr):
        tables.nodes.add_row(time=float(t), flags=int(fl))

    el, er, ep, ec = tsb.dump_edges()
    pos_arr = np.asarray(positions, dtype=np.float64)
    for le, re, pe, ce in zip(el, er, ep, ec):
        lc = float(pos_arr[le])
        rc = float(pos_arr[re]) if re < n_sites else float(sequence_length)
        tables.edges.add_row(left=lc, right=rc, parent=int(pe), child=int(ce))

    alleles = ["0", "1"]
    ms, mn, md, mp = tsb.dump_mutations()
    for s, n, d, par in zip(ms, mn, md, mp):
        tables.mutations.add_row(
            site=int(s), node=int(n), derived_state=alleles[int(d)], parent=int(par)
        )

    tables.sort()
    return tables.tree_sequence()


def _dummy_result(n_sites):
    """A MatchResult with no edges or mutations."""
    return MatchResult(
        path_left=np.array([], dtype=np.int32),
        path_right=np.array([], dtype=np.int32),
        path_parent=np.array([], dtype=np.int32),
        mutation_sites=np.array([], dtype=np.int32),
        mutation_state=np.array([], dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# TestMakeRootTs
# ---------------------------------------------------------------------------


class TestMakeRootTs:
    def test_sequence_length(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.sequence_length == 1000.0

    def test_num_nodes_zero(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.num_nodes == 0

    def test_sites_at_correct_positions(self):
        positions = np.array([5, 15, 25, 100], dtype=np.int32)
        intervals = np.array([[5, 101]], dtype=np.int32)
        ts = make_root_ts(200.0, positions, intervals)
        assert ts.num_sites == 4
        site_positions = [int(s.position) for s in ts.sites()]
        assert site_positions == [5, 15, 25, 100]

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
        assert int(list(ts.sites())[0].position) == 42

    def test_no_edges(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.num_edges == 0

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
# TestComputeGroups
# ---------------------------------------------------------------------------


class TestComputeGroups:
    def _make_inputs(self, times, is_ancestor):
        times = np.asarray(times, dtype=np.float64)
        is_ancestor = np.asarray(is_ancestor, dtype=bool)
        n = len(times)
        start_pos = np.zeros(n, dtype=np.int32)
        end_pos = np.ones(n, dtype=np.int32) * 100
        return times, is_ancestor, start_pos, end_pos

    def test_virtual_root_alone(self):
        # Only virtual root
        times, is_anc, sp, ep = self._make_inputs([1.0], [True])
        groups = compute_groups(times, is_anc, sp, ep)
        assert len(groups) == 1
        assert list(groups[0]) == [0]

    def test_virtual_root_is_group_zero(self):
        # Virtual root always first
        times, is_anc, sp, ep = self._make_inputs([1.0, 0.5, 0.3], [True, True, True])
        groups = compute_groups(times, is_anc, sp, ep)
        assert list(groups[0]) == [0]

    def test_single_ancestor_time_level(self):
        # Root + 2 ancestors at same time
        times, is_anc, sp, ep = self._make_inputs([1.0, 0.5, 0.5], [True, True, True])
        groups = compute_groups(times, is_anc, sp, ep)
        assert len(groups) == 2
        assert list(groups[0]) == [0]
        assert set(groups[1]) == {1, 2}

    def test_multiple_ancestor_time_levels_ordering(self):
        # Root + ancestors at 0.8, 0.5, 0.3 — should be in descending order
        times, is_anc, sp, ep = self._make_inputs(
            [1.0, 0.8, 0.5, 0.3], [True, True, True, True]
        )
        groups = compute_groups(times, is_anc, sp, ep)
        assert len(groups) == 4
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]  # time 0.8
        assert list(groups[2]) == [2]  # time 0.5
        assert list(groups[3]) == [3]  # time 0.3

    def test_samples_only_modern(self):
        # Root + 3 modern samples (time=0)
        times, is_anc, sp, ep = self._make_inputs(
            [1.0, 0.0, 0.0, 0.0], [True, False, False, False]
        )
        groups = compute_groups(times, is_anc, sp, ep)
        assert len(groups) == 2
        assert list(groups[0]) == [0]
        assert set(groups[1]) == {1, 2, 3}

    def test_ancient_samples_before_modern(self):
        # Root + ancient sample (time=0.5) + modern sample (time=0)
        times, is_anc, sp, ep = self._make_inputs([1.0, 0.5, 0.0], [True, False, False])
        groups = compute_groups(times, is_anc, sp, ep)
        assert len(groups) == 3
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]  # ancient sample first
        assert list(groups[2]) == [2]  # modern sample last

    def test_ancestors_before_samples_at_same_time(self):
        # Root + ancestor at t=0.5 + sample at t=0.5
        times, is_anc, sp, ep = self._make_inputs([1.0, 0.5, 0.5], [True, True, False])
        groups = compute_groups(times, is_anc, sp, ep)
        # Should have: root, then ancestor, then sample (even at same time)
        assert len(groups) == 3
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]  # ancestor
        assert list(groups[2]) == [2]  # sample

    def test_mixed_ordering(self):
        # Root, ancestors at 0.9 and 0.4, samples at 0.4 and 0.0
        times, is_anc, sp, ep = self._make_inputs(
            [1.0, 0.9, 0.4, 0.4, 0.0], [True, True, True, False, False]
        )
        groups = compute_groups(times, is_anc, sp, ep)
        # groups: [0], [1], [2], [3], [4]
        # group 0: root
        # group 1: ancestor at t=0.9
        # group 2: ancestor at t=0.4
        # group 3: sample at t=0.4
        # group 4: sample at t=0.0
        assert len(groups) == 5
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]
        assert list(groups[2]) == [2]
        assert list(groups[3]) == [3]
        assert list(groups[4]) == [4]

    def test_returns_int32_arrays(self):
        times, is_anc, sp, ep = self._make_inputs([1.0, 0.5], [True, True])
        groups = compute_groups(times, is_anc, sp, ep)
        for g in groups:
            assert g.dtype == np.int32


# ---------------------------------------------------------------------------
# TestMatcher
# ---------------------------------------------------------------------------


class TestMatcher:
    """Tests for Matcher using a simple known tree sequence."""

    def setup_method(self):
        self.positions = np.array([10, 20, 30], dtype=np.int32)
        self.seq_len = 100.0
        self.n_sites = 3

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
        results = matcher.match(np.array([hap]))
        assert len(results) == 1
        r = results[0]
        # Should have path edges (copy from some ancestor)
        assert len(r.path_left) > 0
        # Path left/right/parent should be consistent
        assert len(r.path_left) == len(r.path_right) == len(r.path_parent)

    def test_match_with_mutation(self):
        """Haplotype that differs from ancestor gets a mutation."""
        ancestor_hap = np.array([0, 0, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(ancestor_hap)
        # Query haplotype has derived state at site 1
        query_hap = np.array([0, 1, 0], dtype=np.int8)
        matcher = Matcher(
            ts, self.positions, recombination_rate=1e-4, mismatch_ratio=1.0
        )
        results = matcher.match(np.array([query_hap]))
        r = results[0]
        # There should be a mutation at site 1 (index 1)
        assert 1 in r.mutation_sites

    def test_match_all_missing(self):
        """All-missing haplotype should still return a MatchResult."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        missing_hap = np.array([-1, -1, -1], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = matcher.match(np.array([missing_hap]))
        assert len(results) == 1

    def test_match_partial_missing(self):
        """Haplotype with some missing sites: active range shrinks."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        query_hap = np.array([-1, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = matcher.match(np.array([query_hap]))
        assert len(results) == 1
        r = results[0]
        # Missing site should not appear in mutation_sites
        assert 0 not in r.mutation_sites

    def test_match_returns_one_result_per_haplotype(self):
        """match() returns one MatchResult per input haplotype."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        haplotypes = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = matcher.match(haplotypes)
        assert len(results) == 3

    def test_match_result_arrays_are_numpy(self):
        """MatchResult arrays should be numpy arrays."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = matcher.match(np.array([hap]))
        r = results[0]
        assert isinstance(r.path_left, np.ndarray)
        assert isinstance(r.path_right, np.ndarray)
        assert isinstance(r.path_parent, np.ndarray)
        assert isinstance(r.mutation_sites, np.ndarray)
        assert isinstance(r.mutation_state, np.ndarray)

    def test_match_path_left_right_consistent(self):
        """path_left, path_right, path_parent should have the same length."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = matcher.match(np.array([hap]))
        r = results[0]
        assert len(r.path_left) == len(r.path_right) == len(r.path_parent)

    def test_match_mutation_sites_state_consistent(self):
        """mutation_sites and mutation_state should have the same length."""
        hap = np.array([0, 0, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        query = np.array([1, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = matcher.match(np.array([query]))
        r = results[0]
        assert len(r.mutation_sites) == len(r.mutation_state)


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

    def _simple_match_result(self, parent_id, n_sites):
        """A MatchResult that copies from parent_id across all sites."""
        return MatchResult(
            path_left=np.array([0], dtype=np.int32),
            path_right=np.array([n_sites], dtype=np.int32),
            path_parent=np.array([parent_id], dtype=np.int32),
            mutation_sites=np.array([], dtype=np.int32),
            mutation_state=np.array([], dtype=np.int8),
        )

    def test_node_count_increases(self):
        ts = self._root_ts()
        # First add virtual root node manually via extend_ts
        result = MatchResult(
            path_left=np.array([], dtype=np.int32),
            path_right=np.array([], dtype=np.int32),
            path_parent=np.array([], dtype=np.int32),
            mutation_sites=np.array([], dtype=np.int32),
            mutation_state=np.array([], dtype=np.int8),
        )
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
            ploidy=1,
        )
        assert ts2.num_nodes == 1

        # Add another node
        ts3 = extend_ts(
            ts2,
            node_times=np.array([0.5]),
            results=[self._simple_match_result(0, len(self.positions))],
            node_metadata=[{}],
            create_individuals=np.array([False]),
            ploidy=1,
        )
        assert ts3.num_nodes == 2

    def test_edges_present_after_extend(self):
        ts = self._root_ts()
        # Add root
        root_result = _dummy_result(len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[root_result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        # Add a node with edges
        ts3 = extend_ts(
            ts2,
            node_times=np.array([0.5]),
            results=[self._simple_match_result(0, len(self.positions))],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts3.num_edges > 0

    def test_mutations_present_after_extend(self):
        ts = self._root_ts()
        root_result = _dummy_result(len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[root_result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        # Add node with a mutation at site 1
        result_with_mut = MatchResult(
            path_left=np.array([0], dtype=np.int32),
            path_right=np.array([3], dtype=np.int32),
            path_parent=np.array([0], dtype=np.int32),
            mutation_sites=np.array([1], dtype=np.int32),
            mutation_state=np.array([1], dtype=np.int8),
        )
        ts3 = extend_ts(
            ts2,
            node_times=np.array([0.5]),
            results=[result_with_mut],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts3.num_mutations > 0

    def test_individual_creation_ploidy2(self):
        ts = self._root_ts()
        root_result = _dummy_result(len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[root_result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        # Two haplotypes should become one individual
        r1 = self._simple_match_result(0, len(self.positions))
        r2 = self._simple_match_result(0, len(self.positions))
        ts3 = extend_ts(
            ts2,
            node_times=np.array([0.0, 0.0]),
            results=[r1, r2],
            node_metadata=[{}, {}],
            create_individuals=np.array([True, True]),
            ploidy=2,
        )
        assert ts3.num_individuals == 1
        ind = list(ts3.individuals())[0]
        assert len(ind.nodes) == 2

    def test_metadata_preserved(self):
        ts = self._root_ts()
        root_result = _dummy_result(len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[root_result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        # Metadata should still be present
        meta = ts2.metadata
        assert "sequence_intervals" in meta

    def test_no_individual_when_create_individuals_false(self):
        ts = self._root_ts()
        root_result = _dummy_result(len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[root_result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        r1 = self._simple_match_result(0, len(self.positions))
        r2 = self._simple_match_result(0, len(self.positions))
        ts3 = extend_ts(
            ts2,
            node_times=np.array([0.0, 0.0]),
            results=[r1, r2],
            node_metadata=[{}, {}],
            create_individuals=np.array([False, False]),
            ploidy=2,
        )
        assert ts3.num_individuals == 0

    def test_sites_preserved(self):
        ts = self._root_ts()
        root_result = _dummy_result(len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[root_result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts2.num_sites == len(self.positions)
        site_positions = [int(s.position) for s in ts2.sites()]
        assert site_positions == self.positions.tolist()

    def test_multiple_individuals_ploidy2(self):
        ts = self._root_ts()
        root_result = _dummy_result(len(self.positions))
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[root_result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        # 4 haplotypes → 2 individuals
        results = [self._simple_match_result(0, len(self.positions)) for _ in range(4)]
        ts3 = extend_ts(
            ts2,
            node_times=np.array([0.0, 0.0, 0.0, 0.0]),
            results=results,
            node_metadata=[{}, {}, {}, {}],
            create_individuals=np.array([True, True, True, True]),
            ploidy=2,
        )
        assert ts3.num_individuals == 2


# ---------------------------------------------------------------------------
# TestMatcherExtendCycle
# ---------------------------------------------------------------------------


class TestMatcherExtendCycle:
    """Integration tests: make_root_ts → extend_ts (root) → Matcher → extend_ts."""

    def setup_method(self):
        self.positions = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        self.seq_len = 100.0
        self.intervals = np.array([[10, 51]], dtype=np.int32)
        self.n_sites = 5

    def _build_root_ts_with_virtual_root(self):
        """Create a root TS and add the virtual root node (time=1.0)."""
        ts = make_root_ts(self.seq_len, self.positions, self.intervals)
        root_result = MatchResult(
            path_left=np.array([], dtype=np.int32),
            path_right=np.array([], dtype=np.int32),
            path_parent=np.array([], dtype=np.int32),
            mutation_sites=np.array([], dtype=np.int32),
            mutation_state=np.array([], dtype=np.int8),
        )
        ts2 = extend_ts(
            ts,
            node_times=np.array([1.0]),
            results=[root_result],
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        return ts2

    def test_node_time_correct_after_cycle(self):
        """Node times in the output TS should match the input times."""
        ts = self._build_root_ts_with_virtual_root()
        # Add an ancestor at time 0.5
        ancestor_hap = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = matcher.match(np.array([ancestor_hap]))
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=results,
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        node_times = [n.time for n in ts2.nodes()]
        assert 1.0 in node_times
        assert 0.5 in node_times

    def test_matching_after_two_extends(self):
        """After adding ancestor, matching a similar haplotype gives a path."""
        ts = self._build_root_ts_with_virtual_root()
        ancestor_hap = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        results = matcher.match(np.array([ancestor_hap]))
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=results,
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )

        # Now match a sample against ts2
        sample_hap = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        matcher2 = Matcher(ts2, self.positions, recombination_rate=1e-4)
        results2 = matcher2.match(np.array([sample_hap]))
        r = results2[0]
        # Should have at least one path edge
        assert len(r.path_left) > 0

    def test_num_nodes_accumulates(self):
        """Each extend_ts call should add the expected number of nodes."""
        ts = self._build_root_ts_with_virtual_root()
        assert ts.num_nodes == 1

        # Add ancestors one at a time at different time levels to avoid the
        # multi-child-of-root constraint in the C ancestor matcher.
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        hap1 = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        results1 = matcher.match(hap1)
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.7]),
            results=results1,
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts2.num_nodes == 2  # root + ancestor1

        matcher2 = Matcher(ts2, self.positions, recombination_rate=1e-4)
        hap2 = np.array([[1, 0, 0, 1, 0]], dtype=np.int8)
        results2 = matcher2.match(hap2)
        ts3 = extend_ts(
            ts2,
            node_times=np.array([0.5]),
            results=results2,
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts3.num_nodes == 3  # root + 2 ancestors

        matcher3 = Matcher(ts3, self.positions, recombination_rate=1e-4)
        sample_hap = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        results4 = matcher3.match(sample_hap)
        ts4 = extend_ts(
            ts3,
            node_times=np.array([0.0]),
            results=results4,
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert ts4.num_nodes == 4

    def test_metadata_survives_multiple_cycles(self):
        """sequence_intervals metadata should survive multiple extend_ts calls."""
        ts = self._build_root_ts_with_virtual_root()
        matcher = Matcher(ts, self.positions, recombination_rate=1e-4)
        hap = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        results = matcher.match(hap)
        ts2 = extend_ts(
            ts,
            node_times=np.array([0.5]),
            results=results,
            node_metadata=[{}],
            create_individuals=np.array([False]),
        )
        assert "sequence_intervals" in ts2.metadata
        assert ts2.metadata["sequence_intervals"] == [[10, 51]]
