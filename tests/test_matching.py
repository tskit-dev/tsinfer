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

from dataclasses import dataclass

import numpy as np
import tskit

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


def _dummy_result(num_sites):
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
        np.testing.assert_array_equal(ts.tables.sites.position, [42])

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


@dataclass
class _GroupInputs:
    times: np.ndarray
    is_ancestor: np.ndarray
    start_positions: np.ndarray
    end_positions: np.ndarray


class TestComputeGroups:
    """
    Tests for compute_groups, which partitions haplotypes into ordered groups
    for sequential matching.

    Rules:
    1. Index 0 (virtual root) is always alone in groups[0].
    2. Remaining haplotypes are ordered by descending time.
    3. At the same time, ancestor groups come before sample groups.
    4. Same-time ancestors with NON-OVERLAPPING intervals can share a group.
    5. Same-time ancestors with OVERLAPPING intervals must be in separate groups.
    6. Samples are always grouped together by time (no interval splitting).
    """

    def _make_inputs(self, times, is_ancestor, starts=None, ends=None):
        times = np.asarray(times, dtype=np.float64)
        is_ancestor = np.asarray(is_ancestor, dtype=bool)
        n = len(times)
        if starts is None:
            starts = np.zeros(n, dtype=np.int32)
        else:
            starts = np.asarray(starts, dtype=np.int32)
        if ends is None:
            ends = np.ones(n, dtype=np.int32) * 100
        else:
            ends = np.asarray(ends, dtype=np.int32)
        return _GroupInputs(
            times=times,
            is_ancestor=is_ancestor,
            start_positions=starts,
            end_positions=ends,
        )

    # -------------------------------------------------------------------
    # Basic ordering (no interval effects)
    # -------------------------------------------------------------------

    def test_virtual_root_alone(self):
        # Only virtual root
        gi = self._make_inputs([1.0], [True])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert list(groups[0]) == [0]

    def test_virtual_root_is_group_zero(self):
        # Virtual root always first
        gi = self._make_inputs([1.0, 0.5, 0.3], [True, True, True])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]

    def test_multiple_ancestor_time_levels_ordering(self):
        # Root + ancestors at 0.8, 0.5, 0.3 — should be in descending order
        gi = self._make_inputs([1.0, 0.8, 0.5, 0.3], [True, True, True, True])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 4
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]  # time 0.8
        assert list(groups[2]) == [2]  # time 0.5
        assert list(groups[3]) == [3]  # time 0.3

    def test_samples_only_modern(self):
        # Root + 3 modern samples (time=0)
        gi = self._make_inputs([1.0, 0.0, 0.0, 0.0], [True, False, False, False])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 2
        assert list(groups[0]) == [0]
        assert set(groups[1]) == {1, 2, 3}

    def test_ancient_samples_before_modern(self):
        # Root + ancient sample (time=0.5) + modern sample (time=0)
        gi = self._make_inputs([1.0, 0.5, 0.0], [True, False, False])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 3
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]  # ancient sample first
        assert list(groups[2]) == [2]  # modern sample last

    def test_ancestors_before_samples_at_same_time(self):
        # Root + ancestor at t=0.5 + sample at t=0.5
        gi = self._make_inputs([1.0, 0.5, 0.5], [True, True, False])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        # Should have: root, then ancestor, then sample (even at same time)
        assert len(groups) == 3
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]  # ancestor
        assert list(groups[2]) == [2]  # sample

    def test_mixed_ordering(self):
        # Root, ancestors at 0.9 and 0.4, samples at 0.4 and 0.0
        gi = self._make_inputs(
            [1.0, 0.9, 0.4, 0.4, 0.0], [True, True, True, False, False]
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        # All ancestors have default intervals [0, 100] → overlapping → separate groups
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
        gi = self._make_inputs([1.0, 0.5], [True, True])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        for g in groups:
            assert g.dtype == np.int32

    # -------------------------------------------------------------------
    # Interval-based grouping for same-time ancestors
    # -------------------------------------------------------------------

    def test_two_same_time_ancestors_overlapping(self):
        """
        Two ancestors at time 0.5, both spanning the full range → overlap
        → must be in SEPARATE groups.

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |================================|
        idx 2 (t=0.5): |================================|
                        0                              100
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5],
            [True, True, True],
            starts=[0, 0, 0],
            ends=[100, 100, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        # Root alone, then two separate ancestor groups
        assert list(groups[0]) == [0]
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2})]
        assert len(t05_groups) == 2
        assert set(t05_groups[0]) | set(t05_groups[1]) == {1, 2}
        assert len(t05_groups[0]) == 1
        assert len(t05_groups[1]) == 1

    def test_two_same_time_ancestors_disjoint(self):
        """
        Two ancestors at time 0.5 with disjoint intervals → no overlap
        → can share ONE group.

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |==========|
        idx 2 (t=0.5):                    |=============|
                        0        30      60           100
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5],
            [True, True, True],
            starts=[0, 0, 60],
            ends=[100, 30, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        # Both ancestors should be in the same group
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2})]
        assert len(t05_groups) == 1
        assert set(t05_groups[0]) == {1, 2}

    def test_two_same_time_ancestors_adjacent(self):
        """
        Two ancestors at time 0.5, intervals touching at boundary (no overlap).

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |===============|
        idx 2 (t=0.5):                 |================|
                        0             50               100

        start=0,end=50 and start=50,end=100: these do NOT overlap
        (the interval is [start, end), so 50 is not in the first interval).
        → can share ONE group.
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5],
            [True, True, True],
            starts=[0, 0, 50],
            ends=[100, 50, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2})]
        assert len(t05_groups) == 1
        assert set(t05_groups[0]) == {1, 2}

    def test_two_same_time_ancestors_one_site_overlap(self):
        """
        Two ancestors at time 0.5, intervals overlapping by one unit.

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |================|
        idx 2 (t=0.5):                |=================|
                        0            49 50             100

        start=0,end=50 and start=49,end=100: overlap at position 49.
        → SEPARATE groups.
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5],
            [True, True, True],
            starts=[0, 0, 49],
            ends=[100, 50, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2})]
        assert len(t05_groups) == 2

    def test_three_same_time_ancestors_all_disjoint(self):
        """
        Three ancestors at time 0.5, all disjoint → one group.

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |=======|
        idx 2 (t=0.5):           |========|
        idx 3 (t=0.5):                       |==========|
                        0      20 30      50 60       100
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5, 0.5],
            [True, True, True, True],
            starts=[0, 0, 30, 60],
            ends=[100, 20, 50, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2, 3})]
        assert len(t05_groups) == 1
        assert set(t05_groups[0]) == {1, 2, 3}

    def test_three_same_time_ancestors_chain_overlap(self):
        """
        Three ancestors at time 0.5 with chain overlap: A overlaps B,
        B overlaps C, but A does not overlap C.

        idx 0 (root):  |==================================|
        idx 1 (t=0.5): |==============|
        idx 2 (t=0.5):          |==============|
        idx 3 (t=0.5):                    |===============|
                        0       25  35   50  55         100

        Overlaps: 1↔2 (25..35), 2↔3 (50..55). No overlap 1↔3.
        → 1 and 3 can share a group; 2 must be separate.
        → Two groups: {1, 3} and {2}.
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5, 0.5],
            [True, True, True, True],
            starts=[0, 0, 25, 50],
            ends=[100, 35, 55, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2, 3})]
        # Should be exactly 2 groups
        assert len(t05_groups) == 2
        group_sets = [set(g) for g in t05_groups]
        # {1,3} can share (non-overlapping); {2} must be alone
        assert {2} in group_sets
        other = [s for s in group_sets if s != {2}][0]
        assert other == {1, 3}

    def test_three_same_time_ancestors_all_overlapping(self):
        """
        Three ancestors at time 0.5, all pairwise overlapping → three groups.

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |======================|
        idx 2 (t=0.5):      |======================|
        idx 3 (t=0.5):           |======================|
                        0   10  20                 80  90 100

        All three overlap each other → each needs its own group.
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5, 0.5],
            [True, True, True, True],
            starts=[0, 0, 10, 20],
            ends=[100, 80, 90, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2, 3})]
        assert len(t05_groups) == 3
        for g in t05_groups:
            assert len(g) == 1

    def test_same_time_ancestors_at_multiple_time_levels(self):
        """
        Two time levels (0.8 and 0.5), each with two disjoint ancestors.

        idx 0 (root):  |================================|
        idx 1 (t=0.8): |==========|
        idx 2 (t=0.8):                    |=============|
        idx 3 (t=0.5): |==========|
        idx 4 (t=0.5):                    |=============|
                        0        30      60           100

        At t=0.8: {1,2} are disjoint → one group.
        At t=0.5: {3,4} are disjoint → one group.
        """
        gi = self._make_inputs(
            [1.0, 0.8, 0.8, 0.5, 0.5],
            [True, True, True, True, True],
            starts=[0, 0, 60, 0, 60],
            ends=[100, 30, 100, 30, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        assert len(groups) == 3  # root + t=0.8 group + t=0.5 group
        assert set(groups[1]) == {1, 2}  # t=0.8 disjoint
        assert set(groups[2]) == {3, 4}  # t=0.5 disjoint

    def test_samples_ignore_intervals(self):
        """
        Samples are always grouped together by time, regardless of intervals.
        Only ancestors get interval-based splitting.

        idx 0 (root):      |================================|
        idx 1 (sample t=0): |================================|
        idx 2 (sample t=0): |================================|
                             0                              100
        """
        gi = self._make_inputs(
            [1.0, 0.0, 0.0],
            [True, False, False],
            starts=[0, 0, 0],
            ends=[100, 100, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        # Samples always in one group, even with overlapping intervals
        sample_groups = [g for g in groups[1:] if set(g).issubset({1, 2})]
        assert len(sample_groups) == 1
        assert set(sample_groups[0]) == {1, 2}

    def test_single_ancestor_no_splitting_needed(self):
        """
        A single ancestor at a time level needs no splitting.

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |================================|
                        0                              100
        """
        gi = self._make_inputs(
            [1.0, 0.5],
            [True, True],
            starts=[0, 0],
            ends=[100, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 2
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]

    def test_nested_intervals_overlap(self):
        """
        One ancestor's interval is fully contained within the other → overlap.

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |================================|
        idx 2 (t=0.5):          |===========|
                        0      20          60          100

        → SEPARATE groups (2 is inside 1).
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5],
            [True, True, True],
            starts=[0, 0, 20],
            ends=[100, 100, 60],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2})]
        assert len(t05_groups) == 2

    def test_empty_interval_ancestor(self):
        """
        An ancestor with start == end (zero-length interval) doesn't overlap
        with anything.

        idx 0 (root):  |================================|
        idx 1 (t=0.5): |================================|
        idx 2 (t=0.5): (empty, start=50, end=50)
                        0                              100

        → can share a group (zero-length interval can't overlap).
        """
        gi = self._make_inputs(
            [1.0, 0.5, 0.5],
            [True, True, True],
            starts=[0, 0, 50],
            ends=[100, 100, 50],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert list(groups[0]) == [0]
        t05_groups = [g for g in groups[1:] if set(g).issubset({1, 2})]
        assert len(t05_groups) == 1
        assert set(t05_groups[0]) == {1, 2}


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

    def _simple_match_result(self, parent_id, num_sites):
        """A MatchResult that copies from parent_id across all sites."""
        return MatchResult(
            path_left=np.array([0], dtype=np.int32),
            path_right=np.array([num_sites], dtype=np.int32),
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
        np.testing.assert_array_equal(ts2.tables.sites.position, self.positions)

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
        self.num_sites = 5

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
