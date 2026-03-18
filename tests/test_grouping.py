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
Tests for tsinfer.grouping: merge_overlapping_ancestors, group_ancestors_by_linesweep,
find_groups, collect_haplotype_metadata, compute_groups, compute_groups_json.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pytest
from helpers import make_sample_vcz

from tsinfer.ancestors import infer_ancestors
from tsinfer.config import AncestorsConfig, Config, MatchConfig, Source
from tsinfer.grouping import (
    HaplotypeMetadata,
    MatchJob,
    collect_haplotype_metadata,
    compute_groups,
    compute_groups_json,
    compute_match_jobs,
    find_groups,
    group_ancestors_by_linesweep,
    merge_overlapping_ancestors,
)


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
    1. Haplotypes are ordered by descending time.
    2. At the same time, ancestor groups come before sample groups.
    3. Same-time ancestors with NON-OVERLAPPING intervals can share a group.
    4. Same-time ancestors with OVERLAPPING intervals are in the SAME group
       (so they don't match against each other).
    5. Samples are always grouped together by time (no interval splitting).
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

    def test_single_ancestor(self):
        # One ancestor only
        gi = self._make_inputs([0.5], [True])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert list(groups[0]) == [0]

    def test_multiple_ancestor_time_levels_ordering(self):
        # Ancestors at 0.8, 0.5, 0.3 — should be in descending order
        gi = self._make_inputs([0.8, 0.5, 0.3], [True, True, True])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 3
        assert list(groups[0]) == [0]  # time 0.8
        assert list(groups[1]) == [1]  # time 0.5
        assert list(groups[2]) == [2]  # time 0.3

    def test_samples_only_modern(self):
        # 3 modern samples (time=0)
        gi = self._make_inputs([0.0, 0.0, 0.0], [False, False, False])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_ancient_samples_before_modern(self):
        # Ancient sample (time=0.5) + modern sample (time=0)
        gi = self._make_inputs([0.5, 0.0], [False, False])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 2
        assert list(groups[0]) == [0]  # ancient sample first
        assert list(groups[1]) == [1]  # modern sample last

    def test_ancestors_before_samples_at_same_time(self):
        # Ancestor at t=0.5 + sample at t=0.5
        gi = self._make_inputs([0.5, 0.5], [True, False])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        # Should have: ancestor, then sample (even at same time)
        assert len(groups) == 2
        assert list(groups[0]) == [0]  # ancestor
        assert list(groups[1]) == [1]  # sample

    def test_mixed_ordering(self):
        # Ancestors at 0.9 and 0.4, samples at 0.4 and 0.0
        gi = self._make_inputs([0.9, 0.4, 0.4, 0.0], [True, True, False, False])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        # All ancestors have default intervals [0, 100] -> overlapping -> separate groups
        # group 0: ancestor at t=0.9
        # group 1: ancestor at t=0.4
        # group 2: sample at t=0.4
        # group 3: sample at t=0.0
        assert len(groups) == 4
        assert list(groups[0]) == [0]
        assert list(groups[1]) == [1]
        assert list(groups[2]) == [2]
        assert list(groups[3]) == [3]

    def test_returns_int32_arrays(self):
        gi = self._make_inputs([0.5], [True])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        for g in groups:
            assert g.dtype == np.int32

    def test_empty_returns_empty(self):
        gi = self._make_inputs([], [])
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 0

    # -------------------------------------------------------------------
    # Interval-based grouping for same-time ancestors
    # -------------------------------------------------------------------

    def test_two_same_time_ancestors_overlapping(self):
        """
        Two ancestors at time 0.5, both spanning the full range -> overlap
        -> must be in SAME group (so they don't match against each other).
        """
        gi = self._make_inputs(
            [0.5, 0.5],
            [True, True],
            starts=[0, 0],
            ends=[100, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_two_same_time_ancestors_disjoint(self):
        """
        Two ancestors at time 0.5 with disjoint intervals -> no overlap
        -> can share ONE group.
        """
        gi = self._make_inputs(
            [0.5, 0.5],
            [True, True],
            starts=[0, 60],
            ends=[30, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_two_same_time_ancestors_adjacent(self):
        """
        Two ancestors at time 0.5, intervals touching at boundary (no overlap).
        -> can share ONE group.
        """
        gi = self._make_inputs(
            [0.5, 0.5],
            [True, True],
            starts=[0, 50],
            ends=[50, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_two_same_time_ancestors_one_site_overlap(self):
        """
        Two ancestors at time 0.5, intervals overlapping by one unit.
        -> SAME group (so they don't match against each other).
        """
        gi = self._make_inputs(
            [0.5, 0.5],
            [True, True],
            starts=[0, 49],
            ends=[50, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_three_same_time_ancestors_all_disjoint(self):
        """Three ancestors at time 0.5, all disjoint -> one group."""
        gi = self._make_inputs(
            [0.5, 0.5, 0.5],
            [True, True, True],
            starts=[0, 30, 60],
            ends=[20, 50, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_three_same_time_ancestors_chain_overlap(self):
        """Three ancestors at time 0.5 with chain overlap -> SAME group."""
        gi = self._make_inputs(
            [0.5, 0.5, 0.5],
            [True, True, True],
            starts=[0, 25, 50],
            ends=[35, 55, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_three_same_time_ancestors_all_overlapping(self):
        """Three ancestors at time 0.5, all pairwise overlapping -> SAME group."""
        gi = self._make_inputs(
            [0.5, 0.5, 0.5],
            [True, True, True],
            starts=[0, 10, 20],
            ends=[80, 90, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_same_time_ancestors_at_multiple_time_levels(self):
        """Two time levels (0.8 and 0.5), each with two disjoint ancestors."""
        gi = self._make_inputs(
            [0.8, 0.8, 0.5, 0.5],
            [True, True, True, True],
            starts=[0, 60, 0, 60],
            ends=[30, 100, 30, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 2  # t=0.8 group + t=0.5 group
        assert set(groups[0]) == {0, 1}  # t=0.8 disjoint
        assert set(groups[1]) == {2, 3}  # t=0.5 disjoint

    def test_samples_ignore_intervals(self):
        """Samples are always grouped together by time, regardless of intervals."""
        gi = self._make_inputs(
            [0.0, 0.0],
            [False, False],
            starts=[0, 0],
            ends=[100, 100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_single_ancestor_no_splitting_needed(self):
        """A single ancestor at a time level needs no splitting."""
        gi = self._make_inputs(
            [0.5],
            [True],
            starts=[0],
            ends=[100],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert list(groups[0]) == [0]

    def test_nested_intervals_overlap(self):
        """Nested intervals -> SAME group."""
        gi = self._make_inputs(
            [0.5, 0.5],
            [True, True],
            starts=[0, 20],
            ends=[100, 60],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_empty_interval_ancestor(self):
        """Zero-length interval can't overlap -> can share a group."""
        gi = self._make_inputs(
            [0.5, 0.5],
            [True, True],
            starts=[0, 50],
            ends=[100, 50],
        )
        groups = compute_groups(
            gi.times, gi.is_ancestor, gi.start_positions, gi.end_positions
        )
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}


class TestGroupAncestorsLinesweep:
    merging_fixed_test_cases = [
        dict(
            name="empty_input",
            start=[],
            end=[],
            time=[],
            new_start=[],
            new_end=[],
            new_time=[],
            old_indexes={},
            sort_indices=[],
        ),
        dict(
            name="single_ancestor",
            start=[0],
            end=[1],
            time=[0],
            new_start=[0],
            new_end=[1],
            new_time=[0],
            old_indexes={0: [0]},
            sort_indices=[0],
        ),
        dict(
            name="two_ancestors_diff_time",
            start=[0, 0],
            end=[1, 1],
            time=[1, 0],
            new_start=[0, 0],
            new_end=[1, 1],
            new_time=[0, 1],
            old_indexes={0: [0], 1: [1]},
            sort_indices=[1, 0],
        ),
        dict(
            name="two_ancestors_complete_overlap",
            start=[0, 0],
            end=[1, 1],
            time=[0, 0],
            new_start=[0],
            new_end=[1],
            new_time=[0],
            old_indexes={0: [0, 1]},
            sort_indices=[0, 1],
        ),
        dict(
            name="two_ancestors_partial_overlap",
            start=[1, 0],
            end=[3, 2],
            time=[0.5, 0.5],
            new_start=[0],
            new_end=[3],
            new_time=[0.5],
            old_indexes={0: [0, 1]},
            sort_indices=[1, 0],
        ),
        dict(
            name="ancestors_inside_other",
            start=[0, 1, 3],
            end=[5, 2, 4],
            time=[0, 0, 0],
            new_start=[0],
            new_end=[5],
            new_time=[0],
            old_indexes={0: [0, 1, 2]},
            sort_indices=[0, 1, 2],
        ),
        dict(
            name="overlap_ancestors_inside_other",
            start=[0, 1, 2],
            end=[5, 3, 4],
            time=[0, 0, 0],
            new_start=[0],
            new_end=[5],
            new_time=[0],
            old_indexes={0: [0, 1, 2]},
            sort_indices=[0, 1, 2],
        ),
        dict(
            name="abutting_ancestors_inside_other",
            start=[1, 3, 0],
            end=[3, 4, 5],
            time=[0, 0, 0],
            new_start=[0],
            new_end=[5],
            new_time=[0],
            old_indexes={0: [0, 1, 2]},
            sort_indices=[2, 0, 1],
        ),
        dict(
            name="abutting_ancestors",
            start=[0, 1, 2],
            end=[1, 2, 3],
            time=[0, 0, 0],
            new_start=[0, 1, 2],
            new_end=[1, 2, 3],
            new_time=[0, 0, 0],
            old_indexes={0: [0], 1: [1], 2: [2]},
            sort_indices=[0, 1, 2],
        ),
        dict(
            name="multiple_overlap",
            start=[2, 3, 5, 1, 2, 1, 1, 2],
            end=[4, 6, 6, 3, 3, 2, 3, 3],
            time=[4, 4, 4, 1, 1, 2, 3, 3],
            new_start=[1, 1, 1, 2],
            new_end=[3, 2, 3, 6],
            new_time=[1, 2, 3, 4],
            old_indexes={0: [0, 1], 1: [2], 2: [3, 4], 3: [5, 6, 7]},
            sort_indices=[3, 4, 5, 6, 7, 0, 1, 2],
        ),
    ]

    @pytest.mark.parametrize(
        "case",
        merging_fixed_test_cases,
        ids=[case["name"] for case in merging_fixed_test_cases],
    )
    def test_merging_fixed_cases(self, case):
        (
            new_start,
            new_end,
            new_time,
            old_indexes,
            sort_indices,
        ) = merge_overlapping_ancestors(
            np.array(case["start"]),
            np.array(case["end"]),
            np.array(case["time"], dtype=np.float32),
        )
        assert list(new_start) == case["new_start"]
        assert list(new_end) == case["new_end"]
        assert list(new_time) == case["new_time"]
        assert old_indexes == case["old_indexes"]
        assert list(sort_indices) == case["sort_indices"]

    grouping_fixed_test_cases = [
        dict(name="empty_input", start=[], end=[], time=[], expected_output={}),
        dict(
            name="singleton_input",
            start=[1],
            end=[5],
            time=[1],
            expected_output={0: [0]},
        ),
        dict(
            name="overlap_same_time",
            start=[1, 2, 2],
            end=[4, 5, 3],
            time=[1, 1, 1],
            expected_output={0: [0, 1, 2]},
        ),
        dict(
            name="overlap_different_time",
            start=[2, 1],
            end=[3, 4],
            time=[2, 1],
            expected_output={0: [0], 1: [1]},
        ),
        dict(
            name="same_start_end_diff_time",
            start=[1, 1],
            end=[5, 5],
            time=[1, 2],
            expected_output={0: [1], 1: [0]},
        ),
        dict(
            name="identical",
            start=[1, 1],
            end=[5, 5],
            time=[1, 1],
            expected_output={0: [0, 1]},
        ),
        dict(
            name="non_overlap",
            start=[1, 5],
            end=[3, 7],
            time=[1, 2],
            expected_output={0: [0, 1]},
        ),
        dict(
            name="all_overlap",
            start=[1, 2, 3],
            end=[4, 5, 6],
            time=[1, 2, 3],
            expected_output={0: [2], 1: [1], 2: [0]},
        ),
        dict(
            name="start_equals_end",
            start=[1, 3],
            end=[3, 5],
            time=[1, 2],
            expected_output={0: [0, 1]},
        ),
        dict(
            name="larger_example",
            start=[1, 3, 2, 5, 4, 1],
            end=[4, 6, 5, 7, 6, 3],
            time=[0.5, 0.7, 0.6, 0.4, 0.3, 0.6],
            expected_output={0: [1], 1: [2, 3, 5], 2: [0, 4]},
        ),
        dict(
            name="overlap_where_pushing_down_breaks_dependency",
            start=[1, 17, 34, 87],
            end=[69, 18, 126, 125],
            time=[16, 19, 16, 15],
            expected_output={0: [1], 1: [0, 2], 2: [3]},
        ),
    ]

    @pytest.mark.parametrize(
        "case",
        grouping_fixed_test_cases,
        ids=[case["name"] for case in grouping_fixed_test_cases],
    )
    def test_grouping_fixed_cases(self, case):
        output = group_ancestors_by_linesweep(
            np.array(case["start"]), np.array(case["end"]), np.array(case["time"])
        )
        for group in output:
            assert list(output[group]) == case["expected_output"][group]

    @pytest.mark.parametrize("seed", range(500))
    def test_grouping_random_cases(self, seed):
        rng = np.random.RandomState(seed)
        n = 100
        start = rng.randint(0, 200, size=n)
        end = start + rng.randint(1, 100, size=n)
        time = rng.randint(0, 40, size=n)
        output = group_ancestors_by_linesweep(start, end, time)

        group_ids = np.full(n, -1, dtype=np.int32)
        for group_id, group in output.items():
            group_ids[group] = group_id
        assert np.sum(group_ids == -1) == 0
        for anc_a, anc_b in itertools.combinations(range(n), 2):
            if start[anc_a] < end[anc_b] and start[anc_b] < end[anc_a]:
                if time[anc_a] == time[anc_b]:
                    assert group_ids[anc_a] == group_ids[anc_b]
                elif time[anc_a] < time[anc_b]:
                    assert group_ids[anc_a] > group_ids[anc_b]
                else:
                    assert group_ids[anc_a] < group_ids[anc_b]

    def test_find_groups_cycle(self):
        children_data = np.array([1, 0], dtype=np.int32)
        children_indices = np.array([0, 1, 2], dtype=np.int32)
        incoming_edge_count = np.array([1, 1], dtype=np.int32)

        with pytest.raises(
            ValueError, match="Erroneous cycle in ancestor dependancies.*"
        ):
            find_groups(children_data, children_indices, incoming_edge_count)


# ---------------------------------------------------------------------------
# Helpers for metadata / groups tests
# ---------------------------------------------------------------------------


def _make_cfg(sample_store, ancestor_store):
    """Build a Config suitable for collect_haplotype_metadata."""
    src = Source(path=sample_store, name="test")
    return Config(
        sources={"test": src},
        ancestors=AncestorsConfig(path=ancestor_store, sources=["test"]),
        match=MatchConfig(
            sources=["test"],
            output="output.trees",
            recombination_rate=1e-4,
        ),
    )


def _build_stores():
    """Create a sample VCZ, infer ancestors, return (sample, ancestor)."""
    sample_store = make_sample_vcz(
        genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
        positions=np.array([100, 200], dtype=np.int32),
        alleles=np.array([["A", "T"], ["A", "T"]]),
        ancestral_state=np.array(["A", "A"]),
        sequence_length=1000,
    )
    anc_cfg = AncestorsConfig(path=None, sources=["test"])
    ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
    return sample_store, ancestor_store


# ---------------------------------------------------------------------------
# TestCollectHaplotypeMetadata
# ---------------------------------------------------------------------------


class TestCollectHaplotypeMetadata:
    def test_returns_haplotype_metadata(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        meta = collect_haplotype_metadata(cfg)
        assert isinstance(meta, HaplotypeMetadata)
        n = len(meta.times)
        assert meta.times.ndim == 1
        assert meta.is_ancestor.ndim == 1
        assert meta.start_positions.ndim == 1
        assert meta.end_positions.ndim == 1
        assert len(meta.is_ancestor) == n
        assert len(meta.source) == n
        assert len(meta.sample_id) == n
        assert len(meta.ploidy_index) == n

    def test_ancestors_are_first(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        meta = collect_haplotype_metadata(cfg)
        assert meta.is_ancestor[0] is True or meta.is_ancestor[0] == True  # noqa: E712
        assert meta.source[0] == "ancestors"
        # No virtual root — first entry is a real ancestor
        assert meta.sample_id[0] != "virtual_root"

    def test_correct_counts(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        meta = collect_haplotype_metadata(cfg)
        num_anc = int(np.sum(meta.is_ancestor))
        num_samp = len(meta.times) - num_anc
        assert num_anc >= 1  # at least one real ancestor
        assert num_samp == 2  # 2 haploid samples

    def test_sample_identity(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        meta = collect_haplotype_metadata(cfg)
        # Sample haplotypes should have source="test"
        sample_indices = [i for i in range(len(meta.times)) if not meta.is_ancestor[i]]
        for i in sample_indices:
            assert meta.source[i] == "test"
            assert meta.sample_id[i].startswith("sample_")
            assert meta.ploidy_index[i] == 0  # haploid

    def test_no_genotype_access(self):
        """Metadata loading works even when call_genotype data is unused."""
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_cfg(sample_store, ancestor_store)
        meta = collect_haplotype_metadata(cfg)
        assert len(meta.times) > 0


# ---------------------------------------------------------------------------
# TestComputeGroupsFromGrouping
# ---------------------------------------------------------------------------


class TestComputeGroupsFromGrouping:
    """Test that compute_groups imported from grouping works correctly."""

    def test_import_from_grouping(self):
        from tsinfer.grouping import compute_groups as cg

        assert callable(cg)

    def test_basic_grouping(self):
        groups = compute_groups(
            times=np.array([1.0, 0.5, 0.0], dtype=np.float64),
            is_ancestor=np.array([True, True, False]),
            start_positions=np.array([0, 0, 0], dtype=np.int32),
            end_positions=np.array([100, 100, 100], dtype=np.int32),
        )
        assert len(groups) == 3
        assert list(groups[0]) == [0]  # ancestor at t=1.0
        assert list(groups[1]) == [1]  # ancestor at t=0.5
        assert list(groups[2]) == [2]  # sample at t=0.0


# ---------------------------------------------------------------------------
# TestComputeGroupsJson
# ---------------------------------------------------------------------------


class TestComputeMatchJobs:
    def test_returns_match_jobs(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        jobs = compute_match_jobs(cfg)
        assert len(jobs) > 0
        assert all(isinstance(j, MatchJob) for j in jobs)

    def test_first_job_is_ancestor(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        jobs = compute_match_jobs(cfg)
        assert jobs[0].haplotype_index == 0
        assert jobs[0].source == "ancestors"
        assert jobs[0].sample_id != "virtual_root"
        assert jobs[0].group == 0

    def test_all_indices_covered(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        jobs = compute_match_jobs(cfg)
        all_indices = {j.haplotype_index for j in jobs}
        assert all_indices == set(range(len(jobs)))

    def test_jobs_have_all_fields(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        jobs = compute_match_jobs(cfg)
        for j in jobs:
            assert isinstance(j.haplotype_index, int)
            assert isinstance(j.source, str)
            assert isinstance(j.sample_id, str)
            assert isinstance(j.ploidy_index, int)
            assert isinstance(j.time, float)
            assert isinstance(j.start_position, int)
            assert isinstance(j.end_position, int)
            assert isinstance(j.group, int)

    def test_ordered_by_group(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        jobs = compute_match_jobs(cfg)
        groups = [j.group for j in jobs]
        assert groups == sorted(groups)


class TestComputeGroupsJsonFromGrouping:
    def test_returns_valid_json_array(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
        assert isinstance(records, list)
        assert len(records) > 0

    def test_records_have_match_job_fields(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
        expected_keys = {
            "haplotype_index",
            "source",
            "sample_id",
            "ploidy_index",
            "time",
            "start_position",
            "end_position",
            "group",
        }
        for rec in records:
            assert set(rec.keys()) == expected_keys

    def test_all_indices_covered(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
        all_indices = {r["haplotype_index"] for r in records}
        assert all_indices == set(range(len(records)))
