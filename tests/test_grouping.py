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
find_groups, compute_groups, assign_groups.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pytest

from tsinfer.grouping import (
    MatchJob,
    assign_groups,
    compute_groups,
    find_groups,
    group_haplotypes_by_linesweep,
    merge_overlapping_haplotypes,
)


@dataclass
class _GroupInputs:
    times: np.ndarray
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

    def _make_inputs(self, times, starts=None, ends=None):
        times = np.asarray(times, dtype=np.float64)
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
            start_positions=starts,
            end_positions=ends,
        )

    # -------------------------------------------------------------------
    # Basic ordering (no interval effects)
    # -------------------------------------------------------------------

    def test_single_haplotype(self):
        gi = self._make_inputs([0.5])
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert list(groups[0]) == [0]

    def test_multiple_time_levels_ordering(self):
        # Haplotypes at 0.8, 0.5, 0.3 — should be in descending order
        gi = self._make_inputs([0.8, 0.5, 0.3])
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 3
        assert list(groups[0]) == [0]  # time 0.8
        assert list(groups[1]) == [1]  # time 0.5
        assert list(groups[2]) == [2]  # time 0.3

    def test_samples_only_modern(self):
        # 3 modern samples (time=0) — all overlap → same group
        gi = self._make_inputs([0.0, 0.0, 0.0])
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_ancient_before_modern(self):
        # Ancient (time=0.5) + modern (time=0)
        gi = self._make_inputs([0.5, 0.0])
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 2
        assert list(groups[0]) == [0]  # ancient first
        assert list(groups[1]) == [1]  # modern last

    def test_same_time_overlapping_haplotypes_same_group(self):
        # Two haplotypes at t=0.5 with overlapping intervals → SAME group
        gi = self._make_inputs([0.5, 0.5])
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_mixed_ordering(self):
        # Haplotypes at 0.9, 0.4, 0.4, 0.0 — all default intervals [0,100]
        gi = self._make_inputs([0.9, 0.4, 0.4, 0.0])
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        # t=0.9: group alone; t=0.4: two overlap → same group; t=0.0: alone
        assert len(groups) == 3
        assert list(groups[0]) == [0]  # t=0.9
        assert set(groups[1]) == {1, 2}  # t=0.4, overlapping → same group
        assert list(groups[2]) == [3]  # t=0.0

    def test_returns_int32_arrays(self):
        gi = self._make_inputs([0.5])
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        for g in groups:
            assert g.dtype == np.int32

    def test_empty_returns_empty(self):
        gi = self._make_inputs([])
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 0

    # -------------------------------------------------------------------
    # Interval-based grouping for same-time ancestors
    # -------------------------------------------------------------------

    def test_two_same_time_overlapping(self):
        """Two at time 0.5, full range -> overlap -> SAME group."""
        gi = self._make_inputs(
            [0.5, 0.5],
            starts=[0, 0],
            ends=[100, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_two_same_time_disjoint(self):
        """Two at time 0.5 with disjoint intervals -> one group."""
        gi = self._make_inputs(
            [0.5, 0.5],
            starts=[0, 60],
            ends=[30, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_two_same_time_adjacent(self):
        """Two at time 0.5, touching boundary -> one group."""
        gi = self._make_inputs(
            [0.5, 0.5],
            starts=[0, 50],
            ends=[50, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_two_same_time_one_site_overlap(self):
        """Two at time 0.5, overlapping by one unit -> SAME group."""
        gi = self._make_inputs(
            [0.5, 0.5],
            starts=[0, 49],
            ends=[50, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_three_same_time_all_disjoint(self):
        """Three at time 0.5, all disjoint -> one group."""
        gi = self._make_inputs(
            [0.5, 0.5, 0.5],
            starts=[0, 30, 60],
            ends=[20, 50, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_three_same_time_chain_overlap(self):
        """Three at time 0.5 with chain overlap -> SAME group."""
        gi = self._make_inputs(
            [0.5, 0.5, 0.5],
            starts=[0, 25, 50],
            ends=[35, 55, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_three_same_time_all_overlapping(self):
        """Three at time 0.5, all pairwise overlapping -> SAME group."""
        gi = self._make_inputs(
            [0.5, 0.5, 0.5],
            starts=[0, 10, 20],
            ends=[80, 90, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_same_time_at_multiple_time_levels(self):
        """Two time levels (0.8 and 0.5), each with two disjoint haplotypes."""
        gi = self._make_inputs(
            [0.8, 0.8, 0.5, 0.5],
            starts=[0, 60, 0, 60],
            ends=[30, 100, 30, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 2  # t=0.8 group + t=0.5 group
        assert set(groups[0]) == {0, 1}  # t=0.8 disjoint
        assert set(groups[1]) == {2, 3}  # t=0.5 disjoint

    def test_same_time_overlapping_full_range(self):
        """Same-time overlapping haplotypes are in the same group."""
        gi = self._make_inputs(
            [0.0, 0.0],
            starts=[0, 0],
            ends=[100, 100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_single_no_splitting_needed(self):
        """A single haplotype needs no splitting."""
        gi = self._make_inputs(
            [0.5],
            starts=[0],
            ends=[100],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert list(groups[0]) == [0]

    def test_nested_intervals_overlap(self):
        """Nested intervals -> SAME group."""
        gi = self._make_inputs(
            [0.5, 0.5],
            starts=[0, 20],
            ends=[100, 60],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_empty_interval(self):
        """Zero-length interval can't overlap -> can share a group."""
        gi = self._make_inputs(
            [0.5, 0.5],
            starts=[0, 50],
            ends=[100, 50],
        )
        groups = compute_groups(gi.times, gi.start_positions, gi.end_positions)
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
        ) = merge_overlapping_haplotypes(
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
        output = group_haplotypes_by_linesweep(
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
        output = group_haplotypes_by_linesweep(start, end, time)

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
            start_positions=np.array([0, 0, 0], dtype=np.int32),
            end_positions=np.array([100, 100, 100], dtype=np.int32),
        )
        assert len(groups) == 3
        assert list(groups[0]) == [0]  # t=1.0
        assert list(groups[1]) == [1]  # t=0.5
        assert list(groups[2]) == [2]  # t=0.0


# ---------------------------------------------------------------------------
# TestAssignGroups
# ---------------------------------------------------------------------------


class TestAssignGroups:
    """Test that assign_groups takes ungrouped MatchJobs and assigns groups."""

    def _make_job(self, haplotype_index, time, start=0, end=100):
        return MatchJob(
            haplotype_index=haplotype_index,
            source="test",
            sample_id=f"s{haplotype_index}",
            ploidy_index=0,
            time=time,
            start_position=start,
            end_position=end,
            group=0,
        )

    def test_empty(self):
        result = assign_groups([])
        assert result == []

    def test_single_job(self):
        jobs = [self._make_job(0, 0.5)]
        result = assign_groups(jobs)
        assert len(result) == 1
        assert result[0].group == 0

    def test_different_times_get_different_groups(self):
        jobs = [
            self._make_job(0, 0.8),
            self._make_job(1, 0.5),
            self._make_job(2, 0.3),
        ]
        result = assign_groups(jobs)
        assert len(result) == 3
        groups = [j.group for j in result]
        assert len(set(groups)) == 3
        # Oldest first
        assert result[0].time == 0.8
        assert result[1].time == 0.5
        assert result[2].time == 0.3

    def test_same_time_overlapping_same_group(self):
        jobs = [
            self._make_job(0, 0.5, start=0, end=100),
            self._make_job(1, 0.5, start=0, end=100),
        ]
        result = assign_groups(jobs)
        assert result[0].group == result[1].group

    def test_sorted_by_group_then_haplotype_index(self):
        jobs = [
            self._make_job(2, 0.3),
            self._make_job(0, 0.8),
            self._make_job(1, 0.5),
        ]
        result = assign_groups(jobs)
        # Should be sorted by (group, haplotype_index)
        for i in range(len(result) - 1):
            assert (result[i].group, result[i].haplotype_index) <= (
                result[i + 1].group,
                result[i + 1].haplotype_index,
            )

    def test_preserves_job_fields(self):
        job = MatchJob(
            haplotype_index=0,
            source="my_source",
            sample_id="my_sample",
            ploidy_index=1,
            time=0.5,
            start_position=10,
            end_position=90,
            group=0,
            node_flags=3,
            individual_id=42,
            population_id=7,
        )
        result = assign_groups([job])
        assert result[0].source == "my_source"
        assert result[0].sample_id == "my_sample"
        assert result[0].ploidy_index == 1
        assert result[0].node_flags == 3
        assert result[0].individual_id == 42
        assert result[0].population_id == 7
