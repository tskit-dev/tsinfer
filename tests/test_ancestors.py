#
# Copyright (C) 2023 University of Oxford
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
Tests for the ancestor handling code.
"""
import itertools

import numpy as np
import pytest

from tsinfer import ancestors


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
        ) = ancestors.merge_overlapping_ancestors(
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
        output = ancestors.group_ancestors_by_linesweep(
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
        output = ancestors.group_ancestors_by_linesweep(start, end, time)

        # Convert to an array of group_ids
        group_ids = np.full(n, -1, dtype=np.int32)
        for group_id, group in output.items():
            group_ids[group] = group_id
        # Check that all ancestors are present
        assert np.sum(group_ids == -1) == 0
        for anc_a, anc_b in itertools.combinations(range(n), 2):
            # Check all the constraints are satisfied for an overlapping pair
            if start[anc_a] < end[anc_b] and start[anc_b] < end[anc_a]:
                if time[anc_a] == time[anc_b]:
                    assert group_ids[anc_a] == group_ids[anc_b]
                elif time[anc_a] < time[anc_b]:
                    assert group_ids[anc_a] > group_ids[anc_b]
                else:
                    assert group_ids[anc_a] < group_ids[anc_b]
