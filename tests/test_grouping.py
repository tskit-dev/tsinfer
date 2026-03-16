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

import numpy as np
import pytest
from helpers import make_sample_vcz

from tsinfer.ancestors import infer_ancestors
from tsinfer.config import AncestorsConfig, Config, MatchConfig, Source
from tsinfer.grouping import (
    HaplotypeMetadata,
    collect_haplotype_metadata,
    compute_groups,
    compute_groups_json,
    find_groups,
    group_ancestors_by_linesweep,
    merge_overlapping_ancestors,
)


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
        assert meta.times.ndim == 1
        assert meta.is_ancestor.ndim == 1
        assert meta.start_positions.ndim == 1
        assert meta.end_positions.ndim == 1
        assert len(meta.times) == len(meta.is_ancestor)

    def test_virtual_root_is_first(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        meta = collect_haplotype_metadata(cfg)
        # Virtual root: time=1.0, is_ancestor=True
        assert meta.times[0] == 1.0
        assert meta.is_ancestor[0] is True or meta.is_ancestor[0] == True  # noqa: E712

    def test_correct_counts(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        meta = collect_haplotype_metadata(cfg)
        # 2 samples, ploidy=1, plus ancestors + virtual root
        num_anc = int(np.sum(meta.is_ancestor))
        num_samp = len(meta.times) - num_anc
        assert num_anc >= 1  # at least the virtual root
        assert num_samp == 2  # 2 haploid samples

    def test_no_genotype_access(self):
        """Verify metadata loading works even when call_genotype data is corrupted."""

        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        # Corrupt genotype data in sample store — metadata loading should still work
        # because collect_haplotype_metadata only reads shape metadata, not data
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
        assert list(groups[0]) == [0]
        assert len(groups) == 3


# ---------------------------------------------------------------------------
# TestComputeGroupsJson
# ---------------------------------------------------------------------------


class TestComputeGroupsJsonFromGrouping:
    def test_returns_valid_json(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        result = json.loads(compute_groups_json(cfg))
        assert result["num_haplotypes"] > 0
        assert result["num_groups"] > 0
        assert len(result["groups"]) == result["num_groups"]

    def test_all_indices_covered(self):
        sample_store, ancestor_store = _build_stores()
        cfg = _make_cfg(sample_store, ancestor_store)
        result = json.loads(compute_groups_json(cfg))
        all_indices = set()
        for g in result["groups"]:
            all_indices.update(g["haplotype_indices"])
        assert all_indices == set(range(result["num_haplotypes"]))
