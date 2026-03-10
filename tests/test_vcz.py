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
Tests for tsinfer.vcz: open_store, resolve_field, and convenience accessors.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from helpers import make_sample_vcz

from tsinfer.vcz import num_contigs, open_store, resolve_field, sequence_length

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_store():
    """A small in-memory sample VCZ with 3 sites and 2 diploid samples."""
    gt = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [1, 1]]], dtype=np.int8)
    return make_sample_vcz(
        gt,
        positions=[100, 300, 500],
        alleles=[["A", "T"], ["C", "G"], ["A", "C"]],
        ancestral_state=["A", "C", "A"],
        sequence_length=1000,
        sample_ids=np.array(["s0", "s1"]),
        site_mask=np.array([False, True, False]),
        sample_time=np.array([0.0, 1.5]),
    )


def _make_annotation_vcz(positions, field_name, values):
    """
    Build a minimal annotation VCZ containing variant_position and one field.
    Uses make_sample_vcz with dummy genotypes so the _ARRAY_DIMENSIONS attrs
    are set correctly, then adds the extra field manually.
    """
    n = len(positions)
    gt = np.zeros((n, 1, 1), dtype=np.int8)
    store = make_sample_vcz(
        gt,
        positions=positions,
        alleles=[["A", "T"]] * n,
        ancestral_state=["A"] * n,
        sequence_length=max(positions) + 100,
        **{field_name: np.asarray(values)},
    )
    return store


# ---------------------------------------------------------------------------
# open_store
# ---------------------------------------------------------------------------


class TestOpenStore:
    def test_passthrough_zarr_group(self, sample_store):
        result = open_store(sample_store)
        assert result is sample_store

    def test_opens_path_on_disk(self, tmp_path):
        # Write and re-open a zarr v2 store from disk
        store_path = tmp_path / "test.vcz"
        root = zarr.open_group(str(store_path), mode="w", zarr_format=2)
        root.create_array("x", data=np.array([1, 2, 3]))
        opened = open_store(store_path)
        assert isinstance(opened, zarr.Group)
        np.testing.assert_array_equal(opened["x"][:], [1, 2, 3])

    def test_opens_string_path(self, tmp_path):
        store_path = tmp_path / "test.vcz"
        root = zarr.open_group(str(store_path), mode="w", zarr_format=2)
        root.create_array("y", data=np.array([7]))
        opened = open_store(str(store_path))
        assert isinstance(opened, zarr.Group)


# ---------------------------------------------------------------------------
# resolve_field — None
# ---------------------------------------------------------------------------


class TestResolveFieldNone:
    def test_none_returns_none(self, sample_store):
        assert resolve_field(sample_store, None, "variant_position", 3) is None


# ---------------------------------------------------------------------------
# resolve_field — string (field in store)
# ---------------------------------------------------------------------------


class TestResolveFieldString:
    def test_site_mask_from_store(self, sample_store):
        result = resolve_field(sample_store, "site_mask", "variant_position", 3)
        np.testing.assert_array_equal(result, [False, True, False])

    def test_sample_time_from_store(self, sample_store):
        result = resolve_field(sample_store, "sample_time", "sample_id", 2)
        np.testing.assert_array_almost_equal(result, [0.0, 1.5])

    def test_missing_field_raises(self, sample_store):
        with pytest.raises((KeyError, Exception)):
            resolve_field(sample_store, "no_such_field", "variant_position", 3)


# ---------------------------------------------------------------------------
# resolve_field — scalar
# ---------------------------------------------------------------------------


class TestResolveFieldScalar:
    def test_integer_scalar(self, sample_store):
        result = resolve_field(sample_store, 1, "variant_position", 3)
        np.testing.assert_array_equal(result, [1, 1, 1])
        assert len(result) == 3

    def test_float_scalar(self, sample_store):
        result = resolve_field(sample_store, 2.5, "sample_id", 2)
        np.testing.assert_array_almost_equal(result, [2.5, 2.5])
        assert len(result) == 2

    def test_zero_scalar(self, sample_store):
        result = resolve_field(sample_store, 0, "variant_position", 3)
        np.testing.assert_array_equal(result, [0, 0, 0])


# ---------------------------------------------------------------------------
# resolve_field — dict join on variant_position
# ---------------------------------------------------------------------------


class TestResolveFieldDictPosition:
    def test_all_positions_match(self, sample_store):
        # Annotation covers all 3 positions in sample_store
        ann = _make_annotation_vcz(
            positions=[100, 300, 500],
            field_name="filter_flag",
            values=[False, True, False],
        )
        spec = {"path": ann, "field": "filter_flag"}
        result = resolve_field(sample_store, spec, "variant_position", 3)
        np.testing.assert_array_equal(result, [False, True, False])

    def test_partial_match_fills_missing(self, sample_store):
        # Annotation only covers positions 100 and 500; 300 is missing
        ann = _make_annotation_vcz(
            positions=[100, 500],
            field_name="filter_flag",
            values=[True, True],
        )
        spec = {"path": ann, "field": "filter_flag"}
        result = resolve_field(
            sample_store, spec, "variant_position", 3, fill_value=False
        )
        np.testing.assert_array_equal(result, [True, False, True])

    def test_no_matches_all_fill(self, sample_store):
        # Annotation has completely different positions
        ann = _make_annotation_vcz(
            positions=[999],
            field_name="flag",
            values=[True],
        )
        spec = {"path": ann, "field": "flag"}
        result = resolve_field(
            sample_store, spec, "variant_position", 3, fill_value=False
        )
        np.testing.assert_array_equal(result, [False, False, False])

    def test_float_values(self, sample_store):
        ann = _make_annotation_vcz(
            positions=[100, 300, 500],
            field_name="score",
            values=[1.0, 2.0, 3.0],
        )
        spec = {"path": ann, "field": "score"}
        result = resolve_field(sample_store, spec, "variant_position", 3, fill_value=0.0)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_fill_value_default_is_none(self, sample_store):
        ann = _make_annotation_vcz(
            positions=[100],
            field_name="flag",
            values=[True],
        )
        spec = {"path": ann, "field": "flag"}
        result = resolve_field(sample_store, spec, "variant_position", 3)
        # positions 300 and 500 have no match → fill with None
        assert result[1] is None
        assert result[2] is None


# ---------------------------------------------------------------------------
# resolve_field — dict join on sample_id
# ---------------------------------------------------------------------------


class TestResolveFieldDictSampleId:
    def test_all_samples_match(self, sample_store):
        gt = np.zeros((1, 2, 1), dtype=np.int8)
        ann = make_sample_vcz(
            gt,
            positions=[1],
            alleles=[["A", "T"]],
            ancestral_state=["A"],
            sequence_length=100,
            sample_ids=np.array(["s0", "s1"]),
            age=np.array([10.0, 20.0]),
        )
        spec = {"path": ann, "field": "age"}
        result = resolve_field(sample_store, spec, "sample_id", 2, fill_value=0.0)
        np.testing.assert_array_almost_equal(result, [10.0, 20.0])

    def test_partial_match_fills_missing(self, sample_store):
        # Annotation only has s0
        gt = np.zeros((1, 1, 1), dtype=np.int8)
        ann = make_sample_vcz(
            gt,
            positions=[1],
            alleles=[["A", "T"]],
            ancestral_state=["A"],
            sequence_length=100,
            sample_ids=np.array(["s0"]),
            age=np.array([5.0]),
        )
        spec = {"path": ann, "field": "age"}
        result = resolve_field(sample_store, spec, "sample_id", 2, fill_value=0.0)
        np.testing.assert_array_almost_equal(result, [5.0, 0.0])


# ---------------------------------------------------------------------------
# resolve_field — bad input
# ---------------------------------------------------------------------------


class TestResolveFieldErrors:
    def test_invalid_type_raises(self, sample_store):
        with pytest.raises(TypeError):
            resolve_field(sample_store, [1, 2, 3], "variant_position", 3)

    def test_dict_missing_path_raises(self, sample_store):
        with pytest.raises(ValueError, match="path"):
            resolve_field(sample_store, {"field": "x"}, "variant_position", 3)

    def test_dict_missing_field_raises(self, sample_store):
        with pytest.raises(ValueError, match="field"):
            resolve_field(sample_store, {"path": "ann.vcz"}, "variant_position", 3)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


class TestConvenienceAccessors:
    def test_sequence_length(self, sample_store):
        assert sequence_length(sample_store) == 1000

    def test_num_contigs(self, sample_store):
        assert num_contigs(sample_store) == 1

    def test_sequence_length_custom(self):
        gt = np.zeros((1, 1, 1), dtype=np.int8)
        store = make_sample_vcz(gt, [10], [["A", "T"]], ["A"], sequence_length=999_999)
        assert sequence_length(store) == 999_999
