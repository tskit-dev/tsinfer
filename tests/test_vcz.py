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

import threading
import time

import numpy as np
import pytest
import zarr
from helpers import make_sample_vcz

from tsinfer.grouping import MatchJob
from tsinfer.vcz import (
    AlleleMapper,
    ChunkCache,
    HaplotypeReader,
    VCZHaplotypeReader,
    num_contigs,
    open_store,
    resolve_field,
    sequence_length,
)

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


# ---------------------------------------------------------------------------
# HaplotypeReader
# ---------------------------------------------------------------------------


def _make_ancestor_and_sample_stores():
    """Build minimal ancestor and sample VCZ stores for HaplotypeReader tests."""
    from helpers import make_ancestor_vcz

    positions = np.array([100, 200, 300], dtype=np.int32)
    # 2 ancestors: a0 = [0, 1, 0], a1 = [1, 0, 1]
    anc_gt = np.array([[[0], [1]], [[1], [0]], [[0], [1]]], dtype=np.int8)
    anc_store = make_ancestor_vcz(
        genotypes=anc_gt,
        positions=positions,
        alleles=np.array([["A", "T"], ["C", "G"], ["A", "C"]]),
        times=np.array([0.8, 0.6]),
        focal_positions=np.array([[100, -2], [300, -2]], dtype=np.int32),
        sequence_intervals=np.array([[100, 300]], dtype=np.int32),
    )

    # Sample store: 2 samples, haploid, at the same 3 positions
    # sample_0: A, G, C → gt [0, 1, 1]
    # sample_1: T, C, A → gt [1, 0, 0]
    sample_gt = np.array([[[0], [1]], [[1], [0]], [[1], [0]]], dtype=np.int8)
    from tsinfer.config import Source

    sample_store = make_sample_vcz(
        genotypes=sample_gt,
        positions=positions,
        alleles=np.array([["A", "T"], ["C", "G"], ["A", "C"]]),
        ancestral_state=np.array(["A", "C", "A"]),
        sequence_length=1000,
    )
    source = Source(path=sample_store, name="test")
    anc_source = Source(path=anc_store, name="ancestors", sample_time="sample_time")
    ancestral_alleles = np.asarray(anc_store["variant_allele"][:])[:, 0]
    return anc_store, sample_store, source, anc_source, positions, ancestral_alleles


class TestHaplotypeReader:
    def test_ancestor_haplotype(self):
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source}, positions, anc_alleles
        )
        # a0 has genotype [0, 1, 0]
        job = MatchJob(
            haplotype_index=1,
            source="ancestors",
            sample_id="a0",
            ploidy_index=0,
            time=0.8,
            start_position=100,
            end_position=300,
            group=1,
        )
        hap = reader.read_haplotype(job)
        np.testing.assert_array_equal(hap, [0, 1, 0])

    def test_ancestor_haplotype_second(self):
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source}, positions, anc_alleles
        )
        # a1 has genotype [1, 0, 1]
        job = MatchJob(
            haplotype_index=2,
            source="ancestors",
            sample_id="a1",
            ploidy_index=0,
            time=0.6,
            start_position=100,
            end_position=300,
            group=2,
        )
        hap = reader.read_haplotype(job)
        np.testing.assert_array_equal(hap, [1, 0, 1])

    def test_sample_haplotype_encoding(self):
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source}, positions, anc_alleles
        )
        # sample_0: gt [0, 1, 1] → encoded [0=anc, 1=derived, 1=derived]
        job = MatchJob(
            haplotype_index=3,
            source="test",
            sample_id="sample_0",
            ploidy_index=0,
            time=0.0,
            start_position=100,
            end_position=300,
            group=3,
        )
        hap = reader.read_haplotype(job)
        np.testing.assert_array_equal(hap, [0, 1, 1])

    def test_sample_haplotype_second(self):
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source}, positions, anc_alleles
        )
        # sample_1: gt [1, 0, 0] → encoded [1=derived, 0=anc, 0=anc]
        job = MatchJob(
            haplotype_index=4,
            source="test",
            sample_id="sample_1",
            ploidy_index=0,
            time=0.0,
            start_position=100,
            end_position=300,
            group=3,
        )
        hap = reader.read_haplotype(job)
        np.testing.assert_array_equal(hap, [1, 0, 0])

    def test_missing_genotype_encoded_as_minus_one(self):
        """Missing genotypes (-1 in call_genotype) should encode as -1."""
        from helpers import make_ancestor_vcz

        from tsinfer.config import Source

        positions = np.array([100, 200], dtype=np.int32)
        anc_store = make_ancestor_vcz(
            genotypes=np.array([[[0]], [[0]]], dtype=np.int8),
            positions=positions,
            alleles=np.array([["A", "T"], ["A", "T"]]),
            times=np.array([0.8]),
            focal_positions=np.array([[100]], dtype=np.int32),
            sequence_intervals=np.array([[100, 200]], dtype=np.int32),
        )
        # Sample with missing data at site 0
        sample_gt = np.array([[[-1]], [[0]]], dtype=np.int8)
        sample_store = make_sample_vcz(
            genotypes=sample_gt,
            positions=positions,
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        source = Source(path=sample_store, name="test")
        anc_source = Source(path=anc_store, name="ancestors", sample_time="sample_time")
        anc_alleles = np.asarray(anc_store["variant_allele"][:])[:, 0]
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source}, positions, anc_alleles
        )
        job = MatchJob(
            haplotype_index=2,
            source="test",
            sample_id="sample_0",
            ploidy_index=0,
            time=0.0,
            start_position=100,
            end_position=200,
            group=2,
        )
        hap = reader.read_haplotype(job)
        assert hap[0] == -1
        assert hap[1] == 0

    def test_multiple_sources(self):
        """HaplotypeReader handles multiple source names correctly."""
        from helpers import make_ancestor_vcz

        from tsinfer.config import Source

        positions = np.array([100, 200], dtype=np.int32)
        anc_store = make_ancestor_vcz(
            genotypes=np.array([[[0]], [[0]]], dtype=np.int8),
            positions=positions,
            alleles=np.array([["A", "T"], ["A", "T"]]),
            times=np.array([0.8]),
            focal_positions=np.array([[100]], dtype=np.int32),
            sequence_intervals=np.array([[100, 200]], dtype=np.int32),
        )
        # Two different sample stores
        store_a = make_sample_vcz(
            genotypes=np.array([[[0]], [[1]]], dtype=np.int8),
            positions=positions,
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        store_b = make_sample_vcz(
            genotypes=np.array([[[1]], [[0]]], dtype=np.int8),
            positions=positions,
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        sources = {
            "ancestors": Source(
                path=anc_store, name="ancestors", sample_time="sample_time"
            ),
            "src_a": Source(path=store_a, name="src_a"),
            "src_b": Source(path=store_b, name="src_b"),
        }
        anc_alleles = np.asarray(anc_store["variant_allele"][:])[:, 0]
        reader = HaplotypeReader(sources, positions, anc_alleles)

        job_a = MatchJob(
            haplotype_index=2,
            source="src_a",
            sample_id="sample_0",
            ploidy_index=0,
            time=0.0,
            start_position=100,
            end_position=200,
            group=2,
        )
        job_b = MatchJob(
            haplotype_index=3,
            source="src_b",
            sample_id="sample_0",
            ploidy_index=0,
            time=0.0,
            start_position=100,
            end_position=200,
            group=2,
        )
        hap_a = reader.read_haplotype(job_a)
        hap_b = reader.read_haplotype(job_b)
        np.testing.assert_array_equal(hap_a, [0, 1])
        np.testing.assert_array_equal(hap_b, [1, 0])

    def test_cache_size_mb_too_small(self):
        """HaplotypeReader raises ValueError when cache can't fit one chunk."""
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        with pytest.raises(ValueError, match="cannot fit a single chunk"):
            HaplotypeReader(
                {"ancestors": anc_source, "test": source},
                positions,
                anc_alleles,
                cache_size_mb=0,
            )

    def test_cache_size_mb_warning(self):
        """HaplotypeReader warns when fewer than 2 chunks fit."""
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        # Ancestor chunk: 3 sites * N samples * 1 ploidy * 1 byte.
        # With default chunk size covering all samples, chunk is small.
        # cache_size_mb=1 → 1 MiB which fits the tiny test chunks fine.
        # We need cache to fit exactly 1 chunk but not 2.
        # Ancestor store has 2 samples, chunk_size = 2 (all in one chunk)
        # → chunk_bytes = 3 * 2 * 1 * 1 = 6 bytes.
        # Sample store has 2 samples → chunk_bytes = 3 * 2 * 1 * 1 = 6 bytes.
        # We need max_bytes >= 6 (fits 1) but < 12 (fits <2 from each).
        # cache_size_mb must be integer MiB, but 1 MiB is way too large.
        # This test is not practical with MiB granularity and tiny stores.
        # Skip this — the ValueError test above covers the fail-early path.


# ---------------------------------------------------------------------------
# VCZHaplotypeReader
# ---------------------------------------------------------------------------


class TestVCZHaplotypeReader:
    def test_ancestor_store_direct(self):
        """VCZHaplotypeReader reads ancestor haplotypes correctly."""
        anc_store, _, _, _, positions, anc_alleles = _make_ancestor_and_sample_stores()
        cache = ChunkCache(max_bytes=1024 * 1024)
        reader = VCZHaplotypeReader(
            anc_store,
            positions,
            anc_alleles,
            source_name="anc",
            cache=cache,
        )
        hap = reader.read_haplotype("a0", ploidy_index=0)
        np.testing.assert_array_equal(hap, [0, 1, 0])

    def test_sample_store_direct(self):
        """VCZHaplotypeReader reads and polarises sample haplotypes."""
        _, sample_store, _, _, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        cache = ChunkCache(max_bytes=1024 * 1024)
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=cache,
        )
        # sample_0: gt [0, 1, 1] → encoded [0, 1, 1]
        hap = reader.read_haplotype("sample_0", ploidy_index=0)
        np.testing.assert_array_equal(hap, [0, 1, 1])

    def test_cache_hit(self):
        """Second read for a sample in the same chunk uses the cache."""
        anc_store, sample_store, _, _, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        cache = ChunkCache(max_bytes=1024 * 1024)
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=cache,
        )
        reader.read_haplotype("sample_0")
        # Both samples are in the same chunk for this small store
        assert len(cache._cache) == 1
        reader.read_haplotype("sample_1")
        # Should still be just 1 cached chunk
        assert len(cache._cache) == 1

    def test_cache_eviction(self):
        """Oldest chunk is evicted when cache is full."""
        positions = np.array([100], dtype=np.int32)
        # Create a store with 6 samples to span multiple chunks
        n_samples = 6
        gt = np.zeros((1, n_samples, 1), dtype=np.int8)
        sample_ids = np.array([f"sample_{i}" for i in range(n_samples)])
        sample_store = make_sample_vcz(
            genotypes=gt,
            positions=positions,
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
            sample_ids=sample_ids,
        )
        # Replace call_genotype with one that has small sample chunks
        del sample_store["call_genotype"]
        cg = sample_store.create_array(
            "call_genotype",
            data=gt,
            chunks=(1, 2, 1),
        )
        cg.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]
        anc_alleles = np.array(["A"])
        # Each chunk is 1 site * 2 samples * 1 ploidy * 1 byte = 2 bytes.
        # max_bytes=4 allows exactly 2 chunks.
        cache = ChunkCache(max_bytes=4)
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=cache,
        )
        # Read from chunk 0
        reader.read_haplotype("sample_0")
        assert len(cache._cache) == 1
        # Read from chunk 1
        reader.read_haplotype("sample_2")
        assert len(cache._cache) == 2
        # Read from chunk 2 — should evict chunk 0
        reader.read_haplotype("sample_4")
        assert len(cache._cache) == 2
        assert ("test", 0) not in cache._cache

    def test_missing_genotype(self):
        """Missing genotypes encode as -1."""
        positions = np.array([100, 200], dtype=np.int32)
        gt = np.array([[[-1]], [[0]]], dtype=np.int8)
        sample_store = make_sample_vcz(
            genotypes=gt,
            positions=positions,
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        anc_alleles = np.array(["A", "A"])
        cache = ChunkCache(max_bytes=1024 * 1024)
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=cache,
        )
        hap = reader.read_haplotype("sample_0")
        assert hap[0] == -1
        assert hap[1] == 0

    def test_samples_selection(self):
        """VCZHaplotypeReader respects samples_selection."""
        positions = np.array([100], dtype=np.int32)
        gt = np.array([[[0], [1], [0]]], dtype=np.int8)
        sample_ids = np.array(["sample_0", "sample_1", "sample_2"])
        sample_store = make_sample_vcz(
            genotypes=gt,
            positions=positions,
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
            sample_ids=sample_ids,
        )
        # Select only sample_1
        selection = np.array([1])
        anc_alleles = np.array(["A"])
        cache = ChunkCache(max_bytes=1024 * 1024)
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=cache,
            samples_selection=selection,
        )
        hap = reader.read_haplotype("sample_1")
        np.testing.assert_array_equal(hap, [1])

    def test_cross_source_eviction(self):
        """Shared cache evicts across sources when at capacity."""
        positions = np.array([100], dtype=np.int32)
        gt = np.zeros((1, 1, 1), dtype=np.int8)
        alleles = np.array([["A", "T"]])
        anc_alleles = np.array(["A"])

        store_a = make_sample_vcz(
            genotypes=gt,
            positions=positions,
            alleles=alleles,
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )
        store_b = make_sample_vcz(
            genotypes=gt,
            positions=positions,
            alleles=alleles,
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )

        # Each chunk is 1 site * 1 sample * 1 ploidy * 1 byte = 1 byte.
        # max_bytes=2 allows exactly 2 chunks.
        cache = ChunkCache(max_bytes=2)
        reader_a = VCZHaplotypeReader(
            store_a,
            positions,
            anc_alleles,
            source_name="src_a",
            cache=cache,
        )
        reader_b = VCZHaplotypeReader(
            store_b,
            positions,
            anc_alleles,
            source_name="src_b",
            cache=cache,
        )

        reader_a.read_haplotype("sample_0")
        assert len(cache._cache) == 1
        reader_b.read_haplotype("sample_0")
        assert len(cache._cache) == 2

        # Third source should evict src_a's chunk
        store_c = make_sample_vcz(
            genotypes=gt,
            positions=positions,
            alleles=alleles,
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )
        reader_c = VCZHaplotypeReader(
            store_c,
            positions,
            anc_alleles,
            source_name="src_c",
            cache=cache,
        )
        reader_c.read_haplotype("sample_0")
        assert len(cache._cache) == 2
        assert ("src_a", 0) not in cache._cache
        assert ("src_b", 0) in cache._cache
        assert ("src_c", 0) in cache._cache

    def test_large_chunk_evicts_multiple_small(self):
        """A single large chunk can evict multiple smaller entries."""
        positions = np.array([100, 200], dtype=np.int32)

        # Small store: 2 sites, 1 sample, 1 ploidy → 2 bytes per chunk
        gt_small = np.zeros((2, 1, 1), dtype=np.int8)
        alleles = np.array([["A", "T"], ["A", "T"]])
        store_small = make_sample_vcz(
            genotypes=gt_small,
            positions=positions,
            alleles=alleles,
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )

        # Large store: 2 sites, 4 samples, 1 ploidy, chunk_size=4 → 8 bytes
        gt_large = np.zeros((2, 4, 1), dtype=np.int8)
        sample_ids = np.array([f"s{i}" for i in range(4)])
        store_large = make_sample_vcz(
            genotypes=gt_large,
            positions=positions,
            alleles=alleles,
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
            sample_ids=sample_ids,
        )

        anc_alleles = np.array(["A", "A"])
        # Cache fits 9 bytes: 2+2+8=12 > 9, and 2+8=10 > 9,
        # so inserting the large chunk must evict both smalls.
        cache = ChunkCache(max_bytes=9)

        reader_small_a = VCZHaplotypeReader(
            store_small,
            positions,
            anc_alleles,
            source_name="small_a",
            cache=cache,
        )
        reader_small_b = VCZHaplotypeReader(
            store_small,
            positions,
            anc_alleles,
            source_name="small_b",
            cache=cache,
        )
        reader_large = VCZHaplotypeReader(
            store_large,
            positions,
            anc_alleles,
            source_name="large",
            cache=cache,
        )

        # Fill with 2-byte chunks: 2+2 = 4 bytes
        reader_small_a.read_haplotype("sample_0")
        reader_small_b.read_haplotype("sample_0")
        assert len(cache._cache) == 2
        assert cache.total_bytes == 4

        # Insert large 8-byte chunk: 4 + 8 = 12 > 10, must evict both smalls
        reader_large.read_haplotype("s0")
        assert ("small_a", 0) not in cache._cache
        assert ("small_b", 0) not in cache._cache
        assert ("large", 0) in cache._cache
        assert cache.total_bytes == 8

    def test_cache_bytes_tracking(self):
        """ChunkCache tracks _current_bytes correctly through inserts/evictions."""
        cache = ChunkCache(max_bytes=100)
        a = np.zeros(30, dtype=np.int8)  # 30 bytes
        b = np.zeros(40, dtype=np.int8)  # 40 bytes
        c = np.zeros(50, dtype=np.int8)  # 50 bytes

        cache.put(("x", 0), a)
        assert cache.total_bytes == 30

        cache.put(("x", 1), b)
        assert cache.total_bytes == 70

        # c (50) + 70 = 120 > 100 → evict a (30) → 40+50=90
        cache.put(("x", 2), c)
        assert cache.total_bytes == 90
        assert ("x", 0) not in cache._cache


# ---------------------------------------------------------------------------
# ChunkCache
# ---------------------------------------------------------------------------


def _arr(n):
    """Return a zero-filled int8 array of exactly *n* bytes."""
    return np.arange(n, dtype=np.int8)


class TestChunkCache:
    # --- Group 1: get() ---

    def test_get_miss_returns_none(self):
        cache = ChunkCache(max_bytes=100)
        assert cache.get(("s", 0)) is None
        assert cache._misses == 1

    def test_get_hit_returns_data(self):
        cache = ChunkCache(max_bytes=100)
        a = _arr(10)
        cache.put(("s", 0), a)
        result = cache.get(("s", 0))
        assert result is a
        assert cache._hits == 1

    def test_get_updates_lru_order(self):
        cache = ChunkCache(max_bytes=30)
        cache.put(("s", 0), _arr(10))  # A
        cache.put(("s", 1), _arr(10))  # B
        # Touch A so B becomes the LRU entry
        cache.get(("s", 0))
        # Insert C — should evict B (LRU), not A
        cache.put(("s", 2), _arr(15))
        assert ("s", 1) not in cache._cache
        assert ("s", 0) in cache._cache

    # --- Group 2: put() ---

    def test_put_inserts_and_tracks_bytes(self):
        cache = ChunkCache(max_bytes=100)
        cache.put(("s", 0), _arr(30))
        assert cache.total_bytes == 30
        assert ("s", 0) in cache._cache

    def test_put_noop_if_key_exists(self):
        cache = ChunkCache(max_bytes=100)
        a = _arr(30)
        cache.put(("s", 0), a)
        cache.put(("s", 0), _arr(50))  # should be ignored
        assert cache.total_bytes == 30
        assert cache._cache[("s", 0)] is a

    def test_put_evicts_lru_single(self):
        cache = ChunkCache(max_bytes=50)
        cache.put(("s", 0), _arr(30))
        cache.put(("s", 1), _arr(30))
        # 30+30=60 > 50 → evict ("s",0)
        assert ("s", 0) not in cache._cache
        assert ("s", 1) in cache._cache
        assert cache.total_bytes == 30

    def test_put_evicts_multiple(self):
        cache = ChunkCache(max_bytes=100)
        cache.put(("s", 0), _arr(30))
        cache.put(("s", 1), _arr(30))
        cache.put(("s", 2), _arr(30))
        # 90 used; inserting 50 → need to evict at least 40
        cache.put(("s", 3), _arr(50))
        assert ("s", 0) not in cache._cache
        assert ("s", 1) not in cache._cache
        assert ("s", 2) in cache._cache
        assert ("s", 3) in cache._cache
        assert cache.total_bytes == 80

    def test_put_item_larger_than_max(self):
        cache = ChunkCache(max_bytes=10)
        cache.put(("s", 0), _arr(5))
        cache.put(("s", 1), _arr(20))
        # All prior entries evicted; oversized item still inserted
        assert ("s", 0) not in cache._cache
        assert ("s", 1) in cache._cache
        assert cache.total_bytes == 20

    # --- Group 3: get_or_wait + finish_load (single-thread) ---

    def test_get_or_wait_hit(self):
        cache = ChunkCache(max_bytes=100)
        a = _arr(10)
        cache.put(("s", 0), a)
        result = cache.get_or_wait(("s", 0))
        assert result is a
        assert cache._hits == 1

    def test_get_or_wait_loader_path(self):
        cache = ChunkCache(max_bytes=100)
        result = cache.get_or_wait(("s", 0))
        assert result is None
        assert ("s", 0) in cache._pending
        assert cache._misses == 1

    def test_finish_load_inserts_and_wakes(self):
        cache = ChunkCache(max_bytes=100)
        cache.get_or_wait(("s", 0))  # creates pending
        a = _arr(20)
        cache.finish_load(("s", 0), a)
        assert ("s", 0) in cache._cache
        assert ("s", 0) not in cache._pending
        assert cache.total_bytes == 20

    def test_finish_load_none_clears_pending(self):
        cache = ChunkCache(max_bytes=100)
        cache.get_or_wait(("s", 0))
        cache.finish_load(("s", 0), None)
        assert ("s", 0) not in cache._cache
        assert ("s", 0) not in cache._pending
        assert cache.total_bytes == 0

    # --- Group 4: Threading coordination ---

    def test_waiter_receives_data(self):
        cache = ChunkCache(max_bytes=100)
        a = _arr(20)
        # Main thread becomes the loader
        assert cache.get_or_wait(("s", 0)) is None

        results = [None]
        loader_called = threading.Event()

        def waiter():
            loader_called.wait()
            results[0] = cache.get_or_wait(("s", 0))

        t = threading.Thread(target=waiter)
        t.start()
        loader_called.set()
        time.sleep(0.05)
        cache.finish_load(("s", 0), a)
        t.join(timeout=5)
        assert not t.is_alive()
        np.testing.assert_array_equal(results[0], a)

    def test_waiter_gets_none_on_load_failure(self):
        cache = ChunkCache(max_bytes=100)
        assert cache.get_or_wait(("s", 0)) is None

        results = [object()]  # sentinel
        loader_called = threading.Event()

        def waiter():
            loader_called.wait()
            results[0] = cache.get_or_wait(("s", 0))

        t = threading.Thread(target=waiter)
        t.start()
        loader_called.set()
        time.sleep(0.05)
        cache.finish_load(("s", 0), None)
        t.join(timeout=5)
        assert not t.is_alive()
        assert results[0] is None

    def test_multiple_waiters(self):
        cache = ChunkCache(max_bytes=100)
        a = _arr(20)
        assert cache.get_or_wait(("s", 0)) is None

        barrier = threading.Barrier(3, timeout=5)
        results = [None, None]

        def waiter(idx):
            barrier.wait()
            results[idx] = cache.get_or_wait(("s", 0))

        threads = [threading.Thread(target=waiter, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        barrier.wait()  # ensure both waiters are ready
        time.sleep(0.05)
        cache.finish_load(("s", 0), a)
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()
        for r in results:
            np.testing.assert_array_equal(r, a)

    # --- Group 5: evict_for() ---

    def test_evict_for_empty_cache(self):
        cache = ChunkCache(max_bytes=100)
        cache.evict_for(50)  # no-op
        assert cache.total_bytes == 0

    def test_evict_for_no_eviction_needed(self):
        cache = ChunkCache(max_bytes=100)
        cache.put(("s", 0), _arr(30))
        cache.evict_for(50)  # 30+50=80 ≤ 100 → nothing evicted
        assert ("s", 0) in cache._cache
        assert cache.total_bytes == 30

    def test_evict_for_evicts_lru(self):
        cache = ChunkCache(max_bytes=100)
        cache.put(("s", 0), _arr(40))
        cache.put(("s", 1), _arr(40))
        cache.evict_for(50)  # 80+50=130 > 100 → evict ("s",0) → 40+50=90
        assert ("s", 0) not in cache._cache
        assert ("s", 1) in cache._cache
        assert cache.total_bytes == 40

    def test_evict_for_evicts_multiple(self):
        cache = ChunkCache(max_bytes=100)
        cache.put(("s", 0), _arr(30))
        cache.put(("s", 1), _arr(30))
        cache.put(("s", 2), _arr(30))
        # 90 used; need room for 50 → must evict 40 worth
        cache.evict_for(50)
        assert ("s", 0) not in cache._cache
        assert ("s", 1) not in cache._cache
        assert ("s", 2) in cache._cache
        assert cache.total_bytes == 30

    def test_evict_for_then_put_no_double_eviction(self):
        cache = ChunkCache(max_bytes=100)
        cache.put(("s", 0), _arr(40))
        cache.put(("s", 1), _arr(40))
        # Pre-evict for upcoming 50-byte insert
        cache.evict_for(50)
        assert cache.total_bytes == 40  # ("s",0) evicted
        # Now put should NOT need to evict ("s",1)
        cache.put(("s", 2), _arr(50))
        assert ("s", 1) in cache._cache
        assert ("s", 2) in cache._cache
        assert len(cache._cache) == 2
        assert cache.total_bytes == 90

    # --- Group 6: Edge cases ---

    def test_finish_load_without_prior_get_or_wait(self):
        cache = ChunkCache(max_bytes=100)
        a = _arr(20)
        cache.finish_load(("s", 0), a)  # no pending entry
        assert ("s", 0) in cache._cache
        assert cache.total_bytes == 20

    def test_hits_and_misses_counting(self):
        cache = ChunkCache(max_bytes=100)
        cache.get(("s", 0))  # miss
        cache.get(("s", 1))  # miss
        cache.put(("s", 0), _arr(10))
        cache.get(("s", 0))  # hit
        cache.get_or_wait(("s", 0))  # hit
        cache.get_or_wait(("s", 2))  # miss (loader path)
        cache.finish_load(("s", 2), _arr(10))
        assert cache._hits == 2
        assert cache._misses == 3


# ---------------------------------------------------------------------------
# AlleleMapper
# ---------------------------------------------------------------------------


class TestAlleleMapper:
    def test_basic(self):
        """Ancestral allele is always code 0, first derived is 1."""
        m = AlleleMapper(2, [["A", "T"], ["C"]])
        assert m.lookup(0, "A") == 0
        assert m.lookup(0, "T") == 1
        # Idempotent
        assert m.lookup(0, "T") == 1
        assert m.lookup(0, "A") == 0
        assert m.num_alleles(0) == 2
        # Unknown allele returns -1
        assert m.lookup(0, "G") == -1

    def test_cross_source_distinct_codes(self):
        """Different derived alleles at the same site get distinct codes."""
        m = AlleleMapper(1, [["A", "T", "G", "C"]])
        assert m.lookup(0, "T") == 1
        assert m.lookup(0, "G") == 2
        assert m.lookup(0, "C") == 3
        assert m.num_alleles(0) == 4

    def test_site_alleles_array(self):
        m = AlleleMapper(2, [["A", "T"], ["C", "G", "T"]])
        arr = m.site_alleles_array()
        assert arr.shape[0] == 2
        assert arr.shape[1] >= 3  # site 1 has 3 alleles
        assert arr[0, 0] == "A"
        assert arr[0, 1] == "T"
        assert arr[1, 0] == "C"
        assert arr[1, 1] == "G"
        assert arr[1, 2] == "T"

    def test_ancestral_alleles(self):
        """ancestral_alleles returns column 0 of forward map."""
        m = AlleleMapper(3, [["A", "T"], ["C", "G"], ["G"]])
        anc = m.ancestral_alleles()
        assert list(anc) == ["A", "C", "G"]

    def test_forward_map_same_as_site_alleles_array(self):
        """forward_map returns the same data as site_alleles_array."""
        m = AlleleMapper(2, [["A", "T"], ["C", "G", "T"]])
        fm = m.forward_map()
        sa = m.site_alleles_array()
        np.testing.assert_array_equal(fm, sa)

    def test_decode_mutations(self):
        """decode_mutations maps codes to correct strings."""
        m = AlleleMapper(2, [["A", "T"], ["C", "G", "T"]])
        site_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        codes = np.array([0, 1, 1, 2], dtype=np.int8)
        result = m.decode_mutations(site_ids, codes)
        np.testing.assert_array_equal(result, ["A", "T", "G", "T"])

    def test_encode_mutations(self):
        """encode_mutations maps strings to correct codes."""
        m = AlleleMapper(2, [["A", "T"], ["C", "G", "T"]])
        site_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        alleles = np.array(["A", "T", "G", "T"])
        result = m.encode_mutations(site_ids, alleles)
        np.testing.assert_array_equal(result, [0, 1, 1, 2])

    def test_encode_decode_roundtrip(self):
        """encode_mutations round-trips with decode_mutations."""
        m = AlleleMapper(2, [["A", "T"], ["C", "G", "T"]])
        site_ids = np.array([0, 1, 1, 0], dtype=np.int32)
        codes = np.array([1, 0, 2, 0], dtype=np.int8)
        alleles = m.decode_mutations(site_ids, codes)
        encoded = m.encode_mutations(site_ids, alleles)
        np.testing.assert_array_equal(encoded, codes)


# ---------------------------------------------------------------------------
# Cached chunk shape and variant selection
# ---------------------------------------------------------------------------


class TestVariantSelection:
    def test_cached_chunk_shape(self):
        """Cached array is (num_selected, chunk_samples, ploidy), not full."""
        # 5 source variants, only 2 match reference positions
        positions_ref = np.array([200, 400], dtype=np.int32)
        positions_src = np.array([100, 200, 300, 400, 500], dtype=np.int32)
        gt = np.zeros((5, 2, 1), dtype=np.int8)
        alleles = np.array([["A", "T"]] * 5)
        store = make_sample_vcz(
            genotypes=gt,
            positions=positions_src,
            alleles=alleles,
            ancestral_state=np.array(["A"] * 5),
            sequence_length=1000,
        )
        anc_alleles = np.array(["A", "A"])
        cache = ChunkCache(max_bytes=1024 * 1024)
        reader = VCZHaplotypeReader(
            store,
            positions_ref,
            anc_alleles,
            source_name="test",
            cache=cache,
        )
        reader.read_haplotype("sample_0")
        # Cache should have one entry with shape (2, 2, 1) not (5, 2, 1)
        cached = list(cache._cache.values())[0]
        assert cached.shape == (2, 2, 1)

    def test_chunk_bytes_reflects_selected(self):
        """chunk_bytes uses num_selected, not total source variants."""
        positions_ref = np.array([200], dtype=np.int32)
        positions_src = np.array([100, 200, 300], dtype=np.int32)
        gt = np.zeros((3, 2, 1), dtype=np.int8)
        store = make_sample_vcz(
            genotypes=gt,
            positions=positions_src,
            alleles=np.array([["A", "T"]] * 3),
            ancestral_state=np.array(["A"] * 3),
            sequence_length=1000,
        )
        cache = ChunkCache(max_bytes=1024 * 1024)
        reader = VCZHaplotypeReader(
            store,
            positions_ref,
            np.array(["A"]),
            source_name="test",
            cache=cache,
        )
        # 1 selected * 2 samples * 1 ploidy * 1 byte = 2
        assert reader.chunk_bytes == 2

    def test_variant_chunks_skipped(self):
        """Store with many variants but few selected returns correct output."""
        # 10 source variants, only positions 300, 700 match reference
        positions_src = np.array(
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=np.int32
        )
        positions_ref = np.array([300, 700], dtype=np.int32)
        # sample_0: allele 1 at pos 300, allele 0 at pos 700
        gt = np.zeros((10, 1, 1), dtype=np.int8)
        gt[2, 0, 0] = 1  # pos 300
        gt[6, 0, 0] = 0  # pos 700
        alleles = np.array([["A", "T"]] * 10)
        store = make_sample_vcz(
            genotypes=gt,
            positions=positions_src,
            alleles=alleles,
            ancestral_state=np.array(["A"] * 10),
            sequence_length=2000,
        )
        anc_alleles = np.array(["A", "A"])
        cache = ChunkCache(max_bytes=1024 * 1024)
        reader = VCZHaplotypeReader(
            store,
            positions_ref,
            anc_alleles,
            source_name="test",
            cache=cache,
        )
        hap = reader.read_haplotype("sample_0")
        np.testing.assert_array_equal(hap, [1, 0])


class TestGetSiteAlleles:
    def test_facade_returns_alleles_after_reads(self):
        """get_site_alleles returns correct allele array after reads."""
        from tsinfer.config import Source

        positions = np.array([100, 200], dtype=np.int32)
        # Source A: site 0 has alleles A/T, site 1 has A/G
        store_a = make_sample_vcz(
            genotypes=np.array([[[1]], [[1]]], dtype=np.int8),
            positions=positions,
            alleles=np.array([["A", "T"], ["A", "G"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        # Source B: site 0 has alleles A/C, site 1 has A/G
        store_b = make_sample_vcz(
            genotypes=np.array([[[1]], [[1]]], dtype=np.int8),
            positions=positions,
            alleles=np.array([["A", "C"], ["A", "G"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        sources = {
            "src_a": Source(path=store_a, name="src_a"),
            "src_b": Source(path=store_b, name="src_b"),
        }
        anc_alleles = np.array(["A", "A"])
        reader = HaplotypeReader(sources, positions, anc_alleles)

        # Read from both sources to trigger allele discovery
        job_a = MatchJob(
            haplotype_index=0,
            source="src_a",
            sample_id="sample_0",
            ploidy_index=0,
            time=0.0,
            start_position=100,
            end_position=200,
            group=0,
        )
        job_b = MatchJob(
            haplotype_index=1,
            source="src_b",
            sample_id="sample_0",
            ploidy_index=0,
            time=0.0,
            start_position=100,
            end_position=200,
            group=0,
        )
        reader.read_haplotype(job_a)
        reader.read_haplotype(job_b)

        alleles = reader.get_site_alleles()
        # Site 0: A (ancestral), T (from src_a), C (from src_b)
        assert alleles[0, 0] == "A"
        assert alleles[0, 1] == "T"
        assert alleles[0, 2] == "C"
        # Site 1: A (ancestral), G (from both sources — same code)
        assert alleles[1, 0] == "A"
        assert alleles[1, 1] == "G"
