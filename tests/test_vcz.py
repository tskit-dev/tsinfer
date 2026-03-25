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
    HaplotypeReader,
    ScheduledCache,
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
        schedule = [("ancestors", "a0", 0)]
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
            schedule=schedule,
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
        schedule = [("ancestors", "a1", 0)]
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
            schedule=schedule,
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
        schedule = [("test", "sample_0", 0)]
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
            schedule=schedule,
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
        schedule = [("test", "sample_1", 0)]
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
            schedule=schedule,
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
        schedule = [("test", "sample_0", 0)]
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
            schedule=schedule,
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
        schedule = [("src_a", "sample_0", 0), ("src_b", "sample_0", 0)]
        reader = HaplotypeReader(sources, positions, anc_alleles, schedule=schedule)

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

    def test_no_schedule(self):
        """HaplotypeReader works when constructed without a schedule."""
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
        )
        # Cache should have empty refcounts
        assert len(reader._cache._refcount) == 0

    def test_shutdown(self):
        """HaplotypeReader.shutdown() shuts down the cache executor."""
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
        )
        reader.shutdown()

    def test_allele_mapper_property(self):
        """allele_mapper property returns an AlleleMapper instance."""
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
        )
        assert isinstance(reader.allele_mapper, AlleleMapper)

    def test_get_num_alleles(self):
        """get_num_alleles returns per-site allele counts."""
        anc_store, sample_store, source, anc_source, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = HaplotypeReader(
            {"ancestors": anc_source, "test": source},
            positions,
            anc_alleles,
        )
        num_alleles = reader.get_num_alleles()
        assert num_alleles.shape == (len(positions),)
        assert num_alleles.dtype == np.uint64
        # Each site has at least 2 alleles (ancestral + 1 derived)
        assert np.all(num_alleles >= 2)


# ---------------------------------------------------------------------------
# VCZHaplotypeReader
# ---------------------------------------------------------------------------


def _make_cache_for_reader(reader, sample_ids, max_bytes=1024 * 1024):
    """Build a ScheduledCache for a VCZHaplotypeReader test.

    Each sample_id is scheduled for exactly one read at ploidy_index=0.
    """
    refcounts = {}
    chunk_order = []
    seen = set()
    for sid in sample_ids:
        chunk_idx = reader._chunk_for_sample(sid)
        key = (reader._source_name, chunk_idx)
        refcounts[key] = refcounts.get(key, 0) + 1
        if key not in seen:
            chunk_order.append(key)
            seen.add(key)
    cache = ScheduledCache(max_bytes, refcounts, chunk_order)
    cache.register_loader(reader._source_name, reader._do_load_chunk, reader.chunk_bytes)
    cache.start()
    return cache


class TestVCZHaplotypeReader:
    def test_ancestor_store_direct(self):
        """VCZHaplotypeReader reads ancestor haplotypes correctly."""
        anc_store, _, _, _, positions, anc_alleles = _make_ancestor_and_sample_stores()
        reader = VCZHaplotypeReader(
            anc_store,
            positions,
            anc_alleles,
            source_name="anc",
            cache=None,
        )
        cache = _make_cache_for_reader(reader, ["a0"])
        reader._cache = cache
        hap = reader.read_haplotype("a0", ploidy_index=0)
        np.testing.assert_array_equal(hap, [0, 1, 0])

    def test_sample_store_direct(self):
        """VCZHaplotypeReader reads and polarises sample haplotypes."""
        _, sample_store, _, _, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=None,
        )
        cache = _make_cache_for_reader(reader, ["sample_0"])
        reader._cache = cache
        # sample_0: gt [0, 1, 1] → encoded [0, 1, 1]
        hap = reader.read_haplotype("sample_0", ploidy_index=0)
        np.testing.assert_array_equal(hap, [0, 1, 1])

    def test_cache_hit(self):
        """Two samples in the same chunk share a single load."""
        anc_store, sample_store, _, _, positions, anc_alleles = (
            _make_ancestor_and_sample_stores()
        )
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=None,
        )
        cache = _make_cache_for_reader(reader, ["sample_0", "sample_1"])
        reader._cache = cache
        reader.read_haplotype("sample_0")
        # Chunk is still alive (refcount=1 remaining)
        assert ("test", 0) in cache._chunks
        reader.read_haplotype("sample_1")
        # After last read, chunk is evicted (refcount=0)
        assert ("test", 0) not in cache._chunks
        cache.shutdown()

    def test_refcount_eviction(self):
        """Chunk is evicted exactly when its last scheduled read completes."""
        positions = np.array([100], dtype=np.int32)
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
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=None,
        )
        # Schedule: read one sample from each of the 3 chunks
        cache = _make_cache_for_reader(
            reader,
            ["sample_0", "sample_2", "sample_4"],
            max_bytes=1024 * 1024,
        )
        reader._cache = cache
        # Read from chunk 0 — evicted immediately (only 1 scheduled read)
        reader.read_haplotype("sample_0")
        assert ("test", 0) not in cache._chunks
        # Read from chunk 1
        reader.read_haplotype("sample_2")
        assert ("test", 1) not in cache._chunks
        # Read from chunk 2
        reader.read_haplotype("sample_4")
        assert ("test", 2) not in cache._chunks
        assert cache.total_bytes == 0

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
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=None,
        )
        cache = _make_cache_for_reader(reader, ["sample_0"])
        reader._cache = cache
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
        reader = VCZHaplotypeReader(
            sample_store,
            positions,
            anc_alleles,
            source_name="test",
            cache=None,
            samples_selection=selection,
        )
        cache = _make_cache_for_reader(reader, ["sample_1"])
        reader._cache = cache
        hap = reader.read_haplotype("sample_1")
        np.testing.assert_array_equal(hap, [1])

    def test_no_overlapping_positions(self):
        """Reader with no overlapping positions returns all -1."""
        # Source has positions 100, 200 but reference has 500, 600
        positions_src = np.array([100, 200], dtype=np.int32)
        positions_ref = np.array([500, 600], dtype=np.int32)
        gt = np.array([[[0]], [[1]]], dtype=np.int8)
        sample_store = make_sample_vcz(
            genotypes=gt,
            positions=positions_src,
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        anc_alleles = np.array(["A", "A"])
        reader = VCZHaplotypeReader(
            sample_store,
            positions_ref,
            anc_alleles,
            source_name="test",
            cache=None,
        )
        assert reader._num_selected == 0
        # _precompute_allele_remap should return early
        reader._precompute_allele_remap()
        cache = _make_cache_for_reader(reader, ["sample_0"])
        reader._cache = cache
        hap = reader.read_haplotype("sample_0")
        np.testing.assert_array_equal(hap, [-1, -1])

    def test_variant_chunk_fully_skipped(self):
        """Variant chunks with no selected sites are skipped during load."""
        # 6 source variants with variant chunk size 2 → 3 variant chunks
        # Only position 500 matches reference → only chunk 2 has selected sites
        positions_src = np.array([100, 200, 300, 400, 500, 600], dtype=np.int32)
        positions_ref = np.array([500], dtype=np.int32)
        gt = np.zeros((6, 1, 1), dtype=np.int8)
        gt[4, 0, 0] = 1  # pos 500
        alleles = np.array([["A", "T"]] * 6)
        sample_store = make_sample_vcz(
            genotypes=gt,
            positions=positions_src,
            alleles=alleles,
            ancestral_state=np.array(["A"] * 6),
            sequence_length=1000,
        )
        # Force variant chunk size to 2
        del sample_store["call_genotype"]
        cg = sample_store.create_array(
            "call_genotype",
            data=gt,
            chunks=(2, 1, 1),
        )
        cg.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]
        anc_alleles = np.array(["A"])
        reader = VCZHaplotypeReader(
            sample_store,
            positions_ref,
            anc_alleles,
            source_name="test",
            cache=None,
        )
        # Only 1 of 3 variant chunks should be in _needed_vc
        assert len(reader._needed_vc) == 1
        cache = _make_cache_for_reader(reader, ["sample_0"])
        reader._cache = cache
        hap = reader.read_haplotype("sample_0")
        np.testing.assert_array_equal(hap, [1])


# ---------------------------------------------------------------------------
# ChunkCache
# ---------------------------------------------------------------------------


def _arr(n):
    """Return a zero-filled int8 array of exactly *n* bytes."""
    return np.arange(n, dtype=np.int8)


class TestScheduledCache:
    # --- Group 1: prime() and get() ---

    def test_prime_loads_chunks(self):
        data = _arr(10)
        cache = ScheduledCache(
            max_bytes=100,
            refcounts={("s", 0): 1},
            chunk_order=[("s", 0)],
        )
        cache.register_loader("s", lambda idx: data, chunk_bytes=10)
        cache.start()
        # get() blocks until prime delivers the chunk
        result = cache.get(("s", 0))
        np.testing.assert_array_equal(result, data)
        assert cache.total_bytes == 10
        cache.shutdown()

    def test_prime_fills_to_capacity(self):
        cache = ScheduledCache(
            max_bytes=25,
            refcounts={("s", 0): 1, ("s", 1): 1, ("s", 2): 1},
            chunk_order=[("s", 0), ("s", 1), ("s", 2)],
        )
        cache.register_loader("s", lambda idx: _arr(10), chunk_bytes=10)
        cache.start()
        # get() on chunk 1 ensures prime has finished
        cache.get(("s", 1))
        # 2 chunks fit (20 <= 25), 3rd would exceed (30 > 25)
        assert ("s", 0) in cache._chunks
        assert ("s", 1) in cache._chunks
        assert ("s", 2) not in cache._chunks
        assert cache.total_bytes == 20
        cache.shutdown()

    def test_get_returns_cached(self):
        call_count = [0]

        def loader(idx):
            call_count[0] += 1
            return _arr(10)

        cache = ScheduledCache(
            max_bytes=100,
            refcounts={("s", 0): 2},
            chunk_order=[("s", 0)],
        )
        cache.register_loader("s", loader, chunk_bytes=10)
        cache.start()
        cache.get(("s", 0))
        result = cache.get(("s", 0))
        assert call_count[0] == 1
        assert cache._hits == 2
        assert result is not None
        cache.shutdown()

    def test_get_waits_for_prime(self):
        """get() blocks until prime delivers the chunk."""
        cache = ScheduledCache(
            max_bytes=100,
            refcounts={("s", 0): 1},
            chunk_order=[("s", 0)],
        )
        cache.register_loader("s", lambda idx: _arr(10), chunk_bytes=10)
        cache.start()
        # get() should block until background prime loads it
        result = cache.get(("s", 0))
        assert result is not None
        cache.shutdown()

    def test_get_waits_for_readahead(self):
        """get() blocks until readahead delivers the chunk."""
        cache = ScheduledCache(
            max_bytes=10,
            refcounts={("s", 0): 1, ("s", 1): 1},
            chunk_order=[("s", 0), ("s", 1)],
        )
        cache.register_loader("s", lambda idx: _arr(10), chunk_bytes=10)
        cache.start()
        # Wait for prime to finish loading chunk 0
        cache.get(("s", 0))

        result = [None]

        def worker():
            result[0] = cache.get(("s", 1))

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.05)
        # Evict chunk 0 — triggers readahead for chunk 1
        cache.record_read(("s", 0))
        t.join(timeout=5)
        assert result[0] is not None
        np.testing.assert_array_equal(result[0], _arr(10))
        cache.shutdown()

    # --- Group 2: record_read() ---

    def test_record_read_decrements_refcount(self):
        cache = ScheduledCache(
            max_bytes=100,
            refcounts={("s", 0): 3},
            chunk_order=[("s", 0)],
        )
        cache.register_loader("s", lambda idx: _arr(10), chunk_bytes=10)
        cache.start()
        cache.get(("s", 0))  # wait for prime
        cache.record_read(("s", 0))
        assert cache._refcount[("s", 0)] == 2
        assert ("s", 0) in cache._chunks
        cache.shutdown()

    def test_record_read_evicts_at_zero(self):
        cache = ScheduledCache(
            max_bytes=100,
            refcounts={("s", 0): 1},
            chunk_order=[("s", 0)],
        )
        cache.register_loader("s", lambda idx: _arr(10), chunk_bytes=10)
        cache.start()
        cache.get(("s", 0))  # wait for prime
        assert cache.total_bytes == 10
        cache.record_read(("s", 0))
        assert ("s", 0) not in cache._chunks
        assert cache.total_bytes == 0
        cache.shutdown()

    def test_record_read_multiple_chunks(self):
        cache = ScheduledCache(
            max_bytes=100,
            refcounts={("s", 0): 1, ("s", 1): 2},
            chunk_order=[("s", 0), ("s", 1)],
        )
        cache.register_loader("s", lambda idx: _arr(10), chunk_bytes=10)
        cache.start()
        cache.get(("s", 1))  # wait for prime to load both
        assert cache.total_bytes == 20

        cache.record_read(("s", 0))
        assert ("s", 0) not in cache._chunks
        assert ("s", 1) in cache._chunks
        assert cache.total_bytes == 10

        cache.record_read(("s", 1))
        assert ("s", 1) in cache._chunks  # refcount=1, not zero

        cache.record_read(("s", 1))
        assert ("s", 1) not in cache._chunks
        assert cache.total_bytes == 0
        cache.shutdown()

    # --- Group 3: read-ahead ---

    def test_readahead_triggers_on_eviction(self):
        loaded = set()

        def loader(idx):
            loaded.add(idx)
            return _arr(10)

        cache = ScheduledCache(
            max_bytes=15,
            refcounts={("s", 0): 1, ("s", 1): 1},
            chunk_order=[("s", 0), ("s", 1)],
        )
        cache.register_loader("s", loader, chunk_bytes=10)
        cache.start()
        cache.get(("s", 0))  # wait for prime
        assert 0 in loaded
        assert 1 not in loaded
        # Evict chunk 0 — should trigger readahead for chunk 1
        cache.record_read(("s", 0))
        time.sleep(0.2)
        assert 1 in loaded
        assert ("s", 1) in cache._chunks
        cache.shutdown()

    def test_readahead_respects_memory_limit(self):
        """Readahead does not fire when next chunk doesn't fit."""
        cache = ScheduledCache(
            max_bytes=15,
            refcounts={("s", 0): 2, ("s", 1): 1},
            chunk_order=[("s", 0), ("s", 1)],
        )
        loaded = set()

        def loader(idx):
            loaded.add(idx)
            return _arr(10)

        cache.register_loader("s", loader, chunk_bytes=10)
        cache.start()
        cache.get(("s", 0))  # wait for prime
        # Decrement but don't evict (refcount 2→1)
        cache.record_read(("s", 0))
        assert cache.total_bytes == 10
        # Readahead should not trigger: 10 + 10 > 15 and chunk 0 still alive
        time.sleep(0.05)
        assert 1 not in loaded
        cache.shutdown()

    def test_readahead_multi_eviction(self):
        """Large chunk needs multiple small evictions before it fits."""
        loaded = {}

        def loader_small(idx):
            loaded[("small", idx)] = True
            return _arr(10)

        def loader_big(idx):
            loaded[("big", idx)] = True
            return _arr(30)

        cache = ScheduledCache(
            max_bytes=35,
            refcounts={
                ("small", 0): 1,
                ("small", 1): 1,
                ("small", 2): 1,
                ("big", 0): 1,
            },
            chunk_order=[
                ("small", 0),
                ("small", 1),
                ("small", 2),
                ("big", 0),
            ],
        )
        cache.register_loader("small", loader_small, chunk_bytes=10)
        cache.register_loader("big", loader_big, chunk_bytes=30)
        cache.start()  # loads small 0,1,2 (30 bytes); big 0 won't fit
        cache.get(("small", 2))  # wait for prime
        assert cache.total_bytes == 30
        assert ("big", 0) not in cache._chunks

        # Evict small 0 → 20 bytes; big needs 30, still doesn't fit
        cache.record_read(("small", 0))
        time.sleep(0.1)
        assert ("big", 0) not in cache._chunks

        # Evict small 1 → 10 bytes; 10 + 30 > 35, still doesn't fit
        cache.record_read(("small", 1))
        time.sleep(0.1)
        assert ("big", 0) not in cache._chunks

        # Evict small 2 → 0 bytes; 0 + 30 <= 35, now it fits
        cache.record_read(("small", 2))
        time.sleep(0.2)
        assert ("big", 0) in cache._chunks
        assert cache.total_bytes == 30
        cache.shutdown()

    def test_readahead_skips_unregistered_source(self):
        cache = ScheduledCache(
            max_bytes=100,
            refcounts={("unknown", 0): 1, ("s", 0): 1},
            chunk_order=[("unknown", 0), ("s", 0)],
        )
        cache.register_loader("s", lambda idx: _arr(10), chunk_bytes=10)
        cache.start()
        cache.get(("s", 0))  # wait for prime
        # Skipped "unknown", loaded "s"
        assert ("s", 0) in cache._chunks
        assert ("unknown", 0) not in cache._chunks
        cache.shutdown()

    # --- Group 4: concurrent get() ---

    def test_concurrent_get_waiters(self):
        """Multiple threads waiting for the same chunk all get it."""
        cache = ScheduledCache(
            max_bytes=10,
            refcounts={("s", 0): 1, ("s", 1): 3},
            chunk_order=[("s", 0), ("s", 1)],
        )
        cache.register_loader("s", lambda idx: _arr(10), chunk_bytes=10)
        cache.start()
        cache.get(("s", 0))  # wait for prime

        results = [None, None, None]

        def worker(i):
            results[i] = cache.get(("s", 1))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        time.sleep(0.05)
        # Evict chunk 0 → readahead loads chunk 1 → all waiters wake
        cache.record_read(("s", 0))
        for t in threads:
            t.join(timeout=5)
        for r in results:
            np.testing.assert_array_equal(r, _arr(10))
        cache.shutdown()

    # --- Group 5: shutdown ---

    def test_shutdown(self):
        cache = ScheduledCache(
            max_bytes=100,
            refcounts={},
            chunk_order=[],
        )
        cache.shutdown()

    # --- Group 6: bytes tracking ---

    def test_total_bytes_tracking(self):
        cache = ScheduledCache(
            max_bytes=100,
            refcounts={("s", 0): 1, ("s", 1): 1},
            chunk_order=[("s", 0), ("s", 1)],
        )
        cache.register_loader("s", lambda idx: _arr(30), chunk_bytes=30)
        cache.start()
        cache.get(("s", 1))  # wait for prime
        assert cache.total_bytes == 60
        cache.record_read(("s", 0))
        assert cache.total_bytes == 30
        cache.record_read(("s", 1))
        assert cache.total_bytes == 0


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
        reader = VCZHaplotypeReader(
            store,
            positions_ref,
            anc_alleles,
            source_name="test",
            cache=None,
        )
        cache = _make_cache_for_reader(reader, ["sample_0", "sample_1"])
        reader._cache = cache
        reader.read_haplotype("sample_0")
        # Cache should have one entry with shape (2, 2, 1) not (5, 2, 1)
        cached = list(cache._chunks.values())[0]
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
        reader = VCZHaplotypeReader(
            store,
            positions_ref,
            np.array(["A"]),
            source_name="test",
            cache=None,
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
        reader = VCZHaplotypeReader(
            store,
            positions_ref,
            anc_alleles,
            source_name="test",
            cache=None,
        )
        cache = _make_cache_for_reader(reader, ["sample_0"])
        reader._cache = cache
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
        schedule = [("src_a", "sample_0", 0), ("src_b", "sample_0", 0)]
        reader = HaplotypeReader(sources, positions, anc_alleles, schedule=schedule)

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


# ---------------------------------------------------------------------------
# MultiSourceView
# ---------------------------------------------------------------------------


class TestMultiSourceView:
    def _anc_state(self, store):
        from tsinfer.config import AncestralState

        return AncestralState(path=store, field="variant_ancestral_allele")

    def test_single_source_positions_and_alleles(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0, 1]], [[1, 0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        assert view.num_sites == 2
        np.testing.assert_array_equal(view.positions, [100, 200])
        # Allele 0 = ancestral
        view.prepare(view.positions)
        assert view.alleles[0, 0] == "A"
        assert view.alleles[0, 1] == "T"
        assert view.alleles[1, 0] == "C"
        assert view.alleles[1, 1] == "G"

    def test_multiple_sources_unified_positions(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store_a = make_sample_vcz(
            genotypes=np.array([[[1]], [[0]]], dtype=np.int8),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
            sample_ids=np.array(["s0"]),
        )
        store_b = make_sample_vcz(
            genotypes=np.array([[[0]], [[1]]], dtype=np.int8),
            positions=np.array([200, 300], dtype=np.int32),
            alleles=np.array([["A", "G"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
            sample_ids=np.array(["s1"]),
        )
        # Use store_a as ancestral state source (has pos 100, 300)
        # Also need pos 200 from store_b's ancestral state
        # Build combined annotation
        ann_store = make_sample_vcz(
            genotypes=np.zeros((3, 1, 1), dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "G"], ["C", "G"]]),
            ancestral_state=np.array(["A", "A", "C"]),
            sequence_length=1000,
        )
        from tsinfer.config import AncestralState

        anc = AncestralState(path=ann_store, field="variant_ancestral_allele")
        view = MultiSourceView(
            [
                Source(path=store_a, name="a"),
                Source(path=store_b, name="b"),
            ],
            anc,
        )

        assert view.num_sites == 3
        np.testing.assert_array_equal(view.positions, [100, 200, 300])
        # Verify multi-source behavior via iter_genotypes
        variants = list(view.iter_genotypes())
        assert len(variants) == 3
        # Site 100: only src a has it (1 hap = T→derived=1), src b missing
        np.testing.assert_array_equal(variants[0].genotypes, [1, -1])
        # Site 200: src a missing, src b has it (genotype 0=A=ancestral=0)
        np.testing.assert_array_equal(variants[1].genotypes, [-1, 0])
        # Site 300: both have it; src a=ref(C)=0, src b=alt(G)=1
        np.testing.assert_array_equal(variants[2].genotypes, [0, 1])

    def test_duplicate_positions_excluded(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        # Position 100 appears twice — should be excluded
        store = make_sample_vcz(
            genotypes=np.array([[[0]], [[1]], [[0]]], dtype=np.int8),
            positions=np.array([100, 100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "A", "C"]),
            sequence_length=1000,
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        assert view.num_sites == 1
        np.testing.assert_array_equal(view.positions, [200])

    def test_haplotypes_enumeration(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0, 1], [1, 0]]], dtype=np.int8),
            positions=np.array([100], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        haps = view.haplotypes()
        assert len(haps) == 4  # 2 samples * 2 ploidy
        assert haps[0] == ("s1", "s0", 0)
        assert haps[1] == ("s1", "s0", 1)
        assert haps[2] == ("s1", "s1", 0)
        assert haps[3] == ("s1", "s1", 1)
        assert view.num_haplotypes == 4

    def test_iter_genotypes_single_source(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        # 2 sites, 2 haploid samples
        store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        rows = [v.genotypes for v in view.iter_genotypes()]
        assert len(rows) == 2
        # Site 0: s0=ref(A)=0, s1=alt(T)=1
        np.testing.assert_array_equal(rows[0], [0, 1])
        # Site 1: s0=alt(G)=1, s1=ref(C)=0
        np.testing.assert_array_equal(rows[1], [1, 0])

    def test_iter_genotypes_subset_positions(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]], [[0], [0]]], dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"], ["A", "T"]]),
            ancestral_state=np.array(["A", "C", "A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        # Only request positions 100 and 300
        subset = np.array([100, 300], dtype=np.int32)
        rows = [v.genotypes for v in view.iter_genotypes(subset)]
        assert len(rows) == 2
        np.testing.assert_array_equal(rows[0], [0, 1])  # site 100
        np.testing.assert_array_equal(rows[1], [0, 0])  # site 300

    def test_iter_genotypes_missing_source(self):
        """Sources missing a site contribute -1 arrays."""
        from tsinfer.config import AncestralState, Source
        from tsinfer.vcz import MultiSourceView

        store_a = make_sample_vcz(
            genotypes=np.array([[[1]]], dtype=np.int8),
            positions=np.array([100], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0"]),
        )
        store_b = make_sample_vcz(
            genotypes=np.array([[[1]]], dtype=np.int8),
            positions=np.array([200], dtype=np.int32),
            alleles=np.array([["C", "G"]]),
            ancestral_state=np.array(["C"]),
            sequence_length=1000,
            sample_ids=np.array(["s1"]),
        )
        ann_store = make_sample_vcz(
            genotypes=np.zeros((2, 1, 1), dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        anc = AncestralState(path=ann_store, field="variant_ancestral_allele")
        view = MultiSourceView(
            [
                Source(path=store_a, name="a"),
                Source(path=store_b, name="b"),
            ],
            anc,
        )

        rows = [v.genotypes for v in view.iter_genotypes()]
        assert len(rows) == 2
        # Site 100: src_a has it (1 hap), src_b missing (-1)
        np.testing.assert_array_equal(rows[0], [1, -1])
        # Site 200: src_a missing (-1), src_b has it (1 hap)
        np.testing.assert_array_equal(rows[1], [-1, 1])

    def test_sample_selection_respected(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]]], dtype=np.int8),
            positions=np.array([100], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1", "s2"]),
        )
        # Select only s0 and s2
        src = Source(path=store, name="s1", samples="s0,s2")
        view = MultiSourceView(src, self._anc_state(store))

        assert view.num_haplotypes == 2
        rows = [v.genotypes for v in view.iter_genotypes()]
        assert len(rows) == 1
        np.testing.assert_array_equal(rows[0], [0, 1])

    def test_prepare_subset(self):
        """prepare() with a position subset builds mapper for that subset."""
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]], [[0], [0]]], dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"], ["A", "T"]]),
            ancestral_state=np.array(["A", "C", "A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        # Prepare with subset
        subset = np.array([100, 300], dtype=np.int32)
        view.prepare(subset)
        assert view.alleles.shape[0] == 2
        rows = [v.genotypes for v in view.iter_genotypes(subset)]
        assert len(rows) == 2
        np.testing.assert_array_equal(rows[0], [0, 1])  # site 100
        np.testing.assert_array_equal(rows[1], [0, 0])  # site 300

    def test_alleles_before_prepare_raises(self):
        """Accessing alleles before prepare() raises RuntimeError."""
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0, 1]]], dtype=np.int8),
            positions=np.array([100], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        with pytest.raises(RuntimeError, match="prepare"):
            _ = view.alleles

    def test_positions_in_intervals(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]], [[0], [1]]], dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"], ["A", "T"]]),
            ancestral_state=np.array(["A", "C", "A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        # Only positions in [150, 350)
        pos = view.positions_in_intervals([[150, 350]])
        np.testing.assert_array_equal(pos, [200, 300])
        variants = list(view.iter_genotypes(pos))
        assert len(variants) == 2
        assert variants[0].position == 200.0
        assert variants[1].position == 300.0
        np.testing.assert_array_equal(variants[0].genotypes, [1, 0])
        np.testing.assert_array_equal(variants[1].genotypes, [0, 1])

    def test_filter_positions(self):
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]], [[0], [1]]], dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"], ["A", "T"]]),
            ancestral_state=np.array(["A", "C", "A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        # Exclude position 200
        pos = view.filter_positions(np.array([200]))
        np.testing.assert_array_equal(pos, [100, 300])
        variants = list(view.iter_genotypes(pos))
        assert len(variants) == 2
        assert variants[0].position == 100.0
        assert variants[1].position == 300.0

    def test_iter_genotypes_sample_identifiers(self):
        from tsinfer.config import AncestralState, Source
        from tsinfer.vcz import MultiSourceView

        store_a = make_sample_vcz(
            genotypes=np.array([[[0], [1]]], dtype=np.int8),
            positions=np.array([100], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        ann_store = store_a
        anc = AncestralState(path=ann_store, field="variant_ancestral_allele")
        view = MultiSourceView(
            [Source(path=store_a, name="src_a")],
            anc,
        )

        # Reverse order: s1 first, then s0
        identifiers = [("src_a", "s1", 0), ("src_a", "s0", 0)]
        variants = list(view.iter_genotypes(sample_identifiers=identifiers))
        assert len(variants) == 1
        # Natural order: s0=0, s1=1 → reversed: s1=1, s0=0
        np.testing.assert_array_equal(variants[0].genotypes, [1, 0])

    def test_iter_genotypes_sample_identifiers_missing(self):
        """Unmatched identifiers get -1."""
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0], [1]]], dtype=np.int8),
            positions=np.array([100], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        # Request s0 and a nonexistent sample
        identifiers = [("s1", "s0", 0), ("s1", "nonexistent", 0)]
        variants = list(view.iter_genotypes(sample_identifiers=identifiers))
        assert len(variants) == 1
        np.testing.assert_array_equal(variants[0].genotypes, [0, -1])

    def test_iter_genotypes_variant_alleles(self):
        """Variant.alleles tuple is correct."""
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0], [1]]], dtype=np.int8),
            positions=np.array([100], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
            sample_ids=np.array(["s0", "s1"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        variants = list(view.iter_genotypes())
        assert len(variants) == 1
        assert variants[0].alleles == ("A", "T")
        assert variants[0].position == 100.0

    def test_positions_in_intervals_and_filter(self):
        """positions_in_intervals and filter_positions compose."""
        from tsinfer.config import Source
        from tsinfer.vcz import MultiSourceView

        store = make_sample_vcz(
            genotypes=np.array([[[0]], [[1]], [[0]], [[1]]], dtype=np.int8),
            positions=np.array([100, 200, 300, 400], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"], ["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C", "A", "C"]),
            sequence_length=1000,
            sample_ids=np.array(["s0"]),
        )
        src = Source(path=store, name="s1")
        view = MultiSourceView(src, self._anc_state(store))

        # Interval [150, 450), then exclude 300
        pos = view.positions_in_intervals([[150, 450]])
        pos = view.filter_positions(np.array([300]), pos)
        np.testing.assert_array_equal(pos, [200, 400])
        variants = list(view.iter_genotypes(pos))
        assert len(variants) == 2
        assert variants[0].position == 200.0
        assert variants[1].position == 400.0
