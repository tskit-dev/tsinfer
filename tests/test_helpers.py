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
Tests for the test data helper functions.
"""

import msprime
import numpy as np
import pytest
import zarr
from helpers import make_ancestor_vcz, make_sample_vcz, ts_to_sample_vcz

# ---------------------------------------------------------------------------
# Shared minimal fixtures
# ---------------------------------------------------------------------------

_POSITIONS = np.array([100, 500], dtype=np.int32)
_ALLELES_2 = np.array([["A", "T"], ["C", "G"]])
_ANC_STATE = np.array(["A", "C"])
_SEQ_LEN = 1000

# diploid, 2 samples, 2 sites
_GT_DIPLOID = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=np.int8)


def _minimal_sample_vcz(**kwargs):
    return make_sample_vcz(
        _GT_DIPLOID, _POSITIONS, _ALLELES_2, _ANC_STATE, _SEQ_LEN, **kwargs
    )


# ---------------------------------------------------------------------------
# make_sample_vcz
# ---------------------------------------------------------------------------


class TestMakeSampleVcz:
    def test_returns_zarr_group(self):
        vcz = _minimal_sample_vcz()
        assert isinstance(vcz, zarr.Group)

    def test_zarr_format_v2(self):
        vcz = _minimal_sample_vcz()
        assert vcz.metadata.zarr_format == 2

    def test_required_arrays_present(self):
        vcz = _minimal_sample_vcz()
        required = {
            "call_genotype",
            "variant_position",
            "variant_allele",
            "variant_ancestral_allele",
            "variant_contig",
            "contig_id",
            "contig_length",
            "sample_id",
        }
        assert required <= set(vcz.keys())

    def test_call_genotype_shape_and_dtype(self):
        vcz = _minimal_sample_vcz()
        arr = vcz["call_genotype"]
        assert arr.shape == (2, 2, 2)  # (num_sites, n_samples, ploidy)
        assert arr.dtype == np.int8

    def test_call_genotype_values(self):
        vcz = _minimal_sample_vcz()
        np.testing.assert_array_equal(vcz["call_genotype"][:], _GT_DIPLOID)

    def test_missing_encoded_as_minus_one(self):
        gt = np.array([[[-1, 1], [0, 0]]], dtype=np.int8)
        vcz = make_sample_vcz(gt, [100], [["A", "T"]], ["A"], 1000)
        assert vcz["call_genotype"][0, 0, 0] == -1
        assert "call_genotype_mask" not in vcz

    def test_variant_position(self):
        vcz = _minimal_sample_vcz()
        np.testing.assert_array_equal(vcz["variant_position"][:], _POSITIONS)
        assert vcz["variant_position"].dtype == np.int32

    def test_variant_allele(self):
        vcz = _minimal_sample_vcz()
        arr = vcz["variant_allele"][:]
        assert arr.shape == (2, 2)
        assert arr[0, 0] == "A"
        assert arr[0, 1] == "T"

    def test_variant_ancestral_allele(self):
        vcz = _minimal_sample_vcz()
        arr = vcz["variant_ancestral_allele"][:]
        np.testing.assert_array_equal(arr, _ANC_STATE)

    def test_variant_contig_all_zeros(self):
        vcz = _minimal_sample_vcz()
        np.testing.assert_array_equal(vcz["variant_contig"][:], [0, 0])

    def test_contig_id_default(self):
        vcz = _minimal_sample_vcz()
        assert vcz["contig_id"][:][0] == "1"

    def test_contig_id_custom(self):
        vcz = _minimal_sample_vcz(contig_id="chr20")
        assert vcz["contig_id"][:][0] == "chr20"

    def test_contig_length(self):
        vcz = _minimal_sample_vcz()
        assert vcz["contig_length"][:][0] == _SEQ_LEN

    def test_sample_id_default(self):
        vcz = _minimal_sample_vcz()
        ids = vcz["sample_id"][:]
        assert list(ids) == ["sample_0", "sample_1"]

    def test_sample_id_custom(self):
        vcz = make_sample_vcz(
            _GT_DIPLOID,
            _POSITIONS,
            _ALLELES_2,
            _ANC_STATE,
            _SEQ_LEN,
            sample_ids=np.array(["NA001", "NA002"]),
        )
        assert list(vcz["sample_id"][:]) == ["NA001", "NA002"]

    def test_haploid(self):
        gt = np.array([[[0], [1]], [[1], [0]]], dtype=np.int8)
        vcz = make_sample_vcz(gt, _POSITIONS, _ALLELES_2, _ANC_STATE, _SEQ_LEN)
        assert vcz["call_genotype"].shape == (2, 2, 1)

    def test_single_site(self):
        gt = np.array([[[0, 1]]], dtype=np.int8)
        vcz = make_sample_vcz(gt, [100], [["A", "T"]], ["A"], 1000)
        assert vcz["call_genotype"].shape == (1, 1, 2)
        assert vcz["variant_position"][:][0] == 100

    def test_kwargs_stored_as_arrays(self):
        site_mask = np.array([True, False])
        sample_time = np.array([0.5, 1.0])
        vcz = _minimal_sample_vcz(site_mask=site_mask, sample_time=sample_time)
        np.testing.assert_array_equal(vcz["site_mask"][:], site_mask)
        np.testing.assert_array_equal(vcz["sample_time"][:], sample_time)

    def test_three_alleles(self):
        gt = np.array([[[0, 1], [2, 0]]], dtype=np.int8)
        alleles = np.array([["A", "T", "C"]])
        vcz = make_sample_vcz(gt, [100], alleles, ["A"], 1000)
        assert vcz["variant_allele"].shape == (1, 3)


# ---------------------------------------------------------------------------
# make_ancestor_vcz
# ---------------------------------------------------------------------------


def _make_two_ancestor_vcz():
    # 3 inference sites, 2 ancestors
    # ancestor 0: covers sites 1-2 (missing left flank)
    # ancestor 1: covers sites 0-1 (missing right flank)
    genotypes = np.array([[[-1], [0]], [[0], [1]], [[1], [-1]]], dtype=np.int8)
    positions = np.array([100, 300, 500], dtype=np.int32)
    alleles = np.array([["A", "T"], ["C", "G"], ["A", "T"]])
    times = np.array([2.0, 1.0])
    focal = np.array([[300, -2], [300, -2]], dtype=np.int32)
    intervals = np.array([[0, 1000]], dtype=np.int32)
    return make_ancestor_vcz(genotypes, positions, alleles, times, focal, intervals)


class TestMakeAncestorVcz:
    def test_returns_zarr_group(self):
        assert isinstance(_make_two_ancestor_vcz(), zarr.Group)

    def test_zarr_format_v2(self):
        assert _make_two_ancestor_vcz().metadata.zarr_format == 2

    def test_required_arrays_present(self):
        vcz = _make_two_ancestor_vcz()
        required = {
            "call_genotype",
            "variant_position",
            "variant_allele",
            "sample_id",
            "sample_time",
            "sample_start_position",
            "sample_end_position",
            "sample_focal_positions",
            "sequence_intervals",
        }
        assert required <= set(vcz.keys())

    def test_call_genotype_shape(self):
        vcz = _make_two_ancestor_vcz()
        assert vcz["call_genotype"].shape == (3, 2, 1)

    def test_call_genotype_dtype(self):
        assert _make_two_ancestor_vcz()["call_genotype"].dtype == np.int8

    def test_sample_time(self):
        vcz = _make_two_ancestor_vcz()
        np.testing.assert_array_equal(vcz["sample_time"][:], [2.0, 1.0])

    def test_sample_id_default(self):
        vcz = _make_two_ancestor_vcz()
        assert list(vcz["sample_id"][:]) == ["ancestor_0", "ancestor_1"]

    def test_start_position_derived_from_missing_pattern(self):
        vcz = _make_two_ancestor_vcz()
        # ancestor 0: first non-missing site is index 1 → position 300
        # ancestor 1: first non-missing site is index 0 → position 100
        np.testing.assert_array_equal(vcz["sample_start_position"][:], [300, 100])

    def test_end_position_derived_from_missing_pattern(self):
        vcz = _make_two_ancestor_vcz()
        # ancestor 0: last non-missing site is index 2 → position 500
        # ancestor 1: last non-missing site is index 1 → position 300
        np.testing.assert_array_equal(vcz["sample_end_position"][:], [500, 300])

    def test_focal_positions(self):
        vcz = _make_two_ancestor_vcz()
        fp = vcz["sample_focal_positions"][:]
        assert fp.shape == (2, 2)
        assert fp[0, 0] == 300
        assert fp[0, 1] == -2  # padding

    def test_sequence_intervals(self):
        vcz = _make_two_ancestor_vcz()
        np.testing.assert_array_equal(vcz["sequence_intervals"][:], [[0, 1000]])

    def test_single_ancestor_all_sites(self):
        gt = np.array([[[0]], [[1]], [[0]]], dtype=np.int8)
        pos = np.array([10, 20, 30], dtype=np.int32)
        alleles = np.array([["A", "T"]] * 3)
        times = np.array([1.0])
        focal = np.array([[20, -2]], dtype=np.int32)
        intervals = np.array([[0, 100]], dtype=np.int32)
        vcz = make_ancestor_vcz(gt, pos, alleles, times, focal, intervals)
        assert vcz["sample_start_position"][:][0] == 10
        assert vcz["sample_end_position"][:][0] == 30

    def test_multiple_focal_positions(self):
        gt = np.array([[[0]], [[1]], [[1]]], dtype=np.int8)
        pos = np.array([10, 20, 30], dtype=np.int32)
        alleles = np.array([["A", "T"]] * 3)
        times = np.array([1.0])
        focal = np.array([[20, 30]], dtype=np.int32)
        intervals = np.array([[0, 100]], dtype=np.int32)
        vcz = make_ancestor_vcz(gt, pos, alleles, times, focal, intervals)
        fp = vcz["sample_focal_positions"][:]
        assert fp[0, 0] == 20
        assert fp[0, 1] == 30


# ---------------------------------------------------------------------------
# ts_to_sample_vcz
# ---------------------------------------------------------------------------


def _sim_ts(n=4, seq_len=10_000, seed=1):
    ts = msprime.sim_ancestry(n, sequence_length=seq_len, random_seed=seed)
    return msprime.sim_mutations(ts, rate=1e-3, random_seed=seed)


class TestTsToSampleVcz:
    def test_returns_zarr_group(self):
        ts = _sim_ts()
        assert isinstance(ts_to_sample_vcz(ts), zarr.Group)

    def test_call_genotype_shape_diploid(self):
        ts = _sim_ts(n=4)
        vcz = ts_to_sample_vcz(ts)
        num_sites = ts.num_sites
        num_individuals = ts.num_individuals
        assert vcz["call_genotype"].shape == (num_sites, num_individuals, 2)

    def test_positions_match_ts(self):
        ts = _sim_ts()
        vcz = ts_to_sample_vcz(ts)
        expected = np.array([int(s.position) for s in ts.sites()], dtype=np.int32)
        np.testing.assert_array_equal(vcz["variant_position"][:], expected)

    def test_contig_length_matches_ts(self):
        ts = _sim_ts(seq_len=50_000)
        vcz = ts_to_sample_vcz(ts)
        assert vcz["contig_length"][:][0] == 50_000

    def test_sample_ids_tsk_prefix(self):
        ts = _sim_ts(n=3)
        vcz = ts_to_sample_vcz(ts)
        ids = list(vcz["sample_id"][:])
        assert ids == ["tsk_0", "tsk_1", "tsk_2"]

    def test_ancestral_allele_ref(self):
        ts = _sim_ts()
        vcz = ts_to_sample_vcz(ts, ancestral_allele="REF")
        anc = vcz["variant_ancestral_allele"][:]
        alleles = vcz["variant_allele"][:]
        for i in range(len(anc)):
            assert anc[i] == alleles[i, 0]

    def test_ancestral_allele_ancestral(self):
        ts = _sim_ts()
        vcz = ts_to_sample_vcz(ts, ancestral_allele="ANCESTRAL")
        anc = vcz["variant_ancestral_allele"][:]
        expected = np.array([s.ancestral_state for s in ts.sites()])
        np.testing.assert_array_equal(anc, expected)

    def test_no_sites_raises(self):
        ts = msprime.sim_ancestry(2, sequence_length=1000, random_seed=1)
        with pytest.raises(ValueError, match="no sites"):
            ts_to_sample_vcz(ts)

    def test_custom_contig_id(self):
        ts = _sim_ts()
        vcz = ts_to_sample_vcz(ts, contig_id="chr1")
        assert vcz["contig_id"][:][0] == "chr1"

    def test_genotypes_consistent_with_ts(self):
        ts = _sim_ts(n=2)
        vcz = ts_to_sample_vcz(ts)
        gt = vcz["call_genotype"][:]
        for s_idx, variant in enumerate(ts.variants()):
            # tskit gives flat haplotypes; reshape to (num_individuals, ploidy)
            expected = variant.genotypes.reshape(ts.num_individuals, 2)
            np.testing.assert_array_equal(gt[s_idx], expected)
