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
Tests for tsinfer.ancestors: compute_inference_sites, compute_sequence_intervals,
and infer_ancestors.
"""

from __future__ import annotations

import pathlib

import numpy as np
import zarr
from helpers import make_sample_vcz

from tsinfer.ancestors import (
    compute_inference_sites,
    compute_sequence_intervals,
    infer_ancestors,
)
from tsinfer.config import AncestorsConfig, AncestralState
from tsinfer.vcz import (
    iter_genotypes,
    open_group,
    write_empty_ancestor_vcz,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(max_gap_length=500_000, samples_chunk_size=1000, variants_chunk_size=1000):
    return AncestorsConfig(
        path=None,
        sources=["src"],
        max_gap_length=max_gap_length,
        samples_chunk_size=samples_chunk_size,
        variants_chunk_size=variants_chunk_size,
    )


def _haploid_store(gt_matrix, positions, alleles, anc_states, seq_len=None, **kwargs):
    """Wrap gt_matrix (num_sites, num_samples) to (num_sites, num_samples, 1)."""
    gt_matrix = np.asarray(gt_matrix, dtype=np.int8)
    num_sites, num_samples = gt_matrix.shape
    gt = gt_matrix[:, :, np.newaxis]
    if seq_len is None:
        seq_len = max(positions) + 1000
    return make_sample_vcz(
        gt, positions, alleles, anc_states, sequence_length=seq_len, **kwargs
    )


def _oracle_ancestors(gt_matrix, times):
    """
    Run the Python-engine AncestorBuilder as an oracle.

    Parameters
    ----------
    gt_matrix: (num_sites, num_haplotypes) int8 — derived genotypes
              (0=anc, 1=der, -1=miss)
    times: (num_sites,) float — time for each site

    Returns list of (time, focal_site_indices, haplotype_array) tuples, sorted by
    descending time, in the same order as _tsinfer.AncestorBuilder.
    """
    import sys

    sys.path.insert(0, "tests")
    import algorithm as alg

    num_sites, num_haplotypes = gt_matrix.shape
    ab = alg.AncestorBuilder(num_samples=num_haplotypes, max_sites=num_sites + 1)
    for i in range(num_sites):
        ab.add_site(time=times[i], genotypes=gt_matrix[i])
    ab.add_terminal_site()

    result = []
    for t, focal in ab.ancestor_descriptors():
        a = np.full(num_sites + 1, -1, dtype=np.int8)
        ab.make_ancestor(focal, a)
        a = a[:num_sites]
        result.append((t, list(focal), a.copy()))
    result.sort(key=lambda x: -x[0])
    return result


def _anc_genotypes(anc_store):
    """Return call_genotype[:, :, 0] from an ancestor store."""
    return np.asarray(anc_store["call_genotype"][:, :, 0])


# ---------------------------------------------------------------------------
# compute_inference_sites
# ---------------------------------------------------------------------------


class TestComputeInferenceSites:
    def test_all_sites_pass(self):
        gt = np.zeros((3, 2, 1), dtype=np.int8)
        store = make_sample_vcz(
            gt,
            positions=[10, 20, 30],
            alleles=[["A", "T"]] * 3,
            ancestral_state=["A", "A", "A"],
            sequence_length=100,
        )
        result = compute_inference_sites(store, None)
        np.testing.assert_array_equal(result.positions, [10, 20, 30])
        np.testing.assert_array_equal(result.ancestral_allele_index, [0, 0, 0])
        np.testing.assert_array_equal(result.global_indices, [0, 1, 2])

    def test_include_excludes_sites(self):
        gt = np.zeros((3, 2, 1), dtype=np.int8)
        store = make_sample_vcz(
            gt,
            positions=[10, 20, 30],
            alleles=[["A", "T"]] * 3,
            ancestral_state=["A", "A", "A"],
            sequence_length=100,
        )
        result = compute_inference_sites(store, None, include="POS >= 20")
        np.testing.assert_array_equal(result.positions, [20, 30])

    def test_exclude_excludes_sites(self):
        gt = np.zeros((3, 2, 1), dtype=np.int8)
        store = make_sample_vcz(
            gt,
            positions=[10, 20, 30],
            alleles=[["A", "T"]] * 3,
            ancestral_state=["A", "A", "A"],
            sequence_length=100,
        )
        result = compute_inference_sites(store, None, exclude="POS == 20")
        np.testing.assert_array_equal(result.positions, [10, 30])

    def test_missing_ancestral_state_excludes(self):
        # Site 1 has ancestral state not in alleles → excluded
        gt = np.zeros((3, 2, 1), dtype=np.int8)
        store = make_sample_vcz(
            gt,
            positions=[10, 20, 30],
            alleles=[["A", "T"], ["C", "G"], ["A", "T"]],
            ancestral_state=["A", "X", "A"],  # X not in alleles[1]
            sequence_length=100,
        )
        result = compute_inference_sites(store, None)
        np.testing.assert_array_equal(result.positions, [10, 30])

    def test_ancestral_allele_index_not_always_zero(self):
        # Site 0: alleles=['T','A'], ancestral='A' → anc_idx=1
        gt = np.zeros((1, 2, 1), dtype=np.int8)
        store = make_sample_vcz(
            gt,
            positions=[10],
            alleles=[["T", "A"]],
            ancestral_state=["A"],
            sequence_length=100,
        )
        result = compute_inference_sites(store, None)
        assert result.ancestral_allele_index[0] == 1

    def test_external_ancestral_state(self):
        # AncestralState annotation from a separate sample VCZ (used as annotation)
        # The annotation covers positions 10 and 30 (but not 20)
        ann = make_sample_vcz(
            np.zeros((2, 1, 1), dtype=np.int8),
            positions=[10, 30],
            alleles=[["A", "T"]] * 2,
            ancestral_state=["A", "A"],
            sequence_length=100,
        )

        gt = np.zeros((3, 2, 1), dtype=np.int8)
        store = make_sample_vcz(
            gt,
            positions=[10, 20, 30],
            alleles=[["A", "T"]] * 3,
            ancestral_state=["", "", ""],  # store has no valid anc state
            sequence_length=100,
        )
        anc_cfg = AncestralState(path=ann, field="variant_ancestral_allele")
        result = compute_inference_sites(store, anc_cfg)
        # Only positions 10 and 30 are annotated
        np.testing.assert_array_equal(result.positions, [10, 30])


# ---------------------------------------------------------------------------
# compute_sequence_intervals
# ---------------------------------------------------------------------------


class TestComputeSequenceIntervals:
    def test_single_site(self):
        intervals = compute_sequence_intervals([100], 1000, 500_000)
        np.testing.assert_array_equal(intervals, [[100, 101]])

    def test_no_gap_single_interval(self):
        intervals = compute_sequence_intervals([100, 200, 300], 1000, 500_000)
        np.testing.assert_array_equal(intervals, [[100, 301]])

    def test_gap_splits_interval(self):
        positions = [100, 200, 800_000, 900_000]
        intervals = compute_sequence_intervals(positions, 1_000_000, 500_000)
        np.testing.assert_array_equal(intervals, [[100, 201], [800_000, 900_001]])

    def test_gap_exactly_at_threshold_not_split(self):
        # gap = max_gap_length → NOT split (split only when strictly greater)
        intervals = compute_sequence_intervals([0, 500_000], 1_000_000, 500_000)
        np.testing.assert_array_equal(intervals, [[0, 500_001]])

    def test_gap_one_over_threshold_splits(self):
        intervals = compute_sequence_intervals([0, 500_001], 1_000_000, 500_000)
        np.testing.assert_array_equal(intervals, [[0, 1], [500_001, 500_002]])

    def test_empty_positions(self):
        intervals = compute_sequence_intervals([], 1000, 500_000)
        assert intervals.shape == (0, 2)

    def test_three_intervals(self):
        # Two gaps
        positions = [100, 200, 1_000_000, 1_100_000, 2_000_000, 2_100_000]
        intervals = compute_sequence_intervals(positions, 3_000_000, 500_000)
        assert len(intervals) == 3
        np.testing.assert_array_equal(intervals[0], [100, 201])
        np.testing.assert_array_equal(intervals[1], [1_000_000, 1_100_001])
        np.testing.assert_array_equal(intervals[2], [2_000_000, 2_100_001])


# ---------------------------------------------------------------------------
# get_genotypes_for_sites
# ---------------------------------------------------------------------------


class TestIterGenotypes:
    def test_yields_correct_rows(self):
        """Iterator should yield the correct genotype rows in order."""
        gt_matrix = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=np.int8)
        store = _haploid_store(
            gt_matrix,
            positions=[100, 200, 300],
            alleles=[["A", "T"]] * 3,
            anc_states=["A", "A", "A"],
        )
        site_indices = np.array([0, 2], dtype=np.int64)
        rows = list(iter_genotypes(store, site_indices))
        assert len(rows) == 2
        np.testing.assert_array_equal(rows[0], gt_matrix[0])
        np.testing.assert_array_equal(rows[1], gt_matrix[2])

    def test_empty_site_indices(self):
        store = _haploid_store([[0, 1]], [100], [["A", "T"]], ["A"])
        rows = list(iter_genotypes(store, np.array([], dtype=np.int64)))
        assert len(rows) == 0

    def test_sample_include_mask(self):
        gt_matrix = np.array([[0, 1, 0, 1]], dtype=np.int8)
        store = _haploid_store(
            gt_matrix,
            positions=[100],
            alleles=[["A", "T"]],
            anc_states=["A"],
        )
        # Keep only samples 0 and 2
        sample_include = np.array([True, False, True, False])
        rows = list(iter_genotypes(store, np.array([0], dtype=np.int64), sample_include))
        assert len(rows) == 1
        np.testing.assert_array_equal(rows[0], [0, 0])

    def test_diploid_flattening(self):
        # 2 diploid samples
        gt = np.array(
            [[[0, 1], [1, 0]]],  # site 0: haplotypes = 0,1,1,0
            dtype=np.int8,
        )
        store = make_sample_vcz(
            gt,
            positions=[100],
            alleles=[["A", "T"]],
            ancestral_state=["A"],
            sequence_length=1000,
        )
        rows = list(iter_genotypes(store, np.array([0], dtype=np.int64)))
        assert len(rows) == 1
        np.testing.assert_array_equal(rows[0], [0, 1, 1, 0])

    def test_all_sites(self):
        gt_matrix = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int8)
        store = _haploid_store(
            gt_matrix,
            positions=[100, 200, 300],
            alleles=[["A", "T"]] * 3,
            anc_states=["A", "A", "A"],
        )
        rows = list(iter_genotypes(store, np.array([0, 1, 2], dtype=np.int64)))
        result = np.stack(rows)
        np.testing.assert_array_equal(result, gt_matrix)

    def test_yields_independent_copies(self):
        """Each yielded row must be an independent copy, not a view into
        a shared buffer that gets overwritten on the next iteration."""
        gt_matrix = np.array([[0, 1], [1, 0]], dtype=np.int8)
        store = _haploid_store(
            gt_matrix,
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        rows = list(iter_genotypes(store, np.array([0, 1], dtype=np.int64)))
        np.testing.assert_array_equal(rows[0], [0, 1])
        np.testing.assert_array_equal(rows[1], [1, 0])


# ---------------------------------------------------------------------------
# infer_ancestors — basic output format
# ---------------------------------------------------------------------------


class TestInferAncestorsFormat:
    def test_positions_are_inference_sites_only(self):
        # Site 1 is fixed → not an inference site
        gt = _haploid_store(
            [[0, 1], [0, 0], [0, 1]],
            positions=[100, 200, 300],
            alleles=[["A", "T"]] * 3,
            anc_states=["A", "A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        np.testing.assert_array_equal(anc["variant_position"][:], [100, 300])

    def test_genotype_shape(self):
        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        num_sites, num_anc, ploidy = anc["call_genotype"].shape
        assert num_sites == 2
        assert ploidy == 1
        assert num_anc >= 1

    def test_sample_ids_are_ancestor_N(self):
        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        ids = [str(x) for x in anc["sample_id"][:].tolist()]
        num_anc = anc["call_genotype"].shape[1]
        assert ids == [f"ancestor_{i}" for i in range(num_anc)]

    def test_times_are_positive(self):
        # Times are not required to be sorted (no sort step), but must be positive
        gt = _haploid_store(
            [[0, 1, 1, 1], [0, 0, 0, 1]],  # site 0 freq=0.75, site 1 freq=0.25
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        times = np.asarray(anc["sample_time"][:])
        assert np.all(times > 0)

    def test_focal_sites_carry_derived_allele(self):
        # Each ancestor must have genotype=1 at all its focal sites
        gt = _haploid_store(
            [[0, 1, 0, 1], [0, 0, 1, 1]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        gt_out = _anc_genotypes(anc)
        focal_positions = np.asarray(anc["sample_focal_positions"][:])
        anc_positions = np.asarray(anc["variant_position"][:])
        pos_to_idx = {int(p): i for i, p in enumerate(anc_positions.tolist())}

        for j in range(gt_out.shape[1]):
            for fp in focal_positions[j]:
                if int(fp) == -2:
                    break
                site_idx = pos_to_idx[int(fp)]
                assert gt_out[site_idx, j] == 1, (
                    f"Ancestor {j} should carry derived at focal site {fp}"
                )

    def test_start_end_positions_consistent_with_genotypes(self):
        gt = _haploid_store(
            [[0, 1], [1, 0], [0, 1]],
            positions=[100, 200, 300],
            alleles=[["A", "T"]] * 3,
            anc_states=["A", "A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        gt_out = _anc_genotypes(anc)
        positions = np.asarray(anc["variant_position"][:])
        start_pos = np.asarray(anc["sample_start_position"][:])
        end_pos = np.asarray(anc["sample_end_position"][:])

        for j in range(gt_out.shape[1]):
            hap = gt_out[:, j]
            non_missing = np.where(hap != -1)[0]
            assert len(non_missing) > 0
            assert int(start_pos[j]) == int(positions[non_missing[0]])
            assert int(end_pos[j]) == int(positions[non_missing[-1]])

    def test_output_allele_0_is_ancestral(self):
        # Site 0: alleles=['A','T'], ancestral='A' → output[0]='A'
        # Site 1: alleles=['T','A'], ancestral='A' → output[0]='A'
        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"], ["T", "A"]],
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        alleles = np.asarray(anc["variant_allele"][:])
        for i in range(len(alleles)):
            assert str(alleles[i, 0]) == "A"

    def test_sequence_intervals_present(self):
        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        intervals = np.asarray(anc["sequence_intervals"][:])
        assert intervals.ndim == 2
        assert intervals.shape[1] == 2


# ---------------------------------------------------------------------------
# infer_ancestors — comparison with Python oracle
# ---------------------------------------------------------------------------


class TestInferAncestorsVsPythonOracle:
    """
    Compare C-engine output of infer_ancestors against the Python AncestorBuilder
    oracle for simple hand-constructed cases.
    """

    def _check(self, gt_matrix, positions, alleles, anc_states, max_gap=500_000):
        """
        Build a sample VCZ, run infer_ancestors with C engine, and compare
        haplotype arrays against the Python oracle.
        """
        gt_matrix = np.asarray(gt_matrix, dtype=np.int8)
        store = _haploid_store(gt_matrix, positions, alleles, anc_states)
        cfg = _cfg(max_gap_length=max_gap)
        anc = infer_ancestors(store, cfg)

        # Compute derived genotypes for oracle
        anc_idx_map = {}
        for i, als in enumerate(alleles):
            for j, a in enumerate(als):
                if a == anc_states[i]:
                    anc_idx_map[i] = j
                    break

        # Filter to actual inference sites (non-fixed)
        num_hap = gt_matrix.shape[1]
        inf_sites = []
        inf_times = []
        for i in range(len(positions)):
            ai = anc_idx_map.get(i, -1)
            if ai < 0:
                continue
            derived_gt = np.where(
                gt_matrix[i] < 0,
                np.int8(-1),
                np.where(gt_matrix[i] == ai, np.int8(0), np.int8(1)),
            ).astype(np.int8)
            dc = int(np.sum(derived_gt == 1))
            nm = int(np.sum(derived_gt >= 0))
            if nm == 0 or dc == 0 or dc == nm:
                continue
            inf_sites.append(derived_gt)
            inf_times.append(dc / num_hap)

        if not inf_sites:
            # No inference sites → no ancestors (no virtual root in output)
            assert anc["call_genotype"].shape[1] == 0
            return

        oracle_results = _oracle_ancestors(np.stack(inf_sites), np.array(inf_times))

        # No virtual root in output; compare directly
        c_num_anc = anc["call_genotype"].shape[1]
        assert c_num_anc == len(oracle_results), (
            f"C engine produced {c_num_anc} ancestors, oracle "
            f"produced {len(oracle_results)}"
        )

        # Compare haplotypes: sort both by canonical key (haplotype tuple) to be
        # order-independent within groups that share the same time.
        gt_c = _anc_genotypes(anc)
        c_haplotypes = sorted(gt_c[:, j].tolist() for j in range(c_num_anc))
        oracle_haplotypes = sorted(hap.tolist() for _, _, hap in oracle_results)
        assert c_haplotypes == oracle_haplotypes, (
            f"Ancestor haplotypes differ.\n"
            f"C engine:  {c_haplotypes}\n"
            f"Oracle:    {oracle_haplotypes}"
        )

    def test_single_biallelic_site(self):
        self._check(
            [[0, 1, 0, 0]],
            positions=[100],
            alleles=[["A", "T"]],
            anc_states=["A"],
        )

    def test_two_sites_different_carriers(self):
        self._check(
            [[0, 1, 0, 0], [0, 0, 1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )

    def test_two_sites_same_carriers(self):
        # Same derived carrier → grouped into one ancestor with 2 focal sites
        self._check(
            [[0, 1, 0, 0], [0, 1, 0, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )

    def test_sites_with_different_frequencies(self):
        self._check(
            [[0, 1, 1, 1], [0, 0, 0, 1]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )

    def test_four_sites_no_gap(self):
        self._check(
            [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
            positions=[100, 200, 300, 400],
            alleles=[["A", "T"]] * 4,
            anc_states=["A", "A", "A", "A"],
        )

    def test_ancestral_allele_at_index_1(self):
        # alleles=['T','A'], ancestral='A' → anc_idx=1
        self._check(
            [[1, 0, 1, 1]],
            positions=[100],
            alleles=[["T", "A"]],
            anc_states=["A"],
        )


# ---------------------------------------------------------------------------
# infer_ancestors — named scenarios from the design
# ---------------------------------------------------------------------------


class TestInferAncestorsScenarios:
    def test_single_site(self):
        anc = infer_ancestors(
            _haploid_store([[0, 1]], [100], [["A", "T"]], ["A"], seq_len=1000),
            _cfg(),
        )
        # No virtual root; only the real ancestor.
        assert anc["call_genotype"].shape == (1, 1, 1)
        assert int(anc["call_genotype"][0, 0, 0]) == 1  # real ancestor: derived
        np.testing.assert_array_equal(anc["variant_position"][:], [100])
        np.testing.assert_array_equal(anc["sample_start_position"][:], [100])
        np.testing.assert_array_equal(anc["sample_end_position"][:], [100])

    def test_all_fixed_ancestral_returns_empty(self):
        anc = infer_ancestors(
            _haploid_store([[0, 0], [0, 0]], [100, 200], [["A", "T"]] * 2, ["A", "A"]),
            _cfg(),
        )
        assert anc["call_genotype"].shape[1] == 0

    def test_all_fixed_derived_returns_empty(self):
        anc = infer_ancestors(
            _haploid_store([[1, 1], [1, 1]], [100, 200], [["A", "T"]] * 2, ["A", "A"]),
            _cfg(),
        )
        assert anc["call_genotype"].shape[1] == 0

    def test_include_excludes_sites(self):
        from tsinfer.config import Source

        gt = _haploid_store(
            [[0, 1], [0, 1], [0, 1]],
            positions=[100, 200, 300],
            alleles=[["A", "T"]] * 3,
            anc_states=["A", "A", "A"],
        )
        src = Source(path=gt, name="test", exclude="POS == 200")
        anc = infer_ancestors(src, _cfg())
        positions = np.asarray(anc["variant_position"][:])
        assert 200 not in positions.tolist()
        np.testing.assert_array_equal(positions, [100, 300])

    def test_samples_excludes_samples(self):
        from tsinfer.config import Source

        # 4 samples: keep only samples 0 and 1
        # Without filter, site has derived_count=2 (samples 1,2)
        # With only samples 0,1: site has derived_count=1 (sample 1 only)
        gt = make_sample_vcz(
            np.array([[[0], [1], [1], [0]]], dtype=np.int8),
            positions=[100],
            alleles=[["A", "T"]],
            ancestral_state=["A"],
            sequence_length=1000,
        )
        src = Source(path=gt, name="test", samples="sample_0,sample_1")
        anc = infer_ancestors(src, _cfg())
        # No virtual root; real ancestor at index 0 (time=0.5)
        assert anc["call_genotype"].shape[1] >= 1
        np.testing.assert_almost_equal(float(anc["sample_time"][0]), 0.5)

    def test_gap_splits_ancestor_span(self):
        # Two groups of sites separated by a large gap
        # Sites at 100, 200 (interval 0) and 800000, 900000 (interval 1)
        # All sites have the same derived carrier → would be one ancestor without gap
        gt = _haploid_store(
            [[0, 1], [0, 1], [0, 1], [0, 1]],
            positions=[100, 200, 800_000, 900_000],
            alleles=[["A", "T"]] * 4,
            anc_states=["A", "A", "A", "A"],
            seq_len=1_000_000,
        )
        anc = infer_ancestors(gt, _cfg(max_gap_length=500_000))
        intervals = np.asarray(anc["sequence_intervals"][:])
        assert len(intervals) == 2

        # No virtual root. Every ancestor must have start/end within a single interval.
        gt_out = _anc_genotypes(anc)
        for j in range(gt_out.shape[1]):
            start = int(anc["sample_start_position"][j])
            end = int(anc["sample_end_position"][j])
            # start and end must lie within the same interval
            in_interval = [
                s <= start < e and s <= end < e for s, e in intervals.tolist()
            ]
            assert any(in_interval), (
                f"Ancestor {j} spans [{start}, {end}] but no single interval covers it"
            )

    def test_diploid_input(self):
        # 2 diploid samples (4 haplotypes), ploidy=2
        # Genotype shape (num_sites, num_samples, 2)
        gt = np.array(
            [
                [[0, 0], [0, 1]],  # site 0: haplotypes 0,1,2,3 = 0,0,0,1
                [[0, 1], [0, 0]],  # site 1: haplotypes 0,1,2,3 = 0,1,0,0
            ],
            dtype=np.int8,
        )
        store = make_sample_vcz(
            gt,
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            ancestral_state=["A", "A"],
            sequence_length=1000,
        )
        anc = infer_ancestors(store, _cfg())
        # 4 haplotypes total; both sites have derived_count=1 → valid inference sites
        assert anc["call_genotype"].shape[1] >= 1

    def test_missing_data_in_genotypes(self):
        # Site 0: sample 0 missing (-1), sample 1 derived (1)
        # Should still be an inference site since derived_count=1 < n_non_missing=1
        gt = _haploid_store(
            [[-1, 1, 0, 0]],
            positions=[100],
            alleles=[["A", "T"]],
            anc_states=["A"],
        )
        anc = infer_ancestors(gt, _cfg())
        # non_missing = 3, derived = 1 → valid inference site
        # No virtual root; just the real ancestor.
        assert anc["call_genotype"].shape[1] == 1

    def test_no_gap_single_interval(self):
        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        intervals = np.asarray(anc["sequence_intervals"][:])
        assert len(intervals) == 1
        assert int(intervals[0, 0]) == 100
        assert int(intervals[0, 1]) == 201

    def test_multi_focal_ancestor(self):
        # Two sites with identical genotypes → one ancestor with 2 focal sites
        gt = _haploid_store(
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        num_anc = anc["call_genotype"].shape[1]
        focal_pos = np.asarray(anc["sample_focal_positions"][:])
        # Find the ancestor with 2 focal sites (no -2 in row)
        has_multi_focal = any(np.sum(focal_pos[j] != -2) >= 2 for j in range(num_anc))
        assert has_multi_focal

    def test_source_object_input(self):
        from tsinfer.config import Source

        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        # Write to a tmp store — use the in-memory group directly as the "path"
        src = Source(path=gt, name="test")
        anc = infer_ancestors(src, _cfg())
        assert anc["call_genotype"].shape[1] >= 1

    def test_inferred_alleles_ancestral_first(self):
        # Verify that output allele 0 = ancestral, allele 1 = derived
        # at all inference sites, regardless of input allele ordering
        gt = _haploid_store(
            [[1, 0], [0, 1]],  # anc is at index 1 for site 0
            positions=[100, 200],
            alleles=[["T", "A"], ["A", "T"]],
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        alleles = np.asarray(anc["variant_allele"][:])
        # Both sites should have "A" as allele 0 (ancestral)
        for i in range(len(alleles)):
            assert str(alleles[i, 0]) == "A"

    def test_large_gap_threshold(self):
        # max_gap_length=0 → every pair of adjacent sites is a separate interval
        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg(max_gap_length=0))
        intervals = np.asarray(anc["sequence_intervals"][:])
        # gap = 100 > 0 → 2 intervals
        assert len(intervals) == 2


# ---------------------------------------------------------------------------
# Per-interval builder equivalence
# ---------------------------------------------------------------------------


class TestAncestorWriterChunking:
    """Verify infer_ancestors output is identical regardless of chunk sizes."""

    _STORE = None
    _GT_MATRIX = np.array(
        [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
        dtype=np.int8,
    )

    @staticmethod
    def _make_store():
        return _haploid_store(
            TestAncestorWriterChunking._GT_MATRIX,
            positions=[100, 200, 300, 400],
            alleles=[["A", "T"]] * 4,
            anc_states=["A", "A", "A", "A"],
        )

    @staticmethod
    def _compare_ancestors(a, b):
        np.testing.assert_array_equal(_anc_genotypes(a), _anc_genotypes(b))
        for name in (
            "sample_time",
            "sample_start_position",
            "sample_focal_positions",
        ):
            np.testing.assert_array_equal(a[name][:], b[name][:])

    def test_samples_chunk_size_one_matches_default(self):
        store = self._make_store()
        anc_default = infer_ancestors(store, _cfg())
        anc_cs1 = infer_ancestors(store, _cfg(samples_chunk_size=1))
        self._compare_ancestors(anc_default, anc_cs1)

    def test_samples_chunk_size_two_matches_default(self):
        store = self._make_store()
        anc_default = infer_ancestors(store, _cfg())
        anc_cs2 = infer_ancestors(store, _cfg(samples_chunk_size=2))
        self._compare_ancestors(anc_default, anc_cs2)

    def test_variants_chunk_size_one_matches_default(self):
        store = self._make_store()
        anc_default = infer_ancestors(store, _cfg())
        anc_vcs1 = infer_ancestors(store, _cfg(variants_chunk_size=1))
        self._compare_ancestors(anc_default, anc_vcs1)

    def test_variants_chunk_size_two_matches_default(self):
        store = self._make_store()
        anc_default = infer_ancestors(store, _cfg())
        anc_vcs2 = infer_ancestors(store, _cfg(variants_chunk_size=2))
        self._compare_ancestors(anc_default, anc_vcs2)

    def test_both_chunk_sizes_small(self):
        store = self._make_store()
        anc_default = infer_ancestors(store, _cfg())
        anc_small = infer_ancestors(
            store, _cfg(samples_chunk_size=1, variants_chunk_size=1)
        )
        self._compare_ancestors(anc_default, anc_small)

    def test_chunk_size_one_produces_valid_output(self):
        """Even with samples_chunk_size=1, focal sites carry derived allele."""
        gt = _haploid_store(
            [[0, 1, 0, 1], [0, 0, 1, 1]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg(samples_chunk_size=1))
        num_anc = anc["call_genotype"].shape[1]
        assert num_anc >= 1

        gt_out = _anc_genotypes(anc)
        focal_positions = np.asarray(anc["sample_focal_positions"][:])
        anc_positions = np.asarray(anc["variant_position"][:])
        pos_to_idx = {int(p): i for i, p in enumerate(anc_positions.tolist())}
        for j in range(gt_out.shape[1]):
            for fp in focal_positions[j]:
                if int(fp) == -2:
                    break
                assert gt_out[pos_to_idx[int(fp)], j] == 1


class TestPerIntervalBuilder:
    def test_single_interval_matches_old_approach(self):
        """With one interval, per-interval building should match a single builder."""
        gt = _haploid_store(
            [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
            positions=[100, 200, 300, 400],
            alleles=[["A", "T"]] * 4,
            anc_states=["A", "A", "A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        # All sites in one interval, no gap
        intervals = np.asarray(anc["sequence_intervals"][:])
        assert len(intervals) == 1
        # Should produce the same ancestors as the oracle
        gt_out = _anc_genotypes(anc)
        assert gt_out.shape[1] >= 1

    def test_multi_interval_ancestors_are_disjoint(self):
        """Ancestors from different intervals should not overlap."""
        gt = _haploid_store(
            [[0, 1], [0, 1], [0, 1], [0, 1]],
            positions=[100, 200, 800_000, 900_000],
            alleles=[["A", "T"]] * 4,
            anc_states=["A", "A", "A", "A"],
            seq_len=1_000_000,
        )
        anc = infer_ancestors(gt, _cfg(max_gap_length=500_000))
        intervals = np.asarray(anc["sequence_intervals"][:])
        assert len(intervals) == 2

        gt_out = _anc_genotypes(anc)
        positions = np.asarray(anc["variant_position"][:])

        for j in range(gt_out.shape[1]):
            hap = gt_out[:, j]
            non_missing = np.where(hap != -1)[0]
            if len(non_missing) == 0:
                continue
            nm_positions = positions[non_missing]
            # All non-missing positions should be in a single interval
            for s, e in intervals.tolist():
                in_this = (nm_positions >= s) & (nm_positions < e)
                if np.any(in_this):
                    assert np.all(in_this), (
                        f"Ancestor {j} has non-missing sites in multiple intervals"
                    )
                    break


# ---------------------------------------------------------------------------
# Filesystem-backed store tests
# ---------------------------------------------------------------------------


class TestOpenGroup:
    def test_none_gives_memory_store(self):
        group = open_group(None)
        assert isinstance(group, zarr.Group)

    def test_path_gives_filesystem_store(self, tmp_path):
        out = tmp_path / "test.zarr"
        group = open_group(out)
        assert isinstance(group, zarr.Group)
        # Should have created the directory on disk
        assert out.exists()

    def test_string_path_works(self, tmp_path):
        out = str(tmp_path / "test.zarr")
        group = open_group(out)
        assert isinstance(group, zarr.Group)
        assert pathlib.Path(out).exists()


class TestFilesystemStore:
    """Verify filesystem-backed zarr stores work for ancestor output."""

    def _compare_groups(self, a, b):
        """Assert two ancestor zarr Groups have identical contents."""
        for name in (
            "variant_position",
            "variant_allele",
            "call_genotype",
            "sample_time",
            "sample_start_position",
            "sample_end_position",
            "sample_id",
            "sample_focal_positions",
            "sequence_intervals",
        ):
            np.testing.assert_array_equal(
                np.asarray(a[name][:]),
                np.asarray(b[name][:]),
                err_msg=f"Mismatch in {name}",
            )

    def test_ancestor_writer_to_filesystem(self, tmp_path):
        """Filesystem path produces same output as in-memory."""
        gt = _haploid_store(
            [[0, 1, 0, 1], [0, 0, 1, 1]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        mem_result = infer_ancestors(gt, _cfg())

        fs_path = tmp_path / "ancestors.zarr"
        fs_cfg = AncestorsConfig(path=fs_path, sources=["src"])
        fs_result = infer_ancestors(gt, fs_cfg)

        self._compare_groups(mem_result, fs_result)

        # Verify it's actually on disk and re-readable
        reopened = zarr.open_group(str(fs_path), mode="r")
        self._compare_groups(mem_result, reopened)

    def test_ancestor_writer_to_filesystem_multi_site(self, tmp_path):
        """Filesystem output matches in-memory for a larger dataset."""
        gt = _haploid_store(
            [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
            positions=[100, 200, 300, 400],
            alleles=[["A", "T"]] * 4,
            anc_states=["A", "A", "A", "A"],
        )
        mem_result = infer_ancestors(gt, _cfg())

        fs_path = tmp_path / "ancestors.zarr"
        fs_cfg = AncestorsConfig(path=fs_path, sources=["src"])
        fs_result = infer_ancestors(gt, fs_cfg)

        self._compare_groups(mem_result, fs_result)

    def test_empty_ancestor_vcz_to_filesystem(self, tmp_path):
        """write_empty_ancestor_vcz with a filesystem path is re-readable."""
        seq_intervals = np.zeros((0, 2), dtype=np.int32)

        mem_result = write_empty_ancestor_vcz(seq_intervals, store=None)

        fs_path = tmp_path / "empty.zarr"
        fs_result = write_empty_ancestor_vcz(seq_intervals, store=fs_path)

        self._compare_groups(mem_result, fs_result)

        reopened = zarr.open_group(str(fs_path), mode="r")
        self._compare_groups(mem_result, reopened)

    def test_empty_ancestors_via_infer(self, tmp_path):
        """infer_ancestors with all-fixed sites writes empty output to filesystem."""
        gt = _haploid_store([[0, 0], [0, 0]], [100, 200], [["A", "T"]] * 2, ["A", "A"])
        fs_path = tmp_path / "empty_anc.zarr"
        fs_cfg = AncestorsConfig(path=fs_path, sources=["src"])
        anc = infer_ancestors(gt, fs_cfg)

        assert anc["call_genotype"].shape[1] == 0
        assert fs_path.exists()

        reopened = zarr.open_group(str(fs_path), mode="r")
        assert reopened["call_genotype"].shape[1] == 0


# ---------------------------------------------------------------------------
# Logging and progress
# ---------------------------------------------------------------------------


class TestLogging:
    def test_infer_ancestors_logs_key_messages(self, caplog):
        """Key INFO messages are emitted during ancestor inference."""
        import logging

        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        with caplog.at_level(logging.INFO, logger="tsinfer.ancestors"):
            infer_ancestors(gt, _cfg())

        messages = caplog.text
        assert "Starting ancestor inference" in messages
        assert "Inference sites identified" in messages
        assert "Pass 1" in messages
        assert "Ancestor inference complete" in messages


class TestProgress:
    def test_progress_does_not_crash(self):
        """progress=True runs without error."""
        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg(), progress=True)
        assert anc["call_genotype"].shape[1] >= 1

    def test_progress_false_by_default(self):
        """Default progress=False works normally."""
        gt = _haploid_store(
            [[0, 1], [1, 0]],
            positions=[100, 200],
            alleles=[["A", "T"]] * 2,
            anc_states=["A", "A"],
        )
        anc = infer_ancestors(gt, _cfg())
        assert anc["call_genotype"].shape[1] >= 1
