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
Ancestor generation: builds the ancestor VCZ store from a samples VCZ store.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import zarr

import _tsinfer

from . import vcz as vcz_mod
from .config import AncestorsConfig, AncestralState, Source


@dataclass
class InferenceSites:
    positions: np.ndarray
    alleles: np.ndarray
    ancestral_allele_index: np.ndarray
    site_mask: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assign_site_intervals(positions, intervals):
    """
    For each position return the index of the interval that contains it.

    Intervals are half-open [start, end) pairs.  Returns -1 for any position
    that is not inside any interval (should not happen for inference sites).
    """
    result = np.full(len(positions), -1, dtype=np.int32)
    for i_idx in range(len(intervals)):
        s, e = int(intervals[i_idx, 0]), int(intervals[i_idx, 1])
        mask = (positions >= s) & (positions < e)
        result[mask] = i_idx
    return result


def _compute_site_stats(store, inf_sites, sample_include, num_haplotypes):
    """
    Pass 1: iterate over inference-site genotypes, compute derived genotype
    stats for each site, filter out fixed/all-missing.

    Returns (keep_mask, times) where:
    - keep_mask: boolean array of length len(inf_sites.positions)
    - times: float64 array of length sum(keep_mask)
    """
    anc_indices = inf_sites.ancestral_allele_index
    num_inf_sites = len(inf_sites.site_mask)

    keep_mask = np.zeros(num_inf_sites, dtype=bool)
    times_list = []

    for i, gt_row in enumerate(
        vcz_mod.iter_genotypes(store, inf_sites.site_mask, sample_include)
    ):
        anc_idx = int(anc_indices[i])

        is_missing = gt_row < 0
        is_ancestral = gt_row == anc_idx
        derived_gt = np.where(
            is_missing,
            np.int8(-1),
            np.where(is_ancestral, np.int8(0), np.int8(1)),
        ).astype(np.int8)

        derived_count = int(np.sum(derived_gt == 1))
        n_non_missing = int(np.sum(derived_gt >= 0))

        if n_non_missing == 0 or derived_count == 0 or derived_count == n_non_missing:
            continue

        keep_mask[i] = True
        times_list.append(derived_count / num_haplotypes)

    times = (
        np.array(times_list, dtype=np.float64)
        if times_list
        else np.zeros(0, dtype=np.float64)
    )
    return keep_mask, times


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_inference_sites(
    store: zarr.Group,
    site_mask_arr,
    ancestral_state: AncestralState | None,
):
    """
    Return an InferenceSites dataclass for sites that pass the mask and have
    a valid ancestral allele.
    """
    positions = np.asarray(store["variant_position"][:], dtype=np.int32)
    alleles = np.asarray(store["variant_allele"][:])
    num_sites = len(positions)

    if ancestral_state is not None:
        ann_store = vcz_mod.open_store(ancestral_state.path)
        ann_positions = np.asarray(ann_store["variant_position"][:])
        ann_values = np.asarray(ann_store[ancestral_state.field][:])
        # NOTE: potential perf bottleneck.
        ann_lookup = {
            int(k): str(v) for k, v in zip(ann_positions.tolist(), ann_values.tolist())
        }
        anc_str = np.array(
            [ann_lookup.get(int(p), "") for p in positions.tolist()], dtype=object
        )
    else:
        anc_str = np.asarray(store["variant_ancestral_allele"][:])

    # Find ancestral allele index at each site (-1 if not found)
    anc_index = np.full(num_sites, -1, dtype=np.int8)
    for i in range(num_sites):
        for j, a in enumerate(alleles[i].tolist()):
            if a is not None and a != "" and a == str(anc_str[i]):
                anc_index[i] = j
                break

    # Build inclusion mask
    include = np.ones(num_sites, dtype=bool)
    if site_mask_arr is not None:
        include &= ~np.asarray(site_mask_arr, dtype=bool)
    include &= anc_index >= 0

    sel = np.where(include)[0]
    return InferenceSites(
        positions=positions[sel],
        alleles=alleles[sel],
        ancestral_allele_index=anc_index[sel],
        site_mask=sel.astype(np.int32),
    )


def compute_sequence_intervals(
    positions,
    sequence_length: int,
    max_gap_length: int,
) -> np.ndarray:
    """
    Return an (n_intervals, 2) int32 array of [start, end) pairs for regions
    containing inference sites, splitting where consecutive positions are more
    than max_gap_length apart.

    start is the genomic position of the first inference site in the interval;
    end is the last position + 1 (Python half-open convention).
    """
    positions = np.asarray(positions, dtype=np.int32)
    n = len(positions)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int32)

    intervals = []
    start_pos = int(positions[0])
    prev_pos = int(positions[0])

    for i in range(1, n):
        cur_pos = int(positions[i])
        if cur_pos - prev_pos > max_gap_length:
            intervals.append([start_pos, prev_pos + 1])
            start_pos = cur_pos
        prev_pos = cur_pos

    intervals.append([start_pos, prev_pos + 1])
    return np.array(intervals, dtype=np.int32)


def infer_ancestors(
    source: Source | zarr.Group,
    cfg: AncestorsConfig,
    ancestral_state: AncestralState | None = None,
) -> zarr.Group:
    """
    Build the ancestor VCZ store from a samples VCZ store.

    Two-pass approach:
      Pass 1 — Identify inference sites and compute times (chunk-aware).
      Pass 2 — Per-interval AncestorBuilder instances; stream ancestors to output.

    No virtual root is inserted; that is the responsibility of the match step.
    Ancestors are not sorted by time.
    """
    # --- 1. Open store and resolve masks ---
    if isinstance(source, zarr.Group):
        store = source
        site_mask_spec = None
        sample_mask_spec = None
    else:
        store = vcz_mod.open_store(source.path)
        site_mask_spec = source.site_mask
        sample_mask_spec = source.sample_mask

    gt_shape = store["call_genotype"].shape
    num_total_sites, num_samples, ploidy = gt_shape

    sample_mask_arr = vcz_mod.resolve_field(
        store, sample_mask_spec, "sample_id", num_samples
    )
    if sample_mask_arr is not None:
        sample_include = ~np.asarray(sample_mask_arr, dtype=bool)
        num_samples_used = int(np.sum(sample_include))
    else:
        sample_include = None
        num_samples_used = num_samples
    num_haplotypes = num_samples_used * ploidy

    site_mask_arr = vcz_mod.resolve_field(
        store, site_mask_spec, "variant_position", num_total_sites
    )

    # --- 2. Compute inference sites ---
    inf_sites = compute_inference_sites(store, site_mask_arr, ancestral_state)

    # --- 3. Pass 1: compute site stats ---
    keep_mask, times = _compute_site_stats(
        store, inf_sites, sample_include, num_haplotypes
    )

    final_positions = inf_sites.positions[keep_mask]
    final_alleles = inf_sites.alleles[keep_mask]
    final_anc_indices = inf_sites.ancestral_allele_index[keep_mask]
    final_orig_indices = inf_sites.site_mask[keep_mask]

    num_inf = len(final_positions)

    if num_inf == 0:
        seq_len = vcz_mod.sequence_length(store)
        seq_intervals = compute_sequence_intervals(
            final_positions, seq_len, cfg.max_gap_length
        )
        return vcz_mod.write_empty_ancestor_vcz(seq_intervals, store=cfg.path)

    # --- 4. Compute sequence intervals ---
    seq_len = vcz_mod.sequence_length(store)
    seq_intervals = compute_sequence_intervals(
        final_positions, seq_len, cfg.max_gap_length
    )

    # --- 5. Pass 2: per-interval ancestor building ---
    site_interval_idx = _assign_site_intervals(final_positions, seq_intervals)

    writer = vcz_mod.AncestorWriter(
        num_inf,
        final_positions,
        final_alleles,
        final_anc_indices,
        seq_intervals,
        store=cfg.path,
        samples_chunk_size=cfg.samples_chunk_size,
        variants_chunk_size=cfg.variants_chunk_size,
    )

    for i_idx in range(len(seq_intervals)):
        in_interval = site_interval_idx == i_idx
        if not np.any(in_interval):
            continue

        local_mask = np.where(in_interval)[0]
        n_local = len(local_mask)

        # Create per-interval AncestorBuilder, streaming genotypes from store
        local_orig_indices = final_orig_indices[local_mask]
        local_anc_indices = final_anc_indices[local_mask]
        local_times = times[local_mask]
        n_ab_sites = n_local + 1  # +1 for terminal
        ab = _tsinfer.AncestorBuilder(num_samples=num_haplotypes, max_sites=n_ab_sites)

        for j, gt_row in enumerate(
            vcz_mod.iter_genotypes(store, local_orig_indices, sample_include)
        ):
            anc_idx = int(local_anc_indices[j])
            is_missing = gt_row < 0
            is_ancestral = gt_row == anc_idx
            derived_gt = np.where(
                is_missing,
                np.int8(-1),
                np.where(is_ancestral, np.int8(0), np.int8(1)),
            ).astype(np.int8)
            ab.add_site(time=float(local_times[j]), genotypes=derived_gt)
        ab.add_terminal_site()

        # Generate ancestors for this interval
        for time, focal_sites in ab.ancestor_descriptors():
            focal_arr = np.asarray(focal_sites, dtype=np.int32)
            a = np.full(n_ab_sites, np.int8(-1), dtype=np.int8)
            ab.make_ancestor(focal_arr.tolist(), a)
            a = a[:n_local]  # trim terminal

            non_missing = np.where(a != -1)[0]
            if len(non_missing) == 0:
                continue

            # Map local haplotype to global inference site coordinates
            global_hap = np.full(num_inf, np.int8(-1), dtype=np.int8)
            global_hap[local_mask] = a

            start_pos = int(final_positions[local_mask[non_missing[0]]])
            end_pos = int(final_positions[local_mask[non_missing[-1]]])
            focal_global = local_mask[focal_arr]
            focal_pos = final_positions[focal_global]

            writer.add_ancestor(float(time), global_hap, focal_pos, start_pos, end_pos)

    return writer.finalize()
