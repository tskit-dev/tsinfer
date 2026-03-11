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
from zarr.core.dtype.npy.string import VariableLengthUTF8

import _tsinfer

from . import vcz as vcz_mod
from .config import AncestorsConfig, AncestralState, Source

_VLEN_STR = VariableLengthUTF8()
_ZARR_FORMAT = 2


@dataclass
class InferenceSites:
    positions: np.ndarray
    alleles: np.ndarray
    ancestral_allele_index: np.ndarray
    site_mask: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_memory_group() -> zarr.Group:
    store = zarr.storage.MemoryStore()
    return zarr.open_group(store, mode="w", zarr_format=_ZARR_FORMAT)


def _arr(group, name, data, dims):
    a = group.create_array(name, data=data)
    a.attrs["_ARRAY_DIMENSIONS"] = dims
    return a


def _str_array(group, name, data, dims):
    data = np.asarray(data)
    a = group.create_array(name, shape=data.shape, dtype=_VLEN_STR)
    a[:] = data
    a.attrs["_ARRAY_DIMENSIONS"] = dims
    return a


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
            # NOTE: not clear what `if a` is covering here, be more explicit.
            if a and a == str(anc_str[i]):
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

    Steps:
      1. Open source store, apply site_mask and sample_mask.
      2. Compute inference sites (compute_inference_sites).
      3. Compute sequence_intervals (compute_sequence_intervals).
      4. Run _tsinfer.AncestorBuilder to get raw ancestor haplotypes.
      5. Clip ancestors to their containing interval; split focal sites that
         span gap boundaries into separate ancestors per interval.
      6. Write and return the ancestor VCZ store.
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

    # NOTE: this is a perf bottleneck. The GT array is too large to load
    # fully into memory. We will need chunk-aware iteration over the sites.
    call_gt = np.asarray(store["call_genotype"][:], dtype=np.int8)
    num_total_sites, num_samples, ploidy = call_gt.shape

    # Resolve sample_mask
    sample_mask_arr = vcz_mod.resolve_field(
        store, sample_mask_spec, "sample_id", num_samples
    )
    if sample_mask_arr is not None:
        include_samples = ~np.asarray(sample_mask_arr, dtype=bool)
        call_gt = call_gt[:, include_samples, :]
        num_samples_used = int(np.sum(include_samples))
    else:
        num_samples_used = num_samples
    num_haplotypes = num_samples_used * ploidy

    # Flatten ploidy axis: (num_sites, num_haplotypes)
    call_gt_flat = call_gt.reshape(num_total_sites, num_haplotypes)

    # Resolve site_mask
    site_mask_arr = vcz_mod.resolve_field(
        store, site_mask_spec, "variant_position", num_total_sites
    )
    # NOTE: we pass the site mask and the sample mask to the VCZ aware iteration
    # function, because we can use this to avoid loading chunks we don't need.

    # --- 2. Compute inference sites ---
    inf_sites = compute_inference_sites(store, site_mask_arr, ancestral_state)
    inf_positions = inf_sites.positions
    inf_alleles = inf_sites.alleles
    anc_indices = inf_sites.ancestral_allele_index
    orig_site_indices = inf_sites.site_mask

    # --- 3. Compute sequence intervals ---
    seq_len = vcz_mod.sequence_length(store)
    seq_intervals = compute_sequence_intervals(
        inf_positions, seq_len, cfg.max_gap_length
    )

    # --- 4. Build derived genotypes and feed to AncestorBuilder ---
    # We first compute derived genotypes for all candidate inference sites,
    # then skip any that turn out to be fixed (derived_count == 0 or == num_haplotypes).

    inf_derived_gts = []  # list of (num_haplotypes,) int8 arrays
    inf_times = []
    final_positions = []
    final_alleles_list = []
    final_anc_indices = []
    final_orig_indices = []

    for i, (pos, anc_idx) in enumerate(
        zip(inf_positions.tolist(), anc_indices.tolist())
    ):
        orig_idx = int(orig_site_indices[i])
        gt_row = call_gt_flat[orig_idx]  # (num_haplotypes,)
        # 0 = ancestral, 1 = derived, -1 = missing
        # NOTE: this line is opaque - split into two statements.
        derived_gt = np.where(
            gt_row < 0, np.int8(-1), np.where(gt_row == anc_idx, np.int8(0), np.int8(1))
        ).astype(np.int8)

        derived_count = int(np.sum(derived_gt == 1))
        n_non_missing = int(np.sum(derived_gt >= 0))
        if n_non_missing == 0 or derived_count == 0 or derived_count == n_non_missing:
            # Fixed or all-missing: not an inference site
            continue

        time = derived_count / num_haplotypes
        inf_derived_gts.append(derived_gt)
        inf_times.append(time)
        final_positions.append(pos)
        final_alleles_list.append(inf_alleles[i])
        final_anc_indices.append(int(anc_idx))
        final_orig_indices.append(orig_idx)

    num_inf = len(final_positions)

    if num_inf == 0:
        # No inference sites — return an empty ancestor store
        return _write_ancestor_vcz(
            genotypes=np.zeros((0, 0, 1), dtype=np.int8),
            positions=np.zeros(0, dtype=np.int32),
            alleles=np.zeros((0, 2), dtype=object),
            times=np.zeros(0, dtype=np.float64),
            start_positions=np.zeros(0, dtype=np.int32),
            end_positions=np.zeros(0, dtype=np.int32),
            focal_positions=np.zeros((0, 1), dtype=np.int32),
            seq_intervals=seq_intervals,
        )

    inf_positions_arr = np.array(final_positions, dtype=np.int32)

    # Recompute seq_intervals now that we have the final inference sites
    seq_intervals = compute_sequence_intervals(
        inf_positions_arr, seq_len, cfg.max_gap_length
    )

    # Build AncestorBuilder (num_inf + 1 for the terminal sentinel site)
    n_ab_sites = num_inf + 1  # includes terminal
    ab = _tsinfer.AncestorBuilder(num_samples=num_haplotypes, max_sites=n_ab_sites)
    for time, derived_gt in zip(inf_times, inf_derived_gts):
        ab.add_site(time=time, genotypes=derived_gt)
    ab.add_terminal_site()

    # --- 5. Generate ancestors with gap clipping ---
    site_interval_idx = _assign_site_intervals(inf_positions_arr, seq_intervals)

    ancestors = []
    for time, focal_sites in ab.ancestor_descriptors():
        focal_arr = np.asarray(focal_sites, dtype=np.int32)
        focal_int_idx = site_interval_idx[focal_arr]

        # Split focal sites by interval (handle rare cross-gap groupings)
        for i_idx in np.unique(focal_int_idx):
            if i_idx < 0:
                continue
            sub_focal = focal_arr[focal_int_idx == i_idx]

            # AncestorBuilder requires a buffer of length ab.num_sites
            a = np.full(n_ab_sites, np.int8(-1), dtype=np.int8)
            ab.make_ancestor(sub_focal.tolist(), a)
            # Trim the terminal sentinel slot; keep only the num_inf inference sites
            a = a[:num_inf]

            # Clip: zero out sites outside the focal interval
            i_start, i_end = int(seq_intervals[i_idx, 0]), int(seq_intervals[i_idx, 1])
            outside = (inf_positions_arr < i_start) | (inf_positions_arr >= i_end)
            a[outside] = np.int8(-1)

            non_missing = np.where(a != -1)[0]
            if len(non_missing) == 0:
                continue

            start_pos = int(inf_positions_arr[non_missing[0]])
            end_pos = int(inf_positions_arr[non_missing[-1]])
            focal_pos = inf_positions_arr[sub_focal]

            ancestors.append(
                {
                    "time": float(time),
                    "haplotype": a.copy(),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "focal_positions": focal_pos,
                }
            )

    # NOTE don't sort any more, we need to be robust to times being mixed up arbitrarily.
    # Sort by descending time (oldest first), stable for equal times
    ancestors.sort(key=lambda x: -x["time"])

    # NOTE: don't add this virtual all-zeros ancestor, we need to adjust for this
    # elsewhere.
    # Prepend the virtual root (time=1.0, all-ancestral haplotype)
    virtual_hap = np.zeros(num_inf, dtype=np.int8)
    ancestors.insert(
        0,
        {
            "time": 1.0,
            "haplotype": virtual_hap,
            "start_pos": int(inf_positions_arr[0]),
            "end_pos": int(inf_positions_arr[-1]),
            "focal_positions": np.array([], dtype=np.int32),
        },
    )

    num_anc = len(ancestors)

    # --- 6. Assemble output arrays ---
    genotypes = np.full((num_inf, num_anc, 1), np.int8(-1), dtype=np.int8)
    for j, anc in enumerate(ancestors):
        genotypes[:, j, 0] = anc["haplotype"]

    times = np.array([a["time"] for a in ancestors], dtype=np.float64)
    start_positions = np.array([a["start_pos"] for a in ancestors], dtype=np.int32)
    end_positions = np.array([a["end_pos"] for a in ancestors], dtype=np.int32)

    max_focal = max(len(a["focal_positions"]) for a in ancestors) if ancestors else 1
    max_focal = max(max_focal, 1)  # at least 1 column (padded with -2)
    focal_positions = np.full((num_anc, max_focal), -2, dtype=np.int32)
    for j, anc in enumerate(ancestors):
        fp = anc["focal_positions"]
        focal_positions[j, : len(fp)] = fp

    # Build output alleles: [ancestral, primary_derived] per site
    out_alleles = np.empty((num_inf, 2), dtype=object)
    out_alleles[:] = ""
    for i in range(num_inf):
        site_alleles = final_alleles_list[i].tolist()
        anc_idx = final_anc_indices[i]
        out_alleles[i, 0] = str(site_alleles[anc_idx])
        # Find the first non-ancestral non-empty allele for slot 1
        for j, al in enumerate(site_alleles):
            if j != anc_idx and al:
                out_alleles[i, 1] = str(al)
                break

    return _write_ancestor_vcz(
        genotypes=genotypes,
        positions=inf_positions_arr,
        alleles=out_alleles,
        times=times,
        start_positions=start_positions,
        end_positions=end_positions,
        focal_positions=focal_positions,
        seq_intervals=seq_intervals,
    )


def _write_ancestor_vcz(
    genotypes,
    positions,
    alleles,
    times,
    start_positions,
    end_positions,
    focal_positions,
    seq_intervals,
):
    """Write ancestor data to an in-memory zarr Group and return it."""
    num_sites = len(positions)
    num_anc = genotypes.shape[1] if num_sites > 0 else 0

    sample_ids = np.array([f"ancestor_{i}" for i in range(num_anc)])

    root = _open_memory_group()

    if num_sites > 0 and num_anc > 0:
        _arr(root, "call_genotype", genotypes, ["variants", "samples", "ploidy"])
    else:
        g = root.create_array(
            "call_genotype", shape=(num_sites, num_anc, 1), dtype=np.int8
        )
        g.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

    _arr(root, "variant_position", positions, ["variants"])

    alleles_arr = root.create_array(
        "variant_allele", shape=alleles.shape, dtype=_VLEN_STR
    )
    alleles_arr[:] = alleles
    alleles_arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    sample_ids_arr = root.create_array(
        "sample_id", shape=sample_ids.shape, dtype=_VLEN_STR
    )
    sample_ids_arr[:] = sample_ids
    sample_ids_arr.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    _arr(root, "sample_time", times, ["samples"])
    _arr(root, "sample_start_position", start_positions, ["samples"])
    _arr(root, "sample_end_position", end_positions, ["samples"])
    _arr(root, "sample_focal_positions", focal_positions, ["samples", "focal_alleles"])
    _arr(root, "sequence_intervals", seq_intervals, ["intervals", "coords"])

    return root
