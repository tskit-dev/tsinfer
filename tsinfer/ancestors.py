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

import logging
import time
from dataclasses import dataclass

import numcodecs
import numpy as np
import psutil
import tqdm as tqdm_mod
import zarr

import _tsinfer

from . import vcz as vcz_mod
from .config import AncestorsConfig, AncestralState, Source

logger = logging.getLogger(__name__)


def _memory_usage_mb():
    """Return current RSS in MiB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


@dataclass
class InferenceSites:
    positions: np.ndarray
    alleles: np.ndarray
    ancestral_allele_index: np.ndarray


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


def _compute_site_stats(
    store, inf_sites, sample_include, num_haplotypes, progress=False
):
    """
    Pass 1: iterate over inference-site genotypes, compute derived genotype
    stats for each site, filter out fixed/all-missing.

    Returns (keep_mask, times) where:
    - keep_mask: boolean array of length len(inf_sites.positions)
    - times: float64 array of length sum(keep_mask)
    """
    num_inf_sites = len(inf_sites.positions)

    keep_mask = np.zeros(num_inf_sites, dtype=bool)
    times_list = []

    site_iter = tqdm_mod.tqdm(
        vcz_mod.iter_genotypes(
            store, inf_sites.positions, inf_sites.ancestral_allele_index, sample_include
        ),
        total=num_inf_sites,
        desc="Pass 1: site stats",
        unit="sites",
        disable=not progress,
    )

    for i, derived_gt in enumerate(site_iter):
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


def _find_duplicate_positions(store) -> set[int]:
    """Return the set of genomic positions that appear more than once."""
    all_positions = np.asarray(store["variant_position"][:], dtype=np.int32)
    unique, counts = np.unique(all_positions, return_counts=True)
    dup_mask = counts > 1
    if not np.any(dup_mask):
        return set()
    dup_positions = set(unique[dup_mask].tolist())
    n_dup_sites = int(np.sum(counts[dup_mask]))
    logger.info(
        "Found %d duplicate position(s) (%d sites total); "
        "these will be excluded from inference",
        len(dup_positions),
        n_dup_sites,
    )
    return dup_positions


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_inference_sites(
    store: zarr.Group,
    ancestral_state: AncestralState | None,
    *,
    include: str | None = None,
    exclude: str | None = None,
    regions: str | None = None,
    targets: str | None = None,
):
    """
    Return an InferenceSites dataclass for sites that pass filtering and
    have a valid ancestral allele.

    Variant filtering (include/exclude/regions/targets) is delegated to
    vcztools via ``iter_variants``.
    """
    # Detect duplicate positions upfront so we can skip them
    dup_positions = _find_duplicate_positions(store)

    # Build external ancestral state lookup if configured
    ann_lookup = None
    if ancestral_state is not None:
        ann_store = vcz_mod.open_store(ancestral_state.path)
        ann_positions = np.asarray(ann_store["variant_position"][:])
        ann_values = np.asarray(ann_store[ancestral_state.field][:])
        ann_lookup = {
            int(k): str(v) for k, v in zip(ann_positions.tolist(), ann_values.tolist())
        }

    # Iterate variants that pass vcztools filters, collecting metadata
    fields = ["variant_position", "variant_allele"]
    if ancestral_state is None:
        fields.append("variant_ancestral_allele")

    pos_list = []
    allele_list = []
    anc_idx_list = []
    num_filtered = 0
    num_no_ancestral = 0
    num_duplicate = 0

    for variant in vcz_mod.iter_variants(
        store,
        fields=fields,
        include=include,
        exclude=exclude,
        regions=regions,
        targets=targets,
    ):
        num_filtered += 1
        pos = int(variant["variant_position"])

        if pos in dup_positions:
            num_duplicate += 1
            continue

        site_alleles = variant["variant_allele"]

        if ann_lookup is not None:
            anc_str = ann_lookup.get(pos, "")
        else:
            anc_str = str(variant["variant_ancestral_allele"])

        # Find ancestral allele index
        anc_idx = -1
        for j, a in enumerate(site_alleles.tolist()):
            if a is not None and a != "" and a == anc_str:
                anc_idx = j
                break

        if anc_idx < 0:
            num_no_ancestral += 1
            continue

        pos_list.append(pos)
        allele_list.append(site_alleles)
        anc_idx_list.append(anc_idx)

    logger.info(
        "Sites passing variant filters: %d; skipped %d (no ancestral allele match)"
        ", %d (duplicate positions)",
        num_filtered,
        num_no_ancestral,
        num_duplicate,
    )

    if not pos_list:
        return InferenceSites(
            positions=np.zeros(0, dtype=np.int32),
            alleles=np.zeros((0, 0), dtype=object),
            ancestral_allele_index=np.zeros(0, dtype=np.int8),
        )

    return InferenceSites(
        positions=np.array(pos_list, dtype=np.int32),
        alleles=np.stack(allele_list),
        ancestral_allele_index=np.array(anc_idx_list, dtype=np.int8),
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


@dataclass
class Ancestor:
    """
    A single ancestor haplotype, storing only the active fragment.

    The haplotype covers only the site index range
    ``[start_site_idx, end_site_idx)``; all other sites are implicitly
    missing (-1).  Call :meth:`expand_haplotype` to materialise the
    full-length array (done at flush time in :class:`AncestorWriter`).
    """

    index: int
    time: float
    haplotype: np.ndarray  # fragment: sites[start_site_idx:end_site_idx]
    focal_positions: np.ndarray
    start_site_idx: int  # global site index, inclusive
    end_site_idx: int  # global site index, exclusive
    start_position: int  # genomic position
    end_position: int  # genomic position

    def expand_haplotype(self, num_sites):
        """Return a full-length haplotype array with missing = -1."""
        full = np.full(num_sites, np.int8(-1), dtype=np.int8)
        full[self.start_site_idx : self.end_site_idx] = self.haplotype
        return full


def _open_source(source):
    """
    Open the source store, returning a zarr.Group.
    """
    logger.info("Source: %s", source.path)
    return vcz_mod.open_store(source.path)


def _resolve_haplotype_count(store, samples_str):
    """
    Resolve sample selection and compute haplotype count.

    Returns (num_haplotypes, sample_include).
    """
    gt_shape = store["call_genotype"].shape
    num_total_sites, num_samples, ploidy = gt_shape
    logger.info(
        "Store: %d sites, %d samples, ploidy %d",
        num_total_sites,
        num_samples,
        ploidy,
    )

    samples_selection = vcz_mod.resolve_samples_selection(store, samples_str)
    if samples_selection is not None:
        sample_include = np.zeros(num_samples, dtype=bool)
        sample_include[samples_selection] = True
        num_samples_used = len(samples_selection)
        logger.info(
            "Sample filter: %d of %d samples selected (%d haplotypes)",
            num_samples_used,
            num_samples,
            num_samples_used * ploidy,
        )
    else:
        sample_include = None
        num_samples_used = num_samples
    num_haplotypes = num_samples_used * ploidy
    logger.info("Using %d samples, %d haplotypes", num_samples_used, num_haplotypes)
    return num_haplotypes, sample_include


def _apply_site_stats(store, inf_sites, sample_include, num_haplotypes, progress):
    """
    Run Pass 1: compute site stats and filter sites.

    Returns (final_positions, final_alleles, final_anc_indices, times).
    """
    t0 = time.monotonic()
    logger.info("Pass 1: computing site stats")
    keep_mask, times = _compute_site_stats(
        store, inf_sites, sample_include, num_haplotypes, progress=progress
    )

    final_positions = inf_sites.positions[keep_mask]
    final_alleles = inf_sites.alleles[keep_mask]
    final_anc_indices = inf_sites.ancestral_allele_index[keep_mask]

    num_inf = len(final_positions)
    num_dropped = len(inf_sites.positions) - num_inf
    elapsed = time.monotonic() - t0
    logger.info(
        "Pass 1 complete: %d sites kept, %d dropped (fixed or all-missing)"
        " in %.1fs (RSS=%.1fMiB)",
        num_inf,
        num_dropped,
        elapsed,
        _memory_usage_mb(),
    )
    if num_inf > 0:
        logger.info(
            "Inference site range: %d–%d",
            int(final_positions[0]),
            int(final_positions[-1]),
        )

    return final_positions, final_alleles, final_anc_indices, times


def _process_interval(
    store,
    i_idx,
    local_mask,
    final_positions,
    final_anc_indices,
    times,
    num_haplotypes,
    cfg,
    sample_include,
    zarr_root,
    num_threads,
    write_threads,
    ancestor_index,
    progress,
    compressor,
):
    """
    Process a single sequence interval: load genotypes, build ancestors.

    Creates a per-interval AncestorWriter that owns its own worker and
    writer threads.

    Returns (updated_ancestor_index, interval_focal_positions).
    """
    n_local = len(local_mask)
    local_positions = final_positions[local_mask]
    local_anc_indices = final_anc_indices[local_mask]
    local_times = times[local_mask]
    n_ab_sites = n_local
    ab = _tsinfer.AncestorBuilder(
        num_samples=num_haplotypes,
        max_sites=n_ab_sites,
        genotype_encoding=cfg.genotype_encoding,
    )

    gt_iter = tqdm_mod.tqdm(
        vcz_mod.iter_genotypes(
            store, local_positions, local_anc_indices, sample_include
        ),
        total=n_local,
        desc=f"Interval {i_idx}: loading sites",
        unit="sites",
        disable=not progress,
    )
    for j, derived_gt in enumerate(gt_iter):
        ab.add_site(time=float(local_times[j]), genotypes=derived_gt)

    ancestor_descriptors = list(ab.ancestor_descriptors())
    ab_mem_mb = ab.mem_size / (1024 * 1024)
    n_ancestors = len(ancestor_descriptors)
    logger.info(
        "Interval %d: %d sites (%d–%d), %d ancestors (builder=%.1fMiB, RSS=%.1fMiB)",
        i_idx,
        n_local,
        int(local_positions[0]),
        int(local_positions[-1]),
        n_ancestors,
        ab_mem_mb,
        _memory_usage_mb(),
    )

    if n_ancestors == 0:
        return ancestor_index, []

    num_sites = len(final_positions)
    writer = vcz_mod.AncestorWriter(
        zarr_root=zarr_root,
        num_sites=num_sites,
        chunk_size=cfg.samples_chunk_size,
        n_ancestors=n_ancestors,
        local_mask=local_mask,
        final_positions=final_positions,
        num_threads=num_threads,
        write_threads=write_threads,
        compressor=compressor,
    )

    t_wall_start = time.monotonic()

    pbar = tqdm_mod.tqdm(
        total=n_ancestors,
        desc=f"Interval {i_idx}: ancestors",
        unit="haps",
        disable=not progress,
    )

    for local_index, (anc_time, focal_sites) in enumerate(ancestor_descriptors):
        writer.submit(
            ab,
            focal_sites,
            anc_time,
            local_index,
        )
        pbar.update(1)

    t_submit_wall = time.monotonic() - t_wall_start
    pbar.close()

    interval_focals = writer.finalize()

    t_wall = time.monotonic() - t_wall_start
    s = writer.stats
    if n_ancestors > 0:
        make_mean = s.worker_make_ancestor / n_ancestors
        make_min = s.worker_make_min if s.worker_make_min < float("inf") else 0
        make_max = s.worker_make_max
        logger.info(
            "Interval %d: %d ancestors, %d chunks in %.3fs | "
            "submit: wall=%.3fs put_block=%.3fs | "
            "workers: make_ancestor=%.3fs (mean=%.4fs min=%.4fs max=%.4fs) "
            "buf_fill=%.3fs registry_wait=%.3fs | "
            "writers: scatter=%.3fs zarr=%.3fs release=%.3fs | "
            "finalize: worker_join=%.3fs seal=%.3fs writer_join=%.3fs",
            i_idx,
            n_ancestors,
            s.writer_chunks,
            t_wall,
            t_submit_wall,
            s.submit_put,
            s.worker_make_ancestor,
            make_mean,
            make_min,
            make_max,
            s.worker_buf_fill,
            s.worker_registry_wait,
            s.writer_scatter,
            s.writer_zarr,
            s.writer_release,
            s.finalize_worker_join,
            s.finalize_seal,
            s.finalize_writer_join,
        )

    ancestor_index += n_ancestors
    return ancestor_index, interval_focals


def infer_ancestors(
    source: Source,
    cfg: AncestorsConfig,
    ancestral_state: AncestralState | None = None,
    progress: bool = False,
    num_threads: int = 0,
) -> zarr.Group:
    """
    Build the ancestor VCZ store from a samples VCZ store.

    Two-pass approach:
      Pass 1 — Identify inference sites and compute times (chunk-aware).
      Pass 2 — Per-interval AncestorBuilder instances; stream ancestors to output.

    No virtual root is inserted; that is the responsibility of the match step.
    Ancestors are not sorted by time.
    """
    t_start = time.monotonic()
    logger.info("Starting ancestor inference (RSS=%.1fMiB)", _memory_usage_mb())

    # --- 1. Open store and resolve filtering ---
    store = _open_source(source)

    # Build compressor from config
    compressor = numcodecs.Blosc(
        cname=cfg.compressor,
        clevel=cfg.compression_level,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    if source.include is not None:
        logger.info("Variant filter (include): %s", source.include)
    if source.exclude is not None:
        logger.info("Variant filter (exclude): %s", source.exclude)
    if source.regions is not None:
        logger.info("Region filter: %s", source.regions)
    if source.targets is not None:
        logger.info("Targets filter: %s", source.targets)

    num_haplotypes, sample_include = _resolve_haplotype_count(store, source.samples)

    # --- 2. Compute inference sites ---
    inf_sites = compute_inference_sites(
        store,
        ancestral_state,
        include=source.include,
        exclude=source.exclude,
        regions=source.regions,
        targets=source.targets,
    )
    logger.info("Inference sites identified: %d", len(inf_sites.positions))

    # --- 3. Pass 1: compute site stats ---
    final_positions, final_alleles, final_anc_indices, times = _apply_site_stats(
        store, inf_sites, sample_include, num_haplotypes, progress
    )
    num_inf = len(final_positions)

    if num_inf == 0:
        seq_len = vcz_mod.sequence_length(store)
        seq_intervals = compute_sequence_intervals(
            final_positions, seq_len, cfg.max_gap_length
        )
        contig_id = str(store["contig_id"][0])
        contig_length = int(store["contig_length"][0])
        return vcz_mod.write_empty_ancestor_vcz(
            seq_intervals,
            store=cfg.path,
            contig_id=contig_id,
            contig_length=contig_length,
            compressor=compressor,
        )

    # --- 4. Compute sequence intervals ---
    seq_len = vcz_mod.sequence_length(store)
    seq_intervals = compute_sequence_intervals(
        final_positions, seq_len, cfg.max_gap_length
    )
    logger.info(
        "Sequence intervals: %d (max_gap_length=%d)",
        len(seq_intervals),
        cfg.max_gap_length,
    )

    # --- 5. Pass 2: per-interval ancestor building ---
    t_pass2 = time.monotonic()
    encoding_name = "one_bit" if cfg.genotype_encoding == 1 else "eight_bit"
    write_threads = max(1, num_threads)
    if num_threads > 0:
        logger.info(
            "Pass 2: building ancestors per interval "
            "(%d worker threads, %d write threads, encoding=%s, RSS=%.1fMiB)",
            num_threads,
            write_threads,
            encoding_name,
            _memory_usage_mb(),
        )
    else:
        logger.info(
            "Pass 2: building ancestors per interval "
            "(synchronous, %d write threads, encoding=%s, RSS=%.1fMiB)",
            write_threads,
            encoding_name,
            _memory_usage_mb(),
        )
    site_interval_idx = _assign_site_intervals(final_positions, seq_intervals)

    contig_id = str(store["contig_id"][0])
    contig_length = int(store["contig_length"][0])

    # Create zarr group with fixed arrays
    zarr_root = vcz_mod.setup_ancestor_zarr(
        num_inf,
        final_positions,
        final_alleles,
        final_anc_indices,
        seq_intervals,
        store=cfg.path,
        samples_chunk_size=cfg.samples_chunk_size,
        variants_chunk_size=cfg.variants_chunk_size,
        contig_id=contig_id,
        contig_length=contig_length,
        compressor=compressor,
    )

    ancestor_index = 0
    all_focal_positions = []

    for i_idx in range(len(seq_intervals)):
        in_interval = site_interval_idx == i_idx
        if not np.any(in_interval):
            continue
        local_mask = np.where(in_interval)[0]
        ancestor_index, interval_focals = _process_interval(
            store,
            i_idx,
            local_mask,
            final_positions,
            final_anc_indices,
            times,
            num_haplotypes,
            cfg,
            sample_include,
            zarr_root,
            num_threads,
            write_threads,
            ancestor_index,
            progress,
            compressor,
        )
        all_focal_positions.extend(interval_focals)

    elapsed_pass2 = time.monotonic() - t_pass2
    logger.info("Pass 2 complete in %.1fs", elapsed_pass2)

    # Write sample_id and sample_focal_positions
    result = vcz_mod.finalize_ancestor_zarr(
        zarr_root, all_focal_positions, compressor=compressor
    )
    elapsed_total = time.monotonic() - t_start
    logger.info(
        "Ancestor inference complete: %d ancestors across %d sites"
        " in %.1fs total (RSS=%.1fMiB)",
        result["call_genotype"].shape[1],
        num_inf,
        elapsed_total,
        _memory_usage_mb(),
    )
    return result
