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

import concurrent.futures
import logging
import time
from dataclasses import dataclass

import numpy as np
import tqdm as tqdm_mod
import zarr

import _tsinfer

from . import vcz as vcz_mod
from .config import AncestorsConfig, AncestralState, Source
from .utils import SynchronousExecutor

logger = logging.getLogger(__name__)


def _memory_usage_mb():
    """Return current RSS in MiB."""
    import psutil

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
        "Sites passing variant filters: %d; skipped %d (no ancestral allele match)",
        num_filtered,
        num_no_ancestral,
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


def _call_make_ancestor(ab, focal_sites_list, a):
    """
    Worker function: call the C make_ancestor and return raw results.

    Only the GIL-releasing C work lives here so that worker threads
    spend almost no time holding the GIL.  All Python post-processing
    (slicing, copying, Ancestor construction) happens on the main thread.

    The output array *a* and the focal_sites list are prepared by the
    caller on the main thread to minimise GIL-held work here.
    """
    t0 = time.monotonic()
    start_local, end_local = ab.make_ancestor(focal_sites_list, a)
    dt = time.monotonic() - t0
    return start_local, end_local, dt


def _finish_ancestor(
    a,
    start_local,
    end_local,
    focal_arr,
    anc_time,
    local_mask,
    final_positions,
    ancestor_index,
):
    """
    Main-thread post-processing: build an Ancestor from raw C output.
    """
    fragment = a[start_local:end_local].copy()
    start_site_idx = int(local_mask[start_local])
    end_site_idx = int(local_mask[end_local - 1]) + 1

    start_pos = int(final_positions[start_site_idx])
    end_pos = int(final_positions[end_site_idx - 1])
    focal_global = local_mask[focal_arr]
    focal_pos = final_positions[focal_global]

    return Ancestor(
        index=ancestor_index,
        time=float(anc_time),
        haplotype=fragment,
        focal_positions=focal_pos,
        start_site_idx=start_site_idx,
        end_site_idx=end_site_idx,
        start_position=start_pos,
        end_position=end_pos,
    )


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
    writer,
    executor,
    num_threads,
    ancestor_index,
    progress,
):
    """
    Process a single sequence interval: load genotypes, build ancestors.

    Returns the updated ancestor_index.
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
    logger.info(
        "Interval %d: %d sites (%d–%d), %d ancestors (builder=%.1fMiB, RSS=%.1fMiB)",
        i_idx,
        n_local,
        int(local_positions[0]),
        int(local_positions[-1]),
        len(ancestor_descriptors),
        ab_mem_mb,
        _memory_usage_mb(),
    )

    # Build ancestors with bounded in-flight futures.
    # Submit up to max_queued futures,
    # then drain completed ones before submitting more.  The writer's
    # index-aware pending dict ensures deterministic output regardless of
    # completion order.
    #
    # Worker threads only run the GIL-releasing C make_ancestor call.
    # All Python post-processing (_finish_ancestor) runs on the main
    # thread when draining, to avoid GIL contention.
    max_queued = max(8 * num_threads, 1)
    # Map future → (anc_time, ancestor_index, ...) for main-thread finishing
    future_meta = {}
    n_ancestors = len(ancestor_descriptors)
    n_consumed = 0
    # Pre-allocate reusable output arrays to avoid per-ancestor allocation.
    # _finish_ancestor copies out the active fragment, so the array can be
    # recycled immediately after draining.
    _free_arrays = [np.empty(n_ab_sites, dtype=np.int8) for _ in range(max_queued)]
    pbar = tqdm_mod.tqdm(
        total=n_ancestors,
        desc=f"Interval {i_idx}: ancestors",
        unit="haps",
        disable=not progress,
    )

    # Timing accumulators for diagnosing throughput bottlenecks.
    # All times in seconds.
    t_wait = 0.0  # blocked in concurrent.futures.wait()
    t_finish = 0.0  # _finish_ancestor post-processing
    t_write = 0.0  # writer.add_ancestor
    t_submit = 0.0  # preparing + submitting futures
    t_make_total = 0.0  # sum of per-call make_ancestor wall times
    t_make_min = float("inf")
    t_make_max = 0.0

    def _drain_completed(future_meta, pbar=pbar):
        nonlocal n_consumed, t_wait, t_finish, t_write, t_make_total
        nonlocal t_make_min, t_make_max

        t0 = time.monotonic()
        done, _pending = concurrent.futures.wait(
            future_meta.keys(),
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        t_wait += time.monotonic() - t0

        for future in done:
            start_local, end_local, dt_make = future.result()
            t_make_total += dt_make
            if dt_make < t_make_min:
                t_make_min = dt_make
            if dt_make > t_make_max:
                t_make_max = dt_make

            anc_time, anc_idx, a, focal_arr = future_meta.pop(future)

            t0 = time.monotonic()
            ancestor = _finish_ancestor(
                a,
                start_local,
                end_local,
                focal_arr,
                anc_time,
                local_mask,
                final_positions,
                anc_idx,
            )
            t_finish += time.monotonic() - t0

            # Return array to pool for reuse
            _free_arrays.append(a)

            t0 = time.monotonic()
            writer.add_ancestor(ancestor)
            t_write += time.monotonic() - t0

            n_consumed += 1
            pbar.update(1)

    for anc_time, focal_sites in ancestor_descriptors:
        if len(future_meta) >= max_queued:
            _drain_completed(future_meta)
        t0 = time.monotonic()
        focal_arr = np.asarray(focal_sites, dtype=np.int32)
        a = _free_arrays.pop()
        a[:] = np.int8(-1)
        future = executor.submit(
            _call_make_ancestor,
            ab,
            focal_arr.tolist(),
            a,
        )
        future_meta[future] = (anc_time, ancestor_index, a, focal_arr)
        t_submit += time.monotonic() - t0
        ancestor_index += 1

    while future_meta:
        _drain_completed(future_meta)

    if n_consumed > 0:
        t_make_mean = t_make_total / n_consumed
        logger.info(
            "Interval %d timing: %d ancestors, "
            "wait=%.3fs finish=%.3fs write=%.3fs submit=%.3fs | "
            "make_ancestor total=%.3fs mean=%.4fs min=%.4fs max=%.4fs",
            i_idx,
            n_consumed,
            t_wait,
            t_finish,
            t_write,
            t_submit,
            t_make_total,
            t_make_mean,
            t_make_min,
            t_make_max,
        )

    pbar.close()
    return ancestor_index


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
    if num_threads > 0:
        logger.info(
            "Pass 2: building ancestors per interval "
            "(%d threads, encoding=%s, RSS=%.1fMiB)",
            num_threads,
            encoding_name,
            _memory_usage_mb(),
        )
    else:
        logger.info(
            "Pass 2: building ancestors per interval "
            "(synchronous, encoding=%s, RSS=%.1fMiB)",
            encoding_name,
            _memory_usage_mb(),
        )
    site_interval_idx = _assign_site_intervals(final_positions, seq_intervals)

    contig_id = str(store["contig_id"][0])
    contig_length = int(store["contig_length"][0])
    writer = vcz_mod.AncestorWriter(
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
    )

    if num_threads > 0:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    else:
        executor = SynchronousExecutor()

    ancestor_index = 0
    with executor:
        for i_idx in range(len(seq_intervals)):
            in_interval = site_interval_idx == i_idx
            if not np.any(in_interval):
                continue
            local_mask = np.where(in_interval)[0]
            ancestor_index = _process_interval(
                store,
                i_idx,
                local_mask,
                final_positions,
                final_anc_indices,
                times,
                num_haplotypes,
                cfg,
                sample_include,
                writer,
                executor,
                num_threads,
                ancestor_index,
                progress,
            )

    elapsed_pass2 = time.monotonic() - t_pass2
    logger.info("Pass 2 complete in %.1fs", elapsed_pass2)

    result = writer.finalize()
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
