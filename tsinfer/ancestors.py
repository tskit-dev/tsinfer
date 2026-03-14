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
import resource
from dataclasses import dataclass

import numpy as np
import zarr

import _tsinfer

from . import vcz as vcz_mod
from .config import AncestorsConfig, AncestralState, Source
from .utils import SynchronousExecutor

logger = logging.getLogger(__name__)


def _memory_usage_mb():
    """Return current max RSS in MiB (Unix) via the resource module."""
    # ru_maxrss is in KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


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

    site_iter = vcz_mod.iter_genotypes(
        store, inf_sites.positions, inf_sites.ancestral_allele_index, sample_include
    )
    if progress:
        import tqdm

        site_iter = tqdm.tqdm(
            site_iter,
            total=num_inf_sites,
            desc="Pass 1: site stats",
            unit="sites",
            postfix={"RSS": f"{_memory_usage_mb():.0f}MiB"},
        )

    for i, derived_gt in enumerate(site_iter):
        derived_count = int(np.sum(derived_gt == 1))
        n_non_missing = int(np.sum(derived_gt >= 0))

        if n_non_missing == 0 or derived_count == 0 or derived_count == n_non_missing:
            continue

        keep_mask[i] = True
        times_list.append(derived_count / num_haplotypes)

        if progress and i % 1000 == 0:
            site_iter.set_postfix(RSS=f"{_memory_usage_mb():.0f}MiB")

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
class AncestorResult:
    """Result of building a single ancestor haplotype."""

    index: int
    time: float
    haplotype: np.ndarray
    focal_positions: np.ndarray
    start_position: int
    end_position: int


def _build_one_ancestor(
    ab,
    time,
    focal_sites,
    n_ab_sites,
    n_local,
    local_mask,
    final_positions,
    num_inf,
    ancestor_index,
):
    """
    Build a single ancestor haplotype from a finalized AncestorBuilder.

    This is the per-ancestor work extracted from the inner loop of Pass 2
    so it can be submitted to a thread pool. The AncestorBuilder's
    ``make_ancestor`` reads only from finalized (immutable) builder state
    and writes into caller-owned arrays, so concurrent calls are safe.

    The C engine always sets focal sites to 1 (derived) and extends
    outward, so the result is never all-missing.
    """
    focal_arr = np.asarray(focal_sites, dtype=np.int32)
    a = np.full(n_ab_sites, np.int8(-1), dtype=np.int8)
    start_local, end_local = ab.make_ancestor(focal_arr.tolist(), a)
    a = a[:n_local]  # trim terminal

    # Map local haplotype to global inference site coordinates
    global_hap = np.full(num_inf, np.int8(-1), dtype=np.int8)
    global_hap[local_mask] = a

    start_pos = int(final_positions[local_mask[start_local]])
    end_pos = int(final_positions[local_mask[end_local - 1]])
    focal_global = local_mask[focal_arr]
    focal_pos = final_positions[focal_global]

    return AncestorResult(
        index=ancestor_index,
        time=float(time),
        haplotype=global_hap,
        focal_positions=focal_pos,
        start_position=start_pos,
        end_position=end_pos,
    )


def infer_ancestors(
    source: Source | zarr.Group,
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
    logger.info("Starting ancestor inference (RSS=%.1fMiB)", _memory_usage_mb())

    # --- 1. Open store and resolve filtering ---
    if isinstance(source, zarr.Group):
        store = source
        include_expr = None
        exclude_expr = None
        samples_str = None
        regions_str = None
        targets_str = None
    else:
        store = vcz_mod.open_store(source.path)
        include_expr = source.include
        exclude_expr = source.exclude
        samples_str = source.samples
        regions_str = source.regions
        targets_str = source.targets
        logger.info("Source: %s", source.path)

    gt_shape = store["call_genotype"].shape
    num_total_sites, num_samples, ploidy = gt_shape
    logger.info(
        "Store: %d sites, %d samples, ploidy %d",
        num_total_sites,
        num_samples,
        ploidy,
    )

    # Log active filters
    if include_expr is not None:
        logger.info("Variant filter (include): %s", include_expr)
    if exclude_expr is not None:
        logger.info("Variant filter (exclude): %s", exclude_expr)
    if regions_str is not None:
        logger.info("Region filter: %s", regions_str)
    if targets_str is not None:
        logger.info("Targets filter: %s", targets_str)

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

    # --- 2. Compute inference sites ---
    inf_sites = compute_inference_sites(
        store,
        ancestral_state,
        include=include_expr,
        exclude=exclude_expr,
        regions=regions_str,
        targets=targets_str,
    )
    logger.info("Inference sites identified: %d", len(inf_sites.positions))

    # --- 3. Pass 1: compute site stats ---
    logger.info("Pass 1: computing site stats")
    keep_mask, times = _compute_site_stats(
        store, inf_sites, sample_include, num_haplotypes, progress=progress
    )

    final_positions = inf_sites.positions[keep_mask]
    final_alleles = inf_sites.alleles[keep_mask]
    final_anc_indices = inf_sites.ancestral_allele_index[keep_mask]

    num_inf = len(final_positions)
    num_dropped = len(inf_sites.positions) - num_inf
    logger.info(
        "Pass 1 complete: %d sites kept, %d dropped (fixed or all-missing)"
        " (RSS=%.1fMiB)",
        num_inf,
        num_dropped,
        _memory_usage_mb(),
    )
    if num_inf > 0:
        logger.info(
            "Inference site range: %d–%d",
            int(final_positions[0]),
            int(final_positions[-1]),
        )

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

    logger.info(
        "Sequence intervals: %d (max_gap_length=%d)",
        len(seq_intervals),
        cfg.max_gap_length,
    )

    # --- 5. Pass 2: per-interval ancestor building ---
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

    if num_threads > 0:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    else:
        executor = SynchronousExecutor()

    interval_range = range(len(seq_intervals))
    if progress:
        import tqdm

        interval_range = tqdm.tqdm(
            interval_range,
            desc="Pass 2: intervals",
            unit="intervals",
            postfix={"RSS": f"{_memory_usage_mb():.0f}MiB"},
        )

    ancestor_index = 0
    with executor:
        for i_idx in interval_range:
            in_interval = site_interval_idx == i_idx
            if not np.any(in_interval):
                continue

            local_mask = np.where(in_interval)[0]
            n_local = len(local_mask)

            # Create per-interval AncestorBuilder, streaming genotypes from store
            local_positions = final_positions[local_mask]
            local_anc_indices = final_anc_indices[local_mask]
            local_times = times[local_mask]
            n_ab_sites = n_local + 1  # +1 for terminal
            ab = _tsinfer.AncestorBuilder(
                num_samples=num_haplotypes,
                max_sites=n_ab_sites,
                genotype_encoding=cfg.genotype_encoding,
            )

            for j, derived_gt in enumerate(
                vcz_mod.iter_genotypes(
                    store, local_positions, local_anc_indices, sample_include
                )
            ):
                ab.add_site(time=float(local_times[j]), genotypes=derived_gt)
            ab.add_terminal_site()

            ancestor_descriptors = list(ab.ancestor_descriptors())
            ab_mem_mb = ab.mem_size / (1024 * 1024)
            logger.info(
                "Interval %d: %d sites (%d–%d), %d ancestors "
                "(builder=%.1fMiB, RSS=%.1fMiB)",
                i_idx,
                n_local,
                int(local_positions[0]),
                int(local_positions[-1]),
                len(ancestor_descriptors),
                ab_mem_mb,
                _memory_usage_mb(),
            )

            if progress:
                interval_range.set_postfix(RSS=f"{_memory_usage_mb():.0f}MiB")

            # Submit all ancestors for this interval to the executor.
            # Use a set so we can discard each future after consumption,
            # freeing the AncestorResult (and its haplotype array).
            futures = set()
            for time, focal_sites in ancestor_descriptors:
                future = executor.submit(
                    _build_one_ancestor,
                    ab,
                    time,
                    focal_sites,
                    n_ab_sites,
                    n_local,
                    local_mask,
                    final_positions,
                    num_inf,
                    ancestor_index,
                )
                futures.add(future)
                ancestor_index += 1

            # Consume as completed for best throughput. The writer's
            # index-aware pending dict ensures deterministic output
            # regardless of completion order.
            completed = concurrent.futures.as_completed(futures)
            pbar = None
            if progress:
                pbar = tqdm.tqdm(
                    completed,
                    total=len(futures),
                    desc="Interval ancestors",
                    unit="haps",
                    postfix={"RSS": f"{_memory_usage_mb():.0f}MiB"},
                )
                completed = pbar
            for future in completed:
                anc = future.result()
                futures.discard(future)
                writer.add_ancestor(
                    anc.index,
                    anc.time,
                    anc.haplotype,
                    anc.focal_positions,
                    anc.start_position,
                    anc.end_position,
                )
                if pbar is not None:
                    pbar.set_postfix(RSS=f"{_memory_usage_mb():.0f}MiB")

    result = writer.finalize()
    logger.info(
        "Ancestor inference complete: %d ancestors across %d sites (RSS=%.1fMiB)",
        result["call_genotype"].shape[1],
        num_inf,
        _memory_usage_mb(),
    )
    return result
