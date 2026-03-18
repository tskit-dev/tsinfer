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
Ancestor grouping by linesweep: groups ancestors for parallel matching
using a linesweep + topological sort approach.

Also provides lightweight metadata loading and the ``compute_groups`` /
``compute_groups_json`` entry points so the module is self-contained.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time as time_
from dataclasses import dataclass

import numba
import numpy as np

from . import vcz as vcz_mod
from .config import Config

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight metadata loader
# ---------------------------------------------------------------------------


@dataclass
class HaplotypeMetadata:
    """Arrays needed by ``compute_groups`` — no genotype data."""

    times: np.ndarray  # (n,) float64
    is_ancestor: np.ndarray  # (n,) bool
    start_positions: np.ndarray  # (n,) int32
    end_positions: np.ndarray  # (n,) int32
    source: list[str]  # (n,) source name per haplotype
    sample_id: list[str]  # (n,) sample ID per haplotype
    ploidy_index: list[int]  # (n,) ploidy index per haplotype


@dataclass
class MatchJob:
    """One row of the compute-groups output — one per haplotype."""

    haplotype_index: int
    source: str
    sample_id: str
    ploidy_index: int
    time: float
    start_position: int
    end_position: int
    group: int


def collect_haplotype_metadata(cfg: Config) -> HaplotypeMetadata:
    """
    Load only the 1-D metadata arrays from the ancestor VCZ and sample
    sources — no ``call_genotype`` data is read.
    """
    logger.info("Loading haplotype metadata (lightweight, no genotypes)")

    # --- Ancestors ---
    anc_store = vcz_mod.open_store(cfg.ancestors.path)
    positions = np.asarray(anc_store["variant_position"][:], dtype=np.int32)

    anc_times = np.asarray(anc_store["sample_time"][:], dtype=np.float64)
    anc_start = np.asarray(anc_store["sample_start_position"][:], dtype=np.int32)
    anc_end = np.asarray(anc_store["sample_end_position"][:], dtype=np.int32)
    num_anc = len(anc_times)
    logger.info("Loaded %d ancestors from %s", num_anc, cfg.ancestors.path)

    anc_is_ancestor = np.ones(num_anc, dtype=bool)

    # Per-haplotype identity for ancestors
    anc_ids = [str(x) for x in anc_store["sample_id"][:].tolist()]
    anc_source: list[str] = ["ancestors"] * num_anc
    anc_sample_id: list[str] = anc_ids
    anc_ploidy_index: list[int] = [0] * num_anc

    # --- Sample sources ---
    sample_times_list: list[np.ndarray] = []
    sample_start_list: list[np.ndarray] = []
    sample_end_list: list[np.ndarray] = []
    sample_is_ancestor_list: list[np.ndarray] = []
    sample_source_list: list[str] = []
    sample_sample_id_list: list[str] = []
    sample_ploidy_index_list: list[int] = []

    for source_name in cfg.match.sources:
        source = cfg.sources[source_name]
        store = vcz_mod.open_store(source.path)

        # Shape metadata only — no data loaded
        gt_shape = store["call_genotype"].shape
        num_samples = gt_shape[1]
        ploidy = gt_shape[2]

        samples_selection = vcz_mod.resolve_samples_selection(store, source.samples)
        if samples_selection is not None:
            num_samples = len(samples_selection)

        num_hap = num_samples * ploidy
        logger.info(
            "Source '%s': %d samples, ploidy %d, %d haplotypes",
            source_name,
            num_samples,
            ploidy,
            num_hap,
        )

        # sample_time
        all_num_samples = gt_shape[1]
        sample_time_arr = vcz_mod.resolve_field(
            store, source.sample_time, "sample_id", all_num_samples, fill_value=0
        )
        if sample_time_arr is None:
            sample_time_arr = np.zeros(num_samples, dtype=np.float64)
        else:
            sample_time_arr = np.asarray(sample_time_arr, dtype=np.float64)
            if samples_selection is not None:
                sample_time_arr = sample_time_arr[samples_selection]

        hap_times = np.repeat(sample_time_arr, ploidy)

        # sample_id
        raw_ids = store["sample_id"][:]
        if samples_selection is not None:
            raw_ids = raw_ids[samples_selection]
        sample_ids = [str(x) for x in raw_ids.tolist()]

        # Per-haplotype identity
        for i in range(num_samples):
            for p in range(ploidy):
                sample_source_list.append(source.name)
                sample_sample_id_list.append(sample_ids[i])
                sample_ploidy_index_list.append(p)

        # Default start/end from positions
        if len(positions) > 0:
            hap_start = np.full(num_hap, int(positions[0]), dtype=np.int32)
            hap_end = np.full(num_hap, int(positions[-1]), dtype=np.int32)
        else:
            hap_start = np.zeros(num_hap, dtype=np.int32)
            hap_end = np.zeros(num_hap, dtype=np.int32)

        sample_times_list.append(hap_times)
        sample_start_list.append(hap_start)
        sample_end_list.append(hap_end)
        sample_is_ancestor_list.append(np.zeros(num_hap, dtype=bool))

    # --- Concatenate ---
    if sample_times_list:
        all_times = np.concatenate([anc_times] + sample_times_list)
        all_start = np.concatenate([anc_start] + sample_start_list)
        all_end = np.concatenate([anc_end] + sample_end_list)
        all_is_ancestor = np.concatenate([anc_is_ancestor] + sample_is_ancestor_list)
        all_source = anc_source + sample_source_list
        all_sample_id = anc_sample_id + sample_sample_id_list
        all_ploidy_index = anc_ploidy_index + sample_ploidy_index_list
    else:
        all_times = anc_times
        all_start = anc_start
        all_end = anc_end
        all_is_ancestor = anc_is_ancestor
        all_source = anc_source
        all_sample_id = anc_sample_id
        all_ploidy_index = anc_ploidy_index

    num_total = len(all_times)
    num_ancestors = int(np.sum(all_is_ancestor))
    num_samples = num_total - num_ancestors
    logger.info(
        "Collected metadata for %d haplotypes (%d ancestors, %d samples)",
        num_total,
        num_ancestors,
        num_samples,
    )

    return HaplotypeMetadata(
        times=all_times,
        is_ancestor=all_is_ancestor,
        start_positions=all_start,
        end_positions=all_end,
        source=all_source,
        sample_id=all_sample_id,
        ploidy_index=all_ploidy_index,
    )


def merge_overlapping_ancestors(start, end, time):
    """
    Merge overlapping, same-time ancestors by scanning along each time epoch
    from left to right, detecting breaks.
    """
    sort_indices = np.lexsort((start, time))
    start = start[sort_indices]
    end = end[sort_indices]
    time = time[sort_indices]
    old_indexes = {}
    new_start = np.full_like(start, -1)
    new_end = np.full_like(end, -1)
    new_time = np.full_like(time, -1)

    i = 0
    new_index_pos = 0
    while i < len(start):
        j = i + 1
        group_overlap = [i]
        max_right = end[i]
        while j < len(start) and time[j] == time[i] and start[j] < max_right:
            max_right = max(max_right, end[j])
            group_overlap.append(j)
            j += 1

        old_indexes[new_index_pos] = group_overlap
        new_start[new_index_pos] = start[i]
        new_end[new_index_pos] = max_right
        new_time[new_index_pos] = time[i]
        new_index_pos += 1
        i = j
    new_start = new_start[:new_index_pos]
    new_end = new_end[:new_index_pos]
    new_time = new_time[:new_index_pos]
    return new_start, new_end, new_time, old_indexes, sort_indices


@numba.njit
def run_linesweep(event_times, event_index, event_type, new_time):
    """
    Run the linesweep over ancestor start-stop events, building a dependency
    graph as a count of dependencies for each ancestor and a list of
    dependent children for each ancestor.
    """
    n = len(new_time)

    # numba really likes to know the type of the list elements, so we tell it by adding
    # a dummy element to the list and then popping it off.
    # `active` is the list of ancestors that overlap with the current linesweep position.
    active = [-1]
    active.pop()
    children = [[-1] for _ in range(n)]
    for c in range(n):
        children[c].pop()
    incoming_edge_count = np.zeros(n, dtype=np.int32)
    for i in range(len(event_times)):
        index = event_index[i]
        e_time = event_times[i]
        if event_type[i] == 1:
            for j in active:
                if new_time[j] > e_time:
                    incoming_edge_count[index] += 1
                    children[j].append(index)
                elif new_time[j] < e_time:
                    incoming_edge_count[j] += 1
                    children[index].append(j)
            active.append(index)
        else:
            active.remove(index)

    # Convert children to ragged array format so we can pass arrays to the
    # next numba function, `find_groups`.
    children_data = []
    children_indices = [0]
    for child_list in children:
        children_data.extend(child_list)
        children_indices.append(len(children_data))
    children_data = np.array(children_data, dtype=np.int32)
    children_indices = np.array(children_indices, dtype=np.int32)
    return children_data, children_indices, incoming_edge_count


@numba.njit
def find_groups(children_data, children_indices, incoming_edge_count):
    """
    Find groups of ancestors that can be matched in parallel by topologically
    sorting the dependency graph.
    """
    n = len(children_indices) - 1
    group_id = np.full(n, -1, dtype=np.int32)
    current_group = 0
    while True:
        no_incoming = np.where(incoming_edge_count == 0)[0]
        if len(no_incoming) == 0:
            break
        for i in no_incoming:
            incoming_edge_count[i] = -1
            incoming_edge_count[
                children_data[children_indices[i] : children_indices[i + 1]]
            ] -= 1
        group_id[no_incoming] = current_group
        current_group += 1

    if np.any(group_id == -1):
        raise ValueError(
            "Erroneous cycle in ancestor dependancies, this is often "
            "caused by too many unique site times. This fixed by discretising "
            "the site times, for example rounding times to the nearest 0.1."
        )
    return group_id


def group_ancestors_by_linesweep(start, end, time):
    """
    Group ancestors for matching in parallel using a linesweep approach.

    For each ancestor, any overlapping older ancestors must be in an earlier
    group, any overlapping younger ancestors in a later group, and any
    overlapping same-age ancestors must be in the same group so they don't
    match to each other.
    """
    assert len(start) == len(end)
    assert len(start) == len(time)
    t = time_.time()
    (
        new_start,
        new_end,
        new_time,
        old_indexes,
        sort_indices,
    ) = merge_overlapping_ancestors(start, end, time)
    logger.info(f"Merged to {len(new_start)} ancestors in {time_.time() - t:.2f}s")

    t = time_.time()
    n = len(new_time)
    event_times = np.concatenate([new_time, new_time])
    event_pos = np.concatenate([new_start, new_end])
    event_index = np.concatenate([np.arange(n), np.arange(n)])
    event_type = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    event_sort_indices = np.lexsort((event_type, event_pos))
    event_times = event_times[event_sort_indices]
    event_index = event_index[event_sort_indices]
    event_type = event_type[event_sort_indices]
    logger.info(f"Built {len(event_times)} events in {time_.time() - t:.2f}s")

    t = time_.time()
    children_data, children_indices, incoming_edge_count = run_linesweep(
        event_times, event_index, event_type, new_time
    )
    logger.info(
        f"Linesweep generated {np.sum(incoming_edge_count)} dependencies in"
        f" {time_.time() - t:.2f}s"
    )

    t = time_.time()
    group_id = find_groups(children_data, children_indices, incoming_edge_count)
    logger.info(f"Found groups in {time_.time() - t:.2f}s")

    t = time_.time()
    ancestor_grouping = {}
    for group in np.unique(group_id):
        ancestor_grouping[group] = np.where(group_id == group)[0]

    for group in ancestor_grouping:
        ancestor_grouping[group] = sorted(
            [
                sort_indices[item]
                for i in ancestor_grouping[group]
                for item in old_indexes[i]
            ]
        )
    logger.info(f"Un-merged in {time_.time() - t:.2f}s")
    logger.info(
        f"{len(ancestor_grouping)} groups with median size "
        f"{np.median([len(ancestor_grouping[group]) for group in ancestor_grouping])}"
    )
    return ancestor_grouping


# ---------------------------------------------------------------------------
# compute_groups / compute_groups_json
# ---------------------------------------------------------------------------


def compute_groups(
    times: np.ndarray,  # (n_haplotypes,) float64
    is_ancestor: np.ndarray,  # (n_haplotypes,) bool
    start_positions: np.ndarray,  # (n_haplotypes,) int32 — from sample_start_position
    end_positions: np.ndarray,  # (n_haplotypes,) int32 — from sample_end_position
) -> list[np.ndarray]:
    """
    Return an ordered list of haplotype-index arrays, oldest group first.

    - Haplotypes are ordered by descending time.
    - At the same time, ancestor groups come before sample groups.
    - Same-time ancestors with overlapping intervals are placed in the SAME
      group so they don't match against each other. Grouping is determined
      by a linesweep + topological sort that respects time dependencies.
    - Sample haplotypes are always grouped together by time (no interval
      splitting), because samples are never used as copying parents.
    """
    times = np.asarray(times, dtype=np.float64)
    is_ancestor = np.asarray(is_ancestor, dtype=bool)
    start_positions = np.asarray(start_positions, dtype=np.int32)
    end_positions = np.asarray(end_positions, dtype=np.int32)

    groups: list[np.ndarray] = []

    # Separate ancestors and samples
    anc_indices = np.where(is_ancestor)[0]
    sample_indices = np.where(~is_ancestor)[0]

    # Ancestor groups via linesweep
    if len(anc_indices) > 0:
        anc_starts = start_positions[anc_indices]
        anc_ends = end_positions[anc_indices]
        anc_times = times[anc_indices]
        ancestor_grouping = group_ancestors_by_linesweep(anc_starts, anc_ends, anc_times)
        # ancestor_grouping: {group_id: [local_indices...]} ordered by group_id
        for group_id in sorted(ancestor_grouping.keys()):
            local_ids = ancestor_grouping[group_id]
            global_ids = np.array(
                [int(anc_indices[i]) for i in local_ids], dtype=np.int32
            )
            groups.append(global_ids)

    # Sample groups: group by descending time
    if len(sample_indices) > 0:
        sample_times_set = sorted(set(times[sample_indices].tolist()), reverse=True)
        for t in sample_times_set:
            samp_at_t = sample_indices[np.isclose(times[sample_indices], t)]
            if len(samp_at_t) > 0:
                groups.append(samp_at_t.astype(np.int32))

    return groups


def compute_match_jobs(cfg: Config) -> list[MatchJob]:
    """
    Compute haplotype groups and return a flat list of :class:`MatchJob`.

    Each haplotype gets exactly one ``MatchJob`` with its identity
    (source, sample_id, ploidy_index), position range, time, and assigned
    group index.  The list is ordered by group (oldest first), then by
    haplotype index within each group — the same order the match loop
    processes them.
    """
    logger.info("Loading haplotype metadata")
    t0 = time_.monotonic()
    hap_meta = collect_haplotype_metadata(cfg)
    elapsed = time_.monotonic() - t0
    num_anc = int(np.sum(hap_meta.is_ancestor))
    num_samples = len(hap_meta.times) - num_anc
    logger.info(
        "Loaded %d haplotypes (%d ancestors, %d samples) in %.2fs",
        len(hap_meta.times),
        num_anc,
        num_samples,
        elapsed,
    )

    logger.info("Computing groups")
    t0 = time_.monotonic()
    groups = compute_groups(
        hap_meta.times,
        hap_meta.is_ancestor,
        hap_meta.start_positions,
        hap_meta.end_positions,
    )
    elapsed = time_.monotonic() - t0
    logger.info("Computed %d groups in %.2fs", len(groups), elapsed)

    # Build flat list ordered by (group, haplotype_index)
    jobs: list[MatchJob] = []
    for group_idx, group_indices in enumerate(groups):
        for idx in group_indices:
            idx = int(idx)
            jobs.append(
                MatchJob(
                    haplotype_index=idx,
                    source=hap_meta.source[idx],
                    sample_id=hap_meta.sample_id[idx],
                    ploidy_index=hap_meta.ploidy_index[idx],
                    time=float(hap_meta.times[idx]),
                    start_position=int(hap_meta.start_positions[idx]),
                    end_position=int(hap_meta.end_positions[idx]),
                    group=group_idx,
                )
            )

    return jobs


def compute_groups_json(cfg: Config) -> str:
    """
    Compute haplotype groups and return as a JSON string.

    The output is a flat JSON array of objects, one per haplotype, ordered
    by group then haplotype index.  Each object has the fields of
    :class:`MatchJob` — suitable for loading into pandas with
    ``pd.read_json`` or ``pd.DataFrame``.
    """
    t0 = time_.monotonic()
    jobs = compute_match_jobs(cfg)
    elapsed = time_.monotonic() - t0

    logger.info("Serialising %d match jobs to JSON", len(jobs))
    t0 = time_.monotonic()
    json_str = json.dumps([dataclasses.asdict(j) for j in jobs], indent=2)
    elapsed = time_.monotonic() - t0
    logger.info("Serialised in %.2fs", elapsed)
    return json_str
