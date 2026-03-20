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
High-level pipeline: match, post_process, run.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import time as time_
from pathlib import Path

import numpy as np
import tqdm
import tskit

from . import vcz as vcz_mod
from .ancestors import infer_ancestors
from .config import Config
from .grouping import (
    MatchJob,
    assign_groups,
)
from .matching import Matcher, extend_ts, make_root_ts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build jobs with metadata, pre-assign individuals and populations
# ---------------------------------------------------------------------------


def _build_jobs_with_metadata(cfg):
    """
    Build MatchJobs with pre-assigned individual_id and population_id.

    Returns (jobs, individual_metadata_rows, population_metadata_rows).
    """
    logger.info("Loading haplotype metadata")
    t0 = time_.monotonic()

    # Reference positions from first ancestor store (for default start/end)
    if len(cfg.ancestors) > 0:
        anc_store = vcz_mod.open_store(cfg.ancestors[0].path)
        positions = np.asarray(anc_store["variant_position"][:], dtype=np.int32)
    else:
        positions = np.array([], dtype=np.int32)

    # Build MatchJobs directly from VCZ stores, assigning individual IDs
    # in the same pass.
    jobs: list[MatchJob] = []
    ind_key_to_id: dict[tuple[str, str], int] = {}
    hap_idx = 0

    for source_name in cfg.match.sources:
        source = cfg.sources[source_name]
        src_cfg = cfg.match.sources[source_name]
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
            store,
            source.sample_time,
            "sample_id",
            all_num_samples,
            fill_value=0,
        )
        if sample_time_arr is None:
            sample_time_arr = np.zeros(num_samples, dtype=np.float64)
        else:
            sample_time_arr = np.asarray(sample_time_arr, dtype=np.float64)
            if samples_selection is not None:
                sample_time_arr = sample_time_arr[samples_selection]

        # sample_id
        raw_ids = store["sample_id"][:]
        if samples_selection is not None:
            raw_ids = raw_ids[samples_selection]
        sample_ids = [str(x) for x in raw_ids.tolist()]

        # start/end positions
        if "sample_start_position" in store and "sample_end_position" in store:
            src_start = np.asarray(store["sample_start_position"][:], dtype=np.int32)
            src_end = np.asarray(store["sample_end_position"][:], dtype=np.int32)
        elif len(positions) > 0:
            default_start = int(positions[0])
            default_end = int(positions[-1])
            src_start = None
            src_end = None
        else:
            default_start = 0
            default_end = 0
            src_start = None
            src_end = None

        # Build one MatchJob per haplotype
        for i in range(num_samples):
            sample_time = float(sample_time_arr[i])
            sid = sample_ids[i]

            if src_start is not None:
                start_pos = int(src_start[i])
                end_pos = int(src_end[i])
            else:
                start_pos = default_start
                end_pos = default_end

            # Determine individual_id
            if src_cfg.create_individuals:
                key = (source.name, sid)
                if key not in ind_key_to_id:
                    ind_key_to_id[key] = len(ind_key_to_id)
                individual_id = ind_key_to_id[key]
            else:
                individual_id = None

            for p in range(ploidy):
                jobs.append(
                    MatchJob(
                        haplotype_index=hap_idx,
                        source=source.name,
                        sample_id=sid,
                        ploidy_index=p,
                        time=sample_time,
                        start_position=start_pos,
                        end_position=end_pos,
                        group=0,
                        node_flags=src_cfg.node_flags,
                        individual_id=individual_id,
                        population_id=None,
                    )
                )
                hap_idx += 1

    elapsed = time_.monotonic() - t0
    logger.info("Built %d match jobs in %.2fs", len(jobs), elapsed)

    # Build individual metadata rows
    ind_meta_cfg = cfg.individual_metadata
    individual_metadata_rows: list[dict] = []
    ind_keys_ordered = sorted(ind_key_to_id.keys(), key=lambda k: ind_key_to_id[k])
    for source_name, sample_id in ind_keys_ordered:
        ind_md = {"source": source_name, "sample_id": sample_id}

        # Add configured metadata fields
        if ind_meta_cfg is not None and ind_meta_cfg.fields:
            if source_name and source_name in cfg.sources:
                source = cfg.sources[source_name]
                store = vcz_mod.open_store(source.path)
                raw_ids = store["sample_id"][:]
                sample_ids = [str(x) for x in raw_ids.tolist()]
                try:
                    sample_idx = sample_ids.index(sample_id)
                except ValueError:
                    sample_idx = -1

                if sample_idx >= 0:
                    for field_name, array_name in ind_meta_cfg.fields.items():
                        if array_name in store:
                            val = store[array_name][sample_idx]
                            if hasattr(val, "item"):
                                val = val.item()
                            elif hasattr(val, "tolist"):
                                val = val.tolist()
                            ind_md[field_name] = val

        individual_metadata_rows.append(ind_md)

    # Build populations if configured
    population_metadata_rows: list[dict] = []

    if ind_meta_cfg is not None and ind_meta_cfg.population:
        # For each individual, determine its population
        ind_pop_values: list[str | None] = []
        for source_name, sample_id in ind_keys_ordered:
            pop_val = None
            if source_name and source_name in cfg.sources:
                source = cfg.sources[source_name]
                store = vcz_mod.open_store(source.path)
                pop_field = ind_meta_cfg.population

                if pop_field in store:
                    raw_ids = store["sample_id"][:]
                    sample_ids = [str(x) for x in raw_ids.tolist()]
                    try:
                        sample_idx = sample_ids.index(sample_id)
                    except ValueError:
                        sample_idx = -1
                    if sample_idx >= 0:
                        val = store[pop_field][sample_idx]
                        if hasattr(val, "item"):
                            val = val.item()
                        elif hasattr(val, "tolist"):
                            val = val.tolist()
                        pop_val = str(val)
            ind_pop_values.append(pop_val)

        # Build unique population list
        unique_pops: list[str] = []
        for v in ind_pop_values:
            if v is not None and v not in unique_pops:
                unique_pops.append(v)

        if unique_pops:
            pop_lookup = {name: i for i, name in enumerate(unique_pops)}
            population_metadata_rows = [{"name": name} for name in unique_pops]

            # Map individual → population
            ind_to_pop: dict[int, int] = {}
            for ind_idx, pop_val in enumerate(ind_pop_values):
                if pop_val is not None and pop_val in pop_lookup:
                    ind_to_pop[ind_idx] = pop_lookup[pop_val]

            # Map population onto jobs via individual_id
            for job in jobs:
                if job.individual_id is not None and job.individual_id in ind_to_pop:
                    job.population_id = ind_to_pop[job.individual_id]

    # Assign groups
    logger.info("Computing groups")
    t0 = time_.monotonic()
    jobs = assign_groups(jobs)
    elapsed = time_.monotonic() - t0
    num_groups = len({j.group for j in jobs})
    logger.info("Computed %d groups in %.2fs", num_groups, elapsed)

    return jobs, individual_metadata_rows, population_metadata_rows


def compute_groups_json(cfg: Config) -> str:
    """
    Compute haplotype groups and return as a JSON string.

    The output is a flat JSON array of objects, one per haplotype, ordered
    by group then haplotype index.  Each object has the fields of
    :class:`MatchJob` — suitable for loading into pandas with
    ``pd.read_json`` or ``pd.DataFrame``.
    """
    t0 = time_.monotonic()
    jobs, _, _ = _build_jobs_with_metadata(cfg)
    elapsed = time_.monotonic() - t0

    logger.info("Serialising %d match jobs to JSON", len(jobs))
    t0 = time_.monotonic()
    json_str = json.dumps([dataclasses.asdict(j) for j in jobs], indent=2)
    elapsed = time_.monotonic() - t0
    logger.info("Serialised in %.2fs", elapsed)
    return json_str


def _load_match_jobs(path) -> list[MatchJob]:
    """Read a match-jobs JSON file and return a list of MatchJob objects."""
    records = json.loads(Path(path).read_text())
    return [MatchJob(**rec) for rec in records]


def _setup_workdir(workdir, cfg):
    """
    Set up workdir: create dir, write or load match-jobs.json.

    Returns (jobs, individual_metadata_rows, population_metadata_rows,
             last_completed_group_idx, starting_ts_or_None).
    """

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    groups_path = workdir / "match-jobs.json"

    if groups_path.exists():
        logger.info("Loading existing groups from %s", groups_path)
        jobs = _load_match_jobs(groups_path)
        individual_metadata_rows: list[dict] = []
        population_metadata_rows: list[dict] = []
    else:
        jobs, individual_metadata_rows, population_metadata_rows = (
            _build_jobs_with_metadata(cfg)
        )
        json_str = json.dumps([dataclasses.asdict(j) for j in jobs], indent=2)
        groups_path.write_text(json_str)
        logger.info("Wrote groups to %s", groups_path)

    max_group_index = -1
    for path in workdir.glob("group_*.trees"):
        group_index = int(path.stem.split("_")[1])
        max_group_index = max(group_index, max_group_index)

    starting_ts = None
    if max_group_index >= 0:
        last_path = workdir / f"group_{max_group_index}.trees"
        starting_ts = tskit.load(str(last_path))
        logger.info("Resuming from group %d ", max_group_index)

    return (
        jobs,
        individual_metadata_rows,
        population_metadata_rows,
        max_group_index,
        starting_ts,
    )


def match(
    cfg: Config,
    reference_ts: tskit.TreeSequence | None = None,
    progress: bool = False,
    num_threads: int = 0,
    cache_size: int = 256,
    group_stop: int | None = None,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Run the unified match loop over all sources listed in cfg.match.

    Uses the ``make_root_ts → Matcher → extend_ts`` cycle: each group of
    haplotypes is matched against the current tree sequence, then the tree
    sequence is extended with the results.
    """
    path_compression = kwargs.get("path_compression", cfg.match.path_compression)

    # Validate group_stop
    if group_stop is not None and group_stop < 0:
        raise ValueError("group_stop must be non-negative")
    if group_stop is not None and cfg.match.workdir is None:
        logger.warning("group_stop without workdir; partial result cannot be resumed")

    # 1. Get ordered MatchJob list (and workdir state if applicable)
    workdir = cfg.match.workdir
    last_completed_group = -1
    workdir_starting_ts = None
    individual_metadata_rows: list[dict] = []
    population_metadata_rows: list[dict] = []
    if workdir is not None:
        (
            jobs,
            individual_metadata_rows,
            population_metadata_rows,
            last_completed_group,
            workdir_starting_ts,
        ) = _setup_workdir(workdir, cfg)
    else:
        jobs, individual_metadata_rows, population_metadata_rows = (
            _build_jobs_with_metadata(cfg)
        )

    # 2. Load lightweight ancestor metadata (no genotypes)
    anc_store = vcz_mod.open_store(cfg.ancestors[0].path)
    positions = np.asarray(anc_store["variant_position"][:], dtype=np.int32)
    site_alleles = np.asarray(anc_store["variant_allele"][:])
    seq_intervals = np.asarray(anc_store["sequence_intervals"][:], dtype=np.int32)

    # Derive seq_len from first non-ancestor sample source (or seq_intervals)
    ancestor_names = {anc.name for anc in cfg.ancestors}
    seq_len = None
    for source_name in cfg.match.sources:
        if source_name not in ancestor_names:
            source = cfg.sources[source_name]
            seq_len = float(vcz_mod.sequence_length(vcz_mod.open_store(source.path)))
            break
    if seq_len is None:
        seq_len = float(np.max(seq_intervals)) if len(seq_intervals) > 0 else 1.0

    # 3. Create lazy reader with all match sources
    ancestral_alleles = site_alleles[:, 0]
    match_sources = {name: cfg.sources[name] for name in cfg.match.sources}
    reader = vcz_mod.HaplotypeReader(
        match_sources,
        positions,
        ancestral_alleles,
        cache_size_mb=cache_size,
    )
    allele_mapper = reader.allele_mapper

    # 4. Build initial root TS (or resume from workdir checkpoint)
    anc_times = np.asarray(anc_store["sample_time"][:], dtype=np.float64)
    max_anc_time = float(np.max(anc_times)) if len(anc_times) > 0 else 0.0
    if workdir_starting_ts is not None:
        ts = workdir_starting_ts
    else:
        ts = make_root_ts(
            seq_len,
            positions,
            seq_intervals,
            max_time=max_anc_time,
            individuals=individual_metadata_rows if individual_metadata_rows else None,
            populations=population_metadata_rows if population_metadata_rows else None,
            allele_mapper=allele_mapper,
        )

    logger.info(
        "Match: %d haplotypes, %d sites, seq_len=%.0f",
        len(jobs),
        len(positions),
        seq_len,
    )

    # 5. Group jobs by group index
    groups_dict = collections.defaultdict(list)
    for idx, job in enumerate(jobs):
        groups_dict[job.group].append((idx, job))
    sorted_groups = sorted(groups_dict.keys())

    # 6. Match loop — process groups in order
    num_groups = len(sorted_groups)
    total_haps = len(jobs)
    completed_haps = 0

    prev_written_group = None
    for gi, group_idx in enumerate(sorted_groups):
        group_jobs = groups_dict[group_idx]
        num_in_group = len(group_jobs)

        # Skip groups already completed
        if group_idx <= last_completed_group:
            completed_haps += num_in_group
            logger.info("Group %d/%d: skipped (already completed)", gi + 1, num_groups)
            continue

        # Stop before reaching group_stop (range semantics)
        if group_stop is not None and group_idx >= group_stop:
            logger.info(
                "Stopping at group %d (group_stop=%d)",
                group_idx,
                group_stop,
            )
            break

        logger.info("Group %d/%d: %d haplotypes", gi + 1, num_groups, num_in_group)

        # Match against current TS (haplotypes read on demand via reader)
        matcher = Matcher(
            ts,
            positions,
            path_compression=path_compression,
            num_alleles=reader.get_num_alleles(),
            allele_mapper=allele_mapper,
        )
        job_list = [job for _, job in group_jobs]
        match_iter = matcher.match(job_list, reader, num_threads=num_threads)
        if progress:
            match_iter = tqdm.tqdm(
                match_iter,
                total=len(job_list),
                desc=f"Group {gi + 1}/{num_groups}",
                unit="haplotypes",
            )
        paired_results = sorted(match_iter, key=lambda pair: pair[0].haplotype_index)

        completed_haps += num_in_group
        logger.info(
            "Group %d/%d done: %d/%d haplotypes completed (%.1f%%)",
            gi + 1,
            num_groups,
            completed_haps,
            total_haps,
            100.0 * completed_haps / total_haps if total_haps > 0 else 0,
        )

        # Extend the tree sequence
        ts = extend_ts(
            ts,
            paired_results=paired_results,
            allele_mapper=allele_mapper,
        )

        # Write checkpoint to workdir if configured
        if workdir is not None:
            wd = Path(workdir)
            ts_path = wd / f"group_{group_idx}.trees"
            ts.dump(str(ts_path))
            logger.info("Wrote checkpoint to %s", ts_path)
            # Clean up previous group file unless keep_intermediates
            if not cfg.match.keep_intermediates and prev_written_group is not None:
                prev_path = wd / f"group_{prev_written_group}.trees"
                if prev_path.exists():
                    prev_path.unlink()
            prev_written_group = group_idx

    logger.info(
        "Match complete: %d nodes, %d individuals",
        ts.num_nodes,
        ts.num_individuals,
    )

    return ts


def post_process(
    ts: tskit.TreeSequence,
    cfg: Config,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Post-process a matched tree sequence.
    """
    pp = cfg.post_process
    if pp is None:
        return ts

    erase_flanks = kwargs.get("erase_flanks", pp.erase_flanks)
    split_ultimate = kwargs.get("split_ultimate", pp.split_ultimate)

    if erase_flanks:
        ts = _erase_flanks(ts)

    if split_ultimate:
        ts = _split_ultimate(ts)

    ts = ts.simplify()
    return ts


def _erase_flanks(ts: tskit.TreeSequence) -> tskit.TreeSequence:
    """Clip edges to the union of sequence_intervals from metadata."""
    meta = ts.metadata if ts.metadata is not None else {}
    intervals = meta.get("sequence_intervals")
    if intervals is None:
        return ts

    tables = ts.dump_tables()
    edges = tables.edges.copy()
    tables.edges.clear()

    for edge in edges:
        left = edge.left
        right = edge.right
        for iv_start, iv_end in intervals:
            clip_left = max(left, float(iv_start))
            clip_right = min(right, float(iv_end))
            if clip_left < clip_right:
                tables.edges.add_row(
                    left=clip_left,
                    right=clip_right,
                    parent=edge.parent,
                    child=edge.child,
                )

    tables.sort()
    tables.build_index()
    return tables.tree_sequence()


def _split_ultimate(ts: tskit.TreeSequence) -> tskit.TreeSequence:
    """Split ultimate ancestor nodes. No-op for now."""
    # TODO: implement ultimate ancestor splitting
    return ts


def augment_sites(
    ts: tskit.TreeSequence,
    cfg: Config,
    progress: bool = False,
) -> tskit.TreeSequence:
    """
    Place additional sites onto an inferred tree sequence using parsimony.

    For each source listed in ``cfg.augment_sites.sources``, iterates over
    variants not already present in *ts* (and within its sequence_intervals),
    then uses ``tree.map_mutations`` to place them parsimoniously.

    Sites that appear at duplicate positions within a single source are
    excluded (same behaviour as ``infer_ancestors``).  The ancestral allele
    is read from ``variant_ancestral_allele`` (or the ``[ancestral_state]``
    config section) and passed to ``map_mutations`` so that the output
    always uses the specified ancestral state.
    """
    aug_cfg = cfg.augment_sites
    if aug_cfg is None or len(aug_cfg.sources) == 0:
        return ts

    logger.info("augment_sites: %d source(s)", len(aug_cfg.sources))

    # --- sequence_intervals membership ---
    meta = ts.metadata if ts.metadata is not None else {}
    intervals = meta.get("sequence_intervals")
    if intervals is not None:
        iv_arr = np.array(intervals, dtype=np.float64)
        iv_starts = iv_arr[:, 0]
        iv_ends = iv_arr[:, 1]
    else:
        iv_starts = np.array([0.0])
        iv_ends = np.array([ts.sequence_length])

    existing = set(ts.sites_position)

    # --- Build ancestral state lookup (same logic as _compute_inference_sites) ---
    ann_lookup = None
    if cfg.ancestral_state is not None:
        ann_store = vcz_mod.open_store(cfg.ancestral_state.path)
        ann_positions = np.asarray(ann_store["variant_position"][:])
        ann_values = np.asarray(ann_store[cfg.ancestral_state.field][:])
        ann_lookup = {
            int(k): str(v) for k, v in zip(ann_positions.tolist(), ann_values.tolist())
        }

    # --- Detect duplicate positions per source ---
    # Positions that appear more than once in a single source are excluded,
    # matching the behaviour of infer_ancestors.
    source_names = aug_cfg.sources
    source_configs = []
    per_source_dup_positions: list[set[int]] = []
    for src_name in source_names:
        source = cfg.sources[src_name]
        source_configs.append(source)
        store = vcz_mod.open_store(source.path)
        all_pos = np.asarray(store["variant_position"][:], dtype=np.int32)
        unique, counts = np.unique(all_pos, return_counts=True)
        dup_mask = counts > 1
        dup_set = set(unique[dup_mask].tolist()) if np.any(dup_mask) else set()
        if dup_set:
            logger.info(
                "augment_sites: source '%s' has %d duplicate position(s); "
                "these will be excluded",
                src_name,
                len(dup_set),
            )
        per_source_dup_positions.append(dup_set)

    # --- Pass 1: collect new positions from each source ---
    # Each entry: (position, alleles_tuple, ancestral_allele_str, source_index)
    pos_to_info: dict[int, tuple] = {}
    for src_idx, _src_name in enumerate(source_names):
        source = source_configs[src_idx]
        store = vcz_mod.open_store(source.path)
        dup_positions = per_source_dup_positions[src_idx]
        fields = ["variant_position", "variant_allele"]
        if ann_lookup is None:
            fields.append("variant_ancestral_allele")
        for var in vcz_mod.iter_variants(
            store,
            fields=fields,
            include=source.include,
            exclude=source.exclude,
            regions=source.regions,
            targets=source.targets,
        ):
            pos = int(var["variant_position"])
            if pos in dup_positions:
                continue
            if pos in existing:
                continue
            if not _position_in_intervals(pos, iv_starts, iv_ends):
                continue
            if pos not in pos_to_info:
                alleles_raw = var["variant_allele"]
                alleles = tuple(str(a) for a in alleles_raw if str(a) != "")
                if ann_lookup is not None:
                    anc_str = ann_lookup.get(pos, "")
                else:
                    anc_str = str(var["variant_ancestral_allele"])
                pos_to_info[pos] = (alleles, anc_str, src_idx)

    if len(pos_to_info) == 0:
        logger.info("augment_sites: no new sites to add")
        return ts

    # Build sorted unified position list and per-source masks
    sorted_positions = sorted(pos_to_info.keys())
    n_sites = len(sorted_positions)
    n_sources = len(source_names)

    site_alleles = []  # alleles tuple per unified site
    site_ancestral = []  # ancestral allele string per unified site
    source_has_site = np.zeros((n_sites, n_sources), dtype=bool)
    # Track which sources have each position
    # Re-scan all sources to build the full source_has_site mask
    source_positions_set: list[set[int]] = [set() for _ in range(n_sources)]
    for src_idx, src_name in enumerate(source_names):
        source = cfg.sources[src_name]
        store = vcz_mod.open_store(source.path)
        for var in vcz_mod.iter_variants(
            store,
            fields=["variant_position"],
            include=source.include,
            exclude=source.exclude,
            regions=source.regions,
            targets=source.targets,
        ):
            pos = int(var["variant_position"])
            if pos in pos_to_info:
                source_positions_set[src_idx].add(pos)

    for ui, pos in enumerate(sorted_positions):
        info = pos_to_info[pos]
        site_alleles.append(info[0])
        site_ancestral.append(info[1])
        for src_idx in range(n_sources):
            source_has_site[ui, src_idx] = pos in source_positions_set[src_idx]

    positions_arr = np.array(sorted_positions, dtype=np.int32)
    logger.info("augment_sites: %d new sites to place", n_sites)

    # --- Build sample node map ---
    # Map (sample_id, ploidy_index) -> ts sample array index
    sample_nodes = ts.samples()
    node_to_sample_idx = {int(n): i for i, n in enumerate(sample_nodes)}

    # Build lookup: (sample_id, ploidy_index) -> ts_sample_idx
    sid_ploidy_to_ts_idx: dict[tuple[str, int], int] = {}
    for node_id in sample_nodes:
        md = ts.node(node_id).metadata
        if md and "sample_id" in md and "ploidy_index" in md:
            key = (md["sample_id"], md["ploidy_index"])
            sid_ploidy_to_ts_idx[key] = node_to_sample_idx[int(node_id)]

    # Per-source: list of (vcz_flat_col, ts_sample_idx) pairs
    per_source_col_map: list[list[tuple[int, int]]] = []
    per_source_num_haps: list[int] = []
    per_source_sample_include: list[np.ndarray | None] = []
    for _src_idx, src_name in enumerate(source_names):
        source = cfg.sources[src_name]
        store = vcz_mod.open_store(source.path)
        gt_shape = store["call_genotype"].shape
        num_vcz_samples = gt_shape[1]
        ploidy = gt_shape[2]

        samples_selection = vcz_mod.resolve_samples_selection(store, source.samples)
        if samples_selection is not None:
            selected_samples = len(samples_selection)
            sample_include = np.zeros(num_vcz_samples, dtype=bool)
            sample_include[samples_selection] = True
        else:
            selected_samples = num_vcz_samples
            sample_include = None

        raw_ids = store["sample_id"][:]
        if samples_selection is not None:
            raw_ids = raw_ids[samples_selection]

        col_map = []
        for i in range(selected_samples):
            sid = str(raw_ids[i])
            for p in range(ploidy):
                flat_col = i * ploidy + p
                key = (sid, p)
                if key in sid_ploidy_to_ts_idx:
                    col_map.append((flat_col, sid_ploidy_to_ts_idx[key]))

        per_source_col_map.append(col_map)
        per_source_num_haps.append(selected_samples * ploidy)
        per_source_sample_include.append(sample_include)

    num_ts_samples = len(sample_nodes)

    # --- Pass 2: stream genotypes and place via map_mutations ---
    # Build per-source genotype iterators
    source_iters = []
    for src_idx, src_name in enumerate(source_names):
        source = cfg.sources[src_name]
        store = vcz_mod.open_store(source.path)
        mask = source_has_site[:, src_idx]
        src_positions = positions_arr[mask]
        if len(src_positions) > 0:
            it = iter(
                vcz_mod.iter_raw_genotypes(
                    store,
                    src_positions,
                    per_source_sample_include[src_idx],
                )
            )
        else:
            it = iter([])
        source_iters.append(it)

    tables = ts.dump_tables()
    tree = ts.first()

    site_iter = range(n_sites)
    if progress:
        site_iter = tqdm.tqdm(
            site_iter, total=n_sites, desc="augment_sites", unit="sites"
        )

    for ui in site_iter:
        # Merge genotypes from all sources
        genotypes = np.full(num_ts_samples, tskit.MISSING_DATA, dtype=np.int32)
        for src_idx in range(n_sources):
            if source_has_site[ui, src_idx]:
                raw_row = next(source_iters[src_idx])
                col_map = per_source_col_map[src_idx]
                for flat_col, ts_sample_idx in col_map:
                    val = int(raw_row[flat_col])
                    if val >= 0:
                        genotypes[ts_sample_idx] = val

        # Skip all-missing sites (no genotype data at all)
        non_missing = genotypes[genotypes >= 0]
        if len(non_missing) == 0:
            continue

        alleles = list(site_alleles[ui])
        anc_allele = site_ancestral[ui]
        pos = float(sorted_positions[ui])
        tree.seek(pos)
        # Use specified ancestral state if it matches a known allele;
        # otherwise let parsimony choose.
        if anc_allele in alleles:
            ancestral_state, mutations = tree.map_mutations(
                genotypes, alleles, ancestral_state=anc_allele
            )
        else:
            ancestral_state, mutations = tree.map_mutations(genotypes, alleles)
        site_id = tables.sites.add_row(position=pos, ancestral_state=ancestral_state)
        mut_id_map = {tskit.NULL: tskit.NULL}
        for list_id, mut in enumerate(mutations):
            new_id = tables.mutations.add_row(
                site=site_id,
                node=mut.node,
                derived_state=mut.derived_state,
                parent=mut_id_map[mut.parent],
            )
            mut_id_map[list_id] = new_id

    tables.sort()
    tables.build_index()
    result = tables.tree_sequence()
    n_added = result.num_sites - ts.num_sites
    logger.info(
        "augment_sites complete: %d sites total (%d added, %d skipped)",
        result.num_sites,
        n_added,
        n_sites - n_added,
    )
    return result


def _position_in_intervals(position, starts, ends):
    """Check if position falls within any [start, end) interval."""
    idx = np.searchsorted(starts, position, side="right") - 1
    if idx < 0:
        return False
    return position < ends[idx]


def run(
    cfg: Config,
    progress: bool = False,
    num_threads: int = 0,
    cache_size: int = 256,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Run the full pipeline: infer_ancestors, match, post_process, augment_sites.
    """
    logger.info("Starting full pipeline")
    anc_cfg = cfg.ancestors[0]
    sources = [cfg.sources[name] for name in anc_cfg.sources]
    ancestor_store = infer_ancestors(
        sources,
        anc_cfg,
        cfg.ancestral_state,
        progress=progress,
        num_threads=num_threads,
    )

    original_path = anc_cfg.path
    anc_cfg.path = ancestor_store
    cfg.sources[anc_cfg.name] = cfg.sources[anc_cfg.name].__class__(
        path=ancestor_store,
        name=anc_cfg.name,
        sample_time="sample_time",
    )

    try:
        ts = match(
            cfg,
            progress=progress,
            num_threads=num_threads,
            cache_size=cache_size,
            **kwargs,
        )
        ts = post_process(ts, cfg, **kwargs)
        if cfg.augment_sites is not None:
            ts = augment_sites(ts, cfg, progress=progress)
    finally:
        anc_cfg.path = original_path

    logger.info(
        "Pipeline complete: %d nodes, %d edges, %d sites",
        ts.num_nodes,
        ts.num_edges,
        ts.num_sites,
    )
    return ts
