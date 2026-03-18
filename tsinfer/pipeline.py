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
from pathlib import Path

import numpy as np
import tqdm
import tskit

from . import vcz as vcz_mod
from .ancestors import infer_ancestors
from .config import Config
from .grouping import (
    MatchJob,
    compute_match_jobs,
)
from .matching import Matcher, extend_ts, make_root_ts

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _IndividualMetadataResult:
    metadata: list[dict] | None
    population_indices: list[int] | None
    population_names: list[str] | None


def _load_match_jobs(path) -> list[MatchJob]:
    """Read a groups JSON file and return a list of MatchJob objects."""
    records = json.loads(Path(path).read_text())
    return [MatchJob(**rec) for rec in records]


def _setup_workdir(workdir, cfg):
    """
    Set up workdir: create dir, write or load groups.json.

    Returns (jobs, last_completed_group_idx, starting_ts_or_None).
    """

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    groups_path = workdir / "groups.json"

    if groups_path.exists():
        logger.info("Loading existing groups from %s", groups_path)
        jobs = _load_match_jobs(groups_path)
    else:
        jobs = compute_match_jobs(cfg)
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

    return jobs, max_group_index, starting_ts


def _build_individual_metadata(cfg, individual_jobs, ploidy):
    """
    Build individual metadata and population assignments from config.

    Parameters
    ----------
    cfg : Config
    individual_jobs : list[MatchJob]
        One MatchJob per individual (the first haplotype of each).
    ploidy : int

    Returns an _IndividualMetadataResult with:
    - metadata: list of dicts, one per individual
    - population_indices: list of population indices, one per individual (or None)
    - population_names: list of unique population names (or None)
    """
    ind_meta_cfg = cfg.individual_metadata
    num_individuals = len(individual_jobs)

    if num_individuals == 0:
        return _IndividualMetadataResult(
            metadata=None, population_indices=None, population_names=None
        )

    ind_metadata_list = []
    pop_indices = None
    pop_names = None

    # For each individual, find its source and sample index
    for job in individual_jobs:
        source_name = job.source
        sample_id = job.sample_id
        ind_md = {}

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
                            # Convert numpy types to Python types
                            if hasattr(val, "item"):
                                val = val.item()
                            elif hasattr(val, "tolist"):
                                val = val.tolist()
                            ind_md[field_name] = val

        ind_metadata_list.append(ind_md)

    # Build populations if configured
    if ind_meta_cfg is not None and ind_meta_cfg.population:
        pop_values = []
        for job in individual_jobs:
            source_name = job.source
            sample_id = job.sample_id
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

            pop_values.append(pop_val)

        # Build unique population list
        unique_pops = []
        for v in pop_values:
            if v is not None and v not in unique_pops:
                unique_pops.append(v)

        if unique_pops:
            pop_names = unique_pops
            pop_lookup = {name: i for i, name in enumerate(pop_names)}
            pop_indices = [pop_lookup.get(v, -1) for v in pop_values]

    return _IndividualMetadataResult(
        metadata=ind_metadata_list,
        population_indices=pop_indices,
        population_names=pop_names,
    )


def match(
    cfg: Config,
    reference_ts: tskit.TreeSequence | None = None,
    progress: bool = False,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Run the unified match loop over all sources listed in cfg.match.

    Uses the ``make_root_ts → Matcher → extend_ts`` cycle: each group of
    haplotypes is matched against the current tree sequence, then the tree
    sequence is extended with the results.
    """
    recombination_rate = kwargs.get("recombination_rate", cfg.match.recombination_rate)
    mismatch_ratio = kwargs.get("mismatch_ratio", cfg.match.mismatch_ratio)
    path_compression = kwargs.get("path_compression", cfg.match.path_compression)

    # 1. Get ordered MatchJob list (and workdir state if applicable)
    workdir = cfg.match.workdir
    last_completed_group = -1
    workdir_starting_ts = None
    if workdir is not None:
        jobs, last_completed_group, workdir_starting_ts = _setup_workdir(workdir, cfg)
    else:
        jobs = compute_match_jobs(cfg)

    # 2. Load lightweight ancestor metadata (no genotypes)
    anc_store = vcz_mod.open_store(cfg.ancestors.path)
    positions = np.asarray(anc_store["variant_position"][:], dtype=np.int32)
    site_alleles = np.asarray(anc_store["variant_allele"][:])
    seq_intervals = np.asarray(anc_store["sequence_intervals"][:], dtype=np.int32)

    # Derive seq_len from first sample source (or seq_intervals fallback)
    seq_len = None
    for source_name in cfg.match.sources:
        source = cfg.sources[source_name]
        seq_len = float(vcz_mod.sequence_length(vcz_mod.open_store(source.path)))
        break
    if seq_len is None:
        seq_len = float(np.max(seq_intervals)) if len(seq_intervals) > 0 else 1.0

    # 3. Create lazy reader
    reader = vcz_mod.HaplotypeReader(cfg.ancestors.path, cfg.sources, positions)

    # 4. Build initial root TS (or resume from workdir checkpoint)
    if workdir_starting_ts is not None:
        ts = workdir_starting_ts
    else:
        ts = make_root_ts(seq_len, positions, seq_intervals, site_alleles)

    logger.info(
        "Match: %d haplotypes, %d sites, seq_len=%.0f",
        len(jobs),
        len(positions),
        seq_len,
    )

    # 5. Derive ploidy from jobs
    sample_jobs = [j for j in jobs if j.source != "ancestors"]
    ploidy = (
        max((j.ploidy_index for j in sample_jobs), default=0) + 1 if sample_jobs else 1
    )

    # 6. Group jobs by group index
    groups_dict = collections.defaultdict(list)
    for idx, job in enumerate(jobs):
        groups_dict[job.group].append((idx, job))
    sorted_groups = sorted(groups_dict.keys())

    # 7. Match loop — process groups in order
    individual_jobs: list[MatchJob] = []  # first job of each individual
    sorted_groups = sorted(groups_dict.keys())
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

        logger.info("Group %d/%d: %d haplotypes", gi + 1, num_groups, num_in_group)

        # Match against current TS (haplotypes read on demand via reader)
        matcher = Matcher(
            ts,
            positions,
            recombination_rate=recombination_rate,
            mismatch_ratio=mismatch_ratio,
            path_compression=path_compression,
        )
        job_list = [job for _, job in group_jobs]
        match_iter = matcher.match(job_list, reader)
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

        # Build per-node arrays for extend_ts from (job, result) pairs
        node_times = []
        node_metadata = []
        results = []
        create_individuals_list = []
        current_ind_count = 0

        for job, result in paired_results:
            node_times.append(job.time)
            results.append(result)
            node_meta = {
                "source": job.source,
                "sample_id": job.sample_id,
            }
            if job.source != "ancestors":
                node_meta["ploidy_index"] = job.ploidy_index
                create_individuals_list.append(True)
                current_ind_count += 1
                if current_ind_count == 1:
                    individual_jobs.append(job)
                if current_ind_count == ploidy:
                    current_ind_count = 0
            else:
                create_individuals_list.append(False)
            node_metadata.append(node_meta)

        # Extend the tree sequence
        ts = extend_ts(
            ts,
            site_alleles=site_alleles,
            node_times=np.array(node_times, dtype=np.float64),
            results=results,
            node_metadata=node_metadata,
            create_individuals=np.array(create_individuals_list, dtype=bool),
            ploidy=ploidy,
            path_compression=path_compression,
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

    # 8. Apply individual/population metadata as post-processing
    ind_result = _build_individual_metadata(cfg, individual_jobs, ploidy)

    if ind_result.metadata is not None or ind_result.population_names is not None:
        ts = _apply_individual_metadata(ts, ind_result)

    logger.info(
        "Match complete: %d nodes, %d individuals",
        ts.num_nodes,
        ts.num_individuals,
    )

    return ts


def _apply_individual_metadata(
    ts: tskit.TreeSequence,
    ind_result: _IndividualMetadataResult,
) -> tskit.TreeSequence:
    """Apply individual metadata and population assignments to a tree sequence."""
    tables = ts.dump_tables()

    # Add populations if specified
    if ind_result.population_names is not None:
        tables.populations.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        for pop_name in ind_result.population_names:
            tables.populations.add_row(metadata={"name": pop_name})

    # Apply individual metadata
    if ind_result.metadata is not None and tables.individuals.num_rows > 0:
        tables.individuals.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        old_inds = tables.individuals.copy()
        tables.individuals.clear()
        for i in range(old_inds.num_rows):
            md = ind_result.metadata[i] if i < len(ind_result.metadata) else None
            if md is not None:
                tables.individuals.add_row(metadata=md)
            else:
                tables.individuals.add_row()

    # Apply population assignments
    if ind_result.population_indices is not None:
        pop_lookup = {
            ind_id: ind_result.population_indices[ind_id]
            for ind_id in range(len(ind_result.population_indices))
            if ind_id < tables.individuals.num_rows
        }
        for node_id in range(tables.nodes.num_rows):
            row = tables.nodes[node_id]
            if row.individual >= 0 and row.individual in pop_lookup:
                pop_id = pop_lookup[row.individual]
                if pop_id >= 0:
                    tables.nodes[node_id] = row.replace(population=pop_id)

    tables.sort()
    tables.build_index()
    return tables.tree_sequence()


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


def run(
    cfg: Config,
    progress: bool = False,
    num_threads: int = 0,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Run the full pipeline: infer_ancestors, match, post_process.
    """
    logger.info("Starting full pipeline")
    source_name = cfg.ancestors.sources[0]
    source = cfg.sources[source_name]
    ancestor_store = infer_ancestors(
        source,
        cfg.ancestors,
        cfg.ancestral_state,
        progress=progress,
        num_threads=num_threads,
    )

    original_path = cfg.ancestors.path
    cfg.ancestors.path = ancestor_store

    try:
        ts = match(cfg, progress=progress, **kwargs)
        ts = post_process(ts, cfg, **kwargs)
    finally:
        cfg.ancestors.path = original_path

    logger.info(
        "Pipeline complete: %d nodes, %d edges, %d sites",
        ts.num_nodes,
        ts.num_edges,
        ts.num_sites,
    )
    return ts
