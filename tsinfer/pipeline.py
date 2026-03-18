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

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm
import tskit

import _tsinfer

from . import vcz as vcz_mod
from .ancestors import infer_ancestors
from .config import Config
from .grouping import (
    MatchJob,
    compute_groups_json,  # noqa: F401 — re-export
    compute_match_jobs,  # noqa: F401 — re-export
)
from .matching import _ts_from_tsb

logger = logging.getLogger(__name__)


@dataclass
class _IndividualMetadataResult:
    metadata: list[dict] | None
    population_indices: list[int] | None
    population_names: list[str] | None


def _load_match_jobs(path) -> list[MatchJob]:
    """Read a groups JSON file and return a list of MatchJob objects."""
    records = json.loads(Path(path).read_text())
    return [MatchJob(**rec) for rec in records]


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

    Iterates over MatchJob objects, lazily reading one haplotype at a time
    via HaplotypeReader, and matches each against the incrementally-built
    tree sequence.
    """
    recombination_rate = kwargs.get("recombination_rate", cfg.match.recombination_rate)
    mismatch_ratio = kwargs.get("mismatch_ratio", cfg.match.mismatch_ratio)
    path_compression = kwargs.get("path_compression", cfg.match.path_compression)

    # 1. Get ordered MatchJob list
    if cfg.match.groups is not None:
        jobs = _load_match_jobs(cfg.match.groups)
    else:
        jobs = compute_match_jobs(cfg)

    # 2. Load lightweight ancestor metadata (no genotypes)
    anc_store = vcz_mod.open_store(cfg.ancestors.path)
    positions = np.asarray(anc_store["variant_position"][:], dtype=np.int32)
    seq_intervals = np.asarray(anc_store["sequence_intervals"][:], dtype=np.int32)
    num_sites = len(positions)

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

    # 4. Build TSB
    num_alleles = [2] * num_sites
    tsb = _tsinfer.TreeSequenceBuilder(num_alleles)
    metadata = {"sequence_intervals": seq_intervals.tolist()}

    logger.info(
        "Match: %d haplotypes, %d sites, seq_len=%.0f",
        len(jobs),
        num_sites,
        seq_len,
    )

    # 5. Perturb times
    match_times = [j.time for j in jobs]
    eps = 1e-10
    seen_times = {}
    for i, t in enumerate(match_times):
        if t in seen_times:
            seen_times[t] += 1
            match_times[i] = t - eps * seen_times[t]
        else:
            seen_times[t] = 0

    # 6. Derive ploidy from jobs
    sample_jobs = [j for j in jobs if j.source != "ancestors"]
    ploidy = (
        max((j.ploidy_index for j in sample_jobs), default=0) + 1 if sample_jobs else 1
    )

    # 7. Match loop — iterate jobs, read haplotypes lazily
    ordered_node_metadata = []
    individual_groups = []
    current_ind_nodes = []
    individual_jobs = []  # first job of each individual

    match_iter = enumerate(jobs)
    if progress:
        match_iter = tqdm.tqdm(
            match_iter, total=len(jobs), desc="Matching", unit="haplotypes"
        )

    for step, job in match_iter:
        hap = reader.read_haplotype(job)
        time = float(match_times[step])

        if step == 0:
            # Virtual root — add directly with no matching
            tsb.add_node(time)
            ordered_node_metadata.append(None)
        else:
            # Freeze, create matcher, match, then add
            tsb.freeze_indexes()

            d = np.diff(positions.astype(np.float64), prepend=float(positions[0]))
            rho = float(recombination_rate) * np.maximum(d, 1.0)
            rho = np.clip(rho, 1e-10, 1.0 - 1e-10)

            num_match = max(1, tsb.num_match_nodes)
            mu = np.full(num_sites, mismatch_ratio / num_match)
            mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

            matcher = _tsinfer.AncestorMatcher(tsb, rho.tolist(), mu.tolist())

            hap_arr = np.asarray(hap, dtype=np.int8)
            match_out = np.zeros(num_sites, dtype=np.int8)
            non_missing = np.where(hap_arr >= 0)[0]
            if len(non_missing) == 0:
                start, end = 0, num_sites
            else:
                start = int(non_missing[0])
                end = int(non_missing[-1]) + 1

            left, right, parent = matcher.find_path(hap_arr, start, end, match_out)

            in_range = np.zeros(num_sites, dtype=bool)
            in_range[start:end] = True
            mutation_mask = in_range & (hap_arr != match_out) & (hap_arr >= 0)
            mutation_sites = np.where(mutation_mask)[0].astype(np.int32)
            mutation_state = hap_arr[mutation_sites].astype(np.int8)

            node_meta = {
                "source": job.source,
                "sample_id": job.sample_id,
            }
            if job.source != "ancestors":
                node_meta["ploidy_index"] = job.ploidy_index

            node_id = tsb.add_node(time)
            ordered_node_metadata.append(node_meta)

            if len(left) > 0:
                tsb.add_path(
                    child=node_id,
                    left=list(left),
                    right=list(right),
                    parent=list(parent),
                    compress=path_compression,
                )

            if len(mutation_sites) > 0:
                tsb.add_mutations(
                    node=node_id,
                    site=mutation_sites.tolist(),
                    derived_state=mutation_state.tolist(),
                )

        # Track individuals — samples only (ancestors never create individuals)
        if job.source != "ancestors":
            current_ind_nodes.append(tsb.num_nodes - 1)
            if len(current_ind_nodes) == 1:
                individual_jobs.append(job)
            if len(current_ind_nodes) == ploidy:
                individual_groups.append(current_ind_nodes)
                current_ind_nodes = []

    # Handle any remaining partial individual
    if current_ind_nodes:
        individual_groups.append(current_ind_nodes)

    # Build individual metadata and populations
    ind_result = _build_individual_metadata(cfg, individual_jobs, ploidy)

    logger.info(
        "Match complete: %d nodes, %d individuals",
        tsb.num_nodes,
        len(individual_groups),
    )

    # Convert TSB to tree sequence
    ts = _ts_from_tsb(
        tsb,
        num_sites,
        positions,
        seq_len,
        metadata,
        individual_groups if individual_groups else None,
        node_metadata=ordered_node_metadata if any(ordered_node_metadata) else None,
        individual_metadata=ind_result.metadata
        if ind_result.metadata is not None
        else None,
        populations=ind_result.population_indices
        if ind_result.population_indices is not None
        else None,
        population_names=ind_result.population_names
        if ind_result.population_names is not None
        else None,
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
