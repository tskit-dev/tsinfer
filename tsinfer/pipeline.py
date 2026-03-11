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

import logging
from dataclasses import dataclass

import numpy as np
import tskit

import _tsinfer

from . import vcz as vcz_mod
from .ancestors import infer_ancestors
from .config import Config
from .matching import (
    _ts_from_tsb,
)

logger = logging.getLogger(__name__)


@dataclass
class _HaplotypeData:
    positions: np.ndarray
    haplotypes: np.ndarray
    times: np.ndarray
    is_ancestor: np.ndarray
    start_positions: np.ndarray
    end_positions: np.ndarray
    metadata: list[dict]
    create_individuals: np.ndarray
    ploidy: int
    seq_intervals: np.ndarray
    seq_len: float


@dataclass
class _IndividualMetadataResult:
    metadata: list[dict] | None
    population_indices: list[int] | None
    population_names: list[str] | None


def _collect_haplotypes(cfg: Config):
    """
    Load ancestor VCZ and all sample sources; return arrays needed by the
    match loop.
    """
    # --- Load ancestors ---
    anc_store = vcz_mod.open_store(cfg.ancestors.path)
    positions = np.asarray(anc_store["variant_position"][:], dtype=np.int32)
    num_sites = len(positions)
    seq_intervals = np.asarray(anc_store["sequence_intervals"][:], dtype=np.int32)

    anc_gt = np.asarray(anc_store["call_genotype"][:, :, 0], dtype=np.int8)
    num_anc = anc_gt.shape[1]
    anc_times = np.asarray(anc_store["sample_time"][:], dtype=np.float64)
    anc_start = np.asarray(anc_store["sample_start_position"][:], dtype=np.int32)
    anc_end = np.asarray(anc_store["sample_end_position"][:], dtype=np.int32)
    anc_ids = [str(x) for x in anc_store["sample_id"][:].tolist()]

    anc_haplotypes = anc_gt.T  # (num_anc, num_sites)

    # Prepend the virtual root (all-ancestral haplotype, time=1.0).
    # The ancestor VCZ no longer contains it; it is a matching concern.
    virtual_hap = np.zeros((1, num_sites), dtype=np.int8)
    if num_sites > 0:
        virtual_start = np.array([int(positions[0])], dtype=np.int32)
        virtual_end = np.array([int(positions[-1])], dtype=np.int32)
    else:
        virtual_start = np.zeros(1, dtype=np.int32)
        virtual_end = np.zeros(1, dtype=np.int32)

    anc_haplotypes = np.concatenate([virtual_hap, anc_haplotypes], axis=0)
    anc_times = np.concatenate([np.array([1.0], dtype=np.float64), anc_times])
    anc_start = np.concatenate([virtual_start, anc_start])
    anc_end = np.concatenate([virtual_end, anc_end])

    total_anc = num_anc + 1  # including virtual root
    anc_metadata = [{"source": "ancestors", "sample_id": "virtual_root"}]
    anc_metadata.extend(
        {"source": "ancestors", "sample_id": anc_ids[i]} for i in range(num_anc)
    )
    anc_is_ancestor = np.ones(total_anc, dtype=bool)
    anc_create_ind = np.zeros(total_anc, dtype=bool)

    # --- Load sample sources ---
    sample_haplotypes_list = []
    sample_times_list = []
    sample_start_list = []
    sample_end_list = []
    sample_metadata_list = []
    sample_create_ind_list = []
    ploidy = 1
    seq_len = None

    for source_name in cfg.match.sources:
        source = cfg.sources[source_name]
        store = vcz_mod.open_store(source.path)

        call_gt = np.asarray(store["call_genotype"][:], dtype=np.int8)
        num_src_sites, num_samples, src_ploidy = call_gt.shape
        ploidy = src_ploidy

        samples_selection = vcz_mod.resolve_samples_selection(store, source.samples)
        if samples_selection is not None:
            call_gt = call_gt[:, samples_selection, :]
            num_samples = len(samples_selection)

        num_hap = num_samples * src_ploidy

        # Load sample_time, then filter to selected samples
        all_num_samples = store["call_genotype"].shape[1]
        sample_time_arr = vcz_mod.resolve_field(
            store, source.sample_time, "sample_id", all_num_samples, fill_value=0
        )
        if sample_time_arr is None:
            sample_time_arr = np.zeros(num_samples, dtype=np.float64)
        else:
            sample_time_arr = np.asarray(sample_time_arr, dtype=np.float64)
            if samples_selection is not None:
                sample_time_arr = sample_time_arr[samples_selection]

        raw_ids = store["sample_id"][:]
        if samples_selection is not None:
            raw_ids = raw_ids[samples_selection]
        sample_ids = [str(x) for x in raw_ids.tolist()]

        src_positions = np.asarray(store["variant_position"][:], dtype=np.int32)
        src_alleles = np.asarray(store["variant_allele"][:])
        if "variant_ancestral_allele" in store:
            src_anc_state = np.asarray(store["variant_ancestral_allele"][:])
        else:
            src_anc_state = np.array([str(a[0]) for a in src_alleles], dtype=object)

        src_anc_idx = np.full(num_src_sites, -1, dtype=np.int8)
        for i in range(num_src_sites):
            for j, a in enumerate(src_alleles[i].tolist()):
                if a and str(a) == str(src_anc_state[i]):
                    src_anc_idx[i] = j
                    break

        src_pos_to_idx = {int(p): i for i, p in enumerate(src_positions.tolist())}
        call_gt_flat = call_gt.reshape(num_src_sites, num_hap)

        hap_matrix = np.full((num_sites, num_hap), np.int8(-1), dtype=np.int8)
        for site_idx, pos in enumerate(positions.tolist()):
            if pos in src_pos_to_idx:
                src_idx = src_pos_to_idx[pos]
                ai = int(src_anc_idx[src_idx])
                if ai < 0:
                    continue
                gt_row = call_gt_flat[src_idx]
                hap_matrix[site_idx] = np.where(
                    gt_row < 0,
                    np.int8(-1),
                    np.where(gt_row == ai, np.int8(0), np.int8(1)),
                ).astype(np.int8)

        hap_t = hap_matrix.T
        hap_times = np.repeat(sample_time_arr, src_ploidy)
        hap_start = np.full(num_hap, int(positions[0]), dtype=np.int32)
        hap_end = np.full(num_hap, int(positions[-1]), dtype=np.int32)

        for i in range(num_samples):
            for p in range(src_ploidy):
                sample_metadata_list.append(
                    {
                        "source": source.name,
                        "sample_id": sample_ids[i],
                        "ploidy_index": p,
                    }
                )

        sample_haplotypes_list.append(hap_t)
        sample_times_list.append(hap_times)
        sample_start_list.append(hap_start)
        sample_end_list.append(hap_end)
        sample_create_ind_list.append(np.ones(num_hap, dtype=bool))

        if seq_len is None:
            seq_len = float(vcz_mod.sequence_length(store))

    # --- Concatenate ---
    if sample_haplotypes_list:
        all_haplotypes = np.concatenate(
            [anc_haplotypes] + sample_haplotypes_list, axis=0
        )
        all_times = np.concatenate([anc_times] + sample_times_list)
        all_start = np.concatenate([anc_start] + sample_start_list)
        all_end = np.concatenate([anc_end] + sample_end_list)
        all_metadata = anc_metadata + sample_metadata_list
        all_is_ancestor = np.concatenate(
            [anc_is_ancestor] + [np.zeros(len(t), dtype=bool) for t in sample_times_list]
        )
        all_create_ind = np.concatenate([anc_create_ind] + sample_create_ind_list)
    else:
        all_haplotypes = anc_haplotypes
        all_times = anc_times
        all_start = anc_start
        all_end = anc_end
        all_metadata = anc_metadata
        all_is_ancestor = anc_is_ancestor
        all_create_ind = anc_create_ind

    if seq_len is None:
        seq_len = float(np.max(seq_intervals)) if len(seq_intervals) > 0 else 1.0

    return _HaplotypeData(
        positions=positions,
        haplotypes=all_haplotypes,
        times=all_times,
        is_ancestor=all_is_ancestor,
        start_positions=all_start,
        end_positions=all_end,
        metadata=all_metadata,
        create_individuals=all_create_ind,
        ploidy=ploidy,
        seq_intervals=seq_intervals,
        seq_len=float(seq_len),
    )


def _order_haplotypes(times, is_ancestor):
    """
    Return an ordering of haplotype indices for sequential matching.

    Order: virtual root first (index 0), then all other haplotypes sorted
    by descending time. Within the same time level, ancestors come before
    samples. Ties within ancestors at the same time are broken by index
    to ensure deterministic ordering.

    The key constraint: the C ancestor matcher requires that node 0 (virtual
    root) has at most one child. By processing one haplotype at a time in
    strict time order, each haplotype becomes a child of a previously-added
    node (never creating a polytomy at the root).
    """
    n = len(times)
    if n == 0:
        return []

    # Virtual root is always first
    order = [0]

    # Sort remaining by (descending time, ancestors first, index)
    remaining = []
    for i in range(1, n):
        # Sort key: (-time, not is_ancestor, index)
        remaining.append((-times[i], not is_ancestor[i], i))
    remaining.sort()

    order.extend(idx for _, _, idx in remaining)
    return order


def _build_individual_metadata(cfg, node_metadata, individual_sample_indices, ploidy):
    """
    Build individual metadata and population assignments from config.

    Returns an _IndividualMetadataResult with:
    - metadata: list of dicts, one per individual
    - population_indices: list of population indices, one per individual (or None)
    - population_names: list of unique population names (or None)
    """
    ind_meta_cfg = cfg.individual_metadata
    num_individuals = len(individual_sample_indices)

    if num_individuals == 0:
        return _IndividualMetadataResult(
            metadata=None, population_indices=None, population_names=None
        )

    ind_metadata_list = []
    pop_indices = None
    pop_names = None

    # For each individual, find its source and sample index
    for ind_idx in range(num_individuals):
        hap_idx = individual_sample_indices[ind_idx]
        hap_meta = node_metadata[hap_idx]
        ind_md = {}

        if hap_meta is not None:
            source_name = hap_meta.get("source", "")
            sample_id = hap_meta.get("sample_id", "")

            if ind_meta_cfg is not None and ind_meta_cfg.fields:
                # Look up fields from VCZ store
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
        for ind_idx in range(num_individuals):
            hap_idx = individual_sample_indices[ind_idx]
            hap_meta = node_metadata[hap_idx]
            pop_val = None

            if hap_meta is not None:
                source_name = hap_meta.get("source", "")
                sample_id = hap_meta.get("sample_id", "")
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

    Collects haplotypes from the ancestor VCZ and all sample sources, then
    matches them one at a time against the incrementally-built tree sequence.
    """
    recombination_rate = kwargs.get("recombination_rate", cfg.match.recombination_rate)
    mismatch_ratio = kwargs.get("mismatch_ratio", cfg.match.mismatch_ratio)
    path_compression = kwargs.get("path_compression", cfg.match.path_compression)

    logger.info("Collecting haplotypes")
    hap_data = _collect_haplotypes(cfg)

    positions = hap_data.positions
    all_haplotypes = hap_data.haplotypes
    all_times = hap_data.times
    is_ancestor = hap_data.is_ancestor
    node_metadata = hap_data.metadata
    create_individuals = hap_data.create_individuals
    ploidy = hap_data.ploidy
    seq_intervals = hap_data.seq_intervals
    seq_len = hap_data.seq_len

    num_sites = len(positions)

    # Build TSB incrementally, matching one haplotype at a time
    num_alleles = [2] * num_sites
    tsb = _tsinfer.TreeSequenceBuilder(num_alleles)

    metadata = {"sequence_intervals": seq_intervals.tolist()}

    logger.info(
        "Match: %d haplotypes, %d sites, seq_len=%.0f",
        len(all_haplotypes),
        num_sites,
        seq_len,
    )

    # Compute matching order
    order = _order_haplotypes(all_times, is_ancestor)

    # Perturb times slightly for same-time ancestors so they have strictly
    # decreasing times, allowing sequential parent-child relationships.
    # The perturbation is tiny (1e-10 scale) so it doesn't affect the
    # biological interpretation.
    match_times = all_times.copy()
    eps = 1e-10
    seen_times = {}
    for idx in order:
        t = match_times[idx]
        if t in seen_times:
            seen_times[t] += 1
            match_times[idx] = t - eps * seen_times[t]
        else:
            seen_times[t] = 0

    # Track individual creation and ordered node metadata
    individual_groups = []
    current_ind_nodes = []
    ordered_node_metadata = []  # one per TSB node, in TSB insertion order
    individual_sample_indices = []  # haplotype indices for first node of each individual

    match_iter = enumerate(order)
    if progress:
        import tqdm

        match_iter = tqdm.tqdm(
            match_iter, total=len(order), desc="Matching", unit="haplotypes"
        )

    for step, idx in match_iter:
        hap = all_haplotypes[idx]
        time = float(match_times[idx])

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

            node_id = tsb.add_node(time)
            ordered_node_metadata.append(node_metadata[idx])

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

        # Track individuals
        if bool(create_individuals[idx]):
            current_ind_nodes.append(tsb.num_nodes - 1)
            if len(current_ind_nodes) == 1:
                individual_sample_indices.append(idx)
            if len(current_ind_nodes) == ploidy:
                individual_groups.append(current_ind_nodes)
                current_ind_nodes = []

    # Handle any remaining partial individual
    if current_ind_nodes:
        individual_groups.append(current_ind_nodes)

    # Build individual metadata and populations
    ind_result = _build_individual_metadata(
        cfg, node_metadata, individual_sample_indices, ploidy
    )

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


def run(cfg: Config, progress: bool = False, **kwargs) -> tskit.TreeSequence:
    """
    Run the full pipeline: infer_ancestors, match, post_process.
    """
    logger.info("Starting full pipeline")
    source_name = cfg.ancestors.sources[0]
    source = cfg.sources[source_name]
    ancestor_store = infer_ancestors(
        source, cfg.ancestors, cfg.ancestral_state, progress=progress
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
