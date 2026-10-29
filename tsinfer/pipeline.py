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
import pathlib
import sys
import time as time_

import numpy as np
import psutil
import tqdm
import tskit

from . import ancestors, arg_ops, config, grouping, matching, provenance
from . import vcz as vcz_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference sites and reference tree sequence
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MatchSites:
    """The sites that haplotypes are matched at."""

    positions: np.ndarray  # (num_sites,) int32
    ancestral_alleles: np.ndarray  # (num_sites,) allele strings
    # Alleles already present at each site of the reference tree sequence,
    # or None when matching starts from the ancestors.
    reference_alleles: list[list[str]] | None


def _load_reference_ts(
    cfg: config.Config, reference_ts: tskit.TreeSequence | None
) -> tskit.TreeSequence | None:
    """Return the reference tree sequence given directly or in the config."""
    if reference_ts is None and cfg.match.reference_ts is not None:
        logger.info("Loading reference tree sequence %s", cfg.match.reference_ts)
        reference_ts = tskit.load(str(cfg.match.reference_ts))
    return reference_ts


def _match_sites(
    cfg: config.Config, reference_ts: tskit.TreeSequence | None
) -> MatchSites:
    """
    Return the match sites: those of the reference tree sequence if given,
    otherwise those of the first ancestor store.
    """
    if reference_ts is None:
        anc_store = vcz_mod.open_store(cfg.ancestors[0].path)
        positions = np.asarray(anc_store["variant_position"][:], dtype=np.int32)
        site_alleles = np.asarray(anc_store["variant_allele"][:])
        return MatchSites(
            positions=positions,
            ancestral_alleles=site_alleles[:, 0],
            reference_alleles=None,
        )

    site_positions = reference_ts.sites_position
    if np.any(site_positions != np.floor(site_positions)):
        raise ValueError("reference_ts site positions must be integers")
    ancestral_alleles = reference_ts.sites_ancestral_state
    reference_alleles = [[str(a)] for a in ancestral_alleles]
    for site_id, derived_state in zip(
        reference_ts.mutations_site, reference_ts.mutations_derived_state
    ):
        site_alleles = reference_alleles[site_id]
        if derived_state not in site_alleles:
            site_alleles.append(str(derived_state))
    return MatchSites(
        positions=site_positions.astype(np.int32),
        ancestral_alleles=ancestral_alleles,
        reference_alleles=reference_alleles,
    )


def _check_reference_ts(
    reference_ts: tskit.TreeSequence,
    jobs: list[config.MatchJob],
    sequence_length: float,
) -> None:
    """Raise a ValueError if haplotypes cannot be matched against reference_ts."""
    if reference_ts.sequence_length != sequence_length:
        raise ValueError(
            f"reference_ts sequence length ({reference_ts.sequence_length}) "
            f"does not match the contig length ({sequence_length})"
        )
    min_node_time = float(np.min(reference_ts.nodes_time))
    max_job_time = max(job.time for job in jobs)
    if min_node_time <= max_job_time:
        raise ValueError(
            f"All reference_ts nodes must be older than the haplotypes being "
            f"matched: youngest reference node has time {min_node_time}, but a "
            f"haplotype has time {max_job_time}"
        )


# ---------------------------------------------------------------------------
# Build jobs with metadata, pre-assign individuals and populations
# ---------------------------------------------------------------------------


def _build_jobs_with_metadata(cfg, positions, reference_ts=None):
    """
    Build MatchJobs with pre-assigned individual_id and population_id.

    *positions* are the match site positions, which give the default
    start/end of each haplotype. If *reference_ts* is given, individual and
    population ids are offset so that the new rows are appended after
    those already in it.

    Returns (jobs, individual_metadata_rows, population_metadata_rows).
    """
    logger.info("Loading haplotype metadata")
    t0 = time_.monotonic()

    individual_offset = 0
    population_offset = 0
    if reference_ts is not None:
        individual_offset = reference_ts.num_individuals
        population_offset = reference_ts.num_populations

    # Build MatchJobs directly from VCZ stores, assigning individual IDs
    # in the same pass.
    jobs: list[config.MatchJob] = []
    ind_key_to_id: dict[tuple[str, str], int] = {}
    hap_idx = 0

    for source_name in cfg.match.sources:
        source = cfg.sources[source_name]
        src_cfg = cfg.match.sources[source_name]
        store = vcz_mod.open_store(source.path)

        # Shape metadata only — no data loaded
        call_gt = store["call_genotype"]
        gt_shape = call_gt.shape
        num_samples = gt_shape[1]
        ploidy = gt_shape[2]
        sample_chunk_size = call_gt.chunks[1]

        samples_selection = vcz_mod.resolve_samples_selection(store, source.samples)
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
            default_end = int(positions[-1]) + 1
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
                individual_id = ind_key_to_id[key] + individual_offset
            else:
                individual_id = None

            # Determine sample chunk index
            if samples_selection is not None:
                store_col = int(samples_selection[i])
            else:
                store_col = i
            sample_chunk = store_col // sample_chunk_size

            for p in range(ploidy):
                jobs.append(
                    config.MatchJob(
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
                        sample_chunk=sample_chunk,
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
            pop_lookup = {
                name: i + population_offset for i, name in enumerate(unique_pops)
            }
            population_metadata_rows = [{"name": name} for name in unique_pops]

            # Map individual → population
            ind_to_pop: dict[int, int] = {}
            for ind_idx, pop_val in enumerate(ind_pop_values):
                if pop_val is not None and pop_val in pop_lookup:
                    ind_to_pop[ind_idx] = pop_lookup[pop_val]

            # Map population onto jobs via individual_id
            for job in jobs:
                if job.individual_id is None:
                    continue
                ind_idx = job.individual_id - individual_offset
                if ind_idx in ind_to_pop:
                    job.population_id = ind_to_pop[ind_idx]

    # Assign groups
    logger.info("Computing groups")
    t0 = time_.monotonic()
    jobs = grouping.assign_groups(jobs)
    elapsed = time_.monotonic() - t0
    num_groups = len({j.group for j in jobs})
    logger.info("Computed %d groups in %.2fs", num_groups, elapsed)

    return jobs, individual_metadata_rows, population_metadata_rows


def compute_groups_json(cfg: config.Config) -> str:
    """
    Compute haplotype groups and return as a JSON string.

    The output is a flat JSON array of objects, one per haplotype, ordered
    by group then haplotype index.  Each object has the fields of
    :class:`MatchJob` — suitable for loading into pandas with
    ``pd.read_json`` or ``pd.DataFrame``.
    """
    reference_ts = _load_reference_ts(cfg, None)
    sites = _match_sites(cfg, reference_ts)
    jobs, _, _ = _build_jobs_with_metadata(cfg, sites.positions, reference_ts)

    logger.info("Serialising %d match jobs to JSON", len(jobs))
    t0 = time_.monotonic()
    json_str = json.dumps([dataclasses.asdict(j) for j in jobs], indent=2)
    elapsed = time_.monotonic() - t0
    logger.info("Serialised in %.2fs", elapsed)
    return json_str


def _load_match_jobs(path) -> list[config.MatchJob]:
    """Read a match-jobs JSON file and return a list of MatchJob objects."""
    records = json.loads(pathlib.Path(path).read_text())
    return [config.MatchJob(**rec) for rec in records]


def _setup_workdir(workdir, cfg, positions, reference_ts):
    """
    Set up workdir: create dir, write or load match-jobs.json.

    When resuming from a checkpoint, the jobs are loaded from
    match-jobs.json and no metadata rows are returned, since the
    checkpoint already holds the individuals and populations. Otherwise
    the jobs and metadata rows are built and match-jobs.json is written.

    Returns (jobs, individual_metadata_rows, population_metadata_rows,
             last_completed_group_idx, starting_ts_or_None).
    """

    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    groups_path = workdir / "match-jobs.json"

    max_group_index = -1
    for path in workdir.glob("group_*.trees"):
        group_index = int(path.stem.split("_")[1])
        max_group_index = max(group_index, max_group_index)

    starting_ts = None
    if max_group_index >= 0:
        last_path = workdir / f"group_{max_group_index}.trees"
        starting_ts = tskit.load(str(last_path))
        logger.info("Resuming from group %d ", max_group_index)
        logger.info("Loading existing groups from %s", groups_path)
        jobs = _load_match_jobs(groups_path)
        individual_metadata_rows: list[dict] = []
        population_metadata_rows: list[dict] = []
    else:
        jobs, individual_metadata_rows, population_metadata_rows = (
            _build_jobs_with_metadata(cfg, positions, reference_ts)
        )
        json_str = json.dumps([dataclasses.asdict(j) for j in jobs], indent=2)
        groups_path.write_text(json_str)
        logger.info("Wrote groups to %s", groups_path)

    return (
        jobs,
        individual_metadata_rows,
        population_metadata_rows,
        max_group_index,
        starting_ts,
    )


def _compute_hmm_parameters(
    cfg: config.Config, positions: np.ndarray
) -> dict[str, matching.HmmParameters]:
    """Compute the HMM parameters for each match source."""
    result = {}
    for name, src_cfg in cfg.match.sources.items():
        params = matching.hmm_parameters(src_cfg, positions)
        recombination = params.recombination[1:]
        if len(recombination) > 0:
            logger.info(
                "HMM parameters for source '%s': recombination min=%.3g "
                "mean=%.3g max=%.3g, mismatch=%.3g",
                name,
                recombination.min(),
                recombination.mean(),
                recombination.max(),
                params.mismatch[0],
            )
        result[name] = params
    return result


def _process_group(
    ts: tskit.TreeSequence,
    *,
    group_idx: int,
    group_jobs: list,
    positions: np.ndarray,
    path_compression: bool,
    allele_mapper,
    reader,
    num_threads: int,
    match_fh,
    config_str: str,
    progress: bool,
    hmm_parameters: dict[str, matching.HmmParameters],
    engine: str,
) -> tskit.TreeSequence:
    """Match one group of haplotypes and return the extended tree sequence."""
    with provenance.TimingAndMemory() as tm:
        m = matching.Matcher(
            ts,
            positions,
            allele_mapper=allele_mapper,
            hmm_parameters=hmm_parameters,
            engine=engine,
        )
        job_list = [job for _, job in group_jobs]
        match_iter = m.match(job_list, reader, num_threads=num_threads)
        pbar = tqdm.tqdm(
            total=len(job_list),
            desc=f"Group {group_idx}",
            unit="haplotypes",
            disable=not progress,
        )
        results = []
        last_mem_log = time_.monotonic()
        mem_log_interval = 30  # seconds
        for job, result in match_iter:
            if match_fh is not None:
                doc = {
                    "group": group_idx,
                    "haplotype_index": job.haplotype_index,
                    "source": job.source,
                    "sample_id": job.sample_id,
                    "ploidy_index": job.ploidy_index,
                    "time": job.time,
                    "path": [
                        {"left": s.left, "right": s.right, "parent": s.parent}
                        for s in result.path
                    ],
                    "mutations": [
                        {"position": m.position, "derived_state": m.derived_state}
                        for m in result.mutations
                    ],
                }
                match_fh.write(json.dumps(doc) + "\n")
            results.append((job, result))
            pbar.update(1)
            now = time_.monotonic()
            if now - last_mem_log >= mem_log_interval:
                rss_mib = psutil.Process().memory_info().rss / (1024 * 1024)
                reader.log_cache_state()
                logger.info(
                    "Group %d progress: %d/%d haplotypes, RSS=%.1f MiB",
                    group_idx,
                    len(results),
                    len(job_list),
                    rss_mib,
                )
                last_mem_log = now
        pbar.close()
        paired_results = sorted(results, key=lambda pair: pair[0].haplotype_index)

    prov_dict = provenance.get_provenance_dict(
        command="match",
        group=group_idx,
        config=config_str,
        cli_args=sys.argv,
        resources=tm.metrics.asdict(),
    )

    result = matching.extend_ts(
        ts,
        paired_results=paired_results,
        allele_mapper=allele_mapper,
        provenance_record=json.dumps(prov_dict),
    )
    if path_compression:
        result = arg_ops.compress_paths(result)
    return result


def match(
    cfg: config.Config,
    reference_ts: tskit.TreeSequence | None = None,
    progress: bool = False,
    num_threads: int | None = None,
    cache_size: int | None = None,
    group_stop: int | None = None,
    read_workers: int | None = None,
    match_file: str | pathlib.Path | None = None,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Run the unified match loop over all sources listed in cfg.match.

    Uses the ``make_root_ts → Matcher → extend_ts`` cycle: each group of
    haplotypes is matched against the current tree sequence, then the tree
    sequence is extended with the results.

    If *reference_ts* is given (or ``cfg.match.reference_ts`` is set), the
    loop starts from that tree sequence instead of a root tree sequence,
    and haplotypes are matched at its sites. All of its nodes must be
    older than the haplotypes being matched.
    """
    if num_threads is None:
        num_threads = config.DEFAULT_NUM_THREADS
    if cache_size is None:
        cache_size = config.DEFAULT_CACHE_SIZE
    path_compression = kwargs.get("path_compression", cfg.match.path_compression)

    # Validate group_stop
    if group_stop is not None and group_stop < 0:
        raise ValueError("group_stop must be non-negative")
    if group_stop is not None and cfg.match.workdir is None:
        logger.warning("group_stop without workdir; partial result cannot be resumed")

    reference_ts = _load_reference_ts(cfg, reference_ts)
    sites = _match_sites(cfg, reference_ts)
    positions = sites.positions
    hmm_parameters = _compute_hmm_parameters(cfg, positions)

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
        ) = _setup_workdir(workdir, cfg, positions, reference_ts)
    else:
        jobs, individual_metadata_rows, population_metadata_rows = (
            _build_jobs_with_metadata(cfg, positions, reference_ts)
        )

    # 2. Resolve the contig in all match sources (ancestors included)
    match_sources = [cfg.sources[name] for name in cfg.match.sources]
    resolved = vcz_mod.resolve_sources(match_sources, None, cfg.contig)
    seq_len = float(resolved.contig.length)
    if reference_ts is not None:
        _check_reference_ts(reference_ts, jobs, seq_len)

    # 3. Group jobs and build read schedule
    groups_dict = collections.defaultdict(list)
    for idx, job in enumerate(jobs):
        groups_dict[job.group].append((idx, job))
    sorted_groups = sorted(groups_dict.keys())

    schedule = []
    for group_idx in sorted_groups:
        if group_idx <= last_completed_group:
            continue
        if group_stop is not None and group_idx >= group_stop:
            break
        for _, job in sorted(groups_dict[group_idx], key=lambda p: p[1].haplotype_index):
            schedule.append((job.source, job.sample_id, job.ploidy_index))

    # 4. Create lazy reader with all match sources
    reader = vcz_mod.HaplotypeReader(
        {source.name: source for source in resolved.sources},
        positions,
        sites.ancestral_alleles,
        cache_size_mb=cache_size,
        schedule=schedule,
        read_workers=read_workers,
        num_threads=num_threads,
        reference_alleles=sites.reference_alleles,
    )
    allele_mapper = reader.allele_mapper

    # 5. Build the starting TS: resume from a workdir checkpoint, start
    # from the reference TS, or build a root TS.
    individuals = individual_metadata_rows if individual_metadata_rows else None
    populations = population_metadata_rows if population_metadata_rows else None
    if workdir_starting_ts is not None:
        ts = workdir_starting_ts
    elif reference_ts is not None:
        ts = matching.prepare_reference_ts(
            reference_ts, individuals=individuals, populations=populations
        )
    else:
        anc_store = vcz_mod.open_store(cfg.ancestors[0].path)
        seq_intervals = np.asarray(anc_store["sequence_intervals"][:], dtype=np.int32)
        anc_times = np.asarray(anc_store["sample_time"][:], dtype=np.float64)
        max_anc_time = float(np.max(anc_times)) if len(anc_times) > 0 else 0.0
        ts = matching.make_root_ts(
            seq_len,
            positions,
            seq_intervals,
            max_time=max_anc_time,
            individuals=individuals,
            populations=populations,
            allele_mapper=allele_mapper,
        )

    logger.info(
        "Match: %d haplotypes, %d sites, seq_len=%.0f, engine=%s",
        len(jobs),
        len(positions),
        seq_len,
        cfg.match.engine,
    )

    # 6. Match loop — process groups in order
    match_fh = open(match_file, "w") if match_file is not None else None
    num_groups = len(sorted_groups)
    total_haps = len(jobs)
    completed_haps = 0
    config_str = cfg.format()

    prev_written_group = None
    for group_idx in sorted_groups:
        group_jobs = groups_dict[group_idx]
        num_in_group = len(group_jobs)

        # Skip groups already completed
        if group_idx <= last_completed_group:
            completed_haps += num_in_group
            logger.info("Group %d: skipped (already completed)", group_idx)
            continue

        # Stop before reaching group_stop (range semantics)
        if group_stop is not None and group_idx >= group_stop:
            logger.info(
                "Stopping at group %d (group_stop=%d)",
                group_idx,
                group_stop,
            )
            break

        reader.log_cache_state()
        logger.info(
            "Group index %d (%d): %d haplotypes", group_idx, num_groups, num_in_group
        )

        ts = _process_group(
            ts,
            group_idx=group_idx,
            group_jobs=group_jobs,
            positions=positions,
            path_compression=path_compression,
            allele_mapper=allele_mapper,
            reader=reader,
            num_threads=num_threads,
            match_fh=match_fh,
            config_str=config_str,
            progress=progress,
            hmm_parameters=hmm_parameters,
            engine=cfg.match.engine,
        )
        completed_haps += num_in_group
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info(
            "Group %d done (total=%d): %d/%d haplotypes completed (%.1f%%),"
            " RSS=%.1f MiB",
            group_idx,
            num_groups,
            completed_haps,
            total_haps,
            100.0 * completed_haps / total_haps if total_haps > 0 else 0,
            rss,
        )

        # Write checkpoint to workdir if configured
        if workdir is not None:
            wd = pathlib.Path(workdir)
            ts_path = wd / f"group_{group_idx}.trees"
            ts.dump(str(ts_path))
            logger.info("Wrote checkpoint to %s", ts_path)
            # Clean up previous group file unless keep_intermediates
            if not cfg.match.keep_intermediates and prev_written_group is not None:
                prev_path = wd / f"group_{prev_written_group}.trees"
                if prev_path.exists():
                    prev_path.unlink()
            prev_written_group = group_idx

    if match_fh is not None:
        match_fh.close()
    reader.shutdown()
    logger.info(
        "Match complete: %d nodes, %d individuals",
        ts.num_nodes,
        ts.num_individuals,
    )

    return ts


def post_process(
    ts: tskit.TreeSequence,
    cfg: config.Config,
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
    cfg: config.Config,
    progress: bool = False,
) -> tskit.TreeSequence:
    """
    Place additional sites onto an inferred tree sequence using parsimony.

    For each source listed in ``cfg.augment_sites.sources``, iterates over
    variants not already present in *ts* (and within its sequence_intervals),
    then uses ``tree.map_mutations`` to place them parsimoniously.

    Sites that appear at duplicate positions within a single source are
    excluded (same behaviour as ``infer_ancestors``).  The ancestral allele
    is read from the ``[ancestral_state]`` config section and passed to
    ``map_mutations`` so that the output always uses the specified ancestral
    state. Sites with an unknown ancestral allele are skipped.
    """
    aug_cfg = cfg.augment_sites
    if aug_cfg is None or len(aug_cfg.sources) == 0:
        return ts

    logger.info("augment_sites: %d source(s)", len(aug_cfg.sources))

    # --- sequence_intervals ---
    meta = ts.metadata if ts.metadata is not None else {}
    intervals = meta.get("sequence_intervals")
    if intervals is None:
        intervals = [[0.0, float(ts.sequence_length)]]

    # --- Build sample_identifiers from ts node metadata ---
    required_keys = {"source", "sample_id", "ploidy_index"}
    sample_identifiers = []
    for u in ts.samples():
        md = ts.node(u).metadata
        if not md or not required_keys.issubset(md):
            missing = required_keys - set(md) if md else required_keys
            raise ValueError(
                f"Node {u} is missing required metadata key(s) "
                f"{missing} for augment_sites"
            )
        sample_identifiers.append((md["source"], md["sample_id"], md["ploidy_index"]))

    # --- Build MultiSourceView and resolve positions ---
    source_configs = [cfg.sources[name] for name in aug_cfg.sources]
    resolved = vcz_mod.resolve_sources(source_configs, cfg.ancestral_state, cfg.contig)
    view = vcz_mod.MultiSourceView(resolved.sources, resolved.ancestral_state)

    positions = view.positions_in_intervals(intervals)
    positions = view.filter_positions(ts.sites_position, positions)
    # Sites with an unknown ancestral allele cannot be polarised
    unknown = view.ancestral_alleles == ""
    unknown_positions = view.positions[unknown]
    is_unknown = np.isin(positions, unknown_positions)
    num_unknown = int(np.sum(is_unknown))
    if num_unknown > 0:
        logger.info(
            "augment_sites: skipping %d sites with unknown ancestral allele",
            num_unknown,
        )
    positions = positions[~is_unknown]
    n_sites = len(positions)

    if n_sites == 0:
        logger.info("augment_sites: no new sites to add")
        return ts

    logger.info("augment_sites: %d new sites to place", n_sites)

    # --- Stream genotypes and place via map_mutations ---
    tables = ts.dump_tables()
    tree = ts.first()

    site_iter = view.iter_genotypes(
        positions,
        sample_identifiers=sample_identifiers,
    )
    if progress:
        site_iter = tqdm.tqdm(
            site_iter, total=n_sites, desc="augment_sites", unit="sites"
        )

    for var in site_iter:
        genotypes = var.genotypes.astype(np.int32)
        alleles = list(var.alleles)

        # Skip all-missing sites
        if np.all(genotypes < 0):
            continue

        # Canonical code 0 = ancestral allele
        tree.seek(var.position)
        ancestral_state, mutations = tree.map_mutations(
            genotypes, alleles, ancestral_state=alleles[0]
        )
        site_id = tables.sites.add_row(
            position=var.position, ancestral_state=ancestral_state
        )
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


def run(
    cfg: config.Config,
    progress: bool = False,
    num_threads: int | None = None,
    cache_size: int | None = None,
    genotype_encoding: int | None = None,
    read_workers: int | None = None,
    match_file: str | pathlib.Path | None = None,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Run the full pipeline: infer_ancestors, match, post_process, augment_sites.
    """
    if num_threads is None:
        num_threads = config.DEFAULT_NUM_THREADS
    if cache_size is None:
        cache_size = config.DEFAULT_CACHE_SIZE
    if genotype_encoding is None:
        genotype_encoding = config.DEFAULT_GENOTYPE_ENCODING
    logger.info("Starting full pipeline")
    if len(cfg.ancestors) == 0:
        # Matching starts from cfg.match.reference_ts; no ancestors to infer.
        ts = match(
            cfg,
            progress=progress,
            num_threads=num_threads,
            cache_size=cache_size,
            read_workers=read_workers,
            match_file=match_file,
            **kwargs,
        )
        return _finish_pipeline(ts, cfg, progress, **kwargs)

    anc_cfg = cfg.ancestors[0]
    sources = [cfg.sources[name] for name in anc_cfg.sources]
    ancestor_store = ancestors.infer_ancestors(
        sources,
        anc_cfg,
        cfg.ancestral_state,
        progress=progress,
        num_threads=num_threads,
        genotype_encoding=genotype_encoding,
        contig=cfg.contig,
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
            read_workers=read_workers,
            match_file=match_file,
            **kwargs,
        )
        ts = _finish_pipeline(ts, cfg, progress, **kwargs)
    finally:
        anc_cfg.path = original_path
    return ts


def _finish_pipeline(ts, cfg, progress, **kwargs):
    """Run the post-match steps of the pipeline: post_process, augment_sites."""
    ts = post_process(ts, cfg, **kwargs)
    if cfg.augment_sites is not None:
        ts = augment_sites(ts, cfg, progress=progress)
    logger.info(
        "Pipeline complete: %d nodes, %d edges, %d sites",
        ts.num_nodes,
        ts.num_edges,
        ts.num_sites,
    )
    return ts
