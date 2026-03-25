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
Command-line interface for tsinfer.

All commands take a TOML config file as their primary argument.
Runtime concerns (parallelism, verbosity, overwrite) are CLI flags.

Commands:
  infer-ancestors   Build the ancestor VCZ from the samples VCZ.
  match             Run the unified match loop.
  post-process      Post-process a matched tree sequence.
  augment-sites     Place non-inference sites via parsimony.
  run               All steps in sequence.
  config show       Print resolved config with defaults filled in.
  config check      Validate config and verify all input paths.
"""

from __future__ import annotations

import collections
import json
import logging
import sys
import warnings
from pathlib import Path

import click
import tskit

from .ancestors import infer_ancestors
from .config import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_CLI_THREADS,
    DEFAULT_GENOTYPE_ENCODING,
    DEFAULT_WRITE_THREADS,
    Config,
)
from .pipeline import augment_sites as pipeline_augment_sites
from .pipeline import match as pipeline_match
from .pipeline import post_process as pipeline_post_process
from .pipeline import run as pipeline_run

logger = logging.getLogger(__name__)


def _setup_logging(verbose: int) -> None:
    """Configure logging level based on verbosity count."""
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Suppress noisy third-party debug logs
    logging.getLogger("zarr").setLevel(max(level, logging.WARNING))
    logging.getLogger("numcodecs").setLevel(max(level, logging.WARNING))


def _check_output(path: str | Path, force: bool) -> None:
    """Raise click.ClickException if output exists and --force is not set."""
    p = Path(path)
    if p.exists() and not force:
        raise click.ClickException(
            f"Output file '{p}' already exists. Use --force to overwrite."
        )


@click.group()
@click.version_option()
def main():
    """tsinfer: infer tree sequences from genetic variation data."""


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_config_arg = click.argument("config", metavar="CONFIG", type=click.Path(exists=True))

_runtime_options = [
    click.option(
        "--threads",
        default=DEFAULT_CLI_THREADS,
        show_default=True,
        help="Worker threads.",
    ),
    click.option("--force", is_flag=True, help="Overwrite existing output files."),
    click.option("--progress", is_flag=True, help="Show per-step progress bars."),
    click.option("-v", "--verbose", count=True, help="Increase log verbosity."),
]


def _add_options(options):
    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f

    return decorator


# ---------------------------------------------------------------------------
# Pipeline commands
# ---------------------------------------------------------------------------


_ENCODING_NAMES = {"one-bit": 1, "eight-bit": 0}
_DEFAULT_ENCODING_NAME = "one-bit" if DEFAULT_GENOTYPE_ENCODING == 1 else "eight-bit"


@main.command("infer-ancestors")
@_config_arg
@_add_options(_runtime_options)
@click.option(
    "--write-threads",
    default=DEFAULT_WRITE_THREADS,
    show_default=True,
    help="Writer threads for Zarr I/O.",
)
@click.option(
    "--genotype-encoding",
    type=click.Choice(sorted(_ENCODING_NAMES)),
    default=_DEFAULT_ENCODING_NAME,
    show_default=True,
    help="Genotype storage encoding. one-bit saves memory; "
    "eight-bit is faster but uses ~8x more. "
    "eight-bit is required when genotypes contain missing data.",
)
def infer_ancestors_cmd(
    config,
    threads,
    write_threads,
    genotype_encoding,
    force,
    progress,
    verbose,
):
    """Build the ancestor VCZ store from the samples VCZ."""
    _setup_logging(verbose)
    cfg = Config.from_toml(config)
    if progress and threads <= 0:
        warnings.warn(
            "--progress has no effect without --threads; "
            "ancestor-level progress requires threads >= 1",
            stacklevel=1,
        )
    anc_cfg = cfg.ancestors[0]
    sources = [cfg.sources[name] for name in anc_cfg.sources]
    logger.info(
        "Inferring ancestors from sources: %s",
        ", ".join(f"'{name}'" for name in anc_cfg.sources),
    )
    infer_ancestors(
        sources,
        anc_cfg,
        cfg.ancestral_state,
        progress=progress,
        num_threads=threads,
        write_threads=write_threads,
        genotype_encoding=_ENCODING_NAMES[genotype_encoding],
    )
    logger.info("Ancestor inference complete")


@main.command("match")
@_config_arg
@click.option(
    "--workdir",
    type=click.Path(),
    default=None,
    help="Directory for checkpoints; enables resume on restart.",
)
@click.option(
    "--keep-intermediates",
    is_flag=True,
    help="Keep all intermediate .trees files in workdir.",
)
@click.option(
    "--cache-size",
    default=DEFAULT_CACHE_SIZE,
    show_default=True,
    help="Genotype chunk cache size in MiB.",
)
@click.option(
    "--group-stop",
    type=int,
    default=None,
    help="Stop before this group index (0-indexed, like range()). "
    "e.g. --group-stop 2 processes groups 0 and 1. "
    "Requires --workdir for useful resume behavior.",
)
@click.option(
    "--read-workers",
    type=int,
    default=None,
    help="Background threads for loading genotype chunks. "
    "[default: threads/2, minimum 1]",
)
@_add_options(_runtime_options)
def match_cmd(
    config,
    workdir,
    keep_intermediates,
    cache_size,
    group_stop,
    read_workers,
    threads,
    force,
    progress,
    verbose,
):
    """Run the unified match loop (ancestors + samples)."""
    _setup_logging(verbose)
    cfg = Config.from_toml(config)
    if workdir is not None:
        cfg.match.workdir = workdir
    if keep_intermediates:
        cfg.match.keep_intermediates = True
    _check_output(cfg.match.output, force)
    logger.info("Running match")
    ts = pipeline_match(
        cfg,
        progress=progress,
        num_threads=threads,
        cache_size=cache_size,
        group_stop=group_stop,
        read_workers=read_workers,
    )
    ts.dump(str(cfg.match.output))
    logger.info("Match complete: %d nodes, %d edges", ts.num_nodes, ts.num_edges)


@main.command("post-process")
@_config_arg
@click.option(
    "--input",
    "input_ts",
    required=True,
    type=click.Path(exists=True),
    help="Input tree sequence file.",
)
@_add_options(_runtime_options)
def post_process_cmd(config, input_ts, threads, force, progress, verbose):
    """Post-process a matched tree sequence."""
    _setup_logging(verbose)
    cfg = Config.from_toml(config)
    ts = tskit.load(input_ts)
    logger.info("Post-processing %s (%d nodes)", input_ts, ts.num_nodes)
    ts = pipeline_post_process(ts, cfg)
    output = str(cfg.match.output)
    _check_output(output, force)
    ts.dump(output)
    logger.info("Post-process complete: %d nodes", ts.num_nodes)


@main.command("augment-sites")
@_config_arg
@click.option(
    "--input",
    "input_ts",
    required=True,
    type=click.Path(exists=True),
    help="Input tree sequence file (output of match or post-process).",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output tree sequence file.",
)
@_add_options(_runtime_options)
def augment_sites_cmd(config, input_ts, output_path, threads, force, progress, verbose):
    """Place non-inference sites onto a tree sequence using parsimony."""
    _setup_logging(verbose)
    cfg = Config.from_toml(config)
    ts = tskit.load(input_ts)
    _check_output(output_path, force)
    logger.info("Augmenting sites on %s (%d sites)", input_ts, ts.num_sites)
    ts = pipeline_augment_sites(ts, cfg, progress=progress)
    ts.dump(str(output_path))
    logger.info("Augment complete: %d sites", ts.num_sites)


@main.command("run")
@_config_arg
@click.option(
    "--cache-size",
    default=DEFAULT_CACHE_SIZE,
    show_default=True,
    help="Genotype chunk cache size in MiB.",
)
@click.option(
    "--genotype-encoding",
    type=click.Choice(sorted(_ENCODING_NAMES)),
    default=_DEFAULT_ENCODING_NAME,
    show_default=True,
    help="Genotype storage encoding. one-bit saves memory; "
    "eight-bit is faster but uses ~8x more. "
    "eight-bit is required when genotypes contain missing data.",
)
@click.option(
    "--read-workers",
    type=int,
    default=None,
    help="Background threads for loading genotype chunks. "
    "[default: threads/2, minimum 1]",
)
@_add_options(_runtime_options)
def run_cmd(
    config,
    cache_size,
    genotype_encoding,
    read_workers,
    threads,
    force,
    progress,
    verbose,
):
    """Run the full pipeline: infer-ancestors, match, post-process, augment-sites."""
    _setup_logging(verbose)
    cfg = Config.from_toml(config)
    _check_output(cfg.match.output, force)
    logger.info("Running full pipeline")
    ts = pipeline_run(
        cfg,
        progress=progress,
        num_threads=threads,
        cache_size=cache_size,
        genotype_encoding=_ENCODING_NAMES[genotype_encoding],
        read_workers=read_workers,
    )
    ts.dump(str(cfg.match.output))
    logger.info(
        "Pipeline complete: %d nodes, %d edges, %d sites",
        ts.num_nodes,
        ts.num_edges,
        ts.num_sites,
    )


# ---------------------------------------------------------------------------
# Inspection commands
# ---------------------------------------------------------------------------


@main.command("show-match-jobs")
@click.argument("json_file", metavar="JSON_FILE", type=click.Path(exists=True))
def show_match_jobs_cmd(json_file):
    """Show a histogram of match-jobs group sizes."""
    records = json.loads(Path(json_file).read_text())

    group_intervals: dict[int, list[float]] = collections.defaultdict(list)
    for rec in records:
        g = rec["group"]
        interval_kb = (rec["end_position"] - rec["start_position"]) / 1000
        group_intervals[g].append(interval_kb)

    if not group_intervals:
        click.echo("No match jobs.")
        return

    sorted_groups = sorted(group_intervals.items())
    max_count = max(len(v) for v in group_intervals.values())
    max_bar = 60

    click.echo(f"{'Group':>6}  {'Count':>6}  {'Mean kb':>8}  {'Var kb':>8}  ")
    for group_idx, intervals in sorted_groups:
        count = len(intervals)
        mean = sum(intervals) / count
        variance = sum((x - mean) ** 2 for x in intervals) / count
        bar_len = round(count / max_count * max_bar) if max_count > 0 else 0
        bar = "#" * bar_len
        click.echo(f"{group_idx:>6}  {count:>6}  {mean:>8.1f}  {variance:>8.3g}  {bar}")

    click.echo(f"\n{len(records)} jobs in {len(group_intervals)} groups")

    # Chunk locality summary (per source)
    # Build {source: {group: set(chunk_indices)}}
    source_group_chunks: dict[str, dict[int, set[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    for rec in records:
        source_group_chunks[rec["source"]][rec["group"]].add(rec["sample_chunk"])

    if source_group_chunks:
        click.echo("\nChunk locality (by source):")
        for source_name in sorted(source_group_chunks):
            click.echo(f"  Source '{source_name}':")
            click.echo(f"    {'Group':>6}  {'Chunks':>6}  Shared w/prev")
            sgc = source_group_chunks[source_name]
            prev_chunks: set[int] | None = None
            for group_idx in sorted(sgc):
                chunks = sgc[group_idx]
                n_chunks = len(chunks)
                if prev_chunks is None:
                    shared = "-"
                else:
                    shared = str(len(chunks & prev_chunks))
                click.echo(f"    {group_idx:>6}  {n_chunks:>6}  {shared}")
                prev_chunks = chunks


@main.command("simulate-cache")
@click.argument("json_file", metavar="JSON_FILE", type=click.Path(exists=True))
@click.option(
    "--cache-slots",
    type=int,
    required=True,
    help="Number of chunk slots in the simulated LRU cache.",
)
def simulate_cache_cmd(json_file, cache_slots):
    """Simulate LRU chunk cache for match-jobs and report load counts."""
    records = json.loads(Path(json_file).read_text())
    if not records:
        click.echo("No match jobs.")
        return

    # Group records, sort each group by haplotype_index
    groups: dict[int, list[dict]] = collections.defaultdict(list)
    for rec in records:
        groups[rec["group"]].append(rec)
    for g in groups.values():
        g.sort(key=lambda r: r["haplotype_index"])

    # Build deduplicated access sequence per group
    # (consecutive jobs for the same sample hit the same chunk)
    group_accesses: dict[int, list[tuple[str, int]]] = {}
    for group_idx in sorted(groups):
        accesses = []
        prev_key = None
        for rec in groups[group_idx]:
            key = (rec["source"], rec["sample_chunk"])
            if key != prev_key:
                accesses.append(key)
                prev_key = key
        group_accesses[group_idx] = accesses

    # LRU simulation
    cache: collections.OrderedDict[tuple[str, int], None] = collections.OrderedDict()
    chunk_loads: dict[tuple[str, int], int] = collections.defaultdict(int)
    group_stats: dict[int, tuple[int, int]] = {}  # group -> (accesses, misses)
    total_hits = 0
    total_misses = 0

    for group_idx in sorted(group_accesses):
        accesses = group_accesses[group_idx]
        group_misses = 0
        group_hits = 0
        for key in accesses:
            if key in cache:
                cache.move_to_end(key)
                group_hits += 1
            else:
                if len(cache) >= cache_slots:
                    cache.popitem(last=False)
                cache[key] = None
                chunk_loads[key] += 1
                group_misses += 1
        group_stats[group_idx] = (len(accesses), group_misses)
        total_hits += group_hits
        total_misses += group_misses

    # Print results
    click.echo(f"Cache simulation (cache_slots={cache_slots}):\n")

    click.echo("Per-group summary:")
    click.echo(
        f"  {'Group':>6}  {'Jobs':>6}  {'Accesses':>8}  {'Misses':>6}  {'Miss%':>6}"
    )
    for group_idx in sorted(group_stats):
        n_jobs = len(groups[group_idx])
        n_acc, n_miss = group_stats[group_idx]
        pct = 100 * n_miss / n_acc if n_acc > 0 else 0
        click.echo(
            f"  {group_idx:>6}  {n_jobs:>6}  {n_acc:>8}  {n_miss:>6}  {pct:>5.1f}%"
        )

    # Per-chunk load counts (only chunks loaded more than once)
    multi_loaded = {k: v for k, v in chunk_loads.items() if v > 1}
    if multi_loaded:
        click.echo("\nChunks loaded more than once:")
        click.echo(f"  {'Source':<20}  {'Chunk':>6}  {'Loads':>5}")
        for (src, chunk), loads in sorted(multi_loaded.items(), key=lambda x: -x[1]):
            click.echo(f"  {src:<20}  {chunk:>6}  {loads:>5}")

    total = total_hits + total_misses
    miss_pct = 100 * total_misses / total if total > 0 else 0
    click.echo(
        f"\nTotals: {total_hits} hits, {total_misses} misses"
        f" ({miss_pct:.1f}% miss rate),"
        f" {sum(chunk_loads.values())} chunk loads"
    )


# ---------------------------------------------------------------------------
# Config utility
# ---------------------------------------------------------------------------


@main.group("config")
def config_group():
    """Utilities for inspecting and validating the config file."""


@config_group.command("show")
@_config_arg
def config_show(config):
    """Print the resolved config with defaults filled in."""
    cfg = Config.from_toml(config)
    click.echo(cfg.format())


@config_group.command("check")
@_config_arg
def config_check(config):
    """Validate the config and verify all input paths exist."""
    cfg = Config.from_toml(config)
    errors = cfg.validate()
    if errors:
        for err in errors:
            click.echo(f"ERROR: {err}", err=True)
        raise SystemExit(1)
    click.echo("Config OK", err=True)
