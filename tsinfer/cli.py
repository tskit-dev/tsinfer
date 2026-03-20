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
  run               All steps in sequence.
  config show       Print resolved config with defaults filled in.
  config check      Validate config and verify all input paths.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import click
import tskit

from .ancestors import infer_ancestors
from .config import Config
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
    click.option("--threads", default=1, show_default=True, help="Worker threads."),
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


@main.command("infer-ancestors")
@_config_arg
@_add_options(_runtime_options)
def infer_ancestors_cmd(config, threads, force, progress, verbose):
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
    default=256,
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
@_add_options(_runtime_options)
def match_cmd(
    config,
    workdir,
    keep_intermediates,
    cache_size,
    group_stop,
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


@main.command("run")
@_config_arg
@click.option(
    "--cache-size",
    default=256,
    show_default=True,
    help="Genotype chunk cache size in MiB.",
)
@_add_options(_runtime_options)
def run_cmd(config, cache_size, threads, force, progress, verbose):
    """Run the full pipeline: infer-ancestors, match, post-process."""
    _setup_logging(verbose)
    cfg = Config.from_toml(config)
    _check_output(cfg.match.output, force)
    logger.info("Running full pipeline")
    ts = pipeline_run(cfg, progress=progress, num_threads=threads, cache_size=cache_size)
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


@main.command("show-match-work")
@click.argument("json_file", metavar="JSON_FILE", type=click.Path(exists=True))
def show_match_work_cmd(json_file):
    """Show a histogram of match-work group sizes."""
    records = json.loads(Path(json_file).read_text())

    group_counts: dict[int, int] = {}
    for rec in records:
        g = rec["group"]
        group_counts[g] = group_counts.get(g, 0) + 1

    if not group_counts:
        click.echo("No match jobs.")
        return

    sorted_groups = sorted(group_counts.items())
    max_count = max(group_counts.values())
    max_bar = 100

    click.echo(f"{'Group':>6}  {'Count':>6}")
    for group_idx, count in sorted_groups:
        bar_len = round(count / max_count * max_bar) if max_count > 0 else 0
        bar = "#" * bar_len
        click.echo(f"{group_idx:>6}  {count:>6}  {bar}")

    click.echo(f"\n{len(records)} jobs in {len(group_counts)} groups")


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
