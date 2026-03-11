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

import logging
import sys
from pathlib import Path

import click
import tskit
import zarr

from .config import Config

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

    from .ancestors import infer_ancestors

    source_name = cfg.ancestors.sources[0]
    source = cfg.sources[source_name]
    logger.info("Inferring ancestors from source '%s'", source_name)
    infer_ancestors(source, cfg.ancestors, cfg.ancestral_state)
    logger.info("Ancestor inference complete")


@main.command("match")
@_config_arg
@_add_options(_runtime_options)
def match_cmd(config, threads, force, progress, verbose):
    """Run the unified match loop (ancestors + samples)."""
    _setup_logging(verbose)
    cfg = Config.from_toml(config)
    _check_output(cfg.match.output, force)

    from .pipeline import match as pipeline_match

    logger.info("Running match")
    ts = pipeline_match(cfg)
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

    from .pipeline import post_process as pipeline_post_process

    ts = tskit.load(input_ts)
    logger.info("Post-processing %s (%d nodes)", input_ts, ts.num_nodes)
    ts = pipeline_post_process(ts, cfg)
    output = str(cfg.match.output)
    _check_output(output, force)
    ts.dump(output)
    logger.info("Post-process complete: %d nodes", ts.num_nodes)


@main.command("run")
@_config_arg
@_add_options(_runtime_options)
def run_cmd(config, threads, force, progress, verbose):
    """Run the full pipeline: infer-ancestors, match, post-process."""
    _setup_logging(verbose)
    cfg = Config.from_toml(config)
    _check_output(cfg.match.output, force)

    from .pipeline import run as pipeline_run

    logger.info("Running full pipeline")
    ts = pipeline_run(cfg)
    ts.dump(str(cfg.match.output))
    logger.info(
        "Pipeline complete: %d nodes, %d edges, %d sites",
        ts.num_nodes,
        ts.num_edges,
        ts.num_sites,
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
    _print_resolved_config(cfg)


@config_group.command("check")
@_config_arg
def config_check(config):
    """Validate the config and verify all input paths exist."""
    cfg = Config.from_toml(config)
    errors = _validate_paths(cfg)
    if errors:
        for err in errors:
            click.echo(f"ERROR: {err}", err=True)
        raise SystemExit(1)
    click.echo("Config OK", err=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_resolved_config(cfg: Config) -> None:
    """Print the resolved config as human-readable text to stdout."""
    for name, src in cfg.sources.items():
        click.echo(f"[source.{name}]")
        click.echo(f"  path = {src.path}")
        if src.site_mask is not None:
            click.echo(f"  site_mask = {src.site_mask}")
        if src.sample_mask is not None:
            click.echo(f"  sample_mask = {src.sample_mask}")
        if src.sample_time is not None:
            click.echo(f"  sample_time = {src.sample_time}")
        click.echo()

    if cfg.ancestral_state is not None:
        click.echo("[ancestral_state]")
        click.echo(f"  path = {cfg.ancestral_state.path}")
        click.echo(f"  field = {cfg.ancestral_state.field}")
        click.echo()

    if cfg.ancestors is not None:
        click.echo("[ancestors]")
        click.echo(f"  path = {cfg.ancestors.path}")
        click.echo(f"  sources = {cfg.ancestors.sources}")
        click.echo(f"  max_gap_length = {cfg.ancestors.max_gap_length}")
        click.echo()

    click.echo("[match]")
    click.echo(f"  sources = {cfg.match.sources}")
    click.echo(f"  output = {cfg.match.output}")
    click.echo(f"  recombination_rate = {cfg.match.recombination_rate}")
    click.echo(f"  mismatch_ratio = {cfg.match.mismatch_ratio}")
    click.echo(f"  path_compression = {cfg.match.path_compression}")
    click.echo(f"  num_threads = {cfg.match.num_threads}")
    if cfg.match.reference_ts is not None:
        click.echo(f"  reference_ts = {cfg.match.reference_ts}")
    click.echo()

    if cfg.post_process is not None:
        click.echo("[post_process]")
        click.echo(f"  split_ultimate = {cfg.post_process.split_ultimate}")
        click.echo(f"  erase_flanks = {cfg.post_process.erase_flanks}")
        click.echo()

    if cfg.individual_metadata is not None:
        click.echo("[individual_metadata]")
        click.echo(f"  fields = {cfg.individual_metadata.fields}")
        if cfg.individual_metadata.population is not None:
            click.echo(f"  population = {cfg.individual_metadata.population}")
        click.echo()


def _validate_paths(cfg: Config) -> list[str]:
    """Check that all input paths in the config exist. Return list of errors."""
    errors = []

    for name, src in cfg.sources.items():
        p = Path(str(src.path))
        if not p.exists():
            errors.append(f"Source '{name}' path does not exist: {src.path}")
        if isinstance(src.site_mask, dict) and "path" in src.site_mask:
            if not Path(str(src.site_mask["path"])).exists():
                errors.append(
                    f"Source '{name}' site_mask path does not exist: "
                    f"{src.site_mask['path']}"
                )
        if isinstance(src.sample_mask, dict) and "path" in src.sample_mask:
            if not Path(str(src.sample_mask["path"])).exists():
                errors.append(
                    f"Source '{name}' sample_mask path does not exist: "
                    f"{src.sample_mask['path']}"
                )
        if isinstance(src.sample_time, dict) and "path" in src.sample_time:
            if not Path(str(src.sample_time["path"])).exists():
                errors.append(
                    f"Source '{name}' sample_time path does not exist: "
                    f"{src.sample_time['path']}"
                )

    # ancestors.path is an output — don't check it for existence.
    # But do check that ancestral state info is available for ancestor sources.
    if cfg.ancestors is not None and cfg.ancestral_state is None:
        for src_name in cfg.ancestors.sources:
            src = cfg.sources.get(src_name)
            if src is None:
                errors.append(f"Ancestors references unknown source: '{src_name}'")
                continue
            p = Path(str(src.path))
            if p.exists():
                try:
                    store = zarr.open(str(p), mode="r")
                    if "variant_ancestral_allele" not in store:
                        errors.append(
                            f"Source '{src_name}' has no "
                            f"'variant_ancestral_allele' array and no "
                            f"[ancestral_state] section is configured"
                        )
                except Exception:
                    pass  # path existence errors are reported above

    if cfg.match.reference_ts is not None:
        p = Path(str(cfg.match.reference_ts))
        if not p.exists():
            errors.append(
                f"Match reference_ts path does not exist: {cfg.match.reference_ts}"
            )

    if cfg.ancestral_state is not None:
        p = Path(str(cfg.ancestral_state.path))
        if not p.exists():
            errors.append(
                f"Ancestral state path does not exist: {cfg.ancestral_state.path}"
            )

    return errors
