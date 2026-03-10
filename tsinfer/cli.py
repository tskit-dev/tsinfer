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

import click


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
    raise NotImplementedError


@main.command("match")
@_config_arg
@_add_options(_runtime_options)
def match_cmd(config, threads, force, progress, verbose):
    """Run the unified match loop (ancestors + samples)."""
    raise NotImplementedError


@main.command("post-process")
@_config_arg
@_add_options(_runtime_options)
def post_process_cmd(config, threads, force, progress, verbose):
    """Post-process a matched tree sequence."""
    raise NotImplementedError


@main.command("run")
@_config_arg
@_add_options(_runtime_options)
def run_cmd(config, threads, force, progress, verbose):
    """Run the full pipeline: infer-ancestors, match, post-process."""
    raise NotImplementedError


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
    raise NotImplementedError


@config_group.command("check")
@_config_arg
def config_check(config):
    """Validate the config and verify all input paths exist."""
    raise NotImplementedError
