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

import tskit

from .config import Config


def match(
    cfg: Config,
    reference_ts: tskit.TreeSequence | None = None,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Run the unified match loop over all sources listed in cfg.match.

    Collects haplotypes from the ancestor VCZ and all sample sources, groups
    them by time using compute_groups, then iterates: match each group against
    the current tree sequence and extend it with the results.

    reference_ts: if provided, skip building from inferred ancestors and match
    against this tree sequence instead.
    kwargs: override any MatchConfig fields for this call.
    """
    raise NotImplementedError


def post_process(
    ts: tskit.TreeSequence,
    cfg: Config,
    **kwargs,
) -> tskit.TreeSequence:
    """
    Post-process a matched tree sequence.

    Applies simplification, splits ultimate ancestors, and erases flanking
    regions outside sequence_intervals, as configured in cfg.
    Post-processing is always an explicit separate call; it is never implicit.
    """
    raise NotImplementedError


def run(cfg: Config, **kwargs) -> tskit.TreeSequence:
    """
    Run the full pipeline: infer_ancestors, match, post_process.

    Equivalent to calling each step in sequence with the given config.
    kwargs are forwarded to each step.
    """
    raise NotImplementedError
