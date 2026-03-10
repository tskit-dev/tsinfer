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
Matching engine: Matcher, grouping algorithm, and tree sequence extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tskit


@dataclass
class MatchResult:
    """The output of matching a single haplotype against the reference panel."""

    path_left: np.ndarray  # (n_edges,) int32 — left breakpoints (site indices)
    path_right: np.ndarray  # (n_edges,) int32 — right breakpoints (site indices)
    path_parent: np.ndarray  # (n_edges,) int32 — parent node ids
    mutation_sites: np.ndarray  # (n_mutations,) int32 — site indices
    mutation_state: np.ndarray  # (n_mutations,) int8  — derived allele index


class Matcher:
    """
    Matches haplotypes against a fixed reference tree sequence using the C engine.

    Wraps _tsinfer.AncestorMatcher via restore_tree_sequence_builder. A fresh
    Matcher is instantiated for each group's tree sequence; no state is maintained
    between groups.
    """

    def __init__(
        self,
        ts: tskit.TreeSequence,
        positions: np.ndarray,  # (n_sites,) int32 — inference site positions
        recombination_rate,  # float or msprime.RateMap
        mismatch_ratio: float = 1.0,
        path_compression: bool = True,
        num_threads: int = 1,
    ):
        raise NotImplementedError

    def match(
        self,
        haplotypes: np.ndarray,  # (n_haplotypes, n_sites) int8
    ) -> list[MatchResult]:
        """
        Run the HMM for each haplotype. Active range per haplotype is derived from
        its missing data pattern (first and last non-missing site).
        """
        raise NotImplementedError


def compute_groups(
    times: np.ndarray,  # (n_haplotypes,) float64
    is_ancestor: np.ndarray,  # (n_haplotypes,) bool
    start_positions: np.ndarray,  # (n_haplotypes,) int32 — from sample_start_position
    end_positions: np.ndarray,  # (n_haplotypes,) int32 — from sample_end_position
) -> list[np.ndarray]:
    """
    Return an ordered list of haplotype-index arrays, oldest group first.

    Ancestor haplotypes use the linesweep regime (for early epochs) or epoch
    regime (for large epochs). Sample haplotypes are grouped strictly by time.
    Ancestor groups at a given time are placed before same-time sample groups.

    The first element always contains only the virtual root ancestor (index 0
    by convention in the ancestor VCZ).
    """
    raise NotImplementedError


def make_root_ts(
    sequence_length: float,
    positions: np.ndarray,  # (n_sites,) int32
    sequence_intervals: np.ndarray,  # (n_intervals, 2) int32
) -> tskit.TreeSequence:
    """
    Build the empty root tree sequence that starts the match loop.
    Stores sequence_intervals in the tree sequence top-level metadata.
    """
    raise NotImplementedError


def extend_ts(
    ts: tskit.TreeSequence,
    node_times: np.ndarray,  # (n_haplotypes,) float64
    results: list[MatchResult],
    node_metadata: list[dict],  # provenance metadata per haplotype
    create_individuals: np.ndarray,  # (n_haplotypes,) bool
    ploidy: int = 1,
    path_compression: bool = True,
) -> tskit.TreeSequence:
    """
    Extend ts with the match results for one group.

    Adds nodes, edges (clipped at gap boundaries from ts metadata), mutations,
    and individuals. When create_individuals[i] is True, nodes are grouped into
    tskit individuals at one individual per ploidy nodes.

    Returns the updated tree sequence, which becomes the reference panel for
    the next group.
    """
    raise NotImplementedError
