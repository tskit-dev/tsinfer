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

import _tsinfer


@dataclass
class MatchResult:
    """The output of matching a single haplotype against the reference panel."""

    path_left: np.ndarray  # (n_edges,) int32 — left breakpoints (site indices)
    path_right: np.ndarray  # (n_edges,) int32 — right breakpoints (site indices)
    path_parent: np.ndarray  # (n_edges,) int32 — parent node ids
    mutation_sites: np.ndarray  # (n_mutations,) int32 — site indices
    mutation_state: np.ndarray  # (n_mutations,) int8  — derived allele index


def _tsb_from_ts(ts: tskit.TreeSequence, n_sites: int, positions: np.ndarray):
    """Restore a _tsinfer.TreeSequenceBuilder from a tskit.TreeSequence."""
    num_alleles = [2] * n_sites
    tsb = _tsinfer.TreeSequenceBuilder(num_alleles)

    if ts.num_nodes > 0:
        node_flags = np.array([n.flags for n in ts.nodes()], dtype=np.uint32)
        node_times = np.array([n.time for n in ts.nodes()], dtype=np.float64)
        tsb.restore_nodes(node_times, node_flags)

    if ts.num_edges > 0:
        pos_arr = positions.astype(np.float64)
        seq_len = ts.sequence_length
        edges = ts.tables.edges
        el = np.searchsorted(pos_arr, edges.left).astype(np.int32)
        er = np.where(
            edges.right < seq_len,
            np.searchsorted(pos_arr, edges.right),
            n_sites,
        ).astype(np.int32)
        ep = edges.parent.astype(np.int32)
        ec = edges.child.astype(np.int32)
        order = np.lexsort([el, ec])
        tsb.restore_edges(el[order], er[order], ep[order], ec[order])

    if ts.num_mutations > 0:
        ms = np.array([m.site for m in ts.mutations()], dtype=np.int32)
        mn = np.array([m.node for m in ts.mutations()], dtype=np.int32)
        md = np.array([int(m.derived_state) for m in ts.mutations()], dtype=np.int8)
        mp = np.array([m.parent for m in ts.mutations()], dtype=np.int32)
        tsb.restore_mutations(ms, mn, md, mp)

    tsb.freeze_indexes()
    return tsb


def _ts_from_tsb(
    tsb,
    n_sites: int,
    positions: np.ndarray,
    sequence_length: float,
    metadata: dict,
    individuals: list[list[int]] | None = None,
    node_metadata: list[dict] | None = None,
    individual_metadata: list[dict] | None = None,
    populations: list[int] | None = None,
    population_names: list[str] | None = None,
) -> tskit.TreeSequence:
    """Convert a _tsinfer.TreeSequenceBuilder to a tskit.TreeSequence.

    Parameters
    ----------
    node_metadata:
        Per-node metadata dicts, one per node in TSB order.
    individual_metadata:
        Per-individual metadata dicts, one per individual group.
    populations:
        Per-individual population index, one per individual group.
    population_names:
        Unique population names; used to create population table rows.
    """
    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    if metadata:
        tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        tables.metadata = metadata

    for pos in positions:
        tables.sites.add_row(position=float(pos), ancestral_state="0")

    # Create populations if specified
    if population_names:
        tables.populations.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        for pop_name in population_names:
            tables.populations.add_row(metadata={"name": pop_name})

    # Set up node metadata schema if we have node metadata
    if node_metadata:
        tables.nodes.metadata_schema = tskit.MetadataSchema({"codec": "json"})

    flags, times = tsb.dump_nodes()
    for i, (t, fl) in enumerate(zip(times, flags)):
        md = node_metadata[i] if node_metadata and i < len(node_metadata) else None
        if md is not None:
            tables.nodes.add_row(time=float(t), flags=int(fl), metadata=md)
        else:
            tables.nodes.add_row(time=float(t), flags=int(fl))

    if individuals:
        if individual_metadata:
            tables.individuals.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        for ind_idx, ind_nodes in enumerate(individuals):
            ind_md = (
                individual_metadata[ind_idx]
                if individual_metadata and ind_idx < len(individual_metadata)
                else None
            )
            pop_id = (
                populations[ind_idx]
                if populations and ind_idx < len(populations)
                else -1
            )
            if ind_md is not None:
                ind_id = tables.individuals.add_row(metadata=ind_md)
            else:
                ind_id = tables.individuals.add_row()
            for node_id in ind_nodes:
                row = tables.nodes[node_id]
                tables.nodes[node_id] = row.replace(
                    individual=ind_id,
                    population=pop_id if pop_id >= 0 else -1,
                )

    pos_arr = positions.astype(np.float64)
    el, er, ep, ec = tsb.dump_edges()
    for le, re, pe, ce in zip(el, er, ep, ec):
        left_coord = float(pos_arr[le])
        right_coord = float(pos_arr[re]) if re < n_sites else float(sequence_length)
        tables.edges.add_row(
            left=left_coord, right=right_coord, parent=int(pe), child=int(ce)
        )

    alleles = ["0", "1"]
    ms, mn, md, _mp = tsb.dump_mutations()
    for s, n, d in zip(ms, mn, md):
        tables.mutations.add_row(
            site=int(s), node=int(n), derived_state=alleles[int(d)], parent=-1
        )

    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def compute_groups(
    times: np.ndarray,  # (n_haplotypes,) float64
    is_ancestor: np.ndarray,  # (n_haplotypes,) bool
    start_positions: np.ndarray,  # (n_haplotypes,) int32 — from sample_start_position
    end_positions: np.ndarray,  # (n_haplotypes,) int32 — from sample_end_position
) -> list[np.ndarray]:
    """
    Return an ordered list of haplotype-index arrays, oldest group first.

    - Index 0 (virtual root, is_ancestor=True with time=1.0) is always returned alone
      as groups[0].
    - Remaining ancestors (is_ancestor=True) are grouped by unique time values (same
      time → same group), ordered by descending time.
    - Sample haplotypes (is_ancestor=False) are grouped strictly by time, ordered by
      descending time (oldest ancient samples first, modern at time=0 last).
    - At the same time, ancestor groups come before sample groups.
    """
    times = np.asarray(times, dtype=np.float64)
    is_ancestor = np.asarray(is_ancestor, dtype=bool)

    # Group 0: virtual root is always index 0
    groups = [np.array([0], dtype=np.int32)]

    # Separate remaining ancestors and samples
    anc_indices = np.where(is_ancestor)[0]
    anc_indices = anc_indices[anc_indices != 0]  # exclude virtual root

    sample_indices = np.where(~is_ancestor)[0]

    # Build groups: all unique times, ancestors before samples at same time
    all_times_set = set()
    if len(anc_indices) > 0:
        all_times_set.update(times[anc_indices].tolist())
    if len(sample_indices) > 0:
        all_times_set.update(times[sample_indices].tolist())

    for t in sorted(all_times_set, reverse=True):
        # Ancestors at this time (if any)
        if len(anc_indices) > 0:
            anc_at_t = anc_indices[np.isclose(times[anc_indices], t)]
        else:
            anc_at_t = np.array([], dtype=np.int32)
        if len(anc_at_t) > 0:
            groups.append(anc_at_t.astype(np.int32))
        # Samples at this time (if any)
        if len(sample_indices) > 0:
            samp_at_t = sample_indices[np.isclose(times[sample_indices], t)]
            if len(samp_at_t) > 0:
                groups.append(samp_at_t.astype(np.int32))

    return groups


def make_root_ts(
    sequence_length: float,
    positions: np.ndarray,  # (n_sites,) int32
    sequence_intervals: np.ndarray,  # (n_intervals, 2) int32
) -> tskit.TreeSequence:
    """
    Build the empty root tree sequence that starts the match loop.
    Stores sequence_intervals in the tree sequence top-level metadata.
    No nodes are added (the virtual root node is added in the match loop).
    """
    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {"sequence_intervals": np.asarray(sequence_intervals).tolist()}

    positions = np.asarray(positions)
    for pos in positions:
        tables.sites.add_row(position=float(pos), ancestral_state="0")

    return tables.tree_sequence()


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
        self._positions = np.asarray(positions, dtype=np.int32)
        self._n_sites = len(positions)
        self._path_compression = path_compression

        tsb = _tsb_from_ts(ts, self._n_sites, self._positions)

        d = np.diff(
            self._positions.astype(np.float64), prepend=float(self._positions[0])
        )
        rho = float(recombination_rate) * np.maximum(d, 1.0)
        rho = np.clip(rho, 1e-10, 1.0 - 1e-10)

        n_match = max(1, tsb.num_match_nodes)
        mu = np.full(self._n_sites, mismatch_ratio / n_match)
        mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

        self._matcher = _tsinfer.AncestorMatcher(tsb, rho.tolist(), mu.tolist())
        self._n_sites_val = self._n_sites

    def match(
        self,
        haplotypes: np.ndarray,  # (n_haplotypes, n_sites) int8
    ) -> list[MatchResult]:
        """
        Run the HMM for each haplotype. Active range per haplotype is derived from
        its missing data pattern (first and last non-missing site).
        """
        n_sites = self._n_sites_val
        results = []
        for h in haplotypes:
            h = np.asarray(h, dtype=np.int8)
            match_out = np.zeros(n_sites, dtype=np.int8)
            non_missing = np.where(h >= 0)[0]
            if len(non_missing) == 0:
                start, end = 0, n_sites
            else:
                start = int(non_missing[0])
                end = int(non_missing[-1]) + 1

            left, right, parent = self._matcher.find_path(h, start, end, match_out)

            in_range = np.zeros(n_sites, dtype=bool)
            in_range[start:end] = True
            mutation_mask = in_range & (h != match_out) & (h >= 0)
            mutation_sites = np.where(mutation_mask)[0].astype(np.int32)
            mutation_state = h[mutation_sites].astype(np.int8)

            results.append(
                MatchResult(
                    path_left=np.asarray(left, dtype=np.int32),
                    path_right=np.asarray(right, dtype=np.int32),
                    path_parent=np.asarray(parent, dtype=np.int32),
                    mutation_sites=mutation_sites,
                    mutation_state=mutation_state,
                )
            )
        return results


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
    positions = np.array([s.position for s in ts.sites()])
    n_sites = ts.num_sites
    seq_len = ts.sequence_length
    meta = dict(ts.metadata) if ts.metadata else {}

    tsb = _tsb_from_ts(ts, n_sites, positions)

    new_node_ids = []
    for time, result in zip(node_times, results):
        node_id = tsb.add_node(float(time))
        new_node_ids.append(node_id)

        if len(result.path_left) > 0:
            tsb.add_path(
                child=node_id,
                left=result.path_left.tolist(),
                right=result.path_right.tolist(),
                parent=result.path_parent.tolist(),
                compress=path_compression,
            )

        if len(result.mutation_sites) > 0:
            tsb.add_mutations(
                node=node_id,
                site=result.mutation_sites.tolist(),
                derived_state=result.mutation_state.tolist(),
            )

    # Build individuals (group consecutive create_individuals=True haplotypes by ploidy)
    individuals = []
    i = 0
    while i < len(new_node_ids):
        if bool(create_individuals[i]):
            ind_nodes = new_node_ids[i : i + ploidy]
            individuals.append(ind_nodes)
            i += ploidy
        else:
            i += 1

    return _ts_from_tsb(
        tsb, n_sites, positions, seq_len, meta, individuals if individuals else None
    )
