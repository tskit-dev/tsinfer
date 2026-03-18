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

import logging
from dataclasses import dataclass

import numpy as np
import tskit

import _tsinfer

logger = logging.getLogger(__name__)


@dataclass
class PathSegment:
    """One edge in the copying path: the haplotype copies from *parent*
    over the genomic interval ``[left, right)``."""

    left: float  # absolute genomic position
    right: float  # absolute genomic position
    parent: int  # node id in the reference tree sequence


@dataclass
class Mutation:
    """A single mutation placed by the matching HMM."""

    position: float  # absolute genomic position
    derived_state: int  # allele index (typically 1)


@dataclass
class MatchResult:
    """The output of matching a single haplotype against the reference panel."""

    path: list[PathSegment]
    mutations: list[Mutation]


def _tsb_from_ts(ts: tskit.TreeSequence, num_sites: int, positions: np.ndarray):
    """Restore a _tsinfer.TreeSequenceBuilder from a tskit.TreeSequence."""
    num_alleles = [2] * num_sites
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
            num_sites,
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
    num_sites: int,
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
    if metadata is not None:
        tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        tables.metadata = metadata

    for pos in positions:
        tables.sites.add_row(position=float(pos), ancestral_state="0")

    # Create populations if specified
    if population_names is not None:
        tables.populations.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        for pop_name in population_names:
            tables.populations.add_row(metadata={"name": pop_name})

    # Set up node metadata schema if we have node metadata
    if node_metadata is not None:
        tables.nodes.metadata_schema = tskit.MetadataSchema({"codec": "json"})

    flags, times = tsb.dump_nodes()
    for i, (t, fl) in enumerate(zip(times, flags)):
        md = (
            node_metadata[i]
            if node_metadata is not None and i < len(node_metadata)
            else None
        )
        if md is not None:
            tables.nodes.add_row(time=float(t), flags=int(fl), metadata=md)
        else:
            tables.nodes.add_row(time=float(t), flags=int(fl))

    if individuals is not None:
        if individual_metadata is not None:
            tables.individuals.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        for ind_idx, ind_nodes in enumerate(individuals):
            ind_md = (
                individual_metadata[ind_idx]
                if individual_metadata is not None and ind_idx < len(individual_metadata)
                else None
            )
            pop_id = (
                populations[ind_idx]
                if populations is not None and ind_idx < len(populations)
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
        right_coord = float(pos_arr[re]) if re < num_sites else float(sequence_length)
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


def make_root_ts(
    sequence_length: float,
    positions: np.ndarray,  # (num_sites,) int32
    sequence_intervals: np.ndarray,  # (n_intervals, 2) int32
) -> tskit.TreeSequence:
    """
    Build the root tree sequence that starts the match loop.

    Creates two nodes — an ultimate root (time=2.0) and a virtual root
    (time=1.0) — connected by an edge spanning ``[positions[0],
    sequence_length)``.  This gives ``AncestorMatcher`` a valid tree to
    copy from when matching the first ancestor.

    Stores *sequence_intervals* in the tree sequence top-level metadata.
    """
    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {"sequence_intervals": np.asarray(sequence_intervals).tolist()}

    positions = np.asarray(positions)
    for pos in positions:
        tables.sites.add_row(position=float(pos), ancestral_state="0")

    # Node 0: ultimate root (time=2.0)
    tables.nodes.add_row(time=2.0, flags=0)
    # Node 1: virtual root (time=1.0)
    tables.nodes.add_row(time=1.0, flags=0)

    if len(positions) > 0:
        tables.edges.add_row(
            left=float(positions[0]),
            right=float(sequence_length),
            parent=0,
            child=1,
        )

    tables.sort()
    tables.build_index()
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
        positions: np.ndarray,  # (num_sites,) int32 — inference site positions
        recombination_rate,  # float or msprime.RateMap
        mismatch_ratio: float = 1.0,
        path_compression: bool = True,
        num_threads: int = 1,
    ):
        self._positions = np.asarray(positions, dtype=np.int32)
        self._num_sites = len(positions)
        self._sequence_length = ts.sequence_length
        self._path_compression = path_compression

        tsb = _tsb_from_ts(ts, self._num_sites, self._positions)

        d = np.diff(
            self._positions.astype(np.float64), prepend=float(self._positions[0])
        )
        rho = float(recombination_rate) * np.maximum(d, 1.0)
        rho = np.clip(rho, 1e-10, 1.0 - 1e-10)

        num_match = max(1, tsb.num_match_nodes)
        mu = np.full(self._num_sites, mismatch_ratio / num_match)
        mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

        self._matcher = _tsinfer.AncestorMatcher(tsb, rho.tolist(), mu.tolist())
        self._num_sites_val = self._num_sites

    def match(self, jobs, reader) -> list[MatchResult]:
        """
        Run the HMM for each job, reading haplotypes on demand via *reader*.

        Parameters
        ----------
        jobs : iterable
            Objects passed to ``reader.read_haplotype(job)`` one at a time.
        reader : object
            Anything with a ``read_haplotype(job) -> np.ndarray`` method.
        """
        num_sites = self._num_sites_val
        pos = self._positions.astype(np.float64)
        seq_len = self._sequence_length
        results = []
        for job in jobs:
            h = np.asarray(reader.read_haplotype(job), dtype=np.int8)
            match_out = np.zeros(num_sites, dtype=np.int8)
            non_missing = np.where(h >= 0)[0]
            if len(non_missing) == 0:
                start, end = 0, num_sites
            else:
                start = int(non_missing[0])
                end = int(non_missing[-1]) + 1

            left, right, parent = self._matcher.find_path(h, start, end, match_out)

            # Convert site-index edges to absolute-position PathSegments
            path = []
            for li, ri, pi in zip(left, right, parent):
                l_pos = float(pos[li])
                r_pos = float(pos[ri]) if ri < num_sites else float(seq_len)
                path.append(PathSegment(left=l_pos, right=r_pos, parent=int(pi)))

            # Detect mutations and convert to absolute positions
            in_range = np.zeros(num_sites, dtype=bool)
            in_range[start:end] = True
            mutation_mask = in_range & (h != match_out) & (h >= 0)
            mut_site_idxs = np.where(mutation_mask)[0]
            mutations = [
                Mutation(
                    position=float(pos[si]),
                    derived_state=int(h[si]),
                )
                for si in mut_site_idxs
            ]

            results.append(MatchResult(path=path, mutations=mutations))
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
    positions = ts.tables.sites.position
    num_sites = ts.num_sites
    seq_len = ts.sequence_length
    meta = dict(ts.metadata) if ts.metadata is not None else {}

    tsb = _tsb_from_ts(ts, num_sites, positions)

    pos_arr = positions.astype(np.float64)

    new_node_ids = []
    for time, result in zip(node_times, results):
        node_id = tsb.add_node(float(time))
        new_node_ids.append(node_id)

        if len(result.path) > 0:
            # Convert absolute positions back to site indices for the TSB
            left_idxs = [int(np.searchsorted(pos_arr, seg.left)) for seg in result.path]
            right_idxs = [
                int(np.searchsorted(pos_arr, seg.right))
                if seg.right < seq_len
                else num_sites
                for seg in result.path
            ]
            parents = [seg.parent for seg in result.path]
            tsb.add_path(
                child=node_id,
                left=left_idxs,
                right=right_idxs,
                parent=parents,
                compress=path_compression,
            )

        if len(result.mutations) > 0:
            site_idxs = [
                int(np.searchsorted(pos_arr, m.position)) for m in result.mutations
            ]
            derived = [m.derived_state for m in result.mutations]
            tsb.add_mutations(
                node=node_id,
                site=site_idxs,
                derived_state=derived,
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

    # Build full metadata list: preserve existing node metadata, append new
    existing_meta = []
    has_existing_meta = ts.tables.nodes.metadata_schema.schema is not None
    for node in ts.nodes():
        if has_existing_meta and node.metadata:
            existing_meta.append(node.metadata)
        else:
            existing_meta.append(None)
    full_node_metadata = existing_meta + list(node_metadata)

    return _ts_from_tsb(
        tsb,
        num_sites,
        positions,
        seq_len,
        meta,
        individuals if len(individuals) > 0 else None,
        node_metadata=full_node_metadata,
    )
