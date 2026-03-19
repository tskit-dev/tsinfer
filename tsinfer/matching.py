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
import time as time_
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import tskit

import _tsinfer

from .grouping import MatchJob

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


def _tsb_from_ts(
    ts: tskit.TreeSequence,
    num_sites: int,
    positions: np.ndarray,
    num_alleles: np.ndarray | None = None,
):
    """Restore a _tsinfer.TreeSequenceBuilder from a tskit.TreeSequence.

    The intermediate tree sequences use integer-string alleles ("0", "1", …)
    so mutation derived_state values can be converted directly to int codes.
    """
    if num_alleles is None:
        num_alleles = np.full(num_sites, 2, dtype=np.uint64)
    tsb = _tsinfer.TreeSequenceBuilder(num_alleles)

    t0 = time_.monotonic()
    if ts.num_nodes > 0:
        tsb.restore_nodes(ts.nodes_time, ts.nodes_flags)
    t_nodes = time_.monotonic() - t0

    t_build_edges = 0
    t_restore_edges = 0
    if ts.num_edges > 0:
        t0 = time_.monotonic()
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
        t_build_edges = time_.monotonic() - t0
        t0 = time_.monotonic()
        tsb.restore_edges(el[order], er[order], ep[order], ec[order])
        t_restore_edges = time_.monotonic() - t0

    t_build_mutations = 0
    t_restore_mutations = 0
    if ts.num_mutations > 0:
        t0 = time_.monotonic()
        # Intermediate TS uses integer-string alleles, so derived_state
        # values like "0", "1", "2" map directly to allele codes.
        mutations_state = np.array(
            [int(ds) for ds in ts.mutations_derived_state], dtype=np.int8
        )
        t_build_mutations = time_.monotonic() - t0
        t0 = time_.monotonic()
        tsb.restore_mutations(
            ts.mutations_site,
            ts.mutations_node,
            mutations_state,
            ts.mutations_parent,
        )
        t_restore_mutations = time_.monotonic() - t0

    logger.info(
        "TSB build: nodes %.3fs; build_edges: %.3fs; restore_edges: %.3fs "
        "build_mutations: %.3fs; restore_mutations %.3fs",
        t_nodes,
        t_build_edges,
        t_restore_edges,
        t_build_mutations,
        t_restore_mutations,
    )

    tsb.freeze_indexes()
    return tsb


def make_root_ts(
    sequence_length: float,
    positions: np.ndarray,  # (num_sites,) int32
    sequence_intervals: np.ndarray,  # (n_intervals, 2) int32
    max_time: float = 0.0,
) -> tskit.TreeSequence:
    """
    Build the root tree sequence that starts the match loop.

    Creates two nodes — an ultimate root and a virtual root — connected by
    an edge spanning ``[positions[0], sequence_length)``.  This gives
    ``AncestorMatcher`` a valid tree to copy from when matching the first
    ancestor.

    The virtual root is placed at ``max_time + 1`` and the ultimate root at
    ``max_time + 2``, ensuring both are strictly older than any ancestor.

    Sites use integer-string alleles ("0" = ancestral) so that intermediate
    tree sequences can round-trip allele codes through _tsb_from_ts without
    a string lookup table.

    Stores *sequence_intervals* in the tree sequence top-level metadata.
    """
    virtual_root_time = max_time + 1
    ultimate_root_time = max_time + 2

    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {"sequence_intervals": np.asarray(sequence_intervals).tolist()}

    positions = np.asarray(positions)
    for pos in positions:
        tables.sites.add_row(position=float(pos), ancestral_state="0")

    tables.nodes.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.individuals.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    # Node 0: ultimate root
    tables.nodes.add_row(time=ultimate_root_time, flags=0, metadata={})
    # Node 1: virtual root
    tables.nodes.add_row(time=virtual_root_time, flags=0, metadata={})

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
        path_compression: bool = True,
        num_alleles: np.ndarray | None = None,
    ):
        self._positions = np.asarray(positions, dtype=np.int32)
        self._num_sites = len(positions)
        self._sequence_length = ts.sequence_length
        self._path_compression = path_compression

        t0 = time_.monotonic()
        self._tsb = _tsb_from_ts(
            ts,
            self._num_sites,
            self._positions,
            num_alleles=num_alleles,
        )
        logger.info("Create matcher tsb in %.3fs", time_.monotonic() - t0)
        self._rho = np.full(self._num_sites, 1e-2)
        self._mu = np.full(self._num_sites, 1e-20)

    def _match_one(self, job, reader) -> tuple:
        """Match a single haplotype: read, run HMM, convert to result."""
        num_sites = self._num_sites
        pos = self._positions.astype(np.float64)
        seq_len = self._sequence_length

        # Each call gets its own AncestorMatcher so threads don't share state
        t0 = time_.monotonic()
        matcher = _tsinfer.AncestorMatcher(self._tsb, self._rho, self._mu)
        t_init = time_.monotonic() - t0

        t0 = time_.monotonic()
        h = np.asarray(reader.read_haplotype(job), dtype=np.int8)
        t_read = time_.monotonic() - t0

        match_out = np.zeros(num_sites, dtype=np.int8)
        non_missing = np.where(h >= 0)[0]
        if len(non_missing) == 0:
            start, end = 0, num_sites
        else:
            start = int(non_missing[0])
            end = int(non_missing[-1]) + 1

        t1 = time_.monotonic()
        left, right, parent = matcher.find_path(h, start, end, match_out)
        t_match = time_.monotonic() - t1

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
            Mutation(position=float(pos[si]), derived_state=int(h[si]))
            for si in mut_site_idxs
        ]

        logger.debug(
            "Matched job %s: init=%.4fs read=%.4fs match=%.4fs "
            "edges=%d mutations=%d sites=%d..%d "
            "matcher_mem=%.1fMiB mean_tb=%.1f",
            getattr(job, "haplotype_index", "?"),
            t_init,
            t_read,
            t_match,
            len(path),
            len(mutations),
            start,
            end,
            matcher.total_memory / (1024 * 1024),
            matcher.mean_traceback_size,
        )

        return (job, MatchResult(path=path, mutations=mutations))

    def match(
        self,
        jobs,
        reader,
        num_threads: int = 1,
    ):
        """
        Run the HMM for each job, yielding ``(job, MatchResult)`` pairs
        as they complete.

        When *num_threads* > 1 the completion order may differ from the
        input *jobs* order.

        Parameters
        ----------
        jobs : iterable
            Objects passed to ``reader.read_haplotype(job)`` one at a time.
        reader : object
            Anything with a ``read_haplotype(job) -> np.ndarray`` method.
        num_threads : int
            Maximum worker threads (default 1 — sequential).
        """
        jobs = sorted(jobs, key=lambda j: j.haplotype_index if j is not None else -1)
        with ThreadPoolExecutor(max_workers=max(1, num_threads)) as executor:
            futures = {
                executor.submit(self._match_one, job, reader): job for job in jobs
            }
            for future in as_completed(futures):
                yield future.result()


def extend_ts(
    ts: tskit.TreeSequence,
    *,
    paired_results: list[tuple[MatchJob, MatchResult]],
) -> tskit.TreeSequence:
    """
    Extend ts with the match results for one group.

    Adds nodes, edges, mutations, and individuals directly via the tskit
    Tables API — no TSB roundtrip.  When ``job.create_individuals`` is True,
    nodes are grouped by ``(source, sample_id)`` into tskit individuals.
    Within each individual, nodes are ordered by ``haplotype_index``.

    Mutation derived states are stored as integer-string allele codes
    ("0", "1", "2", …) matching the codes produced by ``_AlleleMap``.
    The caller is responsible for relabeling to real allele strings before
    the tree sequence is returned to the user.

    Returns the updated tree sequence, which becomes the reference panel for
    the next group.
    """
    t_start = time_.monotonic()

    # Sort by haplotype_index so nodes within each individual are ordered
    paired_results = sorted(paired_results, key=lambda p: p[0].haplotype_index)

    tables = ts.dump_tables()
    pos_arr = tables.sites.position.astype(np.float64)

    # Add nodes, edges, mutations for each (job, result) pair
    new_node_ids = []
    for job, result in paired_results:
        node_meta = {
            "source": job.source,
            "sample_id": job.sample_id,
            "ploidy_index": job.ploidy_index,
        }
        node_id = tables.nodes.add_row(
            time=job.time, flags=job.node_flags, metadata=node_meta
        )
        new_node_ids.append(node_id)

        for seg in result.path:
            tables.edges.add_row(
                left=seg.left, right=seg.right, parent=seg.parent, child=node_id
            )

        for mut in result.mutations:
            site_id = int(np.searchsorted(pos_arr, mut.position))
            tables.mutations.add_row(
                site=site_id,
                node=node_id,
                derived_state=str(mut.derived_state),
                parent=-1,
            )

    # Build individuals: group by (source, sample_id), with individual metadata
    ind_key_to_id: dict[tuple[str, str], int] = {}
    for i, (job, _result) in enumerate(paired_results):
        if job.create_individuals:
            key = (job.source, job.sample_id)
            if key not in ind_key_to_id:
                ind_meta = {"source": job.source, "sample_id": job.sample_id}
                ind_key_to_id[key] = tables.individuals.add_row(metadata=ind_meta)
            nid = new_node_ids[i]
            row = tables.nodes[nid]
            tables.nodes[nid] = row.replace(individual=ind_key_to_id[key])

    t0 = time_.monotonic()
    tables.sort()
    t_sort = time_.monotonic() - t0

    t0 = time_.monotonic()
    tables.build_index()
    t_index = time_.monotonic() - t0

    t0 = time_.monotonic()
    tables.compute_mutation_parents()
    t_mut_parents = time_.monotonic() - t0

    t0 = time_.monotonic()
    result_ts = tables.tree_sequence()
    t_ts = time_.monotonic() - t0

    logger.info(
        "extend_ts: added %d nodes (%d total), %d edges, %d mutations, "
        "%d individuals | sort=%.3fs index=%.3fs mut_parents=%.3fs "
        "tree_sequence=%.3fs total=%.3fs",
        len(new_node_ids),
        result_ts.num_nodes,
        result_ts.num_edges,
        result_ts.num_mutations,
        result_ts.num_individuals,
        t_sort,
        t_index,
        t_mut_parents,
        t_ts,
        time_.monotonic() - t_start,
    )

    return result_ts


def relabel_alleles(
    ts: tskit.TreeSequence,
    site_alleles: np.ndarray,
) -> tskit.TreeSequence:
    """
    Replace integer-string alleles with real allele strings.

    During the match loop, sites use "0" as ancestral_state and mutations
    use "1", "2", … as derived_state.  This function maps them back to the
    actual allele strings from *site_alleles*.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence with integer-string alleles.
    site_alleles : np.ndarray
        ``(num_sites, max_alleles)`` object array where
        ``site_alleles[i, code]`` is the real allele string.
    """
    tables = ts.dump_tables()

    # Relabel ancestral states
    old_sites = tables.sites.copy()
    tables.sites.clear()
    for i in range(old_sites.num_rows):
        code = int(old_sites[i].ancestral_state)
        tables.sites.add_row(
            position=old_sites[i].position,
            ancestral_state=str(site_alleles[i, code]),
            metadata=old_sites[i].metadata,
        )

    # Relabel mutation derived states
    old_muts = tables.mutations.copy()
    tables.mutations.clear()
    for i in range(old_muts.num_rows):
        mut = old_muts[i]
        code = int(mut.derived_state)
        tables.mutations.add_row(
            site=mut.site,
            node=mut.node,
            derived_state=str(site_alleles[mut.site, code]),
            parent=mut.parent,
            time=mut.time,
            metadata=mut.metadata,
        )

    return tables.tree_sequence()
