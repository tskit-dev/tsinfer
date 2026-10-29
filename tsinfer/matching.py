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

import concurrent.futures as cf
import dataclasses
import logging
import pathlib
import time as time_

import numpy as np
import psutil
import tskit

import _tsinfer

from . import arg_ops, config, vcz

try:
    import tsls
except ImportError:
    tsls = None

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PathSegment:
    """One edge in the copying path: the haplotype copies from *parent*
    over the genomic interval ``[left, right)``."""

    left: float  # absolute genomic position
    right: float  # absolute genomic position
    parent: int  # node id in the reference tree sequence


@dataclasses.dataclass
class Mutation:
    """A single mutation placed by the matching HMM."""

    position: float  # absolute genomic position
    derived_state: int  # allele code at this site (0 = ancestral)


@dataclasses.dataclass
class MatchResult:
    """The output of matching a single haplotype against the reference panel."""

    path: list[PathSegment]
    mutations: list[Mutation]


# Defaults used when no recombination or mismatch values are given. A
# high recombination probability and negligible mismatch probability
# means that haplotypes are copied with no mismatches.
DEFAULT_RECOMBINATION = 1e-2
DEFAULT_MISMATCH = 1e-20


@dataclasses.dataclass
class HmmParameters:
    """Per-site parameters of the Li & Stephens copying HMM."""

    recombination: np.ndarray  # (num_sites,) float64 probabilities
    mismatch: np.ndarray  # (num_sites,) float64 probabilities
    likelihood_threshold: float | None = None


def read_genetic_map(path: str | pathlib.Path, positions: np.ndarray) -> np.ndarray:
    """
    Read a HapMap-format genetic map and return the genetic position (in
    Morgans) of each of the specified physical positions.

    The file has a single header line and whitespace-separated columns:
    chromosome, position (bp), rate (cM/Mb) and cumulative map position (cM).
    Only the position and map columns are used. Genetic positions are linearly
    interpolated between map entries, with an implicit entry of 0 cM at
    position 0. Beyond the last entry the recombination rate is zero. This
    follows ``msprime.RateMap.read_hapmap``.
    """
    map_position, map_cm = np.loadtxt(
        path, skiprows=1, usecols=(1, 3), unpack=True, ndmin=2
    )
    if len(map_position) == 0:
        raise ValueError(f"Genetic map {path} is empty")
    if np.any(np.diff(map_position) <= 0):
        raise ValueError(f"Genetic map {path} positions are not strictly increasing")
    if np.any(np.diff(map_cm) < 0):
        raise ValueError(f"Genetic map {path} map positions (cM) decrease")
    map_morgans = map_cm / 100
    if map_position[0] > 0:
        map_position = np.insert(map_position, 0, 0)
        map_morgans = np.insert(map_morgans, 0, 0)
    return np.interp(positions, map_position, map_morgans)


def recombination_probabilities(
    recombination_rate: float | str | pathlib.Path, positions: np.ndarray
) -> np.ndarray:
    """
    Return the probability of recombination between each inference site and
    the previous one, given either a constant rate per bp or the path to a
    HapMap genetic map. The genetic distance d (Morgans) between adjacent sites
    is converted to the probability of an odd number of crossovers using
    Haldane's map function, (1 - exp(-2d)) / 2. The first site has probability 0.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if isinstance(recombination_rate, (str, pathlib.Path)):
        genetic_position = read_genetic_map(recombination_rate, positions)
        distance = np.diff(genetic_position)
    else:
        if recombination_rate < 0:
            raise ValueError("Recombination rate must be non-negative")
        distance = np.diff(positions) * recombination_rate
    probability = np.zeros(len(positions))
    probability[1:] = (1 - np.exp(-2 * distance)) / 2
    return probability


def hmm_parameters(
    source_config: config.MatchSourceConfig, positions: np.ndarray
) -> HmmParameters:
    """Return the HMM parameters to use for haplotypes from a source."""
    num_sites = len(positions)
    if source_config.recombination_rate is not None:
        recombination = recombination_probabilities(
            source_config.recombination_rate, positions
        )
    else:
        value = DEFAULT_RECOMBINATION
        if source_config.recombination is not None:
            value = source_config.recombination
        recombination = np.full(num_sites, value)
    mismatch = DEFAULT_MISMATCH
    if source_config.mismatch is not None:
        mismatch = source_config.mismatch
    return HmmParameters(
        recombination=recombination,
        mismatch=np.full(num_sites, mismatch),
        likelihood_threshold=source_config.likelihood_threshold,
    )


def _rss_mib():
    """Return current process RSS in MiB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def make_root_ts(
    sequence_length: float,
    positions: np.ndarray,  # (num_sites,) int32
    sequence_intervals: np.ndarray,  # (n_intervals, 2) int32
    allele_mapper: vcz.AlleleMapper,
    max_time: float = 0.0,
    individuals: list[dict] | None = None,
    populations: list[dict] | None = None,
) -> tskit.TreeSequence:
    """
    Build the root tree sequence that starts the match loop.

    Creates two nodes — an ultimate root and a virtual root — connected by
    an edge spanning ``[positions[0], sequence_length)``.  This gives
    ``AncestorMatcher`` a valid tree to copy from when matching the first
    ancestor.

    The virtual root is placed at ``max_time + 1`` and the ultimate root at
    ``max_time + 2``, ensuring both are strictly older than any ancestor.

    Sites use real ancestral allele strings from *allele_mapper*.

    Stores *sequence_intervals* in the tree sequence top-level metadata.

    If *populations* is provided, sets a JSON metadata schema on the
    populations table and adds one row per dict.

    If *individuals* is provided, adds one row per dict to the individuals
    table (which already has a JSON metadata schema).
    """
    virtual_root_time = max_time + 1
    ultimate_root_time = max_time + 2

    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {"sequence_intervals": np.asarray(sequence_intervals).tolist()}

    positions = np.asarray(positions)
    ancestral = allele_mapper.ancestral_alleles()
    for i, pos in enumerate(positions):
        tables.sites.add_row(position=float(pos), ancestral_state=str(ancestral[i]))

    tables.nodes.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.individuals.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    _add_individuals_and_populations(tables, individuals, populations)

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


def _add_individuals_and_populations(
    tables: tskit.TableCollection,
    individuals: list[dict] | None,
    populations: list[dict] | None,
) -> None:
    """
    Append one population row per dict in *populations* and one individual
    row per dict in *individuals*. A JSON metadata schema is set on the
    populations table only when it is empty, so an existing schema is kept.
    """
    if populations is not None:
        if tables.populations.num_rows == 0:
            tables.populations.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        for pop_md in populations:
            tables.populations.add_row(metadata=pop_md)
    if individuals is not None:
        for ind_md in individuals:
            tables.individuals.add_row(metadata=ind_md)


def prepare_reference_ts(
    reference_ts: tskit.TreeSequence,
    individuals: list[dict] | None = None,
    populations: list[dict] | None = None,
) -> tskit.TreeSequence:
    """
    Return *reference_ts* with the given individuals and populations
    appended, ready to start the match loop. Match jobs must refer to the
    new rows by ids offset by the existing number of rows.
    """
    tables = reference_ts.dump_tables()
    _add_individuals_and_populations(tables, individuals, populations)
    return tables.tree_sequence()


def tsls_allele_codes(
    ts: tskit.TreeSequence, allele_mapper: vcz.AlleleMapper
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the (to_tsls, from_tsls) tables that translate between the allele
    codes of *allele_mapper* and those used by tsls for *ts*. Both are
    (num_sites, max_alleles) int8 arrays indexed by site and code.

    tsls numbers the alleles at each site itself: the ancestral state is 0,
    and derived states are numbered from 1 in order of their first
    appearance in the mutation table. Codes for alleles that are not in *ts*
    at a site translate to -1 (missing).
    """
    num_sites = ts.num_sites
    max_alleles = allele_mapper.forward_map().shape[1]
    mutation_code = allele_mapper.encode_mutations(
        ts.mutations_site, ts.mutations_derived_state
    ).astype(np.int64)
    if np.any(mutation_code < 0):
        raise ValueError("Tree sequence has a derived state not in the allele table")
    # Mutations back to the ancestral state have code 0 in both numberings.
    is_derived = mutation_code > 0
    derived_site = ts.mutations_site[is_derived].astype(np.int64)
    derived_code = mutation_code[is_derived]
    key = derived_site * max_alleles + derived_code
    unique_key, first_index = np.unique(key, return_index=True)
    allele_site = unique_key // max_alleles
    allele_code = unique_key % max_alleles
    # Order the distinct alleles by site, then by first appearance.
    order = np.lexsort((first_index, allele_site))
    allele_site = allele_site[order]
    allele_code = allele_code[order]
    site_start = np.searchsorted(allele_site, allele_site, side="left")
    tsls_code = np.arange(len(allele_site)) - site_start + 1

    to_tsls = np.full((num_sites, max_alleles), -1, dtype=np.int8)
    to_tsls[:, 0] = 0
    to_tsls[allele_site, allele_code] = tsls_code
    from_tsls = np.full((num_sites, max_alleles), -1, dtype=np.int8)
    from_tsls[:, 0] = 0
    from_tsls[allele_site, tsls_code] = allele_code
    return to_tsls, from_tsls


class TsinferEngine:
    """
    The built-in Li & Stephens HMM (``_tsinfer.AncestorMatcher``).
    """

    def __init__(self, ts: tskit.TreeSequence, allele_mapper: vcz.AlleleMapper):
        # Sites in ts correspond one-to-one with the allele mapper's sites,
        # and their ancestral states are allele code 0.
        num_alleles = allele_mapper.num_alleles_array()
        derived_state = allele_mapper.encode_mutations(
            ts.mutations_site, ts.mutations_derived_state
        )
        self._matcher_indexes = MatcherIndexes(
            ts,
            vestigial_root=False,
            num_alleles=num_alleles,
            derived_state=derived_state,
        )

    def find_path(self, h, start, end, params: HmmParameters):
        """
        Return the (left, right, parent, match, matcher) for haplotype *h*
        over sites [start, end). The matcher is returned for its statistics.
        """
        matcher = AncestorMatcher(
            self._matcher_indexes,
            params.recombination,
            params.mismatch,
            likelihood_threshold=params.likelihood_threshold,
        )
        match = np.zeros(len(h), dtype=np.int8)
        left, right, parent = matcher.find_path(h, start, end, match)
        return left, right, parent, match, matcher


class TslsEngine:
    """
    The Li & Stephens HMM from the tsls package.

    tsls can copy any node in the tree, but tsinfer never copies its
    ultimate root, node 0. Node 0's only child is node 1 and neither has
    mutations, so both have the ancestral state everywhere: we copy node 1
    wherever tsls copies node 0.
    """

    def __init__(self, ts: tskit.TreeSequence, allele_mapper: vcz.AlleleMapper):
        if tsls is None:
            raise ImportError(
                "The tsls engine requires the tsls package: "
                "install it with 'uv sync --extra tsls'"
            )
        node0_children = ts.edges_child[ts.edges_parent == 0]
        if (
            np.any(node0_children != 1)
            or np.any(ts.edges_child == 0)
            or np.any(ts.mutations_node <= 1)
        ):
            raise ValueError(
                "The tsls engine requires node 0 to be the ultimate root, with "
                "node 1 as its only child, and no mutations on either node"
            )
        self._ts = tsls.TreeSequence(ts)
        self._to_tsls, self._from_tsls = tsls_allele_codes(ts, allele_mapper)

    def find_path(self, h, start, end, params: HmmParameters):
        """
        Return the (left, right, parent, match, matcher) for haplotype *h*
        over sites [start, end). The matcher is returned for its statistics.
        """
        hmm = tsls.Hmm(
            self._ts,
            params.recombination,
            params.mismatch,
            likelihood_threshold=params.likelihood_threshold,
        )
        observed_sites = np.where(h >= 0)[0]
        h_tsls = np.full(len(h), -1, dtype=np.int8)
        h_tsls[observed_sites] = self._to_tsls[observed_sites, h[observed_sites]]
        match_tsls = np.zeros(len(h), dtype=np.int8)
        left, right, parent = hmm.find_path(h_tsls, start, end, match_tsls)

        # tsls only sets the matched alleles in [start, end).
        match = np.zeros(len(h), dtype=np.int8)
        sites = np.arange(start, end)
        match[start:end] = self._from_tsls[sites, match_tsls[start:end]]

        left, right, parent = copy_node1_for_node0(left, right, parent)
        return left, right, parent, match, hmm


def copy_node1_for_node0(left, right, parent):
    """
    Return the copying path (left, right, parent), in right-to-left order,
    with node 1 copied wherever node 0 is, merging adjacent segments that
    then have the same parent.
    """
    parent = np.where(parent == 0, 1, parent)
    is_run_start = np.ones(len(parent), dtype=bool)
    is_run_start[1:] = parent[1:] != parent[:-1]
    is_run_end = np.ones(len(parent), dtype=bool)
    is_run_end[:-1] = parent[:-1] != parent[1:]
    return left[is_run_end], right[is_run_start], parent[is_run_start]


ENGINES = {"tsinfer": TsinferEngine, "tsls": TslsEngine}


class Matcher:
    """
    Matches haplotypes against a fixed reference tree sequence using the
    specified HMM engine ("tsinfer" for the built-in engine, or "tsls").

    A fresh Matcher is instantiated for each group's tree sequence; no state
    is maintained between groups.
    """

    def __init__(
        self,
        ts: tskit.TreeSequence,
        positions: np.ndarray,  # (num_sites,) int32 — inference site positions
        allele_mapper: vcz.AlleleMapper,
        hmm_parameters: dict[str, HmmParameters],
        engine: str = "tsinfer",
    ):
        self._positions = np.asarray(positions, dtype=np.int32)
        self._num_sites = len(positions)
        self._sequence_length = ts.sequence_length
        self._hmm_parameters = hmm_parameters

        logger.info(
            "Creating Matcher (%s engine): ts=%d nodes, %d edges, %.1f MiB, "
            "RSS=%.1f MiB",
            engine,
            ts.num_nodes,
            ts.num_edges,
            ts.nbytes / (1024 * 1024),
            _rss_mib(),
        )
        t0 = time_.monotonic()
        self._engine = ENGINES[engine](ts, allele_mapper)
        logger.info(
            "Matcher ready in %.3fs, RSS=%.1f MiB",
            time_.monotonic() - t0,
            _rss_mib(),
        )

    def _match_one(self, job, reader) -> tuple:
        """Match a single haplotype: read, run HMM, convert to result."""
        num_sites = self._num_sites
        pos = self._positions.astype(np.float64)

        params = self._hmm_parameters[job.source]

        t0 = time_.monotonic()
        h = np.asarray(reader.read_haplotype(job), dtype=np.int8)
        t_read = time_.monotonic() - t0

        non_missing = np.where(h >= 0)[0]
        if len(non_missing) == 0:
            start, end = 0, num_sites
        else:
            start = int(non_missing[0])
            end = int(non_missing[-1]) + 1

        # Each call gets its own matcher so threads don't share state
        t1 = time_.monotonic()
        left, right, parent, match_out, matcher = self._engine.find_path(
            h, start, end, params
        )
        t_match = time_.monotonic() - t1

        # Build PathSegments from genomic-position edges
        path = []
        for li, ri, pi in zip(left, right, parent):
            path.append(PathSegment(left=float(li), right=float(ri), parent=int(pi)))

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
            "Matched job %s: read=%.4fs match=%.4fs "
            "edges=%d mutations=%d sites=%d..%d "
            "matcher_mem=%.1fMiB mean_tb=%.1f",
            getattr(job, "haplotype_index", "?"),
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
        with cf.ThreadPoolExecutor(max_workers=max(1, num_threads)) as executor:
            futures = {
                executor.submit(self._match_one, job, reader): job for job in jobs
            }
            for future in cf.as_completed(futures):
                yield future.result()
                del futures[future]


def extend_ts(
    ts: tskit.TreeSequence,
    *,
    paired_results: list[tuple[config.MatchJob, MatchResult]],
    allele_mapper: vcz.AlleleMapper,
    provenance_record: str | None = None,
) -> tskit.TreeSequence:
    """
    Extend ts with the match results for one group.

    Adds nodes, edges, and mutations directly via the tskit Tables API — no
    TSB roundtrip.  When ``job.individual_id`` is set, the node is assigned
    to the pre-existing individual row; when ``job.population_id`` is set,
    the node is assigned to the pre-existing population row.

    Mutation derived states are stored as real allele strings via
    *allele_mapper*.

    Returns the updated tree sequence, which becomes the reference panel for
    the next group.
    """
    t_start = time_.monotonic()

    # Sort by haplotype_index so nodes within each individual are ordered
    paired_results = sorted(paired_results, key=lambda p: p[0].haplotype_index)

    tables = ts.dump_tables()
    pos_arr = tables.sites.position.astype(np.float64)

    # Add nodes, edges, and collect mutation data for each (job, result) pair
    new_node_ids = []
    mut_data = []  # list of (site_id, code, node_id)
    for job, result in paired_results:
        node_meta = {
            "source": job.source,
            "sample_id": job.sample_id,
            "ploidy_index": job.ploidy_index,
        }
        node_id = tables.nodes.add_row(
            time=job.time,
            flags=job.node_flags,
            metadata=node_meta,
            individual=job.individual_id if job.individual_id is not None else -1,
            population=job.population_id if job.population_id is not None else -1,
        )
        new_node_ids.append(node_id)

        for seg in result.path:
            tables.edges.add_row(
                left=seg.left, right=seg.right, parent=seg.parent, child=node_id
            )

        for mut in result.mutations:
            site_id = int(np.searchsorted(pos_arr, mut.position))
            mut_data.append((site_id, mut.derived_state, node_id))

    # Bulk decode mutations and add to tables
    if mut_data:
        site_ids = np.array([m[0] for m in mut_data], dtype=np.int32)
        codes = np.array([m[1] for m in mut_data], dtype=np.int8)
        allele_strings = allele_mapper.decode_mutations(site_ids, codes)
        for (site_id, _, node_id), allele_str in zip(mut_data, allele_strings):
            tables.mutations.add_row(
                site=site_id,
                node=node_id,
                derived_state=allele_str,
                parent=-1,
            )

    t0 = time_.monotonic()
    tables.sort()
    t_sort = time_.monotonic() - t0

    t0 = time_.monotonic()
    tables.build_index()
    t_index = time_.monotonic() - t0

    t0 = time_.monotonic()
    tables.compute_mutation_parents()
    t_mut_parents = time_.monotonic() - t0

    if provenance_record is not None:
        tables.provenances.add_row(record=provenance_record)

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


def _encode_states(states, allele_index, name):
    unique_states, inverse = np.unique(np.asarray(states), return_inverse=True)
    code = np.zeros(len(unique_states), dtype=np.int8)
    for j, state in enumerate(unique_states):
        if state not in allele_index:
            raise ValueError(f"Unknown {name}: {state!r}")
        code[j] = allele_index[state]
    return code[inverse]


def encode_alleles(ts, alleles):
    """
    Return the (ancestral_state, derived_state) arrays for the specified tree
    sequence, encoding each state as its index in the specified list of
    alleles. Raise a ValueError if any state is not in the list.
    """
    alleles = list(alleles)
    if len(alleles) > np.iinfo(np.int8).max:
        raise ValueError("Too many alleles")
    allele_index = {allele: j for j, allele in enumerate(alleles)}
    if len(allele_index) != len(alleles):
        raise ValueError("Alleles must be unique")
    ancestral_state = _encode_states(
        ts.sites_ancestral_state, allele_index, "ancestral state"
    )
    derived_state = _encode_states(
        ts.mutations_derived_state, allele_index, "derived state"
    )
    return ancestral_state, derived_state


class MatcherIndexes(_tsinfer.MatcherIndexes):
    """
    Wrapper around the C MatcherIndexes, built from a tree sequence.

    By default, the ancestral state of every site is 0 and every mutation has
    derived state 1, unless the encoded ``derived_state`` of each mutation is
    given. If ``alleles`` is specified, states are instead encoded as their
    index in this list, and ``num_alleles`` defaults to its length.
    """

    def __init__(
        self,
        ts,
        *,
        vestigial_root=True,
        alleles=None,
        num_alleles=None,
        derived_state=None,
    ):
        if vestigial_root:
            ts = arg_ops.add_vestigial_root(ts)
        ancestral_state = None
        if alleles is not None:
            ancestral_state, derived_state = encode_alleles(ts, alleles)
            if num_alleles is None:
                num_alleles = np.full(ts.num_sites, len(alleles), dtype=np.uint64)
        tables = ts.dump_tables()
        ll_tables = _tsinfer.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        self.sequence_length = ts.sequence_length
        self.num_sites = ts.num_sites
        super().__init__(
            ll_tables,
            num_alleles=num_alleles,
            ancestral_state=ancestral_state,
            derived_state=derived_state,
        )


class AncestorMatcher(_tsinfer.AncestorMatcher):
    """Wrapper with the same ``find_path`` API as ``AncestorMatcher``."""

    def __init__(
        self,
        matcher_indexes,
        recombination,
        mismatch,
        likelihood_threshold=None,
        weight_by_n=None,
    ):
        kwargs = {}
        if likelihood_threshold is not None:
            kwargs["likelihood_threshold"] = likelihood_threshold
        if weight_by_n is not None:
            kwargs["weight_by_n"] = weight_by_n
        super().__init__(
            matcher_indexes,
            recombination=recombination,
            mismatch=mismatch,
            **kwargs,
        )

    def find_path(self, h, start, end, match_out):
        path_len, left, right, parent, matched = super().find_path(h, start, end)
        match_out[:] = matched
        return left[:path_len], right[:path_len], parent[:path_len]
