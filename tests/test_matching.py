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
Tests for tsinfer.matching: make_root_ts, Matcher, extend_ts.
"""

from __future__ import annotations

import numpy as np
import tskit

from tsinfer.grouping import MatchJob
from tsinfer.matching import (
    Matcher,
    MatchResult,
    Mutation,
    PathSegment,
    extend_ts,
    make_root_ts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_ts(positions, sequence_length, node_times, haplotypes):
    """
    Build a minimal tree sequence with one node per haplotype, connected to a
    virtual root (node 0, time=1.0) via edges spanning all sites, with
    mutations where the haplotype has value 1.

    positions: (num_sites,) int array
    node_times: (n_nodes,) float array (excluding root)
    haplotypes: (n_nodes, num_sites) int8 array
    """
    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {}
    tables.nodes.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    for pos in positions:
        tables.sites.add_row(position=float(pos), ancestral_state="0")
    # node 0 = virtual root
    tables.nodes.add_row(time=1.0, flags=0, metadata={})
    for time, hap in zip(node_times, haplotypes):
        node_id = tables.nodes.add_row(time=float(time), flags=0, metadata={})
        # single edge spanning full sequence
        tables.edges.add_row(
            left=float(positions[0]),
            right=float(sequence_length),
            parent=0,
            child=node_id,
        )
        # mutations where hap == 1
        for site_idx in np.where(np.asarray(hap) == 1)[0]:
            tables.mutations.add_row(site=int(site_idx), node=node_id, derived_state="1")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


class _ArrayReader:
    """Adapter that serves rows of a numpy array as read_haplotype(job)."""

    def __init__(self, haplotypes):
        self._haps = np.atleast_2d(np.asarray(haplotypes, dtype=np.int8))
        self._idx = 0

    def read_haplotype(self, job):
        h = self._haps[self._idx]
        self._idx += 1
        return h


def _jobs(n):
    """Return *n* opaque dummy jobs (the reader ignores them)."""
    return [None] * n


def _make_job(
    haplotype_index=0,
    source="test",
    sample_id="s0",
    ploidy_index=0,
    time=0.5,
    start_position=0,
    end_position=100,
    group=0,
    node_flags=1,
    individual_id=None,
    population_id=None,
):
    """Create a MatchJob with sensible defaults for testing."""
    return MatchJob(
        haplotype_index=haplotype_index,
        source=source,
        sample_id=sample_id,
        ploidy_index=ploidy_index,
        time=time,
        start_position=start_position,
        end_position=end_position,
        group=group,
        node_flags=node_flags,
        individual_id=individual_id,
        population_id=population_id,
    )


def _dummy_result():
    """A MatchResult with no edges or mutations."""
    return MatchResult(path=[], mutations=[])


# ---------------------------------------------------------------------------
# TestMakeRootTs
# ---------------------------------------------------------------------------


class TestMakeRootTs:
    def test_sequence_length(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.sequence_length == 1000.0

    def test_num_nodes_two(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.num_nodes == 2

    def test_sites_at_correct_positions(self):
        positions = np.array([5, 15, 25, 100], dtype=np.int32)
        intervals = np.array([[5, 101]], dtype=np.int32)
        ts = make_root_ts(200.0, positions, intervals)
        assert ts.num_sites == 4
        np.testing.assert_array_equal(ts.tables.sites.position, [5, 15, 25, 100])

    def test_sequence_intervals_in_metadata(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31], [100, 200]], dtype=np.int32)
        ts = make_root_ts(500.0, positions, intervals)
        meta = ts.metadata
        assert "sequence_intervals" in meta
        assert meta["sequence_intervals"] == [[10, 31], [100, 200]]

    def test_single_site(self):
        positions = np.array([42], dtype=np.int32)
        intervals = np.array([[42, 43]], dtype=np.int32)
        ts = make_root_ts(100.0, positions, intervals)
        assert ts.num_sites == 1
        assert ts.num_nodes == 2
        assert ts.num_edges == 1
        np.testing.assert_array_equal(ts.tables.sites.position, [42])

    def test_one_edge(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.num_edges == 1

    def test_no_mutations(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.num_mutations == 0

    def test_sequence_length_float(self):
        positions = np.array([1, 2, 3], dtype=np.int32)
        intervals = np.array([[1, 4]], dtype=np.int32)
        ts = make_root_ts(999.5, positions, intervals)
        assert ts.sequence_length == 999.5

    def test_node_metadata_schema_set(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = make_root_ts(1000.0, positions, intervals)
        assert ts.tables.nodes.metadata_schema.schema is not None
        for node in ts.nodes():
            assert node.metadata == {}


# ---------------------------------------------------------------------------
# TestMatcher
# ---------------------------------------------------------------------------


class TestMatcher:
    """Tests for Matcher using a simple known tree sequence."""

    def setup_method(self):
        self.positions = np.array([10, 20, 30], dtype=np.int32)
        self.seq_len = 100.0
        self.num_sites = 3

    def _make_ts_with_root_only(self):
        """Empty tree sequence (just sites, no nodes)."""
        return make_root_ts(
            self.seq_len,
            self.positions,
            np.array([[10, 31]], dtype=np.int32),
        )

    def _make_ts_with_one_ancestor(self, hap):
        """Tree sequence with virtual root + one ancestor haplotype."""
        return _make_simple_ts(
            self.positions,
            self.seq_len,
            node_times=[0.5],
            haplotypes=[hap],
        )

    def test_match_identical_haplotype(self):
        """Matching an identical haplotype returns a MatchResult with path edges."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        assert len(results) == 1
        _, r = results[0]
        # Should have path segments (copy from some ancestor)
        assert len(r.path) > 0

    def test_match_with_mutation(self):
        """Haplotype that differs from ancestor gets a mutation."""
        ancestor_hap = np.array([0, 0, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(ancestor_hap)
        # Query haplotype has derived state at site 1
        query_hap = np.array([0, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(1), _ArrayReader([query_hap])))
        _, r = results[0]
        # There should be a mutation at position 20 (site index 1)
        mut_positions = [m.position for m in r.mutations]
        assert 20.0 in mut_positions

    def test_match_all_missing(self):
        """All-missing haplotype should still return a MatchResult."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        missing_hap = np.array([-1, -1, -1], dtype=np.int8)
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(1), _ArrayReader([missing_hap])))
        assert len(results) == 1

    def test_match_partial_missing(self):
        """Haplotype with some missing sites: active range shrinks."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        query_hap = np.array([-1, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(1), _ArrayReader([query_hap])))
        assert len(results) == 1
        _, r = results[0]
        # Missing site (position 10) should not appear in mutations
        mut_positions = [m.position for m in r.mutations]
        assert 10.0 not in mut_positions

    def test_match_returns_one_result_per_haplotype(self):
        """match() returns one MatchResult per input haplotype."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        haplotypes = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(3), _ArrayReader(haplotypes)))
        assert len(results) == 3

    def test_match_result_types(self):
        """MatchResult should contain PathSegment and Mutation objects."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        _, r = results[0]
        assert isinstance(r.path, list)
        assert isinstance(r.mutations, list)
        for seg in r.path:
            assert isinstance(seg, PathSegment)
            assert isinstance(seg.left, float)
            assert isinstance(seg.right, float)
            assert isinstance(seg.parent, int)

    def test_path_segments_have_valid_coordinates(self):
        """PathSegment left < right for every segment."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        _, r = results[0]
        for seg in r.path:
            assert seg.left < seg.right

    def test_mutations_have_valid_fields(self):
        """Mutation objects should have position and derived_state."""
        hap = np.array([0, 0, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        query = np.array([1, 1, 0], dtype=np.int8)
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(1), _ArrayReader([query])))
        _, r = results[0]
        assert len(r.mutations) > 0
        for m in r.mutations:
            assert isinstance(m, Mutation)
            assert isinstance(m.position, float)
            assert isinstance(m.derived_state, int)


# ---------------------------------------------------------------------------
# TestExtendTs
# ---------------------------------------------------------------------------


class TestExtendTs:
    def setup_method(self):
        self.positions = np.array([10, 20, 30], dtype=np.int32)
        self.seq_len = 100.0
        self.intervals = np.array([[10, 31]], dtype=np.int32)

    def _root_ts(self, individuals=None, populations=None):
        return make_root_ts(
            self.seq_len,
            self.positions,
            self.intervals,
            individuals=individuals,
            populations=populations,
        )

    def _simple_match_result(self, parent_id, num_sites):
        """A MatchResult that copies from parent_id across all sites."""
        left_pos = float(self.positions[0])
        right_pos = self.seq_len
        return MatchResult(
            path=[PathSegment(left=left_pos, right=right_pos, parent=parent_id)],
            mutations=[],
        )

    def test_node_count_increases(self):
        ts = self._root_ts()
        assert ts.num_nodes == 2  # ultimate root + virtual root

        job = _make_job(time=0.5)
        ts2 = extend_ts(
            ts,
            paired_results=[(job, self._simple_match_result(1, len(self.positions)))],
        )
        assert ts2.num_nodes == 3

    def test_edges_present_after_extend(self):
        ts = self._root_ts()
        job = _make_job(time=0.5)
        ts2 = extend_ts(
            ts,
            paired_results=[(job, self._simple_match_result(1, len(self.positions)))],
        )
        assert ts2.num_edges > 0

    def test_mutations_present_after_extend(self):
        ts = self._root_ts()
        result_with_mut = MatchResult(
            path=[PathSegment(left=10.0, right=100.0, parent=1)],
            mutations=[Mutation(position=20.0, derived_state=1)],
        )
        job = _make_job(time=0.5)
        ts2 = extend_ts(
            ts,
            paired_results=[(job, result_with_mut)],
        )
        assert ts2.num_mutations > 0

    def test_individual_creation_diploid(self):
        individuals = [{"source": "test", "sample_id": "s0"}]
        ts = self._root_ts(individuals=individuals)
        r1 = self._simple_match_result(1, len(self.positions))
        r2 = self._simple_match_result(1, len(self.positions))
        j1 = _make_job(
            haplotype_index=0,
            time=0.0,
            ploidy_index=0,
            sample_id="s0",
            individual_id=0,
        )
        j2 = _make_job(
            haplotype_index=1,
            time=0.0,
            ploidy_index=1,
            sample_id="s0",
            individual_id=0,
        )
        ts2 = extend_ts(
            ts,
            paired_results=[(j1, r1), (j2, r2)],
        )
        assert ts2.num_individuals == 1
        ind = list(ts2.individuals())[0]
        assert len(ind.nodes) == 2

    def test_metadata_preserved(self):
        ts = self._root_ts()
        assert "sequence_intervals" in ts.metadata
        job = _make_job(time=0.5)
        ts2 = extend_ts(
            ts,
            paired_results=[(job, self._simple_match_result(1, len(self.positions)))],
        )
        meta = ts2.metadata
        assert "sequence_intervals" in meta

    def test_no_individual_when_individual_id_none(self):
        ts = self._root_ts()
        r1 = self._simple_match_result(1, len(self.positions))
        r2 = self._simple_match_result(1, len(self.positions))
        j1 = _make_job(haplotype_index=0, time=0.0, individual_id=None)
        j2 = _make_job(haplotype_index=1, time=0.0, individual_id=None)
        ts2 = extend_ts(
            ts,
            paired_results=[(j1, r1), (j2, r2)],
        )
        assert ts2.num_individuals == 0

    def test_sites_preserved(self):
        ts = self._root_ts()
        job = _make_job(time=0.5)
        ts2 = extend_ts(
            ts,
            paired_results=[(job, self._simple_match_result(1, len(self.positions)))],
        )
        assert ts2.num_sites == len(self.positions)
        np.testing.assert_array_equal(ts2.tables.sites.position, self.positions)

    def test_multiple_individuals(self):
        individuals = [
            {"source": "test", "sample_id": "s0"},
            {"source": "test", "sample_id": "s1"},
        ]
        ts = self._root_ts(individuals=individuals)
        results = [self._simple_match_result(1, len(self.positions)) for _ in range(4)]
        jobs = [
            _make_job(
                haplotype_index=i,
                time=0.0,
                ploidy_index=i % 2,
                sample_id="s0" if i < 2 else "s1",
                individual_id=0 if i < 2 else 1,
            )
            for i in range(4)
        ]
        ts2 = extend_ts(
            ts,
            paired_results=list(zip(jobs, results)),
        )
        assert ts2.num_individuals == 2

    def test_node_metadata_from_job(self):
        ts = self._root_ts()
        job = _make_job(
            time=0.5,
            source="my_source",
            sample_id="sample_42",
            ploidy_index=1,
        )
        result = self._simple_match_result(1, len(self.positions))
        ts2 = extend_ts(ts, paired_results=[(job, result)])
        # New node is the last one added (node 2)
        node_meta = ts2.node(2).metadata
        assert node_meta["source"] == "my_source"
        assert node_meta["sample_id"] == "sample_42"
        assert node_meta["ploidy_index"] == 1

    def test_mixed_ploidy_individuals(self):
        """2 diploid nodes + 1 haploid node → 2 individuals."""
        individuals = [
            {"source": "test", "sample_id": "diploid"},
            {"source": "test", "sample_id": "haploid"},
        ]
        ts = self._root_ts(individuals=individuals)
        results = [self._simple_match_result(1, len(self.positions)) for _ in range(3)]
        jobs = [
            _make_job(
                haplotype_index=0,
                time=0.0,
                ploidy_index=0,
                sample_id="diploid",
                individual_id=0,
            ),
            _make_job(
                haplotype_index=1,
                time=0.0,
                ploidy_index=1,
                sample_id="diploid",
                individual_id=0,
            ),
            _make_job(
                haplotype_index=2,
                time=0.0,
                ploidy_index=0,
                sample_id="haploid",
                individual_id=1,
            ),
        ]
        ts2 = extend_ts(ts, paired_results=list(zip(jobs, results)))
        assert ts2.num_individuals == 2
        node_counts = sorted(len(ind.nodes) for ind in ts2.individuals())
        assert node_counts == [1, 2]

    def test_individuals_across_sources(self):
        """Same sample_id in different sources → 2 separate individuals."""
        individuals = [
            {"source": "src_a", "sample_id": "s0"},
            {"source": "src_b", "sample_id": "s0"},
        ]
        ts = self._root_ts(individuals=individuals)
        results = [self._simple_match_result(1, len(self.positions)) for _ in range(2)]
        jobs = [
            _make_job(
                haplotype_index=0,
                time=0.0,
                source="src_a",
                sample_id="s0",
                individual_id=0,
            ),
            _make_job(
                haplotype_index=1,
                time=0.0,
                source="src_b",
                sample_id="s0",
                individual_id=1,
            ),
        ]
        ts2 = extend_ts(ts, paired_results=list(zip(jobs, results)))
        assert ts2.num_individuals == 2

    def test_individual_metadata(self):
        """Each individual carries source and sample_id in its metadata."""
        individuals = [{"source": "my_src", "sample_id": "sam1"}]
        ts = self._root_ts(individuals=individuals)
        r = self._simple_match_result(1, len(self.positions))
        job = _make_job(
            haplotype_index=0,
            time=0.0,
            source="my_src",
            sample_id="sam1",
            individual_id=0,
        )
        ts2 = extend_ts(ts, paired_results=[(job, r)])
        assert ts2.num_individuals == 1
        ind_meta = ts2.individual(0).metadata
        assert ind_meta["source"] == "my_src"
        assert ind_meta["sample_id"] == "sam1"

    def test_node_order_by_haplotype_index(self):
        """Nodes within an individual are ordered by haplotype_index."""
        individuals = [{"source": "test", "sample_id": "s0"}]
        ts = self._root_ts(individuals=individuals)
        results = [self._simple_match_result(1, len(self.positions)) for _ in range(2)]
        # Pass jobs in reverse haplotype_index order
        jobs = [
            _make_job(
                haplotype_index=5,
                time=0.0,
                ploidy_index=1,
                sample_id="s0",
                individual_id=0,
            ),
            _make_job(
                haplotype_index=3,
                time=0.0,
                ploidy_index=0,
                sample_id="s0",
                individual_id=0,
            ),
        ]
        ts2 = extend_ts(ts, paired_results=list(zip(jobs, results)))
        assert ts2.num_individuals == 1
        ind = ts2.individual(0)
        # Node added from haplotype_index=3 should come before haplotype_index=5
        assert len(ind.nodes) == 2
        # The node with ploidy_index=0 (haplotype_index=3) should be first
        n0_meta = ts2.node(ind.nodes[0]).metadata
        n1_meta = ts2.node(ind.nodes[1]).metadata
        assert n0_meta["ploidy_index"] == 0
        assert n1_meta["ploidy_index"] == 1


# ---------------------------------------------------------------------------
# TestMatcherExtendCycle
# ---------------------------------------------------------------------------


class TestMatcherExtendCycle:
    """Integration tests: make_root_ts → extend_ts (root) → Matcher → extend_ts."""

    def setup_method(self):
        self.positions = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        self.seq_len = 100.0
        self.intervals = np.array([[10, 51]], dtype=np.int32)
        self.num_sites = 5

    def _build_root_ts(self):
        """Create a root TS with ultimate root + virtual root (2 nodes)."""
        return make_root_ts(self.seq_len, self.positions, self.intervals)

    def _match_and_pair(self, ts, haplotypes):
        """Match haplotypes against ts and return paired_results list."""
        matcher = Matcher(ts, self.positions)
        results = list(matcher.match(_jobs(len(haplotypes)), _ArrayReader(haplotypes)))
        return results

    def test_node_time_correct_after_cycle(self):
        """Node times in the output TS should match the input times."""
        ts = self._build_root_ts()
        ancestor_hap = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        paired = self._match_and_pair(ts, [ancestor_hap])
        job = _make_job(time=0.5, individual_id=None)
        paired_results = [(job, paired[0][1])]
        ts2 = extend_ts(ts, paired_results=paired_results)
        node_times = [n.time for n in ts2.nodes()]
        assert 1.0 in node_times
        assert 0.5 in node_times

    def test_matching_after_two_extends(self):
        """After adding ancestor, matching a similar haplotype gives a path."""
        ts = self._build_root_ts()
        ancestor_hap = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        paired = self._match_and_pair(ts, [ancestor_hap])
        job = _make_job(time=0.5, individual_id=None)
        ts2 = extend_ts(ts, paired_results=[(job, paired[0][1])])

        # Now match a sample against ts2
        sample_hap = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        matcher2 = Matcher(ts2, self.positions)
        results2 = list(matcher2.match(_jobs(1), _ArrayReader([sample_hap])))
        _, r = results2[0]
        assert len(r.path) > 0

    def test_num_nodes_accumulates(self):
        """Each extend_ts call should add the expected number of nodes."""
        ts = self._build_root_ts()
        assert ts.num_nodes == 2

        hap1 = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        paired1 = self._match_and_pair(ts, hap1)
        job1 = _make_job(time=0.7, haplotype_index=0, individual_id=None)
        ts2 = extend_ts(ts, paired_results=[(job1, paired1[0][1])])
        assert ts2.num_nodes == 3

        hap2 = np.array([[1, 0, 0, 1, 0]], dtype=np.int8)
        paired2 = self._match_and_pair(ts2, hap2)
        job2 = _make_job(time=0.5, haplotype_index=1, individual_id=None)
        ts3 = extend_ts(ts2, paired_results=[(job2, paired2[0][1])])
        assert ts3.num_nodes == 4

        sample_hap = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        paired3 = self._match_and_pair(ts3, sample_hap)
        job3 = _make_job(time=0.0, haplotype_index=2, individual_id=None)
        ts4 = extend_ts(ts3, paired_results=[(job3, paired3[0][1])])
        assert ts4.num_nodes == 5

    def test_metadata_survives_multiple_cycles(self):
        """sequence_intervals metadata should survive multiple extend_ts calls."""
        ts = self._build_root_ts()
        hap = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        paired = self._match_and_pair(ts, hap)
        job = _make_job(time=0.5, individual_id=None)
        ts2 = extend_ts(ts, paired_results=[(job, paired[0][1])])
        assert "sequence_intervals" in ts2.metadata
        assert ts2.metadata["sequence_intervals"] == [[10, 51]]
