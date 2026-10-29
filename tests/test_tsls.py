#
# Copyright (C) 2026 University of Oxford
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
Tests for the tsls match engine.
"""

from __future__ import annotations

import helpers
import numpy as np
import pytest

from tsinfer import config, matching, vcz

POSITIONS = np.array([10, 20, 30, 40], dtype=np.int32)
SEQUENCE_LENGTH = 100
# Site 1 is triallelic and site 3 has four alleles.
ALLELE_TABLE = [["A", "T"], ["A", "C", "G"], ["A", "T"], ["A", "C", "G", "T"]]

# The reference panel is matched in two groups. In the first, "G" at site 1
# appears before "C", so tsls numbers the alleles at site 1 differently
# from the allele mapper.
OLD_ANCESTORS = [[1, 0, 0, 0], [0, 2, 0, 0]]
YOUNG_ANCESTORS = [[1, 1, 1, 0], [0, 2, 0, 3]]


class _ArrayReader:
    """Serves the haplotype for each job by its haplotype_index."""

    def __init__(self, haplotypes):
        self._haplotypes = np.asarray(haplotypes, dtype=np.int8)

    def read_haplotype(self, job):
        return self._haplotypes[job.haplotype_index]


def _job(haplotype_index, time):
    return config.MatchJob(
        haplotype_index=haplotype_index,
        source="test",
        sample_id=f"s{haplotype_index}",
        ploidy_index=0,
        time=time,
        start_position=0,
        end_position=SEQUENCE_LENGTH,
        group=0,
        node_flags=0,
    )


def _hmm_parameters():
    params = matching.hmm_parameters(config.MatchSourceConfig(), POSITIONS)
    return {"test": params}


def _match(ts, allele_mapper, haplotypes, engine, time=0.0, num_threads=1):
    """Return the MatchResults for the haplotypes, in order."""
    matcher = matching.Matcher(
        ts,
        POSITIONS,
        allele_mapper=allele_mapper,
        hmm_parameters=_hmm_parameters(),
        engine=engine,
    )
    jobs = [_job(j, time) for j in range(len(haplotypes))]
    results = matcher.match(jobs, _ArrayReader(haplotypes), num_threads=num_threads)
    paired = sorted(results, key=lambda pair: pair[0].haplotype_index)
    return paired


def _extend(ts, allele_mapper, haplotypes, time):
    """Match the haplotypes with the built-in engine and add them to ts."""
    paired = _match(ts, allele_mapper, haplotypes, "tsinfer", time=time)
    return matching.extend_ts(ts, paired_results=paired, allele_mapper=allele_mapper)


@pytest.fixture
def allele_mapper():
    return vcz.AlleleMapper(len(POSITIONS), ALLELE_TABLE)


@pytest.fixture
def root_ts(allele_mapper):
    return matching.make_root_ts(
        SEQUENCE_LENGTH,
        POSITIONS,
        np.array([[10, 41]], dtype=np.int32),
        allele_mapper=allele_mapper,
        max_time=3,
    )


@pytest.fixture
def panel_ts(root_ts, allele_mapper):
    """Nodes 0 and 1 are the roots, 2-3 the old and 4-5 the young ancestors."""
    ts = _extend(root_ts, allele_mapper, OLD_ANCESTORS, time=2)
    return _extend(ts, allele_mapper, YOUNG_ANCESTORS, time=1)


class TestTslsAlleleCodes:
    def check_against_tskit(self, ts, allele_mapper):
        """tsls numbers the alleles at each site as tskit's variants do."""
        to_tsls, from_tsls = matching.tsls_allele_codes(ts, allele_mapper)
        max_alleles = allele_mapper.forward_map().shape[1]
        assert to_tsls.shape == (ts.num_sites, max_alleles)
        assert from_tsls.shape == (ts.num_sites, max_alleles)
        for variant in ts.variants():
            site = variant.site.id
            expected_to = np.full(max_alleles, -1)
            expected_from = np.full(max_alleles, -1)
            for tsls_code, allele in enumerate(variant.alleles):
                code = allele_mapper.lookup(site, allele)
                expected_to[code] = tsls_code
                expected_from[tsls_code] = code
            np.testing.assert_array_equal(to_tsls[site], expected_to)
            np.testing.assert_array_equal(from_tsls[site], expected_from)
        return to_tsls, from_tsls

    def test_root_ts(self, root_ts, allele_mapper):
        to_tsls, from_tsls = self.check_against_tskit(root_ts, allele_mapper)
        assert np.all(to_tsls[:, 0] == 0)
        assert np.all(to_tsls[:, 1:] == -1)

    def test_panel(self, panel_ts, allele_mapper):
        to_tsls, from_tsls = self.check_against_tskit(panel_ts, allele_mapper)
        # "G" (code 2) appears before "C" (code 1) at site 1.
        np.testing.assert_array_equal(to_tsls[1], [0, 2, 1, -1])
        np.testing.assert_array_equal(from_tsls[1], [0, 2, 1, -1])
        # Only "T" (code 3) has been seen at site 3.
        np.testing.assert_array_equal(to_tsls[3], [0, -1, -1, 1])

    def test_mutation_to_ancestral_state(self, panel_ts, allele_mapper):
        tables = panel_ts.dump_tables()
        tables.mutations.add_row(site=2, node=2, derived_state="A")
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        to_tsls, _ = self.check_against_tskit(ts, allele_mapper)
        np.testing.assert_array_equal(to_tsls[2], [0, 1, -1, -1])

    def test_unknown_derived_state(self, panel_ts, allele_mapper):
        tables = panel_ts.dump_tables()
        tables.mutations.add_row(site=2, node=2, derived_state="X")
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        with pytest.raises(ValueError, match="not in the allele table"):
            matching.tsls_allele_codes(tables.tree_sequence(), allele_mapper)


class TestCopyNode1ForNode0:
    def check(self, left, right, parent, expected):
        result = matching.copy_node1_for_node0(
            np.array(left, dtype=np.float64),
            np.array(right, dtype=np.float64),
            np.array(parent, dtype=np.int32),
        )
        for array, expected_array in zip(result, expected):
            np.testing.assert_array_equal(array, expected_array)

    def test_no_node0(self):
        self.check([30, 0], [100, 30], [4, 5], ([30, 0], [100, 30], [4, 5]))

    def test_single_segment(self):
        self.check([0], [100], [0], ([0], [100], [1]))

    def test_merges_node0_next_to_node1(self):
        # Right-to-left order: 5 over [60, 100), 0 over [30, 60),
        # 1 over [10, 30) and 3 over [0, 10).
        self.check(
            [60, 30, 10, 0],
            [100, 60, 30, 10],
            [5, 0, 1, 3],
            ([60, 10, 0], [100, 60, 10], [5, 1, 3]),
        )

    def test_merges_runs_at_both_ends(self):
        self.check(
            [50, 20, 0],
            [100, 50, 20],
            [1, 0, 1],
            ([0], [100], [1]),
        )


class TestTslsEngineErrors:
    def test_tsls_not_installed(self, root_ts, allele_mapper, monkeypatch):
        monkeypatch.setattr(matching, "tsls", None)
        with pytest.raises(ImportError, match="uv sync --extra tsls"):
            matching.TslsEngine(root_ts, allele_mapper)

    @helpers.requires_tsls
    def test_node0_with_other_child(self, root_ts, allele_mapper):
        tables = root_ts.dump_tables()
        child = tables.nodes.add_row(time=1)
        tables.edges.add_row(10, SEQUENCE_LENGTH, 0, child)
        tables.sort()
        with pytest.raises(ValueError, match="node 0 to be the ultimate root"):
            matching.TslsEngine(tables.tree_sequence(), allele_mapper)

    @helpers.requires_tsls
    def test_node0_is_a_child(self, root_ts, allele_mapper):
        tables = root_ts.dump_tables()
        parent = tables.nodes.add_row(time=10)
        tables.edges.add_row(10, SEQUENCE_LENGTH, parent, 0)
        tables.sort()
        with pytest.raises(ValueError, match="node 0 to be the ultimate root"):
            matching.TslsEngine(tables.tree_sequence(), allele_mapper)

    @helpers.requires_tsls
    def test_mutation_on_node1(self, root_ts, allele_mapper):
        tables = root_ts.dump_tables()
        tables.mutations.add_row(site=0, node=1, derived_state="T")
        with pytest.raises(ValueError, match="no mutations on either node"):
            matching.TslsEngine(tables.tree_sequence(), allele_mapper)


@helpers.requires_tsls
class TestTslsEngine:
    """
    Paths match the built-in engine's for haplotypes with a single best path.
    """

    def check_same_as_tsinfer(self, ts, allele_mapper, haplotypes):
        expected = _match(ts, allele_mapper, haplotypes, "tsinfer")
        result = _match(ts, allele_mapper, haplotypes, "tsls")
        for (_, expected_result), (_, tsls_result) in zip(expected, result):
            assert tsls_result == expected_result
        return [match_result for _, match_result in result]

    def test_root_ts(self, root_ts, allele_mapper):
        # tsls would copy node 0 here, as it has the same likelihood as node 1.
        results = self.check_same_as_tsinfer(
            root_ts, allele_mapper, [[1, 2, 0, 3], [0, 0, 0, 0]]
        )
        assert results[0].path == [matching.PathSegment(0, SEQUENCE_LENGTH, 1)]
        assert len(results[0].mutations) == 3
        assert results[1].mutations == []

    def test_copy_ancestor(self, panel_ts, allele_mapper):
        results = self.check_same_as_tsinfer(panel_ts, allele_mapper, YOUNG_ANCESTORS)
        for result in results:
            assert len(result.path) == 1
            assert result.mutations == []

    def test_recombination(self, panel_ts, allele_mapper):
        results = self.check_same_as_tsinfer(panel_ts, allele_mapper, [[1, 1, 1, 3]])
        assert [segment.parent for segment in results[0].path] == [5, 4]
        assert results[0].mutations == []

    def test_allele_not_in_panel(self, panel_ts, allele_mapper):
        # "C" (code 1) is not in the panel at site 3: every node mismatches.
        results = self.check_same_as_tsinfer(panel_ts, allele_mapper, [[0, 0, 0, 1]])
        assert results[0].mutations == [matching.Mutation(40, 1)]

    def test_mismatch(self, panel_ts, allele_mapper):
        # With no recombination, node 4 is the closest, mismatching at site 3.
        haplotypes = [[1, 1, 1, 3]]
        source_config = config.MatchSourceConfig(recombination=1e-30, mismatch=1e-3)
        params = {"test": matching.hmm_parameters(source_config, POSITIONS)}
        results = []
        for engine in matching.ENGINES:
            matcher = matching.Matcher(
                panel_ts,
                POSITIONS,
                allele_mapper=allele_mapper,
                hmm_parameters=params,
                engine=engine,
            )
            jobs = [_job(0, 0.0)]
            results.append(list(matcher.match(jobs, _ArrayReader(haplotypes))))
        assert results[0][0][1] == results[1][0][1]
        assert results[1][0][1].mutations == [matching.Mutation(40, 3)]

    def test_partially_missing(self, panel_ts, allele_mapper):
        results = self.check_same_as_tsinfer(
            panel_ts, allele_mapper, [[-1, 1, 1, -1], [-1, 2, -1, 3]]
        )
        # The path ends at the end of the sequence after the last site.
        assert results[0].path == [matching.PathSegment(20, 40, 4)]
        assert results[1].path == [matching.PathSegment(20, SEQUENCE_LENGTH, 5)]

    def test_all_missing(self, panel_ts, allele_mapper):
        # Every node is equally likely, so the engines can choose different
        # nodes.
        paired = _match(panel_ts, allele_mapper, [[-1, -1, -1, -1]], "tsls")
        result = paired[0][1]
        assert len(result.path) == 1
        assert result.path[0].left == 0
        assert result.path[0].right == SEQUENCE_LENGTH
        assert result.path[0].parent != 0
        assert result.mutations == []

    def test_threads(self, panel_ts, allele_mapper):
        rng = np.random.default_rng(42)
        num_alleles = np.array([len(alleles) for alleles in ALLELE_TABLE])
        haplotypes = rng.integers(0, num_alleles, size=(50, len(POSITIONS)))
        single = _match(panel_ts, allele_mapper, haplotypes, "tsls")
        threaded = _match(panel_ts, allele_mapper, haplotypes, "tsls", num_threads=4)
        assert single == threaded
        for _, result in single:
            parents = [segment.parent for segment in result.path]
            assert 0 not in parents
