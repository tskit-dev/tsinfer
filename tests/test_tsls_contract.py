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
The properties of the tsls HMM that tsinfer relies on.

TestTslsContract calls tsls directly, so that a change in tsls that would
break tsinfer fails here with a clear message. TestMatcher pins the results
of matching through tsinfer's Matcher on a small panel; these were the
same with tsinfer's previous built-in engine.
"""

from __future__ import annotations

import concurrent.futures as cf

import numpy as np
import pytest
import tskit
import tsls

from tsinfer import config, matching, vcz

POSITIONS = np.array([10, 20, 30, 40], dtype=np.int32)
SEQUENCE_LENGTH = 100
# Site 1 is triallelic and site 3 has four alleles.
ALLELE_TABLE = [["A", "T"], ["A", "C", "G"], ["A", "T"], ["A", "C", "G", "T"]]

# The reference panel is matched in two groups. In the first, "G" at site 1
# appears before "C", so tskit numbers the alleles at site 1 differently
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


def _hmm_parameters(source_config=None):
    if source_config is None:
        source_config = config.MatchSourceConfig()
    return {"test": matching.hmm_parameters(source_config, POSITIONS)}


def _match(ts, allele_mapper, haplotypes, time=0.0, num_threads=1, params=None):
    """Return the (job, MatchResult) pairs for the haplotypes, in order."""
    if params is None:
        params = _hmm_parameters()
    matcher = matching.Matcher(
        ts,
        POSITIONS,
        allele_mapper=allele_mapper,
        hmm_parameters=params,
    )
    jobs = [_job(j, time) for j in range(len(haplotypes))]
    results = matcher.match(jobs, _ArrayReader(haplotypes), num_threads=num_threads)
    return sorted(results, key=lambda pair: pair[0].haplotype_index)


def _extend(ts, allele_mapper, haplotypes, time):
    """Match the haplotypes and add them to ts."""
    paired = _match(ts, allele_mapper, haplotypes, time=time)
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


def node_haplotypes(ts, allele_mapper):
    """
    Return the (num_nodes, num_sites) haplotypes of every node in the allele
    mapper's codes.
    """
    H = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.int8)
    for variant in ts.variants(
        samples=np.arange(ts.num_nodes), isolated_as_missing=False
    ):
        site = variant.site.id
        codes = [allele_mapper.lookup(site, allele) for allele in variant.alleles]
        H[:, site] = np.array(codes)[variant.genotypes]
    return H


def tsls_ts(ts, allele_mapper):
    return tsls.TreeSequence(ts, alleles=allele_mapper.forward_map())


def find_path(
    ts, allele_mapper, h, start=0, end=None, recombination=1e-2, mismatch=1e-20
):
    """Run tsls directly, as Matcher does, returning (left, right, parent, match)."""
    if end is None:
        end = ts.num_sites
    m = ts.num_sites
    hmm = tsls.Hmm(
        tsls_ts(ts, allele_mapper),
        np.full(m, recombination),
        np.full(m, mismatch),
    )
    match = np.zeros(m, dtype=np.int8)
    h = np.asarray(h, dtype=np.int8)
    left, right, parent = hmm.find_path(h, start, end, match)
    return left, right, parent, match


class TestTslsContract:
    def test_forward_map_alleles(self, panel_ts, allele_mapper):
        # The allele mapper's padded forward map is accepted as the per-site
        # alleles, and the number of alleles at each site is the VCZ count,
        # which sets the mismatch odds.
        forward_map = allele_mapper.forward_map()
        assert np.any(forward_map == "")
        tts = tsls_ts(panel_ts, allele_mapper)
        np.testing.assert_array_equal(tts.num_alleles, allele_mapper.num_alleles_array())

    def test_codes_pass_through(self, panel_ts, allele_mapper):
        # Copying a node exactly gives its haplotype in the allele mapper's
        # codes, with no translation.
        H = node_haplotypes(panel_ts, allele_mapper)
        for u in range(2, panel_ts.num_nodes):
            _, _, parent, match = find_path(panel_ts, allele_mapper, H[u])
            np.testing.assert_array_equal(match, H[u])
            for v in parent:
                np.testing.assert_array_equal(H[v], H[u])

    def test_allele_not_in_tree_sequence_is_a_mismatch(self, panel_ts, allele_mapper):
        # "C" (code 1) at site 3 is in the allele table but not in the panel:
        # it's a mismatch against every node, not an error, as for the focal
        # sites of an ancestor.
        h = np.array([0, 0, 0, 1], dtype=np.int8)
        _, _, _, match = find_path(panel_ts, allele_mapper, h)
        assert match[3] != 1
        assert np.array_equal(match[:3], h[:3])

    def test_path_format(self, panel_ts, allele_mapper):
        # The path is in right-to-left order, contiguous, and covers
        # [position[start], position[end]), with 0 for the first site and the
        # sequence length for the last. Breakpoints are site positions.
        H = node_haplotypes(panel_ts, allele_mapper)
        h = H[4].copy()
        h[3] = H[5][3]
        for start, end in [(0, 4), (1, 4), (0, 3), (1, 3)]:
            left, right, parent, match = find_path(
                panel_ts, allele_mapper, h, start, end
            )
            assert np.all(np.diff(left) < 0)
            np.testing.assert_array_equal(left[:-1], right[1:])
            expected_left = 0 if start == 0 else POSITIONS[start]
            expected_right = SEQUENCE_LENGTH if end == 4 else POSITIONS[end]
            assert left[-1] == expected_left
            assert right[0] == expected_right
            assert np.all(np.isin(left[:-1], POSITIONS))
            assert np.all(match[:start] == -1)
            assert np.all(match[end:] == -1)
            assert np.all(match[start:end] >= 0)

    def test_missing_data(self, panel_ts, allele_mapper):
        # Missing sites are ignored, and the match there is the copied
        # node's allele.
        H = node_haplotypes(panel_ts, allele_mapper)
        h = H[5].copy()
        h[1] = -1
        _, _, parent, match = find_path(panel_ts, allele_mapper, h)
        assert list(parent) == [5]
        np.testing.assert_array_equal(match, H[5])

    def test_node0_and_node1_tie(self, root_ts, panel_ts, allele_mapper):
        # tsinfer never copies node 0, its ultimate root. Node 0 has the same
        # Viterbi value as node 1 at every site, so copying node 1 where
        # tsls copies node 0 gives a path with the same likelihood.
        for ts in [root_ts, panel_ts]:
            m = ts.num_sites
            hmm = tsls.Hmm(
                tsls_ts(ts, allele_mapper), np.full(m, 1e-2), np.full(m, 1e-3)
            )
            h = np.array([1, 2, 0, 3], dtype=np.int8)
            _, _, matrix = hmm.viterbi_matrix(h)
            V = matrix.decode([0, 1])
            np.testing.assert_array_equal(V[:, 0], V[:, 1])
        _, _, parent, _ = find_path(root_ts, allele_mapper, [0, 0, 0, 0])
        assert set(parent) <= {0, 1}

    def test_root_ts_can_only_copy_roots(self, root_ts, allele_mapper):
        # The first group is matched against the root-only tree sequence,
        # where nodes 0 and 1 are the only states.
        _, _, parent, match = find_path(root_ts, allele_mapper, [1, 2, 0, 3])
        assert len(parent) == 1
        assert parent[0] in (0, 1)
        assert list(match) == [0, 0, 0, 0]

    def test_recombination_preferred_by_default(self, panel_ts, allele_mapper):
        # With tsinfer's default parameters (recombination 1e-2, mismatch
        # 1e-20), a haplotype is copied with switches rather than mismatches.
        _, _, parent, match = find_path(panel_ts, allele_mapper, [1, 1, 1, 3])
        assert list(parent) == [5, 4]
        assert list(match) == [1, 1, 1, 3]

    def test_mismatch_preferred_with_low_recombination(self, panel_ts, allele_mapper):
        _, _, parent, match = find_path(
            panel_ts, allele_mapper, [1, 1, 1, 3], recombination=1e-30, mismatch=1e-3
        )
        assert list(parent) == [4]
        assert list(match) == [1, 1, 1, 0]

    def test_tiny_mismatch_probability(self, panel_ts, allele_mapper):
        # With tsinfer's default mismatch of 1e-20, a haplotype that
        # mismatches every node at every site still has a path.
        h = np.array([1, 1, 1, 1], dtype=np.int8)
        _, _, _, match = find_path(panel_ts, allele_mapper, h)
        assert match[3] != 1

    def test_per_site_rates(self, panel_ts, allele_mapper):
        # tsinfer's recombination probabilities are per site, and 0 at the
        # first site.
        m = panel_ts.num_sites
        recombination = matching.recombination_probabilities(1e-2, POSITIONS)
        assert recombination[0] == 0
        hmm = tsls.Hmm(
            tsls_ts(panel_ts, allele_mapper),
            recombination,
            np.full(m, 1e-20),
            likelihood_threshold=None,
        )
        match = np.zeros(m, dtype=np.int8)
        _, _, parent = hmm.find_path(np.array([1, 1, 1, 3], dtype=np.int8), 0, m, match)
        assert list(parent) == [5, 4]

    def test_shared_tree_sequence_threads(self, panel_ts, allele_mapper):
        # Matcher builds one tsls.TreeSequence per group and shares it between
        # threads, each with its own Hmm.
        tts = tsls_ts(panel_ts, allele_mapper)
        m = panel_ts.num_sites
        rng = np.random.default_rng(42)
        num_alleles = allele_mapper.num_alleles_array().astype(int)
        H = rng.integers(0, num_alleles, size=(50, m)).astype(np.int8)

        def run(h):
            hmm = tsls.Hmm(tts, np.full(m, 1e-2), np.full(m, 1e-3))
            match = np.zeros(m, dtype=np.int8)
            path = hmm.find_path(h, 0, m, match)
            return [array.tolist() for array in path] + [match.tolist()]

        serial = [run(h) for h in H]
        with cf.ThreadPoolExecutor(max_workers=4) as executor:
            threaded = list(executor.map(run, H))
        assert serial == threaded
        # And repeated runs give the same result.
        assert serial == [run(h) for h in H]

    def test_hmm_statistics(self, panel_ts, allele_mapper):
        # Matcher logs these for each haplotype.
        m = panel_ts.num_sites
        hmm = tsls.Hmm(
            tsls_ts(panel_ts, allele_mapper), np.full(m, 1e-2), np.full(m, 1e-20)
        )
        hmm.find_path(np.zeros(m, dtype=np.int8), 0, m, np.zeros(m, dtype=np.int8))
        assert hmm.total_memory > 0
        assert hmm.mean_traceback_size >= 0

    def test_unknown_state_is_value_error(self, panel_ts, allele_mapper):
        tables = panel_ts.dump_tables()
        tables.mutations.add_row(site=2, node=2, derived_state="X")
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        with pytest.raises(ValueError, match="not in its site's list of alleles"):
            tsls_ts(tables.tree_sequence(), allele_mapper)

    def test_bad_haplotype_code_is_value_error(self, panel_ts, allele_mapper):
        # Site 0 has two alleles
        with pytest.raises(ValueError, match="Haplotype allele"):
            find_path(panel_ts, allele_mapper, [2, 0, 0, 0])


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


class TestUltimateRootCheck:
    def test_node0_with_other_child(self, root_ts):
        tables = root_ts.dump_tables()
        child = tables.nodes.add_row(time=1)
        tables.edges.add_row(10, SEQUENCE_LENGTH, 0, child)
        tables.sort()
        with pytest.raises(ValueError, match="node 0 to be the ultimate root"):
            matching.check_ultimate_root(tables.tree_sequence())

    def test_node0_is_a_child(self, root_ts):
        tables = root_ts.dump_tables()
        parent = tables.nodes.add_row(time=10)
        tables.edges.add_row(10, SEQUENCE_LENGTH, parent, 0)
        tables.sort()
        with pytest.raises(ValueError, match="node 0 to be the ultimate root"):
            matching.check_ultimate_root(tables.tree_sequence())

    def test_mutation_on_node1(self, root_ts):
        tables = root_ts.dump_tables()
        tables.mutations.add_row(site=0, node=1, derived_state="T")
        with pytest.raises(ValueError, match="no mutations on either node"):
            matching.check_ultimate_root(tables.tree_sequence())

    def test_matcher_checks(self, root_ts, allele_mapper):
        tables = root_ts.dump_tables()
        tables.mutations.add_row(site=0, node=1, derived_state="T")
        with pytest.raises(ValueError, match="no mutations on either node"):
            _match(tables.tree_sequence(), allele_mapper, [[0, 0, 0, 0]])


class TestMatcher:
    """
    Results of matching through Matcher on the small panel. These are the
    same as tsinfer's previous built-in engine gave.
    """

    def results(self, ts, allele_mapper, haplotypes, **kwargs):
        return [result for _, result in _match(ts, allele_mapper, haplotypes, **kwargs)]

    def test_panel(self, panel_ts):
        # Nodes 2 and 3 copy node 1 (the virtual root), and 4 and 5 copy 2
        # and 3, never node 0.
        assert panel_ts.num_nodes == 6
        edges = {(e.child, e.parent, e.left, e.right) for e in panel_ts.edges()}
        assert edges == {
            (1, 0, 10, SEQUENCE_LENGTH),
            (2, 1, 0, SEQUENCE_LENGTH),
            (3, 1, 0, SEQUENCE_LENGTH),
            (4, 2, 0, SEQUENCE_LENGTH),
            (5, 3, 0, SEQUENCE_LENGTH),
        }
        mutations = [(m.site, m.node, m.derived_state) for m in panel_ts.mutations()]
        assert mutations == [
            (0, 2, "T"),
            (1, 3, "G"),
            (1, 4, "C"),
            (2, 4, "T"),
            (3, 5, "T"),
        ]

    def test_root_ts(self, root_ts, allele_mapper):
        results = self.results(root_ts, allele_mapper, [[1, 2, 0, 3], [0, 0, 0, 0]])
        assert results[0].path == [matching.PathSegment(0, SEQUENCE_LENGTH, 1)]
        assert results[0].mutations == [
            matching.Mutation(10, 1),
            matching.Mutation(20, 2),
            matching.Mutation(40, 3),
        ]
        assert results[1].path == [matching.PathSegment(0, SEQUENCE_LENGTH, 1)]
        assert results[1].mutations == []

    def test_copy_ancestor(self, panel_ts, allele_mapper):
        results = self.results(panel_ts, allele_mapper, YOUNG_ANCESTORS)
        assert results[0].path == [matching.PathSegment(0, SEQUENCE_LENGTH, 4)]
        assert results[1].path == [matching.PathSegment(0, SEQUENCE_LENGTH, 5)]
        for result in results:
            assert result.mutations == []

    def test_recombination(self, panel_ts, allele_mapper):
        (result,) = self.results(panel_ts, allele_mapper, [[1, 1, 1, 3]])
        assert result.path == [
            matching.PathSegment(40, SEQUENCE_LENGTH, 5),
            matching.PathSegment(0, 40, 4),
        ]
        assert result.mutations == []

    def test_allele_not_in_panel(self, panel_ts, allele_mapper):
        # "C" (code 1) is not in the panel at site 3: every node mismatches.
        (result,) = self.results(panel_ts, allele_mapper, [[0, 0, 0, 1]])
        assert result.mutations == [matching.Mutation(40, 1)]

    def test_mismatch(self, panel_ts, allele_mapper):
        # With no recombination, node 4 is the closest, mismatching at site 3.
        source_config = config.MatchSourceConfig(recombination=1e-30, mismatch=1e-3)
        (result,) = self.results(
            panel_ts,
            allele_mapper,
            [[1, 1, 1, 3]],
            params=_hmm_parameters(source_config),
        )
        assert result.path == [matching.PathSegment(0, SEQUENCE_LENGTH, 4)]
        assert result.mutations == [matching.Mutation(40, 3)]

    def test_partially_missing(self, panel_ts, allele_mapper):
        results = self.results(panel_ts, allele_mapper, [[-1, 1, 1, -1], [-1, 2, -1, 3]])
        # The path covers the sites from the first to the last that aren't
        # missing, and ends at the end of the sequence after the last site.
        assert results[0].path == [matching.PathSegment(20, 40, 4)]
        assert results[1].path == [matching.PathSegment(20, SEQUENCE_LENGTH, 5)]

    def test_all_missing(self, panel_ts, allele_mapper):
        # Every node is equally likely, so which is copied isn't specified.
        (result,) = self.results(panel_ts, allele_mapper, [[-1, -1, -1, -1]])
        assert len(result.path) == 1
        assert result.path[0].left == 0
        assert result.path[0].right == SEQUENCE_LENGTH
        assert result.path[0].parent != 0
        assert result.mutations == []

    def test_threads(self, panel_ts, allele_mapper):
        rng = np.random.default_rng(42)
        num_alleles = np.array([len(alleles) for alleles in ALLELE_TABLE])
        haplotypes = rng.integers(0, num_alleles, size=(50, len(POSITIONS)))
        single = _match(panel_ts, allele_mapper, haplotypes)
        threaded = _match(panel_ts, allele_mapper, haplotypes, num_threads=4)
        assert single == threaded
        for _, result in single:
            parents = [segment.parent for segment in result.path]
            assert 0 not in parents


def test_tskit_numbering_differs(panel_ts, allele_mapper):
    # The panel is built so that tskit (and tsls by default) number the
    # alleles at site 1 differently from the allele mapper, so the tests
    # above would catch tsinfer relying on tsls's default numbering.
    variant = next(panel_ts.variants(left=20, right=21))
    assert list(variant.alleles[:3]) == ["A", "G", "C"]
    assert list(allele_mapper.forward_map()[1]) == ["A", "C", "G", ""]
    assert isinstance(panel_ts, tskit.TreeSequence)
