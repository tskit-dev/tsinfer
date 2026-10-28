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

import msprime
import numpy as np
import pytest
import tskit

import _tsinfer
from tsinfer import config, matching, vcz

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_ts(positions, sequence_length, node_times, haplotypes, allele_mapper):
    """
    Build a minimal tree sequence with one node per haplotype, connected to a
    virtual root (node 0, time=1.0) via edges spanning all sites, with
    mutations where the haplotype has value 1.

    positions: (num_sites,) int array
    node_times: (n_nodes,) float array (excluding root)
    haplotypes: (n_nodes, num_sites) int8 array
    allele_mapper: AlleleMapper for encoding allele strings
    """
    tables = tskit.TableCollection(sequence_length=float(sequence_length))
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = {}
    tables.nodes.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    ancestral = allele_mapper.ancestral_alleles()
    for i, pos in enumerate(positions):
        tables.sites.add_row(position=float(pos), ancestral_state=str(ancestral[i]))
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
        site_ids = np.where(np.asarray(hap) == 1)[0]
        if len(site_ids) > 0:
            codes = np.ones(len(site_ids), dtype=np.int8)
            derived = allele_mapper.decode_mutations(
                np.array(site_ids, dtype=np.int32), codes
            )
            for site_idx, ds in zip(site_ids, derived):
                tables.mutations.add_row(
                    site=int(site_idx), node=node_id, derived_state=ds
                )
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
    """Return *n* dummy MatchJobs (the reader ignores their content)."""
    return [_make_job(haplotype_index=i) for i in range(n)]


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
    return config.MatchJob(
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


def _hmm_params(positions, source_config=None):
    """HMM parameters for the single "test" source used in these tests."""
    if source_config is None:
        source_config = config.MatchSourceConfig()
    return {"test": matching.hmm_parameters(source_config, positions)}


# ---------------------------------------------------------------------------
# TestMakeRootTs
# ---------------------------------------------------------------------------


class TestMakeRootTs:
    def _mapper(self, n):
        return vcz.AlleleMapper(n, [["A", "T"]] * n)

    def test_sequence_length(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = matching.make_root_ts(
            1000.0, positions, intervals, allele_mapper=self._mapper(3)
        )
        assert ts.sequence_length == 1000.0

    def test_num_nodes_two(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = matching.make_root_ts(
            1000.0, positions, intervals, allele_mapper=self._mapper(3)
        )
        assert ts.num_nodes == 2

    def test_sites_at_correct_positions(self):
        positions = np.array([5, 15, 25, 100], dtype=np.int32)
        intervals = np.array([[5, 101]], dtype=np.int32)
        ts = matching.make_root_ts(
            200.0, positions, intervals, allele_mapper=self._mapper(4)
        )
        assert ts.num_sites == 4
        np.testing.assert_array_equal(ts.tables.sites.position, [5, 15, 25, 100])

    def test_sequence_intervals_in_metadata(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31], [100, 200]], dtype=np.int32)
        ts = matching.make_root_ts(
            500.0, positions, intervals, allele_mapper=self._mapper(3)
        )
        meta = ts.metadata
        assert "sequence_intervals" in meta
        assert meta["sequence_intervals"] == [[10, 31], [100, 200]]

    def test_single_site(self):
        positions = np.array([42], dtype=np.int32)
        intervals = np.array([[42, 43]], dtype=np.int32)
        ts = matching.make_root_ts(
            100.0, positions, intervals, allele_mapper=self._mapper(1)
        )
        assert ts.num_sites == 1
        assert ts.num_nodes == 2
        assert ts.num_edges == 1
        np.testing.assert_array_equal(ts.tables.sites.position, [42])

    def test_one_edge(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = matching.make_root_ts(
            1000.0, positions, intervals, allele_mapper=self._mapper(3)
        )
        assert ts.num_edges == 1

    def test_no_mutations(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = matching.make_root_ts(
            1000.0, positions, intervals, allele_mapper=self._mapper(3)
        )
        assert ts.num_mutations == 0

    def test_sequence_length_float(self):
        positions = np.array([1, 2, 3], dtype=np.int32)
        intervals = np.array([[1, 4]], dtype=np.int32)
        ts = matching.make_root_ts(
            999.5, positions, intervals, allele_mapper=self._mapper(3)
        )
        assert ts.sequence_length == 999.5

    def test_node_metadata_schema_set(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = matching.make_root_ts(
            1000.0, positions, intervals, allele_mapper=self._mapper(3)
        )
        assert ts.tables.nodes.metadata_schema.schema is not None
        for node in ts.nodes():
            assert node.metadata == {}

    def test_individuals_and_populations(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        ts = matching.make_root_ts(
            1000.0,
            positions,
            intervals,
            allele_mapper=self._mapper(3),
            individuals=[{"sample_id": "a"}, {"sample_id": "b"}],
            populations=[{"name": "p"}],
        )
        assert [ind.metadata for ind in ts.individuals()] == [
            {"sample_id": "a"},
            {"sample_id": "b"},
        ]
        assert [pop.metadata for pop in ts.populations()] == [{"name": "p"}]


class TestPrepareReferenceTs:
    def _reference_ts(self):
        positions = np.array([10, 20, 30], dtype=np.int32)
        intervals = np.array([[10, 31]], dtype=np.int32)
        return matching.make_root_ts(
            1000.0,
            positions,
            intervals,
            allele_mapper=vcz.AlleleMapper(3, [["A", "T"]] * 3),
            individuals=[{"sample_id": "existing"}],
        )

    def test_no_rows_returns_equal_ts(self):
        ref = self._reference_ts()
        ts = matching.prepare_reference_ts(ref)
        ts.tables.assert_equals(ref.tables)

    def test_appends_individuals(self):
        ref = self._reference_ts()
        ts = matching.prepare_reference_ts(ref, individuals=[{"sample_id": "new"}])
        assert [ind.metadata for ind in ts.individuals()] == [
            {"sample_id": "existing"},
            {"sample_id": "new"},
        ]
        # Everything other than the individuals is unchanged
        ts.tables.assert_equals(ref.tables, ignore_tables=["individuals"])

    def test_populations_on_empty_table_get_json_schema(self):
        ref = self._reference_ts()
        ts = matching.prepare_reference_ts(ref, populations=[{"name": "p"}])
        assert ts.tables.populations.metadata_schema.schema == {"codec": "json"}
        assert ts.population(0).metadata == {"name": "p"}

    def test_existing_population_schema_kept(self):
        tables = self._reference_ts().dump_tables()
        schema = tskit.MetadataSchema.permissive_json()
        tables.populations.metadata_schema = schema
        tables.populations.add_row(metadata={"name": "old"})
        ref = tables.tree_sequence()
        ts = matching.prepare_reference_ts(ref, populations=[{"name": "new"}])
        assert ts.tables.populations.metadata_schema == schema
        assert [pop.metadata for pop in ts.populations()] == [
            {"name": "old"},
            {"name": "new"},
        ]


# ---------------------------------------------------------------------------
# TestMatcher
# ---------------------------------------------------------------------------


_HAPMAP_HEADER = "Chromosome Position(bp) Rate(cM/Mb) Map(cM)"


def _write_map_rows(path, rows, header=_HAPMAP_HEADER):
    """Write a HapMap file from (position, rate cM/Mb, map cM) rows."""
    lines = [header]
    for pos, rate, cm in rows:
        lines.append(f"chr1 {pos} {rate!r} {cm!r}")
    path.write_text("\n".join(lines) + "\n")


def _write_hapmap(path, rate_map, omit_origin=False):
    """
    Write an msprime RateMap (rates per bp per generation) as a HapMap file.
    A leading missing (NaN) interval is written by starting the map at the
    end of that interval with 0 cM, which is how HapMap files represent it.
    If omit_origin is True, the row at position 0 is left out, so that the
    map starts at a non-zero genetic position and the rate over the first
    interval is implied.
    """
    left = rate_map.left
    rate = rate_map.rate
    cumulative_cm = rate_map.get_cumulative_mass(left) * 100
    rows = []
    for j in range(len(left)):
        if np.isnan(rate[j]) or (omit_origin and j == 0):
            continue
        rows.append((int(left[j]), float(rate[j] * 1e8), float(cumulative_cm[j])))
    final_cm = float(rate_map.get_cumulative_mass([rate_map.sequence_length])[0] * 100)
    rows.append((int(rate_map.sequence_length), 0.0, final_cm))
    _write_map_rows(path, rows)


# How a random map is laid out in the HapMap file: starting at position 0;
# starting later with 0 cM (missing data before it); or starting later at a
# non-zero genetic position (uniform rate from position 0 implied).
_MAP_LAYOUTS = ["origin", "missing_start", "omit_origin"]


def _random_rate_map(
    seed, sequence_length=100_000, num_intervals=10, missing_start=False
):
    """A random RateMap with integer breakpoints, including zero-rate intervals."""
    rng = np.random.default_rng(seed)
    breaks = rng.choice(np.arange(1, sequence_length), num_intervals - 1, replace=False)
    position = np.concatenate([[0], np.sort(breaks), [sequence_length]])
    rate = rng.exponential(1e-6, num_intervals)
    rate[rng.random(num_intervals) < 0.3] = 0
    if missing_start:
        rate[0] = np.nan
    return msprime.RateMap(position=position, rate=rate)


def _random_sites(seed, sequence_length, rate_map=None, num_sites=50):
    """Sorted unique site positions, including the map breakpoints if given."""
    rng = np.random.default_rng(seed + 1000)
    sites = rng.integers(0, sequence_length, num_sites)
    if rate_map is not None:
        sites = np.concatenate([sites, rate_map.left.astype(int)])
    return np.unique(sites).astype(np.int32)


def _haldane(distance):
    return (1 - np.exp(-2 * np.asarray(distance))) / 2


class TestReadGeneticMap:
    """Tests for matching.read_genetic_map."""

    def _read(self, tmp_path, rows, positions):
        path = tmp_path / "map.txt"
        _write_map_rows(path, rows)
        return matching.read_genetic_map(path, np.asarray(positions))

    # Nominal cases

    def test_values_at_map_entries(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 2.0, 0.1), (3000, 0.0, 0.5)]
        result = self._read(tmp_path, rows, [0, 1000, 3000])
        np.testing.assert_allclose(result, [0, 0.001, 0.005])

    def test_linear_interpolation(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 2.0, 0.1), (3000, 0.0, 0.5)]
        result = self._read(tmp_path, rows, [500, 2000, 2500])
        np.testing.assert_allclose(result, [0.0005, 0.003, 0.004])

    def test_returns_morgans(self, tmp_path):
        rows = [(0, 1e6, 0.0), (100, 0.0, 100.0)]
        result = self._read(tmp_path, rows, [100])
        np.testing.assert_allclose(result, [1.0])

    def test_rate_column_ignored(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 2.0, 0.1), (3000, 0.0, 0.5)]
        garbage = [(pos, -99.0, cm) for pos, _, cm in rows]
        positions = [0, 500, 2000, 3000]
        expected = self._read(tmp_path, rows, positions)
        result = self._read(tmp_path, garbage, positions)
        np.testing.assert_array_equal(result, expected)

    def test_path_as_string(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (100, 0.0, 1.0)])
        result = matching.read_genetic_map(str(path), np.array([50]))
        np.testing.assert_allclose(result, [0.005])

    # Corner cases

    def test_zero_rate_interval_is_flat(self, tmp_path):
        rows = [(0, 1.0, 0.0), (100, 0.0, 1.0), (200, 1.0, 1.0), (300, 0.0, 2.0)]
        result = self._read(tmp_path, rows, [100, 150, 200])
        np.testing.assert_allclose(result, [0.01, 0.01, 0.01])

    def test_flat_before_map_starting_at_zero_cm(self, tmp_path):
        rows = [(1000, 1.0, 0.0), (2000, 0.0, 1.0)]
        result = self._read(tmp_path, rows, [0, 500, 1000, 1500])
        np.testing.assert_allclose(result, [0, 0, 0, 0.005])

    def test_linear_from_origin_before_map_with_offset(self, tmp_path):
        # A map starting at a non-zero genetic position implies a uniform
        # rate from position 0, as in msprime.
        rows = [(1000, 1.0, 2.0), (2000, 0.0, 3.0)]
        result = self._read(tmp_path, rows, [0, 500, 1000, 1500])
        np.testing.assert_allclose(result, [0, 0.01, 0.02, 0.025])

    def test_offset_at_position_zero_preserved(self, tmp_path):
        # Only differences in genetic position matter, so an offset at
        # position zero is harmless.
        rows = [(0, 1.0, 5.0), (1000, 0.0, 6.0)]
        result = self._read(tmp_path, rows, [0, 500, 1000])
        np.testing.assert_allclose(result, [0.05, 0.055, 0.06])

    def test_flat_beyond_last_entry(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 0.0, 1.0)]
        result = self._read(tmp_path, rows, [1000, 1001, 10**9])
        np.testing.assert_allclose(result, [0.01, 0.01, 0.01])

    def test_single_row_at_origin(self, tmp_path):
        result = self._read(tmp_path, [(0, 0.0, 0.0)], [0, 100, 10**6])
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_single_row_after_origin(self, tmp_path):
        result = self._read(tmp_path, [(100, 0.0, 1.0)], [0, 50, 100, 200])
        np.testing.assert_allclose(result, [0, 0.005, 0.01, 0.01])

    def test_no_positions(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 0.0, 1.0)]
        result = self._read(tmp_path, rows, np.array([], dtype=np.int32))
        assert result.shape == (0,)

    def test_large_int32_positions(self, tmp_path):
        top = 2**31 - 1
        rows = [(0, 1.0, 0.0), (top, 0.0, 100.0)]
        positions = np.array([top - 1000, top], dtype=np.int32)
        result = self._read(tmp_path, rows, positions)
        np.testing.assert_allclose(np.diff(result), [1000 / top])

    def test_tab_separated_extra_columns(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text(
            "chr\tpos\trate\tcm\textra\nchr1\t0\t1.0\t0.0\tx\nchr1\t100\t0.0\t1.0\ty\n"
        )
        result = matching.read_genetic_map(path, np.array([50]))
        np.testing.assert_allclose(result, [0.005])

    # Errors

    @pytest.mark.filterwarnings("ignore:loadtxt. input contained no data")
    def test_empty_map(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text(_HAPMAP_HEADER + "\n")
        with pytest.raises(ValueError, match="empty"):
            matching.read_genetic_map(path, np.array([10]))

    def test_unsorted_positions(self, tmp_path):
        rows = [(0, 1.0, 0.0), (200, 1.0, 2.0), (100, 0.0, 1.0)]
        with pytest.raises(ValueError, match="not strictly increasing"):
            self._read(tmp_path, rows, [50])

    def test_duplicate_positions(self, tmp_path):
        rows = [(0, 1.0, 0.0), (100, 1.0, 1.0), (100, 0.0, 1.0)]
        with pytest.raises(ValueError, match="not strictly increasing"):
            self._read(tmp_path, rows, [50])

    def test_decreasing_map(self, tmp_path):
        rows = [(0, 1.0, 0.0), (100, 1.0, 2.0), (200, 0.0, 1.0)]
        with pytest.raises(ValueError, match="decrease"):
            self._read(tmp_path, rows, [50])

    def test_missing_map_column(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text("chr pos rate\nchr1 0 1.0\nchr1 100 0.0\n")
        with pytest.raises(ValueError, match="column"):
            matching.read_genetic_map(path, np.array([50]))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            matching.read_genetic_map(tmp_path / "nope.txt", np.array([50]))

    # msprime oracle

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("layout", _MAP_LAYOUTS)
    def test_matches_rate_map(self, tmp_path, seed, layout):
        rate_map = _random_rate_map(seed, missing_start=layout == "missing_start")
        path = tmp_path / "map.txt"
        _write_hapmap(path, rate_map, omit_origin=layout == "omit_origin")
        sites = _random_sites(seed, rate_map.sequence_length, rate_map)
        result = matching.read_genetic_map(path, sites)
        expected = rate_map.get_cumulative_mass(sites)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("layout", _MAP_LAYOUTS)
    def test_matches_msprime_read_hapmap(self, tmp_path, seed, layout):
        rate_map = _random_rate_map(seed, missing_start=layout == "missing_start")
        path = tmp_path / "map.txt"
        _write_hapmap(path, rate_map, omit_origin=layout == "omit_origin")
        oracle = msprime.RateMap.read_hapmap(path, map_col=3)
        sites = _random_sites(seed, oracle.sequence_length, oracle)
        result = matching.read_genetic_map(path, sites)
        np.testing.assert_allclose(
            result, oracle.get_cumulative_mass(sites), rtol=1e-12, atol=1e-15
        )

    def test_matches_msprime_beyond_last_entry(self, tmp_path):
        rows = [(100, 2.0, 0.0), (5000, 0.0, 0.0098)]
        path = tmp_path / "map.txt"
        _write_map_rows(path, rows)
        oracle = msprime.RateMap.read_hapmap(path, map_col=3, sequence_length=20_000)
        sites = np.array([0, 50, 100, 2500, 5000, 12_000, 20_000], dtype=np.int32)
        result = matching.read_genetic_map(path, sites)
        np.testing.assert_allclose(result, oracle.get_cumulative_mass(sites))


class TestRecombinationProbabilities:
    """Tests for matching.recombination_probabilities."""

    positions = np.array([10, 200, 250, 1000, 5000, 9000], dtype=np.int32)

    # Nominal cases

    def test_constant_rate(self):
        rate = 1e-5
        probs = matching.recombination_probabilities(rate, self.positions)
        assert probs.shape == (6,)
        assert probs[0] == 0
        np.testing.assert_allclose(probs[1:], _haldane(np.diff(self.positions) * rate))

    def test_small_distances_approximately_linear(self):
        probs = matching.recombination_probabilities(1e-8, self.positions)
        np.testing.assert_allclose(probs[1:], np.diff(self.positions) * 1e-8, rtol=1e-4)

    def test_known_value(self):
        # 0.5 Morgans between the sites
        probs = matching.recombination_probabilities(0.005, np.array([0, 100]))
        np.testing.assert_allclose(probs, [0, (1 - np.exp(-1)) / 2])

    def test_proportional_to_spacing_for_small_rate(self):
        positions = np.array([0, 100, 300, 600])
        probs = matching.recombination_probabilities(1e-10, positions)
        np.testing.assert_allclose(probs[2] / probs[1], 2, rtol=1e-6)
        np.testing.assert_allclose(probs[3] / probs[1], 3, rtol=1e-6)

    def test_genetic_map(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (1000, 3.0, 0.1), (3000, 0.0, 0.7)])
        positions = np.array([0, 500, 1000, 2000, 3000])
        probs = matching.recombination_probabilities(path, positions)
        distance = np.array([0.0005, 0.0005, 0.003, 0.003])
        np.testing.assert_allclose(probs, np.concatenate([[0], _haldane(distance)]))

    def test_genetic_map_path_as_string(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (10_000, 0.0, 1.0)])
        from_str = matching.recombination_probabilities(str(path), self.positions)
        from_path = matching.recombination_probabilities(path, self.positions)
        np.testing.assert_array_equal(from_str, from_path)
        np.testing.assert_allclose(
            from_path[1:], _haldane(np.diff(self.positions) * 1e-6)
        )

    def test_uniform_map_equals_constant_rate(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 2.0, 0.0), (10_000, 0.0, 0.02)])
        from_map = matching.recombination_probabilities(path, self.positions)
        from_rate = matching.recombination_probabilities(2e-8, self.positions)
        np.testing.assert_allclose(from_map, from_rate)

    # Corner cases

    def test_zero_rate(self):
        probs = matching.recombination_probabilities(0.0, self.positions)
        np.testing.assert_array_equal(probs, np.zeros(6))

    def test_negative_rate(self):
        with pytest.raises(ValueError, match="non-negative"):
            matching.recombination_probabilities(-1e-8, self.positions)

    def test_single_site(self):
        probs = matching.recombination_probabilities(1e-8, np.array([100]))
        np.testing.assert_array_equal(probs, [0])

    def test_no_sites(self):
        probs = matching.recombination_probabilities(1e-8, np.array([], dtype=np.int32))
        assert probs.shape == (0,)

    def test_saturates_at_half(self):
        probs = matching.recombination_probabilities(1.0, self.positions)
        assert np.all(probs <= 0.5)
        np.testing.assert_allclose(probs[1:], 0.5)

    def test_bounded_and_monotonic_in_distance(self):
        positions = np.cumsum(np.arange(1, 50)) * 1000
        probs = matching.recombination_probabilities(1e-5, positions)
        assert np.all(probs >= 0)
        assert np.all(probs <= 0.5)
        assert np.all(np.diff(probs[1:]) > 0)

    def test_large_int32_positions_do_not_overflow(self):
        positions = np.array([0, 2**31 - 1], dtype=np.int32)
        probs = matching.recombination_probabilities(1e-10, positions)
        np.testing.assert_allclose(probs[1], _haldane((2**31 - 1) * 1e-10))

    def test_sites_in_zero_rate_region(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 0.0, 0.0), (5000, 1.0, 0.0), (6000, 0.0, 0.1)])
        probs = matching.recombination_probabilities(path, np.array([10, 2000, 4999]))
        np.testing.assert_array_equal(probs, [0, 0, 0])

    def test_sites_beyond_map(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (1000, 0.0, 1.0)])
        positions = np.array([500, 1500, 2500])
        probs = matching.recombination_probabilities(path, positions)
        np.testing.assert_allclose(probs, [0, _haldane(0.005), 0])

    # msprime oracle

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("layout", _MAP_LAYOUTS)
    def test_matches_rate_map(self, tmp_path, seed, layout):
        rate_map = _random_rate_map(seed, missing_start=layout == "missing_start")
        path = tmp_path / "map.txt"
        _write_hapmap(path, rate_map, omit_origin=layout == "omit_origin")
        sites = _random_sites(seed, rate_map.sequence_length)
        probs = matching.recombination_probabilities(path, sites)
        distance = np.diff(rate_map.get_cumulative_mass(sites))
        assert probs[0] == 0
        np.testing.assert_allclose(probs[1:], _haldane(distance), rtol=1e-9, atol=1e-15)

    @pytest.mark.parametrize("rate", [0.0, 1e-9, 1e-8, 1e-6])
    def test_constant_rate_matches_rate_map(self, rate):
        sites = _random_sites(0, 1_000_000)
        rate_map = msprime.RateMap.uniform(1_000_000, rate)
        probs = matching.recombination_probabilities(rate, sites)
        # The oracle differences cumulative masses of up to ~1, which loses
        # ~1e-16 in absolute terms, so compare with an absolute tolerance too.
        distance = np.diff(rate_map.get_cumulative_mass(sites))
        np.testing.assert_allclose(probs[1:], _haldane(distance), rtol=1e-12, atol=1e-15)


class TestHmmParameters:
    """Tests for computing per-site HMM parameters from a MatchSourceConfig."""

    positions = np.array([10, 200, 250, 1000, 5000, 9000], dtype=np.int32)

    def test_defaults(self):
        params = matching.hmm_parameters(config.MatchSourceConfig(), self.positions)
        np.testing.assert_array_equal(params.recombination, np.full(6, 1e-2))
        np.testing.assert_array_equal(params.mismatch, np.full(6, 1e-20))
        assert params.likelihood_threshold is None

    def test_scalar_overrides(self):
        source_config = config.MatchSourceConfig(
            recombination=0.25, mismatch=0.1, likelihood_threshold=1e-6
        )
        params = matching.hmm_parameters(source_config, self.positions)
        np.testing.assert_array_equal(params.recombination, np.full(6, 0.25))
        np.testing.assert_array_equal(params.mismatch, np.full(6, 0.1))
        assert params.likelihood_threshold == 1e-6

    def test_constant_rate(self):
        rate = 1e-5
        source_config = config.MatchSourceConfig(recombination_rate=rate)
        params = matching.hmm_parameters(source_config, self.positions)
        np.testing.assert_array_equal(
            params.recombination,
            matching.recombination_probabilities(rate, self.positions),
        )
        np.testing.assert_array_equal(params.mismatch, np.full(6, 1e-20))

    def test_constant_rate_mismatch(self):
        source_config = config.MatchSourceConfig(recombination_rate=1e-8, mismatch=1e-3)
        params = matching.hmm_parameters(source_config, self.positions)
        np.testing.assert_array_equal(params.mismatch, np.full(6, 1e-3))

    def test_genetic_map(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (10_000, 0.0, 1.0)])
        source_config = config.MatchSourceConfig(recombination_rate=str(path))
        params = matching.hmm_parameters(source_config, self.positions)
        np.testing.assert_array_equal(
            params.recombination,
            matching.recombination_probabilities(path, self.positions),
        )


class TestMatcher:
    """Tests for Matcher using a simple known tree sequence."""

    def setup_method(self):
        self.positions = np.array([10, 20, 30], dtype=np.int32)
        self.seq_len = 100.0
        self.num_sites = 3
        self.allele_mapper = vcz.AlleleMapper(3, [["A", "T"]] * 3)

    def _make_ts_with_root_only(self):
        """Empty tree sequence (just sites, no nodes)."""
        return matching.make_root_ts(
            self.seq_len,
            self.positions,
            np.array([[10, 31]], dtype=np.int32),
            allele_mapper=self.allele_mapper,
        )

    def _make_ts_with_one_ancestor(self, hap):
        """Tree sequence with virtual root + one ancestor haplotype."""
        return _make_simple_ts(
            self.positions,
            self.seq_len,
            node_times=[0.5],
            haplotypes=[hap],
            allele_mapper=self.allele_mapper,
        )

    def test_match_identical_haplotype(self):
        """Matching an identical haplotype returns a MatchResult with path edges."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
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
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
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
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
        results = list(matcher.match(_jobs(1), _ArrayReader([missing_hap])))
        assert len(results) == 1

    def test_match_partial_missing(self):
        """Haplotype with some missing sites: active range shrinks."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        query_hap = np.array([-1, 1, 0], dtype=np.int8)
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
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
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
        results = list(matcher.match(_jobs(3), _ArrayReader(haplotypes)))
        assert len(results) == 3

    def test_match_result_types(self):
        """MatchResult should contain PathSegment and Mutation objects."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        _, r = results[0]
        assert isinstance(r.path, list)
        assert isinstance(r.mutations, list)
        for seg in r.path:
            assert isinstance(seg, matching.PathSegment)
            assert isinstance(seg.left, float)
            assert isinstance(seg.right, float)
            assert isinstance(seg.parent, int)

    def test_path_segments_have_valid_coordinates(self):
        """PathSegment left < right for every segment."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        _, r = results[0]
        for seg in r.path:
            assert seg.left < seg.right

    def test_mutations_have_valid_fields(self):
        """Mutation objects should have position and derived_state."""
        hap = np.array([0, 0, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        query = np.array([1, 1, 0], dtype=np.int8)
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
        results = list(matcher.match(_jobs(1), _ArrayReader([query])))
        _, r = results[0]
        assert len(r.mutations) > 0
        for m in r.mutations:
            assert isinstance(m, matching.Mutation)
            assert isinstance(m.position, float)
            assert isinstance(m.derived_state, int)

    def test_match_with_custom_recombination_and_mismatch(self):
        """Custom per-source recombination/mismatch overrides are used."""
        hap = np.array([0, 1, 0], dtype=np.int8)
        ts = self._make_ts_with_one_ancestor(hap)
        source_config = config.MatchSourceConfig(recombination=0.5, mismatch=0.1)
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions, source_config),
        )
        results = list(matcher.match(_jobs(1), _ArrayReader([hap])))
        assert len(results) == 1
        _, r = results[0]
        assert len(r.path) > 0

    def _match_recombinant(self, source_config):
        """Match the recombinant [1, 1, 1] against ancestors [1, 1, 0] and [0, 0, 1]."""
        # Ancestors A and B both descend from an all-ancestral ultimate
        # ancestor below the virtual root.
        ts = _make_simple_ts(
            self.positions,
            self.seq_len,
            node_times=[0.9],
            haplotypes=[[0, 0, 0]],
            allele_mapper=self.allele_mapper,
        )
        tables = ts.dump_tables()
        a = tables.nodes.add_row(time=0.8, flags=0, metadata={})
        b = tables.nodes.add_row(time=0.8, flags=0, metadata={})
        for child in [a, b]:
            tables.edges.add_row(left=10, right=self.seq_len, parent=1, child=child)
        for site, node in [(0, a), (1, a), (2, b)]:
            tables.mutations.add_row(site=site, node=node, derived_state="T")
        tables.sort()
        ts = tables.tree_sequence()
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions, source_config),
        )
        query = np.array([1, 1, 1], dtype=np.int8)
        _, r = list(matcher.match(_jobs(1), _ArrayReader([query])))[0]
        return r

    def test_default_parameters_prefer_recombination(self):
        r = self._match_recombinant(config.MatchSourceConfig())
        assert len(r.path) == 2
        assert len(r.mutations) == 0

    def test_low_recombination_rate_prefers_mismatch(self):
        source_config = config.MatchSourceConfig(recombination_rate=1e-12, mismatch=1e-2)
        r = self._match_recombinant(source_config)
        assert len(r.path) == 1
        assert [m.position for m in r.mutations] == [30.0]

    def test_likelihood_threshold(self):
        source_config = config.MatchSourceConfig(likelihood_threshold=1e-5)
        r = self._match_recombinant(source_config)
        assert len(r.path) == 2


# ---------------------------------------------------------------------------
# TestExtendTs
# ---------------------------------------------------------------------------


class TestMatcherMultiallelic:
    """Matcher uses the real allele code of each mutation in the panel."""

    def setup_method(self):
        self.positions = np.array([10, 20, 30], dtype=np.int32)
        self.seq_len = 100.0
        self.allele_mapper = vcz.AlleleMapper(3, [["A", "T", "G"]] * 3)

    def _make_ts(self, haplotypes):
        """
        Panel with one node per haplotype (ids 2, 3, ...) under the virtual
        root, carrying a mutation to each non-zero allele code.
        """
        ts = matching.make_root_ts(
            self.seq_len,
            self.positions,
            np.array([[10, 31]], dtype=np.int32),
            max_time=1.0,
            allele_mapper=self.allele_mapper,
        )
        paired_results = []
        for j, hap in enumerate(haplotypes):
            mutations = [
                matching.Mutation(position=float(pos), derived_state=int(a))
                for pos, a in zip(self.positions, hap)
                if a > 0
            ]
            path = [matching.PathSegment(left=10.0, right=self.seq_len, parent=1)]
            result = matching.MatchResult(path=path, mutations=mutations)
            job = _make_job(haplotype_index=j, time=0.5, individual_id=None)
            paired_results.append((job, result))
        return matching.extend_ts(
            ts, paired_results=paired_results, allele_mapper=self.allele_mapper
        )

    def _match(self, ts, haplotypes):
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
        jobs = _jobs(len(haplotypes))
        return [r for _, r in matcher.match(jobs, _ArrayReader(haplotypes))]

    def test_copy_second_derived_allele(self):
        ts = self._make_ts([[0, 2, 0]])
        (r,) = self._match(ts, [[0, 2, 0]])
        assert r.mutations == []
        assert [s.parent for s in r.path] == [2]

    def test_mismatch_to_first_derived_allele(self):
        ts = self._make_ts([[0, 2, 0]])
        (r,) = self._match(ts, [[0, 1, 0]])
        assert [(m.position, m.derived_state) for m in r.mutations] == [(20.0, 1)]

    def test_multiple_mutations_at_site(self):
        ts = self._make_ts([[1, 1, 0], [0, 2, 1]])
        assert ts.num_mutations == 4
        assert list(ts.mutations_site) == [0, 1, 1, 2]
        r1, r2 = self._match(ts, [[1, 1, 0], [0, 2, 1]])
        assert r1.mutations == []
        assert [s.parent for s in r1.path] == [2]
        assert r2.mutations == []
        assert [s.parent for s in r2.path] == [3]

    def test_back_mutation_to_ancestral(self):
        ts = self._make_ts([[1, 1, 1]])
        result = matching.MatchResult(
            path=[matching.PathSegment(left=10.0, right=self.seq_len, parent=2)],
            mutations=[matching.Mutation(position=20.0, derived_state=0)],
        )
        job = _make_job(time=0.25, individual_id=None)
        ts = matching.extend_ts(
            ts, paired_results=[(job, result)], allele_mapper=self.allele_mapper
        )
        assert list(ts.mutations_derived_state) == ["T", "T", "A", "T"]
        (r,) = self._match(ts, [[1, 0, 1]])
        assert r.mutations == []
        assert [s.parent for s in r.path] == [3]

    def test_unknown_derived_state(self):
        tables = self._make_ts([[0, 0, 0]]).dump_tables()
        tables.mutations.add_row(site=1, node=2, derived_state="X")
        with pytest.raises(_tsinfer.LibraryError, match="derived state"):
            matching.Matcher(
                tables.tree_sequence(),
                self.positions,
                allele_mapper=self.allele_mapper,
                hmm_parameters=_hmm_params(self.positions),
            )


class TestExtendTs:
    def setup_method(self):
        self.positions = np.array([10, 20, 30], dtype=np.int32)
        self.seq_len = 100.0
        self.intervals = np.array([[10, 31]], dtype=np.int32)
        self.allele_mapper = vcz.AlleleMapper(3, [["A", "T"]] * 3)

    def _root_ts(self, individuals=None, populations=None):
        return matching.make_root_ts(
            self.seq_len,
            self.positions,
            self.intervals,
            individuals=individuals,
            populations=populations,
            allele_mapper=self.allele_mapper,
        )

    def _simple_match_result(self, parent_id, num_sites):
        """A MatchResult that copies from parent_id across all sites."""
        left_pos = float(self.positions[0])
        right_pos = self.seq_len
        return matching.MatchResult(
            path=[
                matching.PathSegment(left=left_pos, right=right_pos, parent=parent_id)
            ],
            mutations=[],
        )

    def test_node_count_increases(self):
        ts = self._root_ts()
        assert ts.num_nodes == 2  # ultimate root + virtual root

        job = _make_job(time=0.5)
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(job, self._simple_match_result(1, len(self.positions)))],
            allele_mapper=self.allele_mapper,
        )
        assert ts2.num_nodes == 3

    def test_edges_present_after_extend(self):
        ts = self._root_ts()
        job = _make_job(time=0.5)
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(job, self._simple_match_result(1, len(self.positions)))],
            allele_mapper=self.allele_mapper,
        )
        assert ts2.num_edges > 0

    def test_mutations_present_after_extend(self):
        ts = self._root_ts()
        result_with_mut = matching.MatchResult(
            path=[matching.PathSegment(left=10.0, right=100.0, parent=1)],
            mutations=[matching.Mutation(position=20.0, derived_state=1)],
        )
        job = _make_job(time=0.5)
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(job, result_with_mut)],
            allele_mapper=self.allele_mapper,
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
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(j1, r1), (j2, r2)],
            allele_mapper=self.allele_mapper,
        )
        assert ts2.num_individuals == 1
        ind = list(ts2.individuals())[0]
        assert len(ind.nodes) == 2

    def test_metadata_preserved(self):
        ts = self._root_ts()
        assert "sequence_intervals" in ts.metadata
        job = _make_job(time=0.5)
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(job, self._simple_match_result(1, len(self.positions)))],
            allele_mapper=self.allele_mapper,
        )
        meta = ts2.metadata
        assert "sequence_intervals" in meta

    def test_no_individual_when_individual_id_none(self):
        ts = self._root_ts()
        r1 = self._simple_match_result(1, len(self.positions))
        r2 = self._simple_match_result(1, len(self.positions))
        j1 = _make_job(haplotype_index=0, time=0.0, individual_id=None)
        j2 = _make_job(haplotype_index=1, time=0.0, individual_id=None)
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(j1, r1), (j2, r2)],
            allele_mapper=self.allele_mapper,
        )
        assert ts2.num_individuals == 0

    def test_sites_preserved(self):
        ts = self._root_ts()
        job = _make_job(time=0.5)
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(job, self._simple_match_result(1, len(self.positions)))],
            allele_mapper=self.allele_mapper,
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
        ts2 = matching.extend_ts(
            ts,
            paired_results=list(zip(jobs, results)),
            allele_mapper=self.allele_mapper,
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
        ts2 = matching.extend_ts(
            ts, paired_results=[(job, result)], allele_mapper=self.allele_mapper
        )
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
        ts2 = matching.extend_ts(
            ts,
            paired_results=list(zip(jobs, results)),
            allele_mapper=self.allele_mapper,
        )
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
        ts2 = matching.extend_ts(
            ts,
            paired_results=list(zip(jobs, results)),
            allele_mapper=self.allele_mapper,
        )
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
        ts2 = matching.extend_ts(
            ts, paired_results=[(job, r)], allele_mapper=self.allele_mapper
        )
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
        ts2 = matching.extend_ts(
            ts,
            paired_results=list(zip(jobs, results)),
            allele_mapper=self.allele_mapper,
        )
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
        self.allele_mapper = vcz.AlleleMapper(5, [["A", "T"]] * 5)

    def _build_root_ts(self):
        """Create a root TS with ultimate root + virtual root (2 nodes)."""
        return matching.make_root_ts(
            self.seq_len,
            self.positions,
            self.intervals,
            allele_mapper=self.allele_mapper,
        )

    def _match_and_pair(self, ts, haplotypes):
        """Match haplotypes against ts and return paired_results list."""
        matcher = matching.Matcher(
            ts,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
        results = list(matcher.match(_jobs(len(haplotypes)), _ArrayReader(haplotypes)))
        return results

    def test_node_time_correct_after_cycle(self):
        """Node times in the output TS should match the input times."""
        ts = self._build_root_ts()
        ancestor_hap = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        paired = self._match_and_pair(ts, [ancestor_hap])
        job = _make_job(time=0.5, individual_id=None)
        paired_results = [(job, paired[0][1])]
        ts2 = matching.extend_ts(
            ts, paired_results=paired_results, allele_mapper=self.allele_mapper
        )
        node_times = [n.time for n in ts2.nodes()]
        assert 1.0 in node_times
        assert 0.5 in node_times

    def test_matching_after_two_extends(self):
        """After adding ancestor, matching a similar haplotype gives a path."""
        ts = self._build_root_ts()
        ancestor_hap = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        paired = self._match_and_pair(ts, [ancestor_hap])
        job = _make_job(time=0.5, individual_id=None)
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(job, paired[0][1])],
            allele_mapper=self.allele_mapper,
        )

        # Now match a sample against ts2
        sample_hap = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        matcher2 = matching.Matcher(
            ts2,
            self.positions,
            allele_mapper=self.allele_mapper,
            hmm_parameters=_hmm_params(self.positions),
        )
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
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(job1, paired1[0][1])],
            allele_mapper=self.allele_mapper,
        )
        assert ts2.num_nodes == 3

        hap2 = np.array([[1, 0, 0, 1, 0]], dtype=np.int8)
        paired2 = self._match_and_pair(ts2, hap2)
        job2 = _make_job(time=0.5, haplotype_index=1, individual_id=None)
        ts3 = matching.extend_ts(
            ts2,
            paired_results=[(job2, paired2[0][1])],
            allele_mapper=self.allele_mapper,
        )
        assert ts3.num_nodes == 4

        sample_hap = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        paired3 = self._match_and_pair(ts3, sample_hap)
        job3 = _make_job(time=0.0, haplotype_index=2, individual_id=None)
        ts4 = matching.extend_ts(
            ts3,
            paired_results=[(job3, paired3[0][1])],
            allele_mapper=self.allele_mapper,
        )
        assert ts4.num_nodes == 5

    def test_metadata_survives_multiple_cycles(self):
        """sequence_intervals metadata should survive multiple extend_ts calls."""
        ts = self._build_root_ts()
        hap = np.array([[0, 1, 0, 0, 1]], dtype=np.int8)
        paired = self._match_and_pair(ts, hap)
        job = _make_job(time=0.5, individual_id=None)
        ts2 = matching.extend_ts(
            ts,
            paired_results=[(job, paired[0][1])],
            allele_mapper=self.allele_mapper,
        )
        assert "sequence_intervals" in ts2.metadata
        assert ts2.metadata["sequence_intervals"] == [[10, 51]]


# ---------------------------------------------------------------------------
# TestAncestorMatcherWrapper
# ---------------------------------------------------------------------------


class TestAncestorMatcherWrapper:
    def test_optional_kwargs(self):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        tables = ts.dump_tables()
        tables.sequence_length = 2
        tables.edges.right = np.full(len(tables.edges), 2, dtype=np.float64)
        tables.sites.add_row(position=1, ancestral_state="A")
        tables.mutations.add_row(site=0, node=1, derived_state="T")
        ts = tables.tree_sequence()
        mi = matching.MatcherIndexes(ts)
        r = np.full(ts.num_sites, 1e-9)
        m = np.full(ts.num_sites, 0.0)
        am = matching.AncestorMatcher(
            mi, r, m, likelihood_threshold=1e-10, weight_by_n=False
        )
        assert am is not None
