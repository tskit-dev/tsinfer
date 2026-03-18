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
End-to-end genotype round-trip tests for the tsinfer pipeline.

Verifies that the output tree sequence from run() reproduces the input
genotype data for simple biallelic cases where all sites are inference sites.
"""

from __future__ import annotations

import msprime
import numpy as np
import pytest
from helpers import make_sample_vcz, ts_to_sample_vcz

from tsinfer.config import (
    AncestorsConfig,
    Config,
    MatchConfig,
    PostProcessConfig,
    Source,
)
from tsinfer.pipeline import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_pipeline(sample_store, recombination_rate=1e-4):
    """Build config, run full pipeline, return output tree sequence."""
    src = Source(path=sample_store, name="test")
    cfg = Config(
        sources={"test": src},
        ancestors=AncestorsConfig(path=None, sources=["test"]),
        match=MatchConfig(
            sources=["test"],
            output="output.trees",
            recombination_rate=recombination_rate,
        ),
        post_process=PostProcessConfig(),
    )
    return run(cfg)


def _check_genotypes(input_store, output_ts, ploidy=1):
    """Assert output TS sample genotypes match input VCZ genotypes."""
    input_gt = input_store["call_genotype"][:]  # (S, N, P)
    input_pos = input_store["variant_position"][:]  # (S,)
    input_alleles = input_store["variant_allele"][:]  # (S, A)
    input_anc = input_store["variant_ancestral_allele"][:]  # (S,)

    num_samples = input_gt.shape[1]

    # Identify the real sample nodes: those assigned to individuals (time=0)
    sample_node_ids = []
    for ind in output_ts.individuals():
        sample_node_ids.extend(ind.nodes)
    assert len(sample_node_ids) == num_samples * ploidy

    # Build position -> input row index map (biallelic, polymorphic sites only)
    pos_to_idx = {}
    for i, p in enumerate(input_pos):
        allele_list = [str(a) for a in input_alleles[i] if str(a) != ""]
        if len(allele_list) != 2:
            continue
        # Map to ancestral/derived and check polymorphism
        anc_allele = str(input_anc[i])
        anc_idx = allele_list.index(anc_allele)
        flat = input_gt[i].reshape(-1)
        mapped = np.where(flat == anc_idx, 0, 1)
        if len(np.unique(mapped)) > 1:
            pos_to_idx[int(p)] = i

    for variant in output_ts.variants():
        pos = int(variant.site.position)
        if pos not in pos_to_idx:
            continue
        i = pos_to_idx.pop(pos)

        # Map input VCZ genotypes to {ancestral=0, derived=1}
        anc_allele = str(input_anc[i])
        allele_list = [str(a) for a in input_alleles[i] if str(a) != ""]
        anc_idx = allele_list.index(anc_allele)
        expected_flat = input_gt[i].reshape(-1)  # (N*P,)
        expected = np.where(expected_flat == anc_idx, 0, 1).astype(np.int8)

        # Extract genotypes only for real sample nodes
        observed = variant.genotypes[sample_node_ids]

        np.testing.assert_array_equal(
            observed,
            expected,
            err_msg=f"Genotype mismatch at position {pos}",
        )

    # All input positions should have been seen
    assert len(pos_to_idx) == 0, f"Missing output positions: {list(pos_to_idx.keys())}"


def _simulate(
    num_samples=6,
    sequence_length=100_000,
    recombination_rate=1e-8,
    mutation_rate=1e-8,
    ploidy=1,
    random_seed=42,
):
    """Simulate a tree sequence with msprime and return it."""
    ts = msprime.sim_ancestry(
        samples=num_samples,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        ploidy=ploidy,
        population_size=10_000,
        random_seed=random_seed,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=random_seed)
    return ts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenotypeRoundtrip:
    def test_hand_constructed_haploid(self):
        """2 haploid samples, 2 biallelic sites — exact genotype match."""
        genotypes = np.array(
            [
                [[0], [1]],
                [[1], [0]],
            ],
            dtype=np.int8,
        )  # (2 sites, 2 samples, ploidy=1)
        positions = np.array([10, 20], dtype=np.int32)
        alleles = np.array([["A", "T"], ["C", "G"]])
        ancestral = np.array(["A", "C"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=100,
        )
        ts = _run_pipeline(store)
        _check_genotypes(store, ts, ploidy=1)

    def test_hand_constructed_diploid(self):
        """2 diploid individuals (4 haplotypes), 3 biallelic sites."""
        genotypes = np.array(
            [
                [[0, 1], [1, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ],
            dtype=np.int8,
        )  # (3 sites, 2 individuals, ploidy=2)
        positions = np.array([10, 20, 30], dtype=np.int32)
        alleles = np.array([["A", "T"], ["C", "G"], ["T", "A"]])
        ancestral = np.array(["A", "C", "T"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=100,
        )
        ts = _run_pipeline(store)
        _check_genotypes(store, ts, ploidy=2)

    def test_simulated_haploid_small(self):
        """4 haploid samples from msprime simulation."""
        sim_ts = _simulate(
            num_samples=4,
            ploidy=1,
            mutation_rate=1e-7,
            random_seed=1,
        )
        assert sim_ts.num_sites > 0, "Simulation produced no sites"
        store = ts_to_sample_vcz(sim_ts)
        ts = _run_pipeline(store)
        _check_genotypes(store, ts, ploidy=1)

    def test_simulated_diploid_small(self):
        """4 diploid individuals from msprime simulation."""
        sim_ts = _simulate(
            num_samples=4,
            ploidy=2,
            mutation_rate=1e-7,
            random_seed=2,
        )
        assert sim_ts.num_sites > 0, "Simulation produced no sites"
        store = ts_to_sample_vcz(sim_ts)
        ts = _run_pipeline(store)
        _check_genotypes(store, ts, ploidy=2)

    def test_simulated_more_samples(self):
        """15 haploid samples — more haplotypes and ancestor groups."""
        sim_ts = _simulate(
            num_samples=15,
            ploidy=1,
            mutation_rate=1e-7,
            random_seed=3,
        )
        assert sim_ts.num_sites > 0, "Simulation produced no sites"
        store = ts_to_sample_vcz(sim_ts)
        ts = _run_pipeline(store)
        _check_genotypes(store, ts, ploidy=1)

    def test_all_biallelic_sites_present(self):
        """Output TS has a site at every biallelic input position."""
        sim_ts = _simulate(
            num_samples=6,
            ploidy=1,
            mutation_rate=1e-7,
            random_seed=4,
        )
        assert sim_ts.num_sites > 0, "Simulation produced no sites"
        store = ts_to_sample_vcz(sim_ts)
        ts = _run_pipeline(store)

        alleles = store["variant_allele"][:]
        gt = store["call_genotype"][:]
        anc_arr = store["variant_ancestral_allele"][:]
        input_pos = set()
        for i, p in enumerate(store["variant_position"][:]):
            allele_list = [str(a) for a in alleles[i] if str(a) != ""]
            if len(allele_list) != 2:
                continue
            anc_allele = str(anc_arr[i])
            anc_idx = allele_list.index(anc_allele)
            flat = gt[i].reshape(-1)
            mapped = np.where(flat == anc_idx, 0, 1)
            if len(np.unique(mapped)) > 1:
                input_pos.add(int(p))
        output_pos = set(int(s.position) for s in ts.sites())
        assert input_pos == output_pos

    @pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
    def test_multiple_seeds(self, seed):
        """Parametrized over seeds to catch edge cases in tree topologies."""
        sim_ts = _simulate(
            num_samples=6,
            ploidy=1,
            mutation_rate=1e-7,
            random_seed=seed,
        )
        if sim_ts.num_sites == 0:
            pytest.skip("Simulation produced no sites with this seed")
        store = ts_to_sample_vcz(sim_ts)
        ts = _run_pipeline(store)
        _check_genotypes(store, ts, ploidy=1)
