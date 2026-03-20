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
genotype data. TestGenotypeRoundtrip covers biallelic inference sites only.
TestAugmentedRoundtrip adds augment_sites to recover singletons and
multi-allelic sites as well.
"""

from __future__ import annotations

import msprime
import numpy as np
import pytest
from helpers import make_sample_vcz, ts_to_sample_vcz

from tsinfer.config import (
    AncestorsConfig,
    AncestralState,
    AugmentSitesConfig,
    Config,
    MatchConfig,
    MatchSourceConfig,
    PostProcessConfig,
    Source,
)
from tsinfer.pipeline import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anc_state(store):
    return AncestralState(path=store, field="variant_ancestral_allele")


def _run_pipeline(sample_store):
    """Build config, run full pipeline, return output tree sequence."""
    src = Source(path=sample_store, name="test")
    anc_src = Source(path=None, name="ancestors", sample_time="sample_time")
    cfg = Config(
        sources={"test": src, "ancestors": anc_src},
        ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
        match=MatchConfig(
            sources={
                "ancestors": MatchSourceConfig(node_flags=0, create_individuals=False),
                "test": MatchSourceConfig(),
            },
            output="output.trees",
        ),
        post_process=PostProcessConfig(),
        ancestral_state=_anc_state(sample_store),
    )
    return run(cfg)


def _run_pipeline_with_augment(sample_store, augment_store=None, ann_store=None):
    """Run pipeline with augment_sites to recover non-inference sites.

    If *augment_store* is None, the same *sample_store* is reused as the
    augment source — augment_sites will add back any positions that the
    inference pipeline dropped (singletons, multi-allelic, fixed, etc.).

    If *ann_store* is None, *sample_store* is used as the annotation store.
    """
    if augment_store is None:
        augment_store = sample_store
    if ann_store is None:
        ann_store = sample_store
    src = Source(path=sample_store, name="test")
    aug_src = Source(path=augment_store, name="augment")
    anc_src = Source(path=None, name="ancestors", sample_time="sample_time")
    cfg = Config(
        sources={
            "test": src,
            "augment": aug_src,
            "ancestors": anc_src,
        },
        ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
        match=MatchConfig(
            sources={
                "ancestors": MatchSourceConfig(node_flags=0, create_individuals=False),
                "test": MatchSourceConfig(),
            },
            output="output.trees",
        ),
        post_process=PostProcessConfig(),
        augment_sites=AugmentSitesConfig(sources=["augment"]),
        ancestral_state=_anc_state(ann_store),
    )
    return run(cfg)


def _check_genotypes(input_store, output_ts, ploidy=1, check_all=False):
    """Assert output TS sample genotypes match input VCZ genotypes.

    When *check_all* is False (default), pre-filters to biallelic polymorphic
    inference sites. When True, checks all sites with at least one non-missing
    genotype (including singletons, multi-allelic, and monomorphic).
    """
    input_gt = input_store["call_genotype"][:]  # (S, N, P)
    input_pos = input_store["variant_position"][:]  # (S,)
    input_alleles = input_store["variant_allele"][:]  # (S, A)

    num_samples = input_gt.shape[1]

    # Identify the real sample nodes: those assigned to individuals
    sample_node_ids = []
    for ind in output_ts.individuals():
        sample_node_ids.extend(ind.nodes)
    assert len(sample_node_ids) == num_samples * ploidy

    # Build set of positions to check
    pos_to_idx = {}
    for i, p in enumerate(input_pos):
        allele_list = [str(a) for a in input_alleles[i] if str(a) != ""]
        gt_flat = input_gt[i].reshape(-1)
        non_missing = gt_flat[gt_flat >= 0]
        if len(non_missing) == 0:
            continue
        if not check_all:
            # Inference-only mode: biallelic polymorphic sites only
            if len(allele_list) != 2:
                continue
            if len(np.unique(non_missing)) < 2:
                continue
        pos_to_idx[int(p)] = i

    for variant in output_ts.variants():
        pos = int(variant.site.position)
        if pos not in pos_to_idx:
            continue
        i = pos_to_idx.pop(pos)

        gt_flat = input_gt[i].reshape(-1)
        expected = np.asarray(input_alleles[i])[gt_flat]
        observed = np.array(variant.alleles)[variant.genotypes[sample_node_ids]]

        np.testing.assert_array_equal(
            observed,
            expected,
            err_msg=f"Genotype mismatch at position {pos}",
        )

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
# Tests — inference sites only (no augment)
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

    def test_ancestral_not_ref(self):
        """Ancestral allele is allele 1 (not REF/allele 0) at every site."""
        genotypes = np.array(
            [
                [[1], [0]],
                [[0], [1]],
            ],
            dtype=np.int8,
        )  # (2 sites, 2 samples, ploidy=1)
        positions = np.array([10, 20], dtype=np.int32)
        # REF is allele 0, but ancestral is allele 1
        alleles = np.array([["A", "T"], ["C", "G"]])
        ancestral = np.array(["T", "G"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=100,
        )
        ts = _run_pipeline(store)
        _check_genotypes(store, ts, ploidy=1)

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
        input_pos = set()
        for i, p in enumerate(store["variant_position"][:]):
            allele_list = [str(a) for a in alleles[i] if str(a) != ""]
            n_distinct = len(np.unique(gt[i].reshape(-1)))
            if len(allele_list) == 2 and n_distinct > 1:
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


# ---------------------------------------------------------------------------
# Tests — with augment_sites (singletons, multi-allelic)
# ---------------------------------------------------------------------------


class TestAugmentedRoundtrip:
    def test_singleton_roundtrip(self):
        """Singleton sites are recovered via augment_sites."""
        # 3 haploid samples: 2 inference sites + 1 singleton
        genotypes = np.array(
            [
                [[0], [1], [1]],  # non-singleton (AC=2)
                [[0], [0], [1]],  # singleton (AC=1) — dropped by inference
                [[1], [0], [1]],  # non-singleton (AC=2)
            ],
            dtype=np.int8,
        )
        positions = np.array([100, 200, 300], dtype=np.int32)
        alleles = np.array([["A", "T"], ["C", "G"], ["A", "T"]])
        ancestral = np.array(["A", "C", "A"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=1000,
        )
        ts = _run_pipeline_with_augment(store)

        # All 3 sites should be in the output
        output_positions = set(ts.sites_position)
        assert 100.0 in output_positions
        assert 200.0 in output_positions
        assert 300.0 in output_positions

        _check_genotypes(store, ts, ploidy=1, check_all=True)

    def test_multiallelic_roundtrip(self):
        """Multi-allelic sites are recovered via augment_sites."""
        # 3 haploid samples: 2 biallelic inference sites + 1 tri-allelic
        genotypes = np.array(
            [
                [[0], [1], [1]],  # biallelic
                [[0], [1], [2]],  # tri-allelic — dropped by inference
                [[1], [0], [1]],  # biallelic
            ],
            dtype=np.int8,
        )
        positions = np.array([100, 200, 300], dtype=np.int32)
        alleles = np.array([["A", "T", ""], ["C", "G", "T"], ["A", "T", ""]])
        ancestral = np.array(["A", "C", "A"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=1000,
        )
        ts = _run_pipeline_with_augment(store)

        # All 3 sites should be in the output
        output_positions = set(ts.sites_position)
        assert 100.0 in output_positions
        assert 200.0 in output_positions
        assert 300.0 in output_positions

        _check_genotypes(store, ts, ploidy=1, check_all=True)

    def test_singleton_and_multiallelic_together(self):
        """Both singletons and multi-allelic sites recovered in one pass."""
        # 4 haploid samples
        genotypes = np.array(
            [
                [[0], [1], [1], [0]],  # biallelic non-singleton
                [[0], [0], [0], [1]],  # singleton
                [[0], [1], [2], [0]],  # tri-allelic
                [[1], [0], [0], [1]],  # biallelic non-singleton
            ],
            dtype=np.int8,
        )
        positions = np.array([100, 200, 300, 400], dtype=np.int32)
        alleles = np.array(
            [["A", "T", ""], ["C", "G", ""], ["A", "T", "G"], ["C", "G", ""]]
        )
        ancestral = np.array(["A", "C", "A", "C"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=1000,
        )
        ts = _run_pipeline_with_augment(store)

        output_positions = set(ts.sites_position)
        for p in [100, 200, 300, 400]:
            assert float(p) in output_positions

        _check_genotypes(store, ts, ploidy=1, check_all=True)

    def test_monomorphic_fixed_ancestral(self):
        """Monomorphic site where all samples carry the ancestral allele."""
        genotypes = np.array(
            [
                [[0], [1], [1]],  # inference site
                [[0], [0], [0]],  # monomorphic: all ancestral
                [[1], [0], [1]],  # inference site
            ],
            dtype=np.int8,
        )
        positions = np.array([100, 200, 300], dtype=np.int32)
        alleles = np.array([["A", "T"], ["C", "G"], ["A", "T"]])
        ancestral = np.array(["A", "C", "A"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=1000,
        )
        ts = _run_pipeline_with_augment(store)

        site = ts.site(position=200.0)
        assert site.ancestral_state == "C"
        # All samples carry the ancestral allele — no mutations needed
        assert len(site.mutations) == 0

        _check_genotypes(store, ts, ploidy=1, check_all=True)

    def test_monomorphic_fixed_derived(self):
        """Monomorphic site where all samples carry the derived allele."""
        genotypes = np.array(
            [
                [[0], [1], [1]],  # inference site
                [[1], [1], [1]],  # monomorphic: all derived
                [[1], [0], [1]],  # inference site
            ],
            dtype=np.int8,
        )
        positions = np.array([100, 200, 300], dtype=np.int32)
        alleles = np.array([["A", "T"], ["C", "G"], ["A", "T"]])
        ancestral = np.array(["A", "C", "A"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=1000,
        )
        ts = _run_pipeline_with_augment(store)

        site = ts.site(position=200.0)
        # Ancestral state is "C" but all samples carry "G"
        assert site.ancestral_state == "C"
        # Should have a mutation to "G" above all samples
        assert len(site.mutations) >= 1
        derived_states = {m.derived_state for m in site.mutations}
        assert "G" in derived_states

        _check_genotypes(store, ts, ploidy=1, check_all=True)

    def test_duplicate_positions_excluded(self):
        """Sites at duplicate positions within a source are excluded."""
        # 2 inference sites + 2 sites at the same position (200)
        # in the augment source — both should be skipped.
        main_store = make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        # Augment source has a duplicate position at 200
        aug_store = make_sample_vcz(
            genotypes=np.array([[[0], [1], [0]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([200, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        ts = _run_pipeline_with_augment(main_store, augment_store=aug_store)

        output_positions = set(ts.sites_position)
        # Position 200 should NOT appear — it's duplicated in the source
        assert 200.0 not in output_positions

    def test_duplicate_positions_other_sites_kept(self):
        """Non-duplicate sites from a source with some duplicates are kept."""
        main_store = make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([100, 400], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        # Augment source: position 200 is duplicated, but 300 is unique
        aug_store = make_sample_vcz(
            genotypes=np.array(
                [
                    [[0], [1], [0]],
                    [[1], [0], [1]],
                    [[0], [0], [1]],
                ],
                dtype=np.int8,
            ),
            positions=np.array([200, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"], ["A", "G"]]),
            ancestral_state=np.array(["A", "C", "A"]),
            sequence_length=1000,
        )
        # Combined annotation store covering all positions
        ann_store = make_sample_vcz(
            genotypes=np.zeros((4, 1, 1), dtype=np.int8),
            positions=np.array([100, 200, 300, 400], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"], ["A", "G"], ["C", "G"]]),
            ancestral_state=np.array(["A", "A", "A", "C"]),
            sequence_length=1000,
        )
        ts = _run_pipeline_with_augment(
            main_store, augment_store=aug_store, ann_store=ann_store
        )

        output_positions = set(ts.sites_position)
        assert 200.0 not in output_positions  # duplicate — excluded
        assert 300.0 in output_positions  # unique — kept

    def test_ancestral_state_overrides_parsimony(self):
        """Specified ancestral state is used even when parsimony would differ.

        With 4 samples carrying genotypes [1, 1, 1, 0] at a singleton site,
        parsimony would choose allele 1 ("T") as ancestral. But the specified
        ancestral state is allele 0 ("A"), so we should get ancestral_state="A"
        with 3 mutations rather than the more parsimonious 1 mutation.
        """
        # 2 inference sites (non-singleton) + 1 singleton where the
        # "rare" allele (allele 0 = "A") is declared ancestral.
        genotypes = np.array(
            [
                [[0], [1], [1], [0]],  # inference site
                [[1], [1], [1], [0]],  # singleton: 3×T, 1×A; ancestral = A
                [[1], [0], [0], [1]],  # inference site
            ],
            dtype=np.int8,
        )
        positions = np.array([100, 200, 300], dtype=np.int32)
        alleles = np.array([["A", "T"], ["A", "T"], ["C", "G"]])
        # Ancestral allele at position 200 is "A" (allele 0), the minority
        ancestral = np.array(["A", "A", "C"])

        store = make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=ancestral,
            sequence_length=1000,
        )
        ts = _run_pipeline_with_augment(store)

        site = ts.site(position=200.0)
        # Ancestral state must be "A" (the specified value), not "T"
        assert site.ancestral_state == "A"
        # With ancestral="A" and genotypes [1,1,1,0], there should be
        # at least one mutation to "T"
        assert len(site.mutations) >= 1
        derived_states = {m.derived_state for m in site.mutations}
        assert "T" in derived_states

        _check_genotypes(store, ts, ploidy=1, check_all=True)

    def test_simulated_with_singletons(self):
        """Simulated data roundtrips correctly with augment_sites."""
        sim_ts = _simulate(
            num_samples=6,
            ploidy=1,
            mutation_rate=1e-7,
            random_seed=100,
        )
        assert sim_ts.num_sites > 0, "Simulation produced no sites"
        store = ts_to_sample_vcz(sim_ts)
        ts = _run_pipeline_with_augment(store)
        # Augmented output should have at least as many sites
        assert ts.num_sites >= 1
        _check_genotypes(store, ts, ploidy=1, check_all=True)

    def test_simulated_diploid_with_singletons(self):
        """Diploid simulated data with singletons via augment_sites."""
        sim_ts = _simulate(
            num_samples=4,
            ploidy=2,
            mutation_rate=1e-7,
            random_seed=101,
        )
        assert sim_ts.num_sites > 0, "Simulation produced no sites"
        store = ts_to_sample_vcz(sim_ts)

        ts = _run_pipeline_with_augment(store)
        _check_genotypes(store, ts, ploidy=2, check_all=True)

    @pytest.mark.parametrize("seed", [60, 70, 80, 90])
    def test_augmented_multiple_seeds(self, seed):
        """Augmented roundtrip across multiple seeds."""
        sim_ts = _simulate(
            num_samples=8,
            ploidy=1,
            mutation_rate=1e-7,
            random_seed=seed,
        )
        if sim_ts.num_sites == 0:
            pytest.skip("Simulation produced no sites with this seed")
        store = ts_to_sample_vcz(sim_ts)
        ts = _run_pipeline_with_augment(store)
        _check_genotypes(store, ts, ploidy=1, check_all=True)

    @pytest.mark.parametrize("seed", [7, 20, 34, 42, 102, 131])
    def test_all_sites_present_after_augment(self, seed):
        """Every input site (including monomorphic) appears in augmented output.

        Parametrized across seeds that produce multi-allelic and monomorphic
        sites to exercise edge cases in augment_sites.
        """
        sim_ts = _simulate(
            num_samples=6,
            ploidy=1,
            mutation_rate=1e-7,
            random_seed=seed,
        )
        assert sim_ts.num_sites > 0, "Simulation produced no sites"
        store = ts_to_sample_vcz(sim_ts)
        ts = _run_pipeline_with_augment(store)

        gt = store["call_genotype"][:]
        input_positions = set()
        for i, p in enumerate(store["variant_position"][:]):
            gt_flat = gt[i].reshape(-1)
            non_missing = gt_flat[gt_flat >= 0]
            if len(non_missing) > 0:
                input_positions.add(int(p))

        output_positions = {int(s.position) for s in ts.sites()}
        assert input_positions == output_positions

        _check_genotypes(store, ts, ploidy=1, check_all=True)
