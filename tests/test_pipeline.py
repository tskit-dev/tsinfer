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
Tier 2 end-to-end tests for the tsinfer pipeline: match, post_process, run.

Uses helpers.ts_to_sample_vcz(msprime_ts) to create synthetic inputs and verifies
the pipeline produces valid tree sequences.
"""

from __future__ import annotations

import io
import json
import logging

import helpers
import msprime
import numpy as np
import pandas as pd
import pytest
import tskit

from tsinfer import ancestors, config, pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate(
    num_samples=6,
    sequence_length=100_000,
    recombination_rate=1e-8,
    mutation_rate=1e-8,
    ploidy=1,
    random_seed=42,
):
    """Simulate a tree sequence with biallelic mutations and return it."""
    ts = msprime.sim_ancestry(
        samples=num_samples,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        ploidy=ploidy,
        population_size=10_000,
        random_seed=random_seed,
    )
    ts = msprime.sim_mutations(
        ts,
        rate=mutation_rate,
        model=msprime.BinaryMutationModel(),
        random_seed=random_seed,
    )
    return ts


def _anc_state(store):
    """Return an AncestralState pointing at the store's own ancestral allele field."""
    return config.AncestralState(path=store, field="variant_ancestral_allele")


def _make_config(sample_store, ancestor_store):
    """Build a Config suitable for match() from in-memory stores."""
    src = config.Source(path=sample_store, name="test")
    anc_src = config.Source(
        path=ancestor_store, name="ancestors", sample_time="sample_time"
    )
    return config.Config(
        sources={"test": src, "ancestors": anc_src},
        ancestors=[
            config.AncestorsConfig(
                name="ancestors", path=ancestor_store, sources=["test"]
            )
        ],
        match=config.MatchConfig(
            sources={
                "ancestors": config.MatchSourceConfig(
                    node_flags=0, create_individuals=False
                ),
                "test": config.MatchSourceConfig(),
            },
            output="output.trees",
        ),
        ancestral_state=_anc_state(sample_store),
    )


def _make_config_for_run(sample_store):
    """Build a Config suitable for run() from in-memory sample store."""
    src = config.Source(path=sample_store, name="test")
    anc_src = config.Source(path=None, name="ancestors", sample_time="sample_time")
    return config.Config(
        sources={"test": src, "ancestors": anc_src},
        ancestors=[
            config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ],
        match=config.MatchConfig(
            sources={
                "ancestors": config.MatchSourceConfig(
                    node_flags=0, create_individuals=False
                ),
                "test": config.MatchSourceConfig(),
            },
            output="output.trees",
        ),
        ancestral_state=_anc_state(sample_store),
    )


def _infer_and_match(sim_ts):
    """Helper: simulate → sample VCZ → infer ancestors → match → return output ts."""
    sample_store = helpers.ts_to_sample_vcz(sim_ts)
    anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
    ancestor_store = ancestors.infer_ancestors(
        config.Source(path=sample_store, name="test"), anc_cfg, _anc_state(sample_store)
    )
    cfg = _make_config(sample_store, ancestor_store)
    return pipeline.match(cfg)


# ---------------------------------------------------------------------------
# TestMatch
# ---------------------------------------------------------------------------


class TestMatch:
    def test_basic_haploid(self):
        sim_ts = _simulate(num_samples=4, random_seed=1)
        out_ts = _infer_and_match(sim_ts)
        assert out_ts.num_nodes > 0
        assert out_ts.num_edges > 0
        assert out_ts.num_sites > 0

    def test_output_has_sites_at_inference_positions(self):
        sim_ts = _simulate(num_samples=6, random_seed=2)
        out_ts = _infer_and_match(sim_ts)
        assert out_ts.num_sites > 0

    def test_output_has_sample_nodes(self):
        sim_ts = _simulate(num_samples=6, random_seed=3)
        out_ts = _infer_and_match(sim_ts)
        num_haplotypes = sim_ts.num_samples
        assert out_ts.num_nodes >= num_haplotypes

    def test_metadata_contains_sequence_intervals(self):
        sim_ts = _simulate(num_samples=4, random_seed=4)
        out_ts = _infer_and_match(sim_ts)
        assert "sequence_intervals" in out_ts.metadata

    def test_larger_sample_count(self):
        sim_ts = _simulate(num_samples=20, random_seed=5)
        out_ts = _infer_and_match(sim_ts)
        assert out_ts.num_nodes > 0

    def test_diploid(self):
        sim_ts = _simulate(num_samples=4, ploidy=2, random_seed=6)
        out_ts = _infer_and_match(sim_ts)
        assert out_ts.num_nodes > 0
        assert out_ts.num_individuals > 0

    def test_hand_constructed_simple(self):
        """Match with a hand-constructed 2-site, 2-sample VCZ."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        out_ts = pipeline.match(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_sites == 2


class TestMatchSourceFilters:
    """Variants excluded by a match source's filters are not matched."""

    def _run(self, **filters):
        sim_ts = _simulate(num_samples=6, random_seed=11)
        store_a = helpers.ts_to_sample_vcz(sim_ts)
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["a"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=store_a, name="a"), anc_cfg, _anc_state(store_a)
        )
        site_position = int(ancestor_store["variant_position"][0])
        # Source b has an indel at the first inference site, carried by all
        positions = store_a["variant_position"][:]
        alleles = store_a["variant_allele"][:].astype(object)
        genotypes = store_a["call_genotype"][:]
        row = int(np.flatnonzero(positions == site_position)[0])
        alleles[row] = [alleles[row, 0], alleles[row, 0] + "TTA"]
        genotypes[row] = 1
        num_samples = genotypes.shape[1]
        store_b = helpers.make_sample_vcz(
            genotypes=genotypes,
            positions=positions,
            alleles=alleles,
            ancestral_state=store_a["variant_ancestral_allele"][:],
            sequence_length=int(sim_ts.sequence_length),
            sample_ids=np.array([f"b{j}" for j in range(num_samples)]),
        )
        cfg = config.Config(
            sources={
                "a": config.Source(path=store_a, name="a"),
                "b": config.Source(path=store_b, name="b", **filters),
                "ancestors": config.Source(
                    path=ancestor_store, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(
                    name="ancestors", path=ancestor_store, sources=["a"]
                )
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "a": config.MatchSourceConfig(),
                    "b": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            ancestral_state=_anc_state(store_a),
        )
        ts = pipeline.match(cfg)
        site = ts.site(position=site_position)
        derived_states = {m.derived_state for m in site.mutations}
        return ts, alleles[row, 1], derived_states

    def test_unfiltered_indel_matched(self):
        _, indel, derived_states = self._run()
        assert indel in derived_states

    def test_filtered_indel_not_matched(self):
        ts, indel, derived_states = self._run(include='TYPE="snp"')
        assert indel not in derived_states
        assert indel not in {m.derived_state for m in ts.mutations()}
        assert indel not in {a for site in ts.sites() for a in site.alleles}


class TestMatchContig:
    def _sample_store(self):
        sim_ts = _simulate(num_samples=6, random_seed=7)
        return helpers.ts_to_sample_vcz(sim_ts)

    def _infer(self, sample_store):
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        return ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )

    def test_sequence_length_from_variant_contig(self):
        sample_store = self._sample_store()
        helpers.set_contigs(sample_store, ["chr1", "chr2"], [900_000, 200_000], 1)
        ancestor_store = self._infer(sample_store)
        out_ts = pipeline.match(_make_config(sample_store, ancestor_store))
        assert out_ts.sequence_length == 200_000

    def test_sequence_length_from_ancestors(self):
        # Sample store omits contig_length; the ancestor store supplies it
        sample_store = self._sample_store()
        helpers.set_contigs(sample_store, ["chr1", "chr2"], [900_000, 200_000], 1)
        ancestor_store = self._infer(sample_store)
        helpers.set_contigs(sample_store, ["chr1", "chr2"], None, 1)
        out_ts = pipeline.match(_make_config(sample_store, ancestor_store))
        assert out_ts.sequence_length == 200_000

    def test_ancestor_sample_length_mismatch(self):
        sample_store = self._sample_store()
        ancestor_store = self._infer(sample_store)
        helpers.set_contigs(sample_store, ["1"], [200_000], 0)
        with pytest.warns(UserWarning, match="conflicting lengths"):
            out_ts = pipeline.match(_make_config(sample_store, ancestor_store))
        assert out_ts.sequence_length == 200_000


class TestMultiContig:
    """Inference on one contig of a multi-contig store matches a single-contig store."""

    @pytest.fixture(scope="class")
    def stores(self):
        ts_a = _simulate(num_samples=6, sequence_length=100_000, random_seed=11)
        ts_b = _simulate(num_samples=6, sequence_length=80_000, random_seed=12)
        store_a = helpers.ts_to_sample_vcz(ts_a, contig_id="A")
        store_b = helpers.ts_to_sample_vcz(ts_b, contig_id="B")
        # C has the same positions as B but different genotypes
        store_c = helpers.ts_to_sample_vcz(ts_b, contig_id="C")
        store_c["call_genotype"][:] = store_b["call_genotype"][:][:, ::-1]
        singles = [store_a, store_b, store_c]
        combined = helpers.combine_contigs(singles, ["A", "B", "C"], 7)
        return singles, combined

    def _run(self, sample_store, contig=None):
        cfg = _make_config_for_run(sample_store)
        cfg.contig = contig
        cfg.augment_sites = config.AugmentSitesConfig(sources=["test"])
        return pipeline.run(cfg)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_matches_single_contig(self, stores, index):
        singles, combined = stores
        contig_id = ["A", "B", "C"][index]
        ts = self._run(combined, contig_id)
        single_ts = self._run(singles[index])
        assert ts.sequence_length == single_ts.sequence_length
        assert ts.num_sites > 0
        ts.tables.assert_equals(single_ts.tables, ignore_provenance=True)

    def test_no_contig(self, stores):
        _, combined = stores
        with pytest.raises(ValueError, match="contig option"):
            self._run(combined)

    def test_unknown_contig(self, stores):
        _, combined = stores
        with pytest.raises(ValueError, match="does not declare contig .D."):
            self._run(combined, "D")


class TestContigNames:
    """Stores may use different names for the contig being inferred."""

    @pytest.fixture(scope="class")
    def stores(self):
        ts = _simulate(num_samples=6, sequence_length=100_000, random_seed=13)
        sample_store = helpers.ts_to_sample_vcz(ts, contig_id="chr20")
        # Ensembl-style ancestral state: contig "20", no contig_length
        ann_store = helpers.ts_to_sample_vcz(ts, contig_id="20")
        helpers.set_contigs(ann_store, ["20"], None, 0)
        return sample_store, ann_store

    def _cfg(self, sample_store, ann_store, contig):
        cfg = _make_config_for_run(sample_store)
        cfg.ancestral_state = _anc_state(ann_store)
        cfg.contig = contig
        cfg.augment_sites = config.AugmentSitesConfig(sources=["test"])
        return cfg

    def test_matches_same_names(self, stores):
        sample_store, ann_store = stores
        cfg = self._cfg(sample_store, ann_store, "chr20")
        with pytest.warns(UserWarning, match=r"chr20 \(test\), 20 \(ancestral_state\)"):
            ts = pipeline.run(cfg)
        same_names_ts = pipeline.run(self._cfg(sample_store, sample_store, None))
        assert ts.num_sites > 0
        ts.tables.assert_equals(same_names_ts.tables, ignore_provenance=True)

    def test_ancestors_use_top_level_name(self, stores):
        sample_store, ann_store = stores
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        source = config.Source(path=sample_store, name="test")
        with pytest.warns(UserWarning, match="Inferring contig chr20"):
            anc = ancestors.infer_ancestors(
                source, anc_cfg, _anc_state(ann_store), contig="chr20"
            )
        assert anc["contig_id"][:].tolist() == ["chr20"]
        assert anc["contig_length"][:].tolist() == [100_000]

    def test_store_contig_option(self, stores):
        sample_store, ann_store = stores
        cfg = self._cfg(sample_store, ann_store, "chr20")
        cfg.ancestral_state.contig = "20"
        with pytest.warns(UserWarning, match="Inferring contig chr20"):
            ts = pipeline.run(cfg)
        assert ts.num_sites > 0

    def test_different_names_without_top_level_contig(self, stores):
        sample_store, ann_store = stores
        cfg = self._cfg(sample_store, ann_store, None)
        with pytest.raises(ValueError, match="different names for the contig"):
            pipeline.run(cfg)


# ---------------------------------------------------------------------------
# TestPostProcess
# ---------------------------------------------------------------------------


class TestPostProcess:
    def test_simplify_reduces_nodes(self):
        sim_ts = _simulate(num_samples=6, random_seed=10)
        matched_ts = _infer_and_match(sim_ts)
        cfg = config.Config(
            sources={
                "test": config.Source(path=None, name="test"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="out.trees",
            ),
            post_process=config.PostProcessConfig(
                split_ultimate=False, erase_flanks=False
            ),
            ancestral_state=config.AncestralState(
                path="dummy", field="variant_ancestral_allele"
            ),
        )
        pp_ts = pipeline.post_process(matched_ts, cfg)
        # Simplify should remove unused nodes
        assert pp_ts.num_nodes <= matched_ts.num_nodes

    def test_no_post_process_config_returns_unchanged(self):
        sim_ts = _simulate(num_samples=4, random_seed=11)
        matched_ts = _infer_and_match(sim_ts)
        cfg = config.Config(
            sources={
                "test": config.Source(path=None, name="test"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="out.trees",
            ),
            post_process=None,
            ancestral_state=config.AncestralState(
                path="dummy", field="variant_ancestral_allele"
            ),
        )
        pp_ts = pipeline.post_process(matched_ts, cfg)
        assert pp_ts.num_nodes == matched_ts.num_nodes

    def test_erase_flanks(self):
        sim_ts = _simulate(num_samples=6, random_seed=12)
        matched_ts = _infer_and_match(sim_ts)
        cfg = config.Config(
            sources={
                "test": config.Source(path=None, name="test"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="out.trees",
            ),
            post_process=config.PostProcessConfig(
                split_ultimate=False, erase_flanks=True
            ),
            ancestral_state=config.AncestralState(
                path="dummy", field="variant_ancestral_allele"
            ),
        )
        pp_ts = pipeline.post_process(matched_ts, cfg)
        assert pp_ts.num_nodes > 0


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------


class TestRun:
    def test_full_pipeline_haploid(self):
        sim_ts = _simulate(num_samples=4, random_seed=20)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        cfg = _make_config_for_run(sample_store)
        out_ts = pipeline.run(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_sites > 0

    def test_full_pipeline_diploid(self):
        sim_ts = _simulate(num_samples=4, ploidy=2, random_seed=21)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        cfg = _make_config_for_run(sample_store)
        out_ts = pipeline.run(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_individuals > 0

    def test_full_pipeline_with_post_process(self):
        sim_ts = _simulate(num_samples=6, random_seed=22)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        src = config.Source(path=sample_store, name="test")
        anc_src = config.Source(path=None, name="ancestors", sample_time="sample_time")
        cfg = config.Config(
            sources={"test": src, "ancestors": anc_src},
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            post_process=config.PostProcessConfig(
                split_ultimate=False, erase_flanks=True
            ),
            ancestral_state=_anc_state(sample_store),
        )
        out_ts = pipeline.run(cfg)
        assert out_ts.num_nodes > 0

    def test_hand_constructed(self):
        """Full pipeline with a hand-constructed sample VCZ."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = pipeline.run(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_sites == 2


# ---------------------------------------------------------------------------
# TestIsReferenceConfig
# ---------------------------------------------------------------------------


class TestIsReferenceConfig:
    def test_is_reference_with_field_raises(self):
        """is_reference=True with field set raises ValueError."""
        with pytest.raises(ValueError, match="field must not be set"):
            config.AncestralState(path="x.vcz", field="variant_AA", is_reference=True)

    def test_neither_field_nor_is_reference_raises(self):
        """Neither field nor is_reference raises ValueError."""
        with pytest.raises(ValueError, match="requires either"):
            config.AncestralState(path="x.vcz")

    def test_valid_field(self):
        """field without is_reference is valid."""
        a = config.AncestralState(path="x.vcz", field="variant_AA")
        assert a.field == "variant_AA"
        assert a.is_reference is None

    def test_valid_is_reference(self):
        """is_reference=True without field is valid."""
        a = config.AncestralState(path="x.vcz", is_reference=True)
        assert a.is_reference is True
        assert a.field is None


# ---------------------------------------------------------------------------
# TestNodeMetadata
# ---------------------------------------------------------------------------


class TestNodeMetadata:
    def test_nodes_have_metadata(self):
        """Nodes should have metadata with source and sample_id."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = pipeline.run(cfg)
        # At least some nodes should have metadata
        has_metadata = False
        for node in out_ts.nodes():
            if node.metadata:
                has_metadata = True
                break
        assert has_metadata

    def test_ancestor_nodes_have_source(self):
        """Ancestor nodes should have source='ancestors'."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = pipeline.run(cfg)
        ancestor_found = False
        for node in out_ts.nodes():
            if node.metadata and node.metadata.get("source") == "ancestors":
                ancestor_found = True
                assert "sample_id" in node.metadata
                break
        assert ancestor_found

    def test_sample_nodes_have_ploidy_index(self):
        """Sample nodes should have ploidy_index in metadata."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = pipeline.run(cfg)
        sample_found = False
        for node in out_ts.nodes():
            md = node.metadata
            if md and md.get("source") == "test":
                sample_found = True
                assert "ploidy_index" in md
                assert "sample_id" in md
                break
        assert sample_found


# ---------------------------------------------------------------------------
# TestIndividualMetadata
# ---------------------------------------------------------------------------


class TestIndividualMetadata:
    def test_individual_metadata_fields(self):
        """Individual metadata should contain fields from config."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        src = config.Source(path=sample_store, name="test")
        anc_src = config.Source(path=None, name="ancestors", sample_time="sample_time")
        cfg = config.Config(
            sources={"test": src, "ancestors": anc_src},
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            individual_metadata=config.IndividualMetadataConfig(
                fields={"sample_id": "sample_id"},
            ),
            ancestral_state=_anc_state(sample_store),
        )
        out_ts = pipeline.run(cfg)
        assert out_ts.num_individuals > 0
        for ind in out_ts.individuals():
            if ind.metadata:
                assert "sample_id" in ind.metadata

    def test_population_assignment(self):
        """Individuals should be assigned to populations from config."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
            sample_population=np.array(["pop_A", "pop_B"]),
        )
        src = config.Source(path=sample_store, name="test")
        anc_src = config.Source(path=None, name="ancestors", sample_time="sample_time")
        cfg = config.Config(
            sources={"test": src, "ancestors": anc_src},
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            individual_metadata=config.IndividualMetadataConfig(
                population="sample_population",
            ),
            ancestral_state=_anc_state(sample_store),
        )
        out_ts = pipeline.run(cfg)
        assert out_ts.num_populations > 0
        # Check population metadata
        pop_names = [pop.metadata.get("name") for pop in out_ts.populations()]
        assert "pop_A" in pop_names or "pop_B" in pop_names

    def test_no_individual_metadata_config(self):
        """Without individual_metadata config, no individual metadata is set."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = pipeline.run(cfg)
        # No individual metadata config → no individual metadata schema
        assert out_ts.num_populations == 0


# ---------------------------------------------------------------------------
# Logging and progress
# ---------------------------------------------------------------------------


class TestPipelineLogging:
    def test_run_logs_key_messages(self, caplog):
        """Key INFO messages from run()."""
        sim_ts = _simulate(num_samples=4, random_seed=30)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        cfg = _make_config_for_run(sample_store)
        with caplog.at_level(logging.INFO, logger="tsinfer"):
            pipeline.run(cfg)

        messages = caplog.text
        assert "Starting full pipeline" in messages
        assert "Pipeline complete" in messages

    def test_match_logs_haplotype_count(self, caplog):
        """match() logs haplotype and site counts."""
        sim_ts = _simulate(num_samples=4, random_seed=31)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        with caplog.at_level(logging.INFO, logger="tsinfer.pipeline"):
            pipeline.match(cfg)

        assert "Match:" in caplog.text


class TestPipelineProgress:
    def test_run_with_progress(self):
        """run(progress=True) does not crash."""
        sim_ts = _simulate(num_samples=4, random_seed=32)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        cfg = _make_config_for_run(sample_store)
        out_ts = pipeline.run(cfg, progress=True)
        assert out_ts.num_nodes > 0

    def test_match_with_progress(self):
        """match(progress=True) does not crash."""
        sim_ts = _simulate(num_samples=4, random_seed=33)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        out_ts = pipeline.match(cfg, progress=True)
        assert out_ts.num_nodes > 0


# ---------------------------------------------------------------------------
# TestComputeGroupsJson
# ---------------------------------------------------------------------------


class TestComputeGroupsJson:
    def test_basic(self):
        """compute_groups_json returns a flat JSON array of records."""
        sim_ts = _simulate(num_samples=4, random_seed=40)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(pipeline.compute_groups_json(cfg))
        assert isinstance(records, list)
        assert len(records) > 0

    def test_first_record_is_ancestor(self):
        """First record should be an ancestor in group 0."""
        sim_ts = _simulate(num_samples=4, random_seed=41)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(pipeline.compute_groups_json(cfg))
        rec0 = records[0]
        assert rec0["haplotype_index"] == 0
        assert rec0["group"] == 0
        assert rec0["source"] == "ancestors"
        assert rec0["sample_id"] != "virtual_root"

    def test_all_indices_covered(self):
        """All haplotype indices should appear exactly once."""
        sim_ts = _simulate(num_samples=6, random_seed=42)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(pipeline.compute_groups_json(cfg))
        all_indices = {r["haplotype_index"] for r in records}
        assert all_indices == set(range(len(records)))

    def test_hand_constructed(self):
        """compute_groups_json with a hand-constructed VCZ."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(pipeline.compute_groups_json(cfg))
        assert len(records) > 0
        expected_keys = {
            "haplotype_index",
            "source",
            "sample_id",
            "ploidy_index",
            "time",
            "start_position",
            "end_position",
            "group",
            "node_flags",
            "individual_id",
            "population_id",
            "sample_chunk",
        }
        for rec in records:
            assert set(rec.keys()) == expected_keys
            assert isinstance(rec["time"], float)
            assert isinstance(rec["group"], int)

    def test_round_trip_pandas(self):
        """match-jobs.json output can be loaded into a pandas DataFrame."""
        sim_ts = _simulate(num_samples=4, random_seed=45)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        json_str = pipeline.compute_groups_json(cfg)
        df = pd.read_json(io.StringIO(json_str))
        assert len(df) > 0
        expected_cols = {
            "haplotype_index",
            "source",
            "sample_id",
            "ploidy_index",
            "time",
            "start_position",
            "end_position",
            "group",
            "node_flags",
        }
        assert expected_cols.issubset(set(df.columns))
        records = json.loads(json_str)
        assert len(df) == len(records)


# ---------------------------------------------------------------------------
# TestMatchWithGroups
# ---------------------------------------------------------------------------


class TestWorkdir:
    def _setup(self, random_seed=50):
        sim_ts = _simulate(num_samples=4, random_seed=random_seed)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = ancestors.infer_ancestors(
            config.Source(path=sample_store, name="test"),
            anc_cfg,
            _anc_state(sample_store),
        )
        cfg = _make_config(sample_store, ancestor_store)
        return cfg

    def test_workdir_creates_groups_and_trees(self, tmp_path):
        """Fresh run creates match-jobs.json and at least one .trees file."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        out_ts = pipeline.match(cfg)
        assert out_ts.num_nodes > 0

        assert (workdir / "match-jobs.json").exists()
        written = list(workdir.glob("group_*.trees"))
        assert len(written) > 0
        for p in written:
            ts = tskit.load(str(p))
            assert ts.num_nodes > 0

    def test_workdir_resumes_from_checkpoint(self, tmp_path):
        """Delete later group files, re-run — skips completed, same result."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        ts1 = pipeline.match(cfg)

        # Find the group files and delete the last one to force re-run
        trees_files = sorted(workdir.glob("group_*.trees"))
        assert len(trees_files) >= 1
        if len(trees_files) > 1:
            trees_files[-1].unlink()

            ts2 = pipeline.match(cfg)
            assert ts2.num_nodes == ts1.num_nodes
            assert ts2.num_edges == ts1.num_edges

    def test_workdir_ignores_gap(self, tmp_path):
        """Remove a middle .trees file → resumes from before the gap."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        ts1 = pipeline.match(cfg)

        trees_files = sorted(workdir.glob("group_*.trees"))
        if len(trees_files) >= 3:
            # Remove a middle file to create a gap
            trees_files[1].unlink()
            ts2 = pipeline.match(cfg)
            assert ts2.num_nodes == ts1.num_nodes
            assert ts2.num_edges == ts1.num_edges

    def test_workdir_default_keeps_only_last(self, tmp_path):
        """Without keep_intermediates, only the latest .trees file remains."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)

        pipeline.match(cfg)

        trees_files = list(workdir.glob("group_*.trees"))
        assert len(trees_files) == 1

    def test_workdir_keep_intermediates(self, tmp_path):
        """With keep_intermediates, all .trees files are retained."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        pipeline.match(cfg)

        trees_files = list(workdir.glob("group_*.trees"))
        # Should have more than 1 if there are multiple groups
        assert len(trees_files) >= 1
        assert (workdir / "match-jobs.json").exists()

    def test_no_workdir_by_default(self, tmp_path):
        """workdir=None produces no files."""
        cfg = self._setup()
        assert cfg.match.workdir is None

        pipeline.match(cfg)
        assert list(tmp_path.glob("*.trees")) == []
        assert list(tmp_path.glob("match-jobs.json")) == []

    def test_group_stop_produces_partial_result(self, tmp_path):
        """group_stop=1 processes only group 0; checkpoint matches returned ts."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        partial_ts = pipeline.match(cfg, group_stop=1)
        full_ts = pipeline.match(self._setup())

        assert partial_ts.num_nodes > 0
        assert partial_ts.num_nodes < full_ts.num_nodes

        # Exactly one checkpoint: group_0.trees (the only group processed)
        groups_json = json.loads((workdir / "match-jobs.json").read_text())
        sorted_groups = sorted({r["group"] for r in groups_json})
        first_group = sorted_groups[0]
        expected_path = workdir / f"group_{first_group}.trees"
        assert expected_path.exists()
        trees_files = list(workdir.glob("group_*.trees"))
        assert len(trees_files) == 1

        # The checkpoint tree sequence should have the same node/edge
        # counts as the returned partial_ts (before relabelling/metadata,
        # so compare the underlying structure via node count).
        checkpoint_ts = tskit.load(str(expected_path))
        assert checkpoint_ts.num_nodes > 0
        assert checkpoint_ts.num_edges > 0
        # The returned ts has relabelled alleles and metadata applied on
        # top of the checkpoint, but the node/edge topology is identical.
        assert checkpoint_ts.num_nodes == partial_ts.num_nodes
        assert checkpoint_ts.num_edges == partial_ts.num_edges

    def test_group_stop_and_resume_matches_full_run(self, tmp_path):
        """Stop at group 1, resume with no stop → same as full run."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        pipeline.match(cfg, group_stop=1)

        resumed_ts = pipeline.match(cfg)
        full_ts = pipeline.match(self._setup())

        assert resumed_ts.num_nodes == full_ts.num_nodes
        assert resumed_ts.num_edges == full_ts.num_edges

    def test_group_stop_every_group_matches_full_run(self, tmp_path):
        """For each group g, stop at g then resume → matches full run."""
        base_cfg = self._setup()
        full_ts = pipeline.match(base_cfg)

        # Determine number of groups
        workdir_probe = tmp_path / "probe"
        base_cfg.match.workdir = str(workdir_probe)
        base_cfg.match.keep_intermediates = True
        pipeline.match(base_cfg)
        groups_json = json.loads((workdir_probe / "match-jobs.json").read_text())
        num_groups = max(r["group"] for r in groups_json) + 1

        for g in range(1, num_groups + 1):
            cfg = self._setup()
            wd = tmp_path / f"wd_g{g}"
            cfg.match.workdir = str(wd)
            cfg.match.keep_intermediates = True

            pipeline.match(cfg, group_stop=g)
            resumed_ts = pipeline.match(cfg)

            assert resumed_ts.num_nodes == full_ts.num_nodes
            assert resumed_ts.num_edges == full_ts.num_edges

    def test_group_stop_multi_step_resume(self, tmp_path):
        """Step through groups one at a time → final matches full run."""
        base_cfg = self._setup()
        full_ts = pipeline.match(base_cfg)

        # Determine number of groups
        workdir_probe = tmp_path / "probe"
        base_cfg.match.workdir = str(workdir_probe)
        base_cfg.match.keep_intermediates = True
        pipeline.match(base_cfg)
        groups_json = json.loads((workdir_probe / "match-jobs.json").read_text())
        num_groups = max(r["group"] for r in groups_json) + 1

        cfg = self._setup()
        wd = tmp_path / "wd_step"
        cfg.match.workdir = str(wd)
        cfg.match.keep_intermediates = True

        for g in range(1, num_groups + 1):
            pipeline.match(cfg, group_stop=g)

        final_ts = pipeline.match(cfg)

        assert final_ts.num_nodes == full_ts.num_nodes
        assert final_ts.num_edges == full_ts.num_edges

    @pytest.mark.parametrize("ploidy", [1, 2])
    def test_group_stop_zero_and_resume_matches_full_run(self, tmp_path, ploidy):
        """Stop before any group, so match-jobs.json exists but there is no
        checkpoint; resuming must still create the individuals."""
        sim_ts = _simulate(num_samples=4, ploidy=ploidy, random_seed=51)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        ancestor_store = _infer_ancestors(sample_store)
        full_ts = pipeline.match(_make_config(sample_store, ancestor_store))
        assert full_ts.num_individuals > 0

        cfg = _make_config(sample_store, ancestor_store)
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        pipeline.match(cfg, group_stop=0)
        assert (workdir / "match-jobs.json").exists()
        assert list(workdir.glob("group_*.trees")) == []

        resumed_ts = pipeline.match(cfg)
        resumed_ts.tables.assert_equals(full_ts.tables, ignore_provenance=True)

    def test_group_stop_zero_and_resume_with_populations(self, tmp_path):
        """Populations are also created when resuming without a checkpoint."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
            sample_population=np.array(["pop_A", "pop_B"]),
        )
        ancestor_store = _infer_ancestors(sample_store)
        cfg = _make_config(sample_store, ancestor_store)
        cfg.individual_metadata = config.IndividualMetadataConfig(
            population="sample_population"
        )
        full_ts = pipeline.match(cfg)
        assert full_ts.num_populations == 2

        cfg.match.workdir = str(tmp_path / "wd")
        pipeline.match(cfg, group_stop=0)
        resumed_ts = pipeline.match(cfg)
        resumed_ts.tables.assert_equals(full_ts.tables, ignore_provenance=True)

    def test_match_jobs_rewritten_without_checkpoint(self, tmp_path):
        """With no checkpoint, a stale match-jobs.json is replaced."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        workdir.mkdir()
        (workdir / "match-jobs.json").write_text("[]")
        cfg.match.workdir = str(workdir)
        out_ts = pipeline.match(cfg)
        jobs = json.loads((workdir / "match-jobs.json").read_text())
        assert len(jobs) > 0
        expected = pipeline.match(self._setup())
        out_ts.tables.assert_equals(expected.tables, ignore_provenance=True)

    def test_group_stop_beyond_last_group(self, tmp_path):
        """group_stop=9999 behaves like a normal full run."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)

        big_stop_ts = pipeline.match(cfg, group_stop=9999)
        full_ts = pipeline.match(self._setup())

        assert big_stop_ts.num_nodes == full_ts.num_nodes
        assert big_stop_ts.num_edges == full_ts.num_edges

    def test_keep_intermediates_without_workdir_errors(self):
        """keep_intermediates=True without workdir raises ValueError."""
        with pytest.raises(ValueError, match="keep_intermediates requires workdir"):
            config.Config(
                sources={
                    "test": config.Source(path="dummy", name="test"),
                    "ancestors": config.Source(
                        path="dummy", name="ancestors", sample_time="sample_time"
                    ),
                },
                ancestors=[
                    config.AncestorsConfig(
                        name="ancestors", path="dummy", sources=["test"]
                    )
                ],
                match=config.MatchConfig(
                    sources={
                        "ancestors": config.MatchSourceConfig(
                            node_flags=0, create_individuals=False
                        ),
                        "test": config.MatchSourceConfig(),
                    },
                    output="out.trees",
                    keep_intermediates=True,
                ),
                ancestral_state=config.AncestralState(
                    path="dummy", field="variant_ancestral_allele"
                ),
            )


# ---------------------------------------------------------------------------
# TestReferenceTs
# ---------------------------------------------------------------------------


def _ancestors_only_config(sample_store, ancestor_store, create_individuals=False):
    """Config that matches the ancestors only, producing a reference ts."""
    anc_src = config.Source(
        path=ancestor_store, name="ancestors", sample_time="sample_time"
    )
    return config.Config(
        sources={"ancestors": anc_src},
        ancestors=[
            config.AncestorsConfig(
                name="ancestors", path=ancestor_store, sources=["test"]
            )
        ],
        match=config.MatchConfig(
            sources={
                "ancestors": config.MatchSourceConfig(
                    node_flags=0, create_individuals=create_individuals
                ),
            },
            output="reference.trees",
        ),
        ancestral_state=_anc_state(sample_store),
    )


def _reference_config(sample_store, reference_ts_path, **kwargs):
    """Config that matches the samples against a reference ts."""
    return config.Config(
        sources={"test": config.Source(path=sample_store, name="test")},
        ancestors=[],
        match=config.MatchConfig(
            sources={"test": config.MatchSourceConfig()},
            output="output.trees",
            reference_ts=reference_ts_path,
        ),
        ancestral_state=_anc_state(sample_store),
        **kwargs,
    )


def _infer_ancestors(sample_store):
    anc_cfg = config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
    return ancestors.infer_ancestors(
        config.Source(path=sample_store, name="test"), anc_cfg, _anc_state(sample_store)
    )


def _edge_set(ts):
    return set(zip(ts.edges_left, ts.edges_right, ts.edges_parent, ts.edges_child))


def _mutation_set(ts):
    return set(zip(ts.mutations_site, ts.mutations_node, ts.mutations_derived_state))


class TestReferenceTs:
    """Matching samples against a reference tree sequence."""

    def _setup(self, tmp_path, random_seed=1, ploidy=1, create_individuals=False):
        """
        Return (sample_store, ancestor_store, reference_ts, reference_ts_path)
        where reference_ts is the result of matching the ancestors only.
        """
        sim_ts = _simulate(num_samples=6, ploidy=ploidy, random_seed=random_seed)
        sample_store = helpers.ts_to_sample_vcz(sim_ts)
        ancestor_store = _infer_ancestors(sample_store)
        anc_cfg = _ancestors_only_config(
            sample_store, ancestor_store, create_individuals=create_individuals
        )
        reference_ts = pipeline.match(anc_cfg)
        reference_ts_path = tmp_path / "reference.trees"
        reference_ts.dump(reference_ts_path)
        return sample_store, ancestor_store, reference_ts, reference_ts_path

    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    @pytest.mark.parametrize("ploidy", [1, 2])
    def test_equivalent_to_single_match(self, tmp_path, random_seed, ploidy):
        """Matching against an ancestors-only reference gives the same
        result as matching ancestors and samples in one pass."""
        sample_store, ancestor_store, _, reference_ts_path = self._setup(
            tmp_path, random_seed=random_seed, ploidy=ploidy
        )
        single_ts = pipeline.match(_make_config(sample_store, ancestor_store))
        cfg = _reference_config(sample_store, reference_ts_path)
        two_step_ts = pipeline.match(cfg)
        assert two_step_ts.num_samples > 0
        two_step_ts.tables.assert_equals(single_ts.tables, ignore_provenance=True)

    def test_does_not_use_root_ts(self, tmp_path):
        """The output is built on the reference, not on a new root ts."""
        sample_store, _, reference_ts, reference_ts_path = self._setup(tmp_path)
        out_ts = pipeline.match(_reference_config(sample_store, reference_ts_path))
        assert out_ts.num_nodes > reference_ts.num_nodes
        np.testing.assert_array_equal(
            out_ts.nodes_time[: reference_ts.num_nodes], reference_ts.nodes_time
        )

    def test_reference_ts_argument(self, tmp_path):
        """A reference_ts argument is used in place of the config path."""
        sample_store, _, reference_ts, reference_ts_path = self._setup(tmp_path)
        expected = pipeline.match(_reference_config(sample_store, reference_ts_path))
        cfg = _reference_config(sample_store, tmp_path / "does-not-exist.trees")
        out_ts = pipeline.match(cfg, reference_ts=reference_ts)
        out_ts.tables.assert_equals(expected.tables, ignore_provenance=True)

    def test_reference_ts_argument_overrides_config(self, tmp_path):
        """The argument wins over a different tree sequence in the config."""
        sample_store, _, reference_ts, reference_ts_path = self._setup(tmp_path)
        tables = reference_ts.dump_tables()
        tables.metadata = {**reference_ts.metadata, "marker": "argument"}
        marked_ts = tables.tree_sequence()
        cfg = _reference_config(sample_store, reference_ts_path)
        out_ts = pipeline.match(cfg, reference_ts=marked_ts)
        assert out_ts.metadata["marker"] == "argument"

    def test_reference_is_kept(self, tmp_path):
        """Nodes, sites, edges and mutations of the reference are kept."""
        sample_store, _, reference_ts, reference_ts_path = self._setup(
            tmp_path, ploidy=2
        )
        out_ts = pipeline.match(_reference_config(sample_store, reference_ts_path))
        num_ref_nodes = reference_ts.num_nodes
        ref_nodes = reference_ts.tables.nodes
        out_nodes = out_ts.tables.nodes
        np.testing.assert_array_equal(out_nodes.time[:num_ref_nodes], ref_nodes.time)
        np.testing.assert_array_equal(out_nodes.flags[:num_ref_nodes], ref_nodes.flags)
        reference_ts.tables.sites.assert_equals(out_ts.tables.sites)
        assert out_ts.metadata == reference_ts.metadata
        assert _edge_set(reference_ts) <= _edge_set(out_ts)
        assert _mutation_set(reference_ts) <= _mutation_set(out_ts)

    def test_new_sample_nodes(self, tmp_path):
        """Each sample haplotype becomes a new sample node with metadata."""
        sample_store, _, reference_ts, reference_ts_path = self._setup(
            tmp_path, ploidy=2
        )
        out_ts = pipeline.match(_reference_config(sample_store, reference_ts_path))
        sample_ids = [str(x) for x in sample_store["sample_id"][:]]
        new_nodes = list(out_ts.nodes())[reference_ts.num_nodes :]
        assert len(new_nodes) == 2 * len(sample_ids)
        assert out_ts.num_samples == len(new_nodes)
        for node in new_nodes:
            assert node.is_sample()
            assert node.time == 0
            assert node.metadata["source"] == "test"
            assert node.metadata["sample_id"] in sample_ids
        assert out_ts.num_individuals == len(sample_ids)

    def test_individual_ids_offset(self, tmp_path):
        """New individuals are appended after those in the reference."""
        sample_store, _, reference_ts, reference_ts_path = self._setup(
            tmp_path, ploidy=2, create_individuals=True
        )
        num_ref_individuals = reference_ts.num_individuals
        assert num_ref_individuals > 0
        out_ts = pipeline.match(_reference_config(sample_store, reference_ts_path))

        reference_ts.tables.individuals.assert_equals(
            out_ts.tables.individuals[:num_ref_individuals]
        )
        sample_ids = [str(x) for x in sample_store["sample_id"][:]]
        new_individuals = list(out_ts.individuals())[num_ref_individuals:]
        assert [ind.metadata["sample_id"] for ind in new_individuals] == sample_ids
        for node in list(out_ts.nodes())[reference_ts.num_nodes :]:
            assert node.individual >= num_ref_individuals
            ind = out_ts.individual(node.individual)
            assert ind.metadata["source"] == "test"
            assert ind.metadata["sample_id"] == node.metadata["sample_id"]
        # Reference nodes keep their individuals
        np.testing.assert_array_equal(
            out_ts.nodes_individual[: reference_ts.num_nodes],
            reference_ts.nodes_individual,
        )

    def test_population_ids_offset(self, tmp_path):
        """New populations are appended after those in the reference."""
        sample_store = helpers.make_sample_vcz(
            genotypes=np.array(
                [[[0], [1], [1]], [[1], [0], [1]], [[1], [1], [0]]], dtype=np.int8
            ),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A", "A"]),
            sequence_length=1000,
            sample_population=np.array(["pop_A", "pop_B", "pop_A"]),
        )
        ancestor_store = _infer_ancestors(sample_store)
        reference_ts = pipeline.match(
            _ancestors_only_config(sample_store, ancestor_store)
        )
        tables = reference_ts.dump_tables()
        tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.add_row(metadata={"name": "existing"})
        reference_ts_path = tmp_path / "reference.trees"
        tables.tree_sequence().dump(reference_ts_path)

        cfg = _reference_config(
            sample_store,
            reference_ts_path,
            individual_metadata=config.IndividualMetadataConfig(
                population="sample_population"
            ),
        )
        out_ts = pipeline.match(cfg)
        pop_names = [pop.metadata["name"] for pop in out_ts.populations()]
        assert pop_names == ["existing", "pop_A", "pop_B"]
        sample_ids = [str(x) for x in sample_store["sample_id"][:]]
        sample_pops = [str(x) for x in sample_store["sample_population"][:]]
        expected_pop = dict(zip(sample_ids, sample_pops))
        for node_id in out_ts.samples():
            node = out_ts.node(node_id)
            pop = out_ts.population(node.population)
            assert pop.metadata["name"] == expected_pop[node.metadata["sample_id"]]

    def test_reference_allele_not_in_samples(self, tmp_path):
        """Reference mutations with alleles that no match source has are
        encoded, and the samples are still matched exactly."""
        sample_store, _, reference_ts, _ = self._setup(tmp_path, random_seed=4)
        assert reference_ts.num_mutations > 0
        tables = reference_ts.dump_tables()
        tables.mutations.packset_derived_state(["G"] * tables.mutations.num_rows)
        reference_ts = tables.tree_sequence()
        reference_ts_path = tmp_path / "reference_g.trees"
        reference_ts.dump(reference_ts_path)

        out_ts = pipeline.match(_reference_config(sample_store, reference_ts_path))
        assert _mutation_set(reference_ts) <= _mutation_set(out_ts)
        # The sample haplotypes are reproduced exactly at every site
        sample_positions = sample_store["variant_position"][:]
        sample_alleles = sample_store["variant_allele"][:]
        sample_gt = sample_store["call_genotype"][:, :, 0]
        for var in out_ts.variants():
            row = int(np.searchsorted(sample_positions, var.site.position))
            expected = sample_alleles[row][sample_gt[row]]
            np.testing.assert_array_equal(np.array(var.states()), expected)

    def test_sequence_length_mismatch(self, tmp_path):
        sample_store, _, reference_ts, _ = self._setup(tmp_path)
        tables = reference_ts.dump_tables()
        tables.sequence_length = reference_ts.sequence_length + 1
        cfg = _reference_config(sample_store, "unused.trees")
        with pytest.raises(ValueError, match="sequence length"):
            pipeline.match(cfg, reference_ts=tables.tree_sequence())

    def test_reference_nodes_not_older(self, tmp_path):
        """A reference that already holds time-0 samples is rejected."""
        sample_store, ancestor_store, _, _ = self._setup(tmp_path)
        full_ts = pipeline.match(_make_config(sample_store, ancestor_store))
        cfg = _reference_config(sample_store, "unused.trees")
        with pytest.raises(ValueError, match="must be older"):
            pipeline.match(cfg, reference_ts=full_ts)

    def test_non_integer_positions(self, tmp_path):
        sample_store, _, reference_ts, _ = self._setup(tmp_path)
        tables = reference_ts.dump_tables()
        tables.sites.position = tables.sites.position + 0.5
        cfg = _reference_config(sample_store, "unused.trees")
        with pytest.raises(ValueError, match="integers"):
            pipeline.match(cfg, reference_ts=tables.tree_sequence())

    def test_workdir(self, tmp_path):
        """Matching against a reference with a workdir gives the same result,
        and the stored match jobs use the reference sites and offset ids."""
        sample_store, _, reference_ts, reference_ts_path = self._setup(
            tmp_path, ploidy=2, create_individuals=True
        )
        expected = pipeline.match(_reference_config(sample_store, reference_ts_path))
        cfg = _reference_config(sample_store, reference_ts_path)
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        out_ts = pipeline.match(cfg)
        out_ts.tables.assert_equals(expected.tables, ignore_provenance=True)

        jobs = json.loads((workdir / "match-jobs.json").read_text())
        positions = reference_ts.sites_position
        for job in jobs:
            assert job["start_position"] == positions[0]
            assert job["end_position"] == positions[-1] + 1
            assert job["individual_id"] >= reference_ts.num_individuals

        # Re-running resumes from the final checkpoint
        resumed_ts = pipeline.match(cfg)
        resumed_ts.tables.assert_equals(out_ts.tables)

    def test_workdir_group_stop_zero_and_resume(self, tmp_path):
        """Resuming with no checkpoint appends the new individuals to the
        reference."""
        sample_store, _, _, reference_ts_path = self._setup(
            tmp_path, ploidy=2, create_individuals=True
        )
        expected = pipeline.match(_reference_config(sample_store, reference_ts_path))
        cfg = _reference_config(sample_store, reference_ts_path)
        cfg.match.workdir = str(tmp_path / "wd")
        pipeline.match(cfg, group_stop=0)
        resumed_ts = pipeline.match(cfg)
        resumed_ts.tables.assert_equals(expected.tables, ignore_provenance=True)

    def test_compute_groups_json(self, tmp_path):
        """Groups for a reference-only config use the reference sites."""
        sample_store, _, reference_ts, reference_ts_path = self._setup(
            tmp_path, create_individuals=True
        )
        cfg = _reference_config(sample_store, reference_ts_path)
        jobs = json.loads(pipeline.compute_groups_json(cfg))
        assert len(jobs) == len(sample_store["sample_id"][:])
        positions = reference_ts.sites_position
        for job in jobs:
            assert job["source"] == "test"
            assert job["start_position"] == positions[0]
            assert job["end_position"] == positions[-1] + 1
            assert job["individual_id"] >= reference_ts.num_individuals

    def test_run_without_ancestors(self, tmp_path):
        """run() with a reference and no [[ancestors]] skips ancestor inference."""
        sample_store, _, _, reference_ts_path = self._setup(tmp_path)
        cfg = _reference_config(
            sample_store,
            reference_ts_path,
            post_process=config.PostProcessConfig(),
        )
        out_ts = pipeline.run(cfg)
        expected = pipeline.post_process(pipeline.match(cfg), cfg)
        out_ts.tables.assert_equals(expected.tables, ignore_provenance=True)
        assert out_ts.num_samples == len(sample_store["sample_id"][:])


# ---------------------------------------------------------------------------
# TestAugmentSites
# ---------------------------------------------------------------------------


def _run_pipeline_no_augment(sample_store):
    """Run the full pipeline (no augment) and return (ts, sample_store)."""
    src = config.Source(path=sample_store, name="test")
    anc_src = config.Source(path=None, name="ancestors", sample_time="sample_time")
    cfg = config.Config(
        sources={"test": src, "ancestors": anc_src},
        ancestors=[
            config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ],
        match=config.MatchConfig(
            sources={
                "ancestors": config.MatchSourceConfig(
                    node_flags=0, create_individuals=False
                ),
                "test": config.MatchSourceConfig(),
            },
            output="output.trees",
        ),
        post_process=config.PostProcessConfig(split_ultimate=False, erase_flanks=True),
        ancestral_state=_anc_state(sample_store),
    )
    return pipeline.run(cfg), cfg


class TestAugmentSites:
    def _make_two_stores(self):
        """
        Create two VCZ stores from a simulation:
        - main_store: all non-singleton sites (for inference)
        - singleton_store: only singleton sites (for augmenting)

        Returns (main_store, singleton_store, all_positions).
        """
        ts = _simulate(
            num_samples=6, sequence_length=100_000, mutation_rate=2e-8, random_seed=100
        )
        assert ts.num_sites > 5, "Need enough sites for the test"

        # Identify singletons
        sample_nodes = ts.samples()
        singleton_sites = []
        non_singleton_sites = []
        for var in ts.variants(samples=sample_nodes):
            ac = np.sum(var.genotypes != 0)
            if ac == 1:
                singleton_sites.append(int(var.site.id))
            else:
                non_singleton_sites.append(int(var.site.id))

        if len(singleton_sites) == 0 or len(non_singleton_sites) < 2:
            # Fallback: manually create stores with known content
            main_store, sing_store, _ = self._make_hand_constructed_stores()
            return main_store, sing_store

        # Build main store (non-singletons)
        main_store = helpers.ts_to_sample_vcz(ts)

        # Build singleton-only store with same samples
        all_positions = np.array(
            [int(ts.site(s).position) for s in range(ts.num_sites)], dtype=np.int32
        )
        sing_positions = np.array(
            [int(ts.site(s).position) for s in singleton_sites], dtype=np.int32
        )

        return main_store, sing_positions, all_positions, ts

    def _make_hand_constructed_stores(self):
        """Hand-construct stores for testing augment_sites."""
        # Main store: 2 sites (non-singletons)
        main_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        # Singleton store: 1 site
        sing_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [0], [1]]], dtype=np.int8),
            positions=np.array([200], dtype=np.int32),
            alleles=np.array([["A", "G"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )
        # Combined annotation store covering all positions
        ann_store = helpers.make_sample_vcz(
            genotypes=np.zeros((3, 1, 1), dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "G"], ["C", "G"]]),
            ancestral_state=np.array(["A", "A", "C"]),
            sequence_length=1000,
        )
        return main_store, sing_store, ann_store

    def test_adds_new_sites(self):
        """Augmenting with a source adds sites not in the original TS."""
        # Use hand-constructed data for determinism
        main_store, sing_store, ann_store = self._make_hand_constructed_stores()

        # Run pipeline on main_store
        ts, _ = _run_pipeline_no_augment(main_store)
        original_num_sites = ts.num_sites

        # Augment with singleton store
        cfg = config.Config(
            sources={
                "test": config.Source(path=main_store, name="test"),
                "singletons": config.Source(path=sing_store, name="singletons"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=config.AugmentSitesConfig(sources=["singletons"]),
            ancestral_state=_anc_state(ann_store),
        )
        aug_ts = pipeline.augment_sites(ts, cfg)
        assert aug_ts.num_sites > original_num_sites
        # Position 200 should now be present
        aug_positions = set(aug_ts.sites_position)
        assert 200.0 in aug_positions

    def test_skips_existing_sites(self):
        """Sites already in the TS are not duplicated."""
        main_store, sing_store, ann_store = self._make_hand_constructed_stores()
        ts, _ = _run_pipeline_no_augment(main_store)

        # Create a source that has the SAME positions as the main store
        cfg = config.Config(
            sources={
                "test": config.Source(path=main_store, name="test"),
                "same": config.Source(path=main_store, name="same"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=config.AugmentSitesConfig(sources=["same"]),
            ancestral_state=_anc_state(ann_store),
        )
        aug_ts = pipeline.augment_sites(ts, cfg)
        # No new sites should be added
        assert aug_ts.num_sites == ts.num_sites

    def test_respects_sequence_intervals(self):
        """Sites outside sequence_intervals are skipped."""
        main_store, sing_store, ann_store = self._make_hand_constructed_stores()
        ts, _ = _run_pipeline_no_augment(main_store)

        # Create a singleton store with a site outside the intervals
        far_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [0], [1]]], dtype=np.int8),
            positions=np.array([999_999], dtype=np.int32),
            alleles=np.array([["A", "G"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1_000_000,
        )

        cfg = config.Config(
            sources={
                "test": config.Source(path=main_store, name="test"),
                "far": config.Source(path=far_store, name="far"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=config.AugmentSitesConfig(sources=["far"]),
            ancestral_state=_anc_state(main_store),
        )
        aug_ts = pipeline.augment_sites(ts, cfg)
        # The far site should be outside intervals, so no new sites
        assert aug_ts.num_sites == ts.num_sites

    def test_no_sources_is_noop(self):
        """Empty sources list returns TS unchanged."""
        main_store, _, _ = self._make_hand_constructed_stores()
        ts, _ = _run_pipeline_no_augment(main_store)

        cfg = config.Config(
            sources={
                "test": config.Source(path=main_store, name="test"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=config.AugmentSitesConfig(sources=[]),
            ancestral_state=_anc_state(main_store),
        )
        aug_ts = pipeline.augment_sites(ts, cfg)
        assert aug_ts.num_sites == ts.num_sites
        assert aug_ts.num_nodes == ts.num_nodes

    def test_sample_mapping_diploid(self):
        """Correct mapping for diploid data."""
        # 2 non-singleton sites for inference, 3 diploid samples
        main_store = helpers.make_sample_vcz(
            genotypes=np.array(
                [[[0, 0], [1, 0], [1, 1]], [[1, 1], [0, 0], [1, 0]]], dtype=np.int8
            ),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        # Extra site at position 200, same 3 diploid samples
        extra_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0, 0], [0, 0], [1, 0]]], dtype=np.int8),
            positions=np.array([200], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )
        # Combined annotation store
        ann_store = helpers.make_sample_vcz(
            genotypes=np.zeros((3, 1, 1), dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "A", "C"]),
            sequence_length=1000,
        )

        ts, _ = _run_pipeline_no_augment(main_store)

        cfg = config.Config(
            sources={
                "test": config.Source(path=main_store, name="test"),
                "extra": config.Source(path=extra_store, name="extra"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=config.AugmentSitesConfig(sources=["extra"]),
            ancestral_state=_anc_state(ann_store),
        )

        aug_ts = pipeline.augment_sites(ts, cfg)
        assert aug_ts.num_sites > ts.num_sites
        assert 200.0 in set(aug_ts.sites_position)

    def test_multiallelic(self):
        """Multi-allelic sites are placed correctly."""
        main_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        # Multi-allelic site at position 200
        multi_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1], [2]]], dtype=np.int8),
            positions=np.array([200], dtype=np.int32),
            alleles=np.array([["A", "T", "G"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )
        # Combined annotation store
        ann_store = helpers.make_sample_vcz(
            genotypes=np.zeros((3, 1, 1), dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "A", "C"]),
            sequence_length=1000,
        )

        ts, _ = _run_pipeline_no_augment(main_store)

        cfg = config.Config(
            sources={
                "test": config.Source(path=main_store, name="test"),
                "multi": config.Source(path=multi_store, name="multi"),
                "ancestors": config.Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=config.AugmentSitesConfig(sources=["multi"]),
            ancestral_state=_anc_state(ann_store),
        )
        aug_ts = pipeline.augment_sites(ts, cfg)
        assert aug_ts.num_sites > ts.num_sites
        site = aug_ts.site(position=200.0)
        assert len(site.mutations) > 0

    def test_run_integration(self):
        """Full run() with augment_sites configured end-to-end."""
        main_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        extra_store = helpers.make_sample_vcz(
            genotypes=np.array([[[0], [0], [1]]], dtype=np.int8),
            positions=np.array([200], dtype=np.int32),
            alleles=np.array([["A", "G"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )
        # Combined annotation store
        ann_store = helpers.make_sample_vcz(
            genotypes=np.zeros((3, 1, 1), dtype=np.int8),
            positions=np.array([100, 200, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "G"], ["C", "G"]]),
            ancestral_state=np.array(["A", "A", "C"]),
            sequence_length=1000,
        )

        src = config.Source(path=main_store, name="test")
        extra_src = config.Source(path=extra_store, name="extra")
        anc_src = config.Source(path=None, name="ancestors", sample_time="sample_time")
        cfg = config.Config(
            sources={
                "test": src,
                "extra": extra_src,
                "ancestors": anc_src,
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": config.MatchSourceConfig(),
                },
                output="output.trees",
            ),
            post_process=config.PostProcessConfig(
                split_ultimate=False, erase_flanks=True
            ),
            augment_sites=config.AugmentSitesConfig(sources=["extra"]),
            ancestral_state=_anc_state(ann_store),
        )
        out_ts = pipeline.run(cfg)
        assert out_ts.num_nodes > 0
        # Should have the augmented site
        positions = set(out_ts.sites_position)
        assert 200.0 in positions


class TestEraseFlanks:
    def _make_ts_with_json_metadata(self, metadata=None):
        """Build a ts with JSON metadata schema so ts.metadata returns a dict."""
        tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
        tables.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.metadata = metadata if metadata is not None else {}
        return tables.tree_sequence()

    def test_no_intervals_returns_unchanged(self):
        """_erase_flanks returns ts unchanged when metadata has no sequence_intervals."""
        ts = self._make_ts_with_json_metadata()
        result = pipeline._erase_flanks(ts)
        assert result.num_edges == ts.num_edges
        assert result.num_nodes == ts.num_nodes

    def test_empty_metadata(self):
        """_erase_flanks handles ts.metadata being empty dict."""
        ts = self._make_ts_with_json_metadata({})
        result = pipeline._erase_flanks(ts)
        assert result.num_edges == ts.num_edges


class TestAugmentSitesEdgeCases:
    def test_missing_node_metadata(self):
        """augment_sites raises when sample nodes lack required metadata."""
        # Build a ts with JSON node metadata schema but nodes missing keys
        tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        md = [{}] * len(tables.nodes)
        tables.nodes.packset_metadata(
            [tables.nodes.metadata_schema.encode_row(m) for m in md]
        )
        tables.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.metadata = {}
        ts = tables.tree_sequence()
        cfg = config.Config(
            sources={"s": config.Source(name="s", path="s.vcz")},
            ancestors=[],
            match=config.MatchConfig(
                sources={}, output="out.trees", reference_ts="ref.trees"
            ),
            ancestral_state=config.AncestralState(path="ann.vcz", field="x"),
            augment_sites=config.AugmentSitesConfig(sources=["s"]),
        )
        with pytest.raises(ValueError, match="missing required metadata key"):
            pipeline.augment_sites(ts, cfg)

    def test_no_augment_config(self):
        """augment_sites returns ts unchanged when no augment config."""
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        cfg = config.Config(
            sources={},
            ancestors=[],
            match=config.MatchConfig(
                sources={}, output="out.trees", reference_ts="ref.trees"
            ),
            ancestral_state=config.AncestralState(path="ann.vcz", field="x"),
            augment_sites=None,
        )
        result = pipeline.augment_sites(ts, cfg)
        assert result.equals(ts)
