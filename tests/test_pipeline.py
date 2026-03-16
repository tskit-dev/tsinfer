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

Uses ts_to_sample_vcz(msprime_ts) to create synthetic inputs and verifies
the pipeline produces valid tree sequences.
"""

from __future__ import annotations

import msprime
import numpy as np
from helpers import make_sample_vcz, ts_to_sample_vcz

from tsinfer.ancestors import infer_ancestors
from tsinfer.config import (
    AncestorsConfig,
    Config,
    IndividualMetadataConfig,
    MatchConfig,
    PostProcessConfig,
    Source,
)
from tsinfer.grouping import compute_groups_json
from tsinfer.pipeline import match, post_process, run

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


def _make_config(sample_store, ancestor_store, recombination_rate=1e-4):
    """Build a Config suitable for match() from in-memory stores."""
    src = Source(path=sample_store, name="test")
    return Config(
        sources={"test": src},
        ancestors=AncestorsConfig(path=ancestor_store, sources=["test"]),
        match=MatchConfig(
            sources=["test"],
            output="output.trees",
            recombination_rate=recombination_rate,
        ),
    )


def _make_config_for_run(sample_store, recombination_rate=1e-4):
    """Build a Config suitable for run() from in-memory sample store."""
    src = Source(path=sample_store, name="test")
    return Config(
        sources={"test": src},
        ancestors=AncestorsConfig(path=None, sources=["test"]),
        match=MatchConfig(
            sources=["test"],
            output="output.trees",
            recombination_rate=recombination_rate,
        ),
    )


def _infer_and_match(sim_ts, recombination_rate=1e-4):
    """Helper: simulate → sample VCZ → infer ancestors → match → return output ts."""
    sample_store = ts_to_sample_vcz(sim_ts)
    anc_cfg = AncestorsConfig(path=None, sources=["test"])
    ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
    cfg = _make_config(sample_store, ancestor_store, recombination_rate)
    return match(cfg)


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
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        out_ts = match(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_sites == 2


# ---------------------------------------------------------------------------
# TestPostProcess
# ---------------------------------------------------------------------------


class TestPostProcess:
    def test_simplify_reduces_nodes(self):
        sim_ts = _simulate(num_samples=6, random_seed=10)
        matched_ts = _infer_and_match(sim_ts)
        cfg = Config(
            sources={"test": Source(path=None, name="test")},
            ancestors=AncestorsConfig(path=None, sources=["test"]),
            match=MatchConfig(
                sources=["test"],
                output="out.trees",
                recombination_rate=1e-4,
            ),
            post_process=PostProcessConfig(split_ultimate=False, erase_flanks=False),
        )
        pp_ts = post_process(matched_ts, cfg)
        # Simplify should remove unused nodes
        assert pp_ts.num_nodes <= matched_ts.num_nodes

    def test_no_post_process_config_returns_unchanged(self):
        sim_ts = _simulate(num_samples=4, random_seed=11)
        matched_ts = _infer_and_match(sim_ts)
        cfg = Config(
            sources={"test": Source(path=None, name="test")},
            ancestors=AncestorsConfig(path=None, sources=["test"]),
            match=MatchConfig(
                sources=["test"],
                output="out.trees",
                recombination_rate=1e-4,
            ),
            post_process=None,
        )
        pp_ts = post_process(matched_ts, cfg)
        assert pp_ts.num_nodes == matched_ts.num_nodes

    def test_erase_flanks(self):
        sim_ts = _simulate(num_samples=6, random_seed=12)
        matched_ts = _infer_and_match(sim_ts)
        cfg = Config(
            sources={"test": Source(path=None, name="test")},
            ancestors=AncestorsConfig(path=None, sources=["test"]),
            match=MatchConfig(
                sources=["test"],
                output="out.trees",
                recombination_rate=1e-4,
            ),
            post_process=PostProcessConfig(split_ultimate=False, erase_flanks=True),
        )
        pp_ts = post_process(matched_ts, cfg)
        assert pp_ts.num_nodes > 0


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------


class TestRun:
    def test_full_pipeline_haploid(self):
        sim_ts = _simulate(num_samples=4, random_seed=20)
        sample_store = ts_to_sample_vcz(sim_ts)
        cfg = _make_config_for_run(sample_store)
        out_ts = run(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_sites > 0

    def test_full_pipeline_diploid(self):
        sim_ts = _simulate(num_samples=4, ploidy=2, random_seed=21)
        sample_store = ts_to_sample_vcz(sim_ts)
        cfg = _make_config_for_run(sample_store)
        out_ts = run(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_individuals > 0

    def test_full_pipeline_with_post_process(self):
        sim_ts = _simulate(num_samples=6, random_seed=22)
        sample_store = ts_to_sample_vcz(sim_ts)
        src = Source(path=sample_store, name="test")
        cfg = Config(
            sources={"test": src},
            ancestors=AncestorsConfig(path=None, sources=["test"]),
            match=MatchConfig(
                sources=["test"],
                output="output.trees",
                recombination_rate=1e-4,
            ),
            post_process=PostProcessConfig(split_ultimate=False, erase_flanks=True),
        )
        out_ts = run(cfg)
        assert out_ts.num_nodes > 0

    def test_hand_constructed(self):
        """Full pipeline with a hand-constructed sample VCZ."""
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = run(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_sites == 2


# ---------------------------------------------------------------------------
# TestNodeMetadata
# ---------------------------------------------------------------------------


class TestNodeMetadata:
    def test_nodes_have_metadata(self):
        """Nodes should have metadata with source and sample_id."""
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = run(cfg)
        # At least some nodes should have metadata
        has_metadata = False
        for node in out_ts.nodes():
            if node.metadata:
                has_metadata = True
                break
        assert has_metadata

    def test_ancestor_nodes_have_source(self):
        """Ancestor nodes should have source='ancestors'."""
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = run(cfg)
        ancestor_found = False
        for node in out_ts.nodes():
            if node.metadata and node.metadata.get("source") == "ancestors":
                ancestor_found = True
                assert "sample_id" in node.metadata
                break
        assert ancestor_found

    def test_sample_nodes_have_ploidy_index(self):
        """Sample nodes should have ploidy_index in metadata."""
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = run(cfg)
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
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        src = Source(path=sample_store, name="test")
        cfg = Config(
            sources={"test": src},
            ancestors=AncestorsConfig(path=None, sources=["test"]),
            match=MatchConfig(
                sources=["test"],
                output="output.trees",
                recombination_rate=1e-4,
            ),
            individual_metadata=IndividualMetadataConfig(
                fields={"sample_id": "sample_id"},
            ),
        )
        out_ts = run(cfg)
        assert out_ts.num_individuals > 0
        for ind in out_ts.individuals():
            if ind.metadata:
                assert "sample_id" in ind.metadata

    def test_population_assignment(self):
        """Individuals should be assigned to populations from config."""
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
            sample_population=np.array(["pop_A", "pop_B"]),
        )
        src = Source(path=sample_store, name="test")
        cfg = Config(
            sources={"test": src},
            ancestors=AncestorsConfig(path=None, sources=["test"]),
            match=MatchConfig(
                sources=["test"],
                output="output.trees",
                recombination_rate=1e-4,
            ),
            individual_metadata=IndividualMetadataConfig(
                population="sample_population",
            ),
        )
        out_ts = run(cfg)
        assert out_ts.num_populations > 0
        # Check population metadata
        pop_names = [pop.metadata.get("name") for pop in out_ts.populations()]
        assert "pop_A" in pop_names or "pop_B" in pop_names

    def test_no_individual_metadata_config(self):
        """Without individual_metadata config, no individual metadata is set."""
        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        cfg = _make_config_for_run(sample_store)
        out_ts = run(cfg)
        # No individual metadata config → no individual metadata schema
        assert out_ts.num_populations == 0


# ---------------------------------------------------------------------------
# Logging and progress
# ---------------------------------------------------------------------------


class TestPipelineLogging:
    def test_run_logs_key_messages(self, caplog):
        """Key INFO messages from run()."""
        import logging

        sim_ts = _simulate(num_samples=4, random_seed=30)
        sample_store = ts_to_sample_vcz(sim_ts)
        cfg = _make_config_for_run(sample_store)
        with caplog.at_level(logging.INFO, logger="tsinfer"):
            run(cfg)

        messages = caplog.text
        assert "Starting full pipeline" in messages
        assert "Pipeline complete" in messages

    def test_match_logs_haplotype_count(self, caplog):
        """match() logs haplotype and site counts."""
        import logging

        sim_ts = _simulate(num_samples=4, random_seed=31)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        with caplog.at_level(logging.INFO, logger="tsinfer.pipeline"):
            match(cfg)

        assert "Match:" in caplog.text


class TestPipelineProgress:
    def test_run_with_progress(self):
        """run(progress=True) does not crash."""
        sim_ts = _simulate(num_samples=4, random_seed=32)
        sample_store = ts_to_sample_vcz(sim_ts)
        cfg = _make_config_for_run(sample_store)
        out_ts = run(cfg, progress=True)
        assert out_ts.num_nodes > 0

    def test_match_with_progress(self):
        """match(progress=True) does not crash."""
        sim_ts = _simulate(num_samples=4, random_seed=33)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        out_ts = match(cfg, progress=True)
        assert out_ts.num_nodes > 0


# ---------------------------------------------------------------------------
# TestComputeGroupsJson
# ---------------------------------------------------------------------------


class TestComputeGroupsJson:
    def test_basic(self):
        """compute_groups_json returns a flat JSON array of records."""
        import json

        sim_ts = _simulate(num_samples=4, random_seed=40)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
        assert isinstance(records, list)
        assert len(records) > 0

    def test_virtual_root_is_first_record(self):
        """First record should be the virtual root in group 0."""
        import json

        sim_ts = _simulate(num_samples=4, random_seed=41)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
        rec0 = records[0]
        assert rec0["haplotype_index"] == 0
        assert rec0["time"] == 1.0
        assert rec0["group"] == 0
        assert rec0["sample_id"] == "virtual_root"

    def test_all_indices_covered(self):
        """All haplotype indices should appear exactly once."""
        import json

        sim_ts = _simulate(num_samples=6, random_seed=42)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
        all_indices = {r["haplotype_index"] for r in records}
        assert all_indices == set(range(len(records)))

    def test_hand_constructed(self):
        """compute_groups_json with a hand-constructed VCZ."""
        import json

        sample_store = make_sample_vcz(
            genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
            positions=np.array([100, 200], dtype=np.int32),
            alleles=np.array([["A", "T"], ["A", "T"]]),
            ancestral_state=np.array(["A", "A"]),
            sequence_length=1000,
        )
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
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
        }
        for rec in records:
            assert set(rec.keys()) == expected_keys
            assert isinstance(rec["time"], float)
            assert isinstance(rec["group"], int)


# ---------------------------------------------------------------------------
# TestMatchWithGroups
# ---------------------------------------------------------------------------


class TestMatchWithGroups:
    def test_match_with_groups_file(self, tmp_path):
        """match() with cfg.match.groups set produces a valid tree sequence."""

        sim_ts = _simulate(num_samples=4, random_seed=50)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)

        # Compute groups and write to file
        groups_json = compute_groups_json(cfg)
        groups_path = tmp_path / "groups.json"
        groups_path.write_text(groups_json)

        # Set groups path on config
        cfg.match.groups = str(groups_path)
        out_ts = match(cfg)
        assert out_ts.num_nodes > 0
        assert out_ts.num_edges > 0
        assert out_ts.num_sites > 0

    def test_match_without_groups_still_works(self):
        """match() with cfg.match.groups=None computes groups internally."""
        sim_ts = _simulate(num_samples=4, random_seed=51)
        out_ts = _infer_and_match(sim_ts)
        assert out_ts.num_nodes > 0
        assert out_ts.num_edges > 0
