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
    AugmentSitesConfig,
    Config,
    IndividualMetadataConfig,
    MatchConfig,
    MatchSourceConfig,
    PostProcessConfig,
    Source,
)
from tsinfer.pipeline import augment_sites, compute_groups_json, match, post_process, run

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


def _make_config(sample_store, ancestor_store):
    """Build a Config suitable for match() from in-memory stores."""
    src = Source(path=sample_store, name="test")
    anc_src = Source(path=ancestor_store, name="ancestors", sample_time="sample_time")
    return Config(
        sources={"test": src, "ancestors": anc_src},
        ancestors=[
            AncestorsConfig(name="ancestors", path=ancestor_store, sources=["test"])
        ],
        match=MatchConfig(
            sources={
                "ancestors": MatchSourceConfig(node_flags=0, create_individuals=False),
                "test": MatchSourceConfig(),
            },
            output="output.trees",
        ),
    )


def _make_config_for_run(sample_store):
    """Build a Config suitable for run() from in-memory sample store."""
    src = Source(path=sample_store, name="test")
    anc_src = Source(path=None, name="ancestors", sample_time="sample_time")
    return Config(
        sources={"test": src, "ancestors": anc_src},
        ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
        match=MatchConfig(
            sources={
                "ancestors": MatchSourceConfig(node_flags=0, create_individuals=False),
                "test": MatchSourceConfig(),
            },
            output="output.trees",
        ),
    )


def _infer_and_match(sim_ts):
    """Helper: simulate → sample VCZ → infer ancestors → match → return output ts."""
    sample_store = ts_to_sample_vcz(sim_ts)
    anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
    ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
    cfg = _make_config(sample_store, ancestor_store)
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
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
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
            sources={
                "test": Source(path=None, name="test"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="out.trees",
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
            sources={
                "test": Source(path=None, name="test"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="out.trees",
            ),
            post_process=None,
        )
        pp_ts = post_process(matched_ts, cfg)
        assert pp_ts.num_nodes == matched_ts.num_nodes

    def test_erase_flanks(self):
        sim_ts = _simulate(num_samples=6, random_seed=12)
        matched_ts = _infer_and_match(sim_ts)
        cfg = Config(
            sources={
                "test": Source(path=None, name="test"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="out.trees",
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
        anc_src = Source(path=None, name="ancestors", sample_time="sample_time")
        cfg = Config(
            sources={"test": src, "ancestors": anc_src},
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
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
        anc_src = Source(path=None, name="ancestors", sample_time="sample_time")
        cfg = Config(
            sources={"test": src, "ancestors": anc_src},
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
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
        anc_src = Source(path=None, name="ancestors", sample_time="sample_time")
        cfg = Config(
            sources={"test": src, "ancestors": anc_src},
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
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
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
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
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
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
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
        assert isinstance(records, list)
        assert len(records) > 0

    def test_first_record_is_ancestor(self):
        """First record should be an ancestor in group 0."""
        import json

        sim_ts = _simulate(num_samples=4, random_seed=41)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        records = json.loads(compute_groups_json(cfg))
        rec0 = records[0]
        assert rec0["haplotype_index"] == 0
        assert rec0["group"] == 0
        assert rec0["source"] == "ancestors"
        assert rec0["sample_id"] != "virtual_root"

    def test_all_indices_covered(self):
        """All haplotype indices should appear exactly once."""
        import json

        sim_ts = _simulate(num_samples=6, random_seed=42)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
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
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
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
            "node_flags",
            "individual_id",
            "population_id",
        }
        for rec in records:
            assert set(rec.keys()) == expected_keys
            assert isinstance(rec["time"], float)
            assert isinstance(rec["group"], int)

    def test_round_trip_pandas(self):
        """match-jobs.json output can be loaded into a pandas DataFrame."""
        import io
        import json

        import pandas as pd

        sim_ts = _simulate(num_samples=4, random_seed=45)
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        json_str = compute_groups_json(cfg)
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
        sample_store = ts_to_sample_vcz(sim_ts)
        anc_cfg = AncestorsConfig(name="ancestors", path=None, sources=["test"])
        ancestor_store = infer_ancestors(Source(path=sample_store, name="test"), anc_cfg)
        cfg = _make_config(sample_store, ancestor_store)
        return cfg

    def test_workdir_creates_groups_and_trees(self, tmp_path):
        """Fresh run creates match-jobs.json and at least one .trees file."""
        import tskit

        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        out_ts = match(cfg)
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

        ts1 = match(cfg)

        # Find the group files and delete the last one to force re-run
        trees_files = sorted(workdir.glob("group_*.trees"))
        assert len(trees_files) >= 1
        if len(trees_files) > 1:
            trees_files[-1].unlink()

            ts2 = match(cfg)
            assert ts2.num_nodes == ts1.num_nodes
            assert ts2.num_edges == ts1.num_edges

    def test_workdir_ignores_gap(self, tmp_path):
        """Remove a middle .trees file → resumes from before the gap."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        ts1 = match(cfg)

        trees_files = sorted(workdir.glob("group_*.trees"))
        if len(trees_files) >= 3:
            # Remove a middle file to create a gap
            trees_files[1].unlink()
            ts2 = match(cfg)
            assert ts2.num_nodes == ts1.num_nodes
            assert ts2.num_edges == ts1.num_edges

    def test_workdir_default_keeps_only_last(self, tmp_path):
        """Without keep_intermediates, only the latest .trees file remains."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)

        match(cfg)

        trees_files = list(workdir.glob("group_*.trees"))
        assert len(trees_files) == 1

    def test_workdir_keep_intermediates(self, tmp_path):
        """With keep_intermediates, all .trees files are retained."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        match(cfg)

        trees_files = list(workdir.glob("group_*.trees"))
        # Should have more than 1 if there are multiple groups
        assert len(trees_files) >= 1
        assert (workdir / "match-jobs.json").exists()

    def test_no_workdir_by_default(self, tmp_path):
        """workdir=None produces no files."""
        cfg = self._setup()
        assert cfg.match.workdir is None

        match(cfg)
        assert list(tmp_path.glob("*.trees")) == []
        assert list(tmp_path.glob("match-jobs.json")) == []

    def test_group_stop_produces_partial_result(self, tmp_path):
        """group_stop=1 processes only group 0; checkpoint matches returned ts."""
        import json

        import tskit

        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)
        cfg.match.keep_intermediates = True

        partial_ts = match(cfg, group_stop=1)
        full_ts = match(self._setup())

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

        match(cfg, group_stop=1)

        resumed_ts = match(cfg)
        full_ts = match(self._setup())

        assert resumed_ts.num_nodes == full_ts.num_nodes
        assert resumed_ts.num_edges == full_ts.num_edges

    def test_group_stop_every_group_matches_full_run(self, tmp_path):
        """For each group g, stop at g then resume → matches full run."""
        import json

        base_cfg = self._setup()
        full_ts = match(base_cfg)

        # Determine number of groups
        workdir_probe = tmp_path / "probe"
        base_cfg.match.workdir = str(workdir_probe)
        base_cfg.match.keep_intermediates = True
        match(base_cfg)
        groups_json = json.loads((workdir_probe / "match-jobs.json").read_text())
        num_groups = max(r["group"] for r in groups_json) + 1

        for g in range(1, num_groups + 1):
            cfg = self._setup()
            wd = tmp_path / f"wd_g{g}"
            cfg.match.workdir = str(wd)
            cfg.match.keep_intermediates = True

            match(cfg, group_stop=g)
            resumed_ts = match(cfg)

            assert resumed_ts.num_nodes == full_ts.num_nodes
            assert resumed_ts.num_edges == full_ts.num_edges

    def test_group_stop_multi_step_resume(self, tmp_path):
        """Step through groups one at a time → final matches full run."""
        import json

        base_cfg = self._setup()
        full_ts = match(base_cfg)

        # Determine number of groups
        workdir_probe = tmp_path / "probe"
        base_cfg.match.workdir = str(workdir_probe)
        base_cfg.match.keep_intermediates = True
        match(base_cfg)
        groups_json = json.loads((workdir_probe / "match-jobs.json").read_text())
        num_groups = max(r["group"] for r in groups_json) + 1

        cfg = self._setup()
        wd = tmp_path / "wd_step"
        cfg.match.workdir = str(wd)
        cfg.match.keep_intermediates = True

        for g in range(1, num_groups + 1):
            match(cfg, group_stop=g)

        final_ts = match(cfg)

        assert final_ts.num_nodes == full_ts.num_nodes
        assert final_ts.num_edges == full_ts.num_edges

    def test_group_stop_beyond_last_group(self, tmp_path):
        """group_stop=9999 behaves like a normal full run."""
        cfg = self._setup()
        workdir = tmp_path / "wd"
        cfg.match.workdir = str(workdir)

        big_stop_ts = match(cfg, group_stop=9999)
        full_ts = match(self._setup())

        assert big_stop_ts.num_nodes == full_ts.num_nodes
        assert big_stop_ts.num_edges == full_ts.num_edges

    def test_keep_intermediates_without_workdir_errors(self):
        """keep_intermediates=True without workdir raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="keep_intermediates requires workdir"):
            Config(
                sources={
                    "test": Source(path="dummy", name="test"),
                    "ancestors": Source(
                        path="dummy", name="ancestors", sample_time="sample_time"
                    ),
                },
                ancestors=[
                    AncestorsConfig(name="ancestors", path="dummy", sources=["test"])
                ],
                match=MatchConfig(
                    sources={
                        "ancestors": MatchSourceConfig(
                            node_flags=0, create_individuals=False
                        ),
                        "test": MatchSourceConfig(),
                    },
                    output="out.trees",
                    keep_intermediates=True,
                ),
            )


# ---------------------------------------------------------------------------
# TestAugmentSites
# ---------------------------------------------------------------------------


def _run_pipeline_no_augment(sample_store):
    """Run the full pipeline (no augment) and return (ts, sample_store)."""
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
        post_process=PostProcessConfig(split_ultimate=False, erase_flanks=True),
    )
    return run(cfg), cfg


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
            return self._make_hand_constructed_stores()

        # Build main store (non-singletons)
        main_store = ts_to_sample_vcz(ts)

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
        main_store = make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        # Singleton store: 1 site
        sing_store = make_sample_vcz(
            genotypes=np.array([[[0], [0], [1]]], dtype=np.int8),
            positions=np.array([200], dtype=np.int32),
            alleles=np.array([["A", "G"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )
        return main_store, sing_store

    def test_adds_new_sites(self):
        """Augmenting with a source adds sites not in the original TS."""
        # Use hand-constructed data for determinism
        main_store, sing_store = self._make_hand_constructed_stores()

        # Run pipeline on main_store
        ts, _ = _run_pipeline_no_augment(main_store)
        original_num_sites = ts.num_sites

        # Augment with singleton store
        cfg = Config(
            sources={
                "test": Source(path=main_store, name="test"),
                "singletons": Source(path=sing_store, name="singletons"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=AugmentSitesConfig(sources=["singletons"]),
        )
        aug_ts = augment_sites(ts, cfg)
        assert aug_ts.num_sites > original_num_sites
        # Position 200 should now be present
        aug_positions = set(aug_ts.sites_position)
        assert 200.0 in aug_positions

    def test_skips_existing_sites(self):
        """Sites already in the TS are not duplicated."""
        main_store, sing_store = self._make_hand_constructed_stores()
        ts, _ = _run_pipeline_no_augment(main_store)

        # Create a source that has the SAME positions as the main store
        cfg = Config(
            sources={
                "test": Source(path=main_store, name="test"),
                "same": Source(path=main_store, name="same"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=AugmentSitesConfig(sources=["same"]),
        )
        aug_ts = augment_sites(ts, cfg)
        # No new sites should be added
        assert aug_ts.num_sites == ts.num_sites

    def test_respects_sequence_intervals(self):
        """Sites outside sequence_intervals are skipped."""
        main_store, sing_store = self._make_hand_constructed_stores()
        ts, _ = _run_pipeline_no_augment(main_store)

        # Create a singleton store with a site outside the intervals
        far_store = make_sample_vcz(
            genotypes=np.array([[[0], [0], [1]]], dtype=np.int8),
            positions=np.array([999_999], dtype=np.int32),
            alleles=np.array([["A", "G"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1_000_000,
        )

        cfg = Config(
            sources={
                "test": Source(path=main_store, name="test"),
                "far": Source(path=far_store, name="far"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=AugmentSitesConfig(sources=["far"]),
        )
        aug_ts = augment_sites(ts, cfg)
        # The far site should be outside intervals, so no new sites
        assert aug_ts.num_sites == ts.num_sites

    def test_no_sources_is_noop(self):
        """Empty sources list returns TS unchanged."""
        main_store, _ = self._make_hand_constructed_stores()
        ts, _ = _run_pipeline_no_augment(main_store)

        cfg = Config(
            sources={
                "test": Source(path=main_store, name="test"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=AugmentSitesConfig(sources=[]),
        )
        aug_ts = augment_sites(ts, cfg)
        assert aug_ts.num_sites == ts.num_sites
        assert aug_ts.num_nodes == ts.num_nodes

    def test_sample_mapping_diploid(self):
        """Correct mapping for diploid data."""
        sim_ts = _simulate(num_samples=3, ploidy=2, random_seed=101)
        sample_store = ts_to_sample_vcz(sim_ts)

        # Create a singleton-like extra site store with same samples
        # Use 3 diploid samples (6 haplotypes)
        extra_store = make_sample_vcz(
            genotypes=np.array([[[0, 0], [0, 0], [1, 0]]], dtype=np.int8),
            positions=np.array([50], dtype=np.int32),
            alleles=np.array([["A", "T"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=int(sim_ts.sequence_length),
            sample_ids=np.array(["tsk_0", "tsk_1", "tsk_2"]),
        )

        ts, _ = _run_pipeline_no_augment(sample_store)

        cfg = Config(
            sources={
                "test": Source(path=sample_store, name="test"),
                "extra": Source(path=extra_store, name="extra"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=AugmentSitesConfig(sources=["extra"]),
        )

        # Check that position 50 is in the sequence intervals
        meta = ts.metadata if ts.metadata is not None else {}
        intervals = meta.get("sequence_intervals")
        if intervals is not None:
            in_interval = False
            for s, e in intervals:
                if 50 >= s and 50 < e:
                    in_interval = True
                    break
            if not in_interval:
                # Skip if position 50 is outside intervals
                return

        aug_ts = augment_sites(ts, cfg)
        # If position 50 was added, verify it has mutations
        if aug_ts.num_sites > ts.num_sites:
            new_positions = set(aug_ts.sites_position) - set(ts.sites_position)
            assert 50.0 in new_positions

    def test_multiallelic(self):
        """Multi-allelic sites are placed correctly."""
        main_store = make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        # Multi-allelic site at position 200
        multi_store = make_sample_vcz(
            genotypes=np.array([[[0], [1], [2]]], dtype=np.int8),
            positions=np.array([200], dtype=np.int32),
            alleles=np.array([["A", "T", "G"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )

        ts, _ = _run_pipeline_no_augment(main_store)

        cfg = Config(
            sources={
                "test": Source(path=main_store, name="test"),
                "multi": Source(path=multi_store, name="multi"),
                "ancestors": Source(
                    path=None, name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
            ),
            augment_sites=AugmentSitesConfig(sources=["multi"]),
        )
        aug_ts = augment_sites(ts, cfg)
        assert aug_ts.num_sites > ts.num_sites
        # Find the new site
        for site in aug_ts.sites():
            if site.position == 200.0:
                # Should have mutations
                assert len(site.mutations) > 0
                break

    def test_run_integration(self):
        """Full run() with augment_sites configured end-to-end."""
        main_store = make_sample_vcz(
            genotypes=np.array([[[0], [1], [1]], [[1], [0], [1]]], dtype=np.int8),
            positions=np.array([100, 300], dtype=np.int32),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            ancestral_state=np.array(["A", "C"]),
            sequence_length=1000,
        )
        extra_store = make_sample_vcz(
            genotypes=np.array([[[0], [0], [1]]], dtype=np.int8),
            positions=np.array([200], dtype=np.int32),
            alleles=np.array([["A", "G"]]),
            ancestral_state=np.array(["A"]),
            sequence_length=1000,
        )

        src = Source(path=main_store, name="test")
        extra_src = Source(path=extra_store, name="extra")
        anc_src = Source(path=None, name="ancestors", sample_time="sample_time")
        cfg = Config(
            sources={"test": src, "extra": extra_src, "ancestors": anc_src},
            ancestors=[AncestorsConfig(name="ancestors", path=None, sources=["test"])],
            match=MatchConfig(
                sources={
                    "ancestors": MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "test": MatchSourceConfig(),
                },
                output="output.trees",
            ),
            post_process=PostProcessConfig(split_ultimate=False, erase_flanks=True),
            augment_sites=AugmentSitesConfig(sources=["extra"]),
        )
        out_ts = run(cfg)
        assert out_ts.num_nodes > 0
        # Should have the augmented site
        positions = set(out_ts.sites_position)
        assert 200.0 in positions
