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
Tests for the tsinfer CLI (click commands).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import zarr
from click.testing import CliRunner
from helpers import make_sample_vcz

from tsinfer.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sample_vcz_to_disk(tmp_dir: str) -> str:
    """Create a sample VCZ on disk and return its path."""
    store = make_sample_vcz(
        genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
        positions=np.array([100, 200], dtype=np.int32),
        alleles=np.array([["A", "T"], ["A", "T"]]),
        ancestral_state=np.array(["A", "A"]),
        sequence_length=1000,
    )
    vcz_path = os.path.join(tmp_dir, "samples.vcz")
    zarr.save(vcz_path, **{k: store[k][:] for k in store})
    # Re-open to verify it's a valid zarr store
    zarr.open(vcz_path, mode="r")
    return vcz_path


def _toml_path(path: str) -> str:
    """Convert a filesystem path to a TOML-safe string (forward slashes)."""
    return path.replace("\\", "/")


def _write_config(tmp_dir: str, sample_path: str, output_name: str = "out.trees"):
    """Write a minimal TOML config and return its path."""
    output_path = _toml_path(os.path.join(tmp_dir, output_name))
    ancestors_path = _toml_path(os.path.join(tmp_dir, "ancestors.vcz"))
    sample_path = _toml_path(sample_path)
    config_content = f"""\
[[source]]
name = "test"
path = "{sample_path}"

[[ancestors]]
name = "ancestors"
path = "{ancestors_path}"
sources = ["test"]

[match]
output = "{output_path}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.test]

[ancestral_state]
path = "{sample_path}"
field = "variant_ancestral_allele"
"""
    config_path = os.path.join(tmp_dir, "config.toml")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


def _write_run_config(tmp_dir: str, sample_path: str, output_name: str = "out.trees"):
    """Write a TOML config for the run command (no ancestors path needed)."""
    output_path = _toml_path(os.path.join(tmp_dir, output_name))
    ancestors_path = _toml_path(os.path.join(tmp_dir, "ancestors.vcz"))
    sample_path = _toml_path(sample_path)
    config_content = f"""\
[[source]]
name = "test"
path = "{sample_path}"

[[ancestors]]
name = "ancestors"
path = "{ancestors_path}"
sources = ["test"]

[match]
output = "{output_path}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.test]

[ancestral_state]
path = "{sample_path}"
field = "variant_ancestral_allele"
"""
    config_path = os.path.join(tmp_dir, "config.toml")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


# ---------------------------------------------------------------------------
# TestMainGroup
# ---------------------------------------------------------------------------


class TestMainGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "tsinfer" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# TestConfigShow
# ---------------------------------------------------------------------------


class TestConfigShow:
    def test_show_prints_config(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["config", "show", config_path])
            assert result.exit_code == 0, result.output
            assert "[source.test]" in result.output
            assert "[match]" in result.output
            assert "path_compression" in result.output

    def test_show_includes_ancestors(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["config", "show", config_path])
            assert result.exit_code == 0, result.output
            assert "[[ancestors]]" in result.output


# ---------------------------------------------------------------------------
# TestConfigCheck
# ---------------------------------------------------------------------------


class TestConfigCheck:
    def test_check_valid(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["config", "check", config_path])
            assert result.exit_code == 0, result.output

    def test_check_missing_source_path(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write config with nonexistent source path
            config_content = """\
[[source]]
name = "test"
path = "/nonexistent/path.vcz"

[[ancestors]]
name = "ancestors"
path = "/nonexistent/ancestors.vcz"
sources = ["test"]

[match]
output = "out.trees"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.test]

[ancestral_state]
path = "/nonexistent/path.vcz"
field = "variant_ancestral_allele"
"""
            config_path = os.path.join(tmp_dir, "config.toml")
            with open(config_path, "w") as f:
                f.write(config_content)
            result = runner.invoke(main, ["config", "check", config_path])
            assert result.exit_code != 0
            assert "does not exist" in result.output

    def test_check_ancestors_path_not_required_to_exist(self):
        """ancestors.path is an output; it should not need to exist."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_config(tmp_dir, sample_path)
            # ancestors.vcz does not exist on disk — should still pass
            result = runner.invoke(main, ["config", "check", config_path])
            assert result.exit_code == 0, result.output

    def test_check_unknown_ancestor_source(self):
        """Error when ancestors.sources references a name not in [[source]]."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            sample_path_toml = _toml_path(sample_path)
            out_path = _toml_path(os.path.join(tmp_dir, "out.trees"))
            anc_path = _toml_path(os.path.join(tmp_dir, "ancestors.vcz"))
            config_content = f"""\
[[source]]
name = "test"
path = "{sample_path_toml}"

[[ancestors]]
name = "ancestors"
path = "{anc_path}"
sources = ["nonexistent"]

[match]
output = "{out_path}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.test]

[ancestral_state]
path = "{sample_path_toml}"
field = "variant_ancestral_allele"
"""
            config_path = os.path.join(tmp_dir, "config.toml")
            with open(config_path, "w") as f:
                f.write(config_content)
            result = runner.invoke(main, ["config", "check", config_path])
            assert result.exit_code != 0
            assert "unknown source" in result.output.lower()


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_produces_output(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_run_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["run", config_path])
            assert result.exit_code == 0, result.output
            output_path = os.path.join(tmp_dir, "out.trees")
            assert Path(output_path).exists()

    def test_run_force_overwrites(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_run_config(tmp_dir, sample_path)
            # First run
            result = runner.invoke(main, ["run", config_path])
            assert result.exit_code == 0, result.output
            # Second run without --force should fail
            result = runner.invoke(main, ["run", config_path])
            assert result.exit_code != 0
            assert "already exists" in result.output
            # With --force should succeed
            result = runner.invoke(main, ["run", config_path, "--force"])
            assert result.exit_code == 0, result.output

    def test_run_verbose(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_run_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["run", config_path, "-v"])
            assert result.exit_code == 0, result.output

    def test_run_nonexistent_config(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "/nonexistent/config.toml"])
        assert result.exit_code != 0

    def test_cache_size_in_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert "--cache-size" in result.output

    def test_run_with_cache_size(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_run_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["run", config_path, "--cache-size", "128"])
            assert result.exit_code == 0, result.output
            output_path = os.path.join(tmp_dir, "out.trees")
            assert Path(output_path).exists()


# ---------------------------------------------------------------------------
# TestInferAncestors
# ---------------------------------------------------------------------------


class TestInferAncestors:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["infer-ancestors", "--help"])
        assert result.exit_code == 0
        assert "ancestor" in result.output.lower()


# ---------------------------------------------------------------------------
# TestMatch
# ---------------------------------------------------------------------------


class TestMatch:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["match", "--help"])
        assert result.exit_code == 0
        assert "match" in result.output.lower()

    def test_cache_size_in_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["match", "--help"])
        assert "--cache-size" in result.output


# ---------------------------------------------------------------------------
# TestPostProcess
# ---------------------------------------------------------------------------


class TestPostProcess:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["post-process", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# TestAugmentSites
# ---------------------------------------------------------------------------


class TestAugmentSites:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["augment-sites", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output

    def test_augment_sites_adds_sites(self):
        """augment-sites places new sites from the configured source."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create sample VCZ with 2 inference sites + 1 singleton
            store = make_sample_vcz(
                genotypes=np.array(
                    [[[0], [1], [1]], [[0], [0], [1]], [[1], [0], [1]]],
                    dtype=np.int8,
                ),
                positions=np.array([100, 200, 300], dtype=np.int32),
                alleles=np.array([["A", "T"], ["C", "G"], ["A", "T"]]),
                ancestral_state=np.array(["A", "C", "A"]),
                sequence_length=1000,
            )
            vcz_path = os.path.join(tmp_dir, "samples.vcz")
            zarr.save(vcz_path, **{k: store[k][:] for k in store})

            # Write config with augment_sites pointing at the same source
            output_path = _toml_path(os.path.join(tmp_dir, "out.trees"))
            ancestors_path = _toml_path(os.path.join(tmp_dir, "ancestors.vcz"))
            vcz_toml = _toml_path(vcz_path)
            augmented_path = _toml_path(os.path.join(tmp_dir, "augmented.trees"))
            config_content = f"""\
[[source]]
name = "test"
path = "{vcz_toml}"

[[source]]
name = "augment"
path = "{vcz_toml}"

[[ancestors]]
name = "ancestors"
path = "{ancestors_path}"
sources = ["test"]

[match]
output = "{output_path}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.test]

[ancestral_state]
path = "{vcz_toml}"
field = "variant_ancestral_allele"

[augment_sites]
sources = ["augment"]
"""
            config_path = os.path.join(tmp_dir, "config.toml")
            with open(config_path, "w") as f:
                f.write(config_content)

            # Run infer-ancestors + match to produce the input TS
            result = runner.invoke(main, ["infer-ancestors", config_path])
            assert result.exit_code == 0, result.output
            result = runner.invoke(main, ["match", config_path])
            assert result.exit_code == 0, result.output

            # Run augment-sites
            result = runner.invoke(
                main,
                [
                    "augment-sites",
                    config_path,
                    "--input",
                    output_path,
                    "--output",
                    augmented_path,
                ],
            )
            assert result.exit_code == 0, result.output
            assert Path(augmented_path).exists()

            import tskit

            original = tskit.load(output_path)
            augmented = tskit.load(augmented_path)
            assert augmented.num_sites >= original.num_sites

    def test_augment_sites_force(self):
        """augment-sites --force overwrites existing output."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            output_path = _toml_path(os.path.join(tmp_dir, "out.trees"))
            ancestors_path = _toml_path(os.path.join(tmp_dir, "ancestors.vcz"))
            sample_toml = _toml_path(sample_path)
            augmented_path = _toml_path(os.path.join(tmp_dir, "augmented.trees"))
            config_content = f"""\
[[source]]
name = "test"
path = "{sample_toml}"

[[source]]
name = "augment"
path = "{sample_toml}"

[[ancestors]]
name = "ancestors"
path = "{ancestors_path}"
sources = ["test"]

[match]
output = "{output_path}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.test]

[ancestral_state]
path = "{sample_toml}"
field = "variant_ancestral_allele"

[augment_sites]
sources = ["augment"]
"""
            config_path = os.path.join(tmp_dir, "config.toml")
            with open(config_path, "w") as f:
                f.write(config_content)

            # Produce the input TS
            result = runner.invoke(main, ["infer-ancestors", config_path])
            assert result.exit_code == 0, result.output
            result = runner.invoke(main, ["match", config_path])
            assert result.exit_code == 0, result.output

            args = [
                "augment-sites",
                config_path,
                "--input",
                output_path,
                "--output",
                augmented_path,
            ]
            # First run
            result = runner.invoke(main, args)
            assert result.exit_code == 0, result.output

            # Second run without --force should fail
            result = runner.invoke(main, args)
            assert result.exit_code != 0
            assert "already exists" in result.output

            # With --force should succeed
            result = runner.invoke(main, args + ["--force"])
            assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# TestMatchWorkdirCLI
# ---------------------------------------------------------------------------


class TestMatchWorkdirCLI:
    def test_match_with_workdir(self):
        """match --workdir creates match-jobs.json and checkpoint files."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_run_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["infer-ancestors", config_path])
            assert result.exit_code == 0, result.output
            workdir = os.path.join(tmp_dir, "workdir")
            result = runner.invoke(main, ["match", config_path, "--workdir", workdir])
            assert result.exit_code == 0, result.output
            output_path = os.path.join(tmp_dir, "out.trees")
            assert Path(output_path).exists()
            assert Path(workdir, "match-jobs.json").exists()

    def test_match_with_keep_intermediates(self):
        """match --workdir --keep-intermediates retains all .trees files."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_run_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["infer-ancestors", config_path])
            assert result.exit_code == 0, result.output
            workdir = os.path.join(tmp_dir, "workdir")
            result = runner.invoke(
                main,
                [
                    "match",
                    config_path,
                    "--workdir",
                    workdir,
                    "--keep-intermediates",
                ],
            )
            assert result.exit_code == 0, result.output
            trees_files = list(Path(workdir).glob("group_*.trees"))
            assert len(trees_files) >= 1


class TestShowMatchJobs:
    def test_show_match_jobs(self):
        """show-match-jobs prints a histogram from a match-jobs.json file."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_run_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["infer-ancestors", config_path])
            assert result.exit_code == 0, result.output
            workdir = os.path.join(tmp_dir, "workdir")
            result = runner.invoke(main, ["match", config_path, "--workdir", workdir])
            assert result.exit_code == 0, result.output
            json_path = os.path.join(workdir, "match-jobs.json")
            result = runner.invoke(main, ["show-match-jobs", json_path])
            assert result.exit_code == 0, result.output
            assert "Group" in result.output
            assert "Mean kb" in result.output
            assert "Var kb" in result.output
            assert "#" in result.output
            assert "jobs in" in result.output
