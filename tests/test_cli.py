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

[ancestors]
path = "{ancestors_path}"
sources = ["test"]

[match]
sources = ["test"]
output = "{output_path}"
recombination_rate = 1e-4
"""
    config_path = os.path.join(tmp_dir, "config.toml")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


def _write_run_config(tmp_dir: str, sample_path: str, output_name: str = "out.trees"):
    """Write a TOML config for the run command (no ancestors path needed)."""
    output_path = _toml_path(os.path.join(tmp_dir, output_name))
    sample_path = _toml_path(sample_path)
    config_content = f"""\
[[source]]
name = "test"
path = "{sample_path}"

[ancestors]
path = "unused"
sources = ["test"]

[match]
sources = ["test"]
output = "{output_path}"
recombination_rate = 1e-4
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
            assert "recombination_rate" in result.output

    def test_show_includes_ancestors(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            config_path = _write_config(tmp_dir, sample_path)
            result = runner.invoke(main, ["config", "show", config_path])
            assert result.exit_code == 0, result.output
            assert "[ancestors]" in result.output


# ---------------------------------------------------------------------------
# TestConfigCheck
# ---------------------------------------------------------------------------


class TestConfigCheck:
    def test_check_valid(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = _write_sample_vcz_to_disk(tmp_dir)
            # Create ancestors.vcz so path check passes
            ancestors_path = os.path.join(tmp_dir, "ancestors.vcz")
            os.makedirs(ancestors_path, exist_ok=True)
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

[ancestors]
path = "/nonexistent/ancestors.vcz"
sources = ["test"]

[match]
sources = ["test"]
output = "out.trees"
recombination_rate = 1e-4
"""
            config_path = os.path.join(tmp_dir, "config.toml")
            with open(config_path, "w") as f:
                f.write(config_content)
            result = runner.invoke(main, ["config", "check", config_path])
            assert result.exit_code != 0
            assert "does not exist" in result.output


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


# ---------------------------------------------------------------------------
# TestPostProcess
# ---------------------------------------------------------------------------


class TestPostProcess:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["post-process", "--help"])
        assert result.exit_code == 0
