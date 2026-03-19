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
End-to-end multi-source genotype round-trip tests.

Each test partitions the same 8-sample, 10-site dataset by sample or by
site, defines a TOML config with multiple [[source]] entries, and verifies
that the output genotypes match the input. All tests are skipped until
multi-source ancestor inference is implemented.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr
from helpers import make_sample_vcz

from tsinfer.config import Config
from tsinfer.pipeline import run

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# 8 haploid samples, 10 biallelic sites (4 zeros + 4 ones per column)
_GENOTYPES = np.array(
    [
        [[0], [1], [0], [1], [0], [1], [0], [1]],  # site 0
        [[0], [1], [0], [1], [1], [0], [0], [1]],  # site 1
        [[0], [1], [1], [0], [0], [1], [0], [1]],  # site 2
        [[0], [1], [1], [0], [1], [0], [1], [0]],  # site 3
        [[0], [1], [0], [1], [0], [1], [1], [0]],  # site 4
        [[1], [0], [0], [1], [1], [0], [1], [0]],  # site 5
        [[1], [0], [1], [0], [0], [1], [1], [0]],  # site 6
        [[1], [0], [1], [0], [1], [0], [0], [1]],  # site 7
        [[1], [0], [0], [1], [0], [1], [0], [1]],  # site 8
        [[1], [0], [0], [1], [1], [0], [0], [1]],  # site 9
    ],
    dtype=np.int8,
)

_POSITIONS = np.array(
    [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=np.int32
)

_ALLELES = np.array([["A", "T"]] * 10)
_ANCESTRAL = np.array(["A"] * 10)
_SAMPLE_IDS = np.array([f"s{i}" for i in range(8)])
_SEQUENCE_LENGTH = 2000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_vcz_to_disk(tmp_path):
    """Write the shared 10-site, 8-sample VCZ to disk, return path string."""
    store = make_sample_vcz(
        genotypes=_GENOTYPES,
        positions=_POSITIONS,
        alleles=_ALLELES,
        ancestral_state=_ANCESTRAL,
        sequence_length=_SEQUENCE_LENGTH,
        sample_ids=_SAMPLE_IDS,
    )
    vcz_path = os.path.join(str(tmp_path), "samples.vcz")
    zarr.save(vcz_path, **{k: store[k][:] for k in store})
    return vcz_path


def _toml_path(path):
    """Forward-slash path for TOML strings."""
    return path.replace("\\", "/")


def _write_toml(tmp_path, content):
    """Write TOML string to config.toml, return path."""
    config_path = os.path.join(str(tmp_path), "config.toml")
    with open(config_path, "w") as f:
        f.write(content)
    return config_path


def _check_genotypes(input_vcz_path, output_ts):
    """Assert output TS sample genotypes match input VCZ genotypes."""
    input_store = zarr.open(input_vcz_path, mode="r")
    input_gt = input_store["call_genotype"][:]
    input_pos = input_store["variant_position"][:]
    input_alleles = input_store["variant_allele"][:]

    num_samples = input_gt.shape[1]

    sample_node_ids = []
    for ind in output_ts.individuals():
        sample_node_ids.extend(ind.nodes)
    assert len(sample_node_ids) == num_samples

    pos_to_idx = {}
    for i, p in enumerate(input_pos):
        allele_list = [str(a) for a in input_alleles[i] if str(a) != ""]
        if len(allele_list) != 2:
            continue
        if len(np.unique(input_gt[i])) < 2:
            continue
        pos_to_idx[int(p)] = i

    for variant in output_ts.variants():
        pos = int(variant.site.position)
        if pos not in pos_to_idx:
            continue
        i = pos_to_idx.pop(pos)

        expected = np.asarray(input_alleles[i])[input_gt[i].reshape(-1)]
        observed = np.array(variant.alleles)[variant.genotypes[sample_node_ids]]

        np.testing.assert_array_equal(
            observed,
            expected,
            err_msg=f"Genotype mismatch at position {pos}",
        )

    assert len(pos_to_idx) == 0, f"Missing output positions: {list(pos_to_idx.keys())}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="multi-source ancestor inference not yet implemented")
class TestMultiSourceRoundtrip:
    def test_split_samples(self, tmp_path):
        """First/second half of samples, ancestors from both."""
        sample_vcz = _write_vcz_to_disk(tmp_path)
        ancestor_vcz = _toml_path(os.path.join(str(tmp_path), "ancestors.vcz"))
        output_trees = _toml_path(os.path.join(str(tmp_path), "output.trees"))

        toml_content = f"""\
[[source]]
name = "first_half"
path = "{_toml_path(sample_vcz)}"
samples = "s0,s1,s2,s3"

[[source]]
name = "second_half"
path = "{_toml_path(sample_vcz)}"
samples = "s4,s5,s6,s7"

[[ancestors]]
name = "ancestors"
path = "{ancestor_vcz}"
sources = ["first_half", "second_half"]

[match]
output = "{output_trees}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.first_half]
[match.sources.second_half]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(sample_vcz, ts)

    def test_interleaved_samples(self, tmp_path):
        """Even/odd indexed samples, ancestors from both."""
        sample_vcz = _write_vcz_to_disk(tmp_path)
        ancestor_vcz = _toml_path(os.path.join(str(tmp_path), "ancestors.vcz"))
        output_trees = _toml_path(os.path.join(str(tmp_path), "output.trees"))

        toml_content = f"""\
[[source]]
name = "even_samples"
path = "{_toml_path(sample_vcz)}"
samples = "s0,s2,s4,s6"

[[source]]
name = "odd_samples"
path = "{_toml_path(sample_vcz)}"
samples = "s1,s3,s5,s7"

[[ancestors]]
name = "ancestors"
path = "{ancestor_vcz}"
sources = ["even_samples", "odd_samples"]

[match]
output = "{output_trees}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.even_samples]
[match.sources.odd_samples]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(sample_vcz, ts)

    def test_split_sites(self, tmp_path):
        """Low/high position split at midpoint, ancestors from both."""
        sample_vcz = _write_vcz_to_disk(tmp_path)
        ancestor_vcz = _toml_path(os.path.join(str(tmp_path), "ancestors.vcz"))
        output_trees = _toml_path(os.path.join(str(tmp_path), "output.trees"))

        toml_content = f"""\
[[source]]
name = "low_sites"
path = "{_toml_path(sample_vcz)}"
include = "POS <= 500"

[[source]]
name = "high_sites"
path = "{_toml_path(sample_vcz)}"
include = "POS > 500"

[[ancestors]]
name = "ancestors"
path = "{ancestor_vcz}"
sources = ["low_sites", "high_sites"]

[match]
output = "{output_trees}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.low_sites]
[match.sources.high_sites]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(sample_vcz, ts)

    def test_interleaved_sites(self, tmp_path):
        """Alternating positions, ancestors from both."""
        sample_vcz = _write_vcz_to_disk(tmp_path)
        ancestor_vcz = _toml_path(os.path.join(str(tmp_path), "ancestors.vcz"))
        output_trees = _toml_path(os.path.join(str(tmp_path), "output.trees"))

        toml_content = f"""\
[[source]]
name = "even_pos"
path = "{_toml_path(sample_vcz)}"
include = "POS % 2 == 0"

[[source]]
name = "odd_pos"
path = "{_toml_path(sample_vcz)}"
include = "POS % 2 != 0"

[[ancestors]]
name = "ancestors"
path = "{ancestor_vcz}"
sources = ["even_pos", "odd_pos"]

[match]
output = "{output_trees}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.even_pos]
[match.sources.odd_pos]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(sample_vcz, ts)
