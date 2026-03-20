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

import numpy as np
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
    vcz_path = tmp_path / "samples.vcz"
    zarr.save(vcz_path, **{k: store[k][:] for k in store})
    return vcz_path


def _toml_path(path):
    """Forward-slash path for TOML strings."""
    return str(path).replace("\\", "/")


def _write_toml(tmp_path, content):
    """Write TOML string to config.toml, return path."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(content)
    return config_path


def _check_genotypes(input_vcz_path, output_ts):
    """Assert output TS sample genotypes match input VCZ genotypes."""
    input_store = zarr.open(input_vcz_path, mode="r")
    input_gt = input_store["call_genotype"][:]
    input_pos = input_store["variant_position"][:]
    input_alleles = input_store["variant_allele"][:]
    input_sample_ids = [str(s) for s in input_store["sample_id"][:]]

    # Build mapping from input sample ID to input sample index
    input_id_to_idx = {sid: i for i, sid in enumerate(input_sample_ids)}

    # Build mapping from output sample index to input sample index via
    # individual metadata sample_id. Each individual may have multiple nodes
    # (ploidy), ordered by haplotype_index.
    samples_array = output_ts.samples()
    node_to_sample_idx = {int(nid): idx for idx, nid in enumerate(samples_array)}

    # For each output sample index, find the corresponding input haplotype index.
    # Only map individuals whose sample_id appears in this input VCZ;
    # skip individuals from other sources.
    output_to_input = np.full(len(samples_array), -1, dtype=int)
    for ind in output_ts.individuals():
        meta = ind.metadata
        sid = meta["sample_id"]
        if sid not in input_id_to_idx:
            continue
        input_sample_idx = input_id_to_idx[sid]
        ploidy = input_gt.shape[2]
        for _ploidy_offset, nid in enumerate(ind.nodes):
            out_idx = node_to_sample_idx[int(nid)]
            input_hap_idx = input_sample_idx * ploidy + _ploidy_offset
            output_to_input[out_idx] = input_hap_idx

    pos_to_idx = {}
    for i, p in enumerate(input_pos):
        allele_list = [str(a) for a in input_alleles[i] if str(a) != ""]
        if len(allele_list) != 2:
            continue
        if len(np.unique(input_gt[i])) < 2:
            continue
        pos_to_idx[int(p)] = i

    # Indices of output samples that belong to this input VCZ
    mapped_mask = output_to_input >= 0
    mapped_out_indices = np.where(mapped_mask)[0]
    mapped_input_indices = output_to_input[mapped_mask]
    # Sort by input haplotype index so observed aligns with expected
    sort_order = np.argsort(mapped_input_indices)
    mapped_out_indices = mapped_out_indices[sort_order]

    for variant in output_ts.variants():
        pos = int(variant.site.position)
        if pos not in pos_to_idx:
            continue
        i = pos_to_idx.pop(pos)

        # Expected genotypes from input VCZ
        input_hap_gt = input_gt[i].reshape(-1)  # (num_haplotypes,)
        expected = np.asarray(input_alleles[i])[input_hap_gt]
        # Observed genotypes from output TS, only for this source's samples
        alleles_arr = np.array(variant.alleles)
        observed_all = alleles_arr[variant.genotypes]
        observed = observed_all[mapped_out_indices]

        np.testing.assert_array_equal(
            observed,
            expected,
            err_msg=f"Genotype mismatch at position {pos}",
        )

    assert len(pos_to_idx) == 0, f"Missing output positions: {list(pos_to_idx.keys())}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiSourceRoundtrip:
    def test_split_samples(self, tmp_path):
        """First/second half of samples, ancestors from both."""
        sample_vcz = _write_vcz_to_disk(tmp_path)
        ancestor_vcz = _toml_path(tmp_path / "ancestors.vcz")
        output_trees = _toml_path(tmp_path / "output.trees")

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
        ancestor_vcz = _toml_path(tmp_path / "ancestors.vcz")
        output_trees = _toml_path(tmp_path / "output.trees")

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
        ancestor_vcz = _toml_path(tmp_path / "ancestors.vcz")
        output_trees = _toml_path(tmp_path / "output.trees")

        toml_content = f"""\
[[source]]
name = "all_samples"
path = "{_toml_path(sample_vcz)}"

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

[match.sources.all_samples]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(sample_vcz, ts)

    def test_interleaved_sites(self, tmp_path):
        """Alternating positions, ancestors from both."""
        sample_vcz = _write_vcz_to_disk(tmp_path)
        ancestor_vcz = _toml_path(tmp_path / "ancestors.vcz")
        output_trees = _toml_path(tmp_path / "output.trees")

        toml_content = f"""\
[[source]]
name = "all_samples"
path = "{_toml_path(sample_vcz)}"

[[source]]
name = "even_pos"
path = "{_toml_path(sample_vcz)}"
include = "POS=100 | POS=300 | POS=500 | POS=700 | POS=900"

[[source]]
name = "odd_pos"
path = "{_toml_path(sample_vcz)}"
include = "POS=200 | POS=400 | POS=600 | POS=800 | POS=1000"

[[ancestors]]
name = "ancestors"
path = "{ancestor_vcz}"
sources = ["even_pos", "odd_pos"]

[match]
output = "{output_trees}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.all_samples]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(sample_vcz, ts)


# ---------------------------------------------------------------------------
# Mixed-ploidy helpers and data
# ---------------------------------------------------------------------------

# Haploid VCZ: 4 haploid samples from columns 0–3 of _GENOTYPES
_HAPLOID_GENOTYPES = _GENOTYPES[:, :4, :]  # shape (10, 4, 1)
_HAPLOID_SAMPLE_IDS = np.array([f"h{i}" for i in range(4)])

# Diploid VCZ: 2 diploid individuals from columns 4–7 of _GENOTYPES
# Reshape (10, 4, 1) -> (10, 2, 2): ind0 gets cols 4,5; ind1 gets cols 6,7
_DIPLOID_GENOTYPES = _GENOTYPES[:, 4:, :].reshape(10, 2, 2)
_DIPLOID_SAMPLE_IDS = np.array([f"d{i}" for i in range(2)])


def _write_mixed_ploidy_vcz(tmp_path):
    """Write haploid and diploid VCZ stores to disk, return paths."""
    haploid_store = make_sample_vcz(
        genotypes=_HAPLOID_GENOTYPES,
        positions=_POSITIONS,
        alleles=_ALLELES,
        ancestral_state=_ANCESTRAL,
        sequence_length=_SEQUENCE_LENGTH,
        sample_ids=_HAPLOID_SAMPLE_IDS,
    )
    haploid_path = tmp_path / "haploid.vcz"
    zarr.save(haploid_path, **{k: haploid_store[k][:] for k in haploid_store})

    diploid_store = make_sample_vcz(
        genotypes=_DIPLOID_GENOTYPES,
        positions=_POSITIONS,
        alleles=_ALLELES,
        ancestral_state=_ANCESTRAL,
        sequence_length=_SEQUENCE_LENGTH,
        sample_ids=_DIPLOID_SAMPLE_IDS,
    )
    diploid_path = tmp_path / "diploid.vcz"
    zarr.save(diploid_path, **{k: diploid_store[k][:] for k in diploid_store})

    return haploid_path, diploid_path


class TestMixedPloidyRoundtrip:
    def test_haploid_and_diploid_sources(self, tmp_path):
        """Both haploid and diploid sources contribute to ancestors."""
        haploid_vcz, diploid_vcz = _write_mixed_ploidy_vcz(tmp_path)
        ancestor_vcz = _toml_path(tmp_path / "ancestors.vcz")
        output_trees = _toml_path(tmp_path / "output.trees")

        toml_content = f"""\
[[source]]
name = "haploid"
path = "{_toml_path(haploid_vcz)}"

[[source]]
name = "diploid"
path = "{_toml_path(diploid_vcz)}"

[[ancestors]]
name = "ancestors"
path = "{ancestor_vcz}"
sources = ["haploid", "diploid"]

[match]
output = "{output_trees}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.haploid]
[match.sources.diploid]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(haploid_vcz, ts)
        _check_genotypes(diploid_vcz, ts)

    def test_haploid_and_diploid_ancestors_from_one(self, tmp_path):
        """Ancestors from haploid only; diploid matched against them."""
        haploid_vcz, diploid_vcz = _write_mixed_ploidy_vcz(tmp_path)
        ancestor_vcz = _toml_path(tmp_path / "ancestors.vcz")
        output_trees = _toml_path(tmp_path / "output.trees")

        toml_content = f"""\
[[source]]
name = "haploid"
path = "{_toml_path(haploid_vcz)}"

[[source]]
name = "diploid"
path = "{_toml_path(diploid_vcz)}"

[[ancestors]]
name = "ancestors"
path = "{ancestor_vcz}"
sources = ["haploid"]

[match]
output = "{output_trees}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.haploid]
[match.sources.diploid]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(haploid_vcz, ts)
        _check_genotypes(diploid_vcz, ts)

    def test_diploid_only_ancestors(self, tmp_path):
        """Ancestors from diploid only; haploid matched against them."""
        haploid_vcz, diploid_vcz = _write_mixed_ploidy_vcz(tmp_path)
        ancestor_vcz = _toml_path(tmp_path / "ancestors.vcz")
        output_trees = _toml_path(tmp_path / "output.trees")

        toml_content = f"""\
[[source]]
name = "haploid"
path = "{_toml_path(haploid_vcz)}"

[[source]]
name = "diploid"
path = "{_toml_path(diploid_vcz)}"

[[ancestors]]
name = "ancestors"
path = "{ancestor_vcz}"
sources = ["diploid"]

[match]
output = "{output_trees}"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.haploid]
[match.sources.diploid]
"""
        config_path = _write_toml(tmp_path, toml_content)
        cfg = Config.from_toml(config_path)
        ts = run(cfg)
        _check_genotypes(haploid_vcz, ts)
        _check_genotypes(diploid_vcz, ts)
