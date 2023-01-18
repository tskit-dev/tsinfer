#
# Copyright (C) 2022 University of Oxford
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
Tests for the data files.
"""
import sys
import tempfile

import msprime
import numpy as np
import pytest
import sgkit

import tsinfer
from tsinfer import formats


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_sampledata(tmp_path):
    import sgkit.io.vcf

    ts = msprime.sim_ancestry(
        samples=50,
        ploidy=3,
        recombination_rate=0.25,
        sequence_length=50,
        random_seed=100,
    )
    ts = msprime.sim_mutations(ts, rate=0.025, model=msprime.BinaryMutationModel())
    with open(tmp_path / "data.vcf", "w") as f:
        ts.write_vcf(f)
    sgkit.io.vcf.vcf_to_zarr(
        # max_alt_alleles=4 tests tsinfer's ability to handle empty string alleles,
        tmp_path / "data.vcf",
        tmp_path / "data.zarr",
        ploidy=3,
        max_alt_alleles=4,
    )
    samples = tsinfer.SgkitSampleData(tmp_path / "data.zarr")
    inf_ts = tsinfer.infer(samples)
    assert np.array_equal(ts.genotype_matrix(), inf_ts.genotype_matrix())
    # Check that the trees are non-trivial (i.e. the sites have actually been used)
    assert inf_ts.num_trees > 10
    assert inf_ts.num_edges > 200


@pytest.mark.skipif(sys.platform == "win32", reason="File permission errors on Windows")
def test_sgkit_ancestor(small_sd_fixture, tmp_path):
    with tempfile.TemporaryDirectory(prefix="tsi_eval") as tmpdir:
        f = f"{tmpdir}/test.ancestors"
        tsinfer.generate_ancestors(small_sd_fixture, path=f)
        store = formats.open_lmbd_readonly(f)
        ds = sgkit.load_dataset(store)
        ds = sgkit.variant_stats(ds, merge=True)
        ds = sgkit.sample_stats(ds, merge=True)
        sgkit.display_genotypes(ds)


class TestSgkitSampleDataErrors:
    def test_missing_phase(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3)
        sgkit.save_dataset(ds, path)
        with pytest.raises(
            ValueError, match="The call_genotype_phased array is missing"
        ):
            tsinfer.SgkitSampleData(path)

    def test_unphased(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3)
        ds["call_genotype_phased"] = (
            ds["call_genotype"].dims,
            np.zeros(ds["call_genotype"].shape, dtype=bool),
        )
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="One or more genotypes are unphased"):
            tsinfer.SgkitSampleData(path)

    def test_phased(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3)
        ds["call_genotype_phased"] = (
            ds["call_genotype"].dims,
            np.ones(ds["call_genotype"].shape, dtype=bool),
        )
        sgkit.save_dataset(ds, path)
        tsinfer.SgkitSampleData(path)

    def test_ploidy1_missing_phase(self, tmp_path):
        path = tmp_path / "data.zarr"
        # Ploidy==1 is always ok
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        sgkit.save_dataset(ds, path)
        tsinfer.SgkitSampleData(path)

    def test_ploidy1_unphased(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        ds["call_genotype_phased"] = (
            ds["call_genotype"].dims,
            np.zeros(ds["call_genotype"].shape, dtype=bool),
        )
        sgkit.save_dataset(ds, path)
        tsinfer.SgkitSampleData(path)

    def test_empty_alleles_not_at_end(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        ds["variant_allele"] = (
            ds["variant_allele"].dims,
            np.array([["", "C"], ["A", "C"], ["A", "C"]], dtype="S1"),
        )
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="Empty alleles must be at the end"):
            samples = tsinfer.SgkitSampleData(path)
            tsinfer.infer(samples)
