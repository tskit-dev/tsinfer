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

import msprime
import numpy as np
import pytest

import tsinfer


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_dataset(tmp_path):
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
        tmp_path / "data.vcf", tmp_path / "data.zarr", ploidy=3, max_alt_alleles=1
    )
    samples = tsinfer.SgkitSampleData(tmp_path / "data.zarr")
    inf_ts = tsinfer.infer(samples)
    assert np.array_equal(ts.genotype_matrix(), inf_ts.genotype_matrix())
