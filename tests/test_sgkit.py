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
import tskit
import xarray as xr

import tsinfer
from tsinfer import formats

EXAMPLE_SCHEMA = tskit.MetadataSchema(
    {"codec": "json", "properties": {"foo": {"type": "integer"}}}
)


def add_array_to_dataset(name, array, zarr_path, dims=None):
    ds = sgkit.load_dataset(zarr_path)
    ds.update({name: xr.DataArray(data=array, dims=dims, name=name)})
    sgkit.save_dataset(ds.drop_vars(set(ds.data_vars) - {name}), zarr_path, mode="a")


def add_attribute_to_dataset(name, contents, zarr_path):
    ds = sgkit.load_dataset(zarr_path)
    ds.attrs[name] = contents
    sgkit.save_dataset(ds, zarr_path, mode="a")


def make_ts_and_zarr(path, add_optional=False):
    import sgkit.io.vcf

    ts = msprime.sim_ancestry(
        samples=50,
        ploidy=3,
        recombination_rate=0.25,
        sequence_length=50,
        random_seed=42,
    )
    ts = msprime.sim_mutations(
        ts, rate=0.025, model=msprime.BinaryMutationModel(), random_seed=42
    )
    tables = ts.dump_tables()
    tables.metadata_schema = EXAMPLE_SCHEMA
    sites_copy = tables.sites.copy()
    tables.sites.clear()
    tables.sites.metadata_schema = EXAMPLE_SCHEMA
    for i, site in enumerate(sites_copy):
        tables.sites.append(site.replace(metadata={"id_site": i}))

    pops_copy = tables.populations.copy()
    tables.populations.clear()
    tables.populations.metadata_schema = EXAMPLE_SCHEMA
    for i, pop in enumerate(pops_copy):
        tables.populations.append(pop.replace(metadata={"id_pop": i}))

    indiv_copy = tables.individuals.copy()
    tables.individuals.clear()
    tables.individuals.metadata_schema = EXAMPLE_SCHEMA
    for i, ind in enumerate(indiv_copy):
        tables.individuals.append(ind.replace(metadata={"id_indiv": i}))

    ts = tables.tree_sequence()

    # For simplicity, we would like go directly from the tree sequence to the sgkit
    # dataset, but for testing it is desirable to have sgkit code write as much of the
    # data as possible.
    with open(path / "data.vcf", "w") as f:
        ts.write_vcf(f)
    sgkit.io.vcf.vcf_to_zarr(
        # max_alt_alleles=4 tests tsinfer's ability to handle empty string alleles,
        path / "data.vcf",
        path / "data.zarr",
        ploidy=3,
        max_alt_alleles=4,
    )

    if add_optional:
        add_attribute_to_dataset(
            "sequence_length",
            ts.sequence_length + 1337,
            path / "data.zarr",
        )
        add_array_to_dataset(
            "sites_metadata",
            np.array(
                [
                    tables.sites.metadata_schema.encode_row(site.metadata)
                    for site in ts.sites()
                ]
            ),
            path / "data.zarr",
            ["variants"],
        )
        add_array_to_dataset(
            "sites_time",
            np.arange(ts.num_sites) / ts.num_sites,
            path / "data.zarr",
            ["variants"],
        )
        add_attribute_to_dataset(
            "sites_metadata_schema",
            repr(tables.sites.metadata_schema),
            path / "data.zarr",
        )
        add_attribute_to_dataset(
            "metadata_schema",
            repr(tables.metadata_schema),
            path / "data.zarr",
        )
        add_array_to_dataset(
            "provenances_timestamp",
            ["2021-01-01T00:00:00", "2021-01-02T00:00:00"],
            path / "data.zarr",
        )
        add_array_to_dataset(
            "provenances_record",
            ['{"foo": 1}', '{"foo": 2}'],
            path / "data.zarr",
        )
        add_attribute_to_dataset(
            "populations_metadata_schema",
            repr(tables.populations.metadata_schema),
            path / "data.zarr",
        )
        add_array_to_dataset(
            "populations_metadata",
            np.array(
                [
                    tables.populations.metadata_schema.encode_row(population.metadata)
                    for population in ts.populations()
                ]
            ),
            path / "data.zarr",
            ["populations"],
        )
        add_array_to_dataset(
            "individuals_time",
            np.arange(ts.num_individuals, dtype=np.float32),
            path / "data.zarr",
            ["samples"],
        )
        add_array_to_dataset(
            "individuals_metadata",
            np.array(
                [
                    tables.individuals.metadata_schema.encode_row(individual.metadata)
                    for individual in ts.individuals()
                ]
            ),
            path / "data.zarr",
            ["samples"],
        )
        add_array_to_dataset(
            "individuals_location",
            np.tile(np.array([["0", "1"]], dtype="float32"), (ts.num_individuals, 1)),
            path / "data.zarr",
            ["samples", "coordinates"],
        )
        add_array_to_dataset(
            "individuals_population",
            np.zeros(ts.num_individuals, dtype="int32"),
            path / "data.zarr",
            ["samples"],
        )
        add_array_to_dataset(
            "individuals_flags",
            np.random.RandomState(42).randint(
                0, 2_000_000, ts.num_individuals, dtype="int32"
            ),
            path / "data.zarr",
            ["samples"],
        )
        add_attribute_to_dataset(
            "individuals_metadata_schema",
            repr(tables.individuals.metadata_schema),
            path / "data.zarr",
        )

    return ts, path / "data.zarr"


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_dataset_roundtrip(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    samples = tsinfer.SgkitSampleData(zarr_path)
    inf_ts = tsinfer.infer(samples)
    assert np.array_equal(ts.genotype_matrix(), inf_ts.genotype_matrix())
    # Check that the trees are non-trivial (i.e. the sites have actually been used)
    assert inf_ts.num_trees > 10
    assert inf_ts.num_edges > 200


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_dataset_accessors(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path, add_optional=True)
    samples = tsinfer.SgkitSampleData(zarr_path)

    assert samples.format_name == "tsinfer-sgkit-sample-data"
    assert samples.format_version == (0, 1)
    assert samples.finalised
    assert samples.sequence_length == ts.sequence_length + 1337
    assert samples.num_sites == ts.num_sites
    assert samples.sites_metadata_schema == ts.tables.sites.metadata_schema.schema
    assert samples.sites_metadata == [site.metadata for site in ts.sites()]
    assert np.array_equal(samples.sites_time, np.arange(ts.num_sites) / ts.num_sites)
    assert np.array_equal(samples.sites_position, ts.tables.sites.position)
    assert np.array_equal(
        samples.sites_alleles,
        np.tile(np.array([["0", "1", "", "", ""]], dtype="O"), (ts.num_sites, 1)),
    )
    assert np.array_equal(samples.sites_mask, np.ones(ts.num_sites, dtype=bool))
    assert np.array_equal(
        samples.sites_ancestral_allele, np.zeros(ts.num_sites, dtype=np.int8)
    )
    assert np.array_equal(samples.sites_genotypes, ts.genotype_matrix())
    assert np.array_equal(
        samples.provenances_timestamp, ["2021-01-01T00:00:00", "2021-01-02T00:00:00"]
    )
    assert samples.provenances_record == [{"foo": 1}, {"foo": 2}]
    assert samples.num_samples == ts.num_samples
    assert np.array_equal(
        samples.samples_individual, np.repeat(np.arange(ts.num_samples // 3), 3)
    )
    assert samples.metadata_schema == EXAMPLE_SCHEMA.schema
    assert samples.metadata == ts.tables.metadata
    assert (
        samples.populations_metadata_schema
        == ts.tables.populations.metadata_schema.schema
    )
    assert samples.populations_metadata == [pop.metadata for pop in ts.populations()]
    assert samples.num_individuals == ts.num_individuals
    assert np.array_equal(
        samples.individuals_time, np.arange(ts.num_individuals, dtype=np.float32)
    )
    assert (
        samples.individuals_metadata_schema
        == ts.tables.individuals.metadata_schema.schema
    )
    assert samples.individuals_metadata == [ind.metadata for ind in ts.individuals()]
    assert np.array_equal(
        samples.individuals_location,
        np.tile(np.array([["0", "1"]], dtype="float32"), (ts.num_individuals, 1)),
    )
    assert np.array_equal(
        samples.individuals_population, np.zeros(ts.num_individuals, dtype="int32")
    )
    assert np.array_equal(
        samples.individuals_flags,
        np.random.RandomState(42).randint(
            0, 2_000_000, ts.num_individuals, dtype="int32"
        ),
    )


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_accessors_defaults(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    samples = tsinfer.SgkitSampleData(zarr_path)

    default_schema = tskit.MetadataSchema.permissive_json().schema
    assert samples.sequence_length == ts.sequence_length
    assert samples.sites_metadata_schema == default_schema
    assert samples.sites_metadata == [{} for _ in range(ts.num_sites)]
    for time in samples.sites_time:
        assert tskit.is_unknown_time(time)
    assert np.array_equal(samples.sites_mask, np.ones(ts.num_sites, dtype=bool))
    assert np.array_equal(
        samples.sites_ancestral_allele, np.zeros(ts.num_sites, dtype=np.int8)
    )
    assert np.array_equal(samples.provenances_timestamp, [])
    assert np.array_equal(samples.provenances_record, [])
    assert samples.metadata_schema == default_schema
    assert samples.metadata == {}
    assert samples.populations_metadata_schema == default_schema
    assert samples.populations_metadata == []
    assert samples.individuals_metadata_schema == default_schema
    assert samples.individuals_metadata == [{} for _ in range(ts.num_individuals)]
    for time in samples.individuals_time:
        assert tskit.is_unknown_time(time)
    assert np.array_equal(
        samples.individuals_location, np.array([[]] * ts.num_individuals, dtype=float)
    )
    assert np.array_equal(
        samples.individuals_population, np.full(ts.num_individuals, tskit.NULL)
    )
    assert np.array_equal(
        samples.individuals_flags, np.zeros(ts.num_individuals, dtype=int)
    )


@pytest.mark.skipif(sys.platform == "win32", reason="File permission errors on Windows")
class TestSgkitMask:
    @pytest.mark.parametrize("sites", [[1, 2, 3, 5, 9, 27], [0], []])
    def test_sgkit_variant_mask(self, tmp_path, sites):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        print(ds)
        sites_mask = np.zeros_like(ds["variant_position"], dtype=bool)
        for i in sites:
            sites_mask[i] = True
        add_array_to_dataset("variant_mask", sites_mask, zarr_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        assert samples.num_sites == len(sites)
        assert np.array_equal(samples.sites_mask, sites_mask)
        assert np.array_equal(
            samples.sites_position, ts.tables.sites.position[sites_mask]
        )
        inf_ts = tsinfer.infer(samples)
        assert np.array_equal(
            ts.genotype_matrix()[sites_mask], inf_ts.genotype_matrix()
        )
        assert np.array_equal(
            ts.tables.sites.position[sites_mask], inf_ts.tables.sites.position
        )
        assert np.array_equal(
            ts.tables.sites.ancestral_state[sites_mask],
            inf_ts.tables.sites.ancestral_state,
        )
        # TODO - site metadata needs merging not replacing
        # assert [site.metadata for site in ts.sites() if site.id in sites] == [
        #     site.metadata for site in inf_ts.sites()
        # ]

    def test_sgkit_variant_bad_mask(self, tmp_path):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.arange(ds.sizes["variants"], dtype=int)
        add_array_to_dataset("variant_mask", sites_mask, zarr_path)
        with pytest.raises(
            ValueError,
            match="The variant_mask array contains values " "other than 0 or 1",
        ):
            tsinfer.SgkitSampleData(zarr_path)

    def test_sgkit_variant_bad_mask_length(self, tmp_path):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.ones(ds.sizes["variants"] + 1, dtype=int)
        add_array_to_dataset("variant_mask", sites_mask, zarr_path)
        with pytest.raises(
            ValueError,
            match="Mask must be the same length as the number of unmasked sites",
        ):
            tsinfer.SgkitSampleData(zarr_path)

    def test_bad_mask_length_at_iterator(self, tmp_path):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.ones(ds.sizes["variants"] + 1, dtype=int)
        from tsinfer.formats import chunk_iterator

        with pytest.raises(
            ValueError, match="Mask must be the same length as the array"
        ):
            for _ in chunk_iterator(ds.variant_position, mask=sites_mask):
                pass


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
