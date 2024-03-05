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
import json
import os
import pickle
import sys
import tempfile

import msprime
import numcodecs
import numpy as np
import pytest
import sgkit
import tskit
import xarray as xr
import zarr

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


def make_ts_and_zarr(path, add_optional=False, shuffle_alleles=True):
    import sgkit.io.vcf

    ts = msprime.sim_ancestry(
        samples=100,
        ploidy=3,
        recombination_rate=0.25,
        sequence_length=250,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=0.025, model=msprime.JC69(), random_seed=42)
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
        path / "data.vcf",
        path / "data.zarr",
        ploidy=3,
        max_alt_alleles=4,  # tests tsinfer's ability to handle empty string alleles
    )

    ancestral_allele = [site.ancestral_state for site in ts.sites()]
    add_array_to_dataset(
        "variant_ancestral_allele",
        ancestral_allele,
        path / "data.zarr",
        dims=["variants"],
    )

    unseen_ancestral_allele_count = 0
    for variant in ts.variants():
        ancestral_index = variant.alleles.index(variant.site.ancestral_state)
        if ancestral_index not in variant.genotypes:
            unseen_ancestral_allele_count += 1
    assert unseen_ancestral_allele_count > 0

    if shuffle_alleles:
        # Tskit will always put the ancestral allele in the REF field, which will then
        # be the zeroth allele in the zarr file.  We need to shuffle the alleles around
        # to make sure that we test ancestral allele handling.
        ds = sgkit.load_dataset(path / "data.zarr")
        site_alleles = ds["variant_allele"].values
        assert np.all(ds.variant_allele.values[:, 0] == ancestral_allele)
        num_alleles = [len([a for a in alleles if a != ""]) for alleles in site_alleles]
        random = np.random.RandomState(42)
        new_ancestral_allele_pos = [random.randint(0, n) for n in num_alleles]
        new_site_alleles = []
        index_remappers = []
        for alleles, new_pos in zip(site_alleles, new_ancestral_allele_pos):
            alleles = list(alleles)
            indexes = list(range(len(alleles)))
            alleles.insert(new_pos, alleles.pop(0))
            indexes.insert(new_pos, indexes.pop(0))
            new_site_alleles.append(alleles)
            indexes = np.argsort(indexes)
            index_remappers.append(np.array(indexes))
        new_site_alleles = np.array(new_site_alleles, dtype=object)
        assert np.any(new_site_alleles[:, 0] != ancestral_allele)
        ds["variant_allele"] = xr.DataArray(
            new_site_alleles, dims=["variants", "alleles"]
        )
        genotypes = ds["call_genotype"].values
        for i, remapper in enumerate(index_remappers):
            genotypes[i] = remapper[genotypes[i]]
        ds["call_genotype"] = xr.DataArray(
            genotypes, dims=["variants", "samples", "ploidy"]
        )
        sgkit.save_dataset(
            ds.drop_vars(set(ds.data_vars) - {"call_genotype", "variant_allele"}),
            path / "data.zarr",
            mode="a",
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
            ["provenances"],
        )
        add_array_to_dataset(
            "provenances_record",
            ['{"foo": 1}', '{"foo": 2}'],
            path / "data.zarr",
            ["provenances"],
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
        ds = sgkit.load_dataset(path / "data.zarr")

    return ts, path / "data.zarr"


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_dataset_roundtrip(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    samples = tsinfer.SgkitSampleData(zarr_path)
    inf_ts = tsinfer.infer(samples)
    ds = sgkit.load_dataset(zarr_path)

    assert ts.num_individuals == inf_ts.num_individuals == ds.dims["samples"]
    for ts_ind, sample_id in zip(inf_ts.individuals(), ds["sample_id"].values):
        assert ts_ind.metadata["sgkit_sample_id"] == sample_id

    assert (
        ts.num_samples == inf_ts.num_samples == ds.dims["samples"] * ds.dims["ploidy"]
    )
    assert ts.num_sites == inf_ts.num_sites == ds.dims["variants"]
    assert ts.sequence_length == inf_ts.sequence_length == ds.attrs["contig_lengths"][0]
    for (
        v,
        inf_v,
    ) in zip(ts.variants(), inf_ts.variants()):
        mapping_dict = {a: i for i, a in enumerate(v.alleles)}
        mapping = np.array([mapping_dict[a] for a in inf_v.alleles])
        mapped_genotypes = mapping[inf_v.genotypes]
        assert np.array_equal(v.genotypes, mapped_genotypes)

    # Check that the trees are non-trivial (i.e. the sites have actually been used)
    assert inf_ts.num_trees > 10
    assert inf_ts.num_edges > 200


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_individual_metadata_not_clobbered(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    # Load the zarr to add metadata for testing
    zarr_root = zarr.open(zarr_path)
    empty_obj = json.dumps({}).encode()
    indiv_metadata = np.array([empty_obj] * ts.num_individuals, dtype=object)
    indiv_metadata[42] = json.dumps({"sgkit_sample_id": "foobar"}).encode()
    zarr_root.create_dataset(
        "individuals_metadata", data=indiv_metadata, object_codec=numcodecs.VLenBytes()
    )
    zarr_root.attrs["individuals_metadata_schema"] = repr(
        tskit.MetadataSchema.permissive_json()
    )

    samples = tsinfer.SgkitSampleData(zarr_path)
    inf_ts = tsinfer.infer(samples)
    ds = sgkit.load_dataset(zarr_path)

    assert ts.num_individuals == inf_ts.num_individuals == ds.dims["samples"]
    for i, (ts_ind, sample_id) in enumerate(
        zip(inf_ts.individuals(), ds["sample_id"].values)
    ):
        if i != 42:
            assert ts_ind.metadata["sgkit_sample_id"] == sample_id
        else:
            assert ts_ind.metadata["sgkit_sample_id"] == "foobar"


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_dataset_accessors(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path, add_optional=True, shuffle_alleles=False)
    samples = tsinfer.SgkitSampleData(zarr_path)
    ds = sgkit.load_dataset(zarr_path)

    assert samples.format_name == "tsinfer-sgkit-sample-data"
    assert samples.format_version == (0, 1)
    assert samples.finalised
    assert samples.sequence_length == ts.sequence_length + 1337
    assert samples.num_sites == ts.num_sites
    assert samples.sites_metadata_schema == ts.tables.sites.metadata_schema.schema
    assert samples.sites_metadata == [site.metadata for site in ts.sites()]
    assert np.array_equal(samples.sites_time, np.arange(ts.num_sites) / ts.num_sites)
    assert np.array_equal(samples.sites_position, ts.tables.sites.position)
    for alleles, v in zip(samples.sites_alleles, ts.variants()):
        # sgkit alleles are padded to be rectangular
        assert np.all(alleles[: len(v.alleles)] == v.alleles)
        assert np.all(alleles[len(v.alleles) :] == "")
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
    assert samples.individuals_metadata == [
        {"sgkit_sample_id": sample_id, **ind.metadata}
        for ind, sample_id in zip(ts.individuals(), ds["sample_id"].values)
    ]
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

    # Need to shuffle for the ancestral allele test
    ts, zarr_path = make_ts_and_zarr(tmp_path, add_optional=True)
    samples = tsinfer.SgkitSampleData(zarr_path)
    for i in range(ts.num_sites):
        assert (
            samples.sites_alleles[i][samples.sites_ancestral_allele[i]]
            == ts.site(i).ancestral_state
        )


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_accessors_defaults(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    samples = tsinfer.SgkitSampleData(zarr_path)
    ds = sgkit.load_dataset(zarr_path)

    default_schema = tskit.MetadataSchema.permissive_json().schema
    assert samples.sequence_length == ts.sequence_length
    assert samples.sites_metadata_schema == default_schema
    assert samples.sites_metadata == [{} for _ in range(ts.num_sites)]
    for time in samples.sites_time:
        assert tskit.is_unknown_time(time)
    assert np.array_equal(samples.sites_mask, np.ones(ts.num_sites, dtype=bool))
    assert np.array_equal(samples.provenances_timestamp, [])
    assert np.array_equal(samples.provenances_record, [])
    assert samples.metadata_schema == default_schema
    assert samples.metadata == {}
    assert samples.populations_metadata_schema == default_schema
    assert samples.populations_metadata == []
    assert samples.individuals_metadata_schema == default_schema
    assert samples.individuals_metadata == [
        {"sgkit_sample_id": sample_id} for sample_id in ds["sample_id"].values
    ]
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
        sites_mask = np.ones_like(ds["variant_position"], dtype=bool)
        for i in sites:
            sites_mask[i] = False
        add_array_to_dataset("variant_mask", sites_mask, zarr_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        assert samples.num_sites == len(sites)
        assert np.array_equal(samples.sites_mask, ~sites_mask)
        assert np.array_equal(
            samples.sites_position, ts.tables.sites.position[~sites_mask]
        )
        inf_ts = tsinfer.infer(samples)
        assert np.array_equal(
            ts.genotype_matrix()[~sites_mask], inf_ts.genotype_matrix()
        )
        assert np.array_equal(
            ts.tables.sites.position[~sites_mask], inf_ts.tables.sites.position
        )
        assert np.array_equal(
            ts.tables.sites.ancestral_state[~sites_mask],
            inf_ts.tables.sites.ancestral_state,
        )
        # TODO - site metadata needs merging not replacing
        # assert [site.metadata for site in ts.sites() if site.id in sites] == [
        #     site.metadata for site in inf_ts.sites()
        # ]

    def test_sgkit_variant_bad_mask_length(self, tmp_path):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.zeros(ds.sizes["variants"] + 1, dtype=int)
        add_array_to_dataset("variant_mask", sites_mask, zarr_path)
        with pytest.raises(
            ValueError,
            match="Mask must be the same length as the number of unmasked sites",
        ):
            tsinfer.SgkitSampleData(zarr_path)

    def test_bad_mask_length_at_iterator(self, tmp_path):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.zeros(ds.sizes["variants"] + 1, dtype=int)
        from tsinfer.formats import chunk_iterator

        with pytest.raises(
            ValueError, match="Mask must be the same length as the array"
        ):
            for _ in chunk_iterator(ds.call_genotype, mask=sites_mask):
                pass

    @pytest.mark.parametrize("sample_list", [[1, 2, 3, 5, 9, 27], [0], []])
    def test_sgkit_sample_mask(self, tmp_path, sample_list):
        ts, zarr_path = make_ts_and_zarr(tmp_path, add_optional=True)
        ds = sgkit.load_dataset(zarr_path)
        samples_mask = np.ones_like(ds["sample_id"], dtype=bool)
        for i in sample_list:
            samples_mask[i] = False
        add_array_to_dataset("samples_mask", samples_mask, zarr_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        assert samples.ploidy == 3
        assert samples.num_individuals == len(sample_list)
        assert samples.num_samples == len(sample_list) * samples.ploidy
        assert np.array_equal(samples.individuals_mask, ~samples_mask)
        assert np.array_equal(samples.samples_mask, np.repeat(~samples_mask, 3))
        assert np.array_equal(
            samples.individuals_time, ds.individuals_time.values[~samples_mask]
        )
        assert np.array_equal(
            samples.individuals_location, ds.individuals_location.values[~samples_mask]
        )
        assert np.array_equal(
            samples.individuals_population,
            ds.individuals_population.values[~samples_mask],
        )
        assert np.array_equal(
            samples.individuals_flags, ds.individuals_flags.values[~samples_mask]
        )
        assert np.array_equal(
            samples.samples_individual, np.repeat(np.arange(len(sample_list)), 3)
        )
        expected_gt = ds.call_genotype.values[:, ~samples_mask, :].reshape(
            samples.num_sites, len(sample_list) * 3
        )
        assert np.array_equal(samples.sites_genotypes, expected_gt)
        for v, gt in zip(samples.variants(), expected_gt):
            assert np.array_equal(v.genotypes, gt)

        for i, (id, haplo) in enumerate(samples.haplotypes()):
            assert id == i
            assert np.array_equal(haplo, expected_gt[:, i])


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_sgkit_ancestral_allele_same_ancestors(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    ts_sampledata = tsinfer.SampleData.from_tree_sequence(ts)
    sg_sampledata = tsinfer.SgkitSampleData(zarr_path)
    ts_ancestors = tsinfer.generate_ancestors(ts_sampledata)
    sg_ancestors = tsinfer.generate_ancestors(sg_sampledata)
    assert np.array_equal(ts_ancestors.sites_position, sg_ancestors.sites_position)
    assert np.array_equal(ts_ancestors.ancestors_start, sg_ancestors.ancestors_start)
    assert np.array_equal(ts_ancestors.ancestors_end, sg_ancestors.ancestors_end)
    assert np.array_equal(ts_ancestors.ancestors_time, sg_ancestors.ancestors_time)
    for ts_focal, sg_focal in zip(
        ts_ancestors.ancestors_focal_sites, sg_ancestors.ancestors_focal_sites
    ):
        assert np.array_equal(ts_focal, sg_focal)
    assert np.array_equal(
        ts_ancestors.ancestors_full_haplotype, sg_ancestors.ancestors_full_haplotype
    )
    assert np.array_equal(
        ts_ancestors.ancestors_full_haplotype_mask,
        sg_ancestors.ancestors_full_haplotype_mask,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="File permission errors on Windows")
def test_missing_ancestral_allele(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    ds = ds.drop_vars(["variant_ancestral_allele"])
    sgkit.save_dataset(ds, str(zarr_path) + ".tmp")
    samples = tsinfer.SgkitSampleData(str(zarr_path) + ".tmp")
    with pytest.raises(ValueError, match="variant_ancestral_allele was not found"):
        tsinfer.infer(samples)


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_ancestral_missingness(tmp_path):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    ancestral_allele = ds.variant_ancestral_allele.values
    ancestral_allele[0] = "N"
    ancestral_allele[11] = "-"
    ancestral_allele[12] = "💩"
    ancestral_allele[15] = "💩"
    ds = ds.drop_vars(["variant_ancestral_allele"])
    sgkit.save_dataset(ds, str(zarr_path) + ".tmp")
    add_array_to_dataset(
        "variant_ancestral_allele",
        ancestral_allele,
        str(zarr_path) + ".tmp",
        ["variants"],
    )
    ds = sgkit.load_dataset(str(zarr_path) + ".tmp")
    sd = tsinfer.SgkitSampleData(str(zarr_path) + ".tmp")
    with pytest.warns(
        UserWarning,
        match=r"not found in the variant_allele array for the 4 [\s\S]*'💩': 2",
    ):
        inf_ts = tsinfer.infer(sd)
    for i, (inf_var, var) in enumerate(zip(inf_ts.variants(), ts.variants())):
        if i in [0, 11, 12, 15]:
            assert inf_var.site.metadata == {"inference_type": "parsimony"}
        else:
            assert inf_var.site.ancestral_state == var.site.ancestral_state


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
            np.array([["", "A", "C"], ["A", "C", ""], ["A", "C", ""]], dtype="S1"),
        )
        ds["variant_ancestral_allele"] = (
            ["variants"],
            np.array(["C", "A", "A"], dtype="S1"),
        )
        sgkit.save_dataset(ds, path)
        samples = tsinfer.SgkitSampleData(path)
        with pytest.raises(ValueError, match="Empty alleles must be at the end"):
            tsinfer.infer(samples)


class TestSgkitMatchSamplesToDisk:
    @pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
    @pytest.mark.parametrize("slice", [(0, 5), (0, 0), (0, 1), (10, 15)])
    def test_match_samples_to_disk_write(
        self, slice, small_sd_fixture, tmp_path, tmpdir
    ):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        ancestors = tsinfer.generate_ancestors(samples)
        anc_ts = tsinfer.match_ancestors(samples, ancestors)
        tsinfer.match_samples_slice_to_disk(
            samples, anc_ts, slice, tmpdir / "test.path"
        )
        file_slice, matches = pickle.load(open(tmpdir / "test.path", "rb"))
        assert slice == file_slice
        assert len(matches) == slice[1] - slice[0]
        for m in matches:
            assert isinstance(m, tsinfer.inference.MatchResult)

    @pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
    def test_match_samples_to_disk_full(self, small_sd_fixture, tmp_path, tmpdir):
        ts, zarr_path = make_ts_and_zarr(tmp_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        ancestors = tsinfer.generate_ancestors(samples)
        anc_ts = tsinfer.match_ancestors(samples, ancestors)
        ts = tsinfer.match_samples(samples, anc_ts)
        start_index = 0
        while start_index < ts.num_samples:
            end_index = min(start_index + 5, ts.num_samples)
            tsinfer.match_samples_slice_to_disk(
                samples,
                anc_ts,
                (start_index, end_index),
                tmpdir / f"test-{start_index}.path",
            )
            start_index = end_index
        batch_ts = tsinfer.match_samples(
            samples, anc_ts, match_file_pattern=str(tmpdir / "*.path")
        )
        ts.tables.assert_equals(batch_ts.tables, ignore_provenance=True)

        tmpdir.join("test-5.path").copy(tmpdir.join("test-5-copy.path"))
        with pytest.raises(ValueError, match="Duplicate sample index 5"):
            tsinfer.match_samples(
                samples, anc_ts, match_file_pattern=str(tmpdir / "*.path")
            )

        os.remove(tmpdir / "test-5.path")
        os.remove(tmpdir / "test-5-copy.path")
        with pytest.raises(ValueError, match="index 5 not found"):
            tsinfer.match_samples(
                samples, anc_ts, match_file_pattern=str(tmpdir / "*.path")
            )
