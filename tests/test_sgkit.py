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

import numcodecs
import numpy as np
import pytest
import sgkit
import tskit
import tsutil
import zarr

import tsinfer
from tsinfer import formats


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_dataset_roundtrip(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
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
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
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
    ts, zarr_path = tsutil.make_ts_and_zarr(
        tmp_path, add_optional=True, shuffle_alleles=False
    )
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
    assert samples.metadata_schema == tsutil.EXAMPLE_SCHEMA.schema
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
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path, add_optional=True)
    samples = tsinfer.SgkitSampleData(zarr_path)
    for i in range(ts.num_sites):
        assert (
            samples.sites_alleles[i][samples.sites_ancestral_allele[i]]
            == ts.site(i).ancestral_state
        )


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_accessors_defaults(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
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
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.ones_like(ds["variant_position"], dtype=bool)
        for i in sites:
            sites_mask[i] = False
        tsutil.add_array_to_dataset("variant_mask_42", sites_mask, zarr_path)
        samples = tsinfer.SgkitSampleData(zarr_path, sites_mask_name="variant_mask_42")
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
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.zeros(ds.sizes["variants"] + 1, dtype=int)
        tsutil.add_array_to_dataset("variant_mask_foobar", sites_mask, zarr_path)
        tsinfer.SgkitSampleData(zarr_path)
        with pytest.raises(
            ValueError,
            match="Mask must be the same length as the number of unmasked sites",
        ):
            tsinfer.SgkitSampleData(zarr_path, sites_mask_name="variant_mask_foobar")

    def test_bad_mask_length_at_iterator(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
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
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path, add_optional=True)
        ds = sgkit.load_dataset(zarr_path)
        samples_mask = np.ones_like(ds["sample_id"], dtype=bool)
        for i in sample_list:
            samples_mask[i] = False
        tsutil.add_array_to_dataset("samples_mask_69", samples_mask, zarr_path)
        samples = tsinfer.SgkitSampleData(
            zarr_path, sgkit_samples_mask_name="samples_mask_69"
        )
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

    def test_sgkit_sample_and_site_mask(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        # Mask out a random 1/3 of sites
        variant_mask = np.zeros(ts.num_sites, dtype=bool)
        random = np.random.RandomState(42)
        variant_mask[
            random.choice(ts.num_sites, ts.num_sites // 3, replace=False)
        ] = True
        # Mask out a random 1/3 of samples
        samples_mask = np.zeros(ts.num_individuals, dtype=bool)
        samples_mask[
            random.choice(ts.num_individuals, ts.num_individuals // 3, replace=False)
        ] = True
        tsutil.add_array_to_dataset(
            "variant_mask_foobar", variant_mask, zarr_path, dims=["variants"]
        )
        tsutil.add_array_to_dataset(
            "samples_mask_foobar", samples_mask, zarr_path, dims=["samples"]
        )
        samples = tsinfer.SgkitSampleData(
            zarr_path,
            sites_mask_name="variant_mask_foobar",
            sgkit_samples_mask_name="samples_mask_foobar",
        )
        genotypes = samples.sites_genotypes
        ds_genotypes = ds.call_genotype.values[~variant_mask][
            :, ~samples_mask, :
        ].reshape(samples.num_sites, samples.num_samples)
        assert np.array_equal(genotypes, ds_genotypes)

        # Check that variants and haplotypes give the same matrix
        for i, (id, haplo) in enumerate(samples.haplotypes()):
            assert id == i
            assert np.array_equal(haplo, genotypes[:, i])
            assert np.array_equal(haplo, ds_genotypes[:, i])

        for i, v in enumerate(samples.variants()):
            assert np.array_equal(v.genotypes, genotypes[i])
            assert np.array_equal(v.genotypes, ds_genotypes[i])

    def test_sgkit_missing_masks(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        samples.individuals_mask
        samples.sites_mask
        with pytest.raises(
            ValueError, match="The sites mask foobar was not found in the dataset."
        ):
            tsinfer.SgkitSampleData(zarr_path, sites_mask_name="foobar")
        with pytest.raises(
            ValueError,
            match="The sgkit samples mask foobar2 was not found in the dataset.",
        ):
            samples = tsinfer.SgkitSampleData(
                zarr_path, sgkit_samples_mask_name="foobar2"
            )
            samples.individuals_mask

    def test_sgkit_subset_equivalence(self, tmp_path, tmpdir):
        (
            mat_sd,
            mask_sd,
            samples_mask,
            variant_mask,
        ) = tsutil.make_materialized_and_masked_sampledata(tmp_path, tmpdir)
        assert mat_sd.num_sites == mat_sd.num_sites
        assert mask_sd.num_samples == mat_sd.num_samples
        assert mask_sd.num_individuals == mat_sd.num_individuals
        assert np.array_equal(mask_sd.individuals_mask, ~samples_mask)
        assert np.array_equal(mask_sd.samples_mask, np.repeat(~samples_mask, 3))
        assert np.array_equal(mask_sd.sites_position, mat_sd.sites_position)
        assert np.array_equal(mask_sd.sites_alleles, mat_sd.sites_alleles)
        assert np.array_equal(mask_sd.sites_mask, ~variant_mask)
        assert np.array_equal(
            mask_sd.sites_ancestral_allele, mat_sd.sites_ancestral_allele
        )
        assert np.array_equal(mask_sd.sites_genotypes, mat_sd.sites_genotypes)
        assert np.array_equal(mask_sd.samples_individual, mat_sd.samples_individual)
        assert np.array_equal(mask_sd.individuals_metadata, mat_sd.individuals_metadata)

        haplotypes = list(mask_sd.haplotypes())
        haplotypes_subset = list(mat_sd.haplotypes())
        assert len(haplotypes) == len(haplotypes_subset)
        for (id, haplo), (id_subset, haplo_subset) in zip(
            haplotypes, haplotypes_subset
        ):
            assert id == id_subset
            assert np.array_equal(haplo, haplo_subset)

        variants = list(mask_sd.variants())
        variants_subset = list(mat_sd.variants())
        assert len(variants) == len(variants_subset)
        for v, v_subset in zip(variants, variants_subset):
            assert np.array_equal(v.genotypes, v_subset.genotypes)

        haplotypes = list(mask_sd._slice_haplotypes((0, mask_sd.num_samples)))
        haplotypes_subset = list(mat_sd._slice_haplotypes((0, mask_sd.num_samples)))
        assert len(haplotypes) == len(haplotypes_subset)
        for (id, haplo), (id_subset, haplo_subset) in zip(
            haplotypes, haplotypes_subset
        ):
            assert id == id_subset
            assert np.array_equal(haplo, haplo_subset)

        ancestors_subset = tsinfer.generate_ancestors(mat_sd)
        ancestors = tsinfer.generate_ancestors(mask_sd)
        assert np.array_equal(ancestors_subset.sites_position, ancestors.sites_position)
        assert np.array_equal(
            ancestors_subset.ancestors_start, ancestors.ancestors_start
        )
        assert np.array_equal(ancestors_subset.ancestors_end, ancestors.ancestors_end)
        assert np.array_equal(ancestors_subset.ancestors_time, ancestors.ancestors_time)
        for ts_focal, sg_focal in zip(
            ancestors_subset.ancestors_focal_sites, ancestors.ancestors_focal_sites
        ):
            assert np.array_equal(ts_focal, sg_focal)
        assert np.array_equal(
            ancestors_subset.ancestors_full_haplotype,
            ancestors.ancestors_full_haplotype,
        )

        mat_anc_ts = tsinfer.match_ancestors(mat_sd, ancestors_subset)
        mask_anc_ts = tsinfer.match_ancestors(mask_sd, ancestors)
        mat_anc_ts.tables.assert_equals(mask_anc_ts.tables, ignore_timestamps=True)

        mat_ts = tsinfer.match_samples(mat_sd, mat_anc_ts)
        mask_ts = tsinfer.match_samples(mask_sd, mask_anc_ts)
        mat_ts.tables.assert_equals(mask_ts.tables, ignore_timestamps=True)


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_sgkit_ancestral_allele_same_ancestors(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
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
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    ds = ds.drop_vars(["variant_ancestral_allele"])
    sgkit.save_dataset(ds, str(zarr_path) + ".tmp")
    samples = tsinfer.SgkitSampleData(str(zarr_path) + ".tmp")
    with pytest.raises(ValueError, match="variant_ancestral_allele was not found"):
        tsinfer.infer(samples)


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_ancestral_missingness(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    ancestral_allele = ds.variant_ancestral_allele.values
    ancestral_allele[0] = "N"
    ancestral_allele[11] = "-"
    ancestral_allele[12] = "💩"
    ancestral_allele[15] = "💩"
    ds = ds.drop_vars(["variant_ancestral_allele"])
    sgkit.save_dataset(ds, str(zarr_path) + ".tmp")
    tsutil.add_array_to_dataset(
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

    def test_duplicate_positions(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        ds["variant_position"][2] = ds["variant_position"][1]
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="duplicate or out-of-order values"):
            tsinfer.SgkitSampleData(path)

    def test_bad_order_positions(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        ds["variant_position"][0] = ds["variant_position"][2] - 0.5
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="duplicate or out-of-order values"):
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


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
class TestSgkitMatchSamplesToDisk:
    @pytest.mark.parametrize("slice", [(0, 6), (0, 0), (0, 3), (12, 15)])
    def test_match_samples_to_disk_write(self, slice, tmp_path, tmpdir):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
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

    def test_match_samples_to_disk_slice_error(self, tmp_path, tmpdir):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        ancestors = tsinfer.generate_ancestors(samples)
        anc_ts = tsinfer.match_ancestors(samples, ancestors)
        with pytest.raises(ValueError, match="Slice must be a multiple of ploidy"):
            tsinfer.match_samples_slice_to_disk(
                samples, anc_ts, (0, 1), tmpdir / "test.path"
            )

    def test_match_samples_to_disk_full(self, tmp_path, tmpdir):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        samples = tsinfer.SgkitSampleData(zarr_path)
        ancestors = tsinfer.generate_ancestors(samples)
        anc_ts = tsinfer.match_ancestors(samples, ancestors)
        ts = tsinfer.match_samples(samples, anc_ts)
        start_index = 0
        while start_index < ts.num_samples:
            end_index = min(start_index + 6, ts.num_samples)
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

        tmpdir.join("test-6.path").copy(tmpdir.join("test-6-copy.path"))
        with pytest.raises(ValueError, match="Duplicate sample index 6"):
            tsinfer.match_samples(
                samples, anc_ts, match_file_pattern=str(tmpdir / "*.path")
            )

        os.remove(tmpdir / "test-6.path")
        os.remove(tmpdir / "test-6-copy.path")
        with pytest.raises(ValueError, match="index 6 not found"):
            tsinfer.match_samples(
                samples, anc_ts, match_file_pattern=str(tmpdir / "*.path")
            )

    def test_match_samples_to_disk_with_mask(self, tmp_path, tmpdir):
        mat_sd, mask_sd, _, _ = tsutil.make_materialized_and_masked_sampledata(
            tmp_path, tmpdir
        )
        mat_ancestors = tsinfer.generate_ancestors(mat_sd)
        mask_ancestors = tsinfer.generate_ancestors(mask_sd)
        mat_anc_ts = tsinfer.match_ancestors(mat_sd, mat_ancestors)
        mask_anc_ts = tsinfer.match_ancestors(mask_sd, mask_ancestors)
        start_index = 0
        while start_index < mat_sd.num_samples:
            end_index = min(start_index + 6, mat_sd.num_samples)
            tsinfer.match_samples_slice_to_disk(
                mat_sd,
                mat_anc_ts,
                (start_index, end_index),
                tmpdir / f"test-mat-{start_index}.path",
            )
            start_index = end_index

        mat_ts_disk = tsinfer.match_samples(
            mat_sd, mat_anc_ts, match_file_pattern=str(tmpdir / "test-mat-*.path")
        )

        start_index = 0
        while start_index < mask_sd.num_samples:
            end_index = min(start_index + 6, mask_sd.num_samples)
            tsinfer.match_samples_slice_to_disk(
                mask_sd,
                mask_anc_ts,
                (start_index, end_index),
                tmpdir / f"test-mask-{start_index}.path",
            )
            start_index = end_index
        mask_ts_disk = tsinfer.match_samples(
            mask_sd, mask_anc_ts, match_file_pattern=str(tmpdir / "test-mask-*.path")
        )

        mask_ts = tsinfer.match_samples(mask_sd, mask_anc_ts)
        mat_ts = tsinfer.match_samples(mat_sd, mat_anc_ts)

        mat_ts.tables.assert_equals(mask_ts.tables, ignore_timestamps=True)
        mask_ts.tables.assert_equals(mask_ts_disk.tables, ignore_timestamps=True)
        mask_ts_disk.tables.assert_equals(mat_ts_disk.tables, ignore_timestamps=True)
