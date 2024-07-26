#
# Copyright (C) 2020 University of Oxford
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
Extra utility functions used in several test files
"""
import json

import msprime
import numpy as np
import sgkit
import tskit
import xarray as xr

import tsinfer


def mark_mutation_times_unknown(ts):
    """
    Msprime now provides times for mutations, which cannot be estimated by tsinfer
    """
    tables = ts.dump_tables()
    tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)
    return tables.tree_sequence()


def json_metadata_is_subset(metadata1, metadata2):
    return metadata1.items() <= metadata2.items()


def add_default_schemas(ts):
    """
    Returns a copy of the specified tree sequence with permissive JSON
    schemas on the tables that are used for round-tripping data in tsinfer.
    """
    tables = ts.dump_tables()
    schema = tskit.MetadataSchema.permissive_json()
    assert len(tables.metadata) == 0
    tables.metadata_schema = schema
    tables.metadata = {}
    tables.populations.metadata_schema = schema
    # msprime 1.0 fills the population metadata, so put it back in here
    for pop in ts.populations():
        tables.populations[pop.id] = pop
    tables.individuals.metadata_schema = schema
    assert len(tables.individuals.metadata) == 0
    tables.sites.metadata_schema = schema
    assert len(tables.sites.metadata) == 0
    tables.nodes.metadata_schema = schema
    assert len(tables.nodes.metadata) == 0
    return tables.tree_sequence()


def get_example_ts(
    sample_size,
    sequence_length=10000,
    mutation_rate=0.0005,
    mutation_model=None,
    discrete_genome=True,
    random_seed=100,
):
    ts = msprime.sim_ancestry(
        sample_size,
        ploidy=1,
        sequence_length=sequence_length,
        recombination_rate=mutation_rate * 0.1,
        discrete_genome=discrete_genome,
        random_seed=random_seed,
    )
    ts = msprime.sim_mutations(
        ts, rate=mutation_rate, model=mutation_model, random_seed=random_seed
    )
    return add_default_schemas(ts)


def get_example_individuals_ts_with_metadata(
    n,
    ploidy=2,
    sequence_length=10000,
    mutation_rate=0.0002,
    *,
    discrete_genome=True,
    skip_last=True,
):
    """
    For testing only, create a ts with lots of arbitrary metadata attached to sites,
    individuals & populations, and also set flags on individuals (*node* flags such as
    tskit.NODE_IS_SAMPLE are not expected to pass through the inference process, as
    they can be set during inference).

    For testing purposes, we can set ``skip_last`` to check what happens if we have
    some samples that are not associated with an individual in the tree sequence.
    """
    ts = msprime.sim_ancestry(
        n,
        ploidy=ploidy,
        recombination_rate=mutation_rate * 0.1,
        sequence_length=sequence_length,
        random_seed=100,
        discrete_genome=discrete_genome,
    )
    ts = msprime.sim_mutations(
        ts, rate=mutation_rate, discrete_genome=discrete_genome, random_seed=100
    )
    ts = add_default_schemas(ts)
    tables = ts.dump_tables()
    tables.metadata = {f"a_{j}": j for j in range(n)}
    tables.populations.clear()
    tables.individuals.clear()
    rng = np.random.default_rng(123)
    for i in range(n):
        location = [i, i]
        individual_meta = {}
        pop_meta = {}
        if i % 2 == 0:
            # Add unicode metadata to every other individual: 8544+i = Roman numerals
            individual_meta = {"unicode id": chr(8544 + i)}
            individual_flags = rng.integers(0, np.iinfo(np.uint32).max, dtype=np.int64)
            # Also for populations: chr(127462) + chr(127462+i) give emoji flags
            pop_meta = {"utf": chr(127462) + chr(127462 + i)}
        tables.populations.add_row(metadata=pop_meta)  # One pop for each individual
        if i < n - 1 or not skip_last:
            tables.individuals.add_row(
                flags=individual_flags, location=location, metadata=individual_meta
            )

    node_populations = tables.nodes.population
    for node in ts.nodes():
        if node.is_sample():
            node_populations[node.id] = node.id // ploidy
    tables.nodes.population = node_populations

    # Manually encode the site metadata to avoid adding the rows one-by-one.
    site_metadata = []
    for site in ts.sites():
        if site.id % 4 == 0:  # Scatter metadata into sites: once every 4th row
            site_metadata.append(json.dumps({"id": f"site {site.id}"}).encode())
        else:
            site_metadata.append(b"{}")
    tables.sites.packset_metadata(site_metadata)

    nodes_individual = tables.nodes.individual  # Assign individuals to sample nodes
    sample_individuals = np.repeat(
        np.arange(n, dtype=tables.nodes.individual.dtype), ploidy
    )
    if skip_last:
        # Should work even if some sample nodes are not assigned to an individual
        sample_individuals[sample_individuals == n - 1] = tskit.NULL
    nodes_individual[ts.samples()] = sample_individuals
    tables.nodes.individual = nodes_individual
    return tables.tree_sequence()


def get_example_historical_sampled_ts(
    individual_times,
    ploidy=2,
    sequence_length=10000,
    mutation_rate=0.0002,
):
    samples = [
        msprime.SampleSet(1, population=0, time=t, ploidy=ploidy)
        for t in individual_times
    ]
    ts = msprime.sim_ancestry(
        samples=samples,
        ploidy=ploidy,
        recombination_rate=mutation_rate * 0.1,
        sequence_length=sequence_length,
        random_seed=100,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=100)
    ts = add_default_schemas(ts)
    tables = ts.dump_tables()
    # Add individuals
    tables.individuals.clear()
    nodes_individual = tables.nodes.individual
    individual_ids = []
    for _ in individual_times:
        individual_ids.append(tables.individuals.add_row(metadata={}))
    is_sample_node = (ts.tables.nodes.flags & tskit.NODE_IS_SAMPLE) != 0
    nodes_individual[is_sample_node] = np.repeat(individual_ids, ploidy)
    tables.nodes.individual = nodes_individual
    return tables.tree_sequence()


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
        ts.write_vcf(f, allow_position_zero=True)
    sgkit.io.vcf.vcf_to_zarr(
        path / "data.vcf",
        path / "data.zarr",
        chunk_length=10,
        chunk_width=12,
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


def make_materialized_and_masked_sampledata(tmp_path, tmpdir):
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    random = np.random.RandomState(42)
    # Mask out a random 1/3 of sites
    variant_mask = np.zeros(ts.num_sites, dtype=bool)
    variant_mask[random.choice(ts.num_sites, ts.num_sites // 3, replace=False)] = True
    # Mask out a random 1/3 of samples
    samples_mask = np.zeros(ts.num_individuals, dtype=bool)
    samples_mask[
        random.choice(ts.num_individuals, ts.num_individuals // 3, replace=False)
    ] = True

    add_array_to_dataset(
        "variant_mask_foobar", variant_mask, zarr_path, dims=["variants"]
    )
    add_array_to_dataset(
        "samples_mask_foobar", samples_mask, zarr_path, dims=["samples"]
    )

    # Create a new sgkit dataset with the subset baked in
    mat_ds = ds.isel(variants=~variant_mask, samples=~samples_mask)
    mat_ds = mat_ds.unify_chunks()
    sgkit.save_dataset(mat_ds, tmpdir / "subset.zarr", auto_rechunk=True)

    mat_sd = tsinfer.VariantData(tmpdir / "subset.zarr", "variant_ancestral_allele")
    mask_sd = tsinfer.VariantData(
        zarr_path,
        "variant_ancestral_allele",
        site_mask="variant_mask_foobar",
        sample_mask="samples_mask_foobar",
    )
    return mat_sd, mask_sd, samples_mask, variant_mask
