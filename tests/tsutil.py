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
import tskit


def json_metadata_is_subset(metadata1, metadata2):
    metadata1_dict = json.loads(metadata1)
    if metadata1_dict is None:
        metadata1_dict = {}
    metadata2_dict = json.loads(metadata2)
    if metadata2_dict is None:
        metadata2_dict = {}
    return metadata1_dict.items() <= metadata2_dict.items()


def get_example_ts(sample_size, sequence_length, mutation_rate=10, random_seed=100):
    return msprime.simulate(
        sample_size,
        recombination_rate=1,
        mutation_rate=mutation_rate,
        length=sequence_length,
        random_seed=random_seed,
    )


def get_example_individuals_ts_with_metadata(
    n, ploidy, length, mutation_rate=1, *, strict_json_metadata=False, skip_last=True
):
    """
    For testing only, create a ts with lots of arbitrary metadata attached to sites,
    individuals & populations, and also set flags on individuals (*node* flags such as
    tskit.NODE_IS_SAMPLE are not expected to pass through the inference process, as
    they can be set during inference).

    Tsinfer requires metadata in JSON format, which doesn't allow zero-length metadata
    fields (instead it requires None to be encoded as b'{}'). If allow_empty_metadata
    is set to False, then empty metadata fields in the returned tree sequence will be
    set to ``b'null'`` to ensure all metadata is valid JSON.

    For testing purposes, we can set ``skip_last`` to check what happens if we have
    some samples that are not associated with an individual in the tree sequence.
    """
    ts = msprime.simulate(
        n * ploidy,
        recombination_rate=1,
        mutation_rate=mutation_rate,
        length=length,
        random_seed=100,
    )
    tables = ts.dump_tables()
    tables.populations.clear()
    no_metadata = b"{}" if strict_json_metadata else b""  # Fix when schemas in tsinfer
    for i in range(n):
        individual_meta = no_metadata
        pop_meta = no_metadata
        location = [i, i]
        if i % 2 == 0:
            # Add unicode metadata to every other individual: 8544+i = Roman numerals
            individual_meta = json.dumps({"unicode id": chr(8544 + i)}).encode()
            # TODO: flags should use np.iinfo(np.uint32).max. Change after solving issue
            # https://github.com/tskit-dev/tskit/issues/1027
            individual_flags = np.random.randint(0, np.iinfo(np.int32).max)
            # Also for populations: chr(127462) + chr(127462+i) give emoji flags
            pop_meta = json.dumps({"utf": chr(127462) + chr(127462 + i)}).encode()
        tables.populations.add_row(metadata=pop_meta)  # One pop for each individual
        if i < n - 1 or skip_last is False:
            tables.individuals.add_row(individual_flags, location, individual_meta)

    node_metadata = []
    node_populations = tables.nodes.population
    for node in ts.nodes():
        if node.is_sample():
            node_populations[node.id] = node.id // ploidy
        if node.id % 3 == 0:  # Scatter metadata into nodes: once every 3rd row
            node_metadata.append(json.dumps({"node id": node.id}).encode())
        else:
            node_metadata.append(no_metadata)
    tables.nodes.population = node_populations
    tables.nodes.packset_metadata(node_metadata)

    site_metadata = []
    for site in ts.sites():
        if site.id % 4 == 0:  # Scatter metadata into sites: once every 4th row
            site_metadata.append(json.dumps({"id": f"site {site.id}"}).encode())
        else:
            site_metadata.append(no_metadata)
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


def get_example_historical_sampled_ts(individual_times, ploidy=2, sequence_length=1):
    no_metadata = b"{}"  # Fix when schemas in tsinfer
    samples = [
        msprime.Sample(population=0, time=t)
        for t in individual_times
        for _ in range(ploidy)
    ]
    ts = msprime.simulate(
        samples=samples,
        recombination_rate=1,
        mutation_rate=10,
        length=sequence_length,
        random_seed=100,
    )
    tables = ts.dump_tables()
    # Add individuals
    nodes_individual = tables.nodes.individual
    individual_ids = []
    for _ in individual_times:
        individual_ids.append(tables.individuals.add_row(metadata=no_metadata))
    is_sample_node = (ts.tables.nodes.flags & tskit.NODE_IS_SAMPLE) != 0
    nodes_individual[is_sample_node] = np.repeat(individual_ids, ploidy)
    tables.nodes.individual = nodes_individual
    # force JSON-valid metadata
    tables.populations.packset_metadata([no_metadata])
    tables.nodes.packset_metadata([no_metadata] * ts.num_nodes)
    return tables.tree_sequence()
