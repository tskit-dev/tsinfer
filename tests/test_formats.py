#
# Copyright (C) 2018 University of Oxford
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
import unittest
import tempfile
import os.path
import datetime
import warnings
import json
import itertools

import numpy as np
import msprime
import numcodecs
import numcodecs.blosc as blosc
import zarr
import lmdb
import tskit

import tsinfer
import tsinfer.formats as formats
import tsinfer.exceptions as exceptions


class DataContainerMixin(object):
    """
    Common tests for the the data container classes."
    """
    def test_load(self):
        self.assertRaises(
            FileNotFoundError, self.tested_class.load, "/file/does/not/exist")
        if sys.platform != "win32":
            self.assertRaises(IsADirectoryError, self.tested_class.load, "/")
            bad_format_files = ["LICENSE", "/dev/urandom"]
        else:
            # Windows raises a PermissionError not IsADirectoryError when opening a dir
            self.assertRaises(PermissionError, self.tested_class.load, "/")
            # No /dev/urandom on Windows
            bad_format_files = ["LICENSE"]
        for bad_format_file in bad_format_files:
            self.assertTrue(os.path.exists(bad_format_file))
            self.assertRaises(
                exceptions.FileFormatError, self.tested_class.load, bad_format_file)


class TestSampleData(unittest.TestCase, DataContainerMixin):
    """
    Test cases for the sample data file format.
    """
    tested_class = formats.SampleData

    def get_example_ts(self, sample_size, sequence_length, mutation_rate=10):
        return msprime.simulate(
            sample_size, recombination_rate=1, mutation_rate=mutation_rate,
            length=sequence_length, random_seed=100)

    def get_example_individuals_ts_with_metadata(
            self, n_individuals, ploidy, sequence_length, mutation_rate=10):
        ts = msprime.simulate(
            n_individuals*ploidy, recombination_rate=1, mutation_rate=mutation_rate,
            length=sequence_length, random_seed=100)
        tables = ts.dump_tables()

        for i in range(n_individuals - 1):  # Create individuals, leaving one out
            individual_meta = None
            pop_meta = None
            if i % 2 == 0:
                # Add unicode metadata to every other individual: 8544+i = Roman numerals
                individual_meta = '{{"unicode id":"{}"}}'.format(chr(8544+i)).encode()
                # Also for populations: chr(127462) + chr(127462+i) give emoji flags
                pop_meta = '{{"utf":"{}"}}'.format(chr(127462) + chr(127462+i)).encode()
            tables.individuals.add_row(location=[i, i], metadata=individual_meta)
            tables.populations.add_row(metadata=pop_meta)  # One pop for each individual

        node_metadata = []
        for node in ts.nodes():
            if node.id % 3 == 0:  # Scatter metadata into nodes: once every 3rd row
                node_metadata.append('{{"node id":{}}}'.format(node.id).encode())
            else:
                node_metadata.append(b'')
        tables.nodes.packset_metadata(node_metadata)

        site_metadata = []
        for site in ts.sites():
            if site.id % 4 == 0:  # Scatter metadata into sites: once every 4th row
                site_metadata.append('{{"id":"site {}"}}'.format(site.id).encode())
            else:
                site_metadata.append(b'')
        tables.sites.packset_metadata(site_metadata)

        nodes_individual = tables.nodes.individual  # Assign individuals to sample nodes
        sample_individuals = np.repeat(
            np.arange(n_individuals, dtype=tables.nodes.individual.dtype), ploidy)
        # Leave the last sample nodes not assigned to an individual, for testing purposes
        sample_individuals[sample_individuals == n_individuals-1] = tskit.NULL
        nodes_individual[ts.samples()] = sample_individuals
        tables.nodes.individual = nodes_individual
        return tables.tree_sequence()

    def get_example_historical_sampled_ts(self, sample_times, sequence_length):
        samples = [msprime.Sample(population=0, time=t) for t in sample_times]
        return msprime.simulate(
            samples=samples, recombination_rate=1, mutation_rate=10,
            length=sequence_length, random_seed=100)

    def verify_data_round_trip(self, ts, input_file):
        self.assertGreater(ts.num_sites, 1)
        for pop in ts.populations():
            input_file.add_population(metadata=json.loads(pop.metadata or "{}"))
        # For testing, depend on the sample nodes being sorted by individual
        for i, group in itertools.groupby(ts.samples(), lambda n: ts.node(n).individual):
            nodes = [ts.node(nd) for nd in group]
            if i == tskit.NULL:
                for node in nodes:
                    input_file.add_individual(
                        ploidy=1, population=node.population, time=node.time,
                        samples_metadata=[json.loads(node.metadata or "{}")])
            else:
                input_file.add_individual(
                    ploidy=len(nodes), population=nodes[0].population,
                    metadata=json.loads(ts.individual(i).metadata or "{}"),
                    location=ts.individual(i).location,
                    time=nodes[0].time,
                    samples_metadata=[json.loads(n.metadata or "{}") for n in nodes])
        for v in ts.variants():
            t = None
            if len(v.site.mutations) == 1:
                t = ts.node(v.site.mutations[0].node).time
            input_file.add_site(
                v.site.position, v.genotypes, v.alleles,
                metadata=json.loads(v.site.metadata or "{}"), time=t)
        input_file.record_provenance("verify_data_round_trip")
        input_file.finalise()
        self.assertEqual(input_file.format_version, formats.SampleData.FORMAT_VERSION)
        self.assertEqual(input_file.format_name, formats.SampleData.FORMAT_NAME)
        self.assertEqual(input_file.num_samples, ts.num_samples)
        self.assertEqual(input_file.sequence_length, ts.sequence_length)
        self.assertEqual(input_file.num_sites, ts.num_sites)
        self.assertEqual(input_file.sites_genotypes.dtype, np.int8)
        self.assertEqual(input_file.sites_position.dtype, np.float64)
        # Take copies to avoid decompressing the data repeatedly.
        pop_metadata = input_file.populations_metadata[:]
        genotypes = input_file.sites_genotypes[:]
        position = input_file.sites_position[:]
        alleles = input_file.sites_alleles[:]
        site_times = input_file.sites_time[:]
        site_metadata = input_file.sites_metadata[:]
        location = input_file.individuals_location[:]
        individual_metadata = input_file.individuals_metadata[:]
        sample_time = input_file.individuals_time[:]
        for j, variant in enumerate(ts.variants()):
            self.assertEqual(variant.site.position, position[j])
            self.assertTrue(np.all(variant.genotypes == genotypes[j]))
            self.assertEqual(alleles[j], list(variant.alleles))
            if len(variant.site.mutations) == 1:
                the_time = ts.node(variant.site.mutations[0].node).time
                self.assertEqual(the_time, site_times[j])
            if variant.site.metadata:
                self.assertEqual(site_metadata[j], json.loads(variant.site.metadata))
        self.assertEqual(input_file.num_populations, ts.num_populations)
        for pop in ts.populations():
            if pop.metadata:
                self.assertEqual(pop_metadata[pop.id], json.loads(pop.metadata))
        if ts.num_individuals == 0:
            self.assertEqual(input_file.num_individuals, ts.num_samples)
        else:
            for individual in ts.individuals():
                self.assertTrue(np.all(individual.location == location[individual.id]))
                if individual.metadata:
                    self.assertEqual(
                        individual_metadata[individual.id],
                        json.loads(individual.metadata))
                for n in individual.nodes:
                    self.assertTrue(ts.node(n).time == sample_time[individual.id])

    @unittest.skipIf(sys.platform == "win32",
                     "windows simultaneous file permissions issue")
    def test_defaults_with_path(self):
        ts = self.get_example_ts(10, 10)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            input_file = formats.SampleData(
                path=filename, sequence_length=ts.sequence_length)
            self.verify_data_round_trip(ts, input_file)
            compressor = formats.DEFAULT_COMPRESSOR
            for _, array in input_file.arrays():
                self.assertEqual(array.compressor, compressor)
            with tsinfer.load(filename) as other:
                self.assertEqual(other, input_file)

    def test_bad_max_file_size(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            for bad_size in ['a', '', -1]:
                self.assertRaises(
                    ValueError, formats.SampleData, path=filename,
                    max_file_size=bad_size)
            for bad_size in [[1, 3], np.array([1, 2])]:
                self.assertRaises(
                    TypeError, formats.SampleData, path=filename, max_file_size=bad_size)

    def test_too_small_max_file_size_init(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            # Fail immediately if the max_size is so small we can't even create a file
            filename = os.path.join(tempdir, "samples.tmp")
            self.assertRaises(
                lmdb.MapFullError, formats.SampleData,
                path=filename, sequence_length=1, max_file_size=1)

    def test_too_small_max_file_size_add(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            base_size = 2**16  # Big enough to allow the initial file to be created
            # Fail during adding a large amount of data
            with self.assertRaises(lmdb.MapFullError):
                filename = os.path.join(tempdir, "samples.tmp")
                with formats.SampleData(path=filename, sequence_length=1,
                                        max_file_size=base_size) as small_sample_file:
                    small_sample_file.add_site(
                        0, alleles=['0', '1'],
                        genotypes=np.zeros(base_size, dtype=np.int8))
            # Work around https://github.com/tskit-dev/tsinfer/issues/201
            small_sample_file.data.store.close()

    def test_acceptable_max_file_size(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            # set a reasonably large number of sites and samples, and check we
            # don't bomb out
            n_samples = 2**10
            n_sites = 2**12
            np.random.seed(123)
            filename = os.path.join(tempdir, "samples.tmp")
            with formats.SampleData(path=filename, sequence_length=n_sites,
                                    compressor=False, max_file_size=None) as samples:
                for pos in range(n_sites):
                    samples.add_site(
                        pos, alleles=['0', '1'],
                        genotypes=np.random.randint(2, size=n_samples, dtype=np.int8))
            self.assertEqual(samples.num_sites, n_sites)
            self.assertEqual(samples.num_samples, n_samples)
            self.assertGreater(samples.file_size, n_samples*n_sites)
            samples.close()

    def test_defaults_no_path(self):
        ts = self.get_example_ts(10, 10)
        with formats.SampleData(sequence_length=ts.sequence_length) as sample_data:
            self.verify_data_round_trip(ts, sample_data)
            for _, array in sample_data.arrays():
                self.assertEqual(array.compressor, formats.DEFAULT_COMPRESSOR)

    def test_with_metadata_and_individuals(self):
        ts = self.get_example_individuals_ts_with_metadata(5, 2, 10, 1)
        with formats.SampleData(sequence_length=ts.sequence_length) as sample_data:
            self.verify_data_round_trip(ts, sample_data)

    def test_from_tree_sequence_bad_times(self):
        n_individuals = 4
        sample_times = np.arange(n_individuals * 2)  # Diploids
        ts = self.get_example_historical_sampled_ts(sample_times, 10)
        tables = ts.dump_tables()
        for _ in range(n_individuals):
            tables.individuals.add_row()
        # Associate nodes at different times with a single individual
        nodes_individual = tables.nodes.individual
        nodes_individual[ts.samples()] = np.repeat(
            np.arange(n_individuals, dtype=tables.nodes.individual.dtype), 2)
        tables.nodes.individual = nodes_individual
        bad_ts = tables.tree_sequence()
        self.assertRaises(ValueError, formats.SampleData.from_tree_sequence, bad_ts)

    def test_from_tree_sequence_bad_populations(self):
        n_individuals = 4
        ts = self.get_example_ts(n_individuals * 2, 10, 1)  # Diploids
        tables = ts.dump_tables()
        # Associate each sample node with a new population
        for _ in range(n_individuals * 2):
            tables.populations.add_row()
        nodes_population = tables.nodes.population
        nodes_population[ts.samples()] = np.arange(n_individuals * 2)
        tables.nodes.population = nodes_population
        for _ in range(n_individuals):
            tables.individuals.add_row()
        # Associate nodes with individuals
        nodes_individual = tables.nodes.individual
        nodes_individual[ts.samples()] = np.repeat(
            np.arange(n_individuals, dtype=tables.nodes.individual.dtype), 2)
        tables.nodes.individual = nodes_individual
        bad_ts = tables.tree_sequence()
        self.assertRaises(ValueError, formats.SampleData.from_tree_sequence, bad_ts)

    def test_from_tree_sequence_simple(self):
        ts = self.get_example_ts(10, 10, 1)
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        sd2 = formats.SampleData.from_tree_sequence(ts)
        self.assertTrue(sd1.data_equal(sd2))

    def test_from_tree_sequence_variable_allele_number(self):
        ts = self.get_example_ts(10, 10)
        # Create > 2 alles by scattering mutations on the tree nodes at the first site
        tables = ts.dump_tables()
        focal_site = ts.site(0)
        tree = ts.at(focal_site.position)
        # Reset the initial mutation to lie above the root, for correct mutation order
        nodes = tables.mutations.node
        nodes[0] = tree.root
        tables.mutations.node = nodes
        for i, node in enumerate(tree.nodes()):
            if i % 3:  # add above every 3rd node - should create many alleles
                tables.mutations.add_row(site=0, node=node, derived_state=str(i+2))
        # Create < 2 alleles by adding a non-variable site at the end
        extra_last_pos = (ts.site(ts.num_sites - 1).position + ts.sequence_length) / 2
        tables.sites.add_row(position=extra_last_pos, ancestral_state="0")
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        self.assertGreater(len(ts.site(0).mutations), 1)
        self.assertEqual(len(ts.site(ts.num_sites-1).mutations), 0)
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        self.assertFalse(sd1.sites_inference[0])
        self.assertFalse(sd1.sites_inference[sd1.num_sites-1])

        num_alleles = sd1.num_alleles()
        for var in ts.variants():
            self.assertEqual(len(var.alleles), num_alleles[var.site.id])
        for inference_sites in [None, True, False]:
            for var, num_alleles in itertools.zip_longest(
                    sd1.variants(inference_sites), sd1.num_alleles(inference_sites)):
                self.assertEqual(len(var.alleles), num_alleles)
        sd2 = formats.SampleData.from_tree_sequence(ts)
        self.assertTrue(sd1.data_equal(sd2))

    def test_from_tree_sequence_with_metadata(self):
        ts = self.get_example_individuals_ts_with_metadata(5, 2, 10)
        # Remove individuals
        tables = ts.dump_tables()
        tables.individuals.clear()
        tables.nodes.individual = np.full(
            ts.num_nodes, tskit.NULL, dtype=tables.nodes.individual.dtype)
        ts_no_individuals = tables.tree_sequence()
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts_no_individuals, sd1)
        sd2 = formats.SampleData.from_tree_sequence(ts_no_individuals)
        self.assertTrue(sd1.data_equal(sd2))

    def test_from_tree_sequence_with_metadata_and_individuals(self):
        ts = self.get_example_individuals_ts_with_metadata(5, 3, 10)
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        sd2 = formats.SampleData.from_tree_sequence(ts)
        self.assertTrue(sd1.data_equal(sd2))

    def test_from_tree_sequence_omitting_metadata(self):
        ts = self.get_example_ts(10, 10, 1)
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        # Add non JSON metadata
        tables = ts.dump_tables()
        tables.sites.packset_metadata([b"Not JSON!" for _ in range(ts.num_sites)])
        ts = tables.tree_sequence()
        self.assertRaises(
            json.decoder.JSONDecodeError, formats.SampleData.from_tree_sequence, ts)
        sd2 = formats.SampleData.from_tree_sequence(ts, use_metadata=False)
        # Copy tests from SampleData.data_equal, except the metadata
        self.assertTrue(np.all(sd2.individuals_time[:] == sd1.individuals_time[:]))
        self.assertTrue(np.all(sd2.samples_individual[:] == sd1.samples_individual[:]))
        self.assertTrue(np.all(sd2.samples_population[:] == sd1.samples_population[:]))
        self.assertTrue(np.all(sd2.sites_position[:] == sd1.sites_position[:]))
        self.assertTrue(np.all(sd2.sites_inference[:] == sd1.sites_inference[:]))
        self.assertTrue(np.all(sd2.sites_genotypes[:] == sd1.sites_genotypes[:]))
        self.assertTrue(np.all(sd2.sites_time[:] == sd1.sites_time[:]))
        self.assertTrue(np.all(sd2.populations_metadata[:] == {}))
        self.assertTrue(np.all(sd2.individuals_metadata[:] == {}))
        self.assertTrue(np.all(sd2.samples_metadata[:] == {}))
        self.assertTrue(np.all(sd2.sites_metadata[:] == {}))

    def test_from_historical_tree_sequence(self):
        sample_times = np.arange(10)
        ts = self.get_example_historical_sampled_ts(sample_times, 10)
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        sd2 = formats.SampleData.from_tree_sequence(ts)
        self.assertTrue(sd1.data_equal(sd2))

    def test_chunk_size(self):
        ts = self.get_example_ts(4, 2)
        self.assertGreater(ts.num_sites, 50)
        for chunk_size in [1, 2, 3, ts.num_sites - 1, ts.num_sites, ts.num_sites + 1]:
            input_file = formats.SampleData(
                sequence_length=ts.sequence_length, chunk_size=chunk_size)
            self.verify_data_round_trip(ts, input_file)
            for name, array in input_file.arrays():
                self.assertEqual(array.chunks[0], chunk_size)
                if name.endswith("genotypes"):
                    self.assertEqual(array.chunks[1], chunk_size)

    def test_filename(self):
        ts = self.get_example_ts(14, 15)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            input_file = formats.SampleData(
                sequence_length=ts.sequence_length, path=filename)
            self.assertTrue(os.path.exists(filename))
            self.assertFalse(os.path.isdir(filename))
            self.verify_data_round_trip(ts, input_file)
            # Make a copy so that we can close the original and reopen it
            # without hitting simultaneous file access problems on Windows
            input_copy = input_file.copy(path=None)
            input_file.close()
            other_input_file = formats.SampleData.load(filename)
            self.assertIsNot(other_input_file, input_copy)
            # Can't use eq here because UUIDs will not be equal.
            self.assertTrue(other_input_file.data_equal(input_copy))
            other_input_file.close()

    def test_chunk_size_file_equal(self):
        ts = self.get_example_ts(13, 15)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            files = []
            for chunk_size in [5, 7]:
                filename = os.path.join(tempdir, "samples_{}.tmp".format(chunk_size))
                files.append(filename)
                with formats.SampleData(
                        sequence_length=ts.sequence_length, path=filename,
                        chunk_size=chunk_size) as input_file:
                    self.verify_data_round_trip(ts, input_file)
                    self.assertEqual(
                        input_file.sites_genotypes.chunks, (chunk_size, chunk_size))
            # Now reload the files and check they are equal
            with formats.SampleData.load(files[0]) as input_file0:
                with formats.SampleData.load(files[1]) as input_file1:
                    # Can't use eq here because UUIDs will not be equal.
                    self.assertTrue(input_file0.data_equal(input_file1))

    def test_compressor(self):
        ts = self.get_example_ts(11, 17)
        compressors = [
           None, formats.DEFAULT_COMPRESSOR,
           blosc.Blosc(cname='zlib', clevel=1, shuffle=blosc.NOSHUFFLE)
        ]
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            for i, compressor in enumerate(compressors):
                filename = os.path.join(tempdir, "samples_{}.tmp".format(i))
                for path in [None, filename]:
                    with formats.SampleData(sequence_length=ts.sequence_length,
                                            path=path, compressor=compressor) as samples:
                        self.verify_data_round_trip(ts, samples)
                        for _, array in samples.arrays():
                            self.assertEqual(array.compressor, compressor)

    def test_multichar_alleles(self):
        ts = self.get_example_ts(5, 17)
        t = ts.dump_tables()
        t.sites.clear()
        t.mutations.clear()
        for site in ts.sites():
            t.sites.add_row(site.position, ancestral_state="A" * (site.id + 1))
            for mutation in site.mutations:
                t.mutations.add_row(
                    site=site.id, node=mutation.node, derived_state="T" * site.id)
        ts = t.tree_sequence()
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)

    def test_str(self):
        ts = self.get_example_ts(5, 3)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        self.assertGreater(len(str(input_file)), 0)

    def test_eq(self):
        ts = self.get_example_ts(5, 3)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        self.assertTrue(input_file == input_file)
        self.assertFalse(input_file == [])
        self.assertFalse({} == input_file)

    def test_provenance(self):
        ts = self.get_example_ts(4, 3)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        self.assertEqual(input_file.num_provenances, 1)
        timestamp = input_file.provenances_timestamp[0]
        iso = datetime.datetime.now().isoformat()
        self.assertEqual(timestamp.split("T")[0], iso.split("T")[0])
        record = input_file.provenances_record[0]
        self.assertEqual(record["software"]["name"], "tsinfer")
        a = list(input_file.provenances())
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0][0], timestamp)
        self.assertEqual(a[0][1], record)

    def test_variant_errors(self):
        input_file = formats.SampleData(sequence_length=10)
        genotypes = [0, 0]
        input_file.add_site(0, alleles=["0", "1"], genotypes=genotypes)
        for bad_position in [-1, 10, 100]:
            self.assertRaises(
                ValueError, input_file.add_site, position=bad_position,
                alleles=["0", "1"], genotypes=genotypes)
        for bad_genotypes in [[0, 2], [-2, 0], [], [0], [0, 0, 0]]:
            self.assertRaises(
                ValueError, input_file.add_site, position=1,
                alleles=["0", "1"], genotypes=bad_genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=1, inference=True,
            alleles=["0", "1", "2"], genotypes=[0, 1])
        self.assertRaises(
            ValueError, input_file.add_site, position=1,
            alleles=["0"], genotypes=[0, 1])
        self.assertRaises(
            ValueError, input_file.add_site, position=1,
            alleles=["0", "1"], genotypes=[0, 2])
        self.assertRaises(
            ValueError, input_file.add_site, position=1,
            alleles=["0", "0"], genotypes=[0, 2])

    def test_invalid_inference_sites(self):
        # Trying to add singletons or fixed sites as inference sites
        # raise and error
        input_file = formats.SampleData()
        # Make sure this is OK
        input_file.add_site(0, [0, 1, 1, tskit.MISSING_DATA], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[0, 0, 0, 0], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[1, 0, 0, 0], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[1, 1, 1, 1], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[tskit.MISSING_DATA, 0, 0, 0], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[tskit.MISSING_DATA, 1, 1, 1], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[tskit.MISSING_DATA, 0, 1, 0], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[tskit.MISSING_DATA] * 4, inference=True)
        # Check we can still add at pos 1
        input_file.add_site(
            position=1, genotypes=[1, 0, 1, tskit.MISSING_DATA], inference=True)

    def test_duplicate_sites(self):
        # Duplicate sites are not accepted.
        input_file = formats.SampleData()
        alleles = ["0", "1"]
        genotypes = [0, 1, 1]
        input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=0, alleles=alleles,
            genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=0, alleles=alleles,
            genotypes=genotypes)
        input_file.add_site(position=1, alleles=alleles, genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=1, alleles=alleles,
            genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=1, alleles=alleles,
            genotypes=genotypes)

    def test_unordered_sites(self):
        # Sites must be specified in sorted order.
        input_file = formats.SampleData()
        alleles = ["0", "1"]
        genotypes = [0, 1, 1]
        input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        input_file.add_site(position=1, alleles=alleles, genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=0.5, alleles=alleles,
            genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=0.9999, alleles=alleles,
            genotypes=genotypes)
        input_file.add_site(position=2, alleles=alleles, genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=0.5, alleles=alleles,
            genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=1.88, alleles=alleles,
            genotypes=genotypes)

    def test_insufficient_samples(self):
        sample_data = formats.SampleData(sequence_length=10)
        self.assertRaises(
            ValueError, sample_data.add_site, position=0, alleles=["0", "1"],
            genotypes=[])
        sample_data = formats.SampleData(sequence_length=10)
        self.assertRaises(
            ValueError, sample_data.add_site, position=0, alleles=["0", "1"],
            genotypes=[0])
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual(ploidy=3)
        self.assertRaises(
            ValueError, sample_data.add_site, position=0, alleles=["0", "1"],
            genotypes=[0])
        sample_data = formats.SampleData(sequence_length=10)
        self.assertRaises(
            ValueError, sample_data.add_individual, ploidy=3, samples_metadata=[None])

    def test_add_population_errors(self):
        sample_data = formats.SampleData(sequence_length=10)
        self.assertRaises(
            TypeError, sample_data.add_population, metadata=234)

    def test_add_state_machine(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual()
        self.assertRaises(ValueError, sample_data.add_population)

        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_site(0.1, genotypes=[0, 1])
        self.assertRaises(ValueError, sample_data.add_population)
        self.assertRaises(ValueError, sample_data.add_individual)

    def test_add_population_return(self):
        sample_data = formats.SampleData(sequence_length=10)
        pid = sample_data.add_population({"a": 1})
        self.assertEqual(pid, 0)
        pid = sample_data.add_population()
        self.assertEqual(pid, 1)
        pid = sample_data.add_population()
        self.assertEqual(pid, 2)

    def test_population_metadata(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_population({"a": 1})
        sample_data.add_population({"b": 2})
        sample_data.add_individual(population=0)
        sample_data.add_individual(population=1)
        sample_data.add_site(position=0, genotypes=[0, 1])
        sample_data.finalise()

        self.assertEqual(sample_data.populations_metadata[0], {"a": 1})
        self.assertEqual(sample_data.populations_metadata[1], {"b": 2})
        self.assertEqual(sample_data.samples_population[0], 0)
        self.assertEqual(sample_data.samples_population[1], 1)

    def test_individual_metadata(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_population({"a": 1})
        sample_data.add_population({"b": 2})
        sample_data.add_individual(population=0)
        sample_data.add_individual(population=1)
        sample_data.add_site(0, [0, 0])
        sample_data.finalise()
        self.assertEqual(sample_data.populations_metadata[0], {"a": 1})
        self.assertEqual(sample_data.populations_metadata[1], {"b": 2})

    def test_add_individual_time(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual()
        sample_data.add_individual(time=0.5)
        sample_data.add_site(0, [0, 0])
        sample_data.finalise()
        self.assertEqual(sample_data.individuals_time[0], 0)
        self.assertEqual(sample_data.individuals_time[1], 0.5)

    def test_add_individual_return(self):
        sample_data = formats.SampleData(sequence_length=10)
        iid, sids = sample_data.add_individual()
        self.assertEqual(iid, 0)
        self.assertEqual(sids, [0])
        iid, sids = sample_data.add_individual(ploidy=1)
        self.assertEqual(iid, 1)
        self.assertEqual(sids, [1])
        iid, sids = sample_data.add_individual(ploidy=5)
        self.assertEqual(iid, 2)
        self.assertEqual(sids, [2, 3, 4, 5, 6])

    def test_numpy_position(self):
        pos = np.array([5.1, 100], dtype=np.float64)
        with formats.SampleData() as sample_data:
            sample_data.add_site(pos[0], [0, 0])
        self.assertEqual(sample_data.sequence_length, 6.1)
        with formats.SampleData(sequence_length=pos[1]) as sample_data:
            sample_data.add_site(pos[0], [0, 0])
        self.assertEqual(sample_data.sequence_length, 100)

    def test_samples_metadata(self):
        with formats.SampleData(sequence_length=10) as sample_data:
            sample_data.add_individual(ploidy=2)
            sample_data.add_site(0, [0, 0])
        individuals_metadata = sample_data.individuals_metadata[:]
        self.assertEqual(len(individuals_metadata), 1)
        self.assertEqual(individuals_metadata[0], {})

        samples_metadata = sample_data.samples_metadata[:]
        self.assertEqual(len(samples_metadata), 2)
        self.assertEqual(samples_metadata[0], {})
        self.assertEqual(samples_metadata[1], {})

    def test_add_individual_errors(self):
        sample_data = formats.SampleData(sequence_length=10)
        self.assertRaises(TypeError, sample_data.add_individual, metadata=234)
        self.assertRaises(ValueError, sample_data.add_individual, population=0)
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_population()
        self.assertRaises(ValueError, sample_data.add_individual, population=1)
        self.assertRaises(ValueError, sample_data.add_individual, location="x234")
        self.assertRaises(ValueError, sample_data.add_individual, ploidy=0)
        self.assertRaises(ValueError, sample_data.add_individual, time=None)
        self.assertRaises(ValueError, sample_data.add_individual, time=[1, 2])

    def test_no_data(self):
        sample_data = formats.SampleData(sequence_length=10)
        self.assertRaises(ValueError, sample_data.finalise)

    def test_no_sites(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual()
        self.assertRaises(ValueError, sample_data.finalise)

    def test_add_site_return(self):
        sample_data = formats.SampleData(sequence_length=10)
        sid = sample_data.add_site(0, [0, 1])
        self.assertEqual(sid, 0)
        sid = sample_data.add_site(1, [0, 1])
        self.assertEqual(sid, 1)

    def test_genotypes(self):
        ts = self.get_example_ts(13, 12)
        self.assertGreater(ts.num_sites, 1)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        for v in ts.variants():
            input_file.add_site(v.site.position, v.genotypes, v.alleles)
        input_file.finalise()

        self.assertLess(np.sum(input_file.sites_inference[:]), ts.num_sites)
        all_genotypes = input_file.genotypes()
        for v in ts.variants():
            site_id, g = next(all_genotypes)
            self.assertEqual(site_id, v.site.id)
            self.assertTrue(np.array_equal(g, v.genotypes))
        self.assertIsNone(next(all_genotypes, None), None)

        inference_genotypes = input_file.genotypes(inference_sites=True)
        non_inference_genotypes = input_file.genotypes(inference_sites=False)
        for v in ts.variants():
            freq = np.sum(v.genotypes)
            if 1 < freq < ts.num_samples:
                site_id, g = next(inference_genotypes)
                self.assertEqual(site_id, v.site.id)
                self.assertTrue(np.array_equal(g, v.genotypes))
            else:
                site_id, g = next(non_inference_genotypes)
                self.assertEqual(site_id, v.site.id)
                self.assertTrue(np.array_equal(g, v.genotypes))
        self.assertIsNone(next(inference_genotypes, None), None)

    def test_sites(self):
        ts = self.get_example_ts(11, 15)
        self.assertGreater(ts.num_sites, 1)
        input_file = formats.SampleData.from_tree_sequence(ts)

        all_sites = input_file.sites()
        for s1, variant in zip(ts.sites(), ts.variants()):
            s2 = next(all_sites)
            self.assertEqual(s1.id, s2.id)
            self.assertEqual(s1.position, s2.position)
            self.assertEqual(s1.ancestral_state, s2.ancestral_state)
            self.assertEqual(variant.alleles, s2.alleles)
        self.assertIsNone(next(all_sites, None), None)

        inference_sites = input_file.sites(inference_sites=True)
        non_inference_sites = input_file.sites(inference_sites=False)
        inference = input_file.sites_inference[:]
        for j, s1 in enumerate(ts.sites()):
            if inference[j]:
                s2 = next(inference_sites)
            else:
                s2 = next(non_inference_sites)
            self.assertEqual(s1.id, s2.id)
            self.assertEqual(s1.position, s2.position)
            self.assertEqual(s1.ancestral_state, s2.ancestral_state)
            self.assertEqual(inference[j], s2.inference)
        self.assertIsNone(next(inference_sites, None), None)
        self.assertIsNone(next(non_inference_sites, None), None)

    def test_variants(self):
        ts = self.get_example_ts(11, 15)
        self.assertGreater(ts.num_sites, 1)
        input_file = formats.SampleData.from_tree_sequence(ts)

        all_variants = input_file.variants()
        for v1 in ts.variants():
            v2 = next(all_variants)
            self.assertEqual(v1.site.id, v2.site.id)
            self.assertEqual(v1.site.position, v2.site.position)
            self.assertEqual(v1.site.ancestral_state, v2.site.ancestral_state)
            self.assertEqual(v1.alleles, v2.alleles)
            self.assertTrue(np.array_equal(v1.genotypes, v2.genotypes))
        self.assertIsNone(next(all_variants, None), None)

        inference_variants = input_file.variants(inference_sites=True)
        non_inference_variants = input_file.variants(inference_sites=False)
        for v1 in ts.variants():
            freq = np.sum(v1.genotypes)
            if 1 < freq < ts.num_samples:
                v2 = next(inference_variants)
                inference = 1
            else:
                v2 = next(non_inference_variants)
                inference = 0
            self.assertEqual(v1.site.id, v2.site.id)
            self.assertEqual(v1.site.position, v2.site.position)
            self.assertEqual(v1.site.ancestral_state, v2.site.ancestral_state)
            self.assertEqual(inference, v2.site.inference)
            self.assertEqual(v1.alleles, v2.alleles)
            self.assertTrue(np.array_equal(v1.genotypes, v2.genotypes))
        self.assertIsNone(next(inference_variants, None), None)
        self.assertIsNone(next(non_inference_variants, None), None)

    def test_all_haplotypes(self):
        ts = self.get_example_ts(13, 12)
        self.assertGreater(ts.num_sites, 1)
        input_file = formats.SampleData.from_tree_sequence(ts)

        G = ts.genotype_matrix()
        j = 0
        for index, h in input_file.haplotypes():
            self.assertTrue(np.array_equal(h, G[:, j]))
            self.assertEqual(index, j)
            j += 1
        self.assertEqual(j, ts.num_samples)

        j = 0
        for index, h in input_file.haplotypes(np.arange(ts.num_samples)):
            self.assertTrue(np.array_equal(h, G[:, j]))
            self.assertEqual(index, j)
            j += 1
        self.assertEqual(j, ts.num_samples)

        selection = input_file.sites_inference[:]
        j = 0
        for index, h in input_file.haplotypes(inference_sites=True):
            self.assertTrue(np.array_equal(h, G[selection, j]))
            self.assertEqual(index, j)
            j += 1
        self.assertEqual(j, ts.num_samples)

        selection = input_file.sites_inference[:] == 0
        j = 0
        for index, h in input_file.haplotypes(inference_sites=False):
            self.assertTrue(np.array_equal(h, G[selection, j]))
            self.assertEqual(index, j)
            j += 1
        self.assertEqual(j, ts.num_samples)

    def test_haplotypes_index_errors(self):
        ts = self.get_example_ts(13, 12)
        self.assertGreater(ts.num_sites, 1)
        input_file = formats.SampleData.from_tree_sequence(ts)
        self.assertRaises(ValueError, list, input_file.haplotypes([1, 0]))
        self.assertRaises(ValueError, list, input_file.haplotypes([0, 1, 2, -1]))
        self.assertRaises(ValueError, list, input_file.haplotypes([0, 1, 2, 2]))
        self.assertRaises(ValueError, list, input_file.haplotypes(np.arange(10)[::-1]))

        # Out of bounds sample index.
        self.assertRaises(ValueError, list, input_file.haplotypes([13]))
        self.assertRaises(ValueError, list, input_file.haplotypes([3, 14]))

    def test_haplotypes_subsets(self):
        ts = self.get_example_ts(25, 12)
        self.assertGreater(ts.num_sites, 1)
        input_file = formats.SampleData.from_tree_sequence(ts)

        subsets = [
            [],
            [0], [1], [21], [22],
            [0, 1], [1, 2], [4, 5], [10, 11], [23, 24],
            [0, 1, 2], [1, 2, 3], [4, 5, 6], [1, 10, 20],
            [0, 1, 2, 3], [0, 10, 11, 20], [10, 15, 20, 21],
            np.arange(24), 1 + np.arange(24),
            np.arange(25),
        ]
        G = ts.genotype_matrix()
        for subset in subsets:
            j = 0
            for index, h in input_file.haplotypes(subset):
                self.assertTrue(np.array_equal(h, G[:, subset[j]]))
                self.assertEqual(index, subset[j])
                j += 1
            self.assertEqual(j, len(subset))

    def test_invariant_sites(self):
        n = 10
        m = 10
        for value in [0, 1]:
            G = np.zeros((m, n), dtype=np.int8) + value
            input_file = formats.SampleData(sequence_length=m)
            for j in range(m):
                input_file.add_site(j, G[j])
            input_file.finalise()
            self.assertEqual(input_file.num_sites, m)
            self.assertTrue(np.all(~input_file.sites_inference[:]))

    def test_ts_with_invariant_sites(self):
        ts = self.get_example_ts(5, 3)
        t = ts.dump_tables()
        positions = set(site.position for site in ts.sites())
        for j in range(10):
            pos = 1 / (j + 1)
            if pos not in positions:
                t.sites.add_row(position=pos, ancestral_state="0")
                positions.add(pos)
        self.assertGreater(len(positions), ts.num_sites)
        t.sort()
        ts = t.tree_sequence()

        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        self.assertGreater(len(str(input_file)), 0)

    def test_ts_with_root_mutations(self):
        ts = self.get_example_ts(5, 3)
        t = ts.dump_tables()
        positions = set(site.position for site in ts.sites())
        for tree in ts.trees():
            pos = tree.interval[0]
            if pos not in positions:
                site_id = t.sites.add_row(position=pos, ancestral_state="0")
                t.mutations.add_row(site=site_id, node=tree.root, derived_state="1")
                positions.add(pos)
        self.assertGreater(len(positions), ts.num_sites)
        t.sort()
        ts = t.tree_sequence()

        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)

    def test_copy_error_wrong_mode(self):
        data = formats.SampleData()
        self.assertRaises(ValueError, data.copy)
        data = formats.SampleData()
        self.assertRaises(ValueError, data.copy)
        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1])
        data.finalise()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Zarr emits a FutureWarning about object arrays here.
            copy = data.copy()
        self.assertRaises(ValueError, copy.copy)

    def test_error_not_build_mode(self):
        # Cannot build after finalising.
        with formats.SampleData() as input_file:
            alleles = ["0", "1"]
            genotypes = [0, 1, 1]
            input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, alleles=alleles, genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_provenance,
            datetime.datetime.now().isoformat(), {})
        self.assertRaises(ValueError, input_file._check_build_mode)

    def test_error_not_write_mode(self):
        # Cannot finalise twice.
        with formats.SampleData() as input_file:
            alleles = ["0", "1"]
            genotypes = [0, 1, 1]
            input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        self.assertRaises(ValueError, input_file.finalise)
        self.assertRaises(ValueError, input_file._check_write_modes)

    def test_error_not_edit_mode(self):
        # Can edit after copy but not after finalising.
        with formats.SampleData() as input_file:
            alleles = ["0", "1"]
            genotypes = [0, 1, 1]
            input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        editable_sample_data = input_file.copy()
        # Try editing: use setter in the normal way
        editable_sample_data.sites_inference = [True]
        editable_sample_data.sites_time = [0.0]
        # Try editing: use setter via setattr
        setattr(editable_sample_data, 'sites_inference', [False])
        setattr(editable_sample_data, 'sites_time', [1.0])
        editable_sample_data.add_provenance(datetime.datetime.now().isoformat(), {})
        editable_sample_data.finalise()
        self.assertRaises(
            ValueError, setattr, editable_sample_data, 'sites_inference', [True])
        self.assertRaises(
            ValueError, setattr, editable_sample_data, 'sites_time', [0.0])
        self.assertRaises(
            ValueError, editable_sample_data.add_provenance,
            datetime.datetime.now().isoformat(), {})

    def test_copy_new_uuid(self):
        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1])
        data.finalise()
        copy = data.copy()
        copy.finalise()
        self.assertNotEqual(copy.uuid, data.uuid)
        self.assertTrue(copy.data_equal(data))

    def test_copy_update_sites_inference(self):
        with formats.SampleData() as data:
            for j in range(4):
                data.add_site(position=j, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        self.assertEqual(list(data.sites_inference), [True, True, True, True])

        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            for copy_path in [None, filename]:
                copy = data.copy(path=copy_path)
                copy.finalise()
                self.assertTrue(copy.data_equal(data))
                if copy_path is not None:
                    copy.close()
                with data.copy(path=copy_path) as copy:
                    inference = [False, True, False, True]
                    copy.sites_inference = inference
                self.assertFalse(copy.data_equal(data))
                self.assertEqual(list(copy.sites_inference), inference)
                self.assertEqual(list(data.sites_inference), [True, True, True, True])
                if copy_path is not None:
                    copy.close()

    def test_copy_update_sites_time(self):
        with formats.SampleData() as data:
            for j in range(4):
                data.add_site(position=j, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        self.assertEqual(list(data.sites_time), [2.0, 2.0, 2.0, 2.0])  # Freq == 2.0

        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            for copy_path in [None, filename]:
                copy = data.copy(path=copy_path)
                copy.finalise()
                self.assertTrue(copy.data_equal(data))
                if copy_path is not None:
                    copy.close()
                with data.copy(path=copy_path) as copy:
                    time = [0.0, 1.1, 2.2, 3.3]
                    copy.sites_time = time
                self.assertFalse(copy.data_equal(data))
                self.assertEqual(list(copy.sites_time), time)
                self.assertEqual(list(data.sites_time), [2.0, 2.0, 2.0, 2.0])
                if copy_path is not None:
                    copy.close()

    def test_update_sites_inference_bad_data(self):
        def set_value(data, value):
            data.sites_inference = value

        data = formats.SampleData()
        for j in range(4):
            data.add_site(position=j, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        data.finalise()
        self.assertEqual(list(data.sites_inference), [True, True, True, True])
        copy = data.copy()
        for bad_shape in [[], np.arange(100), np.zeros((2, 2))]:
            self.assertRaises((ValueError, TypeError), set_value, copy, bad_shape)
        bad_data = [
            ["a", "b", "c", "d"], [2**10 for _ in range(4)], [0, 1, 0, 2],
            [0, 0, 0, -1], [0, 0.5, 0.2]]
        for a in bad_data:
            self.assertRaises((ValueError, TypeError, OverflowError), set_value, copy, a)

    def test_update_sites_time_bad_data(self):
        def set_value(data, value):
            data.sites_time = value

        data = formats.SampleData()
        for j in range(4):
            data.add_site(position=j, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        data.finalise()
        self.assertEqual(list(data.sites_time), [2.0, 2.0, 2.0, 2.0])
        copy = data.copy()
        for bad_shape in [[], np.arange(100, dtype=np.float64), np.zeros((2, 2))]:
            self.assertRaises((ValueError, TypeError), set_value, copy, bad_shape)
        for bad_data in [["a", "b", "c", "d"]]:
            self.assertRaises(ValueError, set_value, copy, bad_data)

    def test_update_sites_inference_non_copy_mode(self):
        def set_value(data, value):
            data.sites_inference = value

        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        self.assertRaises(ValueError, set_value, data, [True])
        data.finalise()
        self.assertRaises(ValueError, set_value, data, [True])

    def test_update_sites_time_non_copy_mode(self):
        def set_value(data, value):
            data.sites_time = value

        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        self.assertRaises(ValueError, set_value, data, [1.0])
        data.finalise()
        self.assertRaises(ValueError, set_value, data, [1.0])

    @unittest.skipIf(sys.platform == "win32",
                     "windows simultaneous file permissions issue")
    def test_overwrite_partial(self):
        # Check that we can correctly overwrite partially written and
        # unfinalised files. See
        # https://github.com/tskit-dev/tsinfer/issues/64
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            for _ in range(10):
                data = formats.SampleData(path=filename)
                for j in range(10):
                    data.add_site(j, [0, 1])

    def test_sequence_length(self):
        data = formats.SampleData(sequence_length=2)
        data.add_site(position=0, genotypes=[0, 1, 1, 0])
        data.finalise()
        self.assertEqual(data.sequence_length, 2)
        # The default sequence length should be the last site + 1.
        data = formats.SampleData()
        data.add_site(position=0, genotypes=[0, 1, 1, 0])
        data.finalise()
        self.assertEqual(data.sequence_length, 1)

    def test_too_many_alleles(self):
        with tsinfer.SampleData() as sample_data:
            sample_data.add_site(0, [0, 0], alleles=[str(x) for x in range(64)])
            for num_alleles in [65, 66, 100]:
                with self.assertRaises(ValueError):
                    sample_data.add_site(
                        0, [0, 0], alleles=[str(x) for x in range(num_alleles)])

    def test_missing_data(self):
        u = tskit.MISSING_DATA
        sites_by_samples = np.array([
            [u, u, u, 1, 1, 0, 1, 1, 1],
            [u, u, u, 1, 1, 0, 1, 1, 0],
            [u, u, u, 1, 0, 1, 1, 0, 1],
            [u, 0, 0, 1, 1, 1, 1, u, u],
            [u, 0, 1, 1, 1, 0, 1, u, u],
            [u, 1, 1, 0, 0, 0, 0, u, u]
            ], dtype=np.int8)
        with tsinfer.SampleData() as data:
            for col in range(sites_by_samples.shape[1]):
                data.add_site(col, sites_by_samples[:, col])

        self.assertEqual(data.sequence_length, 9.0)
        self.assertEqual(data.num_sites, 9)
        # First site is a entirely missing, second is singleton with missing data =>
        # neither should be marked for inference
        inference_sites = data.sites_inference[:]
        self.assertEqual(inference_sites[0], 0)  # Entirely missing data
        self.assertEqual(inference_sites[1], 0)  # Singleton with missing data
        for i in inference_sites[2:]:
            self.assertEqual(i, 1)


class TestAncestorData(unittest.TestCase, DataContainerMixin):
    """
    Test cases for the sample data file format.
    """
    tested_class = formats.AncestorData

    def get_example_data(self, sample_size, sequence_length, num_ancestors):
        ts = msprime.simulate(
            sample_size, recombination_rate=1, mutation_rate=10,
            length=sequence_length, random_seed=100)
        sample_data = formats.SampleData.from_tree_sequence(ts)

        num_sites = sample_data.num_inference_sites
        ancestors = []
        for j in range(num_ancestors):
            haplotype = np.full(num_sites, tskit.MISSING_DATA, dtype=np.int8)
            start = j
            end = max(num_sites - j, start + 1)
            self.assertLess(start, end)
            haplotype[start: end] = 0
            if start + j < end:
                haplotype[start + j: end] = 1
            self.assertTrue(np.all(haplotype[:start] == tskit.MISSING_DATA))
            self.assertTrue(np.all(haplotype[end:] == tskit.MISSING_DATA))
            focal_sites = np.array([start + k for k in range(j)], dtype=np.int32)
            focal_sites = focal_sites[focal_sites < end]
            haplotype[focal_sites] = 1
            ancestors.append((start, end, 2 * j + 1, focal_sites, haplotype))
        return sample_data, ancestors

    def verify_data_round_trip(self, sample_data, ancestor_data, ancestors):
        for start, end, t, focal_sites, haplotype in ancestors:
            ancestor_data.add_ancestor(
                start, end, t, focal_sites, haplotype[start: end])
        ancestor_data.record_provenance("verify_data_round_trip")
        ancestor_data.finalise()

        self.assertGreater(len(ancestor_data.uuid), 0)
        self.assertEqual(ancestor_data.sample_data_uuid, sample_data.uuid)
        self.assertEqual(ancestor_data.sequence_length, sample_data.sequence_length)
        self.assertEqual(ancestor_data.format_name, formats.AncestorData.FORMAT_NAME)
        self.assertEqual(
            ancestor_data.format_version, formats.AncestorData.FORMAT_VERSION)
        self.assertEqual(ancestor_data.num_sites, sample_data.num_inference_sites)
        self.assertEqual(ancestor_data.num_ancestors, len(ancestors))
        inference_position = sample_data.sites_position[:][
            sample_data.sites_inference[:]]
        self.assertTrue(np.array_equal(
            inference_position, ancestor_data.sites_position[:]))

        ancestors_list = [anc.haplotype for anc in ancestor_data.ancestors()]
        stored_start = ancestor_data.ancestors_start[:]
        stored_end = ancestor_data.ancestors_end[:]
        stored_time = ancestor_data.ancestors_time[:]
        stored_ancestors = ancestor_data.ancestors_haplotype[:]
        stored_focal_sites = ancestor_data.ancestors_focal_sites[:]
        stored_length = ancestor_data.ancestors_length[:]
        for j, (start, end, t, focal_sites, haplotype) in enumerate(ancestors):
            self.assertEqual(stored_start[j], start)
            self.assertEqual(stored_end[j], end)
            self.assertEqual(stored_time[j], t)
            self.assertTrue(np.array_equal(stored_focal_sites[j], focal_sites))
            self.assertTrue(np.array_equal(stored_ancestors[j], haplotype[start: end]))
            self.assertTrue(np.array_equal(ancestors_list[j], haplotype[start: end]))
        pos = list(ancestor_data.sites_position[:]) + [ancestor_data.sequence_length]
        for j, anc in enumerate(ancestor_data.ancestors()):
            self.assertEqual(stored_start[j], anc.start)
            self.assertEqual(stored_end[j], anc.end)
            self.assertEqual(stored_time[j], anc.time)
            self.assertTrue(np.array_equal(stored_focal_sites[j], anc.focal_sites))
            self.assertTrue(np.array_equal(stored_ancestors[j], anc.haplotype))
            length = pos[anc.end] - pos[anc.start]
            self.assertEqual(stored_length[j], length)

    def test_defaults_no_path(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        ancestor_data = tsinfer.AncestorData(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        for _, array in ancestor_data.arrays():
            self.assertEqual(array.compressor, formats.DEFAULT_COMPRESSOR)

    @unittest.skipIf(sys.platform == "win32",
                     "windows simultaneous file permissions issue")
    def test_defaults_with_path(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "ancestors.tmp")
            ancestor_data = tsinfer.AncestorData(sample_data, path=filename)
            self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
            compressor = formats.DEFAULT_COMPRESSOR
            for _, array in ancestor_data.arrays():
                self.assertEqual(array.compressor, compressor)
            with tsinfer.load(filename) as other:
                self.assertEqual(other, ancestor_data)

    def test_bad_max_file_size(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "ancestors.tmp")
            for bad_size in ['a', '', -1]:
                self.assertRaises(
                    ValueError, formats.AncestorData, sample_data,
                    path=filename, max_file_size=bad_size)
            for bad_size in [[1, 3], np.array([1, 2])]:
                self.assertRaises(
                    TypeError, formats.AncestorData, sample_data,
                    path=filename, max_file_size=bad_size)

    def test_provenance(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        ancestor_data = tsinfer.AncestorData(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        self.assertEqual(ancestor_data.num_provenances, sample_data.num_provenances + 1)

        timestamp = ancestor_data.provenances_timestamp[-1]
        iso = datetime.datetime.now().isoformat()
        self.assertEqual(timestamp.split("T")[0], iso.split("T")[0])
        record = ancestor_data.provenances_record[-1]
        self.assertEqual(record["software"]["name"], "tsinfer")
        a = list(ancestor_data.provenances())
        self.assertEqual(a[-1][0], timestamp)
        self.assertEqual(a[-1][1], record)
        for j, (timestamp, record) in enumerate(sample_data.provenances()):
            self.assertEqual(timestamp, a[j][0])
            self.assertEqual(record, a[j][1])

    def test_chunk_size(self):
        N = 20
        for chunk_size in [1, 2, 3, N - 1, N, N + 1]:
            sample_data, ancestors = self.get_example_data(6, 1, N)
            self.assertGreater(sample_data.num_inference_sites, 2 * N)
            ancestor_data = tsinfer.AncestorData(
                sample_data, chunk_size=chunk_size)
            self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
            self.assertEqual(ancestor_data.ancestors_haplotype.chunks, (chunk_size,))
            self.assertEqual(ancestor_data.ancestors_focal_sites.chunks, (chunk_size,))
            self.assertEqual(ancestor_data.ancestors_start.chunks, (chunk_size,))
            self.assertEqual(ancestor_data.ancestors_end.chunks, (chunk_size,))
            self.assertEqual(ancestor_data.ancestors_time.chunks, (chunk_size,))

    def test_filename(self):
        sample_data, ancestors = self.get_example_data(10, 2, 40)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "ancestors.tmp")
            ancestor_data = tsinfer.AncestorData(
                sample_data, path=filename)
            self.assertTrue(os.path.exists(filename))
            self.assertFalse(os.path.isdir(filename))
            self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
            # Make a copy so that we can close the original and reopen it
            # without hitting simultaneous file access problems on Windows
            ancestor_copy = ancestor_data.copy(path=None)
            ancestor_data.close()
            other_ancestor_data = formats.AncestorData.load(filename)
            self.assertIsNot(other_ancestor_data, ancestor_copy)
            # Can't use eq here because UUIDs will not be equal.
            self.assertTrue(other_ancestor_data.data_equal(ancestor_copy))
            other_ancestor_data.close()

    def test_chunk_size_file_equal(self):
        N = 60
        sample_data, ancestors = self.get_example_data(22, 16, N)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            files = []
            for chunk_size in [5, 7]:
                filename = os.path.join(tempdir, "samples_{}.tmp".format(chunk_size))
                files.append(filename)
                with tsinfer.AncestorData(sample_data, path=filename,
                                          chunk_size=chunk_size) as ancestor_data:
                    self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
                    self.assertEqual(
                        ancestor_data.ancestors_haplotype.chunks, (chunk_size,))
            # Now reload the files and check they are equal
            with formats.AncestorData.load(files[0]) as file0:
                with formats.AncestorData.load(files[1]) as file1:
                    self.assertTrue(file0.data_equal(file1))

    def test_add_ancestor_errors(self):
        sample_data, ancestors = self.get_example_data(22, 16, 30)
        ancestor_data = tsinfer.AncestorData(sample_data)
        num_sites = ancestor_data.num_sites
        haplotype = np.zeros(num_sites, dtype=np.int8)
        ancestor_data.add_ancestor(
            start=0, end=num_sites, time=1, focal_sites=[], haplotype=haplotype)
        for bad_start in [-1, -100, num_sites, num_sites + 1]:
            self.assertRaises(
                ValueError, ancestor_data.add_ancestor,
                start=bad_start, end=num_sites, time=0, focal_sites=[],
                haplotype=haplotype)
        for bad_end in [-1, 0, num_sites + 1, 10 * num_sites]:
            self.assertRaises(
                ValueError, ancestor_data.add_ancestor,
                start=0, end=bad_end, time=1, focal_sites=[], haplotype=haplotype)
        for bad_time in [-1, 0]:
            self.assertRaises(
                ValueError, ancestor_data.add_ancestor,
                start=0, end=num_sites, time=bad_time, focal_sites=[],
                haplotype=haplotype)
        self.assertRaises(
            ValueError, ancestor_data.add_ancestor,
            start=0, end=num_sites, time=1, focal_sites=[],
            haplotype=np.zeros(num_sites + 1, dtype=np.int8))
        # Haplotypes must be < num_alleles
        self.assertRaises(
            ValueError, ancestor_data.add_ancestor,
            start=0, end=num_sites, time=1, focal_sites=[],
            haplotype=np.zeros(num_sites, dtype=np.int8) + 2)
        # focal sites must be within start:end
        self.assertRaises(
            ValueError, ancestor_data.add_ancestor,
            start=1, end=num_sites, time=1, focal_sites=[0],
            haplotype=np.ones(num_sites - 1, dtype=np.int8))
        self.assertRaises(
            ValueError, ancestor_data.add_ancestor,
            start=0, end=num_sites - 2, time=1, focal_sites=[num_sites - 1],
            haplotype=np.ones(num_sites, dtype=np.int8))

    @unittest.skipIf(sys.platform == "win32",
                     "windows simultaneous file permissions issue")
    def test_zero_sequence_length(self):
        # Mangle a sample data file to force a zero sequence length.
        ts = msprime.simulate(10, mutation_rate=2, random_seed=5)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            with tsinfer.SampleData(path=filename) as sample_data:
                for var in ts.variants():
                    sample_data.add_site(var.site.position, var.genotypes)
            store = zarr.LMDBStore(filename, subdir=False)
            data = zarr.open(store=store, mode="w+")
            data.attrs["sequence_length"] = 0
            store.close()
            sample_data = tsinfer.load(filename)
            self.assertEqual(sample_data.sequence_length, 0)
            self.assertRaises(ValueError, tsinfer.generate_ancestors, sample_data)


class BufferedItemWriterMixin(object):
    """
    Tests to ensure that the buffered item writer works as expected.
    """
    def filter_warnings_verify_round_trip(self, source):
        # Zarr currently emits an error when dealing with object arrays.
        # https://github.com/zarr-developers/zarr/issues/257
        # As a workaround, we filter this warning.
        # This should be removed when the bug has been fixed upstream.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.verify_round_trip(source)

    def verify_round_trip(self, source):
        """
        Verify that we can round trip the specified mapping of arrays
        using the buffered item writer.
        """
        # Create a set of empty arrays like the originals.
        dest = {}
        num_rows = -1
        for key, array in source.items():
            dest[key] = zarr.empty_like(array)
            if num_rows == -1:
                num_rows = array.shape[0]
            assert num_rows == array.shape[0]
        assert num_rows != -1
        writer = formats.BufferedItemWriter(dest, num_threads=self.num_threads)
        for j in range(num_rows):
            row = {key: array[j] for key, array in source.items()}
            writer.add(**row)
        writer.flush()

        for key, source_array in source.items():
            dest_array = dest[key]
            if source_array.dtype.str == "|O":
                # Object arrays have to be treated differently.
                self.assertTrue(source_array.shape == dest_array.shape)
                for a, b in zip(source_array, dest_array):
                    if isinstance(a, np.ndarray):
                        self.assertTrue(np.array_equal(a, b))
                    else:
                        self.assertEqual(a, b)
            else:
                self.assertTrue(np.array_equal(source_array[:], dest_array[:]))
            self.assertEqual(source_array.chunks, dest_array.chunks)
        return dest

    def test_one_array(self):
        self.verify_round_trip({"a": zarr.ones(10)})

    def test_two_arrays(self):
        self.verify_round_trip({"a": zarr.ones(10), "b": zarr.zeros(10)})

    def verify_dtypes(self, chunk_size=None):
        n = 100
        if chunk_size is None:
            chunk_size = 100
        dtypes = [np.int8, np.uint8, np.int32, np.uint32, np.float64, np.float32]
        source = {
            str(dtype): zarr.array(np.arange(n, dtype=dtype), chunks=(chunk_size,))
            for dtype in dtypes}
        dest = self.verify_round_trip(source)
        for dtype in dtypes:
            self.assertEqual(dest[str(dtype)].dtype, dtype)

    def test_mixed_dtypes(self):
        self.verify_dtypes()

    def test_mixed_dtypes_chunk_size_1(self):
        self.verify_dtypes(1)

    def test_mixed_dtypes_chunk_size_2(self):
        self.verify_dtypes(2)

    def test_mixed_dtypes_chunk_size_3(self):
        self.verify_dtypes(3)

    def test_mixed_dtypes_chunk_size_10000(self):
        self.verify_dtypes(10000)

    def test_2d_array(self):
        a = zarr.array(np.arange(100).reshape((10, 10)))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_1_1(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(1, 1))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_1_2(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(1, 2))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_2_1(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(1, 2))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_1_100(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(1, 100))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_100_1(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(100, 1))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_10_10(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(5, 10))
        self.verify_round_trip({"a": a})

    def test_3d_array(self):
        a = zarr.array(np.arange(27).reshape((3, 3, 3)))
        self.verify_round_trip({"a": a})

    def test_3d_array_chunks_size_1_1_1(self):
        a = zarr.array(np.arange(27).reshape((3, 3, 3)), chunks=(1, 1, 1))
        self.verify_round_trip({"a": a})

    def test_ragged_array_int32(self):
        n = 10
        z = zarr.empty(n, dtype="array:i4")
        for j in range(n):
            z[j] = np.arange(j)
        self.filter_warnings_verify_round_trip({"z": z})

    def test_square_object_array_int32(self):
        n = 10
        z = zarr.empty(n, dtype="array:i4")
        for j in range(n):
            z[j] = np.arange(n)
        self.filter_warnings_verify_round_trip({"z": z})

    def test_json_object_array(self):
        for chunks in [1, 2, 5, 10, 100]:
            n = 10
            z = zarr.empty(
                n, dtype=object, object_codec=numcodecs.JSON(), chunks=(chunks,))
            for j in range(n):
                z[j] = {str(k): k for k in range(j)}
            self.filter_warnings_verify_round_trip({"z": z})

    def test_empty_string_list(self):
        z = zarr.empty(1, dtype=object, object_codec=numcodecs.JSON(), chunks=(2,))
        z[0] = ["", ""]
        self.filter_warnings_verify_round_trip({"z": z})

    def test_mixed_chunk_sizes(self):
        source = {"a": zarr.zeros(10, chunks=(1,)), "b": zarr.zeros(10, chunks=(2,))}
        self.assertRaises(ValueError, formats.BufferedItemWriter, source)


class TestBufferedItemWriterSynchronous(unittest.TestCase, BufferedItemWriterMixin):
    num_threads = 0


class TestBufferedItemWriterThreads1(unittest.TestCase, BufferedItemWriterMixin):
    num_threads = 1


class TestBufferedItemWriterThreads2(unittest.TestCase, BufferedItemWriterMixin):
    num_threads = 2


class TestBufferedItemWriterThreads20(unittest.TestCase, BufferedItemWriterMixin):
    num_threads = 20
