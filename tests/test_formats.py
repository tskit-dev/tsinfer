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

import unittest
import tempfile
import os.path
import datetime
import warnings

import numpy as np
import msprime
import numcodecs.blosc as blosc
import zarr

import tsinfer
import tsinfer.formats as formats
import tsinfer.exceptions as exceptions


class DataContainerMixin(object):
    """
    Common tests for the the data container classes."
    """
    def test_load(self):
        self.assertRaises(IsADirectoryError, self.tested_class.load, "/")
        self.assertRaises(
            FileNotFoundError, self.tested_class.load, "/file/does/not/exist")
        bad_format_files = ["LICENSE", "/dev/urandom"]
        for bad_format_file in bad_format_files:
            self.assertTrue(os.path.exists(bad_format_file))
            self.assertRaises(
                exceptions.FileFormatError, formats.SampleData.load, bad_format_file)


class TestSampleData(unittest.TestCase, DataContainerMixin):
    """
    Test cases for the sample data file format.
    """
    tested_class = formats.SampleData

    def get_example_ts(self, sample_size, sequence_length):
        return msprime.simulate(
            sample_size, recombination_rate=1, mutation_rate=10,
            length=sequence_length, random_seed=100)

    def verify_data_round_trip(self, ts, input_file):
        self.assertGreater(ts.num_sites, 1)
        for v in ts.variants():
            input_file.add_site(v.site.position, v.genotypes, v.alleles)
        input_file.finalise()
        self.assertEqual(input_file.format_version, formats.SampleData.FORMAT_VERSION)
        self.assertEqual(input_file.format_name, formats.SampleData.FORMAT_NAME)
        self.assertEqual(input_file.num_samples, ts.num_samples)
        self.assertEqual(input_file.sequence_length, ts.sequence_length)
        self.assertEqual(input_file.num_sites, ts.num_sites)
        self.assertEqual(input_file.sites_genotypes.dtype, np.uint8)
        self.assertEqual(input_file.sites_position.dtype, np.float64)
        # Take copies to avoid decompressing the data repeatedly.
        genotypes = input_file.sites_genotypes[:]
        position = input_file.sites_position[:]
        alleles = input_file.sites_alleles[:]
        for j, variant in enumerate(ts.variants()):
            self.assertEqual(variant.site.position, position[j])
            self.assertTrue(np.all(variant.genotypes == genotypes[j]))
            self.assertEqual(alleles[j], list(variant.alleles))

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

    def test_defaults_no_path(self):
        ts = self.get_example_ts(10, 10)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        for _, array in input_file.arrays():
            self.assertEqual(array.compressor, None)

    def test_from_tree_sequence(self):
        ts = self.get_example_ts(10, 10)
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
            other_input_file = formats.SampleData.load(filename)
            self.assertIsNot(other_input_file, input_file)
            self.assertEqual(other_input_file, input_file)

    def test_chunk_size_file_equal(self):
        ts = self.get_example_ts(13, 15)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            files = []
            for chunk_size in [5, 7]:
                filename = os.path.join(tempdir, "samples_{}.tmp".format(chunk_size))
                files.append(filename)
                input_file = formats.SampleData(
                    sequence_length=ts.sequence_length, path=filename,
                    chunk_size=chunk_size)
                self.verify_data_round_trip(ts, input_file)
                self.assertEqual(
                    input_file.sites_genotypes.chunks, (chunk_size, chunk_size))
            # Now reload the files and check they are equal
            input_file0 = formats.SampleData.load(files[0])
            input_file1 = formats.SampleData.load(files[1])
            # Can't use eq here because UUIDs will be equal.
            self.assertTrue(input_file0.data_equal(input_file1))

    def test_compressor(self):
        ts = self.get_example_ts(11, 17)
        compressors = [
           None, formats.DEFAULT_COMPRESSOR,
           blosc.Blosc(cname='zlib', clevel=1, shuffle=blosc.NOSHUFFLE)
        ]
        for compressor in compressors:
            input_file = formats.SampleData(
                sequence_length=ts.sequence_length, compressor=compressor)
            self.verify_data_round_trip(ts, input_file)
            for _, array in input_file.arrays():
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
        self.assertEqual(record["software"], "tsinfer")
        a = list(input_file.provenances())
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0][0], timestamp)
        self.assertEqual(a[0][1], record)

    def test_variant_errors(self):
        input_file = formats.SampleData(sequence_length=10)
        genotypes = np.zeros(2, np.uint8)
        input_file.add_site(0, alleles=["0", "1"], genotypes=genotypes)
        for bad_position in [-1, 10, 100]:
            self.assertRaises(
                ValueError, input_file.add_site, position=bad_position,
                alleles=["0", "1"], genotypes=genotypes)
        for bad_genotypes in [[0, 2], [-1, 0], [], [0], [0, 0, 0]]:
            genotypes = np.array(bad_genotypes, dtype=np.uint8)
            self.assertRaises(
                ValueError, input_file.add_site, position=1,
                alleles=["0", "1"], genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_site, position=1,
            alleles=["0", "1", "2"], genotypes=np.zeros(2, dtype=np.int8))
        self.assertRaises(
            ValueError, input_file.add_site, position=1,
            alleles=["0"], genotypes=np.array([0, 1], dtype=np.int8))
        self.assertRaises(
            ValueError, input_file.add_site, position=1,
            alleles=["0", "1"], genotypes=np.array([0, 2], dtype=np.int8))
        self.assertRaises(
            ValueError, input_file.add_site, position=1,
            alleles=["0", "0"], genotypes=np.array([0, 2], dtype=np.int8))

    def test_invalid_inference_sites(self):
        # Trying to add singletons or fixed sites as inference sites
        # raise and error
        input_file = formats.SampleData()
        # Make sure this is OK
        input_file.add_site(0, [0, 1, 1], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[0, 0, 0], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[1, 0, 0], inference=True)
        self.assertRaises(
            ValueError, input_file.add_site,
            position=1, genotypes=[1, 1, 1], inference=True)
        input_file.add_site(
            position=1, genotypes=[1, 0, 1], inference=True)

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
        sample_data.finalise()
        self.assertEqual(sample_data.populations_metadata[0], {"a": 1})
        self.assertEqual(sample_data.populations_metadata[1], {"b": 2})

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

    def test_add_individual_errors(self):
        sample_data = formats.SampleData(sequence_length=10)
        self.assertRaises(TypeError, sample_data.add_individual, metadata=234)
        self.assertRaises(ValueError, sample_data.add_individual, population=0)
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_population()
        self.assertRaises(ValueError, sample_data.add_individual, population=1)
        self.assertRaises(ValueError, sample_data.add_individual, location="x234")
        self.assertRaises(ValueError, sample_data.add_individual, ploidy=0)

    def test_no_data(self):
        sample_data = formats.SampleData(sequence_length=10)
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

    def test_haplotypes(self):
        ts = self.get_example_ts(13, 12)
        self.assertGreater(ts.num_sites, 1)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        for v in ts.variants():
            input_file.add_site(v.site.position, v.genotypes, v.alleles)
        input_file.finalise()

        G = ts.genotype_matrix()
        j = 0
        for h in input_file.haplotypes():
            self.assertTrue(np.array_equal(h, G[:, j]))
            j += 1
        self.assertEqual(j, ts.num_samples)

        selection = input_file.sites_inference[:] == 1
        j = 0
        for h in input_file.haplotypes(inference_sites=True):
            self.assertTrue(np.array_equal(h, G[selection, j]))
            j += 1
        self.assertEqual(j, ts.num_samples)

        selection = input_file.sites_inference[:] == 0
        j = 0
        for h in input_file.haplotypes(inference_sites=False):
            self.assertTrue(np.array_equal(h, G[selection, j]))
            j += 1
        self.assertEqual(j, ts.num_samples)

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
            self.assertTrue(
                np.all(input_file.sites_inference[:] == np.zeros(m, dtype=int)))

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
        copy = data.copy()
        self.assertRaises(ValueError, copy.copy)

    def test_copy_new_uuid(self):
        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1])
        data.finalise()
        copy = data.copy()
        copy.finalise()
        self.assertNotEqual(copy.uuid, data.uuid)
        self.assertTrue(copy.data_equal(data))

    def test_copy_update_inference_sites(self):
        with formats.SampleData() as data:
            for j in range(4):
                data.add_site(position=j, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        self.assertEqual(list(data.sites_inference), [1, 1, 1, 1])

        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            for copy_path in [None, filename]:
                copy = data.copy(path=copy_path)
                copy.finalise()
                self.assertTrue(copy.data_equal(data))
                with data.copy(path=copy_path) as copy:
                    inference = [0, 1, 0, 1]
                    copy.sites_inference = inference
                self.assertFalse(copy.data_equal(data))
                self.assertEqual(list(copy.sites_inference), inference)
                self.assertEqual(list(data.sites_inference), [1, 1, 1, 1])

    def test_update_inference_sites_bad_data(self):
        def set_value(data, value):
            data.sites_inference = value

        data = formats.SampleData()
        for j in range(4):
            data.add_site(position=j, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        data.finalise()
        self.assertEqual(list(data.sites_inference), [1, 1, 1, 1])
        copy = data.copy()
        for bad_shape in [[], np.arange(100), np.zeros((2, 2))]:
            self.assertRaises(ValueError, set_value, copy, bad_shape)
        bad_data = [
            ["a", "b", "c", "d"], [2**10 for _ in range(4)], [0, 1, 0, 2],
            [0, 0, 0, -1]]
        for a in bad_data:
            self.assertRaises(ValueError, set_value, copy, a)

    def test_update_inference_sites_non_copy_mode(self):
        def set_value(data, value):
            data.sites_inference = value

        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        self.assertRaises(ValueError, set_value, data, [])
        data.finalise()
        self.assertRaises(ValueError, set_value, data, [])

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


class TestAncestorData(unittest.TestCase, DataContainerMixin):
    """
    Test cases for the sample data file format.
    """
    tested_class = formats.AncestorData

    def get_example_data(self, sample_size, sequence_length, num_ancestors):
        ts = msprime.simulate(
            sample_size, recombination_rate=1, mutation_rate=10,
            length=sequence_length, random_seed=100)
        sample_data = formats.SampleData(sequence_length=ts.sequence_length)
        for v in ts.variants():
            sample_data.add_site(v.site.position, v.genotypes, v.alleles)
        sample_data.finalise()

        num_sites = sample_data.num_inference_sites
        ancestors = []
        for j in range(num_ancestors):
            haplotype = np.zeros(num_sites, dtype=np.uint8) + tsinfer.UNKNOWN_ALLELE
            start = j
            end = max(num_sites - j, start + 1)
            self.assertLess(start, end)
            haplotype[start: end] = 0
            if start + j < end:
                haplotype[start + j: end] = 1
            self.assertTrue(np.all(haplotype[:start] == tsinfer.UNKNOWN_ALLELE))
            self.assertTrue(np.all(haplotype[end:] == tsinfer.UNKNOWN_ALLELE))
            focal_sites = np.array([start + k for k in range(j)], dtype=np.int32)
            focal_sites = focal_sites[focal_sites < end]
            haplotype[focal_sites] = 1
            ancestors.append((start, end, 2 * j + 1, focal_sites, haplotype))
        return sample_data, ancestors

    def verify_data_round_trip(self, sample_data, ancestor_data, ancestors):
        for start, end, time, focal_sites, haplotype in ancestors:
            ancestor_data.add_ancestor(start, end, time, focal_sites, haplotype)
        ancestor_data.finalise()

        self.assertGreater(len(ancestor_data.uuid), 0)
        self.assertEqual(ancestor_data.sample_data_uuid, sample_data.uuid)
        self.assertEqual(ancestor_data.format_name, formats.AncestorData.FORMAT_NAME)
        self.assertEqual(
            ancestor_data.format_version, formats.AncestorData.FORMAT_VERSION)
        self.assertEqual(ancestor_data.num_sites, sample_data.num_inference_sites)
        self.assertEqual(ancestor_data.num_ancestors, len(ancestors))

        ancestors_list = list(ancestor_data.ancestors())
        stored_start = ancestor_data.start[:]
        stored_end = ancestor_data.end[:]
        stored_time = ancestor_data.time[:]
        stored_ancestors = ancestor_data.ancestor[:]
        stored_focal_sites = ancestor_data.focal_sites[:]
        for j, (start, end, time, focal_sites, haplotype) in enumerate(ancestors):
            self.assertEqual(stored_start[j], start)
            self.assertEqual(stored_end[j], end)
            self.assertEqual(stored_time[j], time)
            self.assertTrue(np.array_equal(stored_focal_sites[j], focal_sites))
            self.assertTrue(np.array_equal(stored_ancestors[j], haplotype[start: end]))
            self.assertTrue(np.array_equal(ancestors_list[j], haplotype[start: end]))

    def test_defaults_no_path(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        ancestor_data = tsinfer.AncestorData(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        for _, array in ancestor_data.arrays():
            self.assertEqual(array.compressor, None)

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

    def test_provenance(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        ancestor_data = tsinfer.AncestorData(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        self.assertEqual(ancestor_data.num_provenances, 1)
        timestamp = ancestor_data.provenances_timestamp[0]
        iso = datetime.datetime.now().isoformat()
        self.assertEqual(timestamp.split("T")[0], iso.split("T")[0])
        record = ancestor_data.provenances_record[0]
        self.assertEqual(record["software"], "tsinfer")
        a = list(ancestor_data.provenances())
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0][0], timestamp)
        self.assertEqual(a[0][1], record)

    def test_chunk_size(self):
        N = 20
        for chunk_size in [1, 2, 3, N - 1, N, N + 1]:
            sample_data, ancestors = self.get_example_data(6, 1, N)
            self.assertGreater(sample_data.num_inference_sites, 2 * N)
            ancestor_data = tsinfer.AncestorData(
                sample_data, chunk_size=chunk_size)
            self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
            self.assertEqual(ancestor_data.ancestor.chunks, (chunk_size,))
            self.assertEqual(ancestor_data.focal_sites.chunks, (chunk_size,))
            self.assertEqual(ancestor_data.start.chunks, (chunk_size,))
            self.assertEqual(ancestor_data.end.chunks, (chunk_size,))
            self.assertEqual(ancestor_data.time.chunks, (chunk_size,))

    def test_filename(self):
        sample_data, ancestors = self.get_example_data(10, 2, 40)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "ancestors.tmp")
            ancestor_data = tsinfer.AncestorData(
                sample_data, path=filename)
            self.assertTrue(os.path.exists(filename))
            self.assertFalse(os.path.isdir(filename))
            self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
            other_ancestor_data = formats.AncestorData.load(filename)
            self.assertIsNot(other_ancestor_data, ancestor_data)
            self.assertEqual(other_ancestor_data, ancestor_data)

    def test_chunk_size_file_equal(self):
        N = 60
        sample_data, ancestors = self.get_example_data(22, 16, N)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            files = []
            for chunk_size in [5, 7]:
                filename = os.path.join(tempdir, "samples_{}.tmp".format(chunk_size))
                files.append(filename)
                ancestor_data = tsinfer.AncestorData(
                    sample_data, path=filename, chunk_size=chunk_size)
                self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
                self.assertEqual(ancestor_data.ancestor.chunks, (chunk_size,))
            # Now reload the files and check they are equal
            file0 = formats.AncestorData.load(files[0])
            file1 = formats.AncestorData.load(files[1])
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
            haplotype=np.zeros(num_sites + 1, dtype=np.uint8))
        # Haplotypes must be < 2
        self.assertRaises(
            ValueError, ancestor_data.add_ancestor,
            start=0, end=num_sites, time=1, focal_sites=[],
            haplotype=np.zeros(num_sites, dtype=np.uint8) + 2)

        # focal sites must be within start:end
        self.assertRaises(
            ValueError, ancestor_data.add_ancestor,
            start=1, end=num_sites, time=1, focal_sites=[0],
            haplotype=np.ones(num_sites, dtype=np.uint8))
        self.assertRaises(
            ValueError, ancestor_data.add_ancestor,
            start=0, end=num_sites - 2, time=1, focal_sites=[num_sites - 1],
            haplotype=np.ones(num_sites, dtype=np.uint8))
        # focal sites must be set to 1
        self.assertRaises(
            ValueError, ancestor_data.add_ancestor,
            start=0, end=num_sites, time=1, focal_sites=[0],
            haplotype=np.zeros(num_sites, dtype=np.uint8))


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
                n, dtype=object, object_codec=formats.TempJSON(), chunks=(chunks,))
            for j in range(n):
                z[j] = {str(k): k for k in range(j)}
            self.filter_warnings_verify_round_trip({"z": z})

    def test_empty_string_list(self):
        z = zarr.empty(1, dtype=object, object_codec=formats.TempJSON(), chunks=(2,))
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
