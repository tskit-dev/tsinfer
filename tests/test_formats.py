"""
Tests for the data files.
"""

import unittest
import tempfile
import os.path

import numpy as np
import msprime
import zarr
import numcodecs.blosc as blosc

import tsinfer
import tsinfer.algorithm as algorithm
import tsinfer.formats as formats


class TestSampleData(unittest.TestCase):
    """
    Test cases for the sample data file format.
    """
    def get_example_ts(self, sample_size, sequence_length):
        return msprime.simulate(
            sample_size, recombination_rate=1, mutation_rate=10,
            length=sequence_length, random_seed=100)

    def verify_data_round_trip(self, ts, input_file):
        self.assertGreater(ts.num_sites, 1)
        for v in ts.variants():
            input_file.add_variant(v.site.position, v.alleles, v.genotypes)
        input_file.finalise()
        self.assertEqual(input_file.format_version, formats.SampleData.FORMAT_VERSION)
        self.assertEqual(input_file.format_name, formats.SampleData.FORMAT_NAME)
        self.assertEqual(input_file.num_samples, ts.num_samples)
        self.assertEqual(input_file.sequence_length, ts.sequence_length)
        self.assertEqual(input_file.num_sites, ts.num_sites)

        # Take copies to avoid decompressing the data repeatedly.
        genotypes = input_file.genotypes[:]
        position = input_file.position[:]
        frequency = input_file.frequency[:]
        recombination_rate = input_file.recombination_rate[:]
        ancestral_states = msprime.unpack_strings(
            input_file.ancestral_state[:], input_file.ancestral_state_offset[:])
        derived_states = msprime.unpack_strings(
            input_file.derived_state[:], input_file.derived_state_offset[:])
        j = 0
        variant_sites = []
        for variant in ts.variants():
            f = np.sum(variant.genotypes)
            self.assertEqual(variant.site.position, position[variant.site.id])
            self.assertEqual(f, frequency[variant.site.id])
            self.assertEqual(variant.alleles[0], ancestral_states[variant.site.id])
            self.assertEqual(variant.alleles[1], derived_states[variant.site.id])
            if f > 1 and f < ts.num_samples:
                variant_sites.append(variant.site.id)
                self.assertTrue(np.array_equal(genotypes[j], variant.genotypes))
                self.assertGreaterEqual(recombination_rate[j], 0)
                j += 1
        self.assertEqual(input_file.num_variant_sites, j)
        self.assertTrue(np.array_equal(
            input_file.variant_sites[:], np.array(variant_sites, dtype=np.uint32)))

    def test_defaults(self):
        ts = self.get_example_ts(10, 10)
        input_file = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        compressor = formats.DEFAULT_COMPRESSOR
        self.assertEqual(input_file.position.compressor, compressor)
        self.assertEqual(input_file.frequency.compressor, compressor)
        # self.assertEqual(input_file.alleles.compressor, compressor)
        self.assertEqual(input_file.variant_sites.compressor, compressor)
        self.assertEqual(input_file.recombination_rate.compressor, compressor)
        self.assertEqual(input_file.genotypes.compressor, compressor)

    def test_chunk_size(self):
        ts = self.get_example_ts(4, 20)
        for chunk_size in [1, 2, 3, ts.num_sites - 1, ts.num_sites, ts.num_sites + 1]:
            input_file = formats.SampleData.initialise(
                num_samples=ts.num_samples, sequence_length=ts.sequence_length,
                chunk_size=chunk_size)
            self.verify_data_round_trip(ts, input_file)
            self.assertEqual(
                input_file.genotypes.chunks, (chunk_size, min(chunk_size, ts.num_samples)))

    def test_filename(self):
        ts = self.get_example_ts(14, 15)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            input_file = formats.SampleData.initialise(
                num_samples=ts.num_samples, sequence_length=ts.sequence_length,
                filename=filename)
            self.verify_data_round_trip(ts, input_file)
            self.assertTrue(os.path.exists(filename))
            self.assertGreater(os.path.getsize(filename), 0)
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
                input_file = formats.SampleData.initialise(
                    num_samples=ts.num_samples, sequence_length=ts.sequence_length,
                    filename=filename, chunk_size=chunk_size)
                self.verify_data_round_trip(ts, input_file)
                self.assertEqual(input_file.genotypes.chunks, (chunk_size, chunk_size))
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
            input_file = formats.SampleData.initialise(
                num_samples=ts.num_samples, sequence_length=ts.sequence_length,
                compressor=compressor)
            self.verify_data_round_trip(ts, input_file)
            self.assertEqual(input_file.position.compressor, compressor)
            self.assertEqual(input_file.frequency.compressor, compressor)
            self.assertEqual(input_file.ancestral_state.compressor, compressor)
            self.assertEqual(input_file.ancestral_state_offset.compressor, compressor)
            self.assertEqual(input_file.derived_state.compressor, compressor)
            self.assertEqual(input_file.derived_state_offset.compressor, compressor)
            self.assertEqual(input_file.variant_sites.compressor, compressor)
            self.assertEqual(input_file.recombination_rate.compressor, compressor)
            self.assertEqual(input_file.genotypes.compressor, compressor)

    def test_multichar_alleles(self):
        ts = self.get_example_ts(5, 17)
        t = ts.tables
        t.sites.clear()
        t.mutations.clear()
        for site in ts.sites():
            t.sites.add_row(site.position, ancestral_state="A" * (site.id + 1))
            for mutation in site.mutations:
                t.mutations.add_row(
                    site=site.id, node=mutation.node, derived_state="T" * site.id)
        ts = msprime.load_tables(**t.asdict())
        input_file = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        self.assertTrue(np.array_equal(
            t.sites.ancestral_state, input_file.ancestral_state[:]))
        self.assertTrue(np.array_equal(
            t.sites.ancestral_state_offset, input_file.ancestral_state_offset[:]))
        self.assertTrue(np.array_equal(
            t.mutations.derived_state, input_file.derived_state[:]))
        self.assertTrue(np.array_equal(
            t.mutations.derived_state_offset, input_file.derived_state_offset[:]))

