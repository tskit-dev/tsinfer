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

import numpy as np
import msprime
import numcodecs.blosc as blosc
import lmdb

import tsinfer
import tsinfer.formats as formats


class DataContainerMixin(object):
    """
    Common tests for the the data container classes."
    """
    def test_load(self):
        # TODO there seems to be a bug in zarr in which we an exception occurs
        # during the handling of this one.
        bad_files = ["/", "/file/does/not/exist"]
        for bad_file in bad_files:
            self.assertRaises(lmdb.Error, self.tested_class.load, bad_file)
        bad_format_files = ["LICENSE"]
        for bad_format_file in bad_format_files:
            self.assertTrue(os.path.exists(bad_format_file))
            self.assertRaises(lmdb.Error, formats.SampleData.load, bad_format_file)


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
            input_file.add_variant(v.site.position, v.alleles, v.genotypes)
        input_file.finalise()
        self.assertEqual(input_file.format_version, formats.SampleData.FORMAT_VERSION)
        self.assertEqual(input_file.format_name, formats.SampleData.FORMAT_NAME)
        self.assertEqual(input_file.num_samples, ts.num_samples)
        self.assertEqual(input_file.sequence_length, ts.sequence_length)
        self.assertEqual(input_file.num_sites, ts.num_sites)

        self.assertEqual(input_file.genotypes.dtype, np.uint8)
        self.assertEqual(input_file.position.dtype, np.float64)
        self.assertEqual(input_file.frequency.dtype, np.uint32)
        self.assertEqual(input_file.ancestral_state.dtype, np.int8)
        self.assertEqual(input_file.ancestral_state_offset.dtype, np.uint32)
        self.assertEqual(input_file.derived_state.dtype, np.int8)
        self.assertEqual(input_file.derived_state_offset.dtype, np.uint32)
        self.assertEqual(input_file.variant_site.dtype, np.int32)
        self.assertEqual(input_file.invariant_site.dtype, np.int32)
        self.assertEqual(input_file.singleton_site.dtype, np.int32)
        self.assertEqual(input_file.singleton_sample.dtype, np.int32)

        # Take copies to avoid decompressing the data repeatedly.
        genotypes = input_file.genotypes[:]
        position = input_file.position[:]
        frequency = input_file.frequency[:]
        ancestral_states = msprime.unpack_strings(
            input_file.ancestral_state[:], input_file.ancestral_state_offset[:])
        derived_states = msprime.unpack_strings(
            input_file.derived_state[:], input_file.derived_state_offset[:])
        j = 0
        variant_sites = []
        invariant_sites = []
        singleton_sites = []
        singleton_samples = []
        for variant in ts.variants():
            f = np.sum(variant.genotypes)
            self.assertEqual(variant.site.position, position[variant.site.id])
            self.assertEqual(f, frequency[variant.site.id])
            self.assertEqual(variant.alleles[0], ancestral_states[variant.site.id])
            if len(variant.alleles) > 1:
                self.assertEqual(variant.alleles[1], derived_states[variant.site.id])
            else:
                self.assertEqual(derived_states[variant.site.id], "")
            if f == 1:
                singleton_sites.append(variant.site.id)
                singleton_samples.append(np.where(variant.genotypes == 1)[0][0])
            elif 0 < f < ts.num_samples:
                variant_sites.append(variant.site.id)
                self.assertTrue(np.array_equal(genotypes[j], variant.genotypes))
                j += 1
            else:
                invariant_sites.append(variant.site.id)
        self.assertEqual(input_file.num_variant_sites, j)
        self.assertEqual(input_file.num_singleton_sites, len(singleton_sites))
        self.assertEqual(input_file.num_invariant_sites, len(invariant_sites))
        self.assertTrue(np.array_equal(
            input_file.variant_site[:], np.array(variant_sites, dtype=np.int32)))
        self.assertTrue(np.array_equal(
            input_file.invariant_site[:], np.array(invariant_sites, dtype=np.int32)))
        self.assertTrue(np.array_equal(
            input_file.singleton_site[:], np.array(singleton_sites, dtype=np.int32)))
        self.assertTrue(np.array_equal(
            input_file.singleton_sample[:], np.array(singleton_samples, dtype=np.int32)))

    def test_defaults(self):
        ts = self.get_example_ts(10, 10)
        input_file = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        compressor = formats.DEFAULT_COMPRESSOR
        self.assertEqual(input_file.position.compressor, compressor)
        self.assertEqual(input_file.frequency.compressor, compressor)
        self.assertEqual(input_file.ancestral_state.compressor, compressor)
        self.assertEqual(input_file.ancestral_state_offset.compressor, compressor)
        self.assertEqual(input_file.derived_state.compressor, compressor)
        self.assertEqual(input_file.derived_state_offset.compressor, compressor)
        self.assertEqual(input_file.variant_site.compressor, compressor)
        self.assertEqual(input_file.invariant_site.compressor, compressor)
        self.assertEqual(input_file.singleton_site.compressor, compressor)
        self.assertEqual(input_file.singleton_sample.compressor, compressor)
        self.assertEqual(input_file.genotypes.compressor, compressor)

    def test_chunk_size(self):
        ts = self.get_example_ts(4, 2)
        self.assertGreater(ts.num_sites, 50)
        for chunk_size in [1, 2, 3, ts.num_sites - 1, ts.num_sites, ts.num_sites + 1]:
            input_file = formats.SampleData.initialise(
                num_samples=ts.num_samples, sequence_length=ts.sequence_length,
                chunk_size=chunk_size)
            self.verify_data_round_trip(ts, input_file)
            self.assertEqual(
                input_file.genotypes.chunks,
                (chunk_size, min(chunk_size, ts.num_samples)))

    def test_filename(self):
        ts = self.get_example_ts(14, 15)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            input_file = formats.SampleData.initialise(
                num_samples=ts.num_samples, sequence_length=ts.sequence_length,
                filename=filename)
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
            self.assertEqual(input_file.variant_site.compressor, compressor)
            self.assertEqual(input_file.invariant_site.compressor, compressor)
            self.assertEqual(input_file.singleton_site.compressor, compressor)
            self.assertEqual(input_file.singleton_sample.compressor, compressor)
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

    def test_str(self):
        ts = self.get_example_ts(5, 3)
        input_file = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        self.assertGreater(len(str(input_file)), 0)

    def test_eq(self):
        ts = self.get_example_ts(5, 3)
        input_file = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        self.assertTrue(input_file == input_file)
        self.assertFalse(input_file == [])
        self.assertFalse({} == input_file)

    def test_variant_errors(self):
        input_file = formats.SampleData.initialise(num_samples=2, sequence_length=10)
        genotypes = np.zeros(2, np.uint8)
        input_file.add_variant(0, alleles=["0", "1"], genotypes=genotypes)
        for bad_position in [-1, 10, 100]:
            self.assertRaises(
                ValueError, input_file.add_variant, position=bad_position,
                alleles=["0", "1"], genotypes=genotypes)
        for bad_genotypes in [[0, 2], [-1, 0], [], [0], [0, 0, 0]]:
            genotypes = np.array(bad_genotypes, dtype=np.uint8)
            self.assertRaises(
                ValueError, input_file.add_variant, position=1,
                alleles=["0", "1"], genotypes=genotypes)
        self.assertRaises(
            ValueError, input_file.add_variant, position=1,
            alleles=["0", "1", "2"], genotypes=np.zeros(2, dtype=np.int8))
        self.assertRaises(
            ValueError, input_file.add_variant, position=1,
            alleles=["0"], genotypes=np.array([0, 1], dtype=np.int8))
        self.assertRaises(
            ValueError, input_file.add_variant, position=1,
            alleles=["0", "1"], genotypes=np.array([0, 2], dtype=np.int8))

    def test_variants(self):
        ts = self.get_example_ts(13, 12)
        self.assertGreater(ts.num_sites, 1)
        input_file = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        variants = []
        for v in ts.variants():
            input_file.add_variant(v.site.position, v.alleles, v.genotypes)
            if 1 < np.sum(v.genotypes) < ts.num_samples:
                variants.append(v)
        input_file.finalise()
        self.assertGreater(len(variants), 0)
        self.assertEqual(input_file.num_variant_sites, len(variants))
        j = 0
        for site_id, genotypes in input_file.variants():
            self.assertEqual(variants[j].site.id, site_id)
            self.assertTrue(np.array_equal(variants[j].genotypes, genotypes))
            j += 1
        self.assertEqual(j, len(variants))

    def test_invariant_sites(self):
        n = 10
        m = 10
        for value in [0, 1]:
            G = np.zeros((m, n), dtype=np.int8) + value
            input_file = formats.SampleData.initialise(num_samples=n, sequence_length=m)
            for j in range(m):
                input_file.add_variant(j, ["0", "1"], G[j])
            input_file.finalise()
            self.assertEqual(input_file.num_variant_sites, 0)
            self.assertEqual(input_file.num_invariant_sites, m)
            self.assertEqual(input_file.num_sites, m)
            self.assertTrue(np.array_equal(
                input_file.invariant_site, np.arange(m, dtype=np.int32)))

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
        msprime.sort_tables(**t.asdict())
        ts = msprime.load_tables(**t.asdict())

        input_file = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
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
        msprime.sort_tables(**t.asdict())
        ts = msprime.load_tables(**t.asdict())

        input_file = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)


class TestAncestorData(unittest.TestCase, DataContainerMixin):
    """
    Test cases for the sample data file format.
    """
    tested_class = formats.AncestorData

    def get_example_data(self, sample_size, sequence_length, num_ancestors):
        ts = msprime.simulate(
            sample_size, recombination_rate=1, mutation_rate=10,
            length=sequence_length, random_seed=100)
        sample_data = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        for v in ts.variants():
            sample_data.add_variant(v.site.position, v.alleles, v.genotypes)
        sample_data.finalise()

        num_sites = sample_data.num_variant_sites
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
        self.assertEqual(ancestor_data.num_sites, sample_data.num_variant_sites)
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

    def test_defaults(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        ancestor_data = tsinfer.AncestorData.initialise(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        compressor = formats.DEFAULT_COMPRESSOR
        self.assertEqual(ancestor_data.start.compressor, compressor)
        self.assertEqual(ancestor_data.end.compressor, compressor)
        self.assertEqual(ancestor_data.time.compressor, compressor)
        self.assertEqual(ancestor_data.focal_sites.compressor, compressor)
        self.assertEqual(ancestor_data.ancestor.compressor, compressor)

    def test_chunk_size(self):
        N = 20
        for chunk_size in [1, 2, 3, N - 1, N, N + 1]:
            sample_data, ancestors = self.get_example_data(6, 1, N)
            self.assertGreater(sample_data.num_variant_sites, 2 * N)
            ancestor_data = tsinfer.AncestorData.initialise(
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
            ancestor_data = tsinfer.AncestorData.initialise(
                sample_data, filename=filename)
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
                ancestor_data = tsinfer.AncestorData.initialise(
                    sample_data, filename=filename, chunk_size=chunk_size)
                self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
                self.assertEqual(ancestor_data.ancestor.chunks, (chunk_size,))
            # Now reload the files and check they are equal
            file0 = formats.AncestorData.load(files[0])
            file1 = formats.AncestorData.load(files[1])
            self.assertTrue(file0.data_equal(file1))

    def test_add_ancestor_errors(self):
        sample_data, ancestors = self.get_example_data(22, 16, 30)
        ancestor_data = tsinfer.AncestorData.initialise(sample_data)
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
