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
Tests for the inference code.
"""
import unittest
import random
import string
import json
import math

import numpy as np
import msprime

import tsinfer
import tsinfer.eval_util as eval_util


def get_random_data_example(num_samples, num_sites, seed=42):
    np.random.seed(seed)
    G = np.random.randint(2, size=(num_sites, num_samples)).astype(np.uint8)
    return G, np.arange(num_sites)


class TsinferTestCase(unittest.TestCase):
    """
    Superclass containing assert utilities for tsinfer test cases.
    """
    def assertTreeSequencesEqual(self, ts1, ts2):
        self.assertEqual(ts1.sequence_length, ts2.sequence_length)
        t1 = ts1.tables
        t2 = ts2.tables
        self.assertEqual(t1.nodes, t2.nodes)
        self.assertEqual(t1.edges, t2.edges)
        self.assertEqual(t1.sites, t2.sites)
        self.assertEqual(t1.mutations, t2.mutations)


class TestRoundTrip(unittest.TestCase):
    """
    Test that we can round-trip data tsinfer.
    """
    def verify_data_round_trip(self, genotypes, positions, sequence_length=None):
        if sequence_length is None:
            sequence_length = positions[-1] + 1
        sample_data = tsinfer.SampleData(sequence_length=sequence_length)
        for j in range(genotypes.shape[0]):
            sample_data.add_site(positions[j], genotypes[j])
        sample_data.finalise()
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts = tsinfer.infer(sample_data, engine=engine)
            self.assertEqual(ts.sequence_length, sequence_length)
            self.assertEqual(ts.num_sites, len(positions))
            for v in ts.variants():
                self.assertEqual(v.position, positions[v.index])
                self.assertTrue(np.array_equal(genotypes[v.index], v.genotypes))
            self.assertGreater(ts.num_provenances, 0)

        for simplify in [True, False]:
            ts = tsinfer.infer(sample_data, simplify=simplify)
            self.assertEqual(ts.sequence_length, sequence_length)
            self.assertEqual(ts.num_sites, len(positions))
            for v in ts.variants():
                self.assertEqual(v.position, positions[v.index])
                self.assertTrue(np.array_equal(genotypes[v.index], v.genotypes))

        for path_compression in [True, False]:
            ts = tsinfer.infer(sample_data, path_compression=path_compression)
            self.assertEqual(ts.sequence_length, sequence_length)
            self.assertEqual(ts.num_sites, len(positions))
            for v in ts.variants():
                self.assertEqual(v.position, positions[v.index])
                self.assertTrue(np.array_equal(genotypes[v.index], v.genotypes))

    def verify_round_trip(self, ts):
        positions = [site.position for site in ts.sites()]
        self.verify_data_round_trip(ts.genotype_matrix(), positions, ts.sequence_length)

    def test_simple_example(self):
        rho = 2
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=rho, random_seed=1)
        self.assertGreater(ts.num_sites, 0)
        self.verify_round_trip(ts)

    def test_single_locus(self):
        ts = msprime.simulate(5, mutation_rate=1, recombination_rate=0, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify_round_trip(ts)

    def test_single_locus_two_samples(self):
        ts = msprime.simulate(2, mutation_rate=1, recombination_rate=0, random_seed=3)
        self.assertGreater(ts.num_sites, 0)
        self.verify_round_trip(ts)

    def test_two_samples_one_site(self):
        self.verify_data_round_trip(np.array([[1, 1]]), [0])

    def test_two_samples_two_sites(self):
        self.verify_data_round_trip(np.array([[1, 1], [0, 1]]), [0, 1])

    def test_random_data_invariant_sites_ancestral_state(self):
        G, positions = get_random_data_example(24, 35)
        # Set some sites to be invariant for the ancestral state
        G[10, :] = 0
        G[15, :] = 0
        G[20, :] = 0
        G[22, :] = 0
        self.verify_data_round_trip(G, positions)

    def test_random_data_invariant_sites(self):
        G, positions = get_random_data_example(39, 25)
        # Set some sites to be invariant
        G[10, :] = 1
        G[15, :] = 0
        G[20, :] = 1
        G[22, :] = 0
        self.verify_data_round_trip(G, positions)

    def test_all_ancestral(self):
        G = np.ones((10, 10), dtype=int)
        self.verify_data_round_trip(G, np.arange(G.shape[0]))

    def test_all_derived(self):
        G = np.zeros((10, 10), dtype=int)
        self.verify_data_round_trip(G, np.arange(G.shape[0]))

    def test_all_derived_or_ancestral(self):
        G = np.zeros((10, 10), dtype=int)
        G[::2] = 1
        self.verify_data_round_trip(G, np.arange(G.shape[0]))

    def test_random_data_large_example(self):
        G, positions = get_random_data_example(20, 30)
        self.verify_data_round_trip(G, positions)

    def test_random_data_small_examples(self):
        np.random.seed(4)
        num_random_tests = 10
        for _ in range(num_random_tests):
            S, positions = get_random_data_example(5, 10)
            self.verify_data_round_trip(S, positions)


class TestNonInferenceSitesRoundTrip(unittest.TestCase):
    """
    Test that we can round-trip data when we have various combinations
    of inference and non inference sites.
    """
    def verify_round_trip(self, genotypes, inference):
        self.assertEqual(genotypes.shape[0], inference.shape[0])
        with tsinfer.SampleData() as sample_data:
            for j in range(genotypes.shape[0]):
                sample_data.add_site(j, genotypes[j], inference=inference[j])
        output_ts = tsinfer.infer(sample_data)
        self.assertTrue(np.array_equal(genotypes, output_ts.genotype_matrix()))

    def test_simple_single_tree(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=10)
        self.assertGreater(ts.num_sites, 2)
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        self.verify_round_trip(genotypes, inference)

    def test_half_sites_single_tree(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=20)
        self.assertGreater(ts.num_sites, 2)
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        inference[::2] = False
        self.verify_round_trip(genotypes, inference)

    def test_simple_many_trees(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=10)
        self.assertGreater(ts.num_trees, 2)
        self.assertGreater(ts.num_sites, 2)
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        self.verify_round_trip(genotypes, inference)

    def test_half_sites_many_trees(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=11)
        self.assertGreater(ts.num_trees, 2)
        self.assertGreater(ts.num_sites, 2)
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        inference[::2] = False
        self.verify_round_trip(genotypes, inference)

    def test_zero_inference_sites(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=21)
        self.assertGreater(ts.num_sites, 2)
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        inference[:] = False
        self.verify_round_trip(genotypes, inference)


class TestZeroInferenceSites(unittest.TestCase):
    """
    Tests for the degenerate case in which we have no inference sites.
    """
    def verify(self, genotypes):
        genotypes = np.array(genotypes, dtype=np.int8)
        m = genotypes.shape[0]
        with tsinfer.SampleData(sequence_length=m + 1) as sample_data:
            for j in range(m):
                sample_data.add_site(j, genotypes[j], inference=False)
        output_ts = tsinfer.infer(sample_data)
        for tree in output_ts.trees():
            self.assertEqual(tree.num_roots, 1)

    def test_many_sites(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=21)
        self.assertGreater(ts.num_sites, 2)
        self.verify(ts.genotype_matrix())

    def test_one_site(self):
        self.verify([[0, 0]])
        self.verify([[0, 1]])
        self.verify([[1, 0]])
        self.verify([[1, 1]])

    def test_two_sites(self):
        self.verify([[0, 0], [0, 0]])
        self.verify([[1, 1], [1, 1]])
        self.verify([[0, 0, 0], [0, 0, 0]])
        self.verify([[0, 1, 0], [1, 0, 0]])

    def test_three_sites(self):
        self.verify([[0, 0], [0, 0], [0, 0]])
        self.verify([[1, 1], [1, 1], [1, 1]])


def random_string(rng, max_len=10):
    """
    Uses the specified random generator to generate a random string.
    """
    s = ""
    for _ in range(rng.randint(0, max_len)):
        s += rng.choice(string.ascii_letters)
    return s


class TestMetadataRoundTrip(unittest.TestCase):
    """
    Tests if we can round-trip various forms of metadata.
    """

    def test_multichar_alleles(self):
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=1, random_seed=5)
        self.assertGreater(ts.num_sites, 2)
        sample_data = tsinfer.SampleData(sequence_length=1)
        rng = random.Random(32)
        all_alleles = []
        for variant in ts.variants():
            ancestral = random_string(rng)
            derived = ancestral
            while derived == ancestral:
                derived = random_string(rng)
            alleles = ancestral, derived
            sample_data.add_site(variant.site.position, variant.genotypes, alleles)
            all_alleles.append(alleles)
        sample_data.finalise()

        for j, alleles in enumerate(sample_data.sites_alleles[:]):
            self.assertEqual(all_alleles[j], tuple(alleles))

        output_ts = tsinfer.infer(sample_data)
        inferred_alleles = [variant.alleles for variant in output_ts.variants()]
        self.assertEqual(inferred_alleles, all_alleles)

    def test_site_metadata(self):
        ts = msprime.simulate(
            11, mutation_rate=5, recombination_rate=2, random_seed=15)
        self.assertGreater(ts.num_sites, 2)
        sample_data = tsinfer.SampleData(sequence_length=1)
        rng = random.Random(32)
        all_metadata = []
        for variant in ts.variants():
            metadata = {str(j): random_string(rng) for j in range(rng.randint(0, 5))}
            sample_data.add_site(
                variant.site.position, variant.genotypes, alleles=["A", "T"],
                metadata=metadata)
            all_metadata.append(metadata)
        sample_data.finalise()

        for j, metadata in enumerate(sample_data.sites_metadata[:]):
            self.assertEqual(all_metadata[j], metadata)

        output_ts = tsinfer.infer(sample_data)
        output_metadata = [
            json.loads(site.metadata.decode()) for site in output_ts.sites()]
        self.assertEqual(all_metadata, output_metadata)

    def test_population_metadata(self):
        ts = msprime.simulate(12, mutation_rate=5, random_seed=16)
        self.assertGreater(ts.num_sites, 2)
        sample_data = tsinfer.SampleData(sequence_length=1)
        rng = random.Random(32)
        all_metadata = []
        for j in range(ts.num_samples):
            metadata = {str(j): random_string(rng) for j in range(rng.randint(0, 5))}
            sample_data.add_population(metadata=metadata)
            all_metadata.append(metadata)
        for j in range(ts.num_samples):
            sample_data.add_individual(population=j)
        for variant in ts.variants():
            sample_data.add_site(
                variant.site.position, variant.genotypes, variant.alleles)
        sample_data.finalise()

        for j, metadata in enumerate(sample_data.populations_metadata[:]):
            self.assertEqual(all_metadata[j], metadata)
        output_ts = tsinfer.infer(sample_data)
        output_metadata = [
            json.loads(population.metadata.decode())
            for population in output_ts.populations()]
        self.assertEqual(all_metadata, output_metadata)
        for j, sample in enumerate(output_ts.samples()):
            node = output_ts.node(sample)
            self.assertEqual(node.population, j)

    def test_individual_metadata(self):
        ts = msprime.simulate(11, mutation_rate=5, random_seed=16)
        self.assertGreater(ts.num_sites, 2)
        sample_data = tsinfer.SampleData(sequence_length=1)
        rng = random.Random(32)
        all_metadata = []
        for j in range(ts.num_samples):
            metadata = {str(j): random_string(rng) for j in range(rng.randint(0, 5))}
            sample_data.add_individual(metadata=metadata)
            all_metadata.append(metadata)
        for variant in ts.variants():
            sample_data.add_site(
                variant.site.position, variant.genotypes, variant.alleles)
        sample_data.finalise()

        for j, metadata in enumerate(sample_data.individuals_metadata[:]):
            self.assertEqual(all_metadata[j], metadata)
        output_ts = tsinfer.infer(sample_data)
        output_metadata = [
            json.loads(individual.metadata.decode())
            for individual in output_ts.individuals()]
        self.assertEqual(all_metadata, output_metadata)

    def test_individual_location(self):
        ts = msprime.simulate(12, mutation_rate=5, random_seed=16)
        self.assertGreater(ts.num_sites, 2)
        sample_data = tsinfer.SampleData(sequence_length=1)
        rng = random.Random(32)
        all_locations = []
        for j in range(ts.num_samples // 2):
            location = np.array([rng.random() for _ in range(j)])
            sample_data.add_individual(location=location, ploidy=2)
            all_locations.append(location)
        for variant in ts.variants():
            sample_data.add_site(
                variant.site.position, variant.genotypes, variant.alleles)
        sample_data.finalise()

        for j, location in enumerate(sample_data.individuals_location[:]):
            self.assertTrue(np.array_equal(all_locations[j], location))
        output_ts = tsinfer.infer(sample_data)
        self.assertEqual(output_ts.num_individuals, len(all_locations))
        for location, individual in zip(all_locations, output_ts.individuals()):
            self.assertTrue(np.array_equal(location, individual.location))


class TestThreads(TsinferTestCase):

    def test_equivalance(self):
        rho = 2
        ts = msprime.simulate(5, mutation_rate=2, recombination_rate=rho, random_seed=2)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        ts1 = tsinfer.infer(sample_data, num_threads=1)
        ts2 = tsinfer.infer(sample_data, num_threads=5)
        self.assertTreeSequencesEqual(ts1, ts2)


class TestAncestorGeneratorsEquivalant(unittest.TestCase):
    """
    Tests for the ancestor generation process.
    """

    def verify_ancestor_generator(self, genotypes, num_threads=0):
        m, n = genotypes.shape
        with tsinfer.SampleData() as sample_data:
            for j in range(m):
                sample_data.add_site(j, genotypes[j])

        adc = tsinfer.generate_ancestors(
            sample_data, engine=tsinfer.C_ENGINE, num_threads=num_threads)
        adp = tsinfer.generate_ancestors(
            sample_data, engine=tsinfer.PY_ENGINE, num_threads=num_threads)

        # # TODO clean this up when we're finished mucking around with the
        # # ancestor generator.
        # print()
        # print(adc.ancestors_start[:])
        # print(adp.ancestors_start[:])
        # assert np.array_equal(adc.ancestors_start[:], adp.ancestors_start[:])

        # print("end:")
        # print(adc.ancestors_end[:])
        # print(adp.ancestors_end[:])
        # assert np.array_equal(adc.ancestors_end[:], adp.ancestors_end[:])

        # print("focal_sites:")
        # print(adc.ancestors_focal_sites[:])
        # print(adp.ancestors_focal_sites[:])

        # print("haplotype:")
        # print(adc.ancestors_haplotype[:])
        # print()
        # print(adp.ancestors_haplotype[:])

        # j = 0
        # for h1, h2 in zip(adc.ancestors_haplotype[:], adp.ancestors_haplotype[:]):
        #     if not np.array_equal(h1, h2):
        #         print(h1)
        #         print(h2)
        #         print(adp.ancestors_focal_sites[j])
        #         print(adc.ancestors_focal_sites[j])
        #         print(adc.ancestors_start[j])
        #         print(adc.ancestors_end[j])
        #     j += 1
        # print(adc)
        # print(adp)
        self.assertTrue(adp.data_equal(adc))

    def test_no_recombination(self):
        ts = msprime.simulate(
            20, length=1, recombination_rate=0, mutation_rate=1, random_seed=1)
        assert ts.num_sites > 0 and ts.num_sites < 50
        self.verify_ancestor_generator(ts.genotype_matrix())

    def test_with_recombination_short(self):
        ts = msprime.simulate(
            20, length=1, recombination_rate=1, mutation_rate=1, random_seed=1)
        assert ts.num_trees > 1
        assert ts.num_sites > 0 and ts.num_sites < 50
        self.verify_ancestor_generator(ts.genotype_matrix())

    def test_with_recombination_long(self):
        ts = msprime.simulate(
            20, length=50, recombination_rate=1, mutation_rate=1, random_seed=1)
        assert ts.num_trees > 1
        assert ts.num_sites > 100
        self.verify_ancestor_generator(ts.genotype_matrix())

    def test_random_data(self):
        G, _ = get_random_data_example(20, 50, seed=1234)
        # G, _ = get_random_data_example(20, 10)
        self.verify_ancestor_generator(G)

    def test_random_data_threads(self):
        G, _ = get_random_data_example(20, 50, seed=1234)
        # G, _ = get_random_data_example(20, 10)
        self.verify_ancestor_generator(G, num_threads=4)

    def test_with_recombination_long_threads(self):
        ts = msprime.simulate(
            20, length=50, recombination_rate=1, mutation_rate=1, random_seed=1)
        assert ts.num_trees > 1
        assert ts.num_sites > 100
        self.verify_ancestor_generator(ts.genotype_matrix(), num_threads=3)


class TestGeneratedAncestors(unittest.TestCase):
    """
    Ensures we work correctly with the ancestors recovered from the
    simulations.
    """
    def verify_inserted_ancestors(self, ts):
        # Verifies that we can round-trip the specified tree sequence
        # using the generated ancestors. NOTE: this must be an SMC
        # consistent tree sequence!
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as sample_data:
            for v in ts.variants():
                sample_data.add_site(v.position, v.genotypes, v.alleles)

        ancestor_data = tsinfer.AncestorData(sample_data)
        tsinfer.build_simulated_ancestors(sample_data, ancestor_data, ts)
        ancestor_data.finalise()

        A = np.zeros(
            (ancestor_data.num_sites, ancestor_data.num_ancestors), dtype=np.uint8)
        start = ancestor_data.ancestors_start[:]
        end = ancestor_data.ancestors_end[:]
        ancestors = ancestor_data.ancestors_haplotype[:]
        for j in range(ancestor_data.num_ancestors):
            A[start[j]: end[j], j] = ancestors[j]
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ancestors_ts = tsinfer.match_ancestors(
                sample_data, ancestor_data, engine=engine)
            self.assertEqual(ancestor_data.num_sites, ancestors_ts.num_sites)
            self.assertEqual(ancestor_data.num_ancestors, ancestors_ts.num_samples)
            self.assertTrue(np.array_equal(ancestors_ts.genotype_matrix(), A))
            inferred_ts = tsinfer.match_samples(
                sample_data, ancestors_ts, engine=engine)
            self.assertTrue(np.array_equal(
                inferred_ts.genotype_matrix(), ts.genotype_matrix()))

    def test_no_recombination(self):
        ts = msprime.simulate(
            20, length=1, recombination_rate=0, mutation_rate=1,
            random_seed=1, model="smc_prime")
        assert ts.num_sites > 0 and ts.num_sites < 50
        self.verify_inserted_ancestors(ts)

    def test_small_sample_high_recombination(self):
        ts = msprime.simulate(
            4, length=1, recombination_rate=5, mutation_rate=1,
            random_seed=1, model="smc_prime")
        assert ts.num_sites > 0 and ts.num_sites < 50 and ts.num_trees > 3
        self.verify_inserted_ancestors(ts)

    def test_high_recombination(self):
        ts = msprime.simulate(
            30, length=1, recombination_rate=5, mutation_rate=1,
            random_seed=1, model="smc_prime")
        assert ts.num_sites > 0 and ts.num_sites < 50 and ts.num_trees > 3
        self.verify_inserted_ancestors(ts)


class TestBuildAncestors(unittest.TestCase):
    """
    Tests for the generate_ancestors function.
    """
    def get_simulated_example(self, ts):
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as sample_data:
            for variant in ts.variants():
                sample_data.add_site(
                    variant.site.position, variant.genotypes, variant.alleles)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        return sample_data, ancestor_data

    def verify_ancestors(self, sample_data, ancestor_data):
        ancestors = ancestor_data.ancestors_haplotype[:]
        inference_sites = sample_data.sites_inference[:]
        position = sample_data.sites_position[:][inference_sites == 1]
        sample_genotypes = sample_data.sites_genotypes[:][inference_sites == 1, :]
        start = ancestor_data.ancestors_start[:]
        end = ancestor_data.ancestors_end[:]
        time = ancestor_data.ancestors_time[:]
        focal_sites = ancestor_data.ancestors_focal_sites[:]

        self.assertEqual(ancestor_data.num_ancestors, ancestors.shape[0])
        self.assertEqual(ancestor_data.num_sites, sample_data.num_inference_sites)
        self.assertEqual(ancestor_data.num_ancestors, time.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, start.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, end.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, focal_sites.shape[0])
        self.assertTrue(np.array_equal(ancestor_data.sites_position[:], position))
        # The first ancestor must be all zeros.
        self.assertEqual(start[0], 0)
        self.assertEqual(end[0], ancestor_data.num_sites)
        self.assertEqual(list(focal_sites[0]), [])
        self.assertTrue(np.all(ancestors[0] == 0))

        used_sites = []
        frequency_time_map = {}
        for j in range(ancestor_data.num_ancestors):
            a = ancestors[j]
            self.assertEqual(a.shape[0], end[j] - start[j])
            h = np.zeros(ancestor_data.num_sites, dtype=np.uint8)
            h[start[j]: end[j]] = a
            self.assertTrue(np.all(h[start[j]:end[j]] != tsinfer.UNKNOWN_ALLELE))
            self.assertTrue(np.all(h[focal_sites[j]] == 1))
            used_sites.extend(focal_sites[j])
            self.assertGreater(time[j], 0)
            if j > 0:
                self.assertGreaterEqual(time[j - 1], time[j])
            for site in focal_sites[j]:
                # The time value should be the same for all sites with the same
                # frequency
                freq = np.sum(sample_genotypes[site])
                if freq not in frequency_time_map:
                    frequency_time_map[freq] = time[j]
                self.assertEqual(frequency_time_map[freq], time[j])
        self.assertEqual(sorted(used_sites), list(range(ancestor_data.num_sites)))

    def test_simulated_no_recombination(self):
        ts = msprime.simulate(10, mutation_rate=10, random_seed=10)
        self.assertGreater(ts.num_sites, 10)
        sample_data, ancestor_data = self.get_simulated_example(ts)
        self.verify_ancestors(sample_data, ancestor_data)

    def test_simulated_recombination(self):
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=10, random_seed=10)
        self.assertGreater(ts.num_sites, 10)
        sample_data, ancestor_data = self.get_simulated_example(ts)
        self.verify_ancestors(sample_data, ancestor_data)
        # Make sure we have at least one partial ancestor.
        start = ancestor_data.ancestors_start[:]
        end = ancestor_data.ancestors_end[:]
        self.assertLess(np.min(end - start), ancestor_data.num_sites)

    def test_random_data(self):
        n = 20
        m = 50
        G, positions = get_random_data_example(n, m)
        sample_data = tsinfer.SampleData(sequence_length=m)
        for genotypes, position in zip(G, positions):
            sample_data.add_site(position, genotypes)
        sample_data.finalise()
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        self.verify_ancestors(sample_data, ancestor_data)


class TestAncestorsTreeSequence(unittest.TestCase):
    """
    Tests for the output of the match_ancestors function.
    """
    def verify(self, sample_data):
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        tables = ancestors_ts.tables
        self.assertTrue(np.array_equal(
            tables.sites.position, ancestor_data.sites_position[:]))
        self.assertEqual(ancestors_ts.num_samples, ancestor_data.num_ancestors)
        H = ancestors_ts.genotype_matrix().T
        for ancestor in ancestor_data.ancestors():
            self.assertTrue(np.array_equal(
                H[ancestor.id, ancestor.start: ancestor.end],
                ancestor.haplotype))

    def test_no_recombination(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=234)
        self.verify(tsinfer.SampleData.from_tree_sequence(ts))

    def test_recombination(self):
        ts = msprime.simulate(10, mutation_rate=2, recombination_rate=2, random_seed=233)
        self.verify(tsinfer.SampleData.from_tree_sequence(ts))

    def test_random_data(self):
        n = 25
        m = 50
        G, positions = get_random_data_example(n, m)
        sample_data = tsinfer.SampleData(sequence_length=m)
        for genotypes, position in zip(G, positions):
            sample_data.add_site(position, genotypes)
        sample_data.finalise()
        self.verify(sample_data)


class AlgorithmsExactlyEqualMixin(object):
    """
    For small example tree sequences, check that the Python and C implementations
    return precisely the same tree sequence when fed with perfect mutations.
    """
    def infer(self, ts, engine, path_compression=False):
        sample_data = tsinfer.SampleData(sequence_length=ts.sequence_length)
        for v in ts.variants():
            sample_data.add_site(v.site.position, v.genotypes, v.alleles)
        sample_data.finalise()

        ancestor_data = tsinfer.AncestorData(sample_data)
        tsinfer.build_simulated_ancestors(sample_data, ancestor_data, ts)
        ancestor_data.finalise()
        ancestors_ts = tsinfer.match_ancestors(
            sample_data, ancestor_data, engine=engine,
            path_compression=path_compression, extended_checks=True)
        inferred_ts = tsinfer.match_samples(
            sample_data, ancestors_ts, engine=engine, simplify=True,
            path_compression=path_compression, extended_checks=True)
        return inferred_ts

    def verify(self, ts):
        tsp = self.infer(
            ts, tsinfer.PY_ENGINE, path_compression=self.path_compression_enabled)
        tsc = self.infer(
            ts, tsinfer.C_ENGINE, path_compression=self.path_compression_enabled)
        self.assertEqual(ts.num_sites, tsp.num_sites)
        self.assertEqual(ts.num_sites, tsc.num_sites)
        self.assertEqual(tsc.num_samples, tsp.num_samples)
        tables_p = tsp.dump_tables()
        tables_c = tsc.dump_tables()
        self.assertEqual(tables_p.nodes, tables_c.nodes)
        self.assertEqual(tables_p.edges, tables_c.edges)
        self.assertEqual(tables_p.sites, tables_c.sites)
        self.assertEqual(tables_p.mutations, tables_c.mutations)

    def test_single_tree(self):
        for seed in range(10):
            ts = msprime.simulate(10, random_seed=seed + 1)
            ts = tsinfer.insert_perfect_mutations(ts)
            self.verify(ts)

    def test_three_samples(self):
        for seed in range(10):
            ts = msprime.simulate(
                3, recombination_rate=1, random_seed=seed + 1, model="smc_prime")
            ts = tsinfer.insert_perfect_mutations(ts)
            self.verify(ts)

    def test_four_samples(self):
        for seed in range(5):
            ts = msprime.simulate(
                4, recombination_rate=0.1, random_seed=seed + 1, length=10,
                model="smc_prime")
            ts = tsinfer.insert_perfect_mutations(ts, delta=1/8192)
            self.verify(ts)

    def test_five_samples(self):
        for seed in range(5):
            ts = msprime.simulate(
                5, recombination_rate=0.1, random_seed=seed + 100, length=10,
                model="smc_prime")
            ts = tsinfer.insert_perfect_mutations(ts, delta=1/8192)
            self.verify(ts)

    def test_ten_samples(self):
        for seed in range(5):
            ts = msprime.simulate(
                10, recombination_rate=0.1, random_seed=seed + 200, length=10,
                model="smc_prime")
            ts = tsinfer.insert_perfect_mutations(ts, delta=1/8192)
            self.verify(ts)

    def test_twenty_samples(self):
        for seed in range(5):
            ts = msprime.simulate(
                10, recombination_rate=0.1, random_seed=seed + 500, length=10,
                model="smc_prime")
            ts = tsinfer.insert_perfect_mutations(ts, delta=1/8192)
            self.verify(ts)


class TestAlgorithmsExactlyEqualNoPathCompression(
        unittest.TestCase, AlgorithmsExactlyEqualMixin):
    path_compression_enabled = False


class TestAlgorithmsExactlyEqualPathCompression(
        unittest.TestCase, AlgorithmsExactlyEqualMixin):
    path_compression_enabled = True


class TestPartialAncestorMatching(unittest.TestCase):
    """
    Tests for copying process behaviour when we have partially
    defined ancestors.
    """
    def verify_edges(self, sample_data, ancestor_data, expected_edges):

        def key(e):
            return (e.left, e.right, e.parent, e.child)

        for engine in [tsinfer.C_ENGINE, tsinfer.PY_ENGINE]:
            ts = tsinfer.match_ancestors(sample_data, ancestor_data, engine=engine)
            self.assertEqual(
                sorted(expected_edges, key=key), sorted(ts.edges(), key=key))

    def test_easy_case(self):
        num_sites = 6
        sample_data = tsinfer.SampleData()
        for j in range(num_sites):
            sample_data.add_site(j, [0, 1, 1])
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData(sample_data)

        ancestor_data.add_ancestor(  # ID 0
            start=0, end=6, focal_sites=[], time=5, haplotype=[0, 0, 0, 0, 0, 0])
        ancestor_data.add_ancestor(  # ID 1
            start=0, end=6, focal_sites=[], time=4, haplotype=[0, 0, 0, 0, 0, 0])
        ancestor_data.add_ancestor(  # ID 2
            start=0, end=3, focal_sites=[2], time=3,
            haplotype=[0, 0, 1, -1, -1, -1][0: 3])
        ancestor_data.add_ancestor(  # ID 3
            start=3, end=6, focal_sites=[4], time=2,
            haplotype=[-1, -1, -1, 0, 1, 0][3: 6])
        ancestor_data.add_ancestor(  # ID 4
            start=0, end=6, focal_sites=[0, 1, 3, 5], time=1,
            haplotype=[1, 1, 1, 1, 1, 1])
        ancestor_data.finalise()

        expected_edges = [
            msprime.Edge(0, 6, 0, 1),
            msprime.Edge(0, 3, 2, 4),
            msprime.Edge(3, 6, 3, 4),
            msprime.Edge(3, 6, 1, 3),
            msprime.Edge(0, 3, 1, 2)]
        self.verify_edges(sample_data, ancestor_data, expected_edges)

    def test_partial_overlap(self):
        num_sites = 7
        sample_data = tsinfer.SampleData()
        for j in range(num_sites):
            sample_data.add_site(j, [0, 1, 1])
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData(sample_data)

        ancestor_data.add_ancestor(  # ID 0
            start=0, end=7, focal_sites=[], time=5, haplotype=[0, 0, 0, 0, 0, 0, 0])
        ancestor_data.add_ancestor(  # ID 1
            start=0, end=7, focal_sites=[], time=4, haplotype=[0, 0, 0, 0, 0, 0, 0])
        ancestor_data.add_ancestor(  # ID 2
            start=0, end=3, focal_sites=[2], time=3,
            haplotype=[0, 0, 1, 0, 0, 0, 0][0: 3])
        ancestor_data.add_ancestor(  # ID 3
            start=3, end=7, focal_sites=[4, 6], time=2,
            haplotype=[-1, -1, -1, 0, 1, 0, 1][3: 7])
        ancestor_data.add_ancestor(  # ID 4
            start=0, end=7, focal_sites=[0, 1, 3, 5], time=1,
            haplotype=[1, 1, 1, 1, 1, 1, 1])
        ancestor_data.finalise()

        expected_edges = [
            msprime.Edge(0, 7, 0, 1),
            msprime.Edge(0, 3, 2, 4),
            msprime.Edge(3, 7, 3, 4),
            msprime.Edge(3, 7, 1, 3),
            msprime.Edge(0, 3, 1, 2)]
        self.verify_edges(sample_data, ancestor_data, expected_edges)

    def test_edge_overlap_bug(self):
        num_sites = 12
        with tsinfer.SampleData() as sample_data:
            for j in range(num_sites):
                sample_data.add_site(j, [0, 1, 1])
        ancestor_data = tsinfer.AncestorData(sample_data)

        ancestor_data.add_ancestor(  # ID 0
            start=0, end=12, focal_sites=[], time=8,
            haplotype=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ancestor_data.add_ancestor(  # ID 1
            start=0, end=12, focal_sites=[], time=7,
            haplotype=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ancestor_data.add_ancestor(  # ID 2
            start=0, end=4, focal_sites=[], time=6,
            haplotype=[0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1][0: 4])
        ancestor_data.add_ancestor(  # ID 3
            start=4, end=12, focal_sites=[], time=5,
            haplotype=[-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0][4: 12])
        ancestor_data.add_ancestor(  # ID 4
            start=8, end=12, focal_sites=[9, 11], time=4,
            haplotype=[-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 0, 1][8: 12])
        ancestor_data.add_ancestor(  # ID 5
            start=4, end=8, focal_sites=[5, 7], time=3,
            haplotype=[-1, -1, -1, -1, 0, 1, 0, 1, -1, -1, -1, -1][4: 8])
        ancestor_data.add_ancestor(  # ID 6
            start=0, end=4, focal_sites=[1, 3], time=2,
            haplotype=[0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1][0: 4])
        ancestor_data.add_ancestor(  # ID 7
            start=0, end=12, focal_sites=[0, 2, 4, 6, 8, 10], time=1,
            haplotype=[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0])
        ancestor_data.finalise()

        expected_edges = [
            msprime.Edge(0, 12, 0, 1),
            msprime.Edge(0, 4, 1, 2),
            msprime.Edge(4, 12, 1, 3),
            msprime.Edge(8, 12, 1, 4),
            msprime.Edge(4, 8, 1, 5),
            msprime.Edge(0, 4, 1, 6),
            msprime.Edge(0, 4, 1, 7),
            msprime.Edge(4, 8, 5, 7),
            msprime.Edge(8, 12, 1, 7)]
        self.verify_edges(sample_data, ancestor_data, expected_edges)


class TestBadEngine(unittest.TestCase):
    """
    Check that we catch bad engines parameters.
    """
    bad_engines = ["CCCC", "c", "p", "Py", "python"]

    def get_example(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=3)
        return tsinfer.SampleData.from_tree_sequence(ts)

    def test_infer(self):
        sample_data = self.get_example()
        for bad_engine in self.bad_engines:
            self.assertRaises(ValueError, tsinfer.infer, sample_data, engine=bad_engine)

    def test_generate_ancestors(self):
        sample_data = self.get_example()
        for bad_engine in self.bad_engines:
            self.assertRaises(
                ValueError, tsinfer.generate_ancestors, sample_data, engine=bad_engine)

    def test_match_ancestors(self):
        sample_data = self.get_example()
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        for bad_engine in self.bad_engines:
            self.assertRaises(
                ValueError, tsinfer.match_ancestors, sample_data, ancestor_data,
                engine=bad_engine)

    def test_match_samples(self):
        sample_data = self.get_example()
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        for bad_engine in self.bad_engines:
            self.assertRaises(
                ValueError, tsinfer.match_samples, sample_data, ancestors_ts,
                engine=bad_engine)


class TestWrongTreeSequence(unittest.TestCase):
    """
    Tests covering what happens when we provide an incorrect tree sequence
    as the ancestrors_ts.
    Initial issue: https://github.com/tskit-dev/tsinfer/issues/53
    """
    def test_wrong_ts_match_samples(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        inferred_ts = tsinfer.infer(sample_data)
        self.assertRaises(ValueError, tsinfer.match_samples, sample_data, inferred_ts)

    def test_original_ts_match_samples(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        # This raises an error because we have non-inference sites in the
        # original ts.
        self.assertRaises(ValueError, tsinfer.match_samples, sample_data, sim)

    def test_different_ancestors_ts_match_samples(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)

        sim = msprime.simulate(sample_size=6, random_seed=2, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        self.assertRaises(ValueError, tsinfer.match_samples, sample_data, ancestors_ts)

    def test_bad__edge_position(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)

        tables = ancestors_ts.dump_tables()
        # To make things easy, add a new node we can refer to without mucking
        # up the existing topology
        node = tables.nodes.add_row(flags=1)
        tables.edges.add_row(0.5, 1.0, node - 1, node)
        tables.sort()
        bad_ts = tables.tree_sequence()
        self.assertRaises(ValueError, tsinfer.match_samples, sample_data, bad_ts)

        # Same thing for the right coordinate.
        tables = ancestors_ts.dump_tables()
        node = tables.nodes.add_row(flags=1)
        tables.edges.add_row(0, 0.5, node - 1, node)
        tables.sort()
        bad_ts = tables.tree_sequence()
        self.assertRaises(ValueError, tsinfer.match_samples, sample_data, bad_ts)


class TestSimplify(unittest.TestCase):
    """
    Check that the simplify argument to infer is correctly invoked.
    """
    def verify(self, ts):
        n = ts.num_samples
        self.assertGreater(ts.num_sites, 2)
        sd = tsinfer.SampleData.from_tree_sequence(ts)
        ts1 = tsinfer.infer(sd, simplify=True)
        # When simplify is true the samples should be zero to n.
        self.assertEqual(list(ts1.samples()), list(range(n)))
        for tree in ts1.trees():
            self.assertEqual(tree.num_samples(), len(list(tree.leaves())))

        # When simplify is true and there is no path compression,
        # the samples should be zero to N - n up to n
        ts2 = tsinfer.infer(sd, simplify=False, path_compression=False)
        self.assertEqual(
            list(ts2.samples()),
            list(range(ts2.num_nodes - n, ts2.num_nodes)))

    def test_single_tree(self):
        ts = msprime.simulate(5, random_seed=1, mutation_rate=2)
        self.verify(ts)

    def test_many_trees(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts)


def squash_edges(ts):
    """
    Returns the edges in the tree sequence squashed.
    """
    t = ts.tables.nodes.time
    edges = list(ts.edges())
    edges.sort(key=lambda e: (t[e.parent], e.parent, e.child, e.left))
    if len(edges) == 0:
        return []

    squashed = []
    last_e = edges[0]
    for e in edges[1:]:
        condition = (
            e.parent != last_e.parent or
            e.child != last_e.child or
            e.left != last_e.right)
        if condition:
            squashed.append(last_e)
            last_e = e
        last_e.right = e.right
    squashed.append(last_e)
    return squashed


class TestMinimise(unittest.TestCase):
    """
    Tests for the minimise function.
    """

    def verify(self, ts):
        source_tables = ts.tables
        positions = set(source_tables.sites.position)
        positions.add(0)
        positions.add(ts.sequence_length)
        mts = tsinfer.minimise(ts)
        for edge in mts.edges():
            self.assertIn(edge.left, positions)
            self.assertIn(edge.right, positions)
        minimised_trees = mts.trees()
        minimised_tree = next(minimised_trees)
        minimised_tree_sites = minimised_tree.sites()
        for tree in ts.trees():
            for site in tree.sites():
                minimised_site = next(minimised_tree_sites, None)
                if minimised_site is None:
                    minimised_tree = next(minimised_trees)
                    minimised_tree_sites = minimised_tree.sites()
                    minimised_site = next(minimised_tree_sites)
                self.assertEqual(site, minimised_site)
            if tree.num_sites > 0:
                self.assertEqual(tree.parent_dict, minimised_tree.parent_dict)
        self.assertTrue(np.array_equal(ts.genotype_matrix(), mts.genotype_matrix()))

        edges = list(mts.edges())
        squashed = squash_edges(mts)
        self.assertEqual(len(edges), len(squashed))
        self.assertEqual(edges, squashed)

    def test_simple_recombination(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        self.verify(ts)

    def test_large_recombination(self):
        ts = msprime.simulate(25, random_seed=12, recombination_rate=5, mutation_rate=15)
        self.verify(ts)

    def test_no_recombination(self):
        ts = msprime.simulate(5, random_seed=1, mutation_rate=2)
        self.verify(ts)

    def test_no_mutation(self):
        ts = msprime.simulate(5, random_seed=1)
        self.verify(ts)


class TestMatchSiteSubsets(unittest.TestCase):
    """
    Tests that we can successfully run the algorithm on data in which we have
    a subset of the original sites.
    """
    def subset_sites(self, ts, position):
        """
        Return a copy of the specified tree sequence with sites reduced to those
        with positions in the specified list.
        """
        tables = ts.dump_tables()
        lookup = set(position)
        tables.sites.clear()
        tables.mutations.clear()
        for site in ts.sites():
            if site.position in lookup:
                site_id = tables.sites.add_row(
                    site.position, ancestral_state=site.ancestral_state,
                    metadata=site.metadata)
                for mutation in site.mutations:
                    tables.mutations.add_row(
                        site_id, node=mutation.node, parent=mutation.parent,
                        derived_state=mutation.derived_state,
                        metadata=mutation.metadata)
        self.assertTrue(np.array_equal(tables.sites.position, position))
        return tables.tree_sequence()

    def verify(self, sample_data, position_subset):
        full_ts = tsinfer.infer(sample_data)
        subset_ts = self.subset_sites(full_ts, position_subset)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        subset_ancestors_ts = tsinfer.minimise(
            self.subset_sites(ancestors_ts, position_subset))
        subset_ancestors_ts = subset_ancestors_ts.simplify()
        subset_sample_data = tsinfer.SampleData.from_tree_sequence(subset_ts)
        output_ts = tsinfer.match_samples(subset_sample_data, subset_ancestors_ts)
        self.assertTrue(
            np.array_equal(output_ts.genotype_matrix(), subset_ts.genotype_matrix()))

    def test_simple_case(self):
        ts = msprime.simulate(10, mutation_rate=2, recombination_rate=2, random_seed=3)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        position = sample_data.sites_position[:][sample_data.sites_inference[:] == 1]
        self.verify(sample_data, position[:][::2])

    def test_one_sites(self):
        ts = msprime.simulate(15, mutation_rate=2, recombination_rate=2, random_seed=3)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        position = sample_data.sites_position[:][sample_data.sites_inference[:] == 1]
        self.verify(sample_data, position[:1])

    def test_no_recombination(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=4)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        position = sample_data.sites_position[:][sample_data.sites_inference[:] == 1]
        self.verify(sample_data, position[:][1::2])

    def test_random_data(self):
        n = 25
        m = 50
        G, positions = get_random_data_example(n, m)
        sample_data = tsinfer.SampleData(sequence_length=m)
        for genotypes, position in zip(G, positions):
            sample_data.add_site(position, genotypes)
        sample_data.finalise()
        position = sample_data.sites_position[:][sample_data.sites_inference[:] == 1]
        self.verify(sample_data, position[:][::2])


class PathCompressionMixin(object):
    """
    Common utilities for testing a tree sequence with path compression.
    """
    def verify_tree_sequence(self, ts):
        num_fraction_times = sum(
            math.floor(node.time) != node.time for node in ts.nodes())
        synthetic_nodes = [
            node for node in ts.nodes() if tsinfer.is_synthetic(node.flags)]
        self.assertGreater(len(synthetic_nodes), 0)
        # Synthetic nodes will mostly have fractional times, so this number
        # should at most the nuber of synthetic nodes.
        self.assertGreaterEqual(len(synthetic_nodes), num_fraction_times)
        for node in synthetic_nodes:
            # print("Synthetic node", node)
            parent_edges = [edge for edge in ts.edges() if edge.parent == node.id]
            child_edges = [edge for edge in ts.edges() if edge.child == node.id]
            self.assertGreater(len(parent_edges), 1)
            self.assertGreater(len(child_edges), 1)
            child_edges.sort(key=lambda e: e.left)
            # print("parent edges")
            # for edge in parent_edges:
            #     print("\t", edge)
            # print("child edges")
            # Child edges should always be contiguous
            last_right = child_edges[0].left
            for edge in child_edges:
                # print("\t", edge)
                self.assertEqual(last_right, edge.left)
                last_right = edge.right
            left = child_edges[0].left
            right = child_edges[-1].right
            original_matches = [
                e for e in parent_edges if e.left == left and e.right == right]
            # We must have at least two initial edges that exactly span the
            # synthetic interval.
            self.assertGreater(len(original_matches), 1)

    def test_simple_case(self):
        ts = msprime.simulate(55, mutation_rate=5, random_seed=4, recombination_rate=8)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)

    def test_simulation_with_error(self):
        ts = msprime.simulate(50, mutation_rate=5, random_seed=4, recombination_rate=8)
        ts = eval_util.insert_errors(ts, 0.1, seed=32)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)

    def test_small_random_data(self):
        n = 25
        m = 20
        G, positions = get_random_data_example(n, m)
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for genotypes, position in zip(G, positions):
                sample_data.add_site(position, genotypes)
        self.verify(sample_data)

    def test_large_random_data(self):
        n = 100
        m = 30
        G, positions = get_random_data_example(n, m)
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for genotypes, position in zip(G, positions):
                sample_data.add_site(position, genotypes)
        self.verify(sample_data)


class PathCompressionAncestorsMixin(PathCompressionMixin):
    """
    Tests for the results of path compression on an ancestors tree sequence.
    """
    def verify(self, sample_data):
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ts = tsinfer.match_ancestors(
            sample_data, ancestor_data, engine=self.engine, extended_checks=True)
        self.verify_tree_sequence(ts)


class TestPathCompressionAncestorsPyEngine(
        PathCompressionAncestorsMixin, unittest.TestCase):
    engine = tsinfer.PY_ENGINE


class TestPathCompressionAncestorsCEngine(
        PathCompressionAncestorsMixin, unittest.TestCase):
    engine = tsinfer.C_ENGINE

    def test_c_engine_fail_example(self):
        # Reproduce a failure that occured under the C engine.
        ts = msprime.simulate(
            20, Ne=10**4, length=0.25 * 10**6,
            recombination_rate=1e-8, mutation_rate=1e-8,
            random_seed=4)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)


class PathCompressionSamplesMixin(PathCompressionMixin):
    """
    Tests for the results of path compression just on samples.
    """
    def verify(self, sample_data):
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        # Turn off path compression in the ancestors to make this as difficult
        # as possible.
        ancestors_ts = tsinfer.match_ancestors(
            sample_data, ancestor_data, path_compression=False)
        ts = tsinfer.match_samples(
            sample_data, ancestors_ts, path_compression=True, engine=self.engine,
            extended_checks=True)
        self.verify_tree_sequence(ts)


class TestPathCompressionSamplesPyEngine(
        PathCompressionSamplesMixin, unittest.TestCase):
    engine = tsinfer.PY_ENGINE


class TestPathCompressionSamplesCEngine(
        PathCompressionSamplesMixin, unittest.TestCase):
    engine = tsinfer.C_ENGINE


class PathCompressionFullStackMixin(PathCompressionMixin):
    """
    Tests for the results of path compression just on samples.
    """
    def verify(self, sample_data):
        # We have to turn off simplify because it'll sometimes remove chunks
        # of synthetic ancestors, breaking out continguity requirements.
        ts = tsinfer.infer(
            sample_data, path_compression=True, engine=self.engine,
            simplify=False)
        self.verify_tree_sequence(ts)


class TestPathCompressionFullStackPyEngine(
        PathCompressionFullStackMixin, unittest.TestCase):
    engine = tsinfer.PY_ENGINE


class TestPathCompressionFullStackCEngine(
        PathCompressionFullStackMixin, unittest.TestCase):
    engine = tsinfer.C_ENGINE


class TestSyntheticFlag(unittest.TestCase):
    """
    Tests if we can set and detect the synthetic node flag correctly.
    """
    BIT_POSITION = 16

    def test_is_synthetic(self):
        self.assertFalse(tsinfer.is_synthetic(0))
        self.assertFalse(tsinfer.is_synthetic(1))
        self.assertTrue(tsinfer.is_synthetic(tsinfer.SYNTHETIC_NODE_BIT))
        for bit in range(32):
            flags = 1 << bit
            if bit == self.BIT_POSITION:
                self.assertTrue(tsinfer.is_synthetic(flags))
            else:
                self.assertFalse(tsinfer.is_synthetic(flags))
        flags = tsinfer.SYNTHETIC_NODE_BIT
        for bit in range(32):
            flags |= 1 << bit
            self.assertTrue(tsinfer.is_synthetic(flags))
        flags = 0
        for bit in range(32):
            if bit != self.BIT_POSITION:
                flags |= 1 << bit
            self.assertFalse(tsinfer.is_synthetic(flags))

    def test_count_synthetic(self):
        self.assertEqual(tsinfer.count_synthetic([0]), 0)
        self.assertEqual(tsinfer.count_synthetic([tsinfer.SYNTHETIC_NODE_BIT]), 1)
        self.assertEqual(tsinfer.count_synthetic([0, 0]), 0)
        self.assertEqual(tsinfer.count_synthetic([0, tsinfer.SYNTHETIC_NODE_BIT]), 1)
        self.assertEqual(tsinfer.count_synthetic(
            [tsinfer.SYNTHETIC_NODE_BIT, tsinfer.SYNTHETIC_NODE_BIT]), 2)
        self.assertEqual(tsinfer.count_synthetic([1, tsinfer.SYNTHETIC_NODE_BIT]), 1)
        self.assertEqual(tsinfer.count_synthetic(
            [1 | tsinfer.SYNTHETIC_NODE_BIT, 1 | tsinfer.SYNTHETIC_NODE_BIT]), 2)

    def test_count_synthetic_random(self):
        np.random.seed(42)
        flags = np.random.randint(0, high=2**32, size=100, dtype=np.uint32)
        count = sum(map(tsinfer.is_synthetic, flags))
        self.assertEqual(count, tsinfer.count_synthetic(flags))
