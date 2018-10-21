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


class TestAugmentedAncestorsRoundTrip(TestRoundTrip):
    """
    Tests that we correctly round drip data when we have augmented ancestors.
    """
    def verify_data_round_trip(self, genotypes, positions, sequence_length=None):
        if sequence_length is None:
            sequence_length = positions[-1] + 1
        with tsinfer.SampleData(sequence_length=sequence_length) as sample_data:
            for j in range(genotypes.shape[0]):
                sample_data.add_site(positions[j], genotypes[j])
        ancestors = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestors)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            augmented_ts = tsinfer.augment_ancestors(
                sample_data, ancestors_ts, np.arange(sample_data.num_samples),
                engine=engine)
            ts = tsinfer.match_samples(sample_data, augmented_ts, engine=engine)
            self.assertEqual(ts.sequence_length, sequence_length)
            self.assertEqual(ts.num_sites, len(positions))
            for v in ts.variants():
                self.assertEqual(v.position, positions[v.index])
                self.assertTrue(np.array_equal(genotypes[v.index], v.genotypes))


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
        for simplify in [False, True]:
            output_ts = tsinfer.infer(sample_data, simplify=simplify)
            for tree in output_ts.trees():
                for site in tree.sites():
                    f = np.sum(genotypes[site.id])
                    if f == 0:
                        self.assertEqual(len(site.mutations), 0)
                    elif f == output_ts.num_samples:
                        self.assertEqual(len(site.mutations), 1)
                        self.assertEqual(site.mutations[0].node, tree.root)
                    self.assertLess(len(site.mutations), output_ts.num_samples)
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

    def test_random_data(self):
        genotypes, _ = get_random_data_example(20, 50, seed=12345)
        inference = np.sum(genotypes, axis=1) > 1
        inference[::2] = False
        self.verify_round_trip(genotypes, inference)


class TestZeroNonInferenceSites(unittest.TestCase):
    """
    Test the case where we have no non-inference sites.
    """
    def verify(self, sample_data):
        with self.assertLogs("tsinfer.inference", level="INFO") as logs:
            ts = tsinfer.infer(sample_data)
        messages = [record.msg for record in logs.records]
        self.assertIn("Inserting detailed site information", messages)
        tsinfer.verify(sample_data, ts)

    def test_many_sites(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=21)
        self.assertGreater(ts.num_sites, 2)
        genotypes = ts.genotype_matrix()
        non_singletons = np.sum(genotypes, axis=1) > 1
        genotypes = genotypes[non_singletons]
        m = genotypes.shape[1]
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for g, position in zip(genotypes, np.arange(m)):
                sample_data.add_site(position, g)
        self.verify(sample_data)

    def test_many_sites_letter_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=21)
        self.assertGreater(ts.num_sites, 2)
        genotypes = ts.genotype_matrix()
        non_singletons = np.sum(genotypes, axis=1) > 1
        genotypes = genotypes[non_singletons]
        m = genotypes.shape[1]
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for g, position in zip(genotypes, np.arange(m)):
                sample_data.add_site(position, g, alleles=["A", "G"])
        self.verify(sample_data)

    def test_one_site(self):
        genotypes = np.array([[1, 1, 0]])
        m = genotypes.shape[1]
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for g, position in zip(genotypes, np.arange(m)):
                sample_data.add_site(position, g)
        self.verify(sample_data)


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
        self.assertEqual(sample_data.num_non_inference_sites, m)
        self.assertEqual(sample_data.num_inference_sites, 0)
        for path_compression in [False, True]:
            output_ts = tsinfer.infer(sample_data, path_compression=path_compression)
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
        self.verify([[1, 1, 1]])

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

        for variant in sample_data.variants():
            self.assertEqual(all_metadata[variant.site.id], variant.site.metadata)

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
            tsinfer.check_ancestors_ts(ancestors_ts)
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
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
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

        # The provenance should be same as in the samples data file, plus an
        # extra row.
        self.assertEqual(ancestor_data.num_provenances, sample_data.num_provenances + 1)
        for j in range(sample_data.num_provenances):
            self.assertEqual(
                ancestor_data.provenances_record[j], sample_data.provenances_record[j])
            self.assertEqual(
                ancestor_data.provenances_timestamp[j],
                sample_data.provenances_timestamp[j])

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
        for path_compression in [True, False]:
            ancestors_ts = tsinfer.match_ancestors(
                sample_data, ancestor_data, path_compression=path_compression)
            tsinfer.check_ancestors_ts(ancestors_ts)
            tables = ancestors_ts.tables
            self.assertTrue(np.array_equal(
                tables.sites.position, ancestor_data.sites_position[:]))
            self.assertEqual(ancestors_ts.num_samples, ancestor_data.num_ancestors)
            H = ancestors_ts.genotype_matrix().T
            for ancestor in ancestor_data.ancestors():
                self.assertTrue(np.array_equal(
                    H[ancestor.id, ancestor.start: ancestor.end],
                    ancestor.haplotype))

            # The provenance should be same as in the ancestors data file, plus an
            # extra row.
            self.assertEqual(
                ancestor_data.num_provenances + 1, ancestors_ts.num_provenances)
            for j in range(ancestor_data.num_provenances):
                p = ancestors_ts.provenance(j)
                self.assertEqual(
                    ancestor_data.provenances_record[j], json.loads(p.record))
                self.assertEqual(ancestor_data.provenances_timestamp[j], p.timestamp)

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


class TestAncestorsTreeSequenceFlags(unittest.TestCase):
    """
    Checks that arbitrary flags can be set in the ancestors tree
    sequence and recovered in the final ts.
    """
    def verify(self, sample_data, ancestors_ts):
        source_flags = ancestors_ts.tables.nodes.flags
        for engine in [tsinfer.C_ENGINE, tsinfer.PY_ENGINE]:
            for path_compression in [True, False]:
                ts = tsinfer.match_samples(
                    sample_data, ancestors_ts, path_compression=path_compression,
                    simplify=False, engine=engine)
                nodes = ts.tables.nodes
                flags = nodes.flags[:source_flags.shape[0]]
                # Anything that's marked as a sample in the ancestors should be a
                # 0 in the final outout
                samples = np.where(source_flags == 1)[0]
                self.assertTrue(np.all(flags[samples] == 0))
                # Anything that's not marked as a sample should be equal in both.
                non_samples = np.where(source_flags != 1)[0]
                self.assertTrue(np.all(flags[non_samples] == source_flags[non_samples]))

    def test_no_flags_changes(self):
        ts = msprime.simulate(10, mutation_rate=2, recombination_rate=2, random_seed=233)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ancestors = tsinfer.generate_ancestors(samples)
        ancestors_ts = tsinfer.match_ancestors(samples, ancestors)
        self.verify(samples, ancestors_ts)

    def test_append_nodes(self):
        ts = msprime.simulate(10, mutation_rate=2, recombination_rate=2, random_seed=233)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ancestors = tsinfer.generate_ancestors(samples)
        ancestors_ts = tsinfer.match_ancestors(samples, ancestors)
        tables = ancestors_ts.dump_tables()
        tables.nodes.add_row(flags=1 << 15, time=1.1)
        tables.nodes.add_row(flags=1 << 16, time=1.1)
        tables.nodes.add_row(flags=1 << 17, time=1.1)
        tables.nodes.add_row(flags=1 << 18, time=1.0)
        self.verify(samples, tables.tree_sequence())


class TestAncestorsTreeSequenceIndividuals(unittest.TestCase):
    """
    Checks that we can have individuals in the ancestors tree sequence and
    that they are correctly preserved in the final TS.
    """
    def verify(self, sample_data, ancestors_ts):
        ts = tsinfer.match_samples(sample_data, ancestors_ts, simplify=False)
        self.assertEqual(
            ancestors_ts.num_individuals + sample_data.num_individuals,
            ts.num_individuals)
        # The ancestors individiduals should come first.
        final_individuals = ts.individuals()
        for ind in ancestors_ts.individuals():
            final_ind = next(final_individuals)
            self.assertEqual(final_ind, ind)
            # The nodes for this individual should *not* be samples.
            for u in final_ind.nodes:
                node = ts.node(u)
                self.assertFalse(node.is_sample())

        for ind1, ind2 in zip(final_individuals, sample_data.individuals()):
            self.assertTrue(np.array_equal(ind1.location, ind2.location))
            self.assertEqual(json.loads(ind1.metadata.decode()), ind2.metadata)
            # The nodes for this individual should *not* be samples.
            for u in ind1.nodes:
                node = ts.node(u)
                self.assertTrue(node.is_sample())

    def test_zero_individuals(self):
        ts = msprime.simulate(10, mutation_rate=2, recombination_rate=2, random_seed=233)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ancestors = tsinfer.generate_ancestors(samples)
        ancestors_ts = tsinfer.match_ancestors(samples, ancestors)
        self.verify(samples, ancestors_ts)

    def test_diploid_individuals(self):
        ts = msprime.simulate(10, mutation_rate=2, recombination_rate=2, random_seed=233)
        tables = ts.dump_tables()
        for j in range(ts.num_samples // 2):
            tables.individuals.add_row(flags=j, location=[j, j], metadata=b"X" * j)
        # Add these individuals to the first n nodes.
        individual = np.zeros(ts.num_nodes, dtype=np.int32) - 1
        x = np.arange(ts.num_samples // 2)
        individual[2 * x] = x
        individual[2 * x + 1] = x
        tables.nodes.set_columns(
            flags=tables.nodes.flags,
            time=tables.nodes.time,
            individual=individual)
        ts = tables.tree_sequence()
        with tsinfer.SampleData() as samples:
            for j in range(ts.num_samples // 2):
                samples.add_individual(ploidy=2, location=[100 * j], metadata={"X": j})
            for var in ts.variants():
                samples.add_site(var.site.position, var.genotypes)
        ancestors_ts = eval_util.make_ancestors_ts(samples, ts)
        self.verify(samples, ancestors_ts)


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


class TestWrongAncestorsTreeSequence(unittest.TestCase):
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

    def test_zero_node_times(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        tables = ancestors_ts.dump_tables()
        tables.nodes.add_row(time=0, flags=0)
        with self.assertRaises(ValueError):
            tsinfer.match_samples(sample_data, tables.tree_sequence())

    def test_different_ancestors_ts_match_samples(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)

        sim = msprime.simulate(sample_size=6, random_seed=2, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        self.assertRaises(ValueError, tsinfer.match_samples, sample_data, ancestors_ts)

    def test_bad_edge_position(self):
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
        pc_nodes = [
            node for node in ts.nodes() if tsinfer.is_pc_ancestor(node.flags)]
        self.assertGreater(len(pc_nodes), 0)
        # Synthetic nodes will mostly have fractional times, so this number
        # should at most the number of pc nodes.
        self.assertGreaterEqual(len(pc_nodes), num_fraction_times)
        for node in pc_nodes:
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
            # pc interval.
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
        # of pc ancestors, breaking out continguity requirements.
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


class TestFlags(unittest.TestCase):
    """
    Tests if we can set and detect the pc node flag correctly.
    """
    PC_BIT_POSITION = 16
    SRB_BIT_POSITION = 17

    def test_is_pc_ancestor(self):
        self.assertFalse(tsinfer.is_pc_ancestor(0))
        self.assertFalse(tsinfer.is_pc_ancestor(1))
        self.assertTrue(tsinfer.is_pc_ancestor(tsinfer.NODE_IS_PC_ANCESTOR))
        for bit in range(32):
            flags = 1 << bit
            if bit == self.PC_BIT_POSITION:
                self.assertTrue(tsinfer.is_pc_ancestor(flags))
            else:
                self.assertFalse(tsinfer.is_pc_ancestor(flags))
        flags = tsinfer.NODE_IS_PC_ANCESTOR
        for bit in range(32):
            flags |= 1 << bit
            self.assertTrue(tsinfer.is_pc_ancestor(flags))
        flags = 0
        for bit in range(32):
            if bit != self.PC_BIT_POSITION:
                flags |= 1 << bit
            self.assertFalse(tsinfer.is_pc_ancestor(flags))

    def test_count_pc_ancestors(self):
        self.assertEqual(tsinfer.count_pc_ancestors([0]), 0)
        self.assertEqual(tsinfer.count_pc_ancestors([tsinfer.NODE_IS_PC_ANCESTOR]), 1)
        self.assertEqual(tsinfer.count_pc_ancestors([0, 0]), 0)
        self.assertEqual(tsinfer.count_pc_ancestors([0, tsinfer.NODE_IS_PC_ANCESTOR]), 1)
        self.assertEqual(tsinfer.count_pc_ancestors(
            [tsinfer.NODE_IS_PC_ANCESTOR, tsinfer.NODE_IS_PC_ANCESTOR]), 2)
        self.assertEqual(tsinfer.count_pc_ancestors([1, tsinfer.NODE_IS_PC_ANCESTOR]), 1)
        self.assertEqual(tsinfer.count_pc_ancestors(
            [1 | tsinfer.NODE_IS_PC_ANCESTOR, 1 | tsinfer.NODE_IS_PC_ANCESTOR]), 2)

    def test_count_srb_ancestors_random(self):
        np.random.seed(42)
        flags = np.random.randint(0, high=2**32, size=100, dtype=np.uint32)
        count = sum(map(tsinfer.is_srb_ancestor, flags))
        self.assertEqual(count, tsinfer.count_srb_ancestors(flags))

    def test_is_srb_ancestor(self):
        self.assertFalse(tsinfer.is_srb_ancestor(0))
        self.assertFalse(tsinfer.is_srb_ancestor(1))
        self.assertTrue(tsinfer.is_srb_ancestor(tsinfer.NODE_IS_SRB_ANCESTOR))
        for bit in range(32):
            flags = 1 << bit
            if bit == self.SRB_BIT_POSITION:
                self.assertTrue(tsinfer.is_srb_ancestor(flags))
            else:
                self.assertFalse(tsinfer.is_srb_ancestor(flags))
        flags = tsinfer.NODE_IS_SRB_ANCESTOR
        for bit in range(32):
            flags |= 1 << bit
            self.assertTrue(tsinfer.is_srb_ancestor(flags))
        flags = 0
        for bit in range(32):
            if bit != self.SRB_BIT_POSITION:
                flags |= 1 << bit
            self.assertFalse(tsinfer.is_srb_ancestor(flags))

    def test_count_srb_ancestors(self):
        self.assertEqual(tsinfer.count_srb_ancestors([0]), 0)
        self.assertEqual(tsinfer.count_srb_ancestors([tsinfer.NODE_IS_SRB_ANCESTOR]), 1)
        self.assertEqual(tsinfer.count_srb_ancestors([0, 0]), 0)
        self.assertEqual(tsinfer.count_srb_ancestors(
            [0, tsinfer.NODE_IS_SRB_ANCESTOR]), 1)
        self.assertEqual(tsinfer.count_srb_ancestors(
            [tsinfer.NODE_IS_SRB_ANCESTOR, tsinfer.NODE_IS_SRB_ANCESTOR]), 2)
        self.assertEqual(tsinfer.count_srb_ancestors(
            [1, tsinfer.NODE_IS_SRB_ANCESTOR]), 1)
        self.assertEqual(tsinfer.count_srb_ancestors(
            [1 | tsinfer.NODE_IS_SRB_ANCESTOR, 1 | tsinfer.NODE_IS_SRB_ANCESTOR]), 2)

    def test_count_pc_ancestors_random(self):
        np.random.seed(42)
        flags = np.random.randint(0, high=2**32, size=100, dtype=np.uint32)
        count = sum(map(tsinfer.is_pc_ancestor, flags))
        self.assertEqual(count, tsinfer.count_pc_ancestors(flags))


class TestBugExamples(unittest.TestCase):
    """
    Run tests on some examples that provoked bugs.
    """
    def test_path_compression_bad_times(self):
        # This provoked a bug in which we created a pc ancestor
        # with the same time as its child, creating an invalid topology.
        sample_data = tsinfer.load(
            "tests/data/bugs/invalid_pc_ancestor_time.samples")
        ts = tsinfer.infer(sample_data)
        for var, (_, genotypes) in zip(ts.variants(), sample_data.genotypes()):
            self.assertTrue(np.array_equal(var.genotypes, genotypes))


class TestVerify(unittest.TestCase):
    """
    Checks that we correctly find problems with verify.
    """

    def test_nominal_case(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=1)
        self.assertGreater(ts.num_sites, 0)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred_ts = tsinfer.infer(samples)

        tsinfer.verify(samples, inferred_ts)
        tsinfer.verify(samples, ts)

    def test_bad_num_sites(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        with tsinfer.SampleData() as samples:
            samples.add_site(0, genotypes=[0, 1])

        with self.assertRaises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_num_samples(self):
        n = 5
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        with tsinfer.SampleData() as samples:
            for j in range(ts.num_sites):
                samples.add_site(j, genotypes=[0, 1])

        with self.assertRaises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_sequence_length(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        with tsinfer.SampleData(sequence_length=100) as samples:
            for j in range(ts.num_sites):
                samples.add_site(j, genotypes=[0, 1])

        with self.assertRaises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_site_position(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as samples:
            for var in ts.variants():
                samples.add_site(
                    position=var.site.position + 1e-6, genotypes=var.genotypes)

        with self.assertRaises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_alleles(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as samples:
            for var in ts.variants():
                samples.add_site(
                    position=var.site.position, alleles=["A", "T"],
                    genotypes=var.genotypes)

        with self.assertRaises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_genotypes(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as samples:
            for var in ts.variants():
                samples.add_site(
                    position=var.site.position, alleles=var.alleles,
                    genotypes=[0, 0])

        with self.assertRaises(ValueError):
            tsinfer.verify(samples, ts)


class TestExtractAncestors(unittest.TestCase):
    """
    Checks whether the extract_ancestors function correctly returns an ancestors
    tree sequence with the required properties.
    """
    def verify(self, samples):
        ancestors = tsinfer.generate_ancestors(samples)
        ancestors_ts_1 = tsinfer.match_ancestors(samples, ancestors)
        ts = tsinfer.match_samples(samples, ancestors_ts_1, simplify=False)
        t1 = ancestors_ts_1.dump_tables()
        t2, node_id_map = tsinfer.extract_ancestors(samples, ts)
        self.assertEqual(len(t2.provenances), len(t1.provenances) + 2)
        t1.provenances.clear()
        t2.provenances.clear()

        # Population data isn't carried through in ancestors tree sequences
        # for now.
        t2.populations.clear()

        self.assertEqual(t1.nodes, t2.nodes)
        self.assertEqual(t1.edges, t2.edges)
        self.assertEqual(t1.sites, t2.sites)
        self.assertEqual(t1.mutations, t2.mutations)
        self.assertEqual(t1.populations, t2.populations)
        self.assertEqual(t1.individuals, t2.individuals)
        self.assertEqual(t1.sites, t2.sites)

        self.assertEqual(t1, t2)

        for node in ts.nodes():
            if node_id_map[node.id] != -1:
                self.assertEqual(node.time, t1.nodes.time[node_id_map[node.id]])

    def test_simple_simulation(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=5, random_seed=2)
        self.verify(tsinfer.SampleData.from_tree_sequence(ts))

    def test_non_zero_one_mutations(self):
        ts = msprime.simulate(10, recombination_rate=5, random_seed=2)
        ts = msprime.mutate(
            ts, rate=5, model=msprime.InfiniteSites(msprime.NUCLEOTIDES),
            random_seed=15)
        self.assertGreater(ts.num_mutations, 0)
        self.verify(tsinfer.SampleData.from_tree_sequence(ts))

    def test_random_data_small_examples(self):
        np.random.seed(4)
        num_random_tests = 10
        for _ in range(num_random_tests):
            G, positions = get_random_data_example(5, 10)
            with tsinfer.SampleData(sequence_length=G.shape[0]) as samples:
                for j in range(G.shape[0]):
                    samples.add_site(positions[j], G[j])
            self.verify(samples)


class TestInsertSrbAncestors(unittest.TestCase):
    """
    Tests that the insert_srb_ancestors function behaves as expected.
    """

    def insert_srb_ancestors(self, samples, ts):

        srb_index = {}
        edges = sorted(ts.edges(), key=lambda e: (e.child, e.left))
        last_edge = edges[0]
        for edge in edges[1:]:
            condition = (
                ts.node(edge.child).is_sample() and
                edge.child == last_edge.child and
                edge.left == last_edge.right)
            if condition:
                key = edge.left, last_edge.parent, edge.parent
                if key in srb_index:
                    count, left_bound, right_bound = srb_index[key]
                    srb_index[key] = (
                        count + 1,
                        max(left_bound, last_edge.left),
                        min(right_bound, edge.right))
                else:
                    srb_index[key] = 1, last_edge.left, edge.right
            last_edge = edge

        tables, node_id_map = tsinfer.extract_ancestors(samples, ts)
        time = tables.nodes.time

        num_extra = 0
        for k, v in srb_index.items():
            if v[0] > 1:
                left, right = v[1:]
                x, pl, pr = k
                pl = node_id_map[pl]
                pr = node_id_map[pr]
                t = min(time[pl], time[pr]) - 1e-4
                node = tables.nodes.add_row(flags=tsinfer.NODE_IS_SRB_ANCESTOR, time=t)
                tables.edges.add_row(left, x, pl, node)
                tables.edges.add_row(x, right, pr, node)
                num_extra += 1

        tables.sort()
        ancestors_ts = tables.tree_sequence()
        return ancestors_ts

    def verify(self, samples):
        ts = tsinfer.infer(samples, simplify=False)
        ancestors_ts_1 = self.insert_srb_ancestors(samples, ts)
        ancestors_ts_2 = tsinfer.insert_srb_ancestors(samples, ts)
        t1 = ancestors_ts_1.dump_tables()
        t2 = ancestors_ts_2.dump_tables()
        t1.provenances.clear()
        t2.provenances.clear()
        self.assertEqual(t1, t2)

        tsinfer.check_ancestors_ts(ancestors_ts_1)
        ts2 = tsinfer.match_samples(samples, ancestors_ts_1)
        tsinfer.verify(samples, ts2)

    def test_simple_simulation(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=15, random_seed=2)
        self.verify(tsinfer.SampleData.from_tree_sequence(ts))

    def test_random_data_small_examples(self):
        np.random.seed(4)
        num_random_tests = 10
        for _ in range(num_random_tests):
            G, positions = get_random_data_example(5, 10)
            with tsinfer.SampleData(sequence_length=G.shape[0]) as samples:
                for j in range(G.shape[0]):
                    samples.add_site(positions[j], G[j])
            self.verify(samples)

    def test_random_data_large_example(self):
        np.random.seed(5)
        G, positions = get_random_data_example(15, 100)
        with tsinfer.SampleData(sequence_length=G.shape[0]) as samples:
            for j in range(G.shape[0]):
                samples.add_site(positions[j], G[j])
        self.verify(samples)


class TestAugmentedAncestors(unittest.TestCase):
    """
    Tests for augmenting an ancestors tree sequence with samples.
    """
    def verify_augmented_ancestors(
            self, subset, ancestors_ts, augmented_ancestors, path_compression):

        t1 = ancestors_ts.dump_tables()
        t2 = augmented_ancestors.dump_tables()
        k = len(subset)
        m = len(t1.nodes)
        self.assertTrue(
            np.all(t2.nodes.flags[m: m + k] == tsinfer.NODE_IS_SAMPLE_ANCESTOR))
        self.assertTrue(np.all(t2.nodes.time[m: m + k] == 1))
        for j, node_id in enumerate(subset):
            node = t2.nodes[m + j]
            self.assertEqual(node.flags, tsinfer.NODE_IS_SAMPLE_ANCESTOR)
            self.assertEqual(node.time, 1)
            metadata = json.loads(node.metadata.decode())
            self.assertEqual(node_id, metadata["sample"])

        t2.nodes.truncate(len(t1.nodes))
        t2.nodes.set_columns(
            flags=t2.nodes.flags,
            time=t2.nodes.time - 1,
            metadata=t2.nodes.metadata,
            metadata_offset=t2.nodes.metadata_offset)
        self.assertEqual(t1.nodes, t2.nodes)
        if not path_compression:
            # If we have path compression it's possible that some older edges
            # will be compressed out.
            self.assertGreaterEqual(set(t2.edges), set(t1.edges))
        self.assertEqual(t1.sites, t2.sites)
        t2.mutations.truncate(len(t1.mutations))
        self.assertEqual(t1.mutations, t2.mutations)
        t2.provenances.truncate(len(t1.provenances))
        self.assertEqual(t1.provenances, t2.provenances)
        self.assertEqual(t1.individuals, t2.individuals)
        self.assertEqual(t1.populations, t2.populations)

    def verify_example(self, subset, samples, ancestors, path_compression):
        ancestors_ts = tsinfer.match_ancestors(
            samples, ancestors, path_compression=path_compression)
        augmented_ancestors = tsinfer.augment_ancestors(
            samples, ancestors_ts, subset, path_compression=path_compression)

        self.verify_augmented_ancestors(
            subset, ancestors_ts, augmented_ancestors, path_compression)

        # Run the inference now
        final_ts = tsinfer.match_samples(
            samples, augmented_ancestors, simplify=False)
        t1 = ancestors_ts.dump_tables()
        tables = final_ts.tables
        for j, index in enumerate(subset):
            sample_id = final_ts.samples()[index]
            edges = [e for e in final_ts.edges() if e.child == sample_id]
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0].left, 0)
            self.assertEqual(edges[0].right, final_ts.sequence_length)
            parent = edges[0].parent
            original_node = len(t1.nodes) + j
            self.assertEqual(
                tables.nodes.flags[original_node], tsinfer.NODE_IS_SAMPLE_ANCESTOR)
            # Most of the time the parent is the original node. However, in
            # simple cases it can be somewhere up the tree above it.
            if parent != original_node:
                for tree in final_ts.trees():
                    u = parent
                    while u != msprime.NULL_NODE:
                        siblings = tree.children(u)
                        if original_node in siblings:
                            break
                        u = tree.parent(u)
                    self.assertNotEqual(u, msprime.NULL_NODE)

    def verify(self, samples):
        ancestors = tsinfer.generate_ancestors(samples)
        n = samples.num_samples
        subsets = [
            [0, 1], [n - 2, n - 1],
            [0, n // 2, n - 1],
            range(5),
            range(6),
        ]
        for subset in subsets:
            for path_compression in [True, False]:
                self.verify_example(subset, samples, ancestors, path_compression)

    def test_simple_case(self):
        ts = msprime.simulate(55, mutation_rate=5, random_seed=8, recombination_rate=8)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)

    def test_simulation_with_error(self):
        ts = msprime.simulate(50, mutation_rate=5, random_seed=5, recombination_rate=8)
        ts = eval_util.insert_errors(ts, 0.1, seed=32)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)

    def test_small_random_data(self):
        n = 25
        m = 20
        G, positions = get_random_data_example(n, m, seed=1234)
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for genotypes, position in zip(G, positions):
                sample_data.add_site(position, genotypes)
        self.verify(sample_data)

    def test_large_random_data(self):
        n = 100
        m = 30
        G, positions = get_random_data_example(n, m, seed=1234)
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for genotypes, position in zip(G, positions):
                sample_data.add_site(position, genotypes)
        self.verify(sample_data)


class TestSequentialAugmentedAncestors(TestAugmentedAncestors):
    """
    Test that we can sequentially augment the ancestors.
    """
    def verify_example(self, full_subset, samples, ancestors, path_compression):
        ancestors_ts = tsinfer.match_ancestors(
            samples, ancestors, path_compression=path_compression)
        expected_sample_ancestors = 0
        for j in range(1, len(full_subset)):
            subset = full_subset[:j]
            expected_sample_ancestors += len(subset)
            augmented_ancestors = tsinfer.augment_ancestors(
                samples, ancestors_ts, subset, path_compression=path_compression)
            self.verify_augmented_ancestors(
                subset, ancestors_ts, augmented_ancestors, path_compression)
            # Run the inference now
            final_ts = tsinfer.match_samples(
                samples, augmented_ancestors, simplify=False)

            # Make sure metadata has been preserved in the final ts.
            num_sample_ancestors = 0
            for node in final_ts.nodes():
                if node.flags == tsinfer.NODE_IS_SAMPLE_ANCESTOR:
                    metadata = json.loads(node.metadata.decode())
                    self.assertIn(metadata["sample"], subset)
                    num_sample_ancestors += 1
            self.assertEqual(expected_sample_ancestors, num_sample_ancestors)
            tsinfer.verify(samples, final_ts.simplify())
            ancestors_ts = augmented_ancestors
