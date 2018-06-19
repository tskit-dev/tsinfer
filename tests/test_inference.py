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

import numpy as np
import msprime

import tsinfer
import tsinfer.formats as formats


def get_random_data_example(num_samples, num_sites):
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
        sample_data = formats.SampleData(sequence_length=sequence_length)
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
        sample_data = formats.SampleData.from_tree_sequence(ts)
        ts1 = tsinfer.infer(sample_data, num_threads=1)
        ts2 = tsinfer.infer(sample_data, num_threads=5)
        self.assertTreeSequencesEqual(ts1, ts2)


class TestAncestorGeneratorsEquivalant(unittest.TestCase):
    """
    Tests for the ancestor generation process.
    """

    def verify_ancestor_generator(self, genotypes):
        m, n = genotypes.shape
        with tsinfer.SampleData() as sample_data:
            for j in range(m):
                sample_data.add_site(j, genotypes[j])

        adc = tsinfer.generate_ancestors(sample_data, engine=tsinfer.C_ENGINE)
        adp = tsinfer.generate_ancestors(sample_data, engine=tsinfer.PY_ENGINE)
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
        G, _ = get_random_data_example(20, 50)
        # G, _ = get_random_data_example(20, 10)
        self.verify_ancestor_generator(G)


class TestGeneratedAncestors(unittest.TestCase):
    """
    Ensures we work correctly with the ancestors recovered from the
    simulations.
    """
    def verify_inserted_ancestors(self, ts):
        # Verifies that we can round-trip the specified tree sequence
        # using the generated ancestors. NOTE: this must be an SMC
        # consistent tree sequence!
        with formats.SampleData(sequence_length=ts.sequence_length) as sample_data:
            for v in ts.variants():
                sample_data.add_site(v.position, v.genotypes, v.alleles)

        ancestor_data = formats.AncestorData(sample_data)
        tsinfer.build_simulated_ancestors(sample_data, ancestor_data, ts)
        ancestor_data.finalise()

        A = np.zeros(
            (ancestor_data.num_sites, ancestor_data.num_ancestors), dtype=np.uint8)
        start = ancestor_data.start[:]
        end = ancestor_data.end[:]
        ancestors = ancestor_data.ancestor[:]
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
        ancestors = ancestor_data.ancestor[:]
        inference_sites = sample_data.sites_inference[:]
        sample_genotypes = sample_data.sites_genotypes[:][inference_sites == 1, :]
        start = ancestor_data.start[:]
        end = ancestor_data.end[:]
        time = ancestor_data.time[:]
        focal_sites = ancestor_data.focal_sites[:]

        self.assertEqual(ancestor_data.num_ancestors, ancestors.shape[0])
        self.assertEqual(ancestor_data.num_sites, sample_data.num_inference_sites)
        self.assertEqual(ancestor_data.num_ancestors, time.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, start.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, end.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, focal_sites.shape[0])
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
        start = ancestor_data.start[:]
        end = ancestor_data.end[:]
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
        if self.path_compression_enabled:
            # With path compression we have to check the tree metrics.
            p_breakpoints, distance = tsinfer.compare(ts, tsp)
            self.assertTrue(np.all(distance == 0))
            c_breakpoints, distance = tsinfer.compare(ts, tsc)
            self.assertTrue(np.all(distance == 0))
            self.assertTrue(np.all(p_breakpoints == c_breakpoints))
        else:
            # Without path compression we're guaranteed to return precisely the
            # same tree sequences.
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
            start=0, end=3, focal_sites=[2], time=3, haplotype=[0, 0, 1, -1, -1, -1])
        ancestor_data.add_ancestor(  # ID 3
            start=3, end=6, focal_sites=[4], time=2, haplotype=[-1, -1, -1, 0, 1, 0])
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
            start=0, end=3, focal_sites=[2], time=3, haplotype=[0, 0, 1, 0, 0, 0, 0])
        ancestor_data.add_ancestor(  # ID 3
            start=3, end=7, focal_sites=[4, 6], time=2,
            haplotype=[-1, -1, -1, 0, 1, 0, 1])
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
            haplotype=[0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1])
        ancestor_data.add_ancestor(  # ID 3
            start=4, end=12, focal_sites=[], time=5,
            haplotype=[-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
        ancestor_data.add_ancestor(  # ID 4
            start=8, end=12, focal_sites=[9, 11], time=4,
            haplotype=[-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 0, 1])
        ancestor_data.add_ancestor(  # ID 5
            start=4, end=8, focal_sites=[5, 7], time=3,
            haplotype=[-1, -1, -1, -1, 0, 1, 0, 1, -1, -1, -1, -1])
        ancestor_data.add_ancestor(  # ID 6
            start=0, end=4, focal_sites=[1, 3], time=2,
            haplotype=[0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1])
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
        return formats.SampleData.from_tree_sequence(ts)

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
        self.assertRaises(ValueError, tsinfer.match_samples, sample_data, sim)

    def test_different_ancestors_ts_match_samples(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)

        sim = msprime.simulate(sample_size=6, random_seed=2, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        self.assertRaises(ValueError, tsinfer.match_samples, sample_data, ancestors_ts)


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

        # When simplify is true the samples should be zero to N - n
        # up to n
        ts2 = tsinfer.infer(sd, simplify=False)
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
