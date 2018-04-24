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

import numpy as np
import msprime

import tsinfer
import tsinfer.formats as formats


def get_random_data_example(num_samples, num_sites, remove_invariant_sites=True):
    S = np.random.randint(2, size=(num_sites, num_samples)).astype(np.uint8)
    if remove_invariant_sites:
        for j in range(num_sites):
            if np.sum(S[j, :]) == 0:
                S[j, 0] = 1
            elif np.sum(S[j, :]) == num_samples:
                S[j, 0] = 0
    return S, np.arange(num_sites)


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
        # import daiquiri
        # daiquiri.setup(level="DEBUG")
        for method in ["python", "C"]:
            ts = tsinfer.infer(
                genotypes=genotypes, positions=positions,
                sequence_length=sequence_length, method=method)
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

    def test_random_data_invariant_sites_ancestral_state(self):
        G, positions = get_random_data_example(24, 35)
        # Set some sites to be invariant for the ancestral state
        G[10, :] = 0
        G[15, :] = 0
        G[20, :] = 0
        G[22, :] = 0
        self.verify_data_round_trip(G, positions)

    @unittest.skip("invariant site state")
    def test_random_data_invariant_sites(self):
        G, positions = get_random_data_example(39, 25)
        # Set some sites to be invariant
        G[10, :] = 1
        G[15, :] = 0
        G[20, :] = 1
        G[22, :] = 0
        self.verify_data_round_trip(G, positions)

    def test_random_data_large_example(self):
        G, positions = get_random_data_example(20, 30)
        self.verify_data_round_trip(G, positions)

    def test_random_data_small_examples(self):
        np.random.seed(4)
        num_random_tests = 10
        for _ in range(num_random_tests):
            S, positions = get_random_data_example(5, 10)
            self.verify_data_round_trip(S, positions)


class TestMutationProperties(unittest.TestCase):
    """
    Tests to ensure that mutations have the properties that we expect.
    """

    def test_no_error(self):
        num_sites = 10
        G, positions = get_random_data_example(5, num_sites)
        for method in ["python", "c"]:
            ts = tsinfer.infer(
                genotypes=G, positions=positions, sequence_length=num_sites,
                method=method)
            self.assertEqual(ts.num_sites, num_sites)
            self.assertEqual(ts.num_mutations, num_sites)
            for site in ts.sites():
                self.assertEqual(site.ancestral_state, "0")
                self.assertEqual(len(site.mutations), 1)
                mutation = site.mutations[0]
                self.assertEqual(mutation.derived_state, "1")
                self.assertEqual(mutation.parent, -1)


class TestThreads(TsinferTestCase):

    def test_equivalance(self):
        rho = 2
        ts = msprime.simulate(5, mutation_rate=2, recombination_rate=rho, random_seed=2)
        G = ts.genotype_matrix()
        positions = [site.position for site in ts.sites()]
        ts1 = tsinfer.infer(
            genotypes=G, positions=positions, sequence_length=ts.sequence_length,
            num_threads=1)
        ts2 = tsinfer.infer(
            genotypes=G, positions=positions, sequence_length=ts.sequence_length,
            num_threads=5)
        self.assertTreeSequencesEqual(ts1, ts2)


class TestAncestorGeneratorsEquivalant(unittest.TestCase):
    """
    Tests for the ancestor generation process.
    """

    def verify_ancestor_generator(self, genotypes):
        m, n = genotypes.shape
        sample_data = tsinfer.SampleData.initialise(n, m, compressor=None)
        for j in range(m):
            sample_data.add_variant(j, ["0", "1"], genotypes[j])
        sample_data.finalise()

        adc = tsinfer.AncestorData.initialise(sample_data, compressor=None)
        tsinfer.build_ancestors(sample_data, adc, method="C")
        adc.finalise()

        adp = tsinfer.AncestorData.initialise(sample_data, compressor=None)
        tsinfer.build_ancestors(sample_data, adp, method="P")
        adp.finalise()

        # np.set_printoptions(linewidth=20000)
        # np.set_printoptions(threshold=20000000)
        # A = adp.genotypes[:]
        # B = adc.genotypes[:]
        # print(A)
        # print(B)
        # print(np.all(A == B))
        # print((A == B).astype(np.int))

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
        sample_data = formats.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length,
            compressor=None)
        for v in ts.variants():
            sample_data.add_variant(v.position, v.alleles, v.genotypes)
        sample_data.finalise()

        ancestor_data = formats.AncestorData.initialise(sample_data, compressor=None)
        tsinfer.build_simulated_ancestors(sample_data, ancestor_data, ts)
        ancestor_data.finalise()

        A = np.zeros(
            (ancestor_data.num_sites, ancestor_data.num_ancestors), dtype=np.uint8)
        start = ancestor_data.start[:]
        end = ancestor_data.end[:]
        ancestors = ancestor_data.ancestor[:]
        for j in range(ancestor_data.num_ancestors):
            A[start[j]: end[j], j] = ancestors[j]
        for method in ["P", "C"]:
            ancestors_ts = tsinfer.match_ancestors(
                sample_data, ancestor_data, method=method)
            self.assertEqual(ancestor_data.num_sites, ancestors_ts.num_sites)
            self.assertEqual(ancestor_data.num_ancestors, ancestors_ts.num_samples)
            self.assertTrue(np.array_equal(ancestors_ts.genotype_matrix(), A))
            inferred_ts = tsinfer.match_samples(
                sample_data, ancestors_ts, method=method)
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
    Tests for the build_ancestors function.
    """
    def get_simulated_example(self, ts):
        sample_data = tsinfer.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length)
        for variant in ts.variants():
            sample_data.add_variant(
                variant.site.position, variant.alleles, variant.genotypes)
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData.initialise(sample_data)
        tsinfer.build_ancestors(sample_data, ancestor_data)
        ancestor_data.finalise()
        return sample_data, ancestor_data

    def verify_ancestors(self, sample_data, ancestor_data):
        ancestors = ancestor_data.ancestor[:]
        sample_genotypes = sample_data.genotypes[:]
        start = ancestor_data.start[:]
        end = ancestor_data.end[:]
        time = ancestor_data.time[:]
        focal_sites = ancestor_data.focal_sites[:]

        self.assertEqual(ancestor_data.num_ancestors, ancestors.shape[0])
        self.assertEqual(ancestor_data.num_sites, sample_data.num_variant_sites)
        self.assertEqual(ancestor_data.num_ancestors, time.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, start.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, end.shape[0])
        self.assertEqual(ancestor_data.num_ancestors, focal_sites.shape[0])
        # The first ancestor must be all zeros.
        self.assertEqual(start[0], 0)
        self.assertEqual(end[0], ancestor_data.num_sites)
        self.assertEqual(time[0], 2 + max(
            np.sum(genotypes) for genotypes in sample_genotypes))
        self.assertEqual(list(focal_sites[0]), [])
        self.assertTrue(np.all(ancestors[0] == 0))

        used_sites = []
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
                # The time value should be equal to the original frequency of the
                # site in question.
                freq = np.sum(sample_genotypes[site])
                self.assertEqual(freq, time[j])
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
        sample_data = tsinfer.SampleData.initialise(num_samples=n, sequence_length=m)
        for genotypes, position in zip(G, positions):
            sample_data.add_variant(position, ["0", "1"], genotypes)
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData.initialise(sample_data)
        tsinfer.build_ancestors(sample_data, ancestor_data)
        ancestor_data.finalise()
        self.verify_ancestors(sample_data, ancestor_data)


class AlgorithmsExactlyEqualMixin(object):
    """
    For small example tree sequences, check that the Python and C implementations
    return precisely the same tree sequence when fed with perfect mutations.
    """
    def infer(self, ts, method, path_compression=False):
        sample_data = tsinfer.SampleData.initialise(
            num_samples=ts.num_samples, sequence_length=ts.sequence_length,
            compressor=None)
        for v in ts.variants():
            sample_data.add_variant(v.site.position, v.alleles, v.genotypes)
        sample_data.finalise()

        ancestor_data = tsinfer.AncestorData.initialise(sample_data, compressor=None)
        tsinfer.build_simulated_ancestors(sample_data, ancestor_data, ts)
        ancestor_data.finalise()
        ancestors_ts = tsinfer.match_ancestors(
            sample_data, ancestor_data, method=method,
            path_compression=path_compression, extended_checks=True)
        inferred_ts = tsinfer.match_samples(
            sample_data, ancestors_ts, method=method, simplify=True,
            path_compression=path_compression, extended_checks=True)
        return inferred_ts

    def verify(self, ts):
        tsp = self.infer(ts, "P", path_compression=self.path_compression_enabled)
        tsc = self.infer(ts, "C", path_compression=self.path_compression_enabled)
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

        for method in ["C", "P"]:
            ts = tsinfer.match_ancestors(sample_data, ancestor_data, method=method)
            self.assertEqual(
                sorted(expected_edges, key=key), sorted(ts.edges(), key=key))

    def test_easy_case(self):
        num_sites = 6
        sample_data = tsinfer.SampleData.initialise(3, num_sites)
        for j in range(num_sites):
            sample_data.add_variant(j, ["0", "1"], [0, 1, 1])
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData.initialise(sample_data)

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
        sample_data = tsinfer.SampleData.initialise(3, num_sites)
        for j in range(num_sites):
            sample_data.add_variant(j, ["0", "1"], [0, 1, 1])
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData.initialise(sample_data)

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
        sample_data = tsinfer.SampleData.initialise(3, num_sites)
        for j in range(num_sites):
            sample_data.add_variant(j, ["0", "1"], [0, 1, 1])
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData.initialise(sample_data)

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
