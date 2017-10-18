"""
Tests for the inference code.
"""
import unittest

import numpy as np
import msprime

import tsinfer


def get_random_data_example(num_samples, num_sites):
    S = np.random.randint(2, size=(num_samples, num_sites)).astype(np.uint8)
    # Weed out any invariant sites
    for j in range(num_sites):
        if np.sum(S[:, j]) == 0:
            S[0, j] = 1
        elif np.sum(S[:, j]) == num_samples:
            S[0, j] = 0
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
    def verify_data_round_trip(
            self, samples, positions, sequence_length=None, recombination_rate=1e-9,
            error_rate=None):
        if sequence_length is None:
            sequence_length = positions[-1] + 1
        for method in ["python", "c"]:
            ts = tsinfer.infer(
                samples=samples, positions=positions, sequence_length=sequence_length,
                recombination_rate=recombination_rate, error_rate=error_rate,
                method=method)
            self.assertEqual(ts.sequence_length, sequence_length)
            self.assertEqual(ts.num_sites, len(positions))
            for v in ts.variants():
                self.assertEqual(v.position, positions[v.index])
                self.assertTrue(np.array_equal(samples[:, v.index], v.genotypes))

    def verify_round_trip(self, ts, rho):
        S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
        for variant in ts.variants():
            S[:, variant.index] = variant.genotypes
        positions = [mut.position for mut in ts.mutations()]
        self.verify_data_round_trip(S, positions, ts.sequence_length, 1e-9)

    def test_simple_example(self):
        rho = 2
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=rho, random_seed=1)
        self.assertGreater(ts.num_sites, 0)
        self.verify_round_trip(ts, rho)

    def test_single_locus(self):
        ts = msprime.simulate(5, mutation_rate=1, recombination_rate=0, random_seed=2)
        self.assertGreater(ts.num_sites, 0)
        self.verify_round_trip(ts, 1e-9)

    def test_single_locus_two_samples(self):
        ts = msprime.simulate(2, mutation_rate=1, recombination_rate=0, random_seed=3)
        self.assertGreater(ts.num_sites, 0)
        self.verify_round_trip(ts, 1e-9)

    def test_random_data_high_recombination(self):
        S, positions = get_random_data_example(20, 30)
        # Force recombination to do all the matching.
        self.verify_data_round_trip(S, positions, recombination_rate=1, error_rate=0)

    @unittest.skip("Error handling not supported")
    def test_random_data_no_recombination(self):
        np.random.seed(4)
        num_random_tests = 100
        for _ in range(num_random_tests):
            S, positions = get_random_data_example(5, 10)
            self.verify_data_round_trip(S, positions, recombination_rate=0, error_rate=1)


class TestThreads(TsinferTestCase):

    def test_equivalance(self):
        rho = 2
        ts = msprime.simulate(5, mutation_rate=2, recombination_rate=rho, random_seed=2)
        S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
        for variant in ts.variants():
            S[:, variant.index] = variant.genotypes
        positions = [site.position for site in ts.sites()]
        ts1 = tsinfer.infer(
            samples=S, positions=positions, sequence_length=ts.sequence_length,
            recombination_rate=1e-9, num_threads=1)
        ts2 = tsinfer.infer(
            samples=S, positions=positions, sequence_length=ts.sequence_length,
            recombination_rate=1e-9, num_threads=5)
        self.assertTreeSequencesEqual(ts1, ts2)


class TestAncestorStorage(unittest.TestCase):
    """
    Tests where we build the set of ancestors using the tree sequential update
    process and verify that we get the correct set of ancestors back from
    the resulting tree sequence.
    """

    # TODO clean up this verification method and figure out a better API
    # for specifying the classes to use.

    def verify_ancestor_storage(
            self, ts, method="C", resolve_polytomies=False,
            resolve_shared_recombinations=False):

        samples = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
        for variant in ts.variants():
            samples[:, variant.index] = variant.genotypes
        positions = np.array([site.position for site in ts.sites()])
        manager = tsinfer.InferenceManager(
            samples, positions, ts.sequence_length, 1e-8,
            method=method, num_threads=1,
            resolve_polytomies=resolve_polytomies,
            resolve_shared_recombinations=resolve_shared_recombinations)
        manager.initialise()
        manager.process_ancestors()
        ts_new = manager.get_tree_sequence()

        self.assertEqual(ts_new.num_samples, manager.num_ancestors)
        self.assertEqual(ts_new.num_sites, manager.num_sites)
        A = manager.ancestors()
        B = np.zeros((manager.num_ancestors, manager.num_sites), dtype=np.int8)
        for v in ts_new.variants():
            B[:, v.index] = v.genotypes
        self.assertTrue(np.array_equal(A, B))

    def verify_small_case(
            self, resolve_polytomies=False, resolve_shared_recombinations=False):
        ts = msprime.simulate(
            20, length=10, recombination_rate=1, mutation_rate=0.1, random_seed=1)
        assert ts.num_sites < 50
        for method in ["C", "Python"]:
            self.verify_ancestor_storage(
                ts, method=method, resolve_polytomies=resolve_polytomies,
                resolve_shared_recombinations=resolve_shared_recombinations)

    def test_small_case(self):
        self.verify_small_case(False, False)

    def test_small_case_resolve_polytomies(self):
        self.verify_small_case(True, False)

    def test_small_case_resolve_shared_recom(self):
        self.verify_small_case(False, True)

    def test_small_case_resolve_all(self):
        self.verify_small_case(True, True)
