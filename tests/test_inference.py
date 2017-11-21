"""
Tests for the inference code.
"""
import unittest

import numpy as np
import msprime

import tsinfer


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
    def verify_data_round_trip(
            self, genotypes, positions, sequence_length=None, recombination_rate=1e-9,
            sample_error=0):
        if sequence_length is None:
            sequence_length = positions[-1] + 1
        # import daiquiri
        # daiquiri.setup(level="DEBUG")
        for method in ["python", "C"]:
            ts = tsinfer.infer(
                genotypes=genotypes, positions=positions, sequence_length=sequence_length,
                recombination_rate=recombination_rate, sample_error=sample_error,
                method=method)
            self.assertEqual(ts.sequence_length, sequence_length)
            self.assertEqual(ts.num_sites, len(positions))
            for v in ts.variants():
                self.assertEqual(v.position, positions[v.index])
                self.assertTrue(np.array_equal(genotypes[v.index], v.genotypes))

    def verify_round_trip(self, ts, rho):
        positions = [site.position for site in ts.sites()]
        self.verify_data_round_trip(
            ts.genotype_matrix(), positions, ts.sequence_length, 1e-9)

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
        G, positions = get_random_data_example(20, 30)
        # Force recombination to do all the matching.
        self.verify_data_round_trip(G, positions, recombination_rate=1)

    def test_random_data_invariant_sites(self):
        G, positions = get_random_data_example(24, 35)
        # Set some sites to be invariant
        G[10,:] = 1
        G[15,:] = 0
        G[20,:] = 1
        G[22,:] = 0
        # Force recombination to do all the matching.
        self.verify_data_round_trip(G, positions, recombination_rate=1)

    @unittest.skip("error broken")
    def test_random_data_no_recombination(self):
        np.random.seed(4)
        num_random_tests = 10
        for _ in range(num_random_tests):
            S, positions = get_random_data_example(5, 10)
            self.verify_data_round_trip(
                S, positions, recombination_rate=1e-10, sample_error=0.1)


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
                recombination_rate=0.5, sample_error=0, method=method)
            self.assertEqual(ts.num_sites, num_sites)
            self.assertEqual(ts.num_mutations, num_sites)
            for site in ts.sites():
                self.assertEqual(site.ancestral_state, "0")
                self.assertEqual(len(site.mutations), 1)
                mutation = site.mutations[0]
                self.assertEqual(mutation.derived_state, "1")
                self.assertEqual(mutation.parent, -1)

    @unittest.skip("error broken")
    def test_error(self):
        num_sites = 20
        S, positions = get_random_data_example(5, num_sites)
        for method in ["python", "c"]:
            ts = tsinfer.infer(
                samples=S, positions=positions, sequence_length=num_sites,
                recombination_rate=1e-9, sample_error=0.1, method=method)
            self.assertEqual(ts.num_sites, num_sites)
            self.assertGreater(ts.num_mutations, num_sites)
            back_mutation = False
            recurrent_mutation = False
            for site in ts.sites():
                self.assertEqual(site.ancestral_state, "0")
                for mutation in site.mutations:
                    if mutation.derived_state == "0":
                        back_mutation = True
                        self.assertEqual(mutation.parent, site.mutations[0].id)
                    else:
                        self.assertEqual(mutation.parent, -1)
                        if mutation != site.mutations[0]:
                            recurrent_mutation = True
            self.assertTrue(back_mutation)
            self.assertTrue(recurrent_mutation)


class TestThreads(TsinferTestCase):

    def test_equivalance(self):
        rho = 2
        ts = msprime.simulate(5, mutation_rate=2, recombination_rate=rho, random_seed=2)
        G = ts.genotype_matrix()
        positions = [site.position for site in ts.sites()]
        ts1 = tsinfer.infer(
            genotypes=G, positions=positions, sequence_length=ts.sequence_length,
            recombination_rate=1e-9, num_threads=1)
        ts2 = tsinfer.infer(
            genotypes=G, positions=positions, sequence_length=ts.sequence_length,
            recombination_rate=1e-9, num_threads=5)
        self.assertTreeSequencesEqual(ts1, ts2)


@unittest.skip("Test broken")
class TestAncestorStorage(unittest.TestCase):
    """
    Tests where we build the set of ancestors using the tree sequential update
    process and verify that we get the correct set of ancestors back from
    the resulting tree sequence.
    """

    # TODO clean up this verification method and figure out a better API
    # for specifying the classes to use.

    def verify_ancestor_storage( self, ts, method="C"):
        samples = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
        for variant in ts.variants():
            samples[:, variant.index] = variant.genotypes
        positions = np.array([site.position for site in ts.sites()])
        recombination_rate = np.zeros_like(positions) + 1e-8
        manager = tsinfer.InferenceManager(
            samples, positions, ts.sequence_length, recombination_rate,
            method=method, num_threads=1)
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

    def test_small_case(self):
        ts = msprime.simulate(
            20, length=10, recombination_rate=1, mutation_rate=0.1, random_seed=1)
        assert ts.num_sites < 50
        for method in ["C", "Python"]:
            self.verify_ancestor_storage(ts, method=method)


class TestPhredEncoding(unittest.TestCase):
    """
    Test cases for Phred encoding.
    """
    def test_zero_proba(self):
        q = tsinfer.proba_to_phred(0)
        self.assertEqual(q, tsinfer.PHRED_MAX)
        p = tsinfer.phred_to_proba(q)
        self.assertEqual(p, 0)

    def test_tiny_proba(self):
        q = tsinfer.proba_to_phred(1e-200)
        self.assertEqual(q, tsinfer.PHRED_MAX)
        p = tsinfer.phred_to_proba(q)
        self.assertEqual(p, 0)

    def test_one_proba(self):
        q = tsinfer.proba_to_phred(1)
        self.assertEqual(q, 0)
        p = tsinfer.phred_to_proba(q)
        self.assertEqual(p, 1)

    def test_exact_values(self):
        for k in range(1, 20):
            p = 10**(-k)
            q = tsinfer.proba_to_phred(p, min_value=1e-21)
            self.assertEqual(q, 10 * k)
            self.assertEqual(tsinfer.phred_to_proba(q), p)

    def test_numpy_array(self):
        p = [0.1, 0.01, 0.001, 0.0001]
        q = tsinfer.proba_to_phred(p)
        self.assertTrue(np.array_equal(q, np.array([10, 20, 30, 40], dtype=np.uint8)))
        self.assertTrue(np.array_equal(p, tsinfer.phred_to_proba(q)))

    def test_non_base_10(self):
        for p in [0.125, 0.333, 0.99]:
            q = tsinfer.proba_to_phred(p)
            self.assertAlmostEqual(p, float(tsinfer.phred_to_proba(q)), 1)

    def test_p_greater_than_one(self):
        for p in [1.0001, 10000, 1e200]:
            self.assertRaises(ValueError, tsinfer.proba_to_phred, p)
            self.assertRaises(ValueError, tsinfer.proba_to_phred, [0.1, p])
