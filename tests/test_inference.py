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
    return S


class TestRoundTrip(unittest.TestCase):
    """
    Test that we can round-trip data from a simulation through tsinfer.
    """
    def verify_round_trip(self, ts, rho):
        S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
        for variant in ts.variants():
            S[:, variant.index] = variant.genotypes
        sites = [mut.position for mut in ts.mutations()]
        for algorithm in ["python", "c"]:
            panel = tsinfer.ReferencePanel(
                S, sites, ts.sequence_length, rho=rho, algorithm=algorithm)
            P, mutations = panel.infer_paths(num_workers=1)
            ts_new = panel.convert_records(P, mutations)
            self.assertEqual(ts.num_sites, ts_new.num_sites)
            for m1, m2 in zip(ts.mutations(), ts_new.mutations()):
                self.assertEqual(m1.position, m2.position)
            for v1, v2 in zip(ts.variants(), ts_new.variants()):
                self.assertTrue(np.all(v1.genotypes == v2.genotypes))
            ts_simplified = ts_new.simplify()
            # Check that we get the same variants.
            self.assertEqual(ts.num_sites, ts_simplified.num_sites)
            for v1, v2 in zip(ts.variants(), ts_simplified.variants()):
                self.assertTrue(np.all(v1.genotypes == v2.genotypes))
            for m1, m2 in zip(ts.mutations(), ts_simplified.mutations()):
                self.assertEqual(m1.position, m2.position)

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

    def verify_data_round_trip(self, S, rho, err):
        num_samples, num_sites = S.shape
        sites = np.arange(num_sites)
        for algorithm in ["python", "c"]:
            panel = tsinfer.ReferencePanel(
                S, sites, num_sites, rho=rho, ancestor_error=err, sample_error=err, algorithm=algorithm)
            P, mutations = panel.infer_paths(num_workers=1)
            ts_new = panel.convert_records(P, mutations)
            for variant in ts_new.variants():
                self.assertTrue(np.all(variant.genotypes == S[:, variant.index]))
            self.assertEqual(num_sites, ts_new.num_sites)
            ts_simplified = ts_new.simplify()
            self.assertEqual(num_sites, ts_simplified.num_sites)
            S2 = np.empty(S.shape, np.uint8)
            for j, h in enumerate(ts_simplified.haplotypes()):
                S2[j,:] = np.fromstring(h, np.uint8) - ord('0')
            self.assertTrue(np.all(S2 == S))

    def test_random_data_high_recombination(self):
        S = get_random_data_example(20, 30)
        # Force recombination to do all the matching.
        self.verify_data_round_trip(S, 1, 0)

    def test_random_data_no_recombination(self):
        np.random.seed(4)
        num_random_tests = 100
        for _ in range(num_random_tests):
            S = get_random_data_example(5, 10)
            self.verify_data_round_trip(S, 1e-8, 1e-3)

class TestThreads(unittest.TestCase):

    def test_equivalance(self):
        rho = 2
        ts = msprime.simulate(5, mutation_rate=2, recombination_rate=rho, random_seed=2)
        S = np.zeros((ts.sample_size, ts.num_sites), dtype="u1")
        for variant in ts.variants():
            S[:, variant.index] = variant.genotypes
        sites = [mut.position for mut in ts.mutations()]
        panel = tsinfer.ReferencePanel(S, sites, ts.sequence_length, rho=rho)
        P1, mutations1 = panel.infer_paths(num_workers=1)
        P2, mutations2 = panel.infer_paths(num_workers=4)
        self.assertTrue(np.all(P1 == P2))
        self.assertTrue(np.all(mutations1 == mutations2))
