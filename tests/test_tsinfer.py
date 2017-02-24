"""
Tests for the tsinfer code.
"""
import unittest

import numpy as np

import msprime
import tsinfer


class TestRoundTrip(unittest.TestCase):
    """
    Test that we can round-trip data from a simulation through tsinfer.
    """
    def verify_round_trip(self, ts, rho):
        S = np.zeros((ts.sample_size, ts.num_mutations), dtype="u1")
        for variant in ts.variants():
            S[:,variant.index] = variant.genotypes
        sites = [mut.position for mut in ts.mutations()]
        panel = tsinfer.ReferencePanel(S, sites, ts.sequence_length)
        P = panel.infer_paths(rho, num_workers=1)
        ts_new = panel.convert_records(P)
        self.assertEqual(ts.num_mutations, ts_new.num_mutations)
        for m1, m2 in zip(ts.mutations(), ts_new.mutations()):
            self.assertEqual(m1.position, m2.position)
        for v1, v2 in zip(ts.variants(), ts_new.variants()):
            if not np.all(v1.genotypes == v2.genotypes):
                print("HERE")
                print(v1)
                print(v2)
            self.assertTrue(np.all(v1.genotypes == v2.genotypes))
        ts_simplified = ts_new.simplify()
        # Check that we get the same variants.
        self.assertEqual(ts.num_mutations, ts_simplified.num_mutations)
        for v1, v2 in zip(ts.variants(), ts_simplified.variants()):
            self.assertTrue(np.all(v1.genotypes == v2.genotypes))
        for m1, m2 in zip(ts.mutations(), ts_simplified.mutations()):
            self.assertEqual(m1.position, m2.position)

    def test_simple_example(self):
        rho = 2
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=rho, random_seed=1)
        self.assertGreater(ts.num_mutations, 0)
        self.verify_round_trip(ts, rho)

    def test_single_locus(self):
        ts = msprime.simulate(5, mutation_rate=1, recombination_rate=0, random_seed=2)
        self.assertGreater(ts.num_mutations, 0)
        self.verify_round_trip(ts, 0)

    def test_single_locus_two_samples(self):
        ts = msprime.simulate(2, mutation_rate=1, recombination_rate=0, random_seed=3)
        self.assertGreater(ts.num_mutations, 0)
        self.verify_round_trip(ts, 0)
