"""
Tests for the inference code.
"""
import unittest

import numpy as np
import msprime

import tsinfer
import _tsinfer


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


class TestAncestorStorage(unittest.TestCase):
    """
    Tests where we build the set of ancestors using the tree sequential update
    process and verify that we get the correct set of ancestors back from
    the resulting tree sequence.
    """

    # TODO clean up this verification method and figure out a better API
    # for specifying the classes to use.

    def verify_ancestor_storage(self, ts, method="C"):
        n = ts.sample_size
        num_sites = ts.num_sites

        samples = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
        for variant in ts.variants():
            samples[:, variant.index] = variant.genotypes
        positions = np.array([site.position for site in ts.sites()])

        recombination_rate = 1e-8
        if method == "C":
            ancestor_builder = _tsinfer.AncestorBuilder(samples, positions)
            ts_builder = _tsinfer.TreeSequenceBuilder(num_sites, 10**6, 10**6)
            matcher = _tsinfer.AncestorMatcher(ts_builder, recombination_rate)
        else:
            ancestor_builder = tsinfer.AncestorBuilder(samples, positions)
            ts_builder = tsinfer.TreeSequenceBuilder(num_sites)
            matcher = tsinfer.AncestorMatcher(ts_builder, recombination_rate)

        frequency_classes = ancestor_builder.get_frequency_classes()

        num_ancestors = 1
        for _, ancestor_focal_sites in frequency_classes:
            num_ancestors += len(ancestor_focal_sites)
        # For checking the output.
        A = np.zeros((num_ancestors, num_sites), dtype=np.int8)
        ancestor_id_map = {0: 0}
        ancestor_id = 1

        # TODO this time is out by 1 I think.
        root_time = frequency_classes[0][0] + 1
        ts_builder.update(1, root_time, [], [], [], [], [], [])
        a = np.zeros(num_sites, dtype=np.int8)

        for age, ancestor_focal_sites in frequency_classes:
            e_left = []
            e_right = []
            e_parent = []
            e_child = []
            s_site = []
            s_node = []
            node = ts_builder.num_nodes
            for focal_sites in ancestor_focal_sites:
                ancestor_builder.make_ancestor(focal_sites, a)
                A[ancestor_id] = a
                ancestor_id += 1
                ancestor_id_map[ancestor_id] = node
                for s in focal_sites:
                    assert a[s] == 1
                    a[s] = 0
                    s_site.append(s)
                    s_node.append(node)
                # When we update this API we should pass in arrays for left, right and

                # parent. There's no point in passing child, since we already know what
                # it is. We don't need to pass the 'node' parameter here then.
                edges = matcher.find_path(node, a)
                for left, right, parent, child in zip(*edges):
                    e_left.append(left)
                    e_right.append(right)
                    e_parent.append(parent)
                    e_child.append(child)
                node += 1
            ts_builder.update(
                len(ancestor_focal_sites), age,
                e_left, e_right, e_parent, e_child,
                s_site, s_node)

        ts = tsinfer.finalise(ts_builder, all_ancestors=True)

        B = np.zeros((ts.sample_size, ts.num_sites), dtype=np.int8)
        for v in ts.variants():
            B[:, v.index] = v.genotypes

        for ancestor_id in range(num_ancestors):
            node_id = ancestor_id
            assert np.array_equal(A[ancestor_id], B[node_id])

    def test_small_case(self):
        ts = msprime.simulate(
            20, length=10, recombination_rate=1, mutation_rate=0.1, random_seed=1)
        assert ts.num_sites < 50
        self.verify_ancestor_storage(ts)
        self.verify_ancestor_storage(ts, method="P")
