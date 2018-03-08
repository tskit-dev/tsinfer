"""
Test cases for the evaluation code.
"""

import unittest

import msprime
import numpy as np

import tsinfer


def get_smc_simulation(n, L=1, recombination_rate=0, seed=1):
    """
    Returns an smc simulation for a sample of size n, with sequence length L
    and recombination at the specified rate.
    """
    return msprime.simulate(
        n, length=L, recombination_rate=recombination_rate, random_seed=seed,
        model="smc_prime")


class TestKCMetric(unittest.TestCase):
    """
    Tests on the KC metric distances.
    """
    def test_same_tree_zero_distance(self):
        for n in range(2, 10):
            for seed in range(1, 10):
                ts = msprime.simulate(n, random_seed=seed)
                tree = ts.first()
                self.assertEqual(tsinfer.kc_distance(tree, tree), 0)
                ts = msprime.simulate(n, random_seed=seed)
                tree2 = ts.first()
                self.assertEqual(tsinfer.kc_distance(tree, tree2), 0)

    def test_sample_2_zero_distance(self):
        # All trees with 2 leaves must be equal distance from each other.
        for seed in range(1, 10):
            tree1 = msprime.simulate(2, random_seed=seed).first()
            tree2 = msprime.simulate(2, random_seed=seed + 1).first()
            self.assertEqual(tsinfer.kc_distance(tree1, tree2), 0)

    def test_different_samples_error(self):
        tree1 = msprime.simulate(10, random_seed=1).first()
        tree2 = msprime.simulate(2, random_seed=1).first()
        self.assertRaises(ValueError, tsinfer.kc_distance, tree1, tree2)

    # TODO add more tests checking actual examples.


class TestTreeSequenceCompare(unittest.TestCase):
    """
    Tests of the method to compare to tree sequences.
    """
    def test_same_ts(self):
        n = 15
        for seed in range(1, 10):
            ts = msprime.simulate(n, recombination_rate=10, random_seed=seed)
            self.assertGreater(ts.num_trees, 1)
            bp, distance = tsinfer.compare(ts, ts)
            self.assertEqual(list(bp), list(ts.breakpoints()))
            self.assertEqual(distance.shape, (bp.shape[0] - 1,))
            self.assertTrue(np.all(distance == 0))

    def test_single_tree(self):
        n = 15
        for seed in range(1, 10):
            ts1 = msprime.simulate(n, random_seed=seed)
            ts2 = msprime.simulate(n, random_seed=seed + 1)
            bp, distance = tsinfer.compare(ts1, ts2)
            self.assertEqual(list(bp), [0, 1])
            self.assertEqual(distance.shape, (1,))

    def test_single_tree_many_trees(self):
        n = 5
        for seed in range(1, 10):
            ts1 = msprime.simulate(n, recombination_rate=5, random_seed=seed)
            ts2 = msprime.simulate(n, random_seed=seed + 1)
            self.assertGreater(ts1.num_trees, 1)
            bp, distance = tsinfer.compare(ts1, ts2)
            self.assertEqual(list(bp), list(ts1.breakpoints()))
            self.assertEqual(distance.shape, (ts1.num_trees,))

    def test_single_many_trees(self):
        n = 5
        for seed in range(1, 10):
            ts1 = msprime.simulate(n, recombination_rate=5, random_seed=seed)
            ts2 = msprime.simulate(n, recombination_rate=5, random_seed=seed + 1)
            self.assertGreater(ts1.num_trees, 1)
            self.assertGreater(ts2.num_trees, 1)
            bp, distance = tsinfer.compare(ts1, ts2)
            breakpoints = set(ts1.breakpoints()) | set(ts2.breakpoints())
            self.assertEqual(list(bp), sorted(breakpoints))
            self.assertEqual(distance.shape, (len(breakpoints) - 1,))
    # TODO add some examples testing for specific instances.


class TestTreePairs(unittest.TestCase):
    """
    Tests of the method to compare to tree sequences.
    """
    def test_same_ts(self):
        n = 15
        ts = msprime.simulate(n, recombination_rate=10, random_seed=10)
        self.assertGreater(ts.num_trees, 1)
        count = 0
        for (left, right), tree1, tree2 in tsinfer.tree_pairs(ts, ts):
            self.assertEqual((left, right), tree1.interval)
            self.assertEqual(tree1.interval, tree2.interval)
            self.assertEqual(tree1.parent_dict, tree2.parent_dict)
            count += 1
        self.assertEqual(count, ts.num_trees)

    def test_single_tree(self):
        n = 10
        ts1 = msprime.simulate(n, random_seed=10)
        ts2 = msprime.simulate(n, random_seed=10)
        self.assertEqual(ts1.num_trees, 1)
        self.assertEqual(ts2.num_trees, 1)
        count = 0
        for (left, right), tree1, tree2 in tsinfer.tree_pairs(ts1, ts2):
            self.assertEqual((0, 1), tree1.interval)
            self.assertEqual((0, 1), tree2.interval)
            self.assertIs(tree1.tree_sequence, ts1)
            self.assertIs(tree2.tree_sequence, ts2)
            count += 1
        self.assertEqual(count, 1)

    def test_single_tree_many_trees(self):
        n = 10
        ts1 = msprime.simulate(n, random_seed=10)
        ts2 = msprime.simulate(n, recombination_rate=10, random_seed=10)
        self.assertEqual(ts1.num_trees, 1)
        self.assertGreater(ts2.num_trees, 1)
        trees2 = ts2.trees()
        count = 0
        for (left, right), tree1, tree2 in tsinfer.tree_pairs(ts1, ts2):
            self.assertEqual((0, 1), tree1.interval)
            self.assertIs(tree1.tree_sequence, ts1)
            self.assertIs(tree2.tree_sequence, ts2)
            tree2p = next(trees2, None)
            if tree2p is not None:
                self.assertEqual(tree2.interval, tree2p.interval)
                self.assertEqual(tree2.parent_dict, tree2p.parent_dict)
            count += 1
        self.assertEqual(count, ts2.num_trees)

    def test_many_trees_single_tree(self):
        n = 10
        ts1 = msprime.simulate(n, random_seed=10)
        ts2 = msprime.simulate(n, recombination_rate=10, random_seed=10)
        self.assertEqual(ts1.num_trees, 1)
        self.assertGreater(ts2.num_trees, 1)
        trees2 = ts2.trees()
        count = 0
        for (left, right), tree2, tree1 in tsinfer.tree_pairs(ts2, ts1):
            self.assertEqual((0, 1), tree1.interval)
            self.assertIs(tree1.tree_sequence, ts1)
            self.assertIs(tree2.tree_sequence, ts2)
            tree2p = next(trees2, None)
            if tree2p is not None:
                self.assertEqual(tree2.interval, tree2p.interval)
                self.assertEqual(tree2.parent_dict, tree2p.parent_dict)
            count += 1
        self.assertEqual(count, ts2.num_trees)

    def test_different_lengths_error(self):
        ts1 = msprime.simulate(2, length=10, random_seed=1)
        ts2 = msprime.simulate(2, length=11, random_seed=1)
        self.assertRaises(ValueError, list, tsinfer.tree_pairs(ts1, ts2))
        self.assertRaises(ValueError, list, tsinfer.tree_pairs(ts2, ts1))

    def test_many_trees(self):
        ts1 = msprime.simulate(20, recombination_rate=20, random_seed=10)
        ts2 = msprime.simulate(30, recombination_rate=10, random_seed=10)
        self.assertGreater(ts1.num_trees, 1)
        self.assertGreater(ts2.num_trees, 1)
        breakpoints = [0]
        for (left, right), tree2, tree1 in tsinfer.tree_pairs(ts2, ts1):
            breakpoints.append(right)
            self.assertGreaterEqual(left, tree1.interval[0])
            self.assertGreaterEqual(left, tree2.interval[0])
            self.assertLessEqual(right, tree1.interval[1])
            self.assertLessEqual(right, tree2.interval[1])
            self.assertIs(tree1.tree_sequence, ts1)
            self.assertIs(tree2.tree_sequence, ts2)
        all_breakpoints = set(ts1.breakpoints()) | set(ts2.breakpoints())
        self.assertEqual(breakpoints, sorted(all_breakpoints))


class TestGetAncestralHaplotypes(unittest.TestCase):
    """
    Tests for the method to the actual ancestors from a simulation.
    """
    def get_matrix(self, ts):
        """
        Simple implementation using tree traversals.
        """
        A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.uint8)
        A[:] = tsinfer.UNKNOWN_ALLELE
        for t in ts.trees():
            for site in t.sites():
                for u in t.nodes():
                    A[u, site.id] = 0
                for mutation in site.mutations:
                    # Every node underneath this node will have the value set
                    # at this site.
                    for u in t.nodes(mutation.node):
                        A[u, site.id] = 1
        return A

    def verify_samples(self, ts, A):
        # Samples should be nodes rows 0 to n - 1, and should be equal to
        # the genotypes.
        G = ts.genotype_matrix()
        self.assertTrue(np.array_equal(G.T, A[:ts.num_samples]))

    def verify_single_tree(self, ts, A):
        self.assertTrue(np.all(A[-1] == 0))
        self.assertEqual(ts.num_trees, 1)
        self.assertGreater(ts.num_sites, 1)
        self.verify_haplotypes(ts, A)

    def verify_haplotypes(self, ts, A):
        self.verify_samples(ts, A)
        for tree in ts.trees():
            for site in tree.sites():
                self.assertEqual(len(site.mutations), 1)
                mutation = site.mutations[0]
                below = np.array(list(tree.nodes(mutation.node)), dtype=int)
                self.assertTrue(np.all(A[below, site.id] == 1))
                above = np.array(list(
                    set(tree.nodes()) - set(tree.nodes(mutation.node))), dtype=int)
                self.assertTrue(np.all(A[above, site.id] == 0))
                outside = np.array(list(
                    set(range(ts.num_nodes)) - set(tree.nodes())), dtype=int)
                self.assertTrue(np.all(A[outside, site.id] == tsinfer.UNKNOWN_ALLELE))

    def test_single_tree(self):
        ts = msprime.simulate(5, mutation_rate=10, random_seed=234)
        A = tsinfer.get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        self.assertTrue(np.array_equal(A, B))
        self.verify_single_tree(ts, A)

    def test_single_tree_perfect_mutations(self):
        ts = msprime.simulate(5, random_seed=234)
        ts = tsinfer.insert_perfect_mutations(ts)
        A = tsinfer.get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        self.assertTrue(np.array_equal(A, B))
        self.verify_single_tree(ts, A)

    def test_many_trees(self):
        ts = msprime.simulate(
            8, recombination_rate=10, mutation_rate=10, random_seed=234)
        self.assertGreater(ts.num_trees, 1)
        self.assertGreater(ts.num_sites, 1)
        A = tsinfer.get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        self.assertTrue(np.array_equal(A, B))
        self.verify_haplotypes(ts, A)

    def test_many_trees_perfect_mutations(self):
        ts = get_smc_simulation(10, 100, 0.1, 1234)
        self.assertGreater(ts.num_trees, 1)
        ts = tsinfer.insert_perfect_mutations(ts)
        A = tsinfer.get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        self.assertTrue(np.array_equal(A, B))
        self.verify_haplotypes(ts, A)


class TestAssertSmc(unittest.TestCase):
    """
    Check that our assertion for SMC simulations works correctly.
    """
    def test_single_tree(self):
        ts = msprime.simulate(5, random_seed=234)
        tsinfer.assert_smc(ts)

    def test_non_smc(self):
        ts = msprime.simulate(3, recombination_rate=10, random_seed=14)
        self.assertRaises(ValueError, tsinfer.assert_smc, ts)

    def test_smc(self):
        for seed in range(1, 10):
            ts = get_smc_simulation(10, 100, 0.1, seed)
            tsinfer.assert_smc(ts)
            self.assertGreater(ts.num_trees, 1)


class TestGetAncestorDescriptors(unittest.TestCase):
    """
    Tests that we correctly recover the ancestor descriptors from a
    given set of ancestral haplotypes.
    """
    def verify_single_tree_dense_mutations(self, ts):
        A = tsinfer.get_ancestral_haplotypes(ts)
        A = A[ts.num_samples:][::-1]
        n, m = A.shape
        ancestors, start, end, focal_sites = tsinfer.get_ancestor_descriptors(A)
        self.assertTrue(np.array_equal(A, ancestors[-n:]))
        self.assertEqual(start, [0 for _ in range(ancestors.shape[0])])
        self.assertTrue(end, [m for _ in range(ancestors.shape[0])])
        for j in range(1, ancestors.shape[0]):
            self.assertGreaterEqual(len(focal_sites[j]), 0)
            for site in focal_sites[j]:
                self.assertEqual(ancestors[j, site], 1)

    def verify_many_trees_dense_mutations(self, ts):
        A = tsinfer.get_ancestral_haplotypes(ts)
        A = A[ts.num_samples:][::-1]
        tsinfer.get_ancestor_descriptors(A)

        ancestors, start, end, focal_sites = tsinfer.get_ancestor_descriptors(A)
        n, m = ancestors.shape
        self.assertEqual(m, ts.num_sites)
        self.assertTrue(np.all(ancestors[0, :] == 0))
        for a, s, e, focal in zip(ancestors[1:], start[1:], end[1:], focal_sites[1:]):
            self.assertTrue(0 <= s < e <= m)
            self.assertTrue(np.all(a[:s] == tsinfer.UNKNOWN_ALLELE))
            self.assertTrue(np.all(a[e:] == tsinfer.UNKNOWN_ALLELE))
            self.assertTrue(np.all(a[s:e] != tsinfer.UNKNOWN_ALLELE))
            for site in focal:
                self.assertEqual(a[site], 1)

    def test_single_tree_perfect_mutations(self):
        ts = msprime.simulate(5, random_seed=234)
        ts = tsinfer.insert_perfect_mutations(ts)
        self.verify_single_tree_dense_mutations(ts)

    def test_single_tree_random_mutations(self):
        ts = msprime.simulate(5, mutation_rate=5, random_seed=234)
        self.assertGreater(ts.num_sites, 1)
        self.verify_single_tree_dense_mutations(ts)

    def test_small_smc_perfect_mutations(self):
        for seed in range(1, 10):
            ts = get_smc_simulation(5, 100, 0.02, seed)
            ts = tsinfer.insert_perfect_mutations(ts)
            self.assertGreater(ts.num_trees, 1)
            self.verify_many_trees_dense_mutations(ts)

    def test_large_smc_perfect_mutations(self):
        for seed in range(1, 10):
            ts = get_smc_simulation(10, 100, 0.1, seed)
            ts = tsinfer.insert_perfect_mutations(ts)
            self.assertGreater(ts.num_trees, 1)
            self.verify_many_trees_dense_mutations(ts)


class TestInsertPerfectMutations(unittest.TestCase):
    """
    Test cases for the inserting perfect mutations to allow a tree
    sequence to be exactly recovered.
    """

    def verify_perfect_mutations(self, ts):
        """
        Check that we have exactly two mutations on each edge.
        """
        for tree, ((left, right), e_out, e_in) in zip(ts.trees(), ts.edge_diffs()):
            self.assertEqual(tree.interval, (left, right))
            positions = [site.position for site in tree.sites()]
            # TODO make better tests when we've figured out the exact algorithm.
            self.assertGreater(len(positions), 0)
            self.assertEqual(positions[0], left)
            for site in tree.sites():
                self.assertEqual(len(site.mutations), 1)

    def test_single_tree(self):
        ts = msprime.simulate(5, random_seed=234)
        ts = tsinfer.insert_perfect_mutations(ts)
        self.verify_perfect_mutations(ts)

    def test_small_smc(self):
        for seed in range(1, 10):
            ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=seed)
            ts = tsinfer.insert_perfect_mutations(ts)
            self.assertGreater(ts.num_trees, 1)
            self.verify_perfect_mutations(ts)

    def test_large_smc(self):
        ts = get_smc_simulation(50, L=1, recombination_rate=100, seed=1)
        ts = tsinfer.insert_perfect_mutations(ts)
        self.assertGreater(ts.num_trees, 100)
        self.verify_perfect_mutations(ts)

    def test_multiple_recombinations(self):
        recomb_map = msprime.RecombinationMap.uniform_map(
            length=10, rate=10, num_loci=10)
        ts = msprime.simulate(10, recombination_map=recomb_map, random_seed=1)
        found = False
        for _, e_out, _ in ts.edge_diffs():
            if len(e_out) > 4:
                found = True
                break
        self.assertTrue(found)
        self.assertRaises(ValueError, tsinfer.insert_perfect_mutations, ts)


class TestPerfectInference(unittest.TestCase):
    """
    Test cases for the method to run perfect inference on an input tree sequence.
    """
    def verify_perfect_inference(self, ts, inferred_ts):
        self.assertEqual(ts.sequence_length, inferred_ts.sequence_length)
        self.assertEqual(ts.tables.edges, inferred_ts.tables.edges)
        self.assertEqual(ts.tables.sites, inferred_ts.tables.sites)
        self.assertEqual(ts.tables.mutations, inferred_ts.tables.mutations)
        self.assertTrue(
            np.array_equal(ts.tables.nodes.flags, inferred_ts.tables.nodes.flags))

    def test_single_tree_defaults(self):
        base_ts = msprime.simulate(5, random_seed=234)
        for method in ["P", "C"]:
            ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, method=method)
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=111)
        self.assertGreater(base_ts.num_trees, 1)
        for method in ["P", "C"]:
            ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, method=method)
            self.verify_perfect_inference(ts, inferred_ts)

    @unittest.skip("Path compression breaks perfect inference")
    def test_small_smc_path_compression(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=111)
        self.assertGreater(base_ts.num_trees, 1)
        for method in ["P", "C"]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, method=method, path_compression=True)
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc_threads(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=112)
        self.assertGreater(base_ts.num_trees, 1)
        for method in ["P", "C"]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, method=method, num_threads=4)
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc_no_time_chunking(self):
        base_ts = get_smc_simulation(10, L=1, recombination_rate=10, seed=113)
        self.assertGreater(base_ts.num_trees, 1)
        for method in ["P", "C"]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, method=method, time_chunking=False)
            self.verify_perfect_inference(ts, inferred_ts)
