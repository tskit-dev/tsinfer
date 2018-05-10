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
Test cases for the evaluation code.
"""
import itertools
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


def kc_distance(tree1, tree2):
    """
    Returns the Kendall-Colijn topological distance between the specified
    pair of trees. This is a very simple and direct implementation for testing.
    """
    samples = tree1.tree_sequence.samples()
    if not np.array_equal(samples, tree2.tree_sequence.samples()):
        raise ValueError("Trees must have the same samples")
    k = samples.shape[0]
    n = (k * (k - 1)) // 2
    M = [np.ones(n + k), np.ones(n + k)]
    for tree_index, tree in enumerate([tree1, tree2]):
        for j, (a, b) in enumerate(itertools.combinations(samples, 2)):
            u = tree.mrca(a, b)
            path_len = 0
            v = u
            while tree.parent(v) != msprime.NULL_NODE:
                path_len += 1
                v = tree.parent(v)
            M[tree_index][j] = path_len
    return np.linalg.norm(M[0] - M[1])


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

    def validate_trees(self, n):
        for seed in range(1, 10):
            tree1 = msprime.simulate(n, random_seed=seed).first()
            tree2 = msprime.simulate(n, random_seed=seed + 1).first()
            self.assertAlmostEqual(
                tsinfer.kc_distance(tree1, tree2), kc_distance(tree1, tree2))

    def test_sample_3(self):
        self.validate_trees(3)

    def test_sample_4(self):
        self.validate_trees(4)

    def test_sample_10(self):
        self.validate_trees(10)

    def test_sample_20(self):
        self.validate_trees(20)

    def validate_nonbinary_trees(self, n):
        demographic_events = [
            msprime.SimpleBottleneck(0.02, 0, proportion=0.25),
            msprime.SimpleBottleneck(0.2, 0, proportion=1)]

        for seed in range(1, 10):
            ts = msprime.simulate(
                n, random_seed=seed, demographic_events=demographic_events)
            # Check if this is really nonbinary
            found = False
            for edgeset in ts.edgesets():
                if len(edgeset.children) > 2:
                    found = True
                    break
            self.assertTrue(found)
            tree1 = ts.first()

            ts = msprime.simulate(
                n, random_seed=seed + 1, demographic_events=demographic_events)
            tree2 = ts.first()
            self.assertAlmostEqual(
                tsinfer.kc_distance(tree1, tree2), kc_distance(tree1, tree2))
            self.assertAlmostEqual(
                tsinfer.kc_distance(tree2, tree1), kc_distance(tree2, tree1))
            # compare to a binary tree also
            tree2 = msprime.simulate(n, random_seed=seed + 1).first()
            self.assertAlmostEqual(
                tsinfer.kc_distance(tree1, tree2), kc_distance(tree1, tree2))
            self.assertAlmostEqual(
                tsinfer.kc_distance(tree2, tree1), kc_distance(tree2, tree1))

    def test_non_binary_sample_10(self):
        self.validate_nonbinary_trees(10)

    def test_non_binary_sample_20(self):
        self.validate_nonbinary_trees(20)

    def test_non_binary_sample_30(self):
        self.validate_nonbinary_trees(30)


class TestTreeSequenceCompare(unittest.TestCase):
    """
    Tests of the engine to compare to tree sequences.
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
    Tests of the engine to compare to tree sequences.
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
    Tests for the engine to the actual ancestors from a simulation.
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
    Test cases for the engine to run perfect inference on an input tree sequence.
    """
    def verify_perfect_inference(self, ts, inferred_ts):
        self.assertEqual(ts.sequence_length, inferred_ts.sequence_length)
        inferred = inferred_ts.dump_tables()
        source = ts.dump_tables()
        self.assertEqual(source.edges, inferred.edges)
        # The metadata column will be different in the tables so we have to check
        # column by column for therest.
        self.assertTrue(np.array_equal(source.nodes.flags, inferred.nodes.flags))
        self.assertTrue(np.array_equal(source.sites.position, inferred.sites.position))
        self.assertTrue(np.array_equal(
            source.sites.ancestral_state, inferred.sites.ancestral_state))
        self.assertTrue(np.array_equal(
            source.sites.ancestral_state_offset, inferred.sites.ancestral_state_offset))
        self.assertTrue(np.array_equal(source.mutations.site, inferred.mutations.site))
        self.assertTrue(np.array_equal(source.mutations.node, inferred.mutations.node))
        self.assertTrue(np.array_equal(
            source.mutations.parent, inferred.mutations.parent))
        self.assertTrue(np.array_equal(
            source.mutations.derived_state, inferred.mutations.derived_state))
        self.assertTrue(np.array_equal(
            source.mutations.derived_state_offset,
            inferred.mutations.derived_state_offset))

    def test_single_tree_defaults(self):
        base_ts = msprime.simulate(5, random_seed=234)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, engine=engine)
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=111)
        self.assertGreater(base_ts.num_trees, 1)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, engine=engine)
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc_path_compression(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=111)
        self.assertGreater(base_ts.num_trees, 1)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, engine=engine, path_compression=True)
            # We can't just compare tables when doing path compression because
            # we'll find different ways of expressing the same trees.
            breakpoints, distances = tsinfer.compare(ts, inferred_ts)
            self.assertTrue(np.all(distances == 0))

    def test_sample_20_smc_path_compression(self):
        base_ts = get_smc_simulation(20, L=5, recombination_rate=10, seed=111)
        self.assertGreater(base_ts.num_trees, 5)
        ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, path_compression=True)
        # We can't just compare tables when doing path compression because
        # we'll find different ways of expressing the same trees.
        breakpoints, distances = tsinfer.compare(ts, inferred_ts)
        self.assertTrue(np.all(distances == 0))

    def test_small_smc_threads(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=112)
        self.assertGreater(base_ts.num_trees, 1)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, engine=engine, num_threads=4)
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc_no_time_chunking(self):
        base_ts = get_smc_simulation(10, L=1, recombination_rate=10, seed=113)
        self.assertGreater(base_ts.num_trees, 1)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, engine=engine, time_chunking=False)
            self.verify_perfect_inference(ts, inferred_ts)


class TestErrors(unittest.TestCase):
    """
    Tests for the error generation code.
    """
    def test_zero_error(self):
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=10, random_seed=1)
        self.assertTrue(ts.num_sites > 1)
        self.assertTrue(ts.num_trees > 1)
        tsp = tsinfer.insert_errors(ts, 0)
        t1 = ts.tables
        t2 = tsp.tables
        self.assertEqual(t1.nodes, t2.nodes)
        self.assertEqual(t1.edges, t2.edges)
        self.assertEqual(t1.sites, t2.sites)
        self.assertEqual(t1.mutations, t2.mutations)

    def verify(self, ts, tsp):
        """
        Verifies that the specified tree sequence tsp is correctly
        derived from ts.
        """
        t1 = ts.tables
        t2 = tsp.tables
        self.assertEqual(t1.nodes, t2.nodes)
        self.assertEqual(t1.edges, t2.edges)
        self.assertEqual(t1.sites, t2.sites)
        for site1, site2 in zip(ts.sites(), tsp.sites()):
            mut1 = site1.mutations[0]
            mut2 = site2.mutations[0]
            self.assertEqual(mut1.site, mut2.site)
            self.assertEqual(mut1.node, mut2.node)
            self.assertEqual(mut1.derived_state, mut2.derived_state)
            node = ts.node(mut1.node)
            for mut in site2.mutations[1:]:
                node = ts.node(mut.node)
                self.assertTrue(node.is_sample())
        for v1, v2 in zip(ts.variants(), tsp.variants()):
            diffs = np.sum(v1.genotypes != v2.genotypes)
            self.assertEqual(diffs, len(v2.site.mutations) - 1)

    def test_simple_error(self):
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=10, random_seed=2)
        self.assertTrue(ts.num_sites > 1)
        self.assertTrue(ts.num_trees > 1)
        tsp = tsinfer.insert_errors(ts, 0.1)
        # We should have some extra mutations
        self.assertGreater(tsp.num_mutations, ts.num_mutations)
        self.verify(ts, tsp)
