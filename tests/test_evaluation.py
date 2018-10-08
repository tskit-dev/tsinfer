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
import subprocess
import sys
import tempfile

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

    def test_small_use_ts(self):
        base_ts = msprime.simulate(5, recombination_rate=10, random_seed=112)
        self.assertGreater(base_ts.num_trees, 1)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                    base_ts, use_ts=True, engine=engine)
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

    def test_small_use_ts_path_compression(self):
        base_ts = msprime.simulate(5, recombination_rate=10, random_seed=112)
        self.assertGreater(base_ts.num_trees, 1)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                    base_ts, use_ts=True, path_compression=True, engine=engine)
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

    def test_sample_20_use_ts(self):
        base_ts = msprime.simulate(
            20, length=5, recombination_rate=10, random_seed=111)
        self.assertGreater(base_ts.num_trees, 5)
        ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, use_ts=True)
        self.verify_perfect_inference(ts, inferred_ts)

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


class TestMakeAncestorsTs(unittest.TestCase):
    """
    Tests for the process of generating an ancestors tree sequence.
    """
    def verify_from_source(self, remove_leaves):
        ts = msprime.simulate(15, recombination_rate=1, mutation_rate=2, random_seed=3)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ancestors_ts = tsinfer.make_ancestors_ts(
            samples, ts, remove_leaves=remove_leaves)
        tsinfer.check_ancestors_ts(ancestors_ts)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            final_ts = tsinfer.match_samples(samples, ancestors_ts, engine=engine)
        tsinfer.verify(samples, final_ts)

    def test_infer_from_source_no_leaves(self):
        self.verify_from_source(True)

    def test_infer_from_source(self):
        self.verify_from_source(True)

    def verify_from_inferred(self, remove_leaves):
        ts = msprime.simulate(15, recombination_rate=1, mutation_rate=2, random_seed=3)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred = tsinfer.infer(samples)
        ancestors_ts = tsinfer.make_ancestors_ts(
            samples, inferred, remove_leaves=remove_leaves)
        tsinfer.check_ancestors_ts(ancestors_ts)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            final_ts = tsinfer.match_samples(samples, ancestors_ts, engine=engine)
        tsinfer.verify(samples, final_ts)

    def test_infer_from_inferred_no_leaves(self):
        self.verify_from_inferred(True)

    def test_infer_from_inferred(self):
        self.verify_from_inferred(False)


class TestCheckAncestorsTs(unittest.TestCase):
    """
    Tests that we verify the right conditions from an ancestors TS.
    """

    def test_empty(self):
        tables = msprime.TableCollection(1)
        tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_zero_time(self):
        tables = msprime.TableCollection(1)
        tables.nodes.add_row(time=0, flags=0)
        with self.assertRaises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_zero_edges(self):
        tables = msprime.TableCollection(1)
        tables.nodes.add_row(time=1, flags=0)
        with self.assertRaises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_one_edge(self):
        tables = msprime.TableCollection(1)
        tables.nodes.add_row(time=2, flags=0)
        tables.nodes.add_row(time=1, flags=0)
        tables.edges.add_row(0, 1, 0, 1)
        tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_zero_has_parent(self):
        tables = msprime.TableCollection(1)
        tables.nodes.add_row(time=1, flags=0)
        tables.nodes.add_row(time=2, flags=0)
        tables.edges.add_row(0, 1, 1, 0)
        with self.assertRaises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_zero_has_no_children(self):
        tables = msprime.TableCollection(1)
        tables.nodes.add_row(time=1, flags=0)
        tables.nodes.add_row(time=2, flags=0)
        tables.nodes.add_row(time=3, flags=0)
        tables.edges.add_row(0, 1, 2, 1)
        with self.assertRaises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_disconnected_subtrees(self):
        tables = msprime.TableCollection(1)
        tables.nodes.add_row(time=4, flags=1)
        tables.nodes.add_row(time=3, flags=1)
        tables.nodes.add_row(time=2, flags=1)
        tables.nodes.add_row(time=1, flags=1)
        tables.edges.add_row(0, 1, 2, 3)
        tables.edges.add_row(0, 1, 0, 1)
        with self.assertRaises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_many_mutations(self):
        tables = msprime.TableCollection(1)
        tables.nodes.add_row(time=2, flags=0)
        tables.nodes.add_row(time=1, flags=0)
        tables.edges.add_row(0, 1, 0, 1)
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.mutations.add_row(site=0, node=0, derived_state="1")
        tables.mutations.add_row(site=0, node=1, derived_state="0")
        with self.assertRaises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_msprime_output(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        with self.assertRaises(ValueError):
            tsinfer.check_ancestors_ts(ts)

    def test_tsinfer_output(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ts = tsinfer.infer(samples)
        with self.assertRaises(ValueError):
            tsinfer.check_ancestors_ts(ts)


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


class TestCli(unittest.TestCase):
    """
    Simple tests for the evaluation CLI to make sure the various tests
    at least run.
    """
    def run_command(self, command):
        with tempfile.TemporaryDirectory(prefix="tsi_eval") as tmpdir:
            subprocess.check_output(
                [sys.executable, "evaluation.py"] + command + ["-d", tmpdir])

    def test_help(self):
        self.run_command(["--help"])

    def test_perfect_inference(self):
        self.run_command(["perfect-inference", "-n", "4", "-l", "0.1", "-s", "1"])

    def test_edges_performance(self):
        self.run_command([
            "edges-performance", "-n", "5", "-l", "0.1", "-R", "2", "-s", "1"])

    def test_hotspot_analysis(self):
        self.run_command([
            "hotspot-analysis", "-n", "5", "-l", "0.1", "-R", "1", "-s", "5"])

    def test_ancestor_properties(self):
        self.run_command([
            "ancestor-properties", "-n", "5", "-l", "0.1", "-R", "1", "-s", "5"])

    def test_ancestor_comparison(self):
        self.run_command(["ancestor-properties", "-n", "5", "-l", "0.5"])

    def test_node_degree(self):
        self.run_command(["node-degree", "-n", "5", "-l", "0.1"])

    def test_ancestor_quality(self):
        self.run_command(["ancestor-quality", "-n", "5", "-l", "0.1"])


class TestCountSampleChildEdges(unittest.TestCase):
    """
    Tests the count_sample_child_edges function.
    """
    def verify(self, ts):
        sample_edges = tsinfer.count_sample_child_edges(ts)
        x = np.zeros(ts.num_samples, dtype=np.int)
        for j, node in enumerate(ts.samples()):
            for edge in ts.edges():
                if edge.child == node:
                    x[j] += 1
        self.assertTrue(np.array_equal(x, sample_edges))

    def test_simulated(self):
        ts = msprime.simulate(20, recombination_rate=2, random_seed=2)
        self.verify(ts)

    def test_inferred_no_simplify(self):
        ts = msprime.simulate(10, recombination_rate=2, mutation_rate=10, random_seed=2)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ts = tsinfer.infer(samples, simplify=False)
        self.verify(ts)

    def test_inferred_simplify(self):
        ts = msprime.simulate(10, recombination_rate=2, mutation_rate=10, random_seed=3)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ts = tsinfer.infer(samples)
        self.verify(ts)


class TestNodeSpan(unittest.TestCase):
    def simple_node_span(self, ts):
        """
        Straightforward implementation of node span calculation by iterating over
        over the trees and nodes in each tree.
        """
        S = np.zeros(ts.num_nodes)
        for tree in ts.trees():
            length = tree.interval[1] - tree.interval[0]
            for u in tree.nodes():
                if tree.num_samples(u) > 0:
                    S[u] += length
        return S

    def verify(self, ts):
        S1 = self.simple_node_span(ts)
        S2 = tsinfer.node_span(ts)
        self.assertEqual(S1.shape, S2.shape)
        self.assertTrue(np.allclose(S1, S2))
        self.assertTrue(np.all(S1 > 0))
        self.assertTrue(np.all(S1 <= ts.sequence_length))
        return S1

    def test_single_locus(self):
        ts = msprime.simulate(10, random_seed=1)
        S = self.verify(ts)
        self.assertTrue(np.all(S == 1))

    def test_single_locus_sequence_length(self):
        ts = msprime.simulate(10, length=100, random_seed=1)
        S = self.verify(ts)
        self.assertTrue(np.all(S == 100))

    def test_simple_recombination(self):
        ts = msprime.simulate(20, recombination_rate=5, random_seed=2)
        self.assertGreater(ts.num_trees, 2)
        S = self.verify(ts)
        self.assertFalse(np.all(S == 1))

    def test_simple_recombination_sequence_length(self):
        ts = msprime.simulate(20, recombination_rate=5, length=10, random_seed=3)
        self.assertGreater(ts.num_trees, 2)
        S = self.verify(ts)
        self.assertFalse(np.all(S == 10))

    def test_inferred_no_simplify(self):
        ts = msprime.simulate(10, recombination_rate=2, mutation_rate=10, random_seed=3)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ts = tsinfer.infer(samples, simplify=False)
        self.verify(ts)

    def test_inferred(self):
        ts = msprime.simulate(10, recombination_rate=2, mutation_rate=10, random_seed=3)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ts = tsinfer.infer(samples)
        self.verify(ts)

    def test_inferred_random_data(self):
        np.random.seed(10)
        num_sites = 40
        num_samples = 8
        G = np.random.randint(2, size=(num_sites, num_samples)).astype(np.uint8)
        with tsinfer.SampleData() as sample_data:
            for j in range(num_sites):
                sample_data.add_site(j, G[j])
        ts = tsinfer.infer(sample_data)
        self.verify(ts)


class TestMeanSampleAncestry(unittest.TestCase):
    """
    Tests the mean_sample_ancestry function.
    """

    def simple_mean_sample_ancestry(self, ts, sample_sets):
        """
        Straightforward implementation of mean sample ancestry by iterating
        over the trees and nodes in each tree.
        """
        A = np.zeros((len(sample_sets), ts.num_nodes))
        S = np.zeros(ts.num_nodes)
        tree_iters = [ts.trees(tracked_samples=sample_set) for sample_set in sample_sets]
        for _ in range(ts.num_trees):
            trees = [next(tree_iter) for tree_iter in tree_iters]
            left, right = trees[0].interval
            length = right - left
            for node in trees[0].nodes():
                total_samples = sum(tree.num_tracked_samples(node) for tree in trees)
                if total_samples > 0:
                    for j, tree in enumerate(trees):
                        f = tree.num_tracked_samples(node) / total_samples
                        A[j, node] += f * length
                    S[node] += length

        # The final value for each node is the mean ancestry fraction for this
        # population over the trees that it was defined in, divided by the span
        # of that node.
        index = S != 0
        A[:, index] /= S[index]
        return A

    def verify(self, ts, sample_sets):
        A1 = self.simple_mean_sample_ancestry(ts, sample_sets)
        A2 = tsinfer.mean_sample_ancestry(ts, sample_sets)
        self.assertEqual(A1.shape, A2.shape)
        # for tree in ts.trees():
        #     print(tree.interval)
        #     print(tree.draw(format="unicode"))
        # print()
        # for node in ts.nodes():
        #     print(node.id, np.sum(A2[:, node.id]), A2[:, node.id], sep="\t")
        if set(itertools.chain(*sample_sets)) == set(ts.samples()):
            self.assertTrue(np.allclose(np.sum(A1, axis=0), 1))
        else:
            S = np.sum(A1, axis=0)
            self.assertTrue(np.allclose(S[S != 0], 1))
        self.assertTrue(np.allclose(A1, A2))
        return A1

    def two_populations_high_migration_example(self, mutation_rate=10):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(8),
                msprime.PopulationConfiguration(8)],
            migration_matrix=[[0, 1], [1, 0]],
            recombination_rate=3,
            mutation_rate=mutation_rate,
            random_seed=5)
        self.assertGreater(ts.num_trees, 1)
        return ts

    def get_random_data_example(self, num_sites, num_samples, seed=100):
        np.random.seed(seed)
        G = np.random.randint(2, size=(num_sites, num_samples)).astype(np.uint8)
        with tsinfer.SampleData() as sample_data:
            for j in range(num_sites):
                sample_data.add_site(j, G[j])
        return sample_data

    def test_two_populations_high_migration(self):
        ts = self.two_populations_high_migration_example()
        A = self.verify(ts, [ts.samples(0), ts.samples(1)])
        total = np.sum(A, axis=0)
        self.assertTrue(np.allclose(total[total != 0], 1))

    def test_two_populations_high_migration_no_centromere(self):
        ts = self.two_populations_high_migration_example(mutation_rate=0)
        ts = tsinfer.snip_centromere(ts, 0.4, 0.6)
        # simplify the output to get rid of unreferenced nodes.
        ts = ts.simplify()
        A = self.verify(ts, [ts.samples(0), ts.samples(1)])
        total = np.sum(A, axis=0)
        self.assertTrue(np.allclose(total[total != 0], 1))

    def test_span_zero_nodes(self):
        ts = msprime.simulate(10, random_seed=1)
        tables = ts.dump_tables()
        # Add in a few unreferenced nodes.
        u1 = tables.nodes.add_row(flags=0, time=1234)
        u2 = tables.nodes.add_row(flags=1, time=1234)
        ts = tables.tree_sequence()
        sample_sets = [[j] for j in range(10)]
        A1 = self.simple_mean_sample_ancestry(ts, sample_sets)
        A2 = tsinfer.mean_sample_ancestry(ts, sample_sets)
        S = np.sum(A1, axis=0)
        self.assertTrue(np.allclose(A1, A2))
        self.assertTrue(np.allclose(S[:u1], 1))
        self.assertTrue(np.all(A1[:, u1] == 0))
        self.assertTrue(np.all(A1[:, u2] == 0))

    def test_two_populations_incomplete_samples(self):
        ts = self.two_populations_high_migration_example()
        samples = ts.samples()
        A = self.verify(ts, [samples[:2], samples[-2:]])
        total = np.sum(A, axis=0)
        self.assertTrue(np.allclose(total[total != 0], 1))

    def test_single_tree_incomplete_samples(self):
        ts = msprime.simulate(10, random_seed=1)
        A = self.verify(ts, [[0, 1], [2, 3]])
        total = np.sum(A, axis=0)
        self.assertTrue(np.allclose(total[total != 0], 1))

    def test_two_populations_overlapping_samples(self):
        ts = self.two_populations_high_migration_example()
        with self.assertRaises(ValueError):
            tsinfer.mean_sample_ancestry(ts, [[1], [1]])
        with self.assertRaises(ValueError):
            tsinfer.mean_sample_ancestry(ts, [[1, 1], [2]])

    def test_two_populations_high_migration_inferred(self):
        ts = self.two_populations_high_migration_example()
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred_ts = tsinfer.infer(samples)
        self.assertEqual(inferred_ts.num_populations, ts.num_populations)
        self.verify(inferred_ts, [inferred_ts.samples(0), inferred_ts.samples(1)])

    def test_two_populations_high_migration_inferred_no_simplify(self):
        ts = self.two_populations_high_migration_example()
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred_ts = tsinfer.infer(samples, simplify=False)
        self.assertEqual(inferred_ts.num_populations, ts.num_populations)
        self.verify(inferred_ts, [inferred_ts.samples(0), inferred_ts.samples(1)])

    def test_random_data_inferred(self):
        n = 20
        samples = self.get_random_data_example(num_sites=52, num_samples=n)
        inferred_ts = tsinfer.infer(samples)
        samples = inferred_ts.samples()
        self.verify(inferred_ts, [samples[: n // 2], samples[n // 2:]])

    def test_random_data_inferred_no_simplify(self):
        samples = self.get_random_data_example(num_sites=20, num_samples=3)
        inferred_ts = tsinfer.infer(samples, simplify=False)
        samples = inferred_ts.samples()
        self.verify(inferred_ts, [samples[:1], samples[1:]])

    def test_many_groups(self):
        ts = msprime.simulate(32, recombination_rate=10, random_seed=4)
        samples = ts.samples()
        group_size = 1
        while group_size <= ts.num_samples:
            sample_sets = [
                samples[j * group_size: (j + 1) * group_size]
                for j in range(ts.num_samples // group_size)]
            self.verify(ts, sample_sets)
            group_size *= 2


class TestSnipCentromere(unittest.TestCase):
    """
    Tests that we remove the centromere successfully from tree sequences.
    """
    def snip_centromere(self, ts, left, right):
        """
        Simple implementation of snipping out centromere.
        """
        assert 0 < left < right < ts.sequence_length
        tables = ts.dump_tables()
        tables.edges.clear()
        for edge in ts.edges():
            if right <= edge.left or left >= edge.right:
                tables.edges.add_row(edge.left, edge.right, edge.parent, edge.child)
            else:
                if edge.left < left:
                    tables.edges.add_row(edge.left, left, edge.parent, edge.child)
                if right < edge.right:
                    tables.edges.add_row(right, edge.right, edge.parent, edge.child)
        tables.sort()
        return tables.tree_sequence()

    def verify(self, ts, left, right):
        ts1 = self.snip_centromere(ts, left, right)
        ts2 = tsinfer.snip_centromere(ts, left, right)
        t1 = ts1.dump_tables()
        t2 = ts2.dump_tables()
        t1.provenances.clear()
        t2.provenances.clear()
        self.assertEqual(t1, t2)
        tree_found = False
        for tree in ts1.trees():
            if tree.interval == (left, right):
                tree_found = True
                for node in ts1.nodes():
                    self.assertEqual(tree.parent(node.id), msprime.NULL_NODE)
                break
        self.assertTrue(tree_found)
        return ts1

    def test_single_tree(self):
        ts1 = msprime.simulate(10, random_seed=1)
        ts2 = self.verify(ts1, 0.5, 0.6)
        self.assertEqual(ts2.num_trees, 3)

    def test_many_trees(self):
        ts1 = msprime.simulate(10, length=10, recombination_rate=1, random_seed=1)
        self.assertGreater(ts1.num_trees, 2)
        self.verify(ts1, 5, 6)

    def get_random_data_example(self, position, num_samples, seed=100):
        np.random.seed(seed)
        G = np.random.randint(2, size=(position.shape[0], num_samples)).astype(np.uint8)
        with tsinfer.SampleData() as sample_data:
            for j, x in enumerate(position):
                sample_data.add_site(x, G[j])
        return sample_data

    def test_random_data_inferred_no_simplify(self):
        samples = self.get_random_data_example(
            10 * np.arange(10), num_samples=10, seed=2)
        inferred_ts = tsinfer.infer(samples, simplify=False)
        ts = self.verify(inferred_ts, 55, 57)
        self.assertTrue(np.array_equal(
            ts.genotype_matrix(), inferred_ts.genotype_matrix()))

    def test_random_data_inferred_simplify(self):
        samples = self.get_random_data_example(5 * np.arange(10), num_samples=10, seed=2)
        inferred_ts = tsinfer.infer(samples, simplify=True)
        ts = self.verify(inferred_ts, 12, 15)
        self.assertTrue(np.array_equal(
            ts.genotype_matrix(), inferred_ts.genotype_matrix()))

    def test_coordinate_errors(self):
        ts = msprime.simulate(2, length=10, recombination_rate=1, random_seed=1)
        self.assertRaises(ValueError, tsinfer.snip_centromere, ts, -1, 5)
        self.assertRaises(ValueError, tsinfer.snip_centromere, ts, 0, 5)
        self.assertRaises(ValueError, tsinfer.snip_centromere, ts, 1, 10)
        self.assertRaises(ValueError, tsinfer.snip_centromere, ts, 1, 11)
        self.assertRaises(ValueError, tsinfer.snip_centromere, ts, 6, 5)
        self.assertRaises(ValueError, tsinfer.snip_centromere, ts, 5, 5)

    def test_position_errors(self):
        ts = msprime.simulate(
            2, length=10, recombination_rate=1, random_seed=1, mutation_rate=2)
        X = ts.tables.sites.position
        self.assertGreater(X.shape[0], 3)
        # Left cannot be on a site position.
        self.assertRaises(ValueError, tsinfer.snip_centromere, ts, X[0], X[0] + 0.001)
        # Cannot go either side of a position
        self.assertRaises(
            ValueError, tsinfer.snip_centromere, ts, X[0] - 0.001, X[0] + 0.001)
        # Cannot cover multiple positions
        self.assertRaises(
            ValueError, tsinfer.snip_centromere, ts, X[0] - 0.001, X[2] + 0.001)

    def test_right_on_position(self):
        ts1 = msprime.simulate(
            2, length=10, recombination_rate=1, random_seed=1, mutation_rate=2)
        X = ts1.tables.sites.position
        self.assertGreater(X.shape[0], 1)
        ts2 = self.verify(ts1, X[0] - 0.001, X[0])
        self.assertTrue(np.array_equal(ts1.genotype_matrix(), ts2.genotype_matrix()))
