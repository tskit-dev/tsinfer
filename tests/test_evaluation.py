#
# Copyright (C) 2018-2022 University of Oxford
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
import subprocess
import sys
import tempfile

import msprime
import numpy as np
import pytest
import tskit
import tsutil

import tsinfer


def get_smc_simulation(n, L=1, recombination_rate=0, seed=1):
    """
    Returns an smc simulation for a sample of size n, with sequence length L
    and recombination at the specified rate.
    """
    return msprime.simulate(
        n,
        length=L,
        recombination_rate=recombination_rate,
        random_seed=seed,
        model="smc_prime",
    )


class TestTreeSequenceCompare:
    """
    Tests of the engine to compare to tree sequences.
    """

    def test_same_ts(self):
        n = 15
        for seed in range(1, 10):
            ts = msprime.simulate(n, recombination_rate=10, random_seed=seed)
            assert ts.num_trees > 1
            bp, distance = tsinfer.compare(ts, ts)
            assert list(bp) == list(ts.breakpoints())
            assert distance.shape == (bp.shape[0] - 1,)
            assert np.all(distance == 0)

    def test_single_tree(self):
        n = 15
        for seed in range(1, 10):
            ts1 = msprime.simulate(n, random_seed=seed)
            ts2 = msprime.simulate(n, random_seed=seed + 1)
            bp, distance = tsinfer.compare(ts1, ts2)
            assert list(bp) == [0, 1]
            assert distance.shape == (1,)

    def test_single_tree_many_trees(self):
        n = 5
        for seed in range(1, 10):
            ts1 = msprime.simulate(n, recombination_rate=5, random_seed=seed)
            ts2 = msprime.simulate(n, random_seed=seed + 1)
            assert ts1.num_trees > 1
            bp, distance = tsinfer.compare(ts1, ts2)
            assert list(bp) == list(ts1.breakpoints())
            assert distance.shape == (ts1.num_trees,)

    def test_single_many_trees(self):
        n = 5
        for seed in range(1, 10):
            ts1 = msprime.simulate(n, recombination_rate=5, random_seed=seed)
            ts2 = msprime.simulate(n, recombination_rate=5, random_seed=seed + 1)
            assert ts1.num_trees > 1
            assert ts2.num_trees > 1
            bp, distance = tsinfer.compare(ts1, ts2)
            breakpoints = set(ts1.breakpoints()) | set(ts2.breakpoints())
            assert list(bp) == sorted(breakpoints)
            assert distance.shape == (len(breakpoints) - 1,)

    # TODO add some examples testing for specific instances.


class TestCoiteration:
    """
    Tests of ts.coiterate, used in evaluating.
    """

    def test_same_ts(self):
        n = 15
        ts = msprime.simulate(n, recombination_rate=10, random_seed=10)
        assert ts.num_trees > 1
        count = 0
        for interval, tree1, tree2 in ts.coiterate(ts):
            assert interval == tree1.interval
            assert tree1.interval == tree2.interval
            assert tree1.parent_dict == tree2.parent_dict
            count += 1
        assert count == ts.num_trees

    def test_single_tree(self):
        n = 10
        ts1 = msprime.simulate(n, random_seed=10)
        ts2 = msprime.simulate(n, random_seed=10)
        assert ts1.num_trees == 1
        assert ts2.num_trees == 1
        count = 0
        for (_left, _right), tree1, tree2 in ts1.coiterate(ts2):
            assert (0, 1) == tree1.interval
            assert (0, 1) == tree2.interval
            assert tree1.tree_sequence is ts1
            assert tree2.tree_sequence is ts2
            count += 1
        assert count == 1

    def test_single_tree_many_trees(self):
        n = 10
        ts1 = msprime.simulate(n, random_seed=10)
        ts2 = msprime.simulate(n, recombination_rate=10, random_seed=10)
        assert ts1.num_trees == 1
        assert ts2.num_trees > 1
        trees2 = ts2.trees()
        count = 0
        for (_left, _right), tree1, tree2 in ts1.coiterate(ts2):
            assert (0, 1) == tree1.interval
            assert tree1.tree_sequence is ts1
            assert tree2.tree_sequence is ts2
            tree2p = next(trees2, None)
            if tree2p is not None:
                assert tree2.interval == tree2p.interval
                assert tree2.parent_dict == tree2p.parent_dict
            count += 1
        assert count == ts2.num_trees

    def test_many_trees_single_tree(self):
        n = 10
        ts1 = msprime.simulate(n, random_seed=10)
        ts2 = msprime.simulate(n, recombination_rate=10, random_seed=10)
        assert ts1.num_trees == 1
        assert ts2.num_trees > 1
        trees2 = ts2.trees()
        count = 0
        for (_left, _right), tree2, tree1 in ts2.coiterate(ts1):
            assert (0, 1) == tree1.interval
            assert tree1.tree_sequence is ts1
            assert tree2.tree_sequence is ts2
            tree2p = next(trees2, None)
            if tree2p is not None:
                assert tree2.interval == tree2p.interval
                assert tree2.parent_dict == tree2p.parent_dict
            count += 1
        assert count == ts2.num_trees

    def test_different_lengths_error(self):
        ts1 = msprime.simulate(2, length=10, random_seed=1)
        ts2 = msprime.simulate(2, length=11, random_seed=1)
        with pytest.raises(ValueError):
            list(ts1.coiterate(ts2))
        with pytest.raises(ValueError):
            list(ts2.coiterate(ts1))

    def test_many_trees(self):
        ts1 = msprime.simulate(20, recombination_rate=20, random_seed=10)
        ts2 = msprime.simulate(30, recombination_rate=10, random_seed=10)
        assert ts1.num_trees > 1
        assert ts2.num_trees > 1
        breakpoints = [0]
        for (left, right), tree2, tree1 in ts2.coiterate(ts1):
            breakpoints.append(right)
            assert left >= tree1.interval[0]
            assert left >= tree2.interval[0]
            assert right <= tree1.interval[1]
            assert right <= tree2.interval[1]
            assert tree1.tree_sequence is ts1
            assert tree2.tree_sequence is ts2
        all_breakpoints = set(ts1.breakpoints()) | set(ts2.breakpoints())
        assert breakpoints == sorted(all_breakpoints)


class TestGetAncestralHaplotypes:
    """
    Tests for the engine to the actual ancestors from a simulation.
    """

    def get_matrix(self, ts):
        """
        Simple implementation using tree traversals.
        """
        A = np.full((ts.num_nodes, ts.num_sites), tskit.MISSING_DATA, dtype=np.int8)
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
        assert np.array_equal(G.T, A[: ts.num_samples])

    def verify_single_tree(self, ts, A):
        assert np.all(A[-1] == 0)
        assert ts.num_trees == 1
        assert ts.num_sites > 1
        self.verify_haplotypes(ts, A)

    def verify_haplotypes(self, ts, A):
        self.verify_samples(ts, A)
        for tree in ts.trees():
            for site in tree.sites():
                assert len(site.mutations) == 1
                mutation = site.mutations[0]
                below = np.array(list(tree.nodes(mutation.node)), dtype=int)
                assert np.all(A[below, site.id] == 1)
                above = np.array(
                    list(set(tree.nodes()) - set(tree.nodes(mutation.node))), dtype=int
                )
                assert np.all(A[above, site.id] == 0)
                outside = np.array(
                    list(set(range(ts.num_nodes)) - set(tree.nodes())), dtype=int
                )
                assert np.all(A[outside, site.id] == tskit.MISSING_DATA)

    def test_single_tree(self):
        ts = msprime.simulate(5, mutation_rate=10, random_seed=234)
        A = tsinfer.get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        assert np.array_equal(A, B)
        self.verify_single_tree(ts, A)

    def test_single_tree_perfect_mutations(self):
        ts = msprime.simulate(5, random_seed=234)
        ts = tsinfer.insert_perfect_mutations(ts)
        A = tsinfer.get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        assert np.array_equal(A, B)
        self.verify_single_tree(ts, A)

    def test_many_trees(self):
        ts = msprime.simulate(
            8, recombination_rate=10, mutation_rate=10, random_seed=234
        )
        assert ts.num_trees > 1
        assert ts.num_sites > 1
        A = tsinfer.get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        assert np.array_equal(A, B)
        self.verify_haplotypes(ts, A)

    def test_many_trees_perfect_mutations(self):
        ts = get_smc_simulation(10, 100, 0.1, 1234)
        assert ts.num_trees > 1
        ts = tsinfer.insert_perfect_mutations(ts)
        A = tsinfer.get_ancestral_haplotypes(ts)
        B = self.get_matrix(ts)
        assert np.array_equal(A, B)
        self.verify_haplotypes(ts, A)


class TestAssertSmc:
    """
    Check that our assertion for SMC simulations works correctly.
    """

    def test_single_tree(self):
        ts = msprime.simulate(5, random_seed=234)
        tsinfer.assert_smc(ts)

    def test_non_smc(self):
        ts = msprime.simulate(3, recombination_rate=10, random_seed=14)
        with pytest.raises(ValueError):
            tsinfer.assert_smc(ts)

    def test_smc(self):
        for seed in range(1, 10):
            ts = get_smc_simulation(10, 100, 0.1, seed)
            tsinfer.assert_smc(ts)
            assert ts.num_trees > 1


class TestGetAncestorDescriptors:
    """
    Tests that we correctly recover the ancestor descriptors from a
    given set of ancestral haplotypes.
    """

    def verify_single_tree_dense_mutations(self, ts):
        A = tsinfer.get_ancestral_haplotypes(ts)
        A = A[ts.num_samples :][::-1]
        n, m = A.shape
        ancestors, start, end, focal_sites = tsinfer.get_ancestor_descriptors(A)
        assert np.array_equal(A, ancestors[-n:])
        assert start == [0 for _ in range(ancestors.shape[0])]
        assert end, [m for _ in range(ancestors.shape[0])]
        for j in range(1, ancestors.shape[0]):
            assert len(focal_sites[j]) >= 0
            for site in focal_sites[j]:
                assert ancestors[j, site] == 1

    def verify_many_trees_dense_mutations(self, ts):
        A = tsinfer.get_ancestral_haplotypes(ts)
        A = A[ts.num_samples :][::-1]
        tsinfer.get_ancestor_descriptors(A)

        ancestors, start, end, focal_sites = tsinfer.get_ancestor_descriptors(A)
        n, m = ancestors.shape
        assert m == ts.num_sites
        assert np.all(ancestors[0, :] == 0)
        for a, s, e, focal in zip(ancestors[1:], start[1:], end[1:], focal_sites[1:]):
            assert 0 <= s < e <= m
            assert np.all(a[:s] == tskit.MISSING_DATA)
            assert np.all(a[e:] == tskit.MISSING_DATA)
            assert np.all(a[s:e] != tskit.MISSING_DATA)
            for site in focal:
                assert a[site] == 1

    def test_single_tree_perfect_mutations(self):
        ts = msprime.simulate(5, random_seed=234)
        ts = tsinfer.insert_perfect_mutations(ts)
        self.verify_single_tree_dense_mutations(ts)

    def test_single_tree_random_mutations(self):
        ts = msprime.simulate(5, mutation_rate=5, random_seed=234)
        assert ts.num_sites > 1
        self.verify_single_tree_dense_mutations(ts)

    def test_small_smc_perfect_mutations(self):
        for seed in range(1, 10):
            ts = get_smc_simulation(5, 100, 0.02, seed)
            ts = tsinfer.insert_perfect_mutations(ts)
            assert ts.num_trees > 1
            self.verify_many_trees_dense_mutations(ts)

    def test_large_smc_perfect_mutations(self):
        for seed in range(1, 10):
            ts = get_smc_simulation(10, 100, 0.1, seed)
            ts = tsinfer.insert_perfect_mutations(ts)
            assert ts.num_trees > 1
            self.verify_many_trees_dense_mutations(ts)


class TestInsertPerfectMutations:
    """
    Test cases for the inserting perfect mutations to allow a tree
    sequence to be exactly recovered.
    """

    def verify_perfect_mutations(self, ts):
        """
        Check that we have exactly two mutations on each edge.
        """
        for tree, ((left, right), _e_out, _e_in) in zip(ts.trees(), ts.edge_diffs()):
            assert tree.interval == (left, right)
            positions = [site.position for site in tree.sites()]
            # TODO make better tests when we've figured out the exact algorithm.
            assert len(positions) > 0
            assert positions[0] == left
            for site in tree.sites():
                assert len(site.mutations) == 1

    def test_single_tree(self):
        ts = msprime.simulate(5, random_seed=234)
        ts = tsinfer.insert_perfect_mutations(ts)
        self.verify_perfect_mutations(ts)

    def test_small_smc(self):
        for seed in range(1, 10):
            ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=seed)
            ts = tsinfer.insert_perfect_mutations(ts)
            assert ts.num_trees > 1
            self.verify_perfect_mutations(ts)

    def test_large_smc(self):
        ts = get_smc_simulation(50, L=1, recombination_rate=100, seed=1)
        ts = tsinfer.insert_perfect_mutations(ts)
        assert ts.num_trees > 100
        self.verify_perfect_mutations(ts)

    def test_multiple_recombinations(self):
        ts = msprime.sim_ancestry(
            5, sequence_length=10, recombination_rate=10, random_seed=1
        )
        found = False
        for _, e_out, _ in ts.edge_diffs():
            if len(e_out) > 4:
                found = True
                break
        assert found
        with pytest.raises(ValueError):
            tsinfer.insert_perfect_mutations(ts)


class TestPerfectInference:
    """
    Test cases for the engine to run perfect inference on an input tree sequence.
    """

    def verify_perfect_inference(self, ts, inferred_ts):
        assert ts.sequence_length == inferred_ts.sequence_length
        inferred = inferred_ts.dump_tables()
        source = ts.dump_tables()
        assert source.edges == inferred.edges
        # The metadata column will be different in the tables so we have to check
        # column by column for therest.
        assert np.array_equal(source.nodes.flags, inferred.nodes.flags)
        assert np.array_equal(source.sites.position, inferred.sites.position)
        assert np.array_equal(
            source.sites.ancestral_state, inferred.sites.ancestral_state
        )
        assert np.array_equal(
            source.sites.ancestral_state_offset, inferred.sites.ancestral_state_offset
        )
        assert np.array_equal(source.mutations.site, inferred.mutations.site)
        assert np.array_equal(source.mutations.node, inferred.mutations.node)
        assert np.array_equal(source.mutations.parent, inferred.mutations.parent)
        assert np.array_equal(
            source.mutations.derived_state, inferred.mutations.derived_state
        )
        assert np.array_equal(
            source.mutations.derived_state_offset,
            inferred.mutations.derived_state_offset,
        )

    def test_single_tree_defaults(self):
        base_ts = msprime.simulate(5, random_seed=234)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, engine=engine)
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=111)
        assert base_ts.num_trees > 1
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, engine=engine)
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_use_ts(self):
        base_ts = msprime.simulate(5, recombination_rate=10, random_seed=112)
        assert base_ts.num_trees > 1
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, use_ts=True, engine=engine
            )
            self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc_path_compression(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=111)
        assert base_ts.num_trees > 1
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, engine=engine, path_compression=True
            )
            # We can't just compare tables when doing path compression because
            # we'll find different ways of expressing the same trees.
            breakpoints, distances = tsinfer.compare(ts, inferred_ts)
            assert np.all(distances == 0)

    def test_small_use_ts_path_compression(self):
        base_ts = msprime.simulate(5, recombination_rate=10, random_seed=112)
        assert base_ts.num_trees > 1
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, use_ts=True, path_compression=True, engine=engine
            )
            # We can't just compare tables when doing path compression because
            # we'll find different ways of expressing the same trees.
            breakpoints, distances = tsinfer.compare(ts, inferred_ts)
            assert np.all(distances == 0)

    def test_sample_20_smc_path_compression(self):
        base_ts = get_smc_simulation(20, L=5, recombination_rate=10, seed=111)
        assert base_ts.num_trees > 5
        ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, path_compression=True)
        # We can't just compare tables when doing path compression because
        # we'll find different ways of expressing the same trees.
        breakpoints, distances = tsinfer.compare(ts, inferred_ts)
        assert np.all(distances == 0)

    def test_sample_20_use_ts(self):
        base_ts = msprime.simulate(20, length=5, recombination_rate=10, random_seed=111)
        assert base_ts.num_trees > 5
        ts, inferred_ts = tsinfer.run_perfect_inference(base_ts, use_ts=True)
        self.verify_perfect_inference(ts, inferred_ts)

    def test_small_smc_threads(self):
        base_ts = get_smc_simulation(5, L=1, recombination_rate=10, seed=112)
        assert base_ts.num_trees > 1
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, engine=engine, num_threads=4
            )
            self.verify_perfect_inference(ts, inferred_ts)

    @pytest.mark.slow
    def test_small_smc_no_time_chunking(self):
        base_ts = get_smc_simulation(10, L=1, recombination_rate=10, seed=113)
        assert base_ts.num_trees > 1
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts, inferred_ts = tsinfer.run_perfect_inference(
                base_ts, engine=engine, time_chunking=False
            )
            self.verify_perfect_inference(ts, inferred_ts)


class TestMakeAncestorsTs:
    """
    Tests for the process of generating an ancestors tree sequence.
    """

    def verify_from_source(self, ts, remove_leaves):
        samples = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        ancestors_ts = tsinfer.make_ancestors_ts(ts, remove_leaves=remove_leaves)
        tsinfer.check_ancestors_ts(ancestors_ts)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            final_ts = tsinfer.match_samples(samples, ancestors_ts, engine=engine)
        tsinfer.verify(samples, final_ts)

    @pytest.mark.parametrize("remove_leaves", [True, False])
    def test_infer_from_source(self, remove_leaves):
        ts = msprime.simulate(15, recombination_rate=1, mutation_rate=2, random_seed=3)
        self.verify_from_source(ts, remove_leaves=remove_leaves)

    def verify_from_inferred(self, remove_leaves):
        ts = msprime.simulate(15, recombination_rate=1, mutation_rate=2, random_seed=3)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred = tsinfer.infer(samples)
        ancestors_ts = tsinfer.make_ancestors_ts(inferred, remove_leaves=remove_leaves)
        tsinfer.check_ancestors_ts(ancestors_ts)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            final_ts = tsinfer.match_samples(samples, ancestors_ts, engine=engine)
        tsinfer.verify(samples, final_ts)

    @pytest.mark.parametrize("remove_leaves", [True, False])
    def test_infer_from_inferred(self, remove_leaves):
        self.verify_from_inferred(remove_leaves)

    @pytest.mark.parametrize("remove_leaves", [True, False])
    def test_infer_from_source_multiple_mutations(self, remove_leaves):
        ts = msprime.sim_ancestry(5, sequence_length=100, random_seed=3)
        mts = msprime.sim_mutations(ts, rate=0.1, random_seed=3)
        assert mts.num_mutations > mts.num_sites
        self.verify_from_source(mts, remove_leaves=remove_leaves)


class TestCheckAncestorsTs:
    """
    Tests that we verify the right conditions from an ancestors TS.
    """

    def test_empty(self):
        tables = tskit.TableCollection(1)
        tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_zero_time(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(time=0, flags=0)
        with pytest.raises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_zero_edges(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(time=1, flags=0)
        with pytest.raises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_one_edge(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(time=2, flags=0)
        tables.nodes.add_row(time=1, flags=0)
        tables.edges.add_row(0, 1, 0, 1)
        tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_zero_has_parent(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(time=1, flags=0)
        tables.nodes.add_row(time=2, flags=0)
        tables.edges.add_row(0, 1, 1, 0)
        with pytest.raises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_zero_has_no_children(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(time=1, flags=0)
        tables.nodes.add_row(time=2, flags=0)
        tables.nodes.add_row(time=3, flags=0)
        tables.edges.add_row(0, 1, 2, 1)
        with pytest.raises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_disconnected_subtrees(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(time=4, flags=1)
        tables.nodes.add_row(time=3, flags=1)
        tables.nodes.add_row(time=2, flags=1)
        tables.nodes.add_row(time=1, flags=1)
        tables.edges.add_row(0, 1, 2, 3)
        tables.edges.add_row(0, 1, 0, 1)
        with pytest.raises(ValueError):
            tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_many_mutations(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(time=2, flags=0)
        tables.nodes.add_row(time=1, flags=0)
        tables.edges.add_row(0, 1, 0, 1)
        tables.sites.add_row(position=0.5, ancestral_state="0")
        tables.mutations.add_row(site=0, node=0, derived_state="1")
        tables.mutations.add_row(site=0, node=1, derived_state="0")
        tsinfer.check_ancestors_ts(tables.tree_sequence())

    def test_msprime_output(self, small_ts_fixture):
        with pytest.raises(ValueError):
            tsinfer.check_ancestors_ts(small_ts_fixture)

    def test_tsinfer_output(self, small_sd_fixture):
        ts = tsinfer.infer(small_sd_fixture)
        with pytest.raises(ValueError):
            tsinfer.check_ancestors_ts(ts)


class TestErrors:
    """
    Tests for the error generation code.
    """

    def test_zero_error(self):
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=10, random_seed=1
        )
        assert ts.num_sites > 1
        assert ts.num_trees > 1
        tsp = tsinfer.insert_errors(ts, 0)
        t1 = tsutil.mark_mutation_times_unknown(ts).tables
        t2 = tsp.tables
        assert t1.nodes == t2.nodes
        assert t1.edges == t2.edges
        assert t1.sites == t2.sites
        assert t1.mutations == t2.mutations

    def verify(self, ts, tsp):
        """
        Verifies that the specified tree sequence tsp is correctly
        derived from ts.
        """
        t1 = ts.tables
        t2 = tsp.tables
        assert t1.nodes == t2.nodes
        assert t1.edges == t2.edges
        assert t1.sites == t2.sites
        for site1, site2 in zip(ts.sites(), tsp.sites()):
            mut1 = site1.mutations[0]
            mut2 = site2.mutations[0]
            assert mut1.site == mut2.site
            assert mut1.node == mut2.node
            assert mut1.derived_state == mut2.derived_state
            node = ts.node(mut1.node)
            for mut in site2.mutations[1:]:
                node = ts.node(mut.node)
                assert node.is_sample()
        for v1, v2 in zip(ts.variants(), tsp.variants()):
            diffs = np.sum(v1.genotypes != v2.genotypes)
            assert diffs == len(v2.site.mutations) - 1

    def test_simple_error(self):
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=10, random_seed=2
        )
        assert ts.num_sites > 1
        assert ts.num_trees > 1
        tsp = tsinfer.insert_errors(ts, 0.1)
        # We should have some extra mutations
        assert tsp.num_mutations > ts.num_mutations
        self.verify(ts, tsp)


class TestCli:
    """
    Simple tests for the evaluation CLI to make sure the various tests
    at least run.
    """

    def run_command(self, command):
        with tempfile.TemporaryDirectory(prefix="tsi_eval") as tmpdir:
            subprocess.check_output(
                [sys.executable, "evaluation.py"] + command + ["-d", tmpdir]
            )

    def test_help(self):
        self.run_command(["--help"])

    def test_perfect_inference(self):
        self.run_command(["perfect-inference", "-n", "4", "-l", "0.1", "-s", "1"])

    def test_edges_performance(self):
        self.run_command(
            ["edges-performance", "-n", "5", "-l", "0.1", "-R", "2", "-s", "1"]
        )

    @pytest.mark.slow
    def test_hotspot_analysis(self):
        self.run_command(
            ["hotspot-analysis", "-n", "5", "-l", "0.1", "-R", "1", "-s", "5"]
        )

    def test_ancestor_properties(self):
        self.run_command(
            ["ancestor-properties", "-n", "5", "-l", "0.1", "-R", "1", "-s", "5"]
        )

    @pytest.mark.slow
    def test_ancestor_comparison(self):
        self.run_command(["ancestor-properties", "-n", "5", "-l", "0.5", "-s", "42"])

    @pytest.mark.slow
    def test_node_degree(self):
        self.run_command(["node-degree", "-n", "5", "-l", "0.1", "-s", "42"])

    @pytest.mark.slow
    def test_ancestor_quality(self):
        self.run_command(["ancestor-quality", "-n", "5", "-l", "0.1", "-s", "42"])

    @pytest.mark.slow
    def test_imputation_accuracy(self):
        self.run_command(
            ["imputation-accuracy", "-n", "5", "-l", "0.1", "-s", "40", "-p", "1"]
        )


class TestCountSampleChildEdges:
    """
    Tests the count_sample_child_edges function.
    """

    def verify(self, ts):
        sample_edges = tsinfer.count_sample_child_edges(ts)
        x = np.zeros(ts.num_samples, dtype=sample_edges.dtype)
        for j, node in enumerate(ts.samples()):
            for edge in ts.edges():
                if edge.child == node:
                    x[j] += 1
        assert np.array_equal(x, sample_edges)

    def test_simulated(self):
        ts = msprime.simulate(20, recombination_rate=2, random_seed=2)
        self.verify(ts)

    def test_inferred_no_simplify(self, medium_sd_fixture):
        ts = tsinfer.infer(medium_sd_fixture, simplify=False)
        self.verify(ts)

    def test_inferred_simplify(self, medium_sd_fixture):
        ts = tsinfer.infer(medium_sd_fixture)
        self.verify(ts)


class TestNodeSpan:
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
        assert S1.shape == S2.shape
        assert np.allclose(S1, S2)
        assert np.all(S1 > 0)
        assert np.all(S1 <= ts.sequence_length)
        return S1

    def test_single_locus(self):
        ts = msprime.simulate(10, random_seed=1)
        S = self.verify(ts)
        assert np.all(S == 1)

    def test_single_locus_sequence_length(self):
        ts = msprime.simulate(10, length=100, random_seed=1)
        S = self.verify(ts)
        assert np.all(S == 100)

    def test_simple_recombination(self):
        ts = msprime.simulate(20, recombination_rate=5, random_seed=2)
        assert ts.num_trees > 2
        S = self.verify(ts)
        assert not np.all(S == 1)

    def test_simple_recombination_sequence_length(self):
        ts = msprime.simulate(20, recombination_rate=5, length=10, random_seed=3)
        assert ts.num_trees > 2
        S = self.verify(ts)
        assert not np.all(S == 10)

    @pytest.mark.skip("Broken coincidentally with LS engine update")
    def test_inferred_no_simplify(self, medium_sd_fixture):
        ts = tsinfer.infer(medium_sd_fixture, simplify=False)
        self.verify(ts)

    def test_inferred(self, medium_sd_fixture):
        ts = tsinfer.infer(medium_sd_fixture)
        self.verify(ts)

    def test_inferred_random_data(self):
        np.random.seed(10)
        num_sites = 40
        num_samples = 8
        G = np.random.randint(2, size=(num_sites, num_samples)).astype(np.int8)
        with tsinfer.SampleData() as sample_data:
            for j in range(num_sites):
                sample_data.add_site(j, G[j])
        ts = tsinfer.infer(sample_data)
        self.verify(ts)


class TestMeanSampleAncestry:
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
        tree_iters = [
            ts.trees(tracked_samples=sample_set) for sample_set in sample_sets
        ]
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
        assert A1.shape == A2.shape
        # for tree in ts.trees():
        #     print(tree.interval)
        #     print(tree.draw(format="unicode"))
        # print()
        # for node in ts.nodes():
        #     print(node.id, np.sum(A2[:, node.id]), A2[:, node.id], sep="\t")
        assert np.allclose(A1, A2)
        S = np.sum(A1, axis=0)
        assert np.allclose(S[S != 0], 1)
        return A1

    def two_populations_high_migration_example(self, mutation_rate=10):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(8),
                msprime.PopulationConfiguration(8),
            ],
            migration_matrix=[[0, 1], [1, 0]],
            recombination_rate=3,
            mutation_rate=mutation_rate,
            random_seed=5,
        )
        assert ts.num_trees > 1
        return ts

    def get_random_data_example(self, num_sites, num_samples, seed=100):
        np.random.seed(seed)
        G = np.random.randint(2, size=(num_sites, num_samples)).astype(np.int8)
        with tsinfer.SampleData() as sample_data:
            for j in range(num_sites):
                sample_data.add_site(j, G[j])
        return sample_data

    def test_two_populations_high_migration(self):
        ts = self.two_populations_high_migration_example()
        A = self.verify(ts, [ts.samples(0), ts.samples(1)])
        total = np.sum(A, axis=0)
        assert np.allclose(total[total != 0], 1)

    def test_two_populations_high_migration_no_centromere(self):
        ts = self.two_populations_high_migration_example(mutation_rate=0)
        ts = tsinfer.snip_centromere(ts, 0.4, 0.6)
        # simplify the output to get rid of unreferenced nodes.
        ts = ts.simplify()
        A = self.verify(ts, [ts.samples(0), ts.samples(1)])
        total = np.sum(A, axis=0)
        assert np.allclose(total[total != 0], 1)

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
        assert np.allclose(A1, A2)
        assert np.allclose(S[:u1], 1)
        assert np.all(A1[:, u1] == 0)
        assert np.all(A1[:, u2] == 0)

    def test_two_populations_incomplete_samples(self):
        ts = self.two_populations_high_migration_example()
        samples = ts.samples()
        A = self.verify(ts, [samples[:2], samples[-2:]])
        total = np.sum(A, axis=0)
        assert np.allclose(total[total != 0], 1)

    def test_single_tree_incomplete_samples(self):
        ts = msprime.simulate(10, random_seed=1)
        A = self.verify(ts, [[0, 1], [2, 3]])
        total = np.sum(A, axis=0)
        assert np.allclose(total[total != 0], 1)

    def test_two_populations_overlapping_samples(self):
        ts = self.two_populations_high_migration_example()
        with pytest.raises(ValueError):
            tsinfer.mean_sample_ancestry(ts, [[1], [1]])
        with pytest.raises(ValueError):
            tsinfer.mean_sample_ancestry(ts, [[1, 1], [2]])

    def test_two_populations_high_migration_inferred(self):
        ts = self.two_populations_high_migration_example()
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred_ts = tsinfer.infer(samples)
        assert inferred_ts.num_populations == ts.num_populations
        self.verify(inferred_ts, [inferred_ts.samples(0), inferred_ts.samples(1)])

    def test_two_populations_high_migration_inferred_no_simplify(self):
        ts = self.two_populations_high_migration_example()
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred_ts = tsinfer.infer(samples, simplify=False)
        assert inferred_ts.num_populations == ts.num_populations
        self.verify(inferred_ts, [inferred_ts.samples(0), inferred_ts.samples(1)])

    def test_random_data_inferred(self):
        n = 20
        samples = self.get_random_data_example(num_sites=52, num_samples=n)
        inferred_ts = tsinfer.infer(samples)
        samples = inferred_ts.samples()
        self.verify(inferred_ts, [samples[: n // 2], samples[n // 2 :]])

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
                samples[j * group_size : (j + 1) * group_size]
                for j in range(ts.num_samples // group_size)
            ]
            self.verify(ts, sample_sets)
            group_size *= 2


class TestSnipCentromere:
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
        assert ts1.equals(ts2, ignore_provenance=True)
        tree_found = False
        for tree in ts1.trees():
            if tree.interval == (left, right):
                tree_found = True
                for node in ts1.nodes():
                    assert tree.parent(node.id) == tskit.NULL
                break
        assert tree_found
        return ts1

    def test_single_tree(self):
        ts1 = msprime.simulate(10, random_seed=1)
        ts2 = self.verify(ts1, 0.5, 0.6)
        assert ts2.num_trees == 3

    def test_many_trees(self):
        ts1 = msprime.simulate(10, length=10, recombination_rate=1, random_seed=1)
        assert ts1.num_trees > 2
        self.verify(ts1, 5, 6)

    def get_random_data_example(self, position, num_samples, seed=100):
        np.random.seed(seed)
        G = np.random.randint(2, size=(position.shape[0], num_samples)).astype(np.int8)
        with tsinfer.SampleData() as sample_data:
            for j, x in enumerate(position):
                sample_data.add_site(x, G[j])
        return sample_data

    def test_random_data_inferred_no_simplify(self):
        samples = self.get_random_data_example(
            10 * np.arange(10), num_samples=10, seed=2
        )
        inferred_ts = tsinfer.infer(samples, simplify=False)
        ts = self.verify(inferred_ts, 55, 57)
        assert np.array_equal(ts.genotype_matrix(), inferred_ts.genotype_matrix())

    def test_random_data_inferred_simplify(self):
        samples = self.get_random_data_example(
            5 * np.arange(10), num_samples=10, seed=2
        )
        inferred_ts = tsinfer.infer(samples, simplify=True)
        ts = self.verify(inferred_ts, 12, 15)
        assert np.array_equal(ts.genotype_matrix(), inferred_ts.genotype_matrix())

    def test_coordinate_errors(self):
        ts = msprime.simulate(2, length=10, recombination_rate=1, random_seed=1)
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, -1, 5)
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, 0, 5)
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, 1, 10)
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, 1, 11)
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, 6, 5)
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, 5, 5)

    def test_position_errors(self):
        ts = msprime.simulate(
            2, length=10, recombination_rate=1, random_seed=1, mutation_rate=2
        )
        X = ts.tables.sites.position
        assert X.shape[0] > 3
        # Left cannot be on a site position.
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, X[0], X[0] + 0.001)
        # Cannot go either side of a position
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, X[0] - 0.001, X[0] + 0.001)
        # Cannot cover multiple positions
        with pytest.raises(ValueError):
            tsinfer.snip_centromere(ts, X[0] - 0.001, X[2] + 0.001)

    def test_right_on_position(self):
        ts1 = msprime.simulate(
            2, length=10, recombination_rate=1, random_seed=1, mutation_rate=2
        )
        X = ts1.tables.sites.position
        assert X.shape[0] > 1
        ts2 = self.verify(ts1, X[0] - 0.001, X[0])
        assert np.array_equal(ts1.genotype_matrix(), ts2.genotype_matrix())
