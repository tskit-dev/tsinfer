#
# Copyright (C) 2018-2020 University of Oxford
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
Integrity tests for the low-level module.
"""
import sys

import numpy as np
import pytest

import _tsinfer


IS_WINDOWS = sys.platform == "win32"


class TestOutOfMemory:
    """
    Make sure we raise the correct error when out of memory occurs in
    the library code.
    """

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="windows seems to allow initializing with insane # of nodes"
        " (perhaps memory allocation is optimised out at this stage?)",
    )
    def test_tree_sequence_builder_too_many_nodes(self):
        big = 2**62
        with pytest.raises(MemoryError):
            _tsinfer.TreeSequenceBuilder([2], max_nodes=big)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="windows raises an assert error not a memory error with 2**62 edges"
        " (line 149 of object_heap.c)",
    )
    def test_tree_sequence_builder_too_many_edges(self):
        big = 2**62
        with pytest.raises(MemoryError):
            _tsinfer.TreeSequenceBuilder([2], max_edges=big)


class TestAncestorMatcher:
    """
    Tests for the AncestorMatcher C Python interface.
    """

    def test_init(self):
        with pytest.raises(TypeError):
            _tsinfer.AncestorMatcher()
        with pytest.raises(TypeError):
            _tsinfer.AncestorMatcher(None)
        tsb = _tsinfer.TreeSequenceBuilder([2])
        with pytest.raises(TypeError):
            _tsinfer.AncestorMatcher(tsb)
        with pytest.raises(TypeError):
            _tsinfer.AncestorMatcher(tsb, [1])
        for bad_type in [None, {}]:
            with pytest.raises(TypeError):
                _tsinfer.AncestorMatcher(tsb, [1], [1], extended_checks=bad_type)
            with pytest.raises(TypeError):
                _tsinfer.AncestorMatcher(tsb, [1], [1], precision=bad_type)
        for bad_array in [[], [[], []], None, "sdf", [1, 2, 3]]:
            with pytest.raises(ValueError):
                _tsinfer.AncestorMatcher(tsb, bad_array, [1])
            with pytest.raises(ValueError):
                _tsinfer.AncestorMatcher(tsb, [1], bad_array)

    def test_weight_by_n_changes_path(self):
        # 4 sites (each with 2 alleles). Topology:
        #   0 (root)
        #   └── 1 over [0,4)
        #       ├── 2 over [0,2)
        #       └── 3 over [2,4)
        # Mutations set node 2 to derived state 1 at sites 0,1 and node 3 at 2,3.
        # With rho=0.01 and mu=0.2, disabling weight-by-n prefers recombination at
        # the boundary (two segments), while enabling weight-by-n prefers a single
        # segment (stay with parent 1).
        tsb = _tsinfer.TreeSequenceBuilder([2, 2, 2, 2])
        n0 = tsb.add_node(time=3)
        n1 = tsb.add_node(time=2)
        n2 = tsb.add_node(time=1)
        n3 = tsb.add_node(time=1)
        tsb.add_path(child=n1, left=[0], right=[4], parent=[n0])
        tsb.add_path(child=n2, left=[0], right=[2], parent=[n1])
        tsb.add_path(child=n3, left=[2], right=[4], parent=[n1])
        tsb.add_mutations(node=n2, site=[0, 1], derived_state=[1, 1])
        tsb.add_mutations(node=n3, site=[2, 3], derived_state=[1, 1])
        tsb.freeze_indexes()

        rho = 0.01
        mu = 0.2  # chosen so behaviour differs between n=1 vs n=2
        recomb = [rho] * 4
        mismatch = [mu] * 4
        h = np.array([1, 1, 1, 1], dtype=np.int8)
        match = np.zeros(4, dtype=np.int8)

        # Weighting by n enabled (default): expect to stick with ancestor a0
        m_weight = _tsinfer.AncestorMatcher(tsb, recomb, mismatch, weight_by_n=1)
        left_w, right_w, parent_w = m_weight.find_path(h, 0, 4, match)
        assert len(parent_w) == 1
        assert parent_w[0] == n1
        assert left_w[0] == 0 and right_w[0] == 4

        # Weighting by n disabled (n=1): expect recombination at boundary 3
        m_no_weight = _tsinfer.AncestorMatcher(tsb, recomb, mismatch, weight_by_n=0)
        left_d, right_d, parent_d = m_no_weight.find_path(h, 0, 4, match)
        assert list(parent_d) == [n3, n2]
        assert list(left_d) == [2, 0]
        assert list(right_d) == [4, 2]


class TestTreeSequenceBuilder:
    """
    Tests for the AncestorMatcher C Python interface.
    """

    def test_init(self):
        with pytest.raises(TypeError):
            _tsinfer.TreeSequenceBuilder()
        for bad_array in [None, "serf", [[], []], ["asdf"], {}]:
            with pytest.raises(ValueError):
                _tsinfer.TreeSequenceBuilder(bad_array)

        for bad_type in [None, "sdf", {}]:
            with pytest.raises(TypeError):
                _tsinfer.TreeSequenceBuilder([2], max_nodes=bad_type)
            with pytest.raises(TypeError):
                _tsinfer.TreeSequenceBuilder([2], max_edges=bad_type)

    def test_ancestral_state(self):
        _tsinfer.TreeSequenceBuilder([2, 3])

        # Test valid ancestral_state
        _tsinfer.TreeSequenceBuilder(
            [2, 3], ancestral_state=np.array([0, 1], dtype=np.int8)
        )

        # Test wrong size
        with pytest.raises(ValueError, match="ancestral_state array wrong size"):
            _tsinfer.TreeSequenceBuilder(
                [2, 3], ancestral_state=np.array([0], dtype=np.int8)
            )

        # Test negative ancestral state
        with pytest.raises(_tsinfer.LibraryError, match="Ancestral state must be >= 0"):
            _tsinfer.TreeSequenceBuilder(
                [2, 3], ancestral_state=np.array([-1, 0], dtype=np.int8)
            )

        # Test ancestral state >= num_alleles
        with pytest.raises(_tsinfer.LibraryError, match="Ancestral state must be >= 0"):
            _tsinfer.TreeSequenceBuilder(
                [2, 3], ancestral_state=np.array([0, 3], dtype=np.int8)
            )

    def test_ancestral_state_matching(self):
        # Build two identical tree structures but with different ancestral states
        # and verify the allelic states of nodes differ based on ancestral state

        # Tree 1: ancestral state = 0, node n1 has mutation to state 1
        tsb0 = _tsinfer.TreeSequenceBuilder(
            [2, 2], ancestral_state=np.array([0, 0], dtype=np.int8)
        )
        root0 = tsb0.add_node(time=2)
        n0 = tsb0.add_node(time=1)
        tsb0.add_path(child=n0, left=[0], right=[2], parent=[root0])
        tsb0.add_mutations(node=n0, site=[0, 1], derived_state=[1, 1])
        tsb0.freeze_indexes()

        # Tree 2: ancestral state = 1, node n1 has mutation to state 0
        # This creates the mirror image: root has state 1, n1 has state 0
        tsb1 = _tsinfer.TreeSequenceBuilder(
            [2, 2], ancestral_state=np.array([1, 1], dtype=np.int8)
        )
        root1 = tsb1.add_node(time=2)
        n1 = tsb1.add_node(time=1)
        tsb1.add_path(child=n1, left=[0], right=[2], parent=[root1])
        tsb1.add_mutations(node=n1, site=[0, 1], derived_state=[0, 0])
        tsb1.freeze_indexes()

        recomb = [0.01, 0.01]
        mismatch = [0.1, 0.1]

        # In tree 0: root0 has state [0,0], n0 has state [1,1]
        # In tree 1: root1 has state [1,1], n1 has state [0,0]
        # Query [0,0] should have different match scores to root vs child

        h_zero = np.array([0, 0], dtype=np.int8)
        h_one = np.array([1, 1], dtype=np.int8)
        match = np.zeros(2, dtype=np.int8)

        # In tsb0, root has [0,0] (ancestral), n0 has [1,1] (derived)
        matcher0 = _tsinfer.AncestorMatcher(tsb0, recomb, mismatch)
        # Query [1,1] should match n0 better than root
        _, _, parent_tsb0_one = matcher0.find_path(h_one, 0, 2, match)
        assert parent_tsb0_one[0] == n0

        # In tsb1, root has [1,1] (ancestral), n1 has [0,0] (derived)
        matcher1 = _tsinfer.AncestorMatcher(tsb1, recomb, mismatch)
        # Query [0,0] should match n1 better than root
        _, _, parent_tsb1_zero = matcher1.find_path(h_zero, 0, 2, match)
        assert parent_tsb1_zero[0] == n1


class TestAncestorBuilder:
    """
    Tests for the AncestorBuilder C Python interface.
    """

    def test_init(self):
        with pytest.raises(TypeError):
            _tsinfer.AncestorBuilder()
        for bad_value in [None, "serf", [[], []], ["asdf"], {}]:
            with pytest.raises(TypeError):
                _tsinfer.AncestorBuilder(num_samples=2, max_sites=bad_value)
            with pytest.raises(TypeError):
                _tsinfer.AncestorBuilder(num_samples=bad_value, max_sites=2)
            with pytest.raises(TypeError):
                _tsinfer.AncestorBuilder(
                    num_samples=2, max_sites=2, genotype_encoding=bad_value
                )
            with pytest.raises(TypeError):
                _tsinfer.AncestorBuilder(num_samples=2, max_sites=2, mmap_fd=bad_value)
        for bad_num_samples in [0, 1]:
            with pytest.raises(_tsinfer.LibraryError):
                _tsinfer.AncestorBuilder(num_samples=bad_num_samples, max_sites=0)

    @pytest.mark.skipif(IS_WINDOWS, reason="mmap_fd is a no-op on Windows")
    def test_bad_fd(self):
        with pytest.raises(_tsinfer.LibraryError, match="Bad file desc"):
            _tsinfer.AncestorBuilder(num_samples=2, max_sites=2, mmap_fd=-2)

    def test_add_site(self):
        ab = _tsinfer.AncestorBuilder(num_samples=2, max_sites=10)
        for bad_type in ["sdf", {}, None]:
            with pytest.raises(TypeError):
                ab.add_site(time=bad_type, genotypes=[0, 0])
        for bad_genotypes in ["asdf", [[], []], [0, 1, 2]]:
            with pytest.raises(ValueError):
                ab.add_site(time=0, genotypes=bad_genotypes)

    def test_add_too_many_sites(self):
        for max_sites in range(10):
            ab = _tsinfer.AncestorBuilder(num_samples=2, max_sites=max_sites)
            for _ in range(max_sites):
                ab.add_site(time=1, genotypes=[0, 1])
            for _ in range(2 * max_sites):
                with pytest.raises(_tsinfer.LibraryError) as record:
                    ab.add_site(time=1, genotypes=[0, 1])
                msg = "Cannot add more sites than the specified maximum."
                assert str(record.value) == msg

    # TODO need tester methods for the remaining methonds in the class.
