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
import unittest

import _tsinfer


class TestOutOfMemory(unittest.TestCase):
    """
    Make sure we raise the correct error when out of memory occurs in
    the library code.
    """
    @unittest.skipIf(sys.platform == "win32",
                     "windows seems to allow initializing with insane # of nodes"
                     " (perhaps memory allocation is optimised out at this stage?)")
    def test_tree_sequence_builder_too_many_nodes(self):
        big = 2**62
        self.assertRaises(
            MemoryError, _tsinfer.TreeSequenceBuilder, [2], max_nodes=big)

    @unittest.skipIf(sys.platform == "win32",
                     "windows raises an assert error not a memory error with 2**62 edges"
                     " (line 149 of object_heap.c)")
    def test_tree_sequence_builder_too_many_edges(self):
        big = 2**62
        self.assertRaises(
            MemoryError, _tsinfer.TreeSequenceBuilder, [2], max_edges=big)


class TestAncestorMatcher(unittest.TestCase):
    """
    Tests for the AncestorMatcher C Python interface.
    """
    def test_init(self):
        self.assertRaises(TypeError, _tsinfer.AncestorMatcher)
        self.assertRaises(TypeError, _tsinfer.AncestorMatcher, None)
        tsb = _tsinfer.TreeSequenceBuilder([2])
        self.assertRaises(TypeError, _tsinfer.AncestorMatcher, tsb)
        self.assertRaises(TypeError, _tsinfer.AncestorMatcher, tsb, [1])
        for bad_type in [None, {}]:
            self.assertRaises(
                TypeError, _tsinfer.AncestorMatcher, tsb, [1], [1],
                extended_checks=bad_type)
            self.assertRaises(
                TypeError, _tsinfer.AncestorMatcher, tsb, [1], [1],
                precision=bad_type)
        for bad_array in [[], [[], []], None, "sdf", [1, 2, 3]]:
            with self.assertRaises(ValueError):
                _tsinfer.AncestorMatcher(tsb, bad_array, [1])
            with self.assertRaises(ValueError):
                _tsinfer.AncestorMatcher(tsb, [1], bad_array)


class TestTreeSequenceBuilder(unittest.TestCase):
    """
    Tests for the AncestorMatcher C Python interface.
    """
    def test_init(self):
        self.assertRaises(TypeError, _tsinfer.TreeSequenceBuilder)
        for bad_array in [None, "serf", [[], []], ["asdf"], {}]:
            with self.assertRaises(ValueError):
                _tsinfer.TreeSequenceBuilder(bad_array)

        for bad_type in [None, "sdf", {}]:
            with self.assertRaises(TypeError):
                _tsinfer.TreeSequenceBuilder([2], max_nodes=bad_type)
            with self.assertRaises(TypeError):
                _tsinfer.TreeSequenceBuilder([2], max_edges=bad_type)
