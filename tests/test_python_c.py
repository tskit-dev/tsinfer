#
# Copyright (C) 2018-2026 University of Oxford
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

import inspect
import sys

import numpy as np
import pytest
import tskit

import _tsinfer
from tsinfer import matching

IS_WINDOWS = sys.platform == "win32"


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

    def test_make_ancestor(self):
        ab = _tsinfer.AncestorBuilder(num_samples=2, max_sites=2)
        ab.add_site(time=1, genotypes=[0, 1])
        ab.add_site(time=2, genotypes=[1, 0])
        for _, focal_sites in ab.ancestor_descriptors():
            a = np.zeros(2, dtype=np.int8)
            start, end = ab.make_ancestor(focal_sites, a)
            assert start == 0
            assert end == 2
            assert np.all(a[:2] >= 0)

    def test_getters(self):
        ab = _tsinfer.AncestorBuilder(num_samples=2, max_sites=2)
        ab.add_site(time=1, genotypes=[0, 1])
        ab.add_site(time=2, genotypes=[1, 0])
        A = list(ab.ancestor_descriptors())
        assert ab.num_sites == 2
        assert ab.num_ancestors == len(A)
        assert ab.mem_size > 0

    def test_make_ancestor_errors(self):
        ab = _tsinfer.AncestorBuilder(num_samples=2, max_sites=1)
        ab.add_site(time=1, genotypes=[0, 1])
        a = np.zeros(1, dtype=np.int8)
        with pytest.raises(TypeError):
            ab.make_ancestor()
        with pytest.raises(ValueError, match="num_focal_sites must > 0"):
            ab.make_ancestor([], a)
        with pytest.raises(ValueError, match="x"):
            ab.make_ancestor(["x"], a)
        with pytest.raises(ValueError, match="Dim != 1"):
            ab.make_ancestor([[0, 1]], a)
        with pytest.raises(TypeError, match="Cannot cast"):
            ab.make_ancestor([0], np.array(["type"]))
        with pytest.raises(ValueError, match="wrong size"):
            ab.make_ancestor([0], np.zeros(100, dtype=np.int8))


class TestUninitialised:
    def test_uninitialised(self):
        for _, cls in inspect.getmembers(_tsinfer):
            if (
                isinstance(cls, type)
                and not issubclass(cls, Exception)
                and not issubclass(cls, tuple)
            ):
                methods = []
                attributes = []
                for name, value in inspect.getmembers(cls):
                    if not name.startswith("__"):
                        if inspect.isdatadescriptor(value):
                            attributes.append(name)
                        else:
                            methods.append(name)
                uninitialised = cls.__new__(cls)
                for attr in attributes:
                    with pytest.raises((SystemError, ValueError)):
                        getattr(uninitialised, attr)
                    with pytest.raises((SystemError, ValueError, AttributeError)):
                        setattr(uninitialised, attr, None)
                for method_name in methods:
                    method = getattr(uninitialised, method_name)
                    with pytest.raises((SystemError, ValueError)):
                        method()


def make_matcher_indexes_and_matcher(num_samples=4):
    """Build a MatcherIndexes and AncestorMatcher from a small tree sequence."""
    tables = tskit.Tree.generate_balanced(num_samples).tree_sequence.dump_tables()
    tables.sequence_length = 2
    tables.edges.right = np.full(len(tables.edges), 2, dtype=np.float64)
    tables.sites.add_row(position=1, ancestral_state="A")
    tables.mutations.add_row(site=0, node=1, derived_state="T")
    ts = tables.tree_sequence()
    ts = matching.add_vestigial_root(ts)
    ll_tables = _tsinfer.LightweightTableCollection(ts.sequence_length)
    ll_tables.fromdict(ts.dump_tables().asdict())
    mi = _tsinfer.MatcherIndexes(ll_tables)
    recombination = np.array([1e-9])
    mismatch = np.array([0.0])
    am = _tsinfer.AncestorMatcher(mi, recombination=recombination, mismatch=mismatch)
    return ts, mi, am


class TestAncestorMatcher:
    def test_total_memory_before_find_path(self):
        _, _, am = make_matcher_indexes_and_matcher()
        mem = am.total_memory
        if IS_WINDOWS:
            assert mem >= 2**31
        else:
            assert isinstance(mem, int)
            # Before find_path, only the initial block is allocated
            assert mem > 0

    def test_total_memory_after_find_path(self):
        ts, _, am = make_matcher_indexes_and_matcher()
        h = np.zeros(ts.num_sites, dtype=np.int8)
        am.find_path(h, 0, ts.num_sites)
        mem = am.total_memory
        if IS_WINDOWS:
            assert mem >= 2**31
        else:
            assert isinstance(mem, int)
            assert mem > 0

    def test_mean_traceback_size(self):
        ts, _, am = make_matcher_indexes_and_matcher()
        h = np.zeros(ts.num_sites, dtype=np.int8)
        am.find_path(h, 0, ts.num_sites)
        tb = am.mean_traceback_size
        assert isinstance(tb, float)
        assert tb >= 0


class TestMatcherIndexes:
    def test_single_tree(self):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        tables = ts.dump_tables()
        ll_tables = _tsinfer.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        _ = _tsinfer.MatcherIndexes(ll_tables)

    def test_print_state(self, tmpdir):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        tables = ts.dump_tables()
        ll_tables = _tsinfer.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        mi = _tsinfer.MatcherIndexes(ll_tables)
        with pytest.raises(TypeError):
            mi.print_state()

        path = tmpdir / "output.txt"
        with open(path, "w") as f:
            mi.print_state(f)
        with open(path) as f:
            output = f.read()
        assert len(output) > 0
        assert "indexes" in output
