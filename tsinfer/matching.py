#
# Copyright (C) 2023 University of Oxford
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
import dataclasses

import numpy as np
import tskit

import _tsinfer


def add_vestigial_root(ts):
    """
    Adds the nodes and edges required by tsinfer to the specified tree sequence
    and returns it.
    """
    if not ts.discrete_genome:
        raise ValueError("Only discrete genome coords supported")
    if ts.num_nodes == 0:
        raise ValueError("Emtpy trees not supported")

    base_tables = ts.dump_tables()
    tables = base_tables.copy()
    tables.nodes.clear()
    t = max(ts.nodes_time)
    tables.nodes.add_row(time=t + 1)
    num_additonal_nodes = 1
    tables.mutations.node += num_additonal_nodes
    tables.edges.child += num_additonal_nodes
    tables.edges.parent += num_additonal_nodes
    for node in base_tables.nodes:
        tables.nodes.append(node)
    if ts.num_edges > 0:
        for tree in ts.trees():
            root = tree.root + num_additonal_nodes
            tables.edges.add_row(
                tree.interval.left, tree.interval.right, parent=0, child=root
            )
        tables.edges.squash()
        # FIXME probably don't need to sort here most of the time, or at least we
        # can just sort almost the end of the table.
        tables.sort()
    return tables.tree_sequence()


class MatcherIndexes(_tsinfer.MatcherIndexes):
    def __init__(self, ts):
        # TODO make this polymorphic to accept tables as well
        # This is very wasteful, but we can do better if it all basically works.
        ts = add_vestigial_root(ts)
        tables = ts.dump_tables()
        ll_tables = _tsinfer.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        # TODO should really just reflect these from the low-level C values.
        self.sequence_length = ts.sequence_length
        self.num_sites = ts.num_sites
        super().__init__(ll_tables)


@dataclasses.dataclass
class Path:
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray

    def __iter__(self):
        yield from zip(self.left, self.right, self.parent)

    def __len__(self):
        return len(self.left)

    def assert_equals(self, other):
        np.testing.assert_array_equal(self.left, other.left)
        np.testing.assert_array_equal(self.right, other.right)
        np.testing.assert_array_equal(self.parent, other.parent)


@dataclasses.dataclass
class Match:
    path: Path
    query_haplotype: np.ndarray
    matched_haplotype: np.ndarray

    def assert_equals(self, other):
        self.path.assert_equals(other.path)
        np.testing.assert_array_equal(self.matched_haplotype, other.matched_haplotype)
        np.testing.assert_array_equal(self.query_haplotype, other.query_haplotype)


class AncestorMatcher2(_tsinfer.AncestorMatcher2):
    def __init__(self, matcher_indexes, **kwargs):
        super().__init__(matcher_indexes, **kwargs)
        self.sequence_length = matcher_indexes.sequence_length
        self.num_sites = matcher_indexes.num_sites

    def zero_sites_path(self):
        left = np.array([0], dtype=np.uint32)
        right = np.array([self.sequence_length], dtype=np.uint32)
        parent = np.array([0], dtype=np.uint32)
        return Match(Path(left, right, parent), [], [])

    def find_match(self, h):
        if self.num_sites == 0:
            return self.zero_sites_path()

        # TODO compute these in C - taking a shortcut for now.
        m = len(h)

        start = 0
        while start < m and h[start] == tskit.MISSING_DATA:
            start += 1
        # if start == m:
        #     raise ValueError("All missing data")
        end = m - 1
        while end >= 0 and h[end] == tskit.MISSING_DATA:
            end -= 1
        end += 1

        path_len, left, right, parent, matched_haplotype = self.find_path(h, start, end)
        left = left[:path_len][::-1]
        right = right[:path_len][::-1]
        parent = parent[:path_len][::-1]
        # We added a 0-root everywhere above, so convert node IDs back
        parent -= 1
        # FIXME C code isn't setting match to missing as expected
        matched_haplotype[:start] = tskit.MISSING_DATA
        matched_haplotype[end:] = tskit.MISSING_DATA
        return Match(Path(left, right, parent), h, matched_haplotype)
