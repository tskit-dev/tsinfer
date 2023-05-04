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

import _tsinfer


def add_vestigial_root(ts):
    """
    Adds the nodes and edges required by tsinfer to the specified tree sequence
    and returns it.
    """
    if not ts.discrete_genome:
        raise ValueError("Only discrete genome coords supported")

    base_tables = ts.dump_tables()
    tables = base_tables.copy()
    tables.nodes.clear()
    t = ts.max_root_time
    tables.nodes.add_row(time=t + 1)
    num_additonal_nodes = len(tables.nodes)
    tables.mutations.node += num_additonal_nodes
    tables.edges.child += num_additonal_nodes
    tables.edges.parent += num_additonal_nodes
    for node in base_tables.nodes:
        tables.nodes.append(node)
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
        super().__init__(ll_tables)


@dataclasses.dataclass
class Path:
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray

    def __len__(self):
        return len(self.left)

    def assert_equals(self, other):
        np.testing.assert_array_equal(self.left, other.left)
        np.testing.assert_array_equal(self.right, other.right)
        np.testing.assert_array_equal(self.parent, other.parent)


@dataclasses.dataclass
class Match:
    path: Path
    matched_haplotype: np.ndarray

    def assert_equals(self, other):
        self.path.assert_equals(other.path)
        np.testing.assert_array_equal(self.matched_haplotype, other.matched_haplotype)


class AncestorMatcher2(_tsinfer.AncestorMatcher2):
    def find_match(self, h, left, right):
        path_len, left, right, parent, matched_haplotype = self.find_path(
            h, left, right
        )
        left = left[:path_len][::-1]
        right = right[:path_len][::-1]
        parent = parent[:path_len][::-1]
        # We added a 0-root everywhere above, so convert node IDs back
        parent -= 1
        return Match(Path(left, right, parent), matched_haplotype)
