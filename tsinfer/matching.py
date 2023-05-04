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


class MatcherIndexes(_tsinfer.MatcherIndexes):
    def __init__(self, ts):
        # TODO make this polymorphic to accept tables as well
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

        left = left[:path_len]
        right = right[:path_len]
        parent = parent[:path_len]
        return Match(Path(left, right, parent), matched_haplotype)
