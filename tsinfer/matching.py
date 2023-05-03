#
# Copyright (C) 2018-2023 University of Oxford
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
import _tsinfer


class MatcherIndexes(_tsinfer.MatcherIndexes):
    def __init__(self, ts):
        # TODO make this polymorphic to accept tables as well
        tables = ts.dump_tables()
        ll_tables = _tsinfer.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        super().__init__(ll_tables)


# TODO add the high-level classes fronting the other class
