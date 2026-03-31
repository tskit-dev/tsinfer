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
Operations on ARG (Ancestral Recombination Graph) topology.
"""

import logging

logger = logging.getLogger(__name__)


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
            # if tree.num_roots > 1:
            #     print(ts.draw_text())
            root = tree.root + num_additonal_nodes
            tables.edges.add_row(
                tree.interval.left, tree.interval.right, parent=0, child=root
            )
        tables.edges.squash()
        # FIXME probably don't need to sort here most of the time, or at least
        # we can just sort almost the end of the table.
        tables.sort()
    return tables.tree_sequence()
