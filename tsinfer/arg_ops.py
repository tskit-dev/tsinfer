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

import collections
import logging
import time as time_

logger = logging.getLogger(__name__)

PC_ANCESTOR_INCREMENT = 1.0 / (1 << 32)
NODE_IS_PC_ANCESTOR = 1 << 16


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


def is_pc_ancestor(flags):
    """
    Returns True if the node flags indicate a path compression ancestor.
    """
    return (flags & NODE_IS_PC_ANCESTOR) != 0


def compress_paths(ts):
    """
    Find groups of child nodes whose entire edge sets are identical
    (same set of (left, right, parent) tuples) and create an intermediate
    path compression ancestor node for each group.

    For each group of 2+ children with identical paths, a new node is
    created with time slightly less than the minimum parent time in the
    path. The children's edges are rewritten to go through the new node.
    """
    start_time = time_.time()
    tables = ts.dump_tables()
    node_time = tables.nodes.time
    original_num_nodes = ts.num_nodes
    original_num_edges = ts.num_edges

    # Build path key for each child: the full sorted tuple of its edges
    edges_by_child = collections.defaultdict(list)
    for edge in tables.edges:
        edges_by_child[edge.child].append((edge.left, edge.right, edge.parent))

    path_map = collections.defaultdict(list)
    for child, edges in edges_by_child.items():
        key = tuple(sorted(edges))
        path_map[key].append(child)

    # Groups with 2+ children sharing identical paths get a PC ancestor
    compressed_children = set()
    new_edges = []
    num_pc_nodes = 0
    for path, children in path_map.items():
        if len(children) < 2:
            continue

        # Compute PC ancestor time
        min_parent_time = min(node_time[parent] for _, _, parent in path)
        pc_time = min_parent_time - PC_ANCESTOR_INCREMENT
        max_child_time = max(node_time[child] for child in children)
        if pc_time <= max_child_time:
            logger.debug(
                "Skipping path compression group: computed time %f "
                "is not greater than maximum child time %f",
                pc_time,
                max_child_time,
            )
            continue

        # Create PC ancestor node
        pc_node = tables.nodes.add_row(flags=NODE_IS_PC_ANCESTOR, time=pc_time)
        num_pc_nodes += 1

        # Add edges from each original parent to the PC node
        for left, right, parent in path:
            new_edges.append((left, right, parent, pc_node))
        # Add edges from PC node to each child
        for child in children:
            for left, right, _ in path:
                new_edges.append((left, right, pc_node, child))
        compressed_children.update(children)

    # Rebuild edge table: drop edges for compressed children, add new ones
    tables.edges.clear()
    for edge in ts.edges():
        if edge.child not in compressed_children:
            tables.edges.add_row(
                left=edge.left,
                right=edge.right,
                parent=edge.parent,
                child=edge.child,
            )
    for left, right, parent, child in new_edges:
        tables.edges.add_row(left=left, right=right, parent=parent, child=child)

    tables.sort()
    result = tables.tree_sequence()
    elapsed = time_.time() - start_time
    logger.info(
        "Path compression: %d PC nodes added, %d children compressed, "
        "edges %d -> %d, nodes %d -> %d (%.2fs)",
        num_pc_nodes,
        len(compressed_children),
        original_num_edges,
        result.num_edges,
        original_num_nodes,
        result.num_nodes,
        elapsed,
    )
    return result
