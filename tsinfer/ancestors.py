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
"""
Ancestor handling routines.
"""
import logging
import time as time_

import numba
import numpy as np

logger = logging.getLogger(__name__)


def merge_overlapping_ancestors(start, end, time):
    # Merge overlapping, same-time ancestors. We do this by scanning along a single
    # time epoch from left to right, detecting breaks.
    sort_indices = np.lexsort((start, time))
    start = start[sort_indices]
    end = end[sort_indices]
    time = time[sort_indices]
    old_indexes = {}
    # For efficiency, pre-allocate the output arrays to the maximum possible size.
    new_start = np.full_like(start, -1)
    new_end = np.full_like(end, -1)
    new_time = np.full_like(time, -1)

    i = 0
    new_index_pos = 0
    while i < len(start):
        j = i + 1
        group_overlap = [i]
        max_right = end[i]
        # While we're in the same time epoch, and the next ancestor
        # overlaps with the group, add this ancestor to the group.
        while j < len(start) and time[j] == time[i] and start[j] < max_right:
            max_right = max(max_right, end[j])
            group_overlap.append(j)
            j += 1

        # Emit the found group
        old_indexes[new_index_pos] = group_overlap
        new_start[new_index_pos] = start[i]
        new_end[new_index_pos] = max_right
        new_time[new_index_pos] = time[i]
        new_index_pos += 1
        i = j
    # Trim the output arrays to the actual size.
    new_start = new_start[:new_index_pos]
    new_end = new_end[:new_index_pos]
    new_time = new_time[:new_index_pos]
    return new_start, new_end, new_time, old_indexes, sort_indices


@numba.njit
def run_linesweep(event_times, event_index, event_type, new_time):
    # Run the linesweep over the ancestor start-stop events,
    # building up the dependency graph as a count of dependencies for each ancestor,
    # and a list of dependant children for each ancestor.
    n = len(new_time)

    # numba really likes to know the type of the list elements, so we tell it by adding
    # a dummy element to the list and then popping it off.
    # `active` is the list of ancestors that overlap with the current linesweep position.
    active = [-1]
    active.pop()
    children = [[-1] for _ in range(n)]
    for c in range(n):
        children[c].pop()
    incoming_edge_count = np.zeros(n, dtype=np.int32)
    for i in range(len(event_times)):
        index = event_index[i]
        e_time = event_times[i]
        if event_type[i] == 1:
            for j in active:
                if new_time[j] > e_time:
                    incoming_edge_count[index] += 1
                    children[j].append(index)
                elif new_time[j] < e_time:
                    incoming_edge_count[j] += 1
                    children[index].append(j)
            active.append(index)
        else:
            active.remove(index)

    # Convert children to ragged array format so we can pass arrays to the
    # next numba function, `find_groups`.
    children_data = []
    children_indices = [0]
    for child_list in children:
        children_data.extend(child_list)
        children_indices.append(len(children_data))
    children_data = np.array(children_data, dtype=np.int32)
    children_indices = np.array(children_indices, dtype=np.int32)
    return children_data, children_indices, incoming_edge_count


@numba.njit
def find_groups(children_data, children_indices, incoming_edge_count):
    # We find groups of ancestors that can be matched in parallel by topologically
    # sorting the dependency graph. We do this by deconstructing the graph, removing
    # nodes with no incoming edges, and adding them to a group.
    n = len(children_indices) - 1
    group_id = np.full(n, -1, dtype=np.int32)
    current_group = 0
    while True:
        # Find the nodes with no incoming edges
        no_incoming = np.where(incoming_edge_count == 0)[0]
        if len(no_incoming) == 0:
            break
        # Remove them from the graph
        for i in no_incoming:
            incoming_edge_count[i] = -1
            incoming_edge_count[
                children_data[children_indices[i] : children_indices[i + 1]]
            ] -= 1
        # Add them to the group
        group_id[no_incoming] = current_group
        current_group += 1
    return group_id


def group_ancestors_by_linesweep(start, end, time):
    # For a given set of ancestors, we want to group them for matching in parallel.
    # For each ancestor, any overlapping, older ancestors must be in an earlier group,
    # and any overlapping, younger ancestors in a later group. Any overlapping same-age
    # ancestors must be in the same group so they don't match to each other.
    # We do this by first merging the overlapping same-age ancestors. Then build a
    # dependency graph of the ancestors by linesweep. Then form groups by topological
    # sort. Finally, we un-merge the same-age ancestors.

    assert len(start) == len(end)
    assert len(start) == len(time)
    t = time_.time()
    (
        new_start,
        new_end,
        new_time,
        old_indexes,
        sort_indices,
    ) = merge_overlapping_ancestors(start, end, time)
    logger.info(f"Merged to {len(new_start)} ancestors in {time_.time() - t:.2f}s")

    # Build a list of events for the linesweep
    t = time_.time()
    n = len(new_time)
    # Create events arrays by copying and concatenating inputs
    event_times = np.concatenate([new_time, new_time])
    event_pos = np.concatenate([new_start, new_end])
    event_index = np.concatenate([np.arange(n), np.arange(n)])
    event_type = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    # Sort events by position, then ends before starts
    event_sort_indices = np.lexsort((event_type, event_pos))
    event_times = event_times[event_sort_indices]
    event_index = event_index[event_sort_indices]
    event_type = event_type[event_sort_indices]
    logger.info(f"Built {len(event_times)} events in {time_.time() - t:.2f}s")

    t = time_.time()
    children_data, children_indices, incoming_edge_count = run_linesweep(
        event_times, event_index, event_type, new_time
    )
    logger.info(
        f"Linesweep generated {np.sum(incoming_edge_count)} dependencies in"
        f" {time_.time() - t:.2f}s"
    )

    t = time_.time()
    group_id = find_groups(children_data, children_indices, incoming_edge_count)
    logger.info(f"Found groups in {time_.time() - t:.2f}s")

    t = time_.time()
    # Convert the group id array to lists of ids for each group
    ancestor_grouping = {}
    for group in np.unique(group_id):
        ancestor_grouping[group] = np.where(group_id == group)[0]

    # Now un-merge the same-age ancestors, simultaneously mapping back to the original,
    # unsorted indexes
    for group in ancestor_grouping:
        ancestor_grouping[group] = sorted(
            [
                sort_indices[item]
                for i in ancestor_grouping[group]
                for item in old_indexes[i]
            ]
        )
    logger.info(f"Un-merged in {time_.time() - t:.2f}s")
    logger.info(
        f"{len(ancestor_grouping)} groups with median size "
        f"{np.median([len(ancestor_grouping[group]) for group in ancestor_grouping])}"
    )
    return ancestor_grouping
