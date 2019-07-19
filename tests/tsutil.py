# MIT License
#
# Copyright (c) 2018-2019 Tskit Developers
# Copyright (C) 2017 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
A collection of utilities to edit and construct tree sequences for testing purposes
"""

import numpy as np
import tskit


def truncate_ts_samples(ts, average_span, random_seed, min_span=5):
    """
    Create a tree sequence that has sample nodes which have been truncated
    so that they span only a small region of the genome. The length of the
    truncated spans is given by a poisson distribution whose mean is average_span
    but which cannot go below a fixed min_span, or above the sequence_length

    Samples are truncated by removing the edges that connect them to the rest
    of the tree.
    """
    np.random.seed(random_seed)
    # Make a list of (left,right) tuples giving the new limits of each sample
    # Keyed by sample ID.
    # for simplicity, we pick lengths from a poisson distribution of av 300 bp
    span = np.random.poisson(average_span, ts.num_samples)
    span = np.maximum(span, min_span)
    span = np.minimum(span, ts.sequence_length)
    start = np.random.uniform(0, ts.sequence_length-span)
    to_slice = {id: (a, b) for id, a, b in zip(ts.samples(), start, start + span)}

    tables = ts.dump_tables()
    tables.edges.clear()
    for e in ts.tables.edges:
        if e.child not in to_slice:
            left, right = e.left, e.right
        else:
            if e.right <= to_slice[e.child][0] or e.left >= to_slice[e.child][1]:
                continue  # this edge is outside the focal region
            else:
                left = max(e.left, to_slice[e.child][0])
                right = min(e.right, to_slice[e.child][1])
        tables.edges.add_row(left, right, e.parent, e.child)
    # Remove mutations above isolated nodes. Remove code below once simplify() does this,
    # see https://github.com/tskit-dev/tskit/issues/260#issuecomment-529573529
    mutations = tables.mutations
    keep_mutations = np.ones((mutations.num_rows, ), dtype=bool)
    positions = tables.sites.position[:]
    for i, m in enumerate(mutations):
        if m.node in to_slice:
            if to_slice[m.node][0] <= positions[m.site] < to_slice[m.node][1]:
                keep_mutations[i] = False
    new_ds, new_ds_offset = tskit.tables.keep_with_offset(
        keep_mutations, mutations.derived_state, mutations.derived_state_offset)
    new_md, new_md_offset = tskit.tables.keep_with_offset(
        keep_mutations, mutations.metadata, mutations.metadata_offset)
    mutations_map = np.append(np.cumsum(keep_mutations) - 1, [-1])
    mutations_map = mutations_map.astype(mutations.parent.dtype)
    # parent -1 always maps to parent -1
    tables.mutations.set_columns(
        site=mutations.site[keep_mutations],
        node=mutations.node[keep_mutations],
        derived_state=new_ds,
        derived_state_offset=new_ds_offset,
        parent=mutations_map[mutations.parent[keep_mutations]],
        metadata=new_md,
        metadata_offset=new_md_offset)
    return tables.tree_sequence().simplify(
        filter_populations=False,
        filter_individuals=False,
        filter_sites=False,
        keep_unary=True)
