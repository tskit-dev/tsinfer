#
# Copyright (C) 2018 University of Oxford
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
Tools for evaluating the algorithm.
"""
import collections
import itertools
import bisect
import random

import numpy as np
import msprime

import tsinfer.inference as inference
import tsinfer.formats as formats


def insert_errors(ts, probability, seed=None):
    """
    Each site has a probability p of generating an error. Errors
    are imposed by choosing a sample and inverting its state with
    a back/recurrent mutation as necessary. Errors resulting in
    fixation of either allele are rejected.
    """
    tables = ts.dump_tables()
    rng = random.Random(seed)
    tables.mutations.clear()
    samples = ts.samples()
    for tree in ts.trees():
        for site in tree.sites():
            assert len(site.mutations) == 1
            mutation_node = site.mutations[0].node
            tables.mutations.add_row(site=site.id, node=mutation_node, derived_state="1")
            for sample in samples:
                # We disallow any fixations. There are two possibilities:
                # (1) We have a singleton and the sample
                # we choose is the mutation node; (2) we have a (n-1)-ton
                # and the sample we choose is on the other root branch.
                if mutation_node == sample:
                    continue
                if {mutation_node, sample} == set(tree.children(tree.root)):
                    continue
                # If the input probability is very high we can still get fixations
                # though by placing a mutation over every sample.
                if rng.random() < probability:
                    # If sample is a descendent of the mutation node we
                    # change the state to 0, otherwise change state to 1.
                    u = sample
                    while u != mutation_node and u != msprime.NULL_NODE:
                        u = tree.parent(u)
                    derived_state = str(int(u == msprime.NULL_NODE))
                    parent = msprime.NULL_MUTATION
                    if u == msprime.NULL_NODE:
                        parent = len(tables.mutations) - 1
                    tables.mutations.add_row(
                        site=site.id, node=sample, parent=parent,
                        derived_state=derived_state)
    return tables.tree_sequence()


def kc_distance(tree1, tree2):
    """
    Returns the Kendall-Colijn topological distance between the specified
    pair of trees. This is a very simple and direct implementation for testing.
    """
    # print(tree1.draw(format="unicode"))
    # print(tree2.draw(format="unicode"))
    samples = tree1.tree_sequence.samples()
    if not np.array_equal(samples, tree2.tree_sequence.samples()):
        raise ValueError("Trees must have the same samples")
    k = samples.shape[0]
    n = (k * (k - 1)) // 2
    M = [np.ones(n + k), np.ones(n + k)]
    for tree_index, tree in enumerate([tree1, tree2]):
        stack = [(tree.root, 0)]
        while len(stack) > 0:
            u, depth = stack.pop()
            children = tree.children(u)
            for v in children:
                stack.append((v, depth + 1))
            for c1, c2 in itertools.combinations(children, 2):
                for u in tree.samples(c1):
                    for v in tree.samples(c2):
                        if u < v:
                            a, b = u, v
                        else:
                            a, b = v, u
                        pair_index = a * (a - 2 * k + 1) // -2 + b - a - 1
                        assert M[tree_index][pair_index] == 1
                        M[tree_index][pair_index] = depth
    return np.linalg.norm(M[0] - M[1])


def tree_pairs(ts1, ts2):
    """
    Returns an iterator over the pairs of trees for each distinct
    interval in the specified pair of tree sequences.
    """
    if ts1.sequence_length != ts2.sequence_length:
        raise ValueError("Tree sequences must be equal length.")
    L = ts1.sequence_length
    trees1 = ts1.trees(sample_lists=True)
    trees2 = ts2.trees(sample_lists=True)
    tree1 = next(trees1)
    tree2 = next(trees2)
    right = 0
    while right != L:
        left = right
        right = min(tree1.interval[1], tree2.interval[1])
        yield (left, right), tree1, tree2
        # Advance
        if tree1.interval[1] == right:
            tree1 = next(trees1, None)
        if tree2.interval[1] == right:
            tree2 = next(trees2, None)


def compare(ts1, ts2):
    """
    Returns the KC distance between the specified tree sequences and
    the intervals over which the trees are compared.
    """
    if not np.array_equal(ts1.samples(), ts2.samples()):
        raise ValueError("Tree sequences must have the same samples")
    breakpoints = [0]
    metrics = []
    for (left, right), tree1, tree2 in tree_pairs(ts1, ts2):
        breakpoints.append(right)
        metrics.append(kc_distance(tree1, tree2))
    return np.array(breakpoints), np.array(metrics)


def strip_singletons(ts):
    """
    Returns a copy of the specified tree sequence with singletons removed.
    """
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    for variant in ts.variants():
        if np.sum(variant.genotypes) > 1:
            site_id = tables.sites.add_row(
                position=variant.site.position,
                ancestral_state=variant.site.ancestral_state,
                metadata=variant.site.metadata)
            assert len(variant.site.mutations) >= 1
            mutation = variant.site.mutations[0]
            parent_id = tables.mutations.add_row(
                site=site_id, node=mutation.node,
                derived_state=mutation.derived_state,
                metadata=mutation.metadata)
            for error in variant.site.mutations[1:]:
                parent = -1
                if error.parent != -1:
                    parent = parent_id
                tables.mutations.add_row(
                    site=site_id, node=error.node, derived_state=error.derived_state,
                    parent=parent, metadata=error.metadata)
    return tables.tree_sequence()


def insert_perfect_mutations(ts, delta=None):
    """
    Returns a copy of the specified tree sequence where the left and right
    coordinates of all edgesets are marked by mutations. This *should* be sufficient
    information to recover the tree sequence exactly.

    This has to be fudged slightly because we cannot have two sites with
    precisely the same coordinates. We work around this by having sites at
    some very small delta from the correct location.
    """
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()

    num_children = np.zeros(ts.num_nodes, dtype=int)
    parent = np.zeros(ts.num_nodes, dtype=int) - 1

    current_delta = 0
    if delta is not None:
        current_delta = delta

    for (left, right), edges_out, edges_in in ts.edge_diffs():
        last_num_children = list(num_children)
        children_in = set()
        children_out = set()
        parents_in = set()
        parents_out = set()
        for e in edges_out:
            # print("out:", e)
            parent[e.child] = -1
            num_children[e.parent] -= 1
            children_out.add(e.child)
            parents_out.add(e.parent)
        for e in edges_in:
            # print("in:", e)
            parent[e.child] = e.parent
            num_children[e.parent] += 1
            children_in.add(e.child)
            parents_in.add(e.parent)
        root = 0
        while parent[root] != -1:
            root = parent[root]
        # If we have more than 4 edges in the diff, or we have a 2 edge diff
        # that is not a root change this must be a multiple recombination.
        if len(edges_out) > 4 or (len(edges_out) == 2 and root not in parents_in):
            raise ValueError("Multiple recombination detected")
        # We use the value of delta from the previous iteration
        x = left - current_delta
        for u in list(children_out - children_in) + list(children_in & children_out):
            if last_num_children[u] > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=u, derived_state="1")
                x -= current_delta

        # Now update delta for this interval.
        if delta is None:
            max_nodes = 2 * (len(children_out) + len(children_in)) + len(parents_in) + 1
            current_delta = (right - left) / max_nodes
        x = left
        for c in list(children_in - children_out) + list(children_in & children_out):
            if num_children[c] > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=c, derived_state="1")
                x += current_delta

        # It seems wrong that we have to mark every parent, since a few of these
        # will already have been marked out by the children.
        for u in parents_in:
            if parent[u] != -1:
                # print("marking in parent", u, "at", x)
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=u, derived_state="1")
                x += current_delta
    tables.sort()
    return tables.tree_sequence()


def get_ancestral_haplotypes(ts):
    """
    Returns a numpy array of the haplotypes of the ancestors in the
    specified tree sequence.
    """
    tables = ts.dump_tables()
    nodes = tables.nodes
    flags = nodes.flags[:]
    flags[:] = 1
    nodes.set_columns(time=nodes.time, flags=flags)

    sites = tables.sites.position
    tsp = tables.tree_sequence()
    B = tsp.genotype_matrix().T

    A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.uint8)
    A[:] = inference.UNKNOWN_ALLELE
    for edge in ts.edges():
        start = bisect.bisect_left(sites, edge.left)
        end = bisect.bisect_right(sites, edge.right)
        if sites[end - 1] == edge.right:
            end -= 1
        A[edge.parent, start:end] = B[edge.parent, start:end]
    A[:ts.num_samples] = B[:ts.num_samples]
    return A


def get_ancestor_descriptors(A):
    """
    Given an array of ancestral haplotypes A in forwards time-order (i.e.,
    so that A[0] == 0), return the descriptors for each ancestor within
    this set and remove any ancestors that do not have any novel mutations.
    Returns the list of ancestors, the start and end site indexes for
    each ancestor, and the list of focal sites for each one.

    This assumes that the input is SMC safe, and will return incorrect
    results on ancestors that contain trapped genetic material.
    """
    L = A.shape[1]
    ancestors = [np.zeros(L, dtype=np.uint8)]
    focal_sites = [[]]
    start = [0]
    end = [L]
    # ancestors = []
    # focal_sites = []
    # start = []
    # end = []
    mask = np.ones(L)
    for a in A:
        masked = np.logical_and(a == 1, mask).astype(int)
        new_sites = np.where(masked)[0]
        mask[new_sites] = 0
        segment = np.where(a != inference.UNKNOWN_ALLELE)[0]
        # Skip any ancestors that are entirely unknown
        if segment.shape[0] > 0:
            s = segment[0]
            e = segment[-1] + 1
            assert np.all(a[s:e] != inference.UNKNOWN_ALLELE)
            assert np.all(a[:s] == inference.UNKNOWN_ALLELE)
            assert np.all(a[e:] == inference.UNKNOWN_ALLELE)
            ancestors.append(a)
            focal_sites.append(new_sites)
            start.append(s)
            end.append(e)
    return np.array(ancestors, dtype=np.uint8), start, end, focal_sites


def assert_smc(ts):
    """
    Check if the specified tree sequence fulfils SMC requirements. This
    means that we cannot have any discontinuous parent segments.
    """
    parent_intervals = collections.defaultdict(list)
    for es in ts.edgesets():
        parent_intervals[es.parent].append((es.left, es.right))
    for intervals in parent_intervals.values():
        if len(intervals) > 0:
            intervals.sort()
            for j in range(1, len(intervals)):
                if intervals[j - 1][1] != intervals[j][0]:
                    raise ValueError("Only SMC simulations are supported")


def assert_single_recombination(ts):
    """
    Check if the specified tree tree sequence contains only single
    recombination events.
    """
    counter = collections.Counter()
    for e in ts.edgesets():
        counter[e.right] += 1
        if e.right != ts.sequence_length and counter[e.right] > 2:
            raise ValueError("Multiple recombinations at ", e.right)


def build_simulated_ancestors(sample_data, ancestor_data, ts, time_chunking=False):
    # Any non-smc tree sequences are rejected.
    assert_smc(ts)
    assert sample_data.num_inference_sites > 0
    A = get_ancestral_haplotypes(ts)
    # This is all nodes, but we only want the non samples. We also reverse
    # the order to make it forwards time.
    A = A[ts.num_samples:][::-1]
    # We also only want the inference sites
    A = A[:, sample_data.sites_inference[:] == 1]

    # get_ancestor_descriptors ensures that the ultimate ancestor is included.
    ancestors, start, end, focal_sites = get_ancestor_descriptors(A)
    N = len(ancestors)
    if time_chunking:
        time = np.zeros(N)
        intersect_mask = np.zeros(A.shape[1], dtype=int)
        t = 0
        for j in range(N):
            if np.any(intersect_mask[start[j]: end[j]] == 1):
                t += 1
                intersect_mask[:] = 0
            intersect_mask[start[j]: end[j]] = 1
            time[j] = t
    else:
        time = np.arange(N)
    time = -1 * (time - time[-1]) + 1
    for a, s, e, focal, t in zip(ancestors, start, end, focal_sites, time):
        assert np.all(a[:s] == inference.UNKNOWN_ALLELE)
        assert np.all(a[s:e] != inference.UNKNOWN_ALLELE)
        assert np.all(a[e:] == inference.UNKNOWN_ALLELE)
        assert all(s <= site < e for site in focal)
        ancestor_data.add_ancestor(
            start=s, end=e, time=t, focal_sites=np.array(focal, dtype=np.int32),
            haplotype=a)


def print_tree_pairs(ts1, ts2, compute_distances=True):
    """
    Prints out the trees at each point in the specified tree sequences,
    alone with their KC distance.
    """
    weighted_distance = 0
    total_mismatch_interval = 0
    total_mismatches = 0
    for (left, right), tree1, tree2 in tree_pairs(ts1, ts2):
        print("-" * 20)
        print("Interval          =", left, "--", right)
        print("Source interval   =", tree1.interval)
        print("Inferred interval =", tree2.interval)
        if compute_distances:
            distance = kc_distance(tree1, tree2)
            weighted_distance += (right - left) * distance
            trailer = ""
            if distance != 0:
                total_mismatch_interval += (right - left)
                total_mismatches += 1
                trailer = "[MISMATCH over {:.2f}]".format(right - left)
            print("KC distance       =", distance, trailer)
        print()
        d1 = tree1.draw(format="unicode").splitlines()
        d2 = tree2.draw(format="unicode").splitlines()
        j = 0
        while j < (min(len(d1), len(d2))):
            print(d1[j], " | ", d2[j])
            j += 1
        while j < len(d1):
            print(d1[j], " |")
            j += 1
        while j < len(d2):
            print(" " * len(d1[0]), " | ", d2[j])
            j += 1
        print()
    print("Total weighted tree distance = ", weighted_distance)
    print("Total mismatch interval      = ", total_mismatch_interval)
    print("Total mismatches             = ", total_mismatches)


def run_perfect_inference(
        base_ts, num_threads=1, path_compression=False,
        extended_checks=True, time_chunking=True, progress_monitor=None,
        engine=inference.C_ENGINE):
    """
    Runs the perfect inference process on the specified tree sequence.
    """
    ts = insert_perfect_mutations(base_ts)
    with formats.SampleData(sequence_length=ts.sequence_length) as sample_data:
        for v in ts.variants():
            sample_data.add_site(v.site.position, v.genotypes, v.alleles)

    ancestor_data = formats.AncestorData(sample_data)
    build_simulated_ancestors(
        sample_data, ancestor_data, ts, time_chunking=time_chunking)
    ancestor_data.finalise()

    ancestors_ts = inference.match_ancestors(
        sample_data, ancestor_data, engine=engine, path_compression=path_compression,
        num_threads=num_threads, extended_checks=extended_checks,
        progress_monitor=progress_monitor)
    # If time_chunking is turned on we need to stabilise the node ordering in the output
    # to ensure that the node IDs are comparable.
    inferred_ts = inference.match_samples(
        sample_data, ancestors_ts, engine=engine, path_compression=path_compression,
        num_threads=num_threads, extended_checks=extended_checks,
        progress_monitor=progress_monitor,
        stabilise_node_ordering=time_chunking and not path_compression)
    return ts, inferred_ts
