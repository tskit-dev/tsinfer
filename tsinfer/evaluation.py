"""
Tools for evaluating the algorithm.
"""
import collections
import itertools
import warnings

import numpy as np
import msprime

import tsinfer.inference as inference
import tsinfer.formats as formats


def kc_distance(tree1, tree2):
    """
    Returns the Kendall-Colijn topological distance between the specified
    pair of trees. Note that this does not include the branch length component.
    """
    samples = tree1.tree_sequence.samples()
    if not np.array_equal(samples, tree2.tree_sequence.samples()):
        raise ValueError("Trees must have the same samples")
    k = samples.shape[0]
    n = (k * (k - 1)) // 2
    trees = [tree1, tree2]
    M = [np.ones(n + k), np.ones(n + k)]
    D = [{}, {}]
    for j, (a, b) in enumerate(itertools.combinations(samples, 2)):
        for tree, m, d in zip(trees, M, D):
            u = tree.mrca(a, b)
            if u not in d:
                # Cache the distance values
                path_len = 0
                v = u
                while tree.parent(v) != msprime.NULL_NODE:
                    path_len += 1
                    v = tree.parent(v)
                d[u] = path_len
            m[j] = d[u]
    return np.linalg.norm(M[0] - M[1])


def tree_pairs(ts1, ts2):
    """
    Returns an iterator over the pairs of trees for each distinct
    interval in the specified pair of tree sequences.
    """
    if ts1.sequence_length != ts2.sequence_length:
        raise ValueError("Tree sequences must be equal length.")
    L = ts1.sequence_length
    trees1 = ts1.trees()
    trees2 = ts2.trees()
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
    sites = msprime.SiteTable()
    mutations = msprime.MutationTable()
    for variant in ts.variants():
        if np.sum(variant.genotypes) > 1:
            site_id = sites.add_row(
                position=variant.site.position,
                ancestral_state=variant.site.ancestral_state)
            for mutation in variant.site.mutations:
                assert mutation.parent == -1  # No back mutations
                mutations.add_row(
                    site=site_id, node=mutation.node, derived_state=mutation.derived_state)
    tables = ts.dump_tables()
    return msprime.load_tables(
        nodes=tables.nodes, edges=tables.edges, sites=sites, mutations=mutations)


def insert_perfect_mutations(ts, delta=1/64):
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
    # Edges are sorted by time.
    left_map = collections.defaultdict(list)
    right_map = collections.defaultdict(list)
    for e in ts.edges():
        left_map[e.left].append(e.child)
        right_map[e.right].append(e.child)

    for t in ts.trees():
        left, right = t.interval
        x = left
        for node in left_map[left]:
            if t.num_samples(node) > 1:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=node, derived_state="1")
                x += delta
        x = right - delta
        for node in reversed(right_map[right]):
            if t.num_samples(node) > 1:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=node, derived_state="1")
                x -= delta

    msprime.sort_tables(**tables.asdict())
    return msprime.load_tables(**tables.asdict())


def get_ancestral_haplotypes(ts):
    """
    Returns a numpy array of the haplotypes of the ancestors in the
    specified tree sequence.
    """
    A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.uint8)
    A[:] = inference.UNKNOWN_ALLELE
    for t in ts.trees():
        for site in t.sites():
            for u in t.nodes():
                A[u, site.id] = 0
            for mutation in site.mutations:
                # Every node underneath this node will have the value set
                # at this site.
                for u in t.nodes(mutation.node):
                    A[u, site.id] = 1
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
    mask = np.ones(L)
    for a in A:
        masked = np.logical_and(a == 1, mask).astype(int)
        new_sites = np.where(masked)[0]
        mask[new_sites] = 0
        segment = np.where(a != inference.UNKNOWN_ALLELE)[0]
        # Skip any ancestors that are entirely unknown.
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
        else:
            warnings.warn("Unknown ancestor provided")
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


def build_simulated_ancestors(sample_data, ancestor_data, ts):
    # Any non-smc tree sequences are rejected.
    assert_smc(ts)
    A = get_ancestral_haplotypes(ts)
    # This is all nodes, but we only want the non samples. We also reverse
    # the order to make it forwards time.
    A = A[ts.num_samples:][::-1]
    # We also only want the variant sites
    A = A[:, sample_data.variant_site]
    # print(A.astype(np.int8))

    ancestors, start, end, focal_sites = get_ancestor_descriptors(A)
    time = len(ancestors)
    num_ancestors = len(ancestors)
    for a, s, e, focal in zip(ancestors, start, end, focal_sites):
        assert np.all(a[:s] == inference.UNKNOWN_ALLELE)
        assert np.all(a[s:e] != inference.UNKNOWN_ALLELE)
        assert np.all(a[e:] == inference.UNKNOWN_ALLELE)
        assert all(s <= site < e for site in focal)
        ancestor_data.add_ancestor(
            start=s, end=e, time=time, focal_sites=np.array(focal, dtype=np.int32),
            haplotype=a)
        time -= 1
