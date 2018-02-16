"""
Tools for evaluating the algorithm.
"""
import collections
import itertools
import warnings

import numpy as np
import msprime

import tsinfer.inference as inference


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
                    site=site_id, node=mutation.node,
                    derived_state=mutation.derived_state)
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

    # Commented out version tried to use to some reasoning about SPRs.
    # Not very successful, but seems like the right way to do this.

    # diffs = ts.edge_diffs()
    # (left, right), edges_out, edges_in = next(diffs)
    # children = [[] for _ in range(ts.num_nodes)]
    # parent = np.zeros(ts.num_nodes, dtype=int) - 1
    # for e in edges_in:
    #     parent[e.child] = e.parent
    #     children[e.parent].append(e.child)
    # x = left
    # for e in reversed(edges_in):
    #     if len(children[e.child]) > 0:
    #         site_id = tables.sites.add_row(position=x, ancestral_state="0")
    #         tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")
    #         print("ADDED mutation over", e.child, "at ", x)
    #         x += delta

    # for (left, right), edges_out, edges_in in diffs:
    #     print("=============")
    #     print("left = ", left)
    #     # u = edges_out[0].parent
    #     # nodes = [u]
    #     # if parent[u] == -1:
    #     #     nodes = children[u]
    #     assert len(edges_out) == len(edges_in)
    #     x = left - delta
    #     if len(edges_in) < 4:
    #         # Put a site over the last two children
    #         for e in edges_out[-2:]:
    #             site_id = tables.sites.add_row(position=x, ancestral_state="0")
    #             tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")
    #             x -= delta
    #     else:
    #         # Put a site over the root of the SPR
    #         u = edges_out[0].parent
    #         site_id = tables.sites.add_row(position=x, ancestral_state="0")
    #         tables.mutations.add_row(site=site_id, node=u, derived_state="1")

    #     for e in edges_out:
    #         print("out:", e)
    #         parent[e.child] = -1
    #         children[e.parent].remove(e.child)
    #     for e in edges_in:
    #         print("in :", e)
    #         parent[e.child] = e.parent
    #         children[e.parent].append(e.child)

    #     x = left
    #     if len(edges_in) < 4:
    #         # Put a site over the last two children
    #         for e in edges_in[:2]:
    #             site_id = tables.sites.add_row(position=x, ancestral_state="0")
    #             tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")
    #             x += delta
    #     else:
    #         # Put a site over the root of the SPR
    #         u = edges_in[-1].parent
    #         site_id = tables.sites.add_row(position=x, ancestral_state="0")
    #         tables.mutations.add_row(site=site_id, node=u, derived_state="1")

        # u = edges_in[-1].parent
        # nodes = [u]
        # if parent[u] == -1:
        #     nodes = children[u]
        # nodes = children[edges_in[-1].parent]
        # x = left
        # for u in nodes:
        #     if len(children[u]) > 0:
        #         site_id = tables.sites.add_row(position=x, ancestral_state="0")
        #         tables.mutations.add_row(site=site_id, node=u, derived_state="1")
        #         x += delta

        # print(parent)
        # print(children)

#         u = edges_in[-1].parent
#         if parent[u] != -1:
#             site_id = tables.sites.add_row(position=left, ancestral_state="0")
#             tables.mutations.add_row(site=site_id, node=u, derived_state="1")


    num_children = np.zeros(ts.num_nodes, dtype=int)
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    for (left, right), edges_out, edges_in in ts.edge_diffs():
        print("=============")
        print("left = ", left)
        x = left - delta
        # for e in reversed(edges_out):
        for e in edges_out:
            # if parent[e.parent] != -1:
            #     site_id = tables.sites.add_row(position=x, ancestral_state="0")
            #     tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")
            #     x -= delta
            if num_children[e.child] > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")
                x -= delta
        for e in edges_out:
            print("out:", e)
            parent[e.child] = -1
            num_children[e.parent] -= 1
        for e in edges_in:
            print("in :", e)
            parent[e.child] = e.parent
            num_children[e.parent] += 1
        x = left
        # for e in reversed(edges_in):
        for e in edges_in:
            # if parent[e.parent] != -1:
            #     site_id = tables.sites.add_row(position=x, ancestral_state="0")
            #     tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")
            #     x += delta
            if num_children[e.child] > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")
                x += delta
    x = ts.sequence_length - delta
    for u in reversed(range(ts.num_nodes)):
        if num_children[u] > 0 and parent[u] != -1:
            site_id = tables.sites.add_row(position=x, ancestral_state="0")
            tables.mutations.add_row(site=site_id, node=u, derived_state="1")
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
    # print(A.astype(np.int8))
    # This is all nodes, but we only want the non samples. We also reverse
    # the order to make it forwards time.
    A = A[ts.num_samples:][::-1]
    # We also only want the variant sites
    A = A[:, sample_data.variant_site]
    # print(A.astype(np.int8))
    # print(ts.tables.sites)
    # print(ts.tables.edges)

    ancestors, start, end, focal_sites = get_ancestor_descriptors(A)
    time = len(ancestors)
    for a, s, e, focal in zip(ancestors, start, end, focal_sites):
        assert np.all(a[:s] == inference.UNKNOWN_ALLELE)
        assert np.all(a[s:e] != inference.UNKNOWN_ALLELE)
        assert np.all(a[e:] == inference.UNKNOWN_ALLELE)
        assert all(s <= site < e for site in focal)
        ancestor_data.add_ancestor(
            start=s, end=e, time=time, focal_sites=np.array(focal, dtype=np.int32),
            haplotype=a)
        time -= 1


def print_tree_pairs(ts1, ts2, compute_distances=True):
    """
    Prints out the trees at each point in the specified tree sequences,
    alone with their KC distance.
    """
    for (left, right), tree1, tree2 in tree_pairs(ts1, ts2):
        print("-" * 20)
        print("Interval          =", left, "--", right)
        print("Source interval   =", tree1.interval)
        print("Inferred interval =", tree2.interval)
        if compute_distances:
            distance = kc_distance(tree1, tree2)
            trailer = ""
            if distance != 0:
                trailer = "[MISMATCH]"
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
