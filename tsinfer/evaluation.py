"""
Tools for evaluating the algorithm.
"""
import collections
import itertools

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
    #remove singletons
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


def insert_perfect_mutations(ts, delta=1e-3):
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
    left_map = collections.Counter()
    right_map = collections.Counter()

    # Edgesets are useful in one way as we can tag two different children at the
    # same time with one mutation. However, this means that we have mutations over
    # roots, and therefore fixed sites.

    # edgesets = list(ts.edgesets())
    # edgesets.sort(key=lambda e: -e.parent)
    # for e in edgesets:
    #     left_map[e.left] +=  1
    #     right_map[e.right] += 1
    # for e in edgesets:
    #     left_map[e.left] -= 1
    #     x = e.left + left_map[e.left] * delta
    #     assert e.left <= x < e.right
    #     site_id = tables.sites.add_row(position=x, ancestral_state="0")
    #     tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")

    #     x = e.right - right_map[e.right] * delta
    #     right_map[e.right] += 1
    #     assert e.left <= x < e.right
    #     site_id = tables.sites.add_row(position=x, ancestral_state="0")
    #     tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")

    edges = list(ts.edges())
    # Edges are sorted by time.
    for e in edges:
        left_map[e.left] +=  1
        right_map[e.right] += 1
    for e in edges:
        left_map[e.left] -= 1
        x = e.left + left_map[e.left] * delta
        assert e.left <= x < e.right
        site_id = tables.sites.add_row(position=x, ancestral_state="0")
        tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")

        x = e.right - right_map[e.right] * delta
        right_map[e.right] += 1
        assert e.left <= x < e.right
        site_id = tables.sites.add_row(position=x, ancestral_state="0")
        tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")

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
        if len(new_sites) > 0:
            segment = np.where(a != inference.UNKNOWN_ALLELE)[0]
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


def build_simulated_ancestors(input_hdf5, ancestor_hdf5, ts):
    # Any non-smc tree sequences are rejected.
    assert_smc(ts)
    A = get_ancestral_haplotypes(ts)
    print(A)
    # This is all nodes, but we only want the non samples. We also reverse
    # the order to make it forwards time.
    A = A[ts.num_samples:][::-1]


    for edge in ts.edges():
        print(edge.left, edge.right, edge.parent, edge.child, sep="\t")

    ancestors, start, end, focal_sites = get_ancestor_descriptors(A)
    time = len(ancestors)
    total_num_focal_sites = sum(len(f) for f in focal_sites)
    num_ancestors = len(ancestors)

    # TODO it's very confusing here on whether we have to add the initial
    # ancestor or not. Fix the API in some way.
    input_file = formats.InputFile(input_hdf5)
    ancestor_file = formats.AncestorFile(ancestor_hdf5, input_file, 'w')
    ancestor_file.initialise(num_ancestors, time, total_num_focal_sites)
    time -= 1
    for a, s, e, focal in zip(ancestors[1:], start[1:], end[1:], focal_sites[1:]):
        print(s, e, focal, a[s:e])
        assert np.all(a[:s] == inference.UNKNOWN_ALLELE)
        assert np.all(a[s:e] != inference.UNKNOWN_ALLELE)
        assert np.all(a[e:] == inference.UNKNOWN_ALLELE)
        assert all(s <= site < e for site in focal)
        ancestor_file.add_ancestor(
            start=s, end=e, ancestor_time=time,
            focal_sites=np.array(focal, dtype=np.int32),
            haplotype=a)
        time -= 1
    ancestor_file.finalise()


def build_simulated_ancestors_old(input_hdf5, ancestor_hdf5, ts, guess_unknown=False):
    """
    guess_unknown = False: use the exact ancestors from the coalescent simulation
    guess_unknown = None: fill out left and right of missing ancestral material with 0
    guess_unknown = True: fill out left and right of missing ancestral material by
        copying from nearest known ancestor
    """
    A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.uint8)
    mutation_sites = [[] for _ in range(ts.num_nodes)]

    if guess_unknown==True:
        #fill in all the ancestral genotypes, even for regions which do not contribute to the
        #final samples. This stops the inference algorithm getting confused by known boundaries
        #but we have to construct the ancestral types by iterating over edges for each node
        #and extending the edges left and right where appropriate
        sites = np.array([s.position for s in ts.sites()])
        edges_by_child = collections.defaultdict(list)
        for site in ts.sites():
            for mutation in site.mutations:
                mutation_sites[mutation.node].append(site.id)
        for e in ts.edges():
            edges_by_child[e.child].append([e.left, e.right, e.parent])
        for child in sorted(edges_by_child.keys(), reverse=True):
            #extend the edges leftwards and rightwards to include all parts of the genome
            edges = sorted(edges_by_child[child], key=lambda x: x[0])
            edges[0][0] = 0
            edges[-1][1] = ts.sequence_length
            for i in range(len(edges)-1):
                edges[i][1] = edges[i+1][0] = (edges[i][1] + edges[i+1][0])/2
            #now actually construct the sites array
            for edge in edges:
                #which sites does this edge span?
                mask = np.logical_and(sites >= edge[0], sites < edge[1])
                A[child,mask]=A[edge[2], mask]
                #add mutations
                for m in mutation_sites[child]:
                    A[child,m]=1
    else:
        #Jerome's original routine, where we iterate over trees, not edges
        A[:] = (0 if guess_unknown==None else inference.UNKNOWN_ALLELE)
        for t in ts.trees():
            for site in t.sites():
                for u in t.nodes():
                    A[u, site.id] = 0
                for mutation in site.mutations:
                    mutation_sites[mutation.node].append(site.id)
                    # Every node underneath this node will have the value set
                    # at this site.
                    for u in t.nodes(mutation.node):
                        A[u, site.id] = 1
    # This is all nodes, but we only want the non samples. We also reverse
    # the order to make it forwards time.
    A = A[ts.num_samples:][::-1]
    mutation_sites = mutation_sites[ts.num_samples:][::-1]
    # Now we need to process these a bit, to weed out any ancestors
    # that have no focal sites and also break up any ancestors that
    # have regions of -1 in the middle.
    ancestors = []
    focal_sites = []
    start = []
    end = []
    m = ts.num_sites
    for a, sites in zip(A, mutation_sites):
        if len(sites) > 0:
            offset = 0
            while offset < m:
                s = np.where(a[offset:] != inference.UNKNOWN_ALLELE)[0]
                if len(s) == 0:
                    break
                s = offset + s[0] # convert to actual location, not offset
                e = np.where(a[s:] == inference.UNKNOWN_ALLELE)[0]
                if len(e) == 0:
                    e = m
                else:
                    e = s + e[0]  # convert to actual location, not offset
                offset = e
                ancestor = np.empty(m, dtype=np.uint8)
                ancestor[:] = inference.UNKNOWN_ALLELE
                ancestor[s:e] = a[s:e]
                ancestors.append(ancestor)
                start.append(s)
                end.append(e)
                focal_sites.append([site for site in sites if s <= site < e])

    time = len(ancestors)
    total_num_focal_sites = sum(len(f) for f in focal_sites)
    num_ancestors = sum(len(f) > 0 for f in focal_sites) + 1

    input_file = formats.InputFile(input_hdf5)
    ancestor_file = formats.AncestorFile(ancestor_hdf5, input_file, 'w')
    ancestor_file.initialise(num_ancestors, time + 1, total_num_focal_sites)

    for a, s, e, focal in zip(ancestors, start, end, focal_sites):
        assert np.all(a[:s] == inference.UNKNOWN_ALLELE)
        assert np.all(a[s:e] != inference.UNKNOWN_ALLELE)
        assert np.all(a[e:] == inference.UNKNOWN_ALLELE)
        assert all(s <= site < e for site in focal)
        if len(focal) > 0:
            ancestor_file.add_ancestor(
                start=s, end=e, ancestor_time=time,
                focal_sites=np.array(focal, dtype=np.int32),
                haplotype=a)
            time -= 1
    ancestor_file.finalise()


