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
    # assert_single_recombination(ts)
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()

    direction = 0
    for tree in ts.trees():
        x = tree.interval[0]
        nodes = list(tree.nodes())
        if direction == 1:
            nodes = list(reversed(nodes))
        direction = (direction + 1) % 2
        for node in nodes:
            if tree.parent(node) != -1 and len(tree.children(node)) > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=node, derived_state="1")
                x += delta
        for node in nodes:
            if tree.parent(node) != -1 and len(tree.children(node)) > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=node, derived_state="1")
                x += delta
    msprime.sort_tables(**tables.asdict())
    ts = msprime.load_tables(**tables.asdict())

    tables = ts.dump_tables()
    A = get_ancestral_haplotypes(ts)
    site_map = {site.position: site.id for site in ts.sites()}
    print(A)

    child_edges = collections.defaultdict(list)
    for e in ts.edges():
        child_edges[e.child].append(e)

    breakpoints = collections.Counter()
    for child in reversed(sorted(child_edges.keys())):
        edges = child_edges[child]
        if len(edges) > 1:
            edges = sorted(edges, key=lambda e: -e.left)
            print("child = ", child)
            for j in range(len(edges) - 1):
                p1 = edges[j].parent
                p2 = edges[j + 1].parent
                bp = site_map[edges[j].left]
                print("\tpos = ", edges[j].left)
                print("\t", p1, A[p1, bp-1:bp+1])
                print("\t", p2, A[p2, bp-1:bp+1])
                values = (A[p1, bp - 1], A[p2, bp - 1])
                if (
                        p1 != p2 and #inference.UNKNOWN_ALLELE not in values and \
                        values[0] != values[1]):
                    breakpoints[edges[j].left] += 1
                    x = edges[j].left - breakpoints[edges[j].left] * delta
                    print("\tINSERTING at ", x)
                    site_id = tables.sites.add_row(position=x, ancestral_state="0")
                    tables.mutations.add_row(site=site_id, node=p2, derived_state="1")


#             for e in edges:
#                 print("\t", e)
#                 print("\t child  = ", A[e.child])
#                 print("\t parent = ", A[e.parent])



#     ancestors = collections.defaultdict(list)
#     for e in ts.edgesets():
#         ancestors[e.parent].append(e)


#     for parent, edgesets in ancestors.items():
#         if len(edgesets) > 1:
#             edgesets = sorted(edgesets, key=lambda e: -e.left)
#             print(parent, [e.children for e in edgesets])
#             for j in range(len(edgesets) - 1):
#                 diff = set(edgesets[j + 1].children) - set(edgesets[j].children)
#                 pos = edgesets[j].left
#                 print("diff = ", diff)
#                 print("pos = ", pos)
#                 last_site = site_map[pos] - 1
#                 for node in sorted(diff):
#                     print("node = ", node, A[node, last_site], A[node])
#                     if A[node, last_site] == 1:
#                         x = edgesets[j].left - delta
#                     else:
#                         x = edgesets[j].left - 2 * delta
#                     print("\t", node, parent, edgesets[j].left, x)
#                     print("\tInsertint mutation at ",x, "over ", node)
#                     # site_id = tables.sites.add_row(position=x, ancestral_state="0")
#                     # tables.mutations.add_row(site=site_id, node=node, derived_state="1")

    msprime.sort_tables(**tables.asdict())
    ts = msprime.load_tables(**tables.asdict())
    return ts


def insert_perfect_mutations_spr(ts, delta=1/64):
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

    left_map = {}
    right_map = {}
    parent = [-1 for _ in range(ts.num_nodes)]
    children = [[] for _ in range(ts.num_nodes)]
    diffs = ts.edge_diffs()
    (_, right) , _, edges_in = next(diffs)
    for e in edges_in:
        parent[e.child] = e.parent
        children[e.parent].append(e.child)

    root = edges_in[-1].parent
    stack = list(children[root])
    left = 0
    while len(stack) > 0:
        node = stack.pop()
        if len(children[node]) > 0:
            stack.extend(children[node])
            assert left < right
            site_id = tables.sites.add_row(position=left, ancestral_state="0")
            tables.mutations.add_row(site=site_id, node=node, derived_state="1")
            left += delta
    # left = 0
    # # for e in sorted(edges_in, key=lambda e: (e.right, -e.parent)):
    # for e in edges_in:
    #     node = e.child
    #     print("inserting", e)
    #     if len(children[node]) > 0:
    #         assert left < right
    #         site_id = tables.sites.add_row(position=left, ancestral_state="0")
    #         tables.mutations.add_row(site=site_id, node=node, derived_state="1")
    #         left += delta

    for (left, right), edges_out, edges_in in diffs:
        print("=============")
        print("left = ", left)

        last_parent = list(parent)
        last_children = list(children)
        for e in edges_out:
            print("out:", e)
            parent[e.child] = -1
            children[e.parent].remove(e.child)
        for e in edges_in:
            print("in :", e)
            parent[e.child] = e.parent
            children[e.parent].append(e.child)
        print("parent = ", parent)
        print("children = ", children)

        if len(edges_out) == 4:
            nodes_out = set()
            print("edges_out:", edges_out)
            for e in edges_out:
                nodes_out.add(e.parent)
                nodes_out.add(e.child)
            nodes_in = set()
            for e in edges_in:
                nodes_in.add(e.parent)
                nodes_in.add(e.child)
            node_out = (nodes_out - nodes_in).pop()
            node_in = (nodes_in - nodes_out).pop()
            print("NODE_IN = ", node_in)
            print("NODE_OUT = ", node_out)
            assert parent[node_in] != -1
            assert last_parent[node_out] != -1

            site_id = tables.sites.add_row(position=left, ancestral_state="0")
            tables.mutations.add_row(site=site_id, node=node_in, derived_state="1")
            site_id = tables.sites.add_row(position=left - delta, ancestral_state="0")
            tables.mutations.add_row(site=site_id, node=node_out, derived_state="1")
        else:
            old_root = edges_out[0].parent
            new_root = edges_in[-1].parent
            print("old_root = ", old_root)
            print("new_root = ", new_root)


            for c in last_children[old_root]:
                if len(children[c]) > 0:
                    site_id = tables.sites.add_row(
                            position=left - delta, ancestral_state="0")
                    tables.mutations.add_row(site=site_id, node=c, derived_state="1")
                    break
            for c in children[new_root]:
                if len(children[c]) > 0:
                    site_id = tables.sites.add_row(
                            position=left, ancestral_state="0")
                    tables.mutations.add_row(site=site_id, node=c, derived_state="1")
                    break

        root = 0
        while parent[root] != -1:
            root = parent[root]
        x  = left + (right - left) / 2
        for c in children[root]:
            if len(children[c]) > 0:
                site_id = tables.sites.add_row(
                        position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=c, derived_state="1")
                x += delta
                # Only insert 1.
                break


            # if parent[node_in] != -1:
            #     site_id = tables.sites.add_row(position=left, ancestral_state="0")
            #     tables.mutations.add_row(site=site_id, node=node_in, derived_state="1")
            # else:
            #     c = children[node_in]
            #     child = c[0]
            #     if len(children[child]) == 0:
            #         child = c[1]
            #     site_id = tables.sites.add_row(position=left, ancestral_state="0")
            #     tables.mutations.add_row(site=site_id, node=child, derived_state="1")

            # if last_parent[node_out] != -1:
            #     site_id = tables.sites.add_row(position=left - delta, ancestral_state="0")
            #     tables.mutations.add_row(site=site_id, node=node_out, derived_state="1")
            # else:
            #     c = last_children[node_out]
            #     child = c[0]
            #     if len(children[child]) == 0:
            #         child = c[1]
            #     site_id = tables.sites.add_row(position=left - delta, ancestral_state="0")
            #     tables.mutations.add_row(site=site_id, node=child, derived_state="1")

#         else:
#             print("Root change")
#             c = [edges_out[j].child for j in range(2)]
#             child = c[0]
#             if len(children[child]) == 0:
#                 child = c[1]
#             site_id = tables.sites.add_row(position=left, ancestral_state="0")
#             tables.mutations.add_row(site=site_id, node=child, derived_state="1")



    root = 0
    while parent[root] != -1:
        root = parent[root]
    stack = list(children[root])
    x = ts.sequence_length - delta
    while len(stack) > 0:
        node = stack.pop()
        if len(children[node]) > 0:
            stack.extend(children[node])
            site_id = tables.sites.add_row(position=x, ancestral_state="0")
            tables.mutations.add_row(site=site_id, node=node, derived_state="1")
            x -= delta



#     assert ts.num_samples > 2
#     left_map = {}
#     right_map = {}
#     for e in sorted(ts.edgesets(), key=lambda e: -e.parent):
#         is_root = False
#         for t in ts.trees():
#             if e.left == t.interval[0]:
#                 is_root = e.parent == t.root
#                 break
#         node = e.parent
#         print(e)
#         if is_root:
#             node = e.children[0]
#             if node < ts.num_samples:
#                 node = e.children[1]
#             assert node >= ts.num_samples
#             print("Choosing node", node)
#         if e.left not in left_map:
#             left_map[e.left] = e.left
#         left = left_map[e.left]
#         left_map[e.left] = left + delta
#         site_id = tables.sites.add_row(position=left, ancestral_state="0")
#         tables.mutations.add_row(site=site_id, node=node, derived_state="1")

#         if e.right not in right_map:
#             right_map[e.right] = e.right - delta
#         right = right_map[e.right]
#         right_map[e.right] = right - delta
#         site_id = tables.sites.add_row(position=right, ancestral_state="0")
#         tables.mutations.add_row(site=site_id, node=node, derived_state="1")


    # parent_edgese = collections.defaultdict(list)
    # for e in ts.edges():
    #     parent_edges[e.parent].append(e)

    # left_map = {}
    # right_map = {}
    # for ancestor in range(ts.num_samples, ts.num_nodes):
    #     print("ancestor = ", ancestor, parent_edges[ancestor])
    #     edges = parent_edges[ancestor]
    #     edges.sort(key=lambda e: e.left)
    #     e = edges[0]
    #     is_root = False
    #     for t in ts.trees():
    #         if e.left == t.interval[0]:
    #             if ancestor == t.root:
    #                 is_root = True
    #             break
    #     if not is_root:
    #         if e.left not in left_map:
    #             left_map[e.left] = e.left
    #         left = left_map[e.left]
    #         left_map[e.left] = left + delta
    #         site_id = tables.sites.add_row(position=left, ancestral_state="0")
    #         tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")

    #         e = edges[-1]
    #         if e.right not in right_map:
    #             right_map[e.right] = e.right - delta
    #         right = right_map[e.right]
    #         right_map[e.right] = right - delta
    #         site_id = tables.sites.add_row(position=right, ancestral_state="0")
    #         tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")

    # child_edges = collections.defaultdict(list)
    # for e in ts.edges():
    #     child_edges[e.child].append(e)

    # left_map = {}
    # right_map = {}
    # for child in range(ts.num_samples, ts.num_nodes):
    #     edges = child_edges[child]
    #     edges.sort(key=lambda e: e.left)
    #     print(child, edges)
    #     if len(edges) > 0:
    #         e = edges[0]
    #         if e.left not in left_map:
    #             left_map[e.left] = e.left
    #         left = left_map[e.left]
    #         left_map[e.left] = left + delta
    #         site_id = tables.sites.add_row(position=left, ancestral_state="0")
    #         tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")

    #         e = edges[-1]
    #         if e.right not in right_map:
    #             right_map[e.right] = e.right - delta
    #         right = right_map[e.right]
    #         right_map[e.right] = right - delta
    #         site_id = tables.sites.add_row(position=right, ancestral_state="0")
    #         tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")

    # for e in ts.edges():
    #     if e.left not in left_map:
    #         left_map[e.left] = e.left
    #     left = left_map[e.left]
    #     left_map[e.left] = left + delta
    #     site_id = tables.sites.add_row(position=left, ancestral_state="0")
    #     tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")
    # for e in reversed(list(ts.edges())):
    #     if e.right not in right_map:
    #         right_map[e.right] = e.right - delta
    #     right = right_map[e.right]
    #     right_map[e.right] = right - delta
    #     site_id = tables.sites.add_row(position=right, ancestral_state="0")
    #     tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")


#     for e in ts.edges():
#         if e.child >= ts.num_samples:
#             if e.left not in left_map:
#                 left_map[e.left] = e.left
#             left = left_map[e.left]
#             left_map[e.left] = left + delta
#             site_id = tables.sites.add_row(position=left, ancestral_state="0")
#             tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")

#     for e in list(ts.edges()):
#         if e.child >= ts.num_samples:
#             if e.right not in right_map:
#                 right_map[e.right] = e.right - delta
#             right = right_map[e.right]
#             right_map[e.right] = right - delta
#             site_id = tables.sites.add_row(position=right, ancestral_state="0")
#             tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")

    # for e in sorted(ts.edgesets(), key=lambda e: e.parent):
    #     print(e)
    #     if e.left not in left_map:
    #         left_map[e.left] = e.left
    #     left = left_map[e.left]
    #     left_map[e.left] = left + delta
    #     site_id = tables.sites.add_row(position=left, ancestral_state="0")
    #     tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")

    #     if e.right not in right_map:
    #         right_map[e.right] = e.right - delta
    #     right = right_map[e.right]
    #     right_map[e.right] = right - delta
    #     site_id = tables.sites.add_row(position=right, ancestral_state="0")
    #     tables.mutations.add_row(site=site_id, node=e.parent, derived_state="1")

    print(tables.sites)
    print(tables.mutations)
    msprime.sort_tables(**tables.asdict())

    # ts = msprime.load_tables(**tables.asdict())
    # tables.sites.clear()
    # tables.mutations.clear()
    # for tree in ts.trees():
    #     for site in tree.sites():
    #         assert len(site.mutations) == 1
    #         mutation = site.mutations[0]
    #         if tree.num_samples(mutation.node) < ts.num_samples:
    #             site_id = tables.sites.add_row(
    #                 position=site.position, ancestral_state=site.ancestral_state)
    #             tables.mutations.add_row(
    #                 site=site_id, node=mutation.node,
    #                 derived_state=mutation.derived_state)

    ts = msprime.load_tables(**tables.asdict())
    A = get_ancestral_haplotypes(ts)[::-1]
    print(A.astype(np.int8))
    return ts


def insert_perfect_mutations_old(ts, delta=1/64):
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
    for (left, right), edges_out, edges_in in ts.edge_diffs():
        print("=============")
        print("left = ", left)
        last_parent = list(parent)
        last_num_children = list(num_children)
        for e in edges_out:
        # for e in sorted(edges_out, key=lambda e: e.right - e.left):
            print("out:", e)
            parent[e.child] = -1
            num_children[e.parent] -= 1

        # for e in sorted(edges_in, key=lambda e: -(e.right - e.left)):
        for e in edges_in:
            # print("in :", e)
            parent[e.child] = e.parent
            num_children[e.parent] += 1

        x = left - delta
        spr_out_root = -1
        for e in edges_out:
            if parent[e.parent] == -1 and num_children[e.parent] == 0:
                spr_out_root = e.parent
        for e in edges_out:
            if last_num_children[e.child] > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")
                x -= delta
        print("spr_out_root = ", spr_out_root)
        if spr_out_root != -1 and last_parent[spr_out_root] != -1:
            print("INSERT out root @ ", x)
            site_id = tables.sites.add_row(position=x, ancestral_state="0")
            tables.mutations.add_row(site=site_id, node=spr_out_root, derived_state="1")
            x -= delta

        spr_in_root = -1
        for e in edges_in:
            # print("in: ", e.parent, last_parent[e.parent], last_num_children[e.parent])
            if last_parent[e.parent] == -1 and last_num_children[e.parent] == 0:
                spr_in_root = e.parent

        print("spr in root = ", spr_in_root)
        x = left
        for e in edges_in:
            if num_children[e.child] > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=e.child, derived_state="1")
                x += delta
        if parent[spr_in_root] != -1:
            print("INSERT in root")
            site_id = tables.sites.add_row(position=x, ancestral_state="0")
            tables.mutations.add_row(site=site_id, node=spr_in_root, derived_state="1")
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

    # get_ancestor_descriptors ensures that the ultimate ancestor is included.
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
    print("Total weighted tree distance = ", weighted_distance)
    print("Total mismatch interval      = ", total_mismatch_interval)
    print("Total mismatches             = ", total_mismatches)
