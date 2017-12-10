# TODO copyright.

"""
Python algorithm implementation.

This isn't meant to be used for any real inference, as it is
*many* times slower than the real C implementation. However,
it is a useful development and debugging tool, and so any
updates made to the low-level C engine should be made here
first.
"""
import collections

import numpy as np
import msprime

UNKNOWN_ALLELE = 255

class Edge(object):

    def __init__(self, left=None, right=None, parent=None, child=None):
        self.left = left
        self.right = right
        self.parent = parent
        self.child = child

    def __str__(self):
        return "Edge(left={}, right={}, parent={}, child={})".format(
            self.left, self.right, self.parent, self.child)


class Site(object):
    def __init__(self, id, frequency, genotypes):
        self.id = id
        self.frequency = frequency
        self.genotypes = genotypes


class AncestorBuilder(object):
    """
    Builds inferred ancestors.
    """
    def __init__(self, num_samples, num_sites):
        self.num_samples = num_samples
        self.num_sites = num_sites
        self.sites = [None for _ in range(self.num_sites)]
        self.frequency_map = [{} for _ in range(self.num_samples + 1)]

    def add_site(self, site_id, frequency, genotypes):
        """
        Adds a new site at the specified ID and allele pattern to the builder.
        """
        self.sites[site_id] = Site(site_id, frequency, genotypes)
        if frequency > 1:
            pattern_map = self.frequency_map[frequency]
            # Each unique pattern gets added to the list
            key = genotypes.tobytes()
            if key not in pattern_map:
                pattern_map[key] = []
            pattern_map[key].append(site_id)
        else:
            # Save some memory as we'll never look at these
            self.sites[site_id].genotypes = None

    def print_state(self):
        print("Ancestor builder")
        print("Sites = ")
        for j in range(self.num_sites):
            site = self.sites[j]
            print(site.frequency, "\t", site.genotypes)
        print("Frequency map")
        for f in range(self.num_samples):
            pattern_map = self.frequency_map[f]
            if len(pattern_map) > 0:
                print("f = ", f, "with ", len(pattern_map), "patterns")
                for pattern, sites in pattern_map.items():
                    print("\t", pattern, ":", sites)

    def ancestor_descriptors(self):
        """
        Returns a list of (frequency, focal_sites) tuples describing the
        ancestors in reverse order of frequency.
        """
        ret = []
        for frequency in reversed(range(self.num_samples + 1)):
            for focal_sites in self.frequency_map[frequency].values():
                ret.append((frequency, np.array(focal_sites, dtype=np.int32)))
        return ret

    def __build_ancestor_sites(self, focal_site, sites, a):
        samples = set()
        g = self.sites[focal_site].genotypes
        for j in range(self.num_samples):
            if g[j] == 1:
                samples.add(j)
        for l in sites:
            a[l] = 0
            if self.sites[l].frequency > self.sites[focal_site].frequency:
                # print("\texamining:", self.sites[l])
                # print("\tsamples = ", samples)
                num_ones = 0
                num_zeros = 0
                for j in samples:
                    if self.sites[l].genotypes[j] == 1:
                        num_ones += 1
                    else:
                        num_zeros += 1
                # TODO choose a branch uniformly if we have equality.
                if num_ones >= num_zeros:
                    a[l] = 1
                    samples = set(j for j in samples if self.sites[l].genotypes[j] == 1)
                else:
                    samples = set(j for j in samples if self.sites[l].genotypes[j] == 0)
            if len(samples) == 1:
                # print("BREAK")
                break

    def make_ancestor(self, focal_sites, a):
        a[:] = UNKNOWN_ALLELE
        focal_site = focal_sites[0]
        sites = range(focal_sites[-1] + 1, self.num_sites)
        self.__build_ancestor_sites(focal_site, sites, a)
        focal_site = focal_sites[-1]
        sites = range(focal_sites[0] - 1, -1, -1)
        self.__build_ancestor_sites(focal_site, sites, a)
        for j in range(focal_sites[0], focal_sites[-1] + 1):
            if j in focal_sites:
                a[j] = 1
            else:
                self.__build_ancestor_sites(focal_site, [j], a)
        known = np.where(a != UNKNOWN_ALLELE)[0]
        start = known[0]
        end = known[-1] + 1
        return start, end


def edge_group_equal(edges, group1, group2):
    """
    Returns true if the specified subsets of the list of edges are considered
    equal in terms of a shared recombination.
    """
    s1, e1 = group1
    s2, e2 = group2
    ret = False
    if (e1 - s1) == (e2 - s2):
        ret = True
        for j in range(e1 - s1):
            edge1 = edges[s1 + j]
            edge2 = edges[s2 + j]
            condition = (
                edge1.left != edge2.left or
                edge1.right != edge2.right or
                edge1.parent != edge2.parent)
            if condition:
                ret = False
                break
    return ret



class TreeSequenceBuilder(object):

    def __init__(
            self, sequence_length, positions, recombination_rate,
            max_nodes, max_edges):
        self.num_nodes = 0
        self.sequence_length = sequence_length
        self.positions = positions
        self.recombination_rate = recombination_rate
        self.num_sites = positions.shape[0]
        self.time = []
        self.flags = []
        self.mutations = collections.defaultdict(list)
        self.edges = []
        self.mean_traceback_size = 0

    def __index_edges(self):
        self.edges.sort(key=lambda e: (e.left, self.time[e.parent]))
        M = len(self.edges)
        self.removal_order = sorted(
            range(M), key=lambda j: (
                self.edges[j].right, -self.time[self.edges[j].parent]))

    def restore_nodes(self, time, flags):
        for t, flag in zip(time, flags):
            self.add_node(t, flag == 1)

    def restore_edges(self, left, right, parent, child):
        for l, r, p, c in zip(left, right, parent, child):
            self.edges.append(Edge(int(l), int(r), p, c))
        self.__index_edges()

    def restore_mutations(self, site, node, derived_state, parent):
        for s, u, d in zip(site, node, derived_state):
            self.mutations[s].append((u, d))

    def add_node(self, time, is_sample=True):
        self.num_nodes += 1
        self.time.append(time)
        self.flags.append(int(is_sample))
        return self.num_nodes - 1

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def num_mutations(self):
        return sum(len(node_state_list) for node_state_list in self.mutations.values())

    def print_state(self):
        print("TreeSequenceBuilder state")
        print("num_sites = ", self.num_sites)
        print("num_nodes = ", self.num_nodes)
        nodes = msprime.NodeTable()
        flags, time = self.dump_nodes()
        nodes.set_columns(flags=flags, time=time)
        print("nodes = ")
        print(nodes)

        edges = msprime.EdgeTable()
        left, right, parent, child = self.dump_edges()
        edges.set_columns(left=left, right=right, parent=parent, child=child)
        print("edges = ")
        print(edges)
        print("Removal order = ", self.removal_order)

        if nodes.num_rows > 1:
            msprime.sort_tables(nodes, edges)
            samples = np.where(nodes.flags == 1)[0].astype(np.int32)
            msprime.simplify_tables(samples, nodes, edges)
            print("edges = ")
            print(edges)

    def _replace_recombinations(self):
        # print("START!!")
        # First filter out all edges covering the full interval
        output_edges = []
        active = self.edges
        filtered = []
        for j in range(len(active)):
            condition = not (active[j].left == 0 and active[j].right == self.num_sites)
            if condition:
                filtered.append(active[j])
            else:
                output_edges.append(active[j])
        active = filtered
        if len(active) > 0:
            # Now sort by (l, r, p, c) to group together all identical (l, r, p) values.
            active.sort(key=lambda e: (e.left, e.right, e.parent, e.child))
            filtered = []
            prev_cond = False
            for j in range(len(active) - 1):
                next_cond = (
                    active[j].left == active[j + 1].left and
                    active[j].right == active[j + 1].right and
                    active[j].parent == active[j + 1].parent)
                if prev_cond or next_cond:
                    filtered.append(active[j])
                else:
                    output_edges.append(active[j])
                prev_cond = next_cond
            j = len(active) - 1
            if prev_cond:
                filtered.append(active[j])
            else:
                output_edges.append(active[j])
            active = filtered

        if len(active) > 0:
            # Now sort by (child, left, right) to group together all contiguous
            active.sort(key=lambda x: (x.child, x.left, x.right))
            filtered = []
            prev_cond = False
            for j in range(len(active) - 1):
                next_cond = (
                    active[j].right == active[j + 1].left and
                    active[j].child == active[j + 1].child)
                if next_cond or prev_cond:
                    filtered.append(active[j])
                else:
                    output_edges.append(active[j])
                prev_cond = next_cond
            j = len(active) - 1
            if prev_cond:
                filtered.append(active[j])
            else:
                output_edges.append(active[j])
            active = list(filtered)
        if len(active) > 0:
            # We sort by left, right, parent again to find identical edges.
            # Remove any that there is only one of.
            active.sort(key=lambda x: (x.left, x.right, x.parent, x.child))
            filtered = []
            prev_cond = False
            for j in range(len(active) - 1):
                next_cond = (
                    active[j].left == active[j + 1].left and
                    active[j].right == active[j + 1].right and
                    active[j].parent == active[j + 1].parent)
                if next_cond or prev_cond:
                    filtered.append(active[j])
                else:
                    output_edges.append(active[j])
                prev_cond = next_cond
            j = len(active) - 1
            if prev_cond:
                filtered.append(active[j])
            else:
                output_edges.append(active[j])
            active = filtered

        if len(active) > 0:
            assert len(active) + len(output_edges) == len(self.edges)
            active.sort(key=lambda x: (x.child, x.left, x.right, x.parent))
            used = [True for _ in active]

            group_start = 0
            groups = []
            for j in range(1, len(active)):
                condition = (
                    active[j - 1].right != active[j].left or
                    active[j - 1].child != active[j].child)
                if condition:
                    if j - group_start > 1:
                        groups.append((group_start, j))
                    group_start = j
            j = len(active)
            if j - group_start > 1:
                groups.append((group_start, j))

            shared_recombinations = []
            match_found = [False for _ in groups]
            for j in range(len(groups)):
                # print("Finding matches for group", j, "match_found = ", match_found)
                matches = []
                if not match_found[j]:
                    for k in range(j + 1, len(groups)):
                        # Compare this group to the others.
                        if not match_found[k] and edge_group_equal(
                                active, groups[j], groups[k]):
                            matches.append(k)
                            match_found[k] = True
                if len(matches) > 0:
                    match_found[j] = True
                    shared_recombinations.append([j] + matches)
                # print("Got", matches, match_found[j])

            if len(shared_recombinations) > 0:
                # print("Shared recombinations = ", shared_recombinations)
                index_set = set()
                for group_index_list in shared_recombinations:
                    for index in group_index_list:
                        assert index not in index_set
                        index_set.add(index)
                for group_index_list in shared_recombinations:
                    # print("Shared recombination for group:", group_index_list)
                    left_set = set()
                    right_set = set()
                    parent_set = set()
                    synthetic_child = -1
                    for group_index in group_index_list:
                        start, end = groups[group_index]
                        left_set.add(tuple([active[j].left for j in range(start, end)]))
                        right_set.add(
                            tuple([active[j].right for j in range(start, end)]))
                        parent_set.add(
                            tuple([active[j].parent for j in range(start, end)]))
                        children = set(active[j].child for j in range(start, end))
                        if self.flags[active[start].child] == 0:
                            synthetic_child = active[start].child
                        assert len(children) == 1
                        for j in range(start, end - 1):
                            assert active[j].right == active[j + 1].left
                        # for j in range(start, end):
                        #     print("\t", active[j])
                        # print()
                    assert len(left_set) == 1
                    assert len(right_set) == 1
                    assert len(parent_set) == 1

                    if synthetic_child != -1:
                        # print("Synthetic child!",synthetic_child)
                        # We have a child in the group already that covers the full
                        # region. Instead of adding a new node, we make edges pointing
                        # to this new node.
                        for group_index in group_index_list:
                            start, end = groups[group_index]
                            left = active[start].left
                            right = active[end - 1].right
                            t_parent = self.time[synthetic_child]
                            t_child = self.time[active[start].child]
                            # We need to guard against making links between nodes with
                            # equal time. We should do something better here as this
                            # will mean we consider the same groups over and over, rejecting
                            # them each time.
                            if active[start].child != synthetic_child and t_child < t_parent:
                                # Mark all the edges we don't need as unused.
                                for j in range(start, end):
                                    used[j] = False
                                j = groups[group_index][0]
                                output_edges.append(
                                    Edge(left, right, synthetic_child, active[j].child))
                                # print("g add", output_edges[-1])
                                # print("time = ", self.time[synthetic_child],
                                #         self.time[active[j].child])
                    else:
                        # Mark the edges in these group as unused.
                        for group_index in group_index_list:
                            start, end = groups[group_index]
                            for j in range(start, end):
                                used[j] = False

                        parent_time = 1e200
                        # Get the parents from the first group.
                        start, end = groups[group_index_list[0]]
                        for j in range(start, end):
                            parent_time = min(parent_time, self.time[active[j].parent])
                        # Get the children from the first record in each group.
                        children_time = -1
                        for group_index in group_index_list:
                            j = groups[group_index][0]
                            children_time = max(children_time, self.time[active[j].child])
                        new_time = children_time + (parent_time - children_time) / 2
                        new_node = self.add_node(new_time, is_sample=False)
                        # print("adding node ", new_node, "@time", new_time)
                        # For each segment add a new edge with the new node as child.
                        start, end = groups[group_index_list[0]]
                        for j in range(start, end):
                            output_edges.append(Edge(
                                active[j].left, active[j].right, active[j].parent, new_node))
                            # print("s add", output_edges[-1])
                        left = active[start].left
                        right = active[end - 1].right
                        # For each group, add a new segment covering the full interval.
                        for group_index in group_index_list:
                            start, end = groups[group_index]
                            j = groups[group_index][0]
                            output_edges.append(Edge(left, right, new_node, active[j].child))
                            # print("g add", output_edges[-1])

                        # print("Done\n")

                # print("Setting edges to ", len(output_edges), "new edges")
                for j in range(len(active)):
                    if used[j]:
                        output_edges.append(active[j])
                    # else:
                        # print("Filtering out", active[j])
                # print("BEFORE")
                # for e in self.edges:
                #     print("\t", e)

                # self.replaces_done += 1
                self.edges = output_edges
                # print("AFTER")
                # for e in self.edges:
                #     print("\t", e)



    def update(
            self, num_nodes, time, left, right, parent, child, site, node,
            derived_state):
        for _ in range(num_nodes):
            self.add_node(time)
        for l, r, p, c in zip(left, right, parent, child):
            self.edges.append(Edge(l, r, p, c))

        # print("update at time ", time, "num_edges = ", len(self.edges))
        for s, u, d in zip(site, node, derived_state):
            self.mutations[s].append((u, d))

        self._replace_recombinations()

        self.__index_edges()

    def dump_nodes(self):
        time = self.time[:self.num_nodes]
        flags = self.flags[:]
        return flags, time

    def dump_edges(self):
        left = np.zeros(self.num_edges, dtype=np.int32)
        right = np.zeros(self.num_edges, dtype=np.int32)
        parent = np.zeros(self.num_edges, dtype=np.int32)
        child = np.zeros(self.num_edges, dtype=np.int32)
        for j, edge in enumerate(self.edges):
            left[j] = edge.left
            right[j] = edge.right
            parent[j] = edge.parent
            child[j] = edge.child
        return left, right, parent, child

    def dump_mutations(self):
        num_mutations = sum(len(muts) for muts in self.mutations.values())
        site = np.zeros(num_mutations, dtype=np.int32)
        node = np.zeros(num_mutations, dtype=np.int32)
        parent = np.zeros(num_mutations, dtype=np.int32)
        derived_state = np.zeros(num_mutations, dtype=np.int8)
        j = 0
        for l in sorted(self.mutations.keys()):
            p = j
            for u, d in self.mutations[l]:
                site[j] = l
                node[j] = u
                derived_state[j] = d
                parent[j] = -1
                if d == 0:
                    parent[j] = p
                j += 1
        return site, node, derived_state, parent

def is_descendant(pi, u, v):
    """
    Returns True if the specified node u is a descendent of node v. That is,
    v is on the path to root from u.
    """
    ret = False
    if v != -1:
        w = u
        path = []
        while w != v and w != msprime.NULL_NODE:
            path.append(w)
            w = pi[w]
        # print("DESC:",v, u, path)
        ret = w == v
    if u < v:
        assert not ret
    # print("IS_DESCENDENT(", u, v, ") = ", ret)
    return ret


class AncestorMatcher(object):

    def __init__(self, tree_sequence_builder, error_rate=0):
        self.tree_sequence_builder = tree_sequence_builder
        self.error_rate = error_rate
        self.num_sites = tree_sequence_builder.num_sites
        self.positions = tree_sequence_builder.positions
        self.parent = None
        self.left_child = None
        self.right_sib = None
        self.traceback = None
        self.max_likelihood_node = None
        self.likelihood = None
        self.likelihood_nodes = None
        self.total_memory = 0

    def print_state(self):
        print("Ancestor matcher state")
        print("max_L_node\ttraceback")
        for l in range(self.num_sites):
            print(self.max_likelihood_node[l], self.traceback[l], sep="\t")

    def check_likelihoods(self):
        # Every value in L_nodes must be positive.
        for u in self.likelihood_nodes:
            assert self.likelihood[u] >= 0
        for u, v in enumerate(self.likelihood):
            # Every non-negative value in L should be in L_nodes
            if v >= 0:
                assert u in self.likelihood_nodes
            # Roots other than 0 should have v == -2
            if u != 0 and self.parent[u] == -1 and self.left_child[u] == -1:
                # print("root: u = ", u, self.parent[u], self.left_child[u])
                assert v == -2

    def update_site(self, site, state):
        n = self.tree_sequence_builder.num_nodes
        recombination_rate = self.tree_sequence_builder.recombination_rate
        err = self.error_rate

        r = 1 - np.exp(-recombination_rate[site] / n)
        recomb_proba = r / n
        no_recomb_proba = 1 - r + r / n

        # print("update_site", site, state)

        if site not in self.tree_sequence_builder.mutations:
            mutation_node = msprime.NULL_NODE
        else:
            mutation_node = self.tree_sequence_builder.mutations[site][0][0]
            # Insert an new L-value for the mutation node if needed.
            if self.likelihood[mutation_node] == -1:
                u = mutation_node
                while self.likelihood[u] == -1:
                    u = self.parent[u]
                self.likelihood[mutation_node] = self.likelihood[u]
                self.likelihood_nodes.append(mutation_node)

        # print("Site ", site, "mutation = ", mutation_node, "state = ", state)

        distance = 1
        if site > 0:
            distance = self.positions[site] - self.positions[site - 1]
        # Update the likelihoods for this site.
        # print("Site ", site, "distance = ", distance)
        # print("Computing likelihoods for ", mutation_node, self.likelihood_nodes)
        path_cache = np.zeros(n, dtype=np.int8) - 1
        max_L = -1
        max_L_node = -1
        for u in self.likelihood_nodes:
            d = False
            if mutation_node != -1:
                v = u
                while v != -1 and v != mutation_node and path_cache[v] == -1:
                    v = self.parent[v]
                if v != -1 and path_cache[v] != -1:
                    d = path_cache[v]
                else:
                    d = v == mutation_node
                assert d == is_descendant(self.parent, u, mutation_node)
                # Insert this path into the cache.
                v = u
                while v != -1 and v != mutation_node and path_cache[v] == -1:
                    path_cache[v] = d
                    v = self.parent[v]

            x = self.likelihood[u] * no_recomb_proba * distance
            assert x >= 0
            y = recomb_proba * distance
            # print("\t", u, x, y)
            if x > y:
                z = x
                self.traceback[site][u] = False
            else:
                z = y
                self.traceback[site][u] = True
            if state == 1:
                emission_p = (1 - err) * d + err * (not d)
            else:
                emission_p = err * d + (1 - err) * (not d)
            self.likelihood[u] = z * emission_p
            if self.likelihood[u] > max_L:
                max_L = self.likelihood[u]
                max_L_node = u

        # print("site=", site, "Max L = ", max_L, "node = ", max_L_node)
        self.max_likelihood_node[site] = max_L_node

        # Reset the path cache
        for u in self.likelihood_nodes:
            v = u
            while v != -1 and path_cache[v] != -1:
                path_cache[v] = -1
                v = self.parent[v]
        assert np.all(path_cache == -1)

        self.compress_likelihoods()
        self.normalise_likelihoods()

    def normalise_likelihoods(self):
        max_L = max(self.likelihood[u] for u in self.likelihood_nodes)
        for u in self.likelihood_nodes:
            if self.likelihood[u] == max_L:
                self.likelihood[u] = 1.0
            else:
                self.likelihood[u] /= max_L

    def compress_likelihoods(self):
        L_cache = np.zeros_like(self.likelihood) - 1
        cached_paths = []
        old_likelihood_nodes = list(self.likelihood_nodes)
        self.likelihood_nodes.clear()
        for u in old_likelihood_nodes:
            # We need to find the likelihood of the parent of u. If this is
            # the same as u, we can delete it.
            p = self.parent[u]
            if p != -1:
                cached_paths.append(p)
                v = p
                while self.likelihood[v] == -1 and L_cache[v] == -1:
                    v = self.parent[v]
                L_p = L_cache[v]
                if L_p == -1:
                    L_p = self.likelihood[v]
                # Fill in the L cache
                v = p
                while self.likelihood[v] == -1 and L_cache[v] == -1:
                    L_cache[v] = L_p
                    v = self.parent[v]

                if self.likelihood[u] == L_p:
                    # Delete u from the map
                    self.likelihood[u] = -1
            if self.likelihood[u] >= 0:
                self.likelihood_nodes.append(u)
        # Reset the L cache
        for u in cached_paths:
            v = u
            while v != -1 and L_cache[v] != -1:
                L_cache[v] = -1
                v = self.parent[v]
        assert np.all(L_cache == -1)

    def remove_edge(self, edge):
        p = edge.parent
        c = edge.child
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == msprime.NULL_NODE:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == msprime.NULL_NODE:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = msprime.NULL_NODE
        self.left_sib[c] = msprime.NULL_NODE
        self.right_sib[c] = msprime.NULL_NODE

    def insert_edge(self, edge):
        p = edge.parent
        c = edge.child
        self.parent[c] = p
        u = self.right_child[p]
        if u == msprime.NULL_NODE:
            self.left_child[p] = c
            self.left_sib[c] = msprime.NULL_NODE
            self.right_sib[c] = msprime.NULL_NODE
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = msprime.NULL_NODE
        self.right_child[p] = c

    def is_nonzero_root(self, u):
        return u != 0 and self.parent[u] == -1 and self.left_child[u] == -1

    def approximately_equal(self, a, b):
        # Based on Python is_close, https://www.python.org/dev/peps/pep-0485/
        rel_tol = 1e-9
        abs_tol = 0.0
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def find_path(self, h, start, end, match):

        M = len(self.tree_sequence_builder.edges)
        O = self.tree_sequence_builder.removal_order
        n = self.tree_sequence_builder.num_nodes
        m = self.tree_sequence_builder.num_sites
        edges = self.tree_sequence_builder.edges
        self.parent = np.zeros(n, dtype=int) - 1
        self.left_child = np.zeros(n, dtype=int) - 1
        self.right_child = np.zeros(n, dtype=int) - 1
        self.left_sib = np.zeros(n, dtype=int) - 1
        self.right_sib = np.zeros(n, dtype=int) - 1
        self.traceback = [{} for _ in range(m)]
        self.max_likelihood_node = np.zeros(m, dtype=int) - 1

        self.likelihood = np.zeros(n) - 2
        self.likelihood_nodes = []
        L_cache = np.zeros_like(self.likelihood) - 1

        # print("MATCH: start=", start, "end = ", end)
        j = 0
        k = 0
        left = 0
        pos = 0
        right = m
        while j < M and k < M and edges[j].left <= start:
            # print("top of init loop:", left, right)
            while edges[O[k]].right == pos:
                self.remove_edge(edges[O[k]])
                k += 1
            while j < M and edges[j].left == pos:
                self.insert_edge(edges[j])
                j += 1
            left = pos
            right = m
            if j < M:
                right = min(right, edges[j].left)
            if k < M:
                right = min(right, edges[O[k]].right)
            pos = right
        assert left < right

        self.likelihood_nodes.append(0)
        self.likelihood[0] = 1
        for u in range(n):
            if self.parent[u] != -1:
                self.likelihood[u] = -1

        remove_start = k
        while left < end:
            assert left < right
            # print("START OF TREE LOOP", left, right)
            for l in range(remove_start, k):
                edge = edges[O[l]]
                for u in [edge.parent, edge.child]:
                    if self.is_nonzero_root(u):
                        self.likelihood[u] = -2
                        if u in self.likelihood_nodes:
                            self.likelihood_nodes.remove(u)

            self.normalise_likelihoods()
            self.check_likelihoods()
            for site in range(max(left, start), min(right, end)):
                self.update_site(site, h[site])

            remove_start = k
            while k < M and edges[O[k]].right == right:
                edge = edges[O[k]]
                self.remove_edge(edge)
                k += 1
                if self.likelihood[edge.child] == -1:
                    # If the child has an L value, traverse upwards until we
                    # find the parent that carries it. To avoid repeated traversals
                    # along the same path we make a cache of the L values.
                    u = edge.parent
                    while self.likelihood[u] == -1 and L_cache[u] == -1:
                        u = self.parent[u]
                    L_child = L_cache[u]
                    if L_child == -1:
                        L_child = self.likelihood[u]
                    # Fill in the L_cache
                    u = edge.parent
                    while self.likelihood[u] == -1 and L_cache[u] == -1:
                        L_cache[u] = L_child
                        u = self.parent[u]
                    self.likelihood[edge.child] = L_child
                    self.likelihood_nodes.append(edge.child)
            # Clear the L cache
            for l in range(remove_start, k):
                edge = edges[O[l]]
                u = edge.parent
                while L_cache[u] != -1:
                    L_cache[u] = -1
                    u = self.parent[u]
            assert np.all(L_cache == -1)

            left = right
            while j < M and edges[j].left == left:
                edge = edges[j]
                self.insert_edge(edge)
                j += 1
                # There's no point in compressing the likelihood tree here as we'll be
                # doing it after we update the first site anyway.
                for u in [edge.parent, edge.child]:
                    if self.likelihood[u] == -2:
                        self.likelihood[u] = 0
                        self.likelihood_nodes.append(u)
            right = m
            if j < M:
                right = min(right, edges[j].left)

            if k < M:
                right = min(right, edges[O[k]].right)

        return self.run_traceback(start, end, match)

    def run_traceback(self, start, end, match):
        # self.print_state()

        M = len(self.tree_sequence_builder.edges)
        edges = self.tree_sequence_builder.edges
        u = self.max_likelihood_node[end - 1]
        output_edge = Edge(right=end, parent=u)
        output_edges = [output_edge]
        recombination_required = np.zeros(
            self.tree_sequence_builder.num_nodes, dtype=int) - 1

        # Now go back through the trees.
        j = M - 1
        k = M - 1
        I = self.tree_sequence_builder.removal_order
        # Construct the matched haplotype
        match[:] = 0
        match[:start] = UNKNOWN_ALLELE
        match[end:] = UNKNOWN_ALLELE
        self.parent[:] = -1
        # print("TB: max_likelihood node = ", u)
        pos = self.tree_sequence_builder.num_sites
        while pos > start:
            # print("Top of loop: pos = ", pos)
            while k >= 0 and edges[k].left == pos:
                self.parent[edges[k].child] = -1
                k -= 1
            while j >= 0 and edges[I[j]].right == pos:
                self.parent[edges[I[j]].child] = edges[I[j]].parent
                j -= 1
            right = pos
            left = 0
            if k >= 0:
                left = max(left, edges[k].left)
            if j >= 0:
                left = max(left, edges[I[j]].right)
            pos = left
            # print("tree:", left, right, "j = ", j, "k = ", k)

            assert left < right
            for l in range(min(right, end) - 1, max(left, start) - 1, -1):
                u = output_edge.parent
                if l in self.tree_sequence_builder.mutations:
                    if is_descendant(
                            self.parent, u,
                            self.tree_sequence_builder.mutations[l][0][0]):
                        match[l] = 1
                # print("TB: site = ", l)
                # print("traceback = ", self.traceback[l])
                for u, recombine in self.traceback[l].items():
                    # Mark the traceback nodes on the tree.
                    recombination_required[u] = recombine
                # print("set", recombination_required)
                # Now traverse up the tree from the current node. The first marked node
                # we meet tells us whether we need to recombine.
                u = output_edge.parent
                while recombination_required[u] == -1:
                    u = self.parent[u]
                if recombination_required[u]:
                    output_edge.left = l
                    u = self.max_likelihood_node[l - 1]
                    # print("Switch to ", u)
                    output_edge = Edge(right=l, parent=u)
                    output_edges.append(output_edge)
                # Reset the nodes in the recombination tree.
                for u in self.traceback[l].keys():
                    recombination_required[u] = -1
        output_edge.left = start

        self.mean_traceback_size = sum(len(t) for t in self.traceback) / self.num_sites
        # print("mathc h = ", match)

        left = np.zeros(len(output_edges), dtype=np.uint32)
        right = np.zeros(len(output_edges), dtype=np.uint32)
        parent = np.zeros(len(output_edges), dtype=np.int32)
        # print("returning edges:")
        for j, e in enumerate(output_edges):
            # print("\t", e.left, e.right, e.parent)
            assert e.left >= start
            assert e.right <= end
            # TODO this does happen in the C code, so if it ever happends in a Python
            # instance we need to pop the last edge off the list. Or, see why we're
            # generating it in the first place.
            assert e.left < e.right
            left[j] = e.left
            right[j] = e.right
            parent[j] = e.parent

        return left, right, parent
