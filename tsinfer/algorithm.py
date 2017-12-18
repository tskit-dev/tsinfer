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
import sortedcontainers

UNKNOWN_ALLELE = 255

class Edge(object):

    def __init__(
            self, left=None, right=None, parent=None, child=None, next=None):
        self.left = left
        self.right = right
        self.parent = parent
        self.child = child
        self.next = next

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

        # TMOP - hack to force different ancestors below.
        self.x = 0

    def add_site(self, site_id, frequency, genotypes):
        """
        Adds a new site at the specified ID and allele pattern to the builder.
        """
        self.sites[site_id] = Site(site_id, frequency, genotypes)
        if frequency > 1:
            pattern_map = self.frequency_map[frequency]
            # Each unique pattern gets added to the list
            # key = genotypes.tobytes()
            # FIXME Hacking this to ensure that we make a unique ancestor for
            # each site.
            if False:
                key = self.x
                self.x += 1
            else:
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
        self.print_state()
        ret = []
        for frequency in reversed(range(self.num_samples + 1)):
            # Need to make the order in which these are returned deterministic,
            # or ancestor IDs are not replicable between runs. In the C implementation
            # We sort by the genotype patterns
            keys = sorted(self.frequency_map[frequency].keys())
            focal_sites_list = [self.frequency_map[frequency][k] for k in keys]
            for focal_sites in focal_sites_list:
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
        self.mean_traceback_size = 0
        self.left_index = sortedcontainers.SortedDict()
        self.right_index = sortedcontainers.SortedDict()
        self.path_index = sortedcontainers.SortedDict()
        self.path = []

    def restore_nodes(self, time, flags):
        for t, flag in zip(time, flags):
            self.add_node(t, flag == 1)

    def add_node(self, time, is_sample=True):
        self.num_nodes += 1
        self.time.append(time)
        self.flags.append(int(is_sample))
        self.path.append(None)
        return self.num_nodes - 1

    def index_edge(self, edge):
        # Adds the specified edge to the indexes. Break ties between identical values
        # using the child ID.
        self.left_index[(edge.left, self.time[edge.child], edge.child)] = edge
        self.right_index[(edge.right, -self.time[edge.child], edge.child)] = edge
        # We need to find edges with identical (left, right, parent) values for
        # path compression.
        self.path_index[(edge.left, edge.right, edge.parent, edge.child)] = edge

    def index_edges(self, node_id):
        """
        Indexes the edges for the specified node ID.
        """
        edge = self.path[node_id]
        while edge != None:
            self.index_edge(edge)
            edge = edge.next

    def unindex_edge(self, edge):
        # Removes the specified edge from the indexes.
        del self.left_index[(edge.left, self.time[edge.child], edge.child)]
        del self.right_index[(edge.right, -self.time[edge.child], edge.child)]
        # We need to find edges with identical (left, right, parent) values for
        # path compression.
        del self.path_index[(edge.left, edge.right, edge.parent, edge.child)]

    def unindex_edges(self, node_id):
        """
        Removes the edges for the specified node from all the indexes.
        """
        edge = self.path[node_id]
        while edge is not None:
            self.unindex_edge(edge)
            edge = edge.next

    def squash_edges(self, head):
        """
        Squashes any edges in the specified chain that are redundant and
        returns the resulting segment chain. Specifically, edges
        (l, x, p, c) and (x, r, p, c) become (l, r, p, c).
        """
        # print("Before:", end="")
        # self.print_chain(head)
        prev = head
        x = head.next
        while x is not None:
            if prev.right == x.left and prev.child == x.child and prev.parent == x.parent:
                prev.right = x.right
                prev.next = x.next
            else:
                prev = x
            x = x.next
        # self.print_chain(head)
        return head

    def restore_edges(self, left, right, parent, child):
        edges = [Edge(l, r, p, c) for l, r, p, c in zip(left, right, parent, child)]
        # Sort the edges by child and left so we can add them in order.
        edges.sort(key=lambda e: (e.child, e.left))
        prev = Edge(child=-1)
        for edge in edges:
            if edge.child == prev.child:
                prev.next = edge
            else:
                self.path[edge.child] = edge
            self.index_edge(edge)
            prev = edge

        self.check_state()

    def add_path(self, child, left, right, parent, compress=True):
        assert self.path[child] == None
        prev = None
        head = None
        for l, r, p in reversed(list(zip(left, right, parent))):
            edge = Edge(l, r, p, child)
            if prev is None:
                head = edge
            else:
                prev.next = edge
            prev = edge

        if compress:
            head = self.compress_path(head)

        # Insert the chain into the global state.
        self.path[child] = head
        self.index_edges(child)
        self.print_state()
        self.check_state()

    def update_node_time(self, node_id):
        """
        Updates the node time for the specified synthetic node ID.
        """
        # print("Getting node time for ", node_id)
        assert self.flags[node_id] == 0
        edge = self.path[node_id]
        assert edge is not None
        min_parent_time = self.time[0] + 1
        while edge is not None:
            min_parent_time = min(min_parent_time, self.time[edge.parent])
            edge = edge.next
        assert min_parent_time >= 0
        assert min_parent_time <= self.time[0]
        # print("min_parent_time = ", min_parent_time)
        self.time[node_id] = min_parent_time - 0.1

    def remap_synthetic(self, child_id, matches):
        """
        Remap the edges in the set of matches to point to the already existing
        synthethic node.
        """
        for new, old in matches:
            if old.child == child_id:
                new.parent = child_id

    def create_synthetic_node(self, child_id, matches):

        # If we have more than one edge matching to a given path, then we create
        # synthetic ancestor for this path.
        # Create a new node for this synthetic ancestor.
        synthetic_node = self.add_node(-1, is_sample=False)
        self.unindex_edges(child_id)
        synthetic_head = None
        synthetic_prev = None
        # print("NEW SYNTHETIC FOR ", child_id, "->", mapped)
        for new, old in matches:
            if old.child == child_id:
                # print("\t", new, "\t", old)
                synthetic_edge = Edge(
                    old.left, old.right, old.parent, synthetic_node)
                if synthetic_prev is not None:
                    synthetic_prev.next = synthetic_edge
                if synthetic_head is None:
                    synthetic_head = synthetic_edge
                synthetic_prev = synthetic_edge
                new.parent = synthetic_node
                old.parent = synthetic_node
        # print("END of match loop")
        self.path[synthetic_node] = self.squash_edges(synthetic_head)
        self.path[child_id] = self.squash_edges(self.path[child_id])
        self.update_node_time(synthetic_node)
        self.index_edges(synthetic_node)
        self.index_edges(child_id)
        # self.print_chain(synthetic_head)
        # self.print_chain(self.path[child_id])
        # self.print_chain(head)

    def compress_path(self, head):
        """
        Tries to compress the path for the specified edge chain, and returns
        the resulting path.
        """
        # print("Compress for child:", head.child)
        edge = head
        matches = []
        while edge is not None:
            # print("\tConsidering ", edge.left, edge.right, edge.parent)
            key = (edge.left, edge.right, edge.parent, -1)
            index = self.path_index.bisect(key)
            if index < len(self.path_index) \
                    and self.path_index.iloc[index][:3] == (edge.left, edge.right, edge.parent):
                match = self.path_index.peekitem(index)[1]
                matches.append((edge, match))
            edge = edge.next

        matched_children_count = collections.Counter()
        for edge, match in matches:
            matched_children_count[match.child] += 1

        for child_id, count in matched_children_count.items():
            if count > 1:
                if self.flags[child_id] == 0:
                    self.remap_synthetic(child_id, matches)
                else:
                    self.create_synthetic_node(child_id, matches)
        return self.squash_edges(head)

    def restore_mutations(self, site, node, derived_state, parent):
        for s, u, d in zip(site, node, derived_state):
            self.mutations[s].append((u, d))

    def add_mutations(self, node, site, derived_state):
        for s, d in zip(site, derived_state):
            self.mutations[s].append((node, d))

    @property
    def num_edges(self):
        return len(self.left_index)

    @property
    def num_mutations(self):
        return sum(len(node_state_list) for node_state_list in self.mutations.values())

    def check_state(self):
        total_edges = 0
        for child in range(len(self.time)):
            edge = self.path[child]
            while edge is not None:
                assert edge.child == child
                if edge.next is not None:
                    if self.flags[child] != 0:
                        assert edge.next.left == edge.right
                assert self.left_index[(edge.left, self.time[child], child)] == edge
                assert self.right_index[(edge.right, -self.time[child], child)] == edge
                edge = edge.next
                total_edges += 1
        assert len(self.left_index) == total_edges
        assert len(self.right_index) == total_edges

    def print_chain(self, head):
        edge = head
        while edge is not None:
            print("({}, {}, {}, {})".format(
                edge.left, edge.right, edge.parent, edge.child), end="")
            edge = edge.next
        print()

    def print_state(self):
        print("TreeSequenceBuilder state")
        print("num_sites = ", self.num_sites)
        print("num_nodes = ", self.num_nodes)
        nodes = msprime.NodeTable()
        flags, time = self.dump_nodes()
        nodes.set_columns(flags=flags, time=time)
        print("nodes = ")
        print(nodes)
        for child in range(len(nodes)):
            print("child = ", child, end="\t")
            self.print_chain(self.path[child])
        self.check_state()

    def dump_nodes(self):
        time = self.time[:]
        flags = self.flags[:]
        return flags, time

    def dump_edges(self):
        left = np.zeros(self.num_edges, dtype=np.int32)
        right = np.zeros(self.num_edges, dtype=np.int32)
        parent = np.zeros(self.num_edges, dtype=np.int32)
        child = np.zeros(self.num_edges, dtype=np.int32)
        j = 0
        for c in range(self.num_nodes):
            edge = self.path[c]
            while edge is not None:
                left[j] = edge.left
                right[j] = edge.right
                parent[j] = edge.parent
                child[j] = edge.child
                edge = edge.next
                j += 1
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
            print(l, self.max_likelihood_node[l], self.traceback[l], sep="\t")

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
                # print("inserted likelihood for ", mutation_node, self.likelihood[u])

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
            if mutation_node == -1:
                emission_p = 1 - err
            else:
                if state == 1:
                    emission_p = (1 - err) * d + err * (not d)
                else:
                    emission_p = err * d + (1 - err) * (not d)
            self.likelihood[u] = z * emission_p
            if self.likelihood[u] > max_L:
                max_L = self.likelihood[u]
                max_L_node = u

        # print("site=", site, "Max L = ", max_L, "node = ", max_L_node)
        # print("L = ", {u: self.likelihood[u] for u in self.likelihood_nodes})

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

        # M = len(self.tree_sequence_builder.edges)
        Il = self.tree_sequence_builder.left_index
        Ir = self.tree_sequence_builder.right_index
        M = len(Il)
        n = self.tree_sequence_builder.num_nodes
        m = self.tree_sequence_builder.num_sites
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

        print("MATCH: start=", start, "end = ", end, "h = ", h)
        j = 0
        k = 0
        left = 0
        pos = 0
        right = m
        while j < M and k < M and Il.peekitem(j)[1].left <= start:
            # while edges[O[k]].right == pos:
            while Ir.peekitem(k)[1].right == pos:
                # self.remove_edge(edges[O[k]])
                self.remove_edge(Ir.peekitem(k)[1])
                k += 1
            # while j < M and edges[j].left == pos:
            while j < M and Il.peekitem(j)[1].left == pos:
                # self.insert_edge(edges[j])
                self.insert_edge(Il.peekitem(j)[1])
                j += 1
            left = pos
            right = m
            if j < M:
                # right = min(right, edges[j].left)
                right = min(right, Il.peekitem(j)[1].left)
            if k < M:
                # right = min(right, edges[O[k]].right)
                right = min(right, Ir.peekitem(k)[1].right)
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
                # edge = edges[O[l]]
                edge = Ir.peekitem(l)[1]
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
            # while k < M and edges[O[k]].right == right:
            while k < M and Ir.peekitem(k)[1].right == right:
                # edge = edges[O[k]]
                edge = Ir.peekitem(k)[1]
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
                # edge = edges[O[l]]
                edge = Ir.peekitem(l)[1]
                u = edge.parent
                while L_cache[u] != -1:
                    L_cache[u] = -1
                    u = self.parent[u]
            assert np.all(L_cache == -1)

            left = right
            # while j < M and edges[j].left == left:
            #     edge = edges[j]
            while j < M and Il.peekitem(j)[1].left == left:
                edge = Il.peekitem(j)[1]
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
                # right = min(right, edges[j].left)
                right = min(right, Il.peekitem(j)[1].left)
            if k < M:
                # right = min(right, edges[O[k]].right)
                right = min(right, Ir.peekitem(k)[1].right)

        return self.run_traceback(start, end, match)

    def run_traceback(self, start, end, match):
        self.print_state()
        Il = self.tree_sequence_builder.left_index
        Ir = self.tree_sequence_builder.right_index
        M = len(Il)
        u = self.max_likelihood_node[end - 1]
        output_edge = Edge(right=end, parent=u)
        output_edges = [output_edge]
        recombination_required = np.zeros(
            self.tree_sequence_builder.num_nodes, dtype=int) - 1

        # Now go back through the trees.
        j = M - 1
        k = M - 1
        # Construct the matched haplotype
        match[:] = 0
        match[:start] = UNKNOWN_ALLELE
        match[end:] = UNKNOWN_ALLELE
        self.parent[:] = -1
        # print("TB: max_likelihood node = ", u)
        pos = self.tree_sequence_builder.num_sites
        while pos > start:
            # print("Top of loop: pos = ", pos)
            # while k >= 0 and edges[k].left == pos:
            #     self.parent[edges[k].child] = -1
            while k >= 0 and Il.peekitem(k)[1].left == pos:
                edge = Il.peekitem(k)[1]
                self.parent[edge.child] = -1
                k -= 1
            # while j >= 0 and edges[I[j]].right == pos:
            #     self.parent[edges[I[j]].child] = edges[I[j]].parent
            while j >= 0 and Ir.peekitem(j)[1].right == pos:
                edge = Ir.peekitem(j)[1]
                self.parent[edge.child] = edge.parent
                j -= 1
            right = pos
            left = 0
            if k >= 0:
                # left = max(left, edges[k].left)
                left = max(left, Il.peekitem(k)[1].left)
            if j >= 0:
                # left = max(left, edges[I[j]].right)
                left = max(left, Ir.peekitem(j)[1].right)
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
        print("returning edges:")
        for j, e in enumerate(output_edges):
            print("\t", e.left, e.right, e.parent)
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


class MatrixAncestorMatcher(object):

    def __init__(self, tree_sequence_builder, error_rate=0):
        self.tree_sequence_builder = tree_sequence_builder
        self.error_rate = error_rate
        self.num_sites = tree_sequence_builder.num_sites
        self.positions = tree_sequence_builder.positions
        # Just to given the right API.
        self.mean_traceback_size = 0
        self.total_memory = 0

    def ancestor_matrix(self):
        tsb = self.tree_sequence_builder
        flags, time = tsb.dump_nodes()
        nodes = msprime.NodeTable()
        nodes.set_columns(flags=flags, time=time)

        left, right, parent, child = tsb.dump_edges()
        position = np.arange(tsb.num_sites)
        sequence_length = tsb.num_sites

        edges = msprime.EdgeTable()
        edges.set_columns(left=left, right=right, parent=parent, child=child)

        sites = msprime.SiteTable()
        sites.set_columns(
            position=position,
            ancestral_state=np.zeros(tsb.num_sites, dtype=np.int8) + ord('0'),
            ancestral_state_length=np.ones(tsb.num_sites, dtype=np.uint32))
        mutations = msprime.MutationTable()
        site = np.zeros(tsb.num_mutations, dtype=np.int32)
        node = np.zeros(tsb.num_mutations, dtype=np.int32)
        parent = np.zeros(tsb.num_mutations, dtype=np.int32)
        derived_state = np.zeros(tsb.num_mutations, dtype=np.int8)
        site, node, derived_state, parent = tsb.dump_mutations()
        derived_state += ord('0')
        mutations.set_columns(
            site=site, node=node, derived_state=derived_state,
            derived_state_length=np.ones(tsb.num_mutations, dtype=np.uint32),
            parent=parent)
        msprime.sort_tables(nodes, edges, sites=sites, mutations=mutations)
        ts = msprime.load_tables(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            sequence_length=sequence_length)
        n = 0
        if len(edges) > 0:
            n = np.max(edges.child)
        H = ts.genotype_matrix().T[:n + 1]

        mask = np.zeros_like(H)
        # Set all the edges
        for edge in ts.edges():
            mask[edge.child, int(edge.left): int(edge.right)] = 1
        mask[0,:] = 1
        H[mask == 0] = -1
        return H


    def find_path(self, h, start, end, match):
        H = self.ancestor_matrix().astype(np.int8)
        print("H = ")
        print(H)
        print("Find path", start, end)
        print(h)
        recombination_rate = 1
        mutations = self.tree_sequence_builder.mutations

        n, m = H.shape
        r = 1 - np.exp(-recombination_rate / n)
        recomb_proba = r / n
        no_recomb_proba = 1 - r + r / n
        L = np.ones(n)
        T = [set() for _ in range(m)]
        T_dest = np.zeros(m, dtype=int)
        match[:] = -1

        for l in range(start, end):
            L_next = np.zeros(n)
            for j in range(n):
                x = L[j] * no_recomb_proba
                y = recomb_proba
                if x > y:
                    z = x
                else:
                    z = y
                    T[l].add(j)
                if H[j, l] == -1:
                    # Can never match to the missing data.
                    emission_p = 0
                else:
                    if l in mutations:
                        emission_p = int(H[j, l] == h[l])
                    else:
                        emission_p = 1
                L_next[j] = z * emission_p
            # Find the max and renormalise
            L = L_next
            j = np.argmax(L)
            T_dest[l] = j
            L /= L[j]
            print(l, ":", L)
        print("T_dest = ", T_dest)
        print("T = ", T)

        p = T_dest[end - 1]
        parent = [p]
        left = []
        right = [end]
        for l in range(end - 1, start - 1, -1):
            print("TB: l = ", l, "p  = " ,p)
            match[l] = H[p, l]
            if p in T[l]:
                assert l != 0
                print("SWITCH")
                p = T_dest[l - 1]
                parent.append(p)
                right.append(l)
                left.append(l)
        left.append(start)
        return (
            np.array(left, dtype=np.int32),
            np.array(right, dtype=np.int32),
            np.array(parent, dtype=np.int32))


