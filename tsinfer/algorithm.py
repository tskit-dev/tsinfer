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

import tsinfer.constants as constants


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
    # TODO this implementation currently partially allows for multiple focal
    # sites per ancestor, but the final generation algorithm assumes a single
    # focal site. Once this is finalise should refactor to remove the complexity
    # needed for matching up ancestors with identical focal sites.
    def __init__(self, num_samples, num_sites):
        self.num_samples = num_samples
        self.num_sites = num_sites
        self.sites = [None for _ in range(self.num_sites)]
        self.frequency_map = [{} for _ in range(self.num_samples + 1)]

    def add_site(self, site_id, frequency, genotypes):
        """
        Adds a new site at the specified ID and allele pattern to the builder.
        """
        assert frequency > 1
        self.sites[site_id] = Site(site_id, frequency, genotypes)
        pattern_map = self.frequency_map[frequency]
        # Each unique pattern gets added to the list
        key = genotypes.tobytes()
        if key not in pattern_map:
            pattern_map[key] = []
        pattern_map[key].append(site_id)

    def print_state(self):
        print("Ancestor builder")
        print("Sites = ")
        for j in range(self.num_sites):
            site = self.sites[j]
            print(j, site.frequency, site.genotypes, sep="\t")
        print("Frequency map")
        for f in range(self.num_samples):
            pattern_map = self.frequency_map[f]
            if len(pattern_map) > 0:
                print("f = ", f, "with ", len(pattern_map), "patterns")
                for pattern, sites in pattern_map.items():
                    print("\t", pattern, ":", sites)

    def break_ancestor(self, a, b, samples):
        """
        Returns True if we should split the ancestor with focal sites at
        a and b into two separate ancestors.
        """
        # return True
        index = np.where(samples == 1)[0]
        for j in range(a + 1, b):
            if self.sites[j].frequency > self.sites[a].frequency:
                gj = self.sites[j].genotypes[index]
                if not (np.all(gj == 1) or np.all(gj == 0)):
                    return True
        return False

    def ancestor_descriptors(self):
        """
        Returns a list of (frequency, focal_sites) tuples describing the
        ancestors in reverse order of frequency.
        """
        # self.print_state()
        ret = []
        for frequency in reversed(range(self.num_samples + 1)):
            # Need to make the order in which these are returned deterministic,
            # or ancestor IDs are not replicable between runs. In the C implementation
            # We sort by the genotype patterns
            keys = sorted(self.frequency_map[frequency].keys())
            for key in keys:
                focal_sites = np.array(
                    self.frequency_map[frequency][key], dtype=np.int32)
                samples = np.frombuffer(key, dtype=np.uint8)
                # print("focal_sites = ", key, samples, focal_sites)
                start = 0
                for j in range(len(focal_sites) - 1):
                    if self.break_ancestor(focal_sites[j], focal_sites[j + 1], samples):
                        ret.append((frequency, focal_sites[start: j + 1]))
                        start = j + 1
                ret.append((frequency, focal_sites[start:]))
        return ret

    def compute_ancestral_states(self, a, focal_site, sites):
        focal_frequency = self.sites[focal_site].frequency
        min_sample_set_size = focal_frequency // 2
        S = set(np.where(self.sites[focal_site].genotypes == 1)[0])
        remove_buffer = []
        last_site = focal_site
        # print("Computing for ", focal_site)
        for l in sites:
            a[l] = 0
            last_site = l
            if self.sites[l].frequency > focal_frequency:
                g_l = self.sites[l].genotypes
                ones = sum(g_l[u] for u in S)
                zeros = len(S) - ones
                # print("\t", l, ones, zeros, sep="\t")
                consensus = 0
                if ones >= zeros:
                    consensus = 1
                # print("\tP", l, "\t", len(S), ":ones=", ones, consensus)
                for u in remove_buffer:
                    if g_l[u] != consensus:
                        # print("\t\tremoving", u)
                        S.remove(u)
                # print("\t", len(S), remove_buffer, consensus, sep="\t")
                if len(S) <= min_sample_set_size:
                    # print("BREAKING", len(S), min_sample_set_size)
                    break
                remove_buffer.clear()
                for u in S:
                    if g_l[u] != consensus:
                        remove_buffer.append(u)
                a[l] = consensus
        return last_site

    def make_ancestor(self, focal_sites, a):
        a[:] = constants.UNKNOWN_ALLELE
        for focal_site in focal_sites:
            a[focal_site] = 1
        focal_frequency = self.sites[focal_sites[0]].frequency
        S = set(np.where(self.sites[focal_sites[0]].genotypes == 1)[0])
        for j in range(len(focal_sites) - 1):
            for l in range(focal_sites[j] + 1, focal_sites[j + 1]):
                a[l] = 0
                if self.sites[l].frequency > focal_frequency:
                    g_l = self.sites[l].genotypes
                    ones = sum(g_l[u] for u in S)
                    zeros = len(S) - ones
                    # print("\t", l, ones, zeros, sep="\t")
                    if ones >= zeros:
                        a[l] = 1
        focal_site = focal_sites[-1]
        last_site = self.compute_ancestral_states(
                a, focal_site, range(focal_site + 1, self.num_sites))
        assert a[last_site] != constants.UNKNOWN_ALLELE
        end = last_site + 1
        focal_site = focal_sites[0]
        last_site = self.compute_ancestral_states(
                a, focal_site, range(focal_site - 1, -1, -1))
        assert a[last_site] != constants.UNKNOWN_ALLELE
        start = last_site
        return start, end

        # Version with 1 focal site
        # assert len(focal_sites) == 1
        # focal_site = focal_sites[0]
        # a[:] = UNKNOWN_ALLELE
        # a[focal_site] = 1

        # last_site = self.compute_ancestral_states(
        #         a, focal_site, range(focal_site + 1, self.num_sites))
        # assert a[last_site] != UNKNOWN_ALLELE
        # end = last_site + 1
        # last_site = self.compute_ancestral_states(
        #         a, focal_site, range(focal_site - 1, -1, -1))
        # assert a[last_site] != UNKNOWN_ALLELE
        # start = last_site
        # return start, end


class TreeSequenceBuilder(object):

    def __init__(self, num_sites, max_nodes, max_edges):
        self.num_sites = num_sites
        self.num_nodes = 0
        self.time = []
        self.flags = []
        self.mutations = collections.defaultdict(list)
        self.mean_traceback_size = 0
        self.left_index = sortedcontainers.SortedDict()
        self.right_index = sortedcontainers.SortedDict()
        self.path_index = sortedcontainers.SortedDict()
        self.path = []

    def freeze_indexes(self):
        # This is a no-op in this implementation.
        pass

    def restore_nodes(self, time, flags):
        for t, flag in zip(time, flags):
            self.add_node(t, flag)

    def add_node(self, time, flags=1):
        self.num_nodes += 1
        self.time.append(time)
        self.flags.append(flags)
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
        while edge is not None:
            self.index_edge(edge)
            edge = edge.next

    def unindex_edge(self, edge):
        # Removes the specified edge from the indexes.
        del self.left_index[(edge.left, self.time[edge.child], edge.child)]
        del self.right_index[(edge.right, -self.time[edge.child], edge.child)]
        # We need to find edges with identical (left, right, parent) values for
        # path compression.
        del self.path_index[(edge.left, edge.right, edge.parent, edge.child)]

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
            assert prev.child == x.child
            if prev.right == x.left and prev.parent == x.parent:
                prev.right = x.right
                prev.next = x.next
            else:
                prev = x
            x = x.next
        # self.print_chain(head)
        return head

    def squash_edges_indexed(self, head, child_id):
        """
        Works as squash_edges above, but is aware that some of the edges are
        indexed and some are not. Edges that have been unindexed are marked
        with a child value of -1.
        """
        prev = head
        x = head.next
        while x is not None:
            if prev.right == x.left and prev.parent == x.parent:
                if prev.child != -1:
                    self.unindex_edge(prev)
                    prev.child = -1
                if x.child != -1:
                    self.unindex_edge(x)
                prev.right = x.right
                prev.next = x.next
            else:
                prev = x
            x = x.next

        # Now go through and index all the remaining edges.
        x = head
        while x is not None:
            if x.child == -1:
                x.child = child_id
                self.index_edge(x)
            x = x.next
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

    def add_path(self, child, left, right, parent, compress=True, extended_checks=False):
        assert self.path[child] is None
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
        # self.print_state()
        if extended_checks:
            self.check_state()

    def update_node_time(self, child_id, pc_parent_id):
        """
        Updates the node time for the specified pc parent node ID.
        """
        assert self.flags[pc_parent_id] == constants.NODE_IS_PC_ANCESTOR
        edge = self.path[pc_parent_id]
        assert edge is not None
        min_parent_time = self.time[0] + 1
        while edge is not None:
            min_parent_time = min(min_parent_time, self.time[edge.parent])
            edge = edge.next
        assert min_parent_time >= 0
        assert min_parent_time <= self.time[0]
        # For the asserttion to be violated we would need to have 64K pc
        # ancestors sequentially copying from each other.
        self.time[pc_parent_id] = min_parent_time - (1 / 2**16)
        assert self.time[pc_parent_id] > self.time[child_id]

    def create_pc_node(self, matches):
        # If we have more than one edge matching to a given path, then we create
        # a path-compression ancestor for this path.
        # Create a new node for this pc ancestor.
        pc_node = self.add_node(-1, constants.NODE_IS_PC_ANCESTOR)
        pc_head = None
        pc_prev = None
        child_id = matches[0][1].child
        # print("NEW SYNTHETIC", child_id, "@", self.time[child_id], "=", pc_node)
        # print("BEFORE")
        # self.print_chain(self.path[child_id])
        for new, old in matches:
            # print("\t", old)
            # print("\t", new, "\t", old)
            assert new.left == old.left
            assert new.right == old.right
            assert new.parent == old.parent
            pc_edge = Edge(old.left, old.right, old.parent, pc_node)
            if pc_prev is not None:
                pc_prev.next = pc_edge
            if pc_head is None:
                pc_head = pc_edge
            pc_prev = pc_edge
            new.parent = pc_node
            # We are modifying this existing edge, so remove it from the
            # index. Also mark it as unindexed by setting the child_id to -1.
            # We check for this in squash_edges_indexed and make sure it
            # is indexed afterwards.
            self.unindex_edge(old)
            old.parent = pc_node
            old.child = -1

        self.path[pc_node] = self.squash_edges(pc_head)
        self.path[child_id] = self.squash_edges_indexed(self.path[child_id], child_id)
        self.update_node_time(child_id, pc_node)
        self.index_edges(pc_node)
        # print("AFTER")
        # self.print_chain(pc_head)
        # self.print_chain(self.path[child_id])

    def compress_path(self, head):
        """
        Tries to compress the path for the specified edge chain, and returns
        the resulting path.
        """
        # print("Compress for child:", head.child)
        edge = head
        # Find all edges in the index that have the same (left, right, parent)
        # values as edges in the edge path for this child.
        matches = []
        contig_offsets = []
        last_match = msprime.Edge(-1, -1, -1, -1)
        while edge is not None:
            # print("\tConsidering ", edge.left, edge.right, edge.parent)
            key = (edge.left, edge.right, edge.parent, -1)
            index = self.path_index.bisect(key)
            if index < len(self.path_index) \
                    and self.path_index.iloc[index][:3] == (
                            edge.left, edge.right, edge.parent):
                match = self.path_index.peekitem(index)[1]
                matches.append((edge, match))
                condition = (
                    edge.left == last_match.right and
                    match.child == last_match.child)
                if not condition:
                    contig_offsets.append(len(matches) - 1)
                last_match = match
            edge = edge.next
        contig_offsets.append(len(matches))

        # FIXME This is just to check the contig finding code above. Remove.
        contiguous_matches = [[(None, msprime.Edge(-1, -1, -1, -1))]]  # Sentinel
        for edge, match in matches:
            condition = (
                edge.left == contiguous_matches[-1][-1][1].right and
                match.child == contiguous_matches[-1][-1][1].child)
            if condition:
                contiguous_matches[-1].append((edge, match))
            else:
                contiguous_matches.append([(edge, match)])
        other_matches = [None]
        for j in range(len(contig_offsets) - 1):
            contigs = matches[contig_offsets[j]: contig_offsets[j + 1]]
            other_matches.append(contigs)
        assert len(other_matches) == len(contiguous_matches)
        for c1, c2 in zip(contiguous_matches[1:], other_matches[1:]):
            assert c1 == c2

        for j in range(len(contig_offsets) - 1):
            match_list = matches[contig_offsets[j]: contig_offsets[j + 1]]
            if len(match_list) > 1:
                child_id = match_list[0][1].child
                # print("MATCH:", child_id)
                if self.flags[child_id] == constants.NODE_IS_PC_ANCESTOR:
                    # print("EXISTING SYNTHETIC")
                    for edge, match in match_list:
                        # print("\t", edge, match)
                        edge.parent = child_id
                else:
                    # print("NEW SYNTHETIC")
                    self.create_pc_node(match_list)
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
                        assert edge.next.left >= edge.right
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
        time = np.array(self.time[:])
        flags = np.array(self.flags[:], dtype=np.uint32)
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
    # print("IS_DESCENDENT(", u, v, ") = ", ret)
    return ret

# We have a 3 valued number system for match likelihoods, plus two
# more values to indicate compressed paths tree and nodes that are
# currently not in the tree.


MISMATCH = 0
RECOMB = 1
MATCH = 2
COMPRESSED = -1
MISSING = -2


class AncestorMatcher(object):

    def __init__(self, tree_sequence_builder, extended_checks=False):
        self.tree_sequence_builder = tree_sequence_builder
        self.extended_checks = extended_checks
        self.num_sites = tree_sequence_builder.num_sites
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
        assert len(set(self.likelihood_nodes)) == len(self.likelihood_nodes)
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

        mutation_node = msprime.NULL_NODE
        if site in self.tree_sequence_builder.mutations:
            mutation_node = self.tree_sequence_builder.mutations[site][0][0]
            # Insert an new L-value for the mutation node if needed.
            if self.likelihood[mutation_node] == COMPRESSED:
                u = mutation_node
                while self.likelihood[u] == COMPRESSED:
                    u = self.parent[u]
                self.likelihood[mutation_node] = self.likelihood[u]
                self.likelihood_nodes.append(mutation_node)
                # print("inserted likelihood for ", mutation_node, self.likelihood[u])

        # print("update_site", site, state, mutation_node)
        # print("Site ", site, "mutation = ", mutation_node, "state = ", state,
        #         {u:self.likelihood[u] for u in self.likelihood_nodes})

        path_cache = np.zeros(n, dtype=np.int8) - 1
        max_L = -1
        max_L_node = -1
        for u in self.likelihood_nodes:
            d = 0
            if mutation_node != -1:
                v = u
                while v != -1 and v != mutation_node and path_cache[v] == -1:
                    v = self.parent[v]
                if v != -1 and path_cache[v] != -1:
                    d = path_cache[v]
                else:
                    d = int(v == mutation_node)
                assert d == is_descendant(self.parent, u, mutation_node)
                # Insert this path into the cache.
                v = u
                while v != -1 and v != mutation_node and path_cache[v] == -1:
                    path_cache[v] = d
                    v = self.parent[v]

            self.traceback[site][u] = False
            if self.likelihood[u] == MISMATCH:
                self.traceback[site][u] = True
            if mutation_node != -1 and d != state:
                self.likelihood[u] = MISMATCH
            elif self.likelihood[u] == MISMATCH:
                self.likelihood[u] = RECOMB

            if self.likelihood[u] > max_L:
                max_L = self.likelihood[u]
                max_L_node = u

        if max_L != MATCH:
            for u in self.likelihood_nodes:
                if self.likelihood[u] == max_L:
                    self.likelihood[u] = MATCH
        self.max_likelihood_node[site] = max_L_node

        # Reset the path cache
        for u in self.likelihood_nodes:
            v = u
            while v != -1 and path_cache[v] != -1:
                path_cache[v] = -1
                v = self.parent[v]
        assert np.all(path_cache == -1)
        self.compress_likelihoods()

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

    def find_path(self, h, start, end, match):
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

        self.likelihood = np.zeros(n, dtype=np.int8) - 2
        self.likelihood_nodes = []
        L_cache = np.zeros_like(self.likelihood) - 1

        # print("MATCH: start=", start, "end = ", end, "h = ", h)
        j = 0
        k = 0
        left = 0
        pos = 0
        right = m
        if j < M and start < Il.peekitem(j)[1].left:
            right = Il.peekitem(j)[1].left
        while j < M and k < M and Il.peekitem(j)[1].left <= start:
            while Ir.peekitem(k)[1].right == pos:
                self.remove_edge(Ir.peekitem(k)[1])
                k += 1
            while j < M and Il.peekitem(j)[1].left == pos:
                self.insert_edge(Il.peekitem(j)[1])
                j += 1
            left = pos
            right = m
            if j < M:
                right = min(right, Il.peekitem(j)[1].left)
            if k < M:
                right = min(right, Ir.peekitem(k)[1].right)
            pos = right
        assert left < right

        for u in range(n):
            if self.parent[u] != -1:
                self.likelihood[u] = -1

        last_root = 0
        if self.left_child[0] != -1:
            last_root = self.left_child[0]
            assert self.right_sib[last_root] == -1
        self.likelihood_nodes.append(last_root)
        self.likelihood[last_root] = MATCH

        remove_start = k
        while left < end:
            # print("START OF TREE LOOP", left, right)
            assert left < right
            for l in range(remove_start, k):
                edge = Ir.peekitem(l)[1]
                for u in [edge.parent, edge.child]:
                    if self.is_nonzero_root(u):
                        self.likelihood[u] = MISSING
                        if u in self.likelihood_nodes:
                            self.likelihood_nodes.remove(u)
            root = 0
            if self.left_child[0] != -1:
                root = self.left_child[0]
                assert self.right_sib[root] == -1

            if root != last_root:
                if last_root == 0:
                    self.likelihood[last_root] = MISSING
                    self.likelihood_nodes.remove(last_root)
                if self.likelihood[root] == MISSING:
                    self.likelihood[root] = 0
                    self.likelihood_nodes.append(root)
                last_root = root

            if self.extended_checks:
                self.check_likelihoods()
            for site in range(max(left, start), min(right, end)):
                self.update_site(site, h[site])

            remove_start = k
            while k < M and Ir.peekitem(k)[1].right == right:
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
                edge = Ir.peekitem(l)[1]
                u = edge.parent
                while L_cache[u] != -1:
                    L_cache[u] = -1
                    u = self.parent[u]
            assert np.all(L_cache == -1)

            left = right
            while j < M and Il.peekitem(j)[1].left == left:
                edge = Il.peekitem(j)[1]
                self.insert_edge(edge)
                j += 1
                # There's no point in compressing the likelihood tree here as we'll be
                # doing it after we update the first site anyway.
                for u in [edge.parent, edge.child]:
                    if u != 0 and self.likelihood[u] == MISSING:
                        self.likelihood[u] = MISMATCH
                        self.likelihood_nodes.append(u)
            right = m
            if j < M:
                right = min(right, Il.peekitem(j)[1].left)
            if k < M:
                right = min(right, Ir.peekitem(k)[1].right)

        return self.run_traceback(start, end, match)

    def run_traceback(self, start, end, match):
        # print("traceback", start, end)
        # self.print_state()
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
        match[:start] = constants.UNKNOWN_ALLELE
        match[end:] = constants.UNKNOWN_ALLELE
        # Reset the tree.
        self.parent[:] = -1
        self.left_child[:] = -1
        self.right_child[:] = -1
        self.left_sib[:] = -1
        self.right_sib[:] = -1
        # print("TB: max_likelihood node = ", u)
        pos = self.tree_sequence_builder.num_sites
        while pos > start:
            # print("Top of loop: pos = ", pos)
            while k >= 0 and Il.peekitem(k)[1].left == pos:
                edge = Il.peekitem(k)[1]
                self.remove_edge(edge)
                k -= 1
            while j >= 0 and Ir.peekitem(j)[1].right == pos:
                edge = Ir.peekitem(j)[1]
                self.insert_edge(edge)
                j -= 1
            right = pos
            left = 0
            if k >= 0:
                left = max(left, Il.peekitem(k)[1].left)
            if j >= 0:
                left = max(left, Ir.peekitem(j)[1].right)
            pos = left
            # print("tree:", left, right, "j = ", j, "k = ", k)

            assert left < right
            for l in range(min(right, end) - 1, max(left, start) - 1, -1):
                u = output_edge.parent
                # print("TB: site = ", l, u)
                if l in self.tree_sequence_builder.mutations:
                    if is_descendant(
                            self.parent, u,
                            self.tree_sequence_builder.mutations[l][0][0]):
                        match[l] = 1
                # print("traceback = ", self.traceback[l])
                for u, recombine in self.traceback[l].items():
                    # Mark the traceback nodes on the tree.
                    recombination_required[u] = recombine
                # Now traverse up the tree from the current node. The first marked node
                # we meet tells us whether we need to recombine.
                u = output_edge.parent
                while u != 0 and recombination_required[u] == -1:
                    u = self.parent[u]
                if recombination_required[u] and l > start:
                    output_edge.left = l
                    u = self.max_likelihood_node[l - 1]
                    # print("\tSwitch to ", u)
                    output_edge = Edge(right=l, parent=u)
                    output_edges.append(output_edge)
                # Reset the nodes in the recombination tree.
                for u in self.traceback[l].keys():
                    recombination_required[u] = -1
        output_edge.left = start

        self.mean_traceback_size = sum(len(t) for t in self.traceback) / self.num_sites
        # print("match = ", match)
        # for j, e in enumerate(output_edges):

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
