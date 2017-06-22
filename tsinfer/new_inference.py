
import collections
import concurrent
import random
import threading
import math

import numpy as np
import attr

import msprime

import _tsinfer

def split_parent_array(P):
    num_sites = P.shape[0]
    start = 0
    for j in range(num_sites - 1):
        if P[j + 1] != P[j]:
            if P[j] != -1:
                yield start, j + 1, P[j]
            start = j + 1
    if P[-1] != -1:
        yield start, num_sites, P[-1]



def build_ancestors(samples, positions, num_threads=1):
    num_samples, num_sites = samples.shape
    builder = _tsinfer.AncestorBuilder(samples, positions)
    store_builder = _tsinfer.AncestorStoreBuilder(
            builder.num_sites, 8192 * builder.num_sites)

    def build_frequency_class(work):
        frequency, focal_sites = work
        num_ancestors = len(focal_sites)
        A = np.zeros((num_ancestors, builder.num_sites), dtype=np.int8)
        p = np.zeros(num_ancestors, dtype=np.uint32)
        # print("frequency:", frequency, "sites = ", focal_sites)
        for j, focal_site in enumerate(focal_sites):
            builder.make_ancestor(focal_site, A[j, :])
        _tsinfer.sort_ancestors(A, p)
        # p = np.arange(num_ancestors, dtype=np.uint32)
        return frequency, A, p

    frequency_classes = builder.get_frequency_classes()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for result in executor.map(build_frequency_class, frequency_classes):
            frequency, A, p = result
            for index in p:
                store_builder.add(A[index, :])

    N = store_builder.total_segments
    site = np.zeros(N, dtype=np.uint32)
    start = np.zeros(N, dtype=np.int32)
    end = np.zeros(N, dtype=np.int32)
    state = np.zeros(N, dtype=np.int8)
    store_builder.dump_segments(site, start, end, state)

    store = _tsinfer.AncestorStore(
        num_sites=builder.num_sites, site=site, start=start, end=end, state=state)
    return store

def match_ancestors(
        store, recombination_rate, tree_sequence_builder, num_threads=1, method="C"):
    ancestor_ids = list(range(1, store.num_ancestors))
    def ancestor_match_worker(ancestor_id):
        if method == "C":
            matcher = _tsinfer.AncestorMatcher(store, recombination_rate)
            traceback = _tsinfer.Traceback(store, 2**10)
        else:
            matcher = AncestorMatcher(store, recombination_rate)
            traceback = Traceback(store)
        h = np.zeros(store.num_sites, dtype=np.int8)
        P = np.zeros(store.num_sites, dtype=np.int32)
        M = np.zeros(store.num_sites, dtype=np.uint32)
        start_site, end_site = store.get_ancestor(ancestor_id, h)
        # print(start_site, end_site)
        # a = "".join(str(x) if x != -1 else '*' for x in h)
        # print(ancestor_id, "\t", a)
        best_match = matcher.best_path(
                ancestor_id, h, start_site, end_site, 1e-200, traceback)
        num_mutations = traceback.run(h, start_site, end_site, best_match, P, M)
        traceback.reset()
        assert num_mutations == 1
        return ancestor_id, h, P, M[:num_mutations]

    if num_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for result in executor.map(ancestor_match_worker, ancestor_ids):
                ancestor_id, h, P, M = result
                tree_sequence_builder.add_path(ancestor_id, P, h, M)
    else:
        for result in map(ancestor_match_worker, ancestor_ids):
            ancestor_id, h, P, M = result
            print("consumed:", ancestor_id)
            tree_sequence_builder.add_path(ancestor_id, P, h, M)

def match_samples(
        store, samples, recombination_rate, error_rate, tree_sequence_builder,
        num_threads=1, method="C"):
    sample_ids = list(range(samples.shape[0]))

    def sample_match_worker(sample_id):

        if method == "C":
            traceback = _tsinfer.Traceback(store, 2**10)
            matcher = _tsinfer.AncestorMatcher(store, recombination_rate)
        else:
            traceback = Traceback(store)
            matcher = AncestorMatcher(store, recombination_rate)
        h = np.zeros(store.num_sites, dtype=np.int8)
        P = np.zeros(store.num_sites, dtype=np.int32)
        M = np.zeros(store.num_sites, dtype=np.uint32)
        h = samples[sample_id, :]
        best_match = matcher.best_path(
                store.num_ancestors, h, 0, store.num_sites, error_rate, traceback)
        num_mutations = traceback.run(h, 0, store.num_sites, best_match, P, M)
        return sample_id, h, P, M[:num_mutations]

    if num_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for result in executor.map(sample_match_worker, sample_ids):
                sample_id, h, P, M = result
                tree_sequence_builder.add_path(
                    store.num_ancestors + sample_id + 1, P, h, M)
    else:
        for result in map(sample_match_worker, sample_ids):
            sample_id, h, P, M = result
            sample_id, h, P, M = result
            tree_sequence_builder.add_path(
                store.num_ancestors + sample_id + 1, P, h, M)


def infer(samples, positions, recombination_rate, error_rate, method="C",
        num_threads=1):
    store = build_ancestors(samples, positions, num_threads=num_threads)
    num_samples, num_sites = samples.shape
    matcher = _tsinfer.AncestorMatcher(store, recombination_rate)

    tree_sequence_builder = TreeSequenceBuilder(num_samples, store.num_ancestors, num_sites)
    match_ancestors(
        store, recombination_rate, tree_sequence_builder, method=method,
        num_threads=num_threads)
    match_samples(
        store, samples, recombination_rate, error_rate, tree_sequence_builder,
        method=method, num_threads=num_threads)

    tree_sequence_builder.print_state()
    ts = tree_sequence_builder.finalise()

    # for e in ts.edgesets():
    #     print(e)
    # for t in ts.trees():
    #     print(t)

    # print()
    # tss = ts.simplify()
    # for e in tss.edgesets():
    #     print(e)
    # for t in tss.trees():
    #     print(t)
    # return tss
    return ts


@attr.s
class LinkedSegment(object):
    """
    A mapping of a half-open interval to a specific value in a linked list.
    Lists of segments must not contain overlapping intervals.
    """
    start = attr.ib(default=None)
    end = attr.ib(default=None)
    value = attr.ib(default=None)
    next = attr.ib(default=None)

@attr.s
class Segment(object):
    """
    A mapping of a half-open interval to a specific value.
    """
    start = attr.ib(default=None)
    end = attr.ib(default=None)
    value = attr.ib(default=None)


def chain_str(head):
    """
    Returns the specified chain of segments as a string.
    """
    ret = ""
    u = head
    while u is not None:
        ret += "({}-{}:{})".format(u.start, u.end, u.value)
        u = u.next
        if u is not None:
            ret += "=>"
    return ret

class TreeSequenceBuilder(object):
    """
    Builds a tree sequence from the copying paths of ancestors and samples.
    This uses a simpler extensional list algorithm.
    """
    # TODO move this into test code.
    def __init__(self, num_samples, num_ancestors, num_sites):
        self.num_sites = num_sites
        self.num_samples = num_samples
        self.num_ancestors = num_ancestors
        # The list of children at every site.
        self.children = [
            [[] for _ in range(num_sites)] for _ in range(num_ancestors + 1)]
        self.mutations = [[] for _ in range(num_sites)]

    def print_state(self):
        print("Tree sequence builder state:")
        for l, children in enumerate(self.children):
            print(l, ":", children)
        print("mutations = :")
        for j, mutations in enumerate(self.mutations):
            print(j, ":", mutations)

    def add_path(self, child, P, A, mutations):
        # print("Add path:", child, P, A,mutations)
        for l in range(self.num_sites):
            if P[l] != -1:
                self.children[P[l]][l].append(child)
            else:
                self.children[0][l].append(child)
        for site in mutations:
            self.mutations[site].append((child, str(A[site])))

    def finalise(self):

        # Allocate the nodes.
        nodes = msprime.NodeTable(self.num_ancestors + self.num_samples + 1)
        nodes.add_row(time=self.num_ancestors + 1)
        for j in range(self.num_ancestors):
            nodes.add_row(time=self.num_ancestors - j)
        for j in range(self.num_samples):
            nodes.add_row(time=0, flags=msprime.NODE_IS_SAMPLE)

        # sort all the children lists
        for children_lists in self.children:
            for children in children_lists:
                children.sort()

        edgesets = msprime.EdgesetTable()
        for j in range(self.num_ancestors, -1, -1):
            row = self.children[j]
            last_c = row[0]
            left = 0
            for l in range(1, self.num_sites):
                if row[l] != last_c:
                    if len(last_c) > 0:
                        edgesets.add_row(
                            left=left, right=l, parent=j, children=tuple(last_c))
                    left = l
                    last_c = row[l]
            if len(last_c) > 0:
                edgesets.add_row(
                    left=left, right=self.num_sites, parent=j, children=tuple(last_c))

        sites = msprime.SiteTable()
        mutations = msprime.MutationTable()
        for j in range(self.num_sites):
            sites.add_row(j, "0")
            for node, derived_state in self.mutations[j]:
                mutations.add_row(j, node, derived_state)

        # self.print_state()
        # print(nodes)
        # print(edgesets)
        # print(sites)
        # print(mutations)
        # right = set(edgesets.right)
        # left = set(edgesets.left)
        # print("Diff:", right - left)

        ts = msprime.load_tables(
            nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)
        return ts


class AncestorStore(object):
    """
    Stores ancestors in a per site run length encoding. The store is always
    initialised with a single ancestor consisting of all zeros.
    """
    def __init__(self, num_sites):
        self.num_sites = num_sites
        self.num_ancestors = 1
        self.sites = [[Segment(0, 1, 0)] for j in range(self.num_sites)]

    def add(self, a):
        """
        Adds the specified ancestor to this store.
        """
        assert a.shape == (self.num_sites,)
        x = self.num_ancestors
        for j in range(self.num_sites):
            if a[j] != -1:
                tail = self.sites[j][-1]
                if tail.end == x and tail.value == a[j]:
                    tail.end += 1
                else:
                    self.sites[j].append(Segment(x, x + 1, a[j]))
        self.num_ancestors += 1

    def get_state(self, site, ancestor):
        """
        Returns the state of the specified ancestor at the specified site.
        """
        for seg in self.sites[site]:
            if seg.start <= ancestor < seg.end:
                break
        if seg.start <= ancestor < seg.end:
            return seg.value
        else:
            return -1

    def export(self):
        num_segments = sum(len(self.sites[j]) for j in range(self.num_sites))
        site = np.zeros(num_segments, dtype=np.int32)
        start = np.zeros(num_segments, dtype=np.int32)
        end = np.zeros(num_segments, dtype=np.int32)
        state = np.zeros(num_segments, dtype=np.int8)
        j = 0
        for l in range(self.num_sites):
            for seg in self.sites[l]:
                site[j] = l
                start[j] = seg.start
                end[j] = seg.end
                state[j] = seg.value
                j += 1
        return site, start, end, state


@attr.s
class Site(object):
    id = attr.ib(default=None)
    frequency = attr.ib(default=None)


class AncestorBuilder(object):
    """
    Builds inferred ancestors.
    """
    def __init__(self, S):
        self.haplotypes = S
        self.num_samples = S.shape[0]
        self.num_sites = S.shape[1]
        self.sites = [None for j in range(self.num_sites)]
        self.sorted_sites = [None for j in range(self.num_sites)]
        for j in range(self.num_sites):
            self.sites[j] = Site(j, np.sum(S[:, j]))
            self.sorted_sites[j] = Site(j, np.sum(S[:, j]))
        self.sorted_sites.sort(key=lambda x: (-x.frequency, x.id))
        self.frequency_classes = collections.defaultdict(list)
        for site in self.sorted_sites:
            if site.frequency > 1:
                self.frequency_classes[site.frequency].append(site)
        # for k, v in self.frequency_classes.items():
        #     print(k, "->", v)

    def __build_ancestor_sites(self, focal_site, sites, a):
        S = self.haplotypes
        samples = set()
        for j in range(self.num_samples):
            if S[j, focal_site.id] == 1:
                samples.add(j)
        for l in sites:
            a[l] = 0
            if self.sites[l].frequency > focal_site.frequency:
                # print("\texamining:", self.sites[l])
                # print("\tsamples = ", samples)
                num_ones = 0
                num_zeros = 0
                for j in samples:
                    if S[j, l] == 1:
                        num_ones += 1
                    else:
                        num_zeros += 1
                # TODO choose a branch uniformly if we have equality.
                if num_ones >= num_zeros:
                    a[l] = 1
                    samples = set(j for j in samples if S[j, l] == 1)
                else:
                    samples = set(j for j in samples if S[j, l] == 0)
            if len(samples) == 1:
                # print("BREAK")
                break

    def __build_ancestor(self, focal_site):
        # print("Building ancestor for ", focal_site)
        a = np.zeros(self.num_sites, dtype=np.int8) - 1
        a[focal_site.id] = 1
        sites = range(focal_site.id + 1, self.num_sites)
        self.__build_ancestor_sites(focal_site, sites, a)
        sites = range(focal_site.id - 1, -1, -1)
        self.__build_ancestor_sites(focal_site, sites, a)
        return a

    def build_ancestors(self):
        for site in self.sorted_sites:
            if site.frequency == 1:
                break
            yield self.__build_ancestor(site)


class Traceback(object):
    def __init__(self, store):
        self.store = store
        self.reset()

    def add_recombination(self, site, start, end, ancestor):
        if self.site_head[site] is None:
            self.site_head[site] = LinkedSegment(start, end, ancestor)
            self.site_tail[site] = self.site_head[site]
        else:
            if self.site_tail[site].end == start and self.site_tail[site].value == ancestor:
                self.site_tail[site].end = end
            else:
                tail = LinkedSegment(start, end, ancestor)
                self.site_tail[site].next = tail
                self.site_tail[site] = tail

    def reset(self):
        self.site_head = [None for _ in range(self.store.num_sites)]
        self.site_tail = [None for _ in range(self.store.num_sites)]

    def run(self, h, start_site, end_site, end_site_value, P, M):
        """
        Returns the array of haplotype indexes that the specified encoded traceback
        defines for the given startin point at the last site.
        """
        # print("Running traceback on ", start_site, end_site, end_site_value)
        # print(self.decode_traceback(T))
        l = end_site - 1
        P[:] = -1
        P[l] = end_site_value
        T = self.site_head
        num_mutations = 0
        while l > start_site:
            state = self.store.get_state(l, int(P[l]))
            if state == -1:
                break
            if state != h[l]:
                M[num_mutations] = l
                num_mutations += 1
            value = None
            u = T[l]
            while u is not None:
                if u.start <= P[l] < u.end:
                    value = u.value
                    break
                if u.start > P[l]:
                    break
                u = u.next
            if value is None:
                value = P[l]
            P[l - 1] = value
            l -= 1
        state = self.store.get_state(l, int(P[l]))
        assert state != -1
        if state != h[l]:
            M[num_mutations] = l
            num_mutations += 1
        return num_mutations


class AncestorMatcher(object):
    """
    """
    def __init__(self, store, recombination_rate):
        self.store = store
        self.recombination_rate = recombination_rate

    def best_path(
            self, num_ancestors, h, start_site, end_site, error_rate, traceback):
        """
        Returns the best path through the list of ancestors for the specified
        haplotype.
        """
        assert h.shape == (self.store.num_sites,)
        m = self.store.num_sites
        print("store = ", self.store)
        n = num_ancestors
        rho = self.recombination_rate
        L = [Segment(0, n, 1)]
        best_haplotype = 0
        # Initialise this to n to ensure that we do not calculate a recombination
        # as the most likely event in the first site.
        possible_recombinants = n

        print("BEST PATH", num_ancestors, h, start_site, end_site)

        for site in range(start_site, end_site):
            L_next = []
            S = [Segment(*s) for s in self.store.get_site(site)]
            # Compute the recombination rate.
            # TODO also need to get the position here so we can get the length of the
            # region.
            r = 1 - np.exp(-rho / possible_recombinants)
            pr = r / possible_recombinants
            qr = 1 - r + r / possible_recombinants
            # print()
            # print("site = ", site)
            # print("n = ", n)
            # print("re= ", possible_recombinants)
            # print("pr= ", pr)
            # print("qr= ", qr)
            # print("L = ", L)
            # print("S = ", S)
            # print("h = ", h[site])
            # print("b = ", best_haplotype)

            l = 0
            s = 0
            start = 0
            while start != n:
                end = n
                if l < len(L):
                    if L[l].start > start:
                        end = min(end, L[l].start)
                    else:
                        end = min(end, L[l].end)
                if s < len(S):
                    if S[s].start > start:
                        end = min(end, S[s].start)
                    else:
                        end = min(end, S[s].end)
                # print("\tLOOP HEAD: start = ", start, "end = ", end)
                # print("\ts = ", s)
                # print("\tl = ", l)
                assert start < end
                # The likelihood of this interval is always 0 if it does not intersect
                # with S
                if s < len(S) and not (S[s].start >= end or S[s].end <= start):
                    state = S[s].value
                    # If this interval does not intersect with L, the likelihood is 0
                    likelihood = 0
                    if l < len(L) and not (L[l].start >= end or L[l].end <= start):
                        likelihood = L[l].value

                    x = likelihood * qr
                    y = pr  # v for maximum is 1 by normalisation
                    # print("\t", start, end, x, y)
                    if x >= y:
                        z = x
                    else:
                        z = y
                        traceback.add_recombination(site, start, end, best_haplotype)
                    # Determine the likelihood for this segment.
                    if state == h[site]:
                        likelihood_next = z * (1 - error_rate)
                    else:
                        likelihood_next = z * error_rate

                    # Update the L_next array
                    if len(L_next) == 0:
                        L_next = [Segment(start, end, likelihood_next)]
                    else:
                        if L_next[-1].end == start and L_next[-1].value == likelihood_next:
                            L_next[-1].end = end
                        else:
                            L_next.append(Segment(start, end, likelihood_next))
                start = end
                if l < len(L) and L[l].end <= start:
                    l += 1
                if s < len(S) and S[s].end <= start:
                    s += 1

            L = L_next
            max_value = -1
            best_haplotype = -1
            for seg in L:
                assert seg.start < seg.end
                if seg.value >= max_value:
                    max_value = seg.value
                    best_haplotype = seg.end - 1
            assert max_value > 0
            # Renormalise L
            for seg in L:
                seg.value /= max_value
            # Compute the possible recombination destinations for the next iteration.
            s = 0
            possible_recombinants = 0
            while s < len(S) and S[s].start < n:
                possible_recombinants += min(n, S[s].end) - S[s].start
                s += 1

        return best_haplotype


    def print_state(self):
        print("Matcher state")
        print("num_ancestors = ", self.num_ancestors)
        print("num_sites = ", self.num_sites)


    def decode_traceback(self, E):
        """
        Decode the specified encoded traceback matrix into the standard integer
        matrix.
        """
        assert len(E) == self.num_sites
        T = np.zeros((self.num_ancestors, self.num_sites), dtype=int)
        for l in range(1, self.num_sites):
            T[:,l] = np.arange(self.num_ancestors)
            u = E[l]
            while u is not None:
                T[u.start:u.end, l] = u.value
                u = u.next
        return T
