
import collections

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


def infer(samples, recombination_rate, mutation_rate, matcher_algorithm="C"):
    num_samples, num_sites = samples.shape
    builder = AncestorBuilder(samples)
    if matcher_algorithm == "C":
        matcher = _tsinfer.AncestorMatcher(num_sites)
    else:
        matcher = AncestorMatcher(num_sites)
    num_ancestors = builder.num_ancestors
    # tree_sequence_builder = TreeSequenceBuilder(num_samples, num_ancestors, num_sites)
    tree_sequence_builder = TreeSequenceBuilder(num_samples, num_ancestors, num_sites)

    A = np.zeros(num_sites, dtype=np.int8)
    P = np.zeros(num_sites, dtype=np.int32)
    M = np.zeros(num_sites, dtype=np.uint32)
    # for j in range(builder.num_ancestors):
        # focal_site = builder.site_order[j]
        # builder.build(j, A)
    for j, A in enumerate(builder.build_all_ancestors()):
        num_mutations = matcher.best_path(A, P, M, recombination_rate, mutation_rate)
        # print(A)
        # print(P)
        # print("num_mutations = ", num_mutations, M[:num_mutations])
        assert num_mutations == 1
        # assert M[0] == focal_site
        matcher.add(A)
        tree_sequence_builder.add_path(j + 1, P, A, M[:num_mutations])
    # tree_sequence_builder.print_state()
    # print("HERE")
    # matcher.print_state()
    # builder.print_all_ancestors()
    # H = matcher.decode_ancestors()
    # print(H)

    for j in range(num_samples):
        num_mutations = matcher.best_path(samples[j], P, M, recombination_rate, mutation_rate)
        u = num_ancestors + j + 1
        tree_sequence_builder.add_path(u, P, samples[j], M[:num_mutations])
    # tree_sequence_builder.print_state()
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

# class TreeSequenceBuilder(object):
#     """
#     Builds a tree sequence from the copying paths of ancestors and samples.
#     """
#     def __init__(self, num_samples, num_ancestors, num_sites):
#         self.num_sites = num_sites
#         self.num_samples = num_samples
#         self.num_ancestors = num_ancestors
#         self.parent_mappings = {}

#     def print_state(self):
#         print("Tree sequence builder state:")
#         for j in sorted(list(self.parent_mappings.keys())):
#             print(j, ":", chain_str(self.parent_mappings[j]))

#     def __add_mapping(self, start, end, parent, child):
#         """
#         Adds a mapping for the specified parent-child relationship over the
#         specified interval.
#         """
#         print("\tadd mapping:", start, end, parent, child)
#         if parent not in self.parent_mappings:
#             self.parent_mappings[parent] = Segment(start, end, [child])
#         else:
#             u = self.parent_mappings[parent]
#             print("\tInserting into", chain_str(u))
#             # Skip any leading segments.
#             t = None
#             while u is not None and u.end <= start:
#                 t = u
#                 u = u.next
#             if u is None:
#                 # We just have a new segment at the end of the chain.
#                 t.next = Segment(start, end, [child])
#             else:

#                 # Trim of the leading edge of a segment overlapping start
#                 if u.start < start and u.end > start:
#                     print("TRIM")
#                     v = Segment(start, u.end, list(u.value), u.next)
#                     u.end = start
#                     u.next = v
#                     u = v
#                 # Consume all segments that are within (start, end)
#                 print("Processing: u = ", u.start, u.end, "start, end= ", start, end)
#                 while u is not None and u.start < end:
#                     print("\tENCLOSED", u.start, u.end, (start, end))
#                     if u.end > end:
#                         v = Segment(end, u.end, list(u.value), u.next)
#                         u.next = v
#                         u.end = end
#                     u.value = sorted(u.value + [child])
#                     u = u.next

#         print("\tDONE:", parent, "->", chain_str(self.parent_mappings[parent]))
#         # check the integrity
#         u = self.parent_mappings[parent]
#         while u.next is not None:
#             assert u.end <= u.next.start
#             u = u.next


#     def add_path(self, child, P):
#         print("Add path:", child, P)
#         # Quick check to ensure we're correct. TODO remove
#         Pp = np.zeros(self.num_sites, dtype=int) - 1
#         for left, right, parent in split_parent_array(P):
#             self.__add_mapping(left, right, parent, child)
#             Pp[left:right] = parent
#         assert np.all(Pp == P)

#     def finalise(self):
#         # Allocate the nodes.
#         nodes = msprime.NodeTable(self.num_ancestors + self.num_samples + 1)
#         nodes.add_row(time=self.num_ancestors + 1)
#         for j in range(self.num_ancestors):
#             nodes.add_row(time=self.num_ancestors - j)
#         for j in range(self.num_samples):
#             nodes.add_row(time=0, flags=msprime.NODE_IS_SAMPLE)

#         edgesets = msprime.EdgesetTable()
#         for j in sorted(list(self.parent_mappings.keys()), reverse=True):
#             u = self.parent_mappings[j]
#             while u is not None:
#                 edgesets.add_row(u.start, u.end, j, tuple(sorted(u.value)))
#                 u = u.next
#         ts = msprime.load_tables(nodes=nodes, edgesets=edgesets)
#         return ts

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




class AncestorBuilder(object):
    """
    Sequentially build ancestors for non-singleton sites in a supplied
    sample-haplotype matrix.
    """
    def __init__(self, sample_matrix):
        self.sample_matrix = sample_matrix
        self.num_samples = sample_matrix.shape[0]
        self.num_sites = sample_matrix.shape[1]
        # Compute the frequencies at each site.
        self.frequency = np.sum(self.sample_matrix, axis=0)
        self.num_ancestors = np.sum(self.frequency > 1)
        self.site_order = self.frequency.argsort(kind="mergesort")[::-1]
        self.site_mask = np.zeros(self.num_sites, dtype=np.int8)

    def print_state(self):
        print("Builder state")
        print("num_samples = ", self.num_samples)
        print("num_sites = ", self.num_sites)
        print("num_ancestors = ", self.num_ancestors)
        print("sample_matrix = ")
        print(self.sample_matrix)
        print("frequency = ")
        print(self.frequency)
        print("site order = ")
        print(self.site_order)

    def print_all_ancestors(self):
        import pandas as pd
        pd.set_option("display.max_rows",10001)
        pd.set_option("display.max_columns",10001)
        pd.set_option("display.width",10001)
        H = np.zeros((self.num_ancestors, self.num_sites), dtype=int)
        for j, A in enumerate(self.build_all_ancestors()):
            H[j, :] = A
        df = pd.DataFrame(H)
        frequencies = self.frequency[self.site_order][:self.num_ancestors]
        sites = self.site_order[:self.num_ancestors]
        df["F"] = frequencies
        df["s"] = sites
        print(df)

    def __build_ancestors(self, frequency_class):
        """
        Builds the ancestors for the specified frequency class.
        """
        sites = np.where(self.frequency == frequency_class)[0]
        site_mask = np.zeros(self.num_sites, dtype=np.int8)
        site_mask[np.where(self.frequency > frequency_class)] = 1
        S = self.sample_matrix
        B = np.zeros((sites.shape[0], self.num_sites), dtype=np.int8)
        # print(frequency_class, sites)
        for index, site in enumerate(sites):
            A = B[index, :]
            # Find all samples that have a 1 at this site
            R = S[S[:,site] == 1]
            # Mask out mutations that haven't happened yet.
            M = np.logical_and(R, site_mask).astype(int)
            A[:] = -1
            A[site] = 1
            l = site - 1
            consistent_samples = {k: {(1, 1)} for k in range(R.shape[0])}
            while l >= 0 and len(consistent_samples) > 0:
                # print("l = ", l, consistent_samples)
                # Get the consensus among the consistent samples for this locus.
                # Only mutations older than this site are considered.
                s = 0
                for k in consistent_samples.keys():
                    s += M[k, l]
                A[l] = int(s >= len(consistent_samples) / 2)
                # Now we have computed the ancestor, go through the samples and
                # update their four-gametes patterns with the ancestor. Any
                # samples inconsistent with the ancestor are dropped.
                dropped = []
                for k, patterns in consistent_samples.items():
                    patterns.add((A[l], S[k, l]))
                    if len(patterns) == 4:
                        dropped.append(k)
                for k in dropped:
                    del consistent_samples[k]
                l -= 1
            l = site + 1
            consistent_samples = {k: {(1, 1)} for k in range(R.shape[0])}
            while l < self.num_sites and len(consistent_samples) > 0:
                # print("l = ", l, consistent_samples)
                # Get the consensus among the consistent samples for this locus.
                s = 0
                for k in consistent_samples.keys():
                    s += M[k, l]
                # print("s = ", s)
                A[l] = int(s >= len(consistent_samples) / 2)
                # Now we have computed the ancestor, go through the samples and
                # update their four-gametes patterns with the ancestor. Any
                # samples inconsistent with the ancestor are dropped.
                dropped = []
                for k, patterns in consistent_samples.items():
                    patterns.add((A[l], S[k, l]))
                    if len(patterns) == 4:
                        dropped.append(k)
                for k in dropped:
                    del consistent_samples[k]
                l += 1
        return B

    def __sort_slice(self, A, start, end, column, sort_order, depth=0):
        """
        Find the first column in the rows start:end in A that is not sorted, and
        resort to establish sortedness. Then, for each subslice of this array,
        recursively call this sorting until the destination slice is empty.
        """
        # print("  " * depth, "SORT SLICE", start, end, column, sort_order)
        col = column
        while col < A.shape[1] and np.all(np.sort(A[start:end, col]) == A[start:end, col]):
            # print("Skipping", col, A[start:end, col])
            col += 1
        if col < A.shape[1]:
            # print("Mismatch col = ", col, A[start:end, col])
            # print("Before:")
            # print(A[start:end])
            order = A[start:end, col].argsort(kind="mergesort")
            if sort_order == 1:
                order = order[::-1]

            A[start:end,:] = A[start:end,:][order]
            # print("Sorted:")
            # print(A[start:end])
            if col < A.shape[1] - 1:
                # Partition A[start:end] into distinct values.
                values, indexes = np.unique(A[start:end, col], return_index=True)
                # print(indexes)
                # print(values)
                indexes.sort()
                assert indexes[0] == 0
                partitions = list(indexes) + [end - start]
                for j in range(len(partitions) - 1):
                    partition_start = start + partitions[j]
                    partition_end = start + partitions[j + 1]
                    assert partition_start >= start and partition_end <= end
                    if partition_end - partition_start > 1:
                        # print("PARTITION:", partitions[j], partitions[j + 1], col + 1)
                        self.__sort_slice(
                            A, partition_start, partition_end, col + 1, sort_order,
                            depth + 1)

    def __sort_ancestor_slice(self, A, start, end, sort_order, depth=0):
        # print("  " * depth, "SORT ANC SLICE", start, end, sort_order)
        candidates = collections.defaultdict(list)
        for col in range(A.shape[1]):
            if not np.all(np.sort(A[start:end, col]) == A[start:end, col]):
                v, c = np.unique(A[start:end, col], return_counts=True)
                c = sorted(c, reverse=True)
                # We're only interested in columns in which the minor value is greater
                # than 1
                if c[1] > 1:
                    # print("col:", col, v, c)
                    candidates[c[0]].append(col)
        if len(candidates) > 0:
            sites = candidates[max(candidates.keys())]
            # print("Candidatate = ", sites)
            col = sites[len(sites) // 2]
            # print("CHOOSE", col)

            order = A[start:end, col].argsort(kind="mergesort")
            if sort_order == 1:
                order = order[::-1]

            A[start:end,:] = A[start:end,:][order]
            # print("Sorted:")
            # print(A[start:end])
            if col < A.shape[1] - 1:
                # Partition A[start:end] into distinct values.
                values, indexes = np.unique(A[start:end, col], return_index=True)
                # print(indexes)
                # print(values)
                indexes.sort()
                assert indexes[0] == 0
                partitions = list(indexes) + [end - start]
                for j in range(len(partitions) - 1):
                    partition_start = start + partitions[j]
                    partition_end = start + partitions[j + 1]
                    assert partition_start >= start and partition_end <= end
                    if partition_end - partition_start > 1:
                        # print("PARTITION:", partitions[j], partitions[j + 1], col + 1)
                        self.__sort_ancestor_slice(
                            A, partition_start, partition_end, sort_order,
                            depth + 1)


    def build_all_ancestors(self):
        order = 0
        for frequency_class in np.unique(self.frequency)[::-1][:-1]:
            B = self.__build_ancestors(frequency_class)
            # print("FREQUENCY CLASS", frequency_class)
            # print("pre-sort")
            # print(B)
            # self.__sort_slice(B, 0, B.shape[0], 0, order)
            self.__sort_ancestor_slice(B, 0, B.shape[0], order)
            for A in B:
                yield A
            order = (order + 1) % 2
            # print("DONE")
            # print(B)

    def build_old(self, site_index, A):
        # TODO check that these are called sequentially. We currently have
        # state in the site_mask variable which requires that there are generated
        # in order.

        S = self.sample_matrix
        site = self.site_order[site_index]
        self.site_mask[site] = 1
        # Find all samples that have a 1 at this site
        R = S[S[:,site] == 1]
        # Mask out mutations that haven't happened yet.
        M = np.logical_and(R, self.site_mask).astype(int)
        A[:] = -1
        A[site] = 1
        l = site - 1
        consistent_samples = {k: {(1, 1)} for k in range(R.shape[0])}
        while l >= 0 and len(consistent_samples) > 0:
            # print("l = ", l, consistent_samples)
            # Get the consensus among the consistent samples for this locus.
            # Only mutations older than this site are considered.
            s = 0
            for k in consistent_samples.keys():
                s += M[k, l]
            A[l] = int(s >= len(consistent_samples) / 2)
            # Now we have computed the ancestor, go through the samples and
            # update their four-gametes patterns with the ancestor. Any
            # samples inconsistent with the ancestor are dropped.
            dropped = []
            for k, patterns in consistent_samples.items():
                patterns.add((A[l], S[k, l]))
                if len(patterns) == 4:
                    dropped.append(k)
            for k in dropped:
                del consistent_samples[k]
            l -= 1
        l = site + 1
        consistent_samples = {k: {(1, 1)} for k in range(R.shape[0])}
        while l < self.num_sites and len(consistent_samples) > 0:
            # print("l = ", l, consistent_samples)
            # Get the consensus among the consistent samples for this locus.
            s = 0
            for k in consistent_samples.keys():
                s += M[k, l]
            # print("s = ", s)
            A[l] = int(s >= len(consistent_samples) / 2)
            # Now we have computed the ancestor, go through the samples and
            # update their four-gametes patterns with the ancestor. Any
            # samples inconsistent with the ancestor are dropped.
            dropped = []
            for k, patterns in consistent_samples.items():
                patterns.add((A[l], S[k, l]))
                if len(patterns) == 4:
                    dropped.append(k)
            for k in dropped:
                del consistent_samples[k]
            l += 1

class AncestorMatcher(object):
    """
    Stores (possibly incomplete) haplotypes representing ancestors using a site-wise
    run-length encoding and allows them to be matched against input haplotypes. The
    ancestor store initially contains a single ancestor that is all zeros, representing
    the ancestor of everybody.
    """
    def __init__(self, num_sites):
        self.num_sites = num_sites
        self.num_ancestors = 1
        self.sites = [[Segment(0, self.num_ancestors, 0)] for _ in range(num_sites)]

    def add(self, h):
        """
        Adds the specified ancestor into the store by appending its allelic
        values to the run length encoding.
        """
        x = self.num_ancestors
        assert h.shape == (self.num_sites,)
        for j in range(self.num_sites):
            if h[j] != -1:
                tail = self.sites[j][-1]
                if tail.end == x and tail.value == h[j]:
                    tail.end += 1
                else:
                    self.sites[j].append(Segment(x, x + 1, h[j]))
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


    def run_traceback(self, T, h, start_site, end_site, end_site_value, P, M):
        """
        Returns the array of haplotype indexes that the specified encoded traceback
        defines for the given startin point at the last site.
        """
        # print("Running traceback on ", start_site, end_site, end_site_value)
        # print(self.decode_traceback(T))
        P[:] = -1
        P[end_site] = end_site_value
        num_mutations = 0
        for l in range(end_site, start_site, -1):
            state = self.get_state(l, P[l])
            if state == -1:
                print("state error at ", l)
            assert state != -1
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
        l = start_site
        state = self.get_state(l, P[l])
        if state == -1:
            print("state error at ", l)
        assert state != -1
        if state != h[l]:
            M[num_mutations] = l
            num_mutations += 1
        return num_mutations

    def best_path(self, h, P, M, rho, theta):
        """
        Returns the best path through the list of ancestors for the specified
        haplotype.
        """
        assert h.shape == (self.num_sites,)
        m = self.num_sites
        n = self.num_ancestors
        r = 1 - np.exp(-rho / n)
        pr = r / n
        qr = 1 - r + r / n
        # pm = mutation; qm no mutation
        pm = 0.5 * theta / (n + theta)
        qm = n / (n + theta) + 0.5 * theta / (n + theta)

        # Skip any leading unset values
        start_site = 0
        while h[start_site] == -1:
            start_site += 1

        L = [Segment(0, n, 1)]
        T_head = [None for l in range(m)]
        T_tail = [None for l in range(m)]
        best_haplotype = 0

        for site in range(start_site, m):
            if h[site] == -1:
                break
            end_site = site

            L_next = []
            S = self.sites[site]
            # print()
            # print("site = ", site)
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
                    # else:
                    #     print("LGAP")

                    x = likelihood * qr
                    y = pr  # v for maximum is 1 by normalisation
                    if x >= y:
                        z = x
                    else:
                        z = y
                        # Update the traceback to reflect a recombination
                        if T_head[site] is None:
                            T_head[site] = LinkedSegment(start, end, best_haplotype)
                            T_tail[site] = T_head[site]
                        else:
                            if T_tail[site].end == start and T_tail[site].value == best_haplotype:
                                T_tail[site].end = end
                            else:
                                tail = LinkedSegment(start, end, best_haplotype)
                                T_tail[site].next = tail
                                T_tail[site] = tail
                    # Determine the likelihood for this segment.
                    if state == h[site]:
                        likelihood_next = z * qm
                    else:
                        likelihood_next = z * pm

                    # Update the L_next array
                    if len(L_next) == 0:
                        L_next = [Segment(start, end, likelihood_next)]
                    else:
                        if L_next[-1].end == start and L_next[-1].value == likelihood_next:
                            L_next[-1].end = end
                        else:
                            L_next.append(Segment(start, end, likelihood_next))
                # else:
                #     print("SGAP")
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
            # Renormalise L
            for seg in L:
                seg.value /= max_value

        return self.run_traceback(T_head, h, start_site, end_site, best_haplotype, P, M)



    def best_path_old(self, h, P, M, rho, theta):
        """
        Returns the best path through the list of ancestors for the specified
        haplotype.
        """
        assert h.shape == (self.num_sites,)
        m = self.num_sites
        n = self.num_ancestors
        r = 1 - np.exp(-rho / n)
        pr = r / n
        qr = 1 - r + r / n
        # pm = mutation; qm no mutation
        pm = 0.5 * theta / (n + theta)
        qm = n / (n + theta) + 0.5 * theta / (n + theta)

        # Skip any leading unset values
        start_site = 0
        while h[start_site] == -1:
            start_site += 1

        V = [Segment(0, self.num_ancestors, 1)]
        T_head = [None for l in range(m)]
        T_tail = [None for l in range(m)]

        # print("V = ", V)
        for l in range(start_site, m):
            if h[l] == -1:
                break
            end_site = l

            max_value = -1
            best_haplotype = -1
            assert V[0].start == 0
            assert V[-1].end == self.num_ancestors
            for v in V:
                assert v.start < v.end
                if v.value >= max_value:
                    max_value = v.value
                    best_haplotype = v.end - 1
            # Renormalise V
            for v in V:
                v.value /= max_value
            V_next = []
            R = self.sites[l]

            print()
            print("l = ", l)
            print("R = ", R)
            print("V = ", V)
            print("h = ", h[l])
            print("b = ", best_haplotype)

            r_index = 0
            v_index = 0
            last_R_end = 0
            while r_index < len(R):
                print("R LOOP HEAD:", r_index, R[r_index])
                if R[r_index].start != last_R_end:
                    # print("R GAP!!", last_R_end, R[r_index].start)
                    # print("V = ", V)
                    # print("V_next = ", V_next)
                    start = last_R_end
                    end = R[r_index].start
                    value = 0
                    if len(V_next) == 0:
                        V_next = [Segment(start, end, value)]
                    else:
                        if V_next[-1].end == start and V_next[-1].value == value:
                            V_next[-1].end = end
                        else:
                            V_next.append(Segment(start, end, value))
                    # Consume any V values intersecting with this gap
                    while V[v_index].end <= end:
                        v_index += 1
                    V[v_index].start = R[r_index].start
                    last_R_end = R[r_index].start
                else:
                    # Consume all segments in V
                    assert V[v_index].start == R[r_index].start
                    while v_index < len(V) and V[v_index].start < R[r_index].end:
                        print("V LOOP HEAD:", v_index, V[v_index])
                        start = V[v_index].start
                        end = min(V[v_index].end, R[r_index].end)
                        value = V[v_index].value
                        state = R[r_index].value

                        x = value * qr
                        y = pr  # v for maximum is 1 by normalisation
                        # print("\tx = ", x, "y = ", y)
                        if x >= y:
                            z = x
                        else:
                            z = y
                            # Update the traceback to reflect a recombination
                            if T_head[l] is None:
                                T_head[l] = LinkedSegment(start, end, best_haplotype)
                                T_tail[l] = T_head[l]
                            else:
                                if T_tail[l].end == start and T_tail[l].value == best_haplotype:
                                    T_tail[l].end = end
                                else:
                                    tail = LinkedSegment(start, end, best_haplotype)
                                    T_tail[l].next = tail
                                    T_tail[l] = tail
                        # Determine the likelihood for this segment.
                        if state == -1:
                            value = 0
                        elif state == h[l]:
                            value = z * qm
                        else:
                            value = z * pm

                        if len(V_next) == 0:
                            V_next = [Segment(start, end, value)]
                        else:
                            if V_next[-1].end == start and V_next[-1].value == value:
                                V_next[-1].end = end
                            else:
                                V_next.append(Segment(start, end, value))
                        v_index += 1

                    if V[v_index - 1].end > R[r_index].end:
                        # Make sure we account for the overhang in the next iteration.
                        # print("OVERRUN PATCHUP")
                        v_index -= 1
                        V[v_index].start = R[r_index].end
                    last_R_end = R[r_index].end
                    r_index += 1

            if last_R_end != self.num_ancestors:
                # print("ADDING MISSING")
                value = 0
                start = end
                end = self.num_ancestors
                if len(V_next) == 0:
                    V_next = [Segment(start, end, value)]
                else:
                    if V_next[-1].end == start and V_next[-1].value == value:
                        V_next[-1].end = end
                    else:
                        V_next.append(Segment(start, end, value))
            # print("END of loop:", V_next)
            V = V_next
            assert V[0].start == 0
            for v_index in range(1, len(V)):
                assert V[v_index].start == V[v_index - 1].end
            assert V[-1].end == self.num_ancestors

        # print("finding best value for ", end_site)
        # print("V = ", chain_str(V_head))
        max_value = -1
        best_haplotype = -1
        for v in V:
            if v.value >= max_value:
                max_value = v.value
                best_haplotype = v.end - 1
        # print(self.decode_ancestors())
        return self.run_traceback(T_head, h, start_site, end_site, best_haplotype, P, M)

    def print_state(self):
        print("Matcher state")
        print("num_ancestors = ", self.num_ancestors)
        print("num_sites = ", self.num_sites)
        print("Sites:")
        for j, R in enumerate(self.sites):
            print(j, "\t:", R)

    def decode_ancestors(self):
        """
        Returns the full matrix of ancestor values.
        """
        H = np.zeros((self.num_ancestors, self.num_sites), dtype=int) - 1
        for j in range(self.num_sites):
            for u in self.sites[j]:
                H[u.start:u.end, j] = u.value
        return H

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
