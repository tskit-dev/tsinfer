

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


def infer(samples, matcher_algorithm="C"):
    num_samples, num_sites = samples.shape
    builder = AncestorBuilder(samples)
    if matcher_algorithm == "C":
        matcher = _tsinfer.AncestorMatcher(num_sites)
    else:
        matcher = AncestorMatcher(num_sites)
    num_ancestors = builder.num_ancestors
    # tree_sequence_builder = TreeSequenceBuilder(num_samples, num_ancestors, num_sites)
    tree_sequence_builder = TestTreeSequenceBuilder(num_samples, num_ancestors, num_sites)

    A = np.zeros(num_sites, dtype=np.int8)
    P = np.zeros(num_sites, dtype=np.int32)
    for j in range(builder.num_ancestors):
        focal_site = builder.site_order[j]
        builder.build(j, A)
        mutations = matcher.best_path(A, P, 0.01, 1e-200)
        assert len(mutations) == 1
        assert mutations[0][0] == focal_site
        matcher.add(A)
        tree_sequence_builder.add_path(j + 1, P, mutations)
    # tree_sequence_builder.print_state()

    for j in range(num_samples):
        mutations = matcher.best_path(samples[j], P, 0.01, 1e-200)
        u = num_ancestors + j + 1
        tree_sequence_builder.add_path(u, P, mutations)
    # tree_sequence_builder.print_state()
    ts = tree_sequence_builder.finalise()
    # fe = open("05-edgesets.txt", "w")
    # fn = open("05-nodes.txt", "w")
    # fs = open("05-sites.txt", "w")
    # fm = open("05-mutations.txt", "w")
    # ts.dump_text(nodes=fn, edgesets=fe, sites=fs, mutations=fm)

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
class Segment(object):
    """
    A mapping of a half-open interval to a specific value in a linked list.
    Lists of segments must not contain overlapping intervals.
    """
    start = attr.ib(default=None)
    end = attr.ib(default=None)
    value = attr.ib(default=None)
    next = attr.ib(default=None)


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
    """
    def __init__(self, num_samples, num_ancestors, num_sites):
        self.num_sites = num_sites
        self.num_samples = num_samples
        self.num_ancestors = num_ancestors
        self.parent_mappings = {}

    def print_state(self):
        print("Tree sequence builder state:")
        for j in sorted(list(self.parent_mappings.keys())):
            print(j, ":", chain_str(self.parent_mappings[j]))

    def __add_mapping(self, start, end, parent, child):
        """
        Adds a mapping for the specified parent-child relationship over the
        specified interval.
        """
        print("\tadd mapping:", start, end, parent, child)
        if parent not in self.parent_mappings:
            self.parent_mappings[parent] = Segment(start, end, [child])
        else:
            u = self.parent_mappings[parent]
            print("\tInserting into", chain_str(u))
            # Skip any leading segments.
            t = None
            while u is not None and u.end <= start:
                t = u
                u = u.next
            if u is None:
                # We just have a new segment at the end of the chain.
                t.next = Segment(start, end, [child])
            else:

                # Trim of the leading edge of a segment overlapping start
                if u.start < start and u.end > start:
                    print("TRIM")
                    v = Segment(start, u.end, list(u.value), u.next)
                    u.end = start
                    u.next = v
                    u = v
                # Consume all segments that are within (start, end)
                print("Processing: u = ", u.start, u.end, "start, end= ", start, end)
                while u is not None and u.start < end:
                    print("\tENCLOSED", u.start, u.end, (start, end))
                    if u.end > end:
                        v = Segment(end, u.end, list(u.value), u.next)
                        u.next = v
                        u.end = end
                    u.value = sorted(u.value + [child])
                    u = u.next

        print("\tDONE:", parent, "->", chain_str(self.parent_mappings[parent]))
        # check the integrity
        u = self.parent_mappings[parent]
        while u.next is not None:
            assert u.end <= u.next.start
            u = u.next


    def add_path(self, child, P):
        print("Add path:", child, P)
        # Quick check to ensure we're correct. TODO remove
        Pp = np.zeros(self.num_sites, dtype=int) - 1
        for left, right, parent in split_parent_array(P):
            self.__add_mapping(left, right, parent, child)
            Pp[left:right] = parent
        assert np.all(Pp == P)

    def finalise(self):
        # Allocate the nodes.
        nodes = msprime.NodeTable(self.num_ancestors + self.num_samples + 1)
        nodes.add_row(time=self.num_ancestors + 1)
        for j in range(self.num_ancestors):
            nodes.add_row(time=self.num_ancestors - j)
        for j in range(self.num_samples):
            nodes.add_row(time=0, flags=msprime.NODE_IS_SAMPLE)

        edgesets = msprime.EdgesetTable()
        for j in sorted(list(self.parent_mappings.keys()), reverse=True):
            u = self.parent_mappings[j]
            while u is not None:
                edgesets.add_row(u.start, u.end, j, tuple(sorted(u.value)))
                u = u.next
        ts = msprime.load_tables(nodes=nodes, edgesets=edgesets)
        return ts

class TestTreeSequenceBuilder(object):
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

    def add_path(self, child, P, mutations):
        # print("Add path:", child, P, mutations)
        for l in range(self.num_sites):
            if P[l] != -1:
                self.children[P[l]][l].append(child)
            else:
                self.children[0][l].append(child)
        for site, ancestral_state, derived_state in mutations:
            self.mutations[site].append((child, ancestral_state, derived_state))

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
            for node, ancestral_state, derived_state in self.mutations[j]:
                mutations.add_row(j, node, str(derived_state))

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

    def build(self, site_index, A):
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
        self.sites_head = [Segment(0, self.num_ancestors, 0) for _ in range(num_sites)]
        self.sites_tail = list(self.sites_head)

    def add(self, h):
        """
        Adds the specified ancestor into the store by appending its allelic
        values to the run length encoding.
        """
        x = self.num_ancestors
        assert h.shape == (self.num_sites,)
        for j in range(self.num_sites):
            tail = self.sites_tail[j]
            if tail.end == x and tail.value == h[j]:
                tail.end += 1
            else:
                seg = Segment(x, x + 1, h[j])
                tail.next = seg
                self.sites_tail[j] = seg
        self.num_ancestors += 1

    def get_state(self, site, ancestor):
        """
        Returns the state of the specified ancestor at the specified site.
        """
        seg = self.sites_head[site]
        while seg.end <= ancestor:
            seg = seg.next
        assert seg.start <= ancestor < seg.end
        return seg.value


    def run_traceback(self, T, h, start_site, end_site, end_site_value, P):
        """
        Returns the array of haplotype indexes that the specified encoded traceback
        defines for the given startin point at the last site.
        """
        # print("Running traceback on ", start_site, end_site, end_site_value)
        # print(self.decode_traceback(T))
        P[:] = -1
        P[end_site] = end_site_value
        mutations = []
        for l in range(end_site, start_site, -1):
            state = self.get_state(l, P[l])
            if state != h[l]:
                mutations.append((l, state, h[l]))
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
        if state != h[l]:
            mutations.append((l, state, h[l]))
        return mutations

    def best_path(self, h, P, rho, theta):
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
        u = self.sites_head[start_site]

        V_head = Segment(0, self.num_ancestors, 1)
        V_tail = V_head
        T_head = [None for l in range(m)]
        T_tail = [None for l in range(m)]

        # print("V = ", chain_str(V))
        for l in range(start_site, m):
            if h[l] == -1:
                break
            end_site = l

            max_value = -1
            best_haplotype = -1
            assert V_head.start == 0
            assert V_tail.end == self.num_ancestors
            v = V_head
            while v is not None:
                if v.value >= max_value:
                    max_value = v.value
                    best_haplotype = v.end - 1
                v = v.next
            # Renormalise V
            v = V_head
            while v is not None:
                v.value /= max_value
                v = v.next
            V_next_head = None
            V_next_tail = None

            # print("l = ", l)
            # print("R = ", chain_str(self.sites_head[l]))
            # print("V = ", chain_str(V_head))
            # print("h = ", h[l])
            # print("b = ", best_haplotype)

            R = self.sites_head[l]
            V = V_head
            while R is not None and V is not None:
                # print("\tLOOP HEAD")
                # print("\tR = ", chain_str(R))
                # print("\tV = ", chain_str(V))
                # print("\tV_next = ", chain_str(V_next_head))
                start = max(V.start, R.start)
                end = min(V.end, R.end)
                value = V.value
                state = R.value
                if R.end == V.end:
                    R = R.next
                    V = V.next
                elif R.end < V.end:
                    R = R.next
                elif V.end < R.end:
                    V = V.next
                # print("", start, end, value, state, sep="\t")

                x = value * qr
                y = pr  # v for maximum is 1 by normalisation
                # print("\tx = ", x, "y = ", y)
                if x >= y:
                    z = x
                else:
                    z = y
                    if T_head[l] is None:
                        T_head[l] = Segment(start, end, best_haplotype)
                        T_tail[l] = T_head[l]
                    else:
                        if T_tail[l].end == start and T_tail[l].value == best_haplotype:
                            T_tail[l].end = end
                        else:
                            tail = Segment(start, end, best_haplotype)
                            T_tail[l].next = tail
                            T_tail[l] = tail
                if state == -1:
                    value = 0
                elif state == h[l]:
                    value = z * qm
                else:
                    value = z * pm
                if V_next_head is None:
                    V_next_head = Segment(start, end, value)
                    V_next_tail = V_next_head
                else:
                    if V_next_tail.end == start and V_next_tail.value == value:
                        V_next_tail.end = end
                    else:
                        tail = Segment(start, end, value)
                        V_next_tail.next = tail
                        V_next_tail = tail
            # print("T = ", chain_str(T_head[l]))
            # print()
            V_head = V_next_head
            V_tail = V_next_tail
            # Make sure V is complete.
            v = V_head
            assert v.start == 0
            while v.next is not None:
                assert v.end == v.next.start
                v = v.next
            assert v.end == self.num_ancestors

        # print("finding best value for ", end_site)
        # print("V = ", chain_str(V_head))
        max_value = -1
        best_haplotype = -1
        v = V_head
        while v is not None:
            if v.value >= max_value:
                max_value = v.value
                best_haplotype = v.end - 1
            v = v.next

        return self.run_traceback(T_head, h, start_site, end_site, best_haplotype, P)

    def print_state(self):
        print("Matcher state")
        print("num_ancestors = ", self.num_ancestors)
        print("num_sites = ", self.num_sites)
        print("Sites:")
        for j, u in enumerate(self.sites_head):
            print(j, "\t:", chain_str(u))

    def decode_ancestors(self):
        """
        Returns the full matrix of ancestor values.
        """
        H = np.zeros((self.num_ancestors, self.num_sites), dtype=int) - 1
        for j in range(self.num_sites):
            u = self.sites_head[j]
            # Check for complete list.
            assert u.start == 0
            while u is not None:
                H[u.start:u.end, j] = u.value
                prev = u
                u = u.next
                if u is not None:
                    assert prev.end == u.start
            assert prev.end == self.num_ancestors
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





