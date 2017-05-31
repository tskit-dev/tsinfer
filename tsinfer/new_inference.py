

import numpy as np
import attr


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


def segments_intersection(A, B):
    """
    Returns an iterator over the intersection of the specified linked lists of segments.
    For each (start, end) intersection that we find, return the pair of values in
    order A, B.
    """
    # print("INTERSECT")
    # print("\t", chain_str(A))
    # print("\t", chain_str(B))
    A_head = A
    B_head = B
    while A_head is not None and B_head is not None:
        A_s, A_e, A_v = A_head.start, A_head.end, A_head.value
        B_s, B_e, B_v = B_head.start, B_head.end, B_head.value
        # print("A_head = ", A_head)
        # print("B_head = ", B_head)
        if A_s <= B_s:
            if A_e > B_s:
                yield max(A_s, B_s), min(A_e, B_e), A_v, B_v
        else:
            if B_e > A_s:
                yield max(A_s, B_s), min(A_e, B_e), A_v, B_v

        if A_e <= B_e:
            A_head = A_head.next
        if B_e <= A_e:
            B_head = B_head.next


class AncestorBuilder(object):
    """
    Sequentially build ancestors for non-singleton sites in a supplied
    sample-haplotype matrix.
    """
    def __init__(self, num_samples, num_sites, sample_matrix):
        self.num_samples = num_samples
        self.num_sites = num_sites
        self.sample_matrix = sample_matrix
        assert self.sample_matrix.shape == (num_samples, num_sites)
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

    def build(self, site_index):
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
        A = -1 * np.ones(self.num_sites, dtype=int)
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
        return A

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
            elif h[j] != -1:
                seg = Segment(x, x + 1, h[j])
                tail.next = seg
                self.sites_tail[j] = seg
        self.num_ancestors += 1

    def run_traceback(self, T, start_site, end_site, end_site_value):
        """
        Returns the array of haplotype indexes that the specified encoded traceback
        defines for the given startin point at the last site.
        """
        print("Running traceback on ", start_site, end_site, end_site_value)
        print(self.decode_traceback(T))
        P = np.zeros(self.num_sites, dtype=int) - 1
        P[end_site] = end_site_value
        for l in range(end_site, start_site, -1):
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
        return P

    def best_path(self, h, rho, theta):
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
        V = None
        prev = None
        while u is not None:
            # TODO deal with -1
            v = Segment(u.start, u.end, pm if h[start_site] == u.value else pm)
            if V is None:
                V = v
            else:
                prev.next = v
            prev = v
            u = u.next
        T_head = [None for l in range(m)]
        T_tail = [None for l in range(m)]

        # print("V = ", chain_str(V))
        for l in range(start_site, m):
            if h[l] == -1:
                break
            end_site = l

            max_value = -1
            best_haplotype = -1
            v = V
            while v is not None:
                if v.value >= max_value:
                    max_value = v.value
                    best_haplotype = v.end - 1
                v = v.next
            # Renormalise V
            v = V
            while v is not None:
                v.value /= max_value
                v = v.next
            V_next_head = None
            V_next_tail = None
            # print("R = ", chain_str(self.sites_head[l]))
            # print("V = ", chain_str(V))
            for start, end, value, state in segments_intersection(V, self.sites_head[l]):
                # print("\t", start, end, v, state)
                x = value * qr
                y = pr  # v for maximum is 1 by normalisation
                if x >= y:
                    z = x
                else:
                    z = y
                    if T_head[l] is None:
                        T_head[l] = Segment(start, end, best_haplotype)
                        T_tail[l] = T_head[l]
                    else:
                        if T_tail[l].end == start:
                            T_tail[l].end = end
                        else:
                            tail = Segment(start, end, best_haplotype)
                            T_tail[l].next = tail
                            T_tail[l] = tail
                # TODO deal with -1
                if state == h[l]:
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
                V = V_next_head

        print("finding best value for ", end_site)
        print("V = ", chain_str(V))
        max_value = -1
        best_haplotype = -1
        v = V
        while v is not None:
            if v.value >= max_value:
                max_value = v.value
                best_haplotype = v.end - 1
            v = v.next

        P = self.run_traceback(T_head, start_site, end_site, best_haplotype)
        return P

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
            while u is not None:
                H[u.start:u.end, j] = u.value
                u = u.next
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
