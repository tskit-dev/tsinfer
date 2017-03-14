
import threading
import collections

import numpy as np
import msprime

import _tsinfer


class ReferencePanel(object):
    """
    Class representing the reference panel for inferring a tree sequence
    from observed data.
    """
    def __init__(self, samples, positions, sequence_length):
        self._ll_reference_panel = _tsinfer.ReferencePanel(
                samples, positions, sequence_length)

    def _infer_paths_simple(self, rho, err=0):
        N = self._ll_reference_panel.num_haplotypes
        n = self._ll_reference_panel.num_samples
        m = self._ll_reference_panel.num_sites
        threader = _tsinfer.Threader(self._ll_reference_panel)
        P = np.zeros((N, m), dtype=np.uint32)
        p = np.zeros(m, dtype=np.uint32)
        P[-1, :] = -1
        mutations = collections.defaultdict(list)
        for j in range(N - 2, n - 1, -1):
            # Error=0 when threading the ancestors because we assume that
            # these are correct.
            mut_positions = threader.run(j, N - j - 1, rho, 0, p)
            for l in mut_positions:
                mutations[l].append(j)
            P[j] = p
        for j in range(n):
            mut_positions = threader.run(j, N - n, rho, err, p)
            P[j] = p
            for l in mut_positions:
                mutations[l].append(j)
        mutations = [
            (position, tuple(sorted(nodes))) for (position, nodes) in mutations.items()]
        return P, mutations

    def _infer_paths_threads(self, rho, err=0, num_workers=None):
        N = self._ll_reference_panel.num_haplotypes
        n = self._ll_reference_panel.num_samples
        m = self._ll_reference_panel.num_sites
        P = np.zeros((N, m), dtype=np.uint32)
        P[-1, :] = -1

        work = []
        for j in range(N - 2, n - 1, -1):
            # For the ancestors we have an error_p of 0
            work.append((j, N - j - 1, 0))
        for j in range(n):
            work.append((j, N - n, err))
        work_index = 0
        mutations = collections.defaultdict(list)
        lock = threading.Lock()

        def worker():
            nonlocal work_index
            nonlocal mutations
            threader = _tsinfer.Threader(self._ll_reference_panel)
            while True:
                with lock:
                    if work_index >= len(work):
                        break
                    haplotype_index, panel_size, error_p = work[work_index]
                    work_index += 1
                mutation_positions = threader.run(
                    haplotype_index, panel_size, rho, error_p, P[haplotype_index])
                with lock:
                    for l in mutation_positions:
                        mutations[l].append(haplotype_index)

        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        mutations = [
            (position, tuple(sorted(nodes))) for (position, nodes) in mutations.items()]
        return P, mutations

    def infer_paths(self, rho, err=0, num_workers=None):
        if num_workers == 1:
            return self._infer_paths_simple(rho, err=err)
        else:
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()
            return self._infer_paths_threads(rho, err=err, num_workers=num_workers)

    def convert_records(self, P, mutations):
        N = self._ll_reference_panel.num_haplotypes
        n = self._ll_reference_panel.num_samples
        m = self._ll_reference_panel.num_sites
        H = self._ll_reference_panel.get_haplotypes()
        X = self._ll_reference_panel.get_positions()
        L = self._ll_reference_panel.sequence_length

        N, m = P.shape
        assert H.shape == P.shape
        C = [[[] for _ in range(m)] for _ in range(N)]

        for j in range(N - 1):
            for l in range(m):
                C[P[j][l]][l].append(j)

        node_table = msprime.NodeTable(N)
        edgeset_table = msprime.EdgesetTable(N, 2 * N)
        for j in range(n):
            node_table.add_row(flags=msprime.NODE_IS_SAMPLE)
        for u in range(n, N):
            node_table.add_row(time=u)
            row = C[u]
            last_c = row[0]
            left = 0
            for l in range(1, m):
                if row[l] != last_c:
                    if len(last_c) > 0:
                        edgeset_table.add_row(
                            left=left, right=X[l], parent=u, children=tuple(last_c))
                    left = X[l]
                    last_c = row[l]
            if len(last_c) > 0:
                edgeset_table.add_row(
                    left=left, right=L, parent=u, children=tuple(last_c))
        site_table = msprime.SiteTable(len(mutations), len(mutations))
        mutation_table = msprime.MutationTable(len(mutations), len(mutations))
        for j, (index, nodes) in enumerate(mutations):
            site_table.add_row(position=X[index], ancestral_state="0")
            for node in nodes:
                mutation_table.add_row(site=j, node=node, derived_state="1")
        return msprime.load_tables(
            nodes=node_table, edgesets=edgeset_table, sites=site_table,
            mutations=mutation_table)
