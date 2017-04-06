
import threading
import collections
import tempfile
import subprocess

import numpy as np
import msprime

import seaborn as sns

import _tsinfer

class PythonThreader(object):
    """
    Simple Python implementation of the threading algorithm.
    """
    def __init__(self, reference_panel):
        self._reference_panel = reference_panel

    def run(self, haplotype_index, n, rho, err, P):
        """
        Thread the haplotype with the specified index through the
        reference panel of the n oldest haplotypes.
        """
        H = self._reference_panel.haplotypes
        X = self._reference_panel.positions
        h = H[haplotype_index]
        N, m = H.shape
        H = np.vstack((H, -1 * np.ones(m, dtype=np.int)))
        N += 1
        r = 1 - np.exp(-rho / n)
        recomb_proba = r / n
        no_recomb_proba = 1 - r + r / n

        V = np.ones(N)
        T = np.zeros((N, m), dtype=int)
        condition = np.logical_and(h == 1, np.sum(H[N - n - 1: N - 1] == 1, axis=0) == 0)
        mutated_loci = set(np.argwhere(condition).flatten())

        max_V_index = N - n - 1
        d = 1
        assert rho > 0 or err > 0
        for l in range(m):
            V /= V[max_V_index]
            Vn = np.zeros(N)
            max_Vn_index = N - n - 1
            for j in range(N - n - 1, N):
                x = V[j] * no_recomb_proba * d
                assert V[max_V_index] == 1.0
                y = V[max_V_index] * recomb_proba * d
                if x > y:
                    max_p = x
                    max_k = j
                else:
                    max_p = y
                    max_k = max_V_index
                T[j, l] = max_k
                if h[l] == -1:
                    if j == N - 1:
                        emission_p = 1
                    else:
                        emission_p = 0
                elif H[j, l] == -1:
                    emission_p = 0
                else:
                    emission_p = 1
                    if l in mutated_loci:
                        assert h[l] == 1 and H[j, l] == 0
                    else:
                        emission_p = err
                        if H[j, l] == h[l]:
                            emission_p = 1
                Vn[j] = max_p * emission_p
                if Vn[j] >= Vn[max_Vn_index]:
                    # If we have several haplotypes with equal liklihood, we choose the
                    # oldest.
                    max_Vn_index = j
            V = Vn
            max_V_index = max_Vn_index
            d = X[l + 1] - X[l]
        P[m - 1] = max_V_index
        for l in range(m - 1, 0, -1):
            P[l - 1] = T[P[l], l]
        mutations = []
        for l in range(m):
            if H[haplotype_index, l] != H[P[l], l]:
                mutations.append(l)
        return mutations


class ReferencePanel(object):
    """
    Class representing the reference panel for inferring a tree sequence
    from observed data.
    """
    def __init__(
            self, samples, positions, sequence_length, rho=None,
            sample_error=0, ancestor_error=0, algorithm=None, haplotypes=None):
        self._ll_reference_panel = _tsinfer.ReferencePanel(
                samples, positions, sequence_length)
        self._threader_class = _tsinfer.Threader
        self.rho = rho
        self.sample_error = sample_error
        self.ancestor_error = ancestor_error
        if algorithm is not None:
            algorithm_map = {
                "c": _tsinfer.Threader,
                "python": PythonThreader
            }
            self._threader_class = algorithm_map[algorithm]
        # Note: this isn't a long term interface. It's just here to allow for
        # experimentation with the set of haplotypes used by the Python
        # algorithm.
        if haplotypes is None:
            self.haplotypes = self._ll_reference_panel.get_haplotypes()
        else:
            self.haplotypes = haplotypes

    def _infer_paths_simple(self):
        N = self._ll_reference_panel.num_haplotypes
        n = self._ll_reference_panel.num_samples
        m = self._ll_reference_panel.num_sites
        H = self._ll_reference_panel.get_haplotypes()
        threader = self._threader_class(self)
        P = np.zeros((N, m), dtype=np.uint32)
        p = np.zeros(m, dtype=np.uint32)
        P[-1, :] = -1
        mutations = collections.defaultdict(list)
        for j in range(N - 2, n - 1, -1):
            mut_positions = threader.run(
                j, N - j - 1, self.rho, self.ancestor_error, p)
            for l in mut_positions:
                mutations[l].append(j)
            P[j] = p
        for j in range(n):
            mut_positions = threader.run(j, N - n, self.rho, self.sample_error, p)
            P[j] = p
            for l in mut_positions:
                mutations[l].append(j)
        return P, mutations

    def _infer_paths_threads(self, num_workers=None):
        N = self._ll_reference_panel.num_haplotypes
        n = self._ll_reference_panel.num_samples
        m = self._ll_reference_panel.num_sites
        P = np.zeros((N, m), dtype=np.uint32)
        P[-1, :] = -1

        work = []
        for j in range(N - 2, n - 1, -1):
            work.append((j, N - j - 1, self.ancestor_error))
        for j in range(n):
            work.append((j, N - n, self.sample_error))
        work_index = 0
        mutations = collections.defaultdict(list)
        lock = threading.Lock()

        def worker():
            nonlocal work_index
            nonlocal mutations
            threader = self._threader_class(self._ll_reference_panel)
            while True:
                with lock:
                    if work_index >= len(work):
                        break
                    haplotype_index, panel_size, error_p = work[work_index]
                    work_index += 1
                mutation_positions = threader.run(
                    haplotype_index, panel_size, self.rho, error_p, P[haplotype_index])
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

        return P, mutations

    @property
    def num_haplotypes(self):
        return self._ll_reference_panel.num_haplotypes

    @property
    def num_samples(self):
        return self._ll_reference_panel.num_samples

    @property
    def positions(self):
        return self._ll_reference_panel.get_positions()

    def infer_paths(self, num_workers=None):
        if num_workers == 1:
            return self._infer_paths_simple()
        else:
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()
            return self._infer_paths_threads(num_workers=num_workers)

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
                if P[j][l] != N:
                    C[P[j][l]][l].append(j)
                else:
                    # This is inelegant, but these links should all get filtered out
                    # by simplify.
                    C[N - 1][l].append(j)

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
        site_table = msprime.SiteTable(m, m)
        for j in range(m):
            site_table.add_row(X[j], ancestral_state="0")
        mutation_table = msprime.MutationTable(len(mutations), len(mutations))
        for site, rows in mutations.items():
            for row in rows:
                derived_state = str(H[row, site])
                mutation_table.add_row(site=site, node=row, derived_state=derived_state)
        return msprime.load_tables(
            nodes=node_table, edgesets=edgeset_table, sites=site_table,
            mutations=mutation_table)


class Illustrator(object):
    """
    Class to illustrate the copy process for a reference panel.
    """
    def __init__(self, reference_panel, P, mutations):
        self.reference_panel = reference_panel
        self.P = P
        self.mutations = mutations
        self.errors = []

    def run(self, focal, filename, H):
        # H = self.reference_panel.haplotypes
        N, M = H.shape
        n = self.reference_panel.num_samples
        P = self.P.astype(np.int32)
        mutation_nodes = list(self.mutations.values())
        out = tempfile.NamedTemporaryFile("w", prefix="ls_fig_")
        matrix_gap = 2
        haplotype_colours = ["0.95 * white", "0.95 * white"]
        palette = sns.color_palette("Dark2", N - n)
        copy_colours = {-1: "0.95 * white"}
        for j, (r, g, b) in enumerate(palette):
            copy_colours[n + j] = "rgb({}, {}, {})".format(r, g, b)
        print('size(20cm);', file=out)
        print('path cellbox = scale(0.95) * unitsquare;', file=out)
        print('path error_marker = scale(0.45) * polygon(6);', file=out)
        print('path mutation_marker = scale(0.45) * polygon(4);', file=out)
        print('path focal_marker = scale(0.25) * unitcircle;', file=out)
        print('pen cellpen = fontsize(5);', file=out)
        print('defaultpen(fontsize(8));', file=out)
        print('frame f;', file=out)
        print('label(f, "Samples", W);', file=out)
        y = 1
        print('label("Haplotypes", ({}, {}), N);'.format(M / 2, y), file=out)
        print('label("Copying matrix", ({}, {}), N);'.format(M + matrix_gap + M / 2, y), file=out)
        y = -n / 2
        x = -3
        print('add(rotate(90) * f, ({}, {}));'.format(x, y), file=out)
        print('frame f;', file=out)
        print('label(f, "Ancestors", W);', file=out)
        y = -n - (N - n) / 2
        print('add(rotate(90) * f, ({}, {}));'.format(x, y), file=out)
        ancestors = list(range(n)) + list(range(max(focal, n), N))
        for j in ancestors:
            x = -1.5
            y = -j
            print('label("{}", ({}, {}), cellpen);'.format(j, x, y), file=out)
            for k in range(M):
                x = k
                y = -j
                colour = haplotype_colours[H[j, k]]
                if P[focal][k] == j:
                    colour = copy_colours[j]
                elif j == focal:
                    colour = "0.5 * white"
                print('fill(shift(({}, {})) * cellbox, {});'.format(
                    x - 0.5, y - 0.5, colour), file=out)
                print('label("{}", ({}, {}), cellpen);'.format(H[j, k], x, y), file=out)
                if (j, k) in self.errors:
                    print('draw(shift(({}, {})) * error_marker, red);'.format(x, y), file=out)
                if j in mutation_nodes[k]:
                    colour = "blue"
                    if len(mutation_nodes[k]) > 1:
                        colour = "orange"
                    print('draw(shift(({}, {})) * mutation_marker, {});'.format(x, y, colour), file=out)

        for j in range(N - 1, focal - 1, -1):
            x = 2 * M + matrix_gap + 0.5
            y = -j
            print('label("{}", ({}, {}), cellpen);'.format(j, x, y), file=out)
            for k in range(M):
                x = k + M + matrix_gap
                y = -j
                if P[j, k] < N:
                    print('fill(shift(({}, {})) * cellbox, {});'.format(
                        x - 0.5, y - 0.5, copy_colours[P[j, k]]), file=out)
                    print('label("{}", ({}, {}), cellpen);'.format(P[j, k], x, y), file=out)


        print('filldraw(shift(({}, {})) * focal_marker, black);'.format(
            M + matrix_gap / 2 - 0.5, -focal), file=out)
        # Draw the frames around the matrices.
        y_middle = -n + 0.5
        y_bottom = -N + 0.5
        y_top = 0.5
        for x_min in [-0.5, M + matrix_gap - 0.5]:
            x_max = x_min + M
            print('draw(({}, {})--({}, {}));'.format(x_min, y_top, x_max, y_top), file=out)
            print('draw(({}, {})--({}, {}));'.format(x_min, y_middle, x_max, y_middle), file=out)
            print('draw(({}, {})--({}, {}));'.format(x_min, y_bottom, x_max, y_bottom), file=out)
            print('draw(({}, {})--({}, {}));'.format(x_min, y_bottom, x_min, y_top), file=out)
            print('draw(({}, {})--({}, {}));'.format(x_max, y_bottom, x_max, y_top), file=out)

        out.flush()
        subprocess.check_call(["asy", "-f", "pdf", out.name, "-o", filename])
        out.close()
