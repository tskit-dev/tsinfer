# TODO copyright and license.
"""
TODO module docs.
"""

import contextlib
import collections
import queue
import threading
import _thread
import traceback

import numpy as np
import tqdm
import humanize
import daiquiri
import msprime

import _tsinfer


class ResultBuffer(object):
    """
    A wrapper for numpy arrays representing the results of a copying operations.
    """
    def __init__(self, chunk_size=1024):
        if chunk_size < 1:
            raise ValueError("chunk size must be > 0")
        self.chunk_size = chunk_size
        # edges
        self.__left = np.empty(chunk_size, dtype=np.uint32)
        self.__right = np.empty(chunk_size, dtype=np.uint32)
        self.__parent = np.empty(chunk_size, dtype=np.int32)
        self.__child = np.empty(chunk_size, dtype=np.int32)
        self.num_edges = 0
        self.max_edges = chunk_size
        # mutations
        self.__site = np.empty(chunk_size, dtype=np.uint32)
        self.__node = np.empty(chunk_size, dtype=np.int32)
        self.__derived_state = np.empty(chunk_size, dtype=np.int8)
        self.num_mutations = 0
        self.max_mutations = chunk_size

    @property
    def left(self):
        return self.__left[:self.num_edges]

    @property
    def right(self):
        return self.__right[:self.num_edges]

    @property
    def parent(self):
        return self.__parent[:self.num_edges]

    @property
    def child(self):
        return self.__child[:self.num_edges]

    @property
    def site(self):
        return self.__site[:self.num_mutations]

    @property
    def node(self):
        return self.__node[:self.num_mutations]

    @property
    def derived_state(self):
        return self.__derived_state[:self.num_mutations]

    def clear(self):
        """
        Clears this result buffer.
        """
        self.num_edges = 0
        self.num_mutations = 0

    def print_state(self):
        print("Edges = ")
        print("\tnum_edges = {} max_edges = {}".format(self.num_edges, self.max_edges))
        print("\tleft\tright\tparent\tchild")
        for j in range(self.num_edges):
            print("\t{}\t{}\t{}\t{}".format(
                self.__left[j], self.__right[j], self.__parent[j], self.__child[j]))
        print("Mutations = ")
        print("\tnum_mutations = {} max_mutations = {}".format(
            self.num_mutations, self.max_mutations))
        print("\tleft\tright\tparent\tchild")
        for j in range(self.num_mutations):
            print("\t{}\t{}".format(self.__site[j], self.__node[j]))

    def check_edges_size(self, additional):
        """
        Ensures that there is enough space for the specified number of additional
        edges.
        """
        if self.num_edges + additional > self.max_edges:
            new_size = self.max_edges + max(additional, self.chunk_size)
            self.__left.resize(new_size)
            self.__right.resize(new_size)
            self.__parent.resize(new_size)
            self.__child.resize(new_size)
            self.max_edges = new_size

    def add_edges(self, left, right, parent, child):
        """
        Adds the specified edges from the specified values. Left, right and parent
        must be numpy arrays of the same size. Child may be either a numpy array of
        the same size, or a single value.
        """
        size = left.shape[0]
        assert right.shape == (size,)
        assert parent.shape == (size,)
        self.check_edges_size(size)
        self.__left[self.num_edges: self.num_edges + size] = left
        self.__right[self.num_edges: self.num_edges + size] = right
        self.__parent[self.num_edges: self.num_edges + size] = parent
        self.__child[self.num_edges: self.num_edges + size] = child
        self.num_edges += size

    def check_mutations_size(self, additional):
        """
        Ensures that there is enough space for the specified number of additional
        mutations.
        """
        if self.num_mutations + additional > self.max_mutations:
            new_size = self.max_mutations + max(additional, self.chunk_size)
            self.__site.resize(new_size)
            self.__node.resize(new_size)
            self.__derived_state.resize(new_size)
            self.max_mutations = new_size

    def add_mutations(self, site, node, derived_state=None):
        """
        Adds the specified mutations from the specified values. Site must be a
        numpy array. Node may be either a numpy array of the same size, or a
        single value.
        """
        size = site.shape[0]
        self.check_mutations_size(size)
        self.__site[self.num_mutations: self.num_mutations + size] = site
        self.__node[self.num_mutations: self.num_mutations + size] = node
        fill = derived_state
        if derived_state is None:
            fill = 1
        self.__derived_state[self.num_mutations: self.num_mutations + size] = fill
        self.num_mutations += size

    def add_back_mutation(self, site, node):
        """
        Adds a single back mutation for the specified site.
        """
        self.check_mutations_size(1)
        self.__site[self.num_mutations] = site
        self.__node[self.num_mutations] = node
        self.__derived_state[self.num_mutations] = 0
        self.num_mutations += 1

    @classmethod
    def combine(cls, result_buffers):
        """
        Combines the specfied list of result buffers into a single new buffer.
        """
        # There is an inefficiency here where we are allocating too much
        # space for mutations. Should add a second parameter for mutations size.
        size = sum(result.num_edges for result in result_buffers)
        combined = cls(size)
        for result in result_buffers:
            combined.add_edges(
                result.left, result.right, result.parent, result.child)
            combined.add_mutations(result.site, result.node, result.derived_state)
        return combined


class InferenceManager(object):

    def __init__(
            self, samples, positions, sequence_length, recombination_rate,
            num_threads=1, method="C", progress=False, log_level="WARNING",
            resolve_shared_recombinations=False, resolve_polytomies=False):
        self.samples = samples
        self.num_samples = samples.shape[0]
        self.num_sites = samples.shape[1]
        assert self.num_sites == positions.shape[0]
        self.positions = positions
        self.resolve_shared_recombinations = resolve_shared_recombinations
        self.resolve_polytomies = resolve_polytomies
        self.sequence_length = sequence_length
        self.recombination_rate = recombination_rate
        self.num_threads = num_threads
        # Set up logging.
        daiquiri.setup(level=log_level)
        self.logger = daiquiri.getLogger()
        self.progress = progress
        self.progress_monitor_lock = None

        self.ancestor_builder_class = AncestorBuilder
        self.tree_sequence_builder_class = TreeSequenceBuilder
        self.ancestor_matcher_class = AncestorMatcher
        if method == "C":
            self.ancestor_builder_class = _tsinfer.AncestorBuilder
            self.tree_sequence_builder_class = _tsinfer.TreeSequenceBuilder
            self.ancestor_matcher_class = _tsinfer.AncestorMatcher
        self.ancestor_builder = None
        self.tree_sequence_builder = None
        self.sample_ids = None

    @contextlib.contextmanager
    def catch_thread_error(self):
        """
        Makes sure that errors that occur in the target of threads results in an
        error being raised in the main thread. Threads **must** be run with daemon
        status or a deadlock will occur!
        """
        try:
            yield
        except Exception:
            self.logger.critical("Exception occured in thread; exiting")
            self.logger.critical(traceback.format_exc())
            _thread.interrupt_main()

    def initialise(self):
        # This is slow, so we should figure out a way to report progress on it.
        self.logger.info(
            "Initialising ancestor builder for {} samples and {} sites".format(
                self.num_samples, self.num_sites))
        self.ancestor_builder = self.ancestor_builder_class(self.samples, self.positions)
        self.num_ancestors = self.ancestor_builder.num_ancestors
        self.tree_sequence_builder = self.tree_sequence_builder_class(
            self.sequence_length, self.positions, 10**6, 10**7,
            resolve_shared_recombinations=self.resolve_shared_recombinations,
            resolve_polytomies=self.resolve_polytomies)

        frequency_classes = self.ancestor_builder.get_frequency_classes()
        # TODO change the builder API here to just return the list of focal sites.
        # We really don't care about the actual frequency here.
        self.num_epochs = len(frequency_classes)
        self.epoch_ancestors = [None for _ in range(self.num_epochs)]
        self.epoch_time = [0 for _ in range(self.num_epochs)]
        for j, fc in enumerate(frequency_classes):
            self.epoch_time[j] = self.num_epochs - j
            self.epoch_ancestors[j] = fc[1]
        root_time = self.num_epochs + 1
        self.tree_sequence_builder.update(1, root_time, [], [], [], [], [], [], [])

        if self.progress:
            total = self.num_samples + self.num_ancestors - 1
            self.progress_monitor = tqdm.tqdm(total=total)
            self.progress_monitor_lock = threading.Lock()

    def __update_progress(self):
        if self.progress:
            with self.progress_monitor_lock:
                self.progress_monitor.update()

    def __find_path_ancestor(self, ancestor, node_id, focal_sites, matcher, results):
        # TODO make this an array natively.
        focal_sites = np.array(focal_sites)
        results.add_mutations(focal_sites, node_id)
        # TODO change the ancestor builder so that we don't need to do this.
        assert np.all(ancestor[focal_sites] == 1)
        ancestor[focal_sites] = 0
        (left, right, parent), matched_haplotype = matcher.find_path(ancestor)
        assert np.all(matched_haplotype == ancestor)
        results.add_edges(left, right, parent, node_id)
        self.__update_progress()

    def process_ancestors(self):
        self.logger.info("Copying {} ancestors".format(self.num_ancestors))
        if self.num_threads == 1:
            self.__process_ancestors_single_threaded()
        else:
            self.__process_ancestors_multi_threaded()

    def __process_ancestors_multi_threaded(self):
        build_queue = queue.Queue()
        match_queue = queue.Queue()
        matchers = [
            self.ancestor_matcher_class(
                self.tree_sequence_builder, self.recombination_rate)
            for _ in range(self.num_threads)]
        results = [ResultBuffer() for _ in range(self.num_threads)]
        mean_traceback_size = np.zeros(self.num_threads)
        num_matches = np.zeros(self.num_threads)

        def build_worker():
            with self.catch_thread_error():
                self.logger.info("Build worker thread starting")
                a = np.zeros(self.num_sites, dtype=np.int8)
                while True:
                    work = build_queue.get()
                    if work is None:
                        break
                    node_id, focal_sites = work
                    self.ancestor_builder.make_ancestor(focal_sites, a)
                    match_queue.put((node_id, focal_sites, a.copy()))
                    build_queue.task_done()
                build_queue.task_done()
                self.logger.info("Ancestor build worker thread exiting")

        def match_worker(thread_index):
            with self.catch_thread_error():
                self.logger.info(
                    "Ancestor match thread {} starting".format(thread_index))
                matcher = matchers[thread_index]
                result_buffer = results[thread_index]
                while True:
                    work = match_queue.get()
                    if work is None:
                        break
                    node_id, focal_sites, a = work
                    self.__find_path_ancestor(
                            a, node_id, focal_sites, matcher, result_buffer)
                    mean_traceback_size[thread_index] += matcher.mean_traceback_size
                    num_matches[thread_index] += 1
                    match_queue.task_done()
                match_queue.task_done()
                self.logger.info("Ancestor match thread {} exiting".format(thread_index))

        build_thread = threading.Thread(target=build_worker, daemon=True)
        build_thread.start()
        match_threads = [
            threading.Thread(target=match_worker, args=(j,), daemon=True)
            for j in range(self.num_threads)]
        for j in range(self.num_threads):
            match_threads[j].start()

        for epoch in range(self.num_epochs):
            time = self.epoch_time[epoch]
            ancestor_focal_sites = self.epoch_ancestors[epoch]
            self.logger.debug("Epoch {}; time = {}; {} ancestors to process".format(
                epoch, time, len(ancestor_focal_sites)))
            child = self.tree_sequence_builder.num_nodes
            for focal_sites in ancestor_focal_sites:
                build_queue.put((child, focal_sites))
                child += 1
            # Wait until the build_queue and match queue are both empty.
            build_queue.join()
            match_queue.join()
            epoch_results = ResultBuffer.combine(results)
            self.tree_sequence_builder.update(
                len(ancestor_focal_sites), time,
                epoch_results.left, epoch_results.right, epoch_results.parent,
                epoch_results.child, epoch_results.site, epoch_results.node)
            mean_memory = np.mean([matcher.total_memory for matcher in matchers])
            self.logger.debug(
                "Finished epoch {}; mean_tb_size={:.2f} edges={}; "
                "mean_matcher_mem={}".format(
                    epoch, np.sum(mean_traceback_size) / np.sum(num_matches),
                    self.tree_sequence_builder.num_edges,
                    humanize.naturalsize(mean_memory, binary=True)))
            mean_traceback_size[:] = 0
            num_matches[:] = 0
            for j in range(self.num_threads):
                results[j].clear()

        # Signal to the workers to quit and clean up.
        build_queue.put(None)
        for j in range(self.num_threads):
            match_queue.put(None)
        for j in range(self.num_threads):
            match_threads[j].join()

    def __process_ancestors_single_threaded(self):
        a = np.zeros(self.num_sites, dtype=np.int8)
        matcher = self.ancestor_matcher_class(
            self.tree_sequence_builder, self.recombination_rate)
        results = ResultBuffer()

        for epoch in range(self.num_epochs):
            time = self.epoch_time[epoch]
            ancestor_focal_sites = self.epoch_ancestors[epoch]
            self.logger.info("Epoch {}; time = {}; {} ancestors to process".format(
                epoch, time, len(ancestor_focal_sites)))
            child = self.tree_sequence_builder.num_nodes
            for focal_sites in ancestor_focal_sites:
                self.ancestor_builder.make_ancestor(focal_sites, a)
                self.__find_path_ancestor(a, child, focal_sites, matcher, results)
                child += 1
            self.tree_sequence_builder.update(
                len(ancestor_focal_sites), time,
                results.left, results.right, results.parent, results.child,
                results.site, results.node, results.derived_state)
            results.clear()

    def __process_sample(self, sample_index, matcher, results):
        child = self.sample_ids[sample_index]
        haplotype = self.samples[sample_index]
        (left, right, parent), matched_haplotype = matcher.find_path(haplotype)
        diffs = np.where(haplotype != matched_haplotype)[0]
        derived_state = haplotype[diffs]
        results.add_mutations(diffs, child, derived_state)
        results.add_edges(left, right, parent, child)
        self.__update_progress()

    def __process_samples_threads(self):
        results = [ResultBuffer() for _ in range(self.num_threads)]
        matchers = [
            self.ancestor_matcher_class(
                self.tree_sequence_builder, self.recombination_rate, self.error_rate)
            for _ in range(self.num_threads)]
        work_queue = queue.Queue()

        def worker(thread_index):
            with self.catch_thread_error():
                self.logger.info("Started sample worker thread {}".format(thread_index))
                mean_traceback_size = 0
                num_matches = 0
                while True:
                    sample_index = work_queue.get()
                    if sample_index is None:
                        break
                    self.__process_sample(
                        sample_index, matchers[thread_index], results[thread_index])
                    mean_traceback_size += matchers[thread_index].mean_traceback_size
                    num_matches += 1
                    work_queue.task_done()
                if num_matches > 0:
                    mean_traceback_size /= num_matches
                self.logger.info(
                    "Thread {} done: mean_tb_size={:.2f}; total_edges={}".format(
                        thread_index, mean_traceback_size,
                        results[thread_index].num_edges))
                work_queue.task_done()

        threads = [
            threading.Thread(target=worker, args=(j,), daemon=True)
            for j in range(self.num_threads)]
        for t in threads:
            t.start()
        for sample_index in range(self.num_samples):
            work_queue.put(sample_index)
        for _ in range(self.num_threads):
            work_queue.put(None)
        for t in threads:
            t.join()
        return ResultBuffer.combine(results)

    def process_samples(self, sample_error=0.0):
        self.logger.info("Processing {} samples".format(self.num_samples))
        self.sample_ids = self.tree_sequence_builder.num_nodes + np.arange(
                self.num_samples, dtype=np.int32)
        if self.num_threads == 1:
            results = ResultBuffer(self.num_samples)
            matcher = self.ancestor_matcher_class(
                self.tree_sequence_builder, self.recombination_rate, sample_error)
            for j in range(self.num_samples):
                self.__process_sample(j, matcher, results)
        else:
            results = self.__process_samples_threads()
        self.tree_sequence_builder.update(
            self.num_samples, 0,
            results.left, results.right, results.parent, results.child,
            results.site, results.node, results.derived_state)

    def finalise(self):
        self.logger.info("Finalising tree sequence")
        ts = self.get_tree_sequence()
        return ts.simplify(
            samples=self.sample_ids, filter_zero_mutation_sites=False)

    def get_tree_sequence(self):
        """
        Returns the current state of the build tree sequence. All samples and
        ancestors will have the sample node flag set.
        """
        tsb = self.tree_sequence_builder
        nodes = msprime.NodeTable()
        flags = np.zeros(tsb.num_nodes, dtype=np.uint32)
        time = np.zeros(tsb.num_nodes, dtype=np.float64)
        tsb.dump_nodes(flags=flags, time=time)
        nodes.set_columns(flags=flags, time=time)

        edges = msprime.EdgeTable()
        left = np.zeros(tsb.num_edges, dtype=np.float64)
        right = np.zeros(tsb.num_edges, dtype=np.float64)
        parent = np.zeros(tsb.num_edges, dtype=np.int32)
        child = np.zeros(tsb.num_edges, dtype=np.int32)
        tsb.dump_edges(left=left, right=right, parent=parent, child=child)
        edges.set_columns(left=left, right=right, parent=parent, child=child)

        sites = msprime.SiteTable()
        sites.set_columns(
            position=self.positions,
            ancestral_state=np.zeros(tsb.num_sites, dtype=np.int8) + ord('0'),
            ancestral_state_length=np.ones(tsb.num_sites, dtype=np.uint32))
        mutations = msprime.MutationTable()
        site = np.zeros(tsb.num_mutations, dtype=np.int32)
        node = np.zeros(tsb.num_mutations, dtype=np.int32)
        parent = np.zeros(tsb.num_mutations, dtype=np.int32)
        derived_state = np.zeros(tsb.num_mutations, dtype=np.int8)
        # TODO add parent to dump_mutations
        tsb.dump_mutations(
            site=site, node=node, derived_state=derived_state, parent=parent)
        derived_state += ord('0')
        mutations.set_columns(
            site=site, node=node, derived_state=derived_state,
            derived_state_length=np.ones(tsb.num_mutations, dtype=np.uint32),
            parent=parent)
        msprime.sort_tables(nodes, edges, sites=sites, mutations=mutations)
        return msprime.load_tables(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations)

    def ancestors(self):
        """
        Convenience method to get an array of the ancestors.
        """
        builder = self.ancestor_builder_class(self.samples, self.positions)
        frequency_classes = builder.get_frequency_classes()
        A = np.zeros((builder.num_ancestors, self.num_sites), dtype=np.int8)
        ancestor_id = 1
        for age, ancestor_focal_sites in frequency_classes:
            for focal_sites in ancestor_focal_sites:
                builder.make_ancestor(focal_sites, A[ancestor_id, :])
                ancestor_id += 1
        return A


def infer(
        samples, positions, sequence_length, recombination_rate, sample_error=0,
        method="C", num_threads=1, progress=False, log_level="WARNING",
        resolve_shared_recombinations=False, resolve_polytomies=False):
    # Primary entry point.
    manager = InferenceManager(
        samples, np.array(positions), sequence_length, recombination_rate,
        num_threads=num_threads, method=method, progress=progress, log_level=log_level,
        resolve_shared_recombinations=resolve_shared_recombinations,
        resolve_polytomies=resolve_polytomies)
    manager.initialise()
    manager.process_ancestors()
    manager.process_samples(sample_error)
    ts = manager.finalise()
    return ts


###############################################################
#
# Python algorithm implementation.
#
# This isn't meant to be used for any real inference, as it is
# *many* times slower than the real C implementation. However,
# it is a useful development and debugging tool, and so any
# updates made to the low-level C engine should be made here
# first.
#
###############################################################


class Edge(object):

    def __init__(self, left=None, right=None, parent=None, child=None):
        self.left = left
        self.right = right
        self.parent = parent
        self.child = child


class Site(object):
    def __init__(self, id, frequency):
        self.id = id
        self.frequency = frequency


class AncestorBuilder(object):
    """
    Builds inferred ancestors.
    """
    def __init__(self, S, positions):
        self.haplotypes = S
        self.num_samples = S.shape[0]
        self.num_sites = S.shape[1]
        self.sites = [None for j in range(self.num_sites)]
        self.sorted_sites = [None for j in range(self.num_sites)]
        for j in range(self.num_sites):
            self.sites[j] = Site(j, np.sum(S[:, j]))
            self.sorted_sites[j] = Site(j, np.sum(S[:, j]))
        self.sorted_sites.sort(key=lambda x: (-x.frequency, x.id))
        frequency_sites = collections.defaultdict(list)
        for site in self.sorted_sites:
            if site.frequency > 1:
                frequency_sites[site.frequency].append(site)
        # Group together identical sites within a frequency class
        self.frequency_classes = {}
        self.num_ancestors = 1
        for frequency, sites in frequency_sites.items():
            patterns = collections.defaultdict(list)
            for site in sites:
                state = tuple(self.haplotypes[:, site.id])
                patterns[state].append(site.id)
            self.frequency_classes[frequency] = list(patterns.values())
            self.num_ancestors += len(self.frequency_classes[frequency])

    def get_frequency_classes(self):
        ret = []
        for frequency in reversed(sorted(self.frequency_classes.keys())):
            ret.append((frequency, self.frequency_classes[frequency]))
        return ret

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

    def make_ancestor(self, focal_sites, a):
        # a[:] = -1
        # Setting to 0 for now to see if we can take advantage of other RLE.:w

        a[:] = 0
        focal_site = self.sites[focal_sites[0]]
        sites = range(focal_sites[-1] + 1, self.num_sites)
        self.__build_ancestor_sites(focal_site, sites, a)
        focal_site = self.sites[focal_sites[-1]]
        sites = range(focal_sites[0] - 1, -1, -1)
        self.__build_ancestor_sites(focal_site, sites, a)
        for j in range(focal_sites[0], focal_sites[-1] + 1):
            if j in focal_sites:
                a[j] = 1
            else:
                self.__build_ancestor_sites(focal_site, [j], a)
        return a


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
            self, sequence_length, positions, max_nodes, max_edges,
            resolve_shared_recombinations=True, resolve_polytomies=True):
        self.num_nodes = 0
        self.sequence_length = sequence_length
        self.positions = positions
        self.num_sites = positions.shape[0]
        self.time = []
        self.flags = []
        self.mutations = collections.defaultdict(list)
        self.edges = []
        self.mean_traceback_size = 0
        self.resolve_shared_recombinations = resolve_shared_recombinations
        self.resolve_polytomies = resolve_polytomies

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
        flags = np.zeros(self.num_nodes, dtype=np.uint32)
        time = np.zeros(self.num_nodes, dtype=np.float64)
        self.dump_nodes(flags=flags, time=time)
        nodes.set_columns(flags=flags, time=time)
        print("nodes = ")
        print(nodes)

        edges = msprime.EdgeTable()
        left = np.zeros(self.num_edges, dtype=np.float64)
        right = np.zeros(self.num_edges, dtype=np.float64)
        parent = np.zeros(self.num_edges, dtype=np.int32)
        child = np.zeros(self.num_edges, dtype=np.int32)
        self.dump_edges(left=left, right=right, parent=parent, child=child)
        edges.set_columns(left=left, right=right, parent=parent, child=child)
        print("edges = ")
        print(edges)

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
                    for group_index in group_index_list:
                        start, end = groups[group_index]
                        left_set.add(tuple([active[j].left for j in range(start, end)]))
                        right_set.add(
                            tuple([active[j].right for j in range(start, end)]))
                        parent_set.add(
                            tuple([active[j].parent for j in range(start, end)]))
                        children = set(active[j].child for j in range(start, end))
                        assert len(children) == 1
                        for j in range(start, end - 1):
                            assert active[j].right == active[j + 1].left
                        # for j in range(start, end):
                        #     print("\t", active[j])
                        # print()
                    assert len(left_set) == 1
                    assert len(right_set) == 1
                    assert len(parent_set) == 1

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
                    if left != 0:
                        output_edges.append(Edge(0, left, 0, new_node))
                        # print("X add", output_edges[-1])
                    if right != self.num_sites:
                        output_edges.append(Edge(right, self.num_sites, 0, new_node))
                        # print("Y add", output_edges[-1])
                    # For each group, add a new segment covering the full interval.
                    for group_index in group_index_list:
                        j = groups[group_index][0]
                        output_edges.append(Edge(left, right, new_node, active[j].child))
                        # print("g add", output_edges[-1])

                    # print("Done\n")

                # print("Setting edges to ", len(output_edges), "new edges")
                for j in range(len(active)):
                    if used[j]:
                        output_edges.append(active[j])
                    # else:
                    #     print("Filtering out", active[j])
                # print("BEFORE")
                # for e in self.edges:
                #     print("\t", e)

                # self.replaces_done += 1
                self.edges = output_edges
                # print("AFTER")
                # for e in self.edges:
                #     print("\t", e)

    def insert_polytomy_ancestor(self, edges):
        """
        Insert a new ancestor for the specified edges and update the parents
        to point to this new ancestor.
        """
        # print("BREAKING POLYTOMY FOR")
        children_time = max(self.time[e.child] for e in edges)
        parent_time = self.time[edges[0].parent]
        time = children_time + (parent_time - children_time) / 2
        new_node = self.add_node(time, is_sample=False)
        e = edges[0]
        # Add the new edge.
        self.edges.append(Edge(0, self.num_sites, e.parent, new_node))
        # Update the edges to point to this new node.
        for e in edges:
            # print("\t", e)
            e.parent = new_node

    def _resolve_polytomies(self):
        # Gather all the egdes pointing to a given parent.
        active = list(self.edges)
        active.sort(key=lambda e: (e.parent, e.left, e.right, e.child))
        parent_count = [0 for _ in range(self.num_nodes)]
        # print("ACTIVE")
        for e in active:
            parent_count[e.parent] += 1
            # print(e.left, e.right, e.parent, e.child, sep="\t")

        group_start = 0
        groups = []
        for j in range(1, len(active)):
            condition = (
                active[j - 1].left != active[j].left or
                active[j - 1].right != active[j].right or
                active[j - 1].parent != active[j].parent)
            if condition:
                size = j - group_start
                if size > 1 and size != parent_count[active[j - 1].parent]:
                    groups.append((group_start, j))
                group_start = j
        j = len(active)
        size = j - group_start
        if size > 1 and size != parent_count[active[j - 1].parent]:
            groups.append((group_start, j))

        for start, end in groups:
            # print("BREAKING POLYTOMY FOR group:", start, end)
            # for j in range(start, end):
            #     print("\t", active[j])

            parent = active[start].parent
            children_time = max(self.time[active[j].child] for j in range(start, end))
            parent_time = self.time[parent]
            time = children_time + (parent_time - children_time) / 2
            new_node = self.add_node(time, is_sample=False)

            # Update the edges to point to this new node.
            for j in range(start, end):
                active[j].parent = new_node
            # Add the new edge.
            active.append(Edge(0, self.num_sites, parent, new_node))

        self.edges = active

    def update(
            self, num_nodes, time, left, right, parent, child, site, node,
            derived_state):
        for _ in range(num_nodes):
            self.add_node(time)
        for l, r, p, c in zip(left, right, parent, child):
            self.edges.append(Edge(l, r, p, c))

        for s, u, d in zip(site, node, derived_state):
            self.mutations[s].append((u, d))

        if self.resolve_polytomies:
            self._resolve_polytomies()

        # print("replaces_done = ", self.replaces_done)
        # if self.replaces_done < 2:
        if self.resolve_shared_recombinations:
            self._replace_recombinations()

        # Index the edges
        M = len(self.edges)
        self.insertion_order = sorted(
            range(M), key=lambda j: (
                self.edges[j].left, self.time[self.edges[j].parent]))
        self.removal_order = sorted(
            range(M), key=lambda j: (
                self.edges[j].right, -self.time[self.edges[j].parent]))
        # print("AFTER UPDATE")
        # self.print_state()

    def dump_nodes(self, flags, time):
        time[:] = self.time[:self.num_nodes]
        flags[:] = self.flags

    def dump_edges(self, left, right, parent, child):
        x = np.hstack([self.positions, [self.sequence_length]])
        x[0] = 0
        for j, edge in enumerate(self.edges):
            left[j] = x[edge.left]
            right[j] = x[edge.right]
            parent[j] = edge.parent
            child[j] = edge.child

    def dump_mutations(self, site, node, derived_state, parent):
        j = 0
        for l in sorted(self.mutations.keys()):
            p = j
            for u, d in self.mutations[l]:
                site[j] = l
                node[j] = u
                derived_state[j] = d
                parent[j] = -1
                if j != p:
                    parent[j] = p
                j += 1


def is_descendant(pi, u, v):
    """
    Returns True if the specified node u is a descendent of node v. That is,
    v is on the path to root from u.
    """
    ret = False
    if v != -1:
        # print("IS_DESCENDENT(", u, v, ")")
        while u != v and u != msprime.NULL_NODE:
            # print("\t", u)
            u = pi[u]
        # print("END, ", u, v)
        ret = u == v
    return ret


class AncestorMatcher(object):

    def __init__(self, tree_sequence_builder, recombination_rate, error_rate=0):
        self.tree_sequence_builder = tree_sequence_builder
        self.recombination_rate = recombination_rate
        self.error_rate = error_rate
        self.num_sites = tree_sequence_builder.num_sites
        self.positions = tree_sequence_builder.positions

    def get_max_likelihood_node(self, L):
        """
        Returns the node with the maxmimum likelihood from the specified map.
        """
        u = -1
        max_likelihood = -1
        for node, likelihood in L.items():
            if likelihood > max_likelihood:
                u = node
                max_likelihood = likelihood
        assert u != -1
        return u

    def find_path(self, h):

        M = len(self.tree_sequence_builder.edges)
        I = self.tree_sequence_builder.insertion_order
        O = self.tree_sequence_builder.removal_order
        n = self.tree_sequence_builder.num_nodes
        m = self.tree_sequence_builder.num_sites
        pi = np.zeros(n, dtype=int) - 1
        L = {u: 1.0 for u in range(n)}
        traceback = [dict(L) for _ in range(m)]
        edges = self.tree_sequence_builder.edges

        r = 1 - np.exp(-self.recombination_rate / n)
        recomb_proba = r / n
        no_recomb_proba = 1 - r + r / n
        err = self.error_rate

        j = 0
        k = 0
        while j < M:
            left = edges[I[j]].left
            while edges[O[k]].right == left:
                parent = edges[O[k]].parent
                child = edges[O[k]].child
                k = k + 1
                pi[child] = -1
                if child not in L:
                    # If the child does not already have a u value, traverse
                    # upwards until we find an L value for u
                    u = parent
                    while u not in L:
                        u = pi[u]
                    L[child] = L[u]
            right = edges[O[k]].right
            while j < M and edges[I[j]].left == left:
                parent = edges[I[j]].parent
                child = edges[I[j]].child
                # print("INSERT", parent, child)
                pi[child] = parent
                j += 1
                # Traverse upwards until we find the L value for the parent.
                u = parent
                while u not in L:
                    u = pi[u]
                # The child must have an L value. If it is the same as the parent
                # we can delete it.
                if L[child] == L[u]:
                    del L[child]

            # print("END OF TREE LOOP", left, right)
            # print("left = ", left)
            # print("right = ", right)
            # print(L)
            # print(pi)
            for site in range(left, right):
                state = h[site]
                if site not in self.tree_sequence_builder.mutations:
                    if err == 0:
                        # This is a special case for ancestor matching. Very awkward
                        # flow control here. FIXME
                        if state == 0:
                            assert len(L) > 0
                        traceback[site] = dict(L)
                        if site > 0 and (site - 1) \
                                not in self.tree_sequence_builder.mutations:
                            assert traceback[site] == traceback[site - 1]
                        # NASTY!!!!
                        continue
                    mutation_node = msprime.NULL_NODE
                else:
                    mutation_node = self.tree_sequence_builder.mutations[site][0][0]
                    # Insert an new L-value for the mutation node if needed.
                    if mutation_node not in L:
                        u = mutation_node
                        while u not in L:
                            u = pi[u]
                        L[mutation_node] = L[u]

                # print("Site ", site, "mutation = ", mutation_node, "state = ", state)
                traceback[site] = dict(L)

                distance = 1
                if site > 0:
                    distance = self.positions[site] - self.positions[site - 1]
                # Update the likelihoods for this site.
                # print("Site ", site, "distance = ", distance)
                max_L = -1
                for v in L.keys():
                    x = L[v] * no_recomb_proba * distance
                    assert x >= 0
                    y = recomb_proba * distance
                    if x > y:
                        z = x
                    else:
                        z = y
                    d = is_descendant(pi, v, mutation_node)
                    if state == 1:
                        emission_p = (1 - err) * d + err * (not d)
                    else:
                        emission_p = err * d + (1 - err) * (not d)
                    L[v] = z * emission_p
                    if L[v] > max_L:
                        max_L = L[v]
                assert max_L > 0

                # Normalise
                for v in L.keys():
                    L[v] /= max_L

                # Compress
                # TODO we probably don't need the second dict here and can just take
                # a copy of the keys.
                L_next = {}
                for u in L.keys():
                    if pi[u] != -1:
                        # Traverse upwards until we find another L value
                        v = pi[u]
                        while v not in L:
                            v = pi[v]
                        if L[u] != L[v]:
                            L_next[u] = L[u]
                    else:
                        L_next[u] = L[u]
                L = L_next

        # print("LIKELIHOODS")
        # for l in range(self.num_sites):
        #     print("\t", l, traceback[l])

        u = self.get_max_likelihood_node(L)
        output_edge = Edge(right=m, parent=u)
        output_edges = [output_edge]

        # Now go back through the trees.
        j = M - 1
        k = M - 1
        I = self.tree_sequence_builder.removal_order
        O = self.tree_sequence_builder.insertion_order
        # Construct the matched haplotype
        hp = np.zeros(m, np.int8)
        while j >= 0:
            right = edges[I[j]].right
            while edges[O[k]].left == right:
                pi[edges[O[k]].child] = -1
                k -= 1
            left = edges[O[k]].left
            while j >= 0 and edges[I[j]].right == right:
                pi[edges[I[j]].child] = edges[I[j]].parent
                j -= 1
            for l in range(right - 1, max(left - 1, 0), -1):
                u = output_edge.parent
                if l in self.tree_sequence_builder.mutations:
                    if is_descendant(
                            pi, u, self.tree_sequence_builder.mutations[l][0][0]):
                        hp[l] = 1
                L = traceback[l]
                v = u
                while v not in L:
                    v = pi[v]
                x = L[v]
                # TODO check this approximately!!
                if x != 1.0:
                    output_edge.left = l
                    u = self.get_max_likelihood_node(L)
                    output_edge = Edge(right=l, parent=u)
                    output_edges.append(output_edge)
                assert l > 0
        output_edge.left = 0
        l = 0
        if l in self.tree_sequence_builder.mutations:
            if is_descendant(pi, u, self.tree_sequence_builder.mutations[l][0][0]):
                hp[l] = 1

        self.mean_traceback_size = sum(len(t) for t in traceback) / self.num_sites

        left = np.zeros(len(output_edges), dtype=np.uint32)
        right = np.zeros(len(output_edges), dtype=np.uint32)
        parent = np.zeros(len(output_edges), dtype=np.int32)
        for j, e in enumerate(output_edges):
            left[j] = e.left
            right[j] = e.right
            parent[j] = e.parent
        return (left, right, parent), hp
