# TODO copyright and license.
"""
TODO module docs.
"""

import contextlib
import collections
import queue
import threading
import time
import _thread
import traceback
import pickle
import logging

import numpy as np
import tqdm
import humanize
import msprime
import zarr

import _tsinfer
import tsinfer.formats as formats
import tsinfer.algorithm as algorithm

logger = logging.getLogger(__name__)

# prctl is an optional extra; it allows us assign meaninful names to threads
# for debugging.
_prctl_available = False
try:
    import prctl
    _prctl_available = True
except ImportError:
    pass

_numa_available = False
try:
    import numa
    _numa_available = True
except ImportError:
    pass


UNKNOWN_ALLELE = 255

def infer(
        genotypes, positions, sequence_length, recombination_rate, sample_error=0,
        method="C", num_threads=0, progress=False):
    positions_array = np.array(positions)
    # If the input recombination rate is a single number set this value for all sites.
    recombination_rate_array = np.zeros(positions_array.shape[0], dtype=np.float64)
    recombination_rate_array[:] = recombination_rate

    input_root = zarr.group()
    formats.InputFile.build(
        input_root, genotypes=genotypes, position=positions,
        recombination_rate=recombination_rate_array, sequence_length=sequence_length,
        compress=False)
    ancestors_root = zarr.group()
    build_ancestors(input_root, ancestors_root, method=method, compress=False)
    ancestors_ts = match_ancestors(
        input_root, ancestors_root, method=method, num_threads=num_threads)
    inferred_ts = match_samples(
        input_root, ancestors_ts, method=method, num_threads=num_threads)
    return inferred_ts


def build_ancestors(
        input_hdf5, ancestor_hdf5, progress=False, method="C", compress=True,
        chunk_size=None):

    input_file = formats.InputFile(input_hdf5)
    ancestor_file = formats.AncestorFile(ancestor_hdf5, input_file, 'w')

    num_sites = input_file.num_sites
    num_samples = input_file.num_samples
    if method == "P":
        logger.debug("Using Python AncestorBuilder implementation")
        ancestor_builder = algorithm.AncestorBuilder(num_samples, num_sites)
    else:
        logger.debug("Using C AncestorBuilder implementation")
        ancestor_builder = _tsinfer.AncestorBuilder(num_samples, num_sites)

    progress_monitor = tqdm.tqdm(total=num_sites, disable=not progress)
    logger.info("Starting site addition")
    for j, v in enumerate(input_file.site_genotypes()):
        ancestor_builder.add_site(j, int(np.sum(v)), v)
        progress_monitor.update()
    progress_monitor.close()
    logger.info("Finished adding sites")

    descriptors = ancestor_builder.ancestor_descriptors()
    num_ancestors = 1 + len(descriptors)
    total_num_focal_sites = sum(len(d[1]) for d in descriptors)
    oldest_time = 1
    if len(descriptors) > 0:
        oldest_time = descriptors[0][0] + 1
    else:
        raise ValueError(
            "Zero ancestors current not supported due to bug in Zarr. See "
            "https://github.com/alimanfoo/zarr/issues/187")
    ancestor_file.initialise(
        num_ancestors, oldest_time, total_num_focal_sites, chunk_size=chunk_size,
        compress=compress)

    a = np.zeros(num_sites, dtype=np.uint8)
    progress_monitor = tqdm.tqdm(total=num_ancestors, initial=1, disable=not progress)
    for freq, focal_sites in descriptors:
        before = time.perf_counter()
        s, e = ancestor_builder.make_ancestor(focal_sites, a)
        duration = time.perf_counter() - before
        logger.debug(
            "Made ancestor with {} focal sites and length={} in {:.2f}s.".format(
                focal_sites.shape[0], e - s, duration))
        ancestor_file.add_ancestor(
            start=s, end=e, ancestor_time=freq, focal_sites=focal_sites,
            haplotype=a)
        progress_monitor.update()
    ancestor_file.finalise()
    progress_monitor.close()
    logger.info("Finished building ancestors")


def match_ancestors(
        input_hdf5, ancestors_hdf5, method="C", progress=False, num_threads=0):
    """
    Runs the copying process of the specified input and ancestors and returns
    the resulting tree sequence.
    """
    input_file = formats.InputFile(input_hdf5)
    ancestors_file = formats.AncestorFile(ancestors_hdf5, input_file, 'r')

    matcher = AncestorMatcher(
        input_file, ancestors_file, method=method, progress=progress,
        num_threads=num_threads)
    matcher.match_ancestors()
    return matcher.get_tree_sequence(rescale_positions=False)


def match_samples(input_data, ancestors_ts, method="C", progress=False, num_threads=0):
    input_file = formats.InputFile(input_data)
    manager = SampleMatcher(
        input_file, ancestors_ts, method=method, progress=progress,
        num_threads=num_threads)
    manager.match_samples()
    return manager.finalise()


class Matcher(object):

    def __init__(
            self, input_file, num_threads=1, method="C", progress=False,
            traceback_file_pattern=None):
        self.input_file = input_file
        self.num_threads = num_threads
        self.num_samples = self.input_file.num_samples
        self.num_sites = self.input_file.num_sites
        self.sequence_length = self.input_file.sequence_length
        self.positions = self.input_file.position
        self.recombination_rate = self.input_file.recombination_rate
        self.progress = progress
        self.tree_sequence_builder_class = algorithm.TreeSequenceBuilder
        self.ancestor_matcher_class = algorithm.AncestorMatcher
        if method == "C":
            self.tree_sequence_builder_class = _tsinfer.TreeSequenceBuilder
            self.ancestor_matcher_class = _tsinfer.AncestorMatcher
        self.tree_sequence_builder = None
        # Debugging. Set this to a file path like "traceback_{}.pkl" to store the
        # the tracebacks for each node ID and other debugging information.
        self.traceback_file_pattern = traceback_file_pattern

        # Allocate 64K edges initially. This will double as needed and will quickly be
        # big enough even for very large instances.
        max_edges = 64 * 1024
        max_nodes = self.num_samples + self.num_sites
        self.tree_sequence_builder = self.tree_sequence_builder_class(
            self.sequence_length, self.positions, self.recombination_rate,
            max_nodes=max_nodes, max_edges=max_edges)

    def get_tree_sequence(self, rescale_positions=True):
        """
        Returns the current state of the build tree sequence. All samples and
        ancestors will have the sample node flag set.
        """
        tsb = self.tree_sequence_builder
        flags, time = tsb.dump_nodes()
        nodes = msprime.NodeTable()
        nodes.set_columns(flags=flags, time=time)

        left, right, parent, child = tsb.dump_edges()
        if rescale_positions:
            sequence_length = self.sequence_length
            position = self.positions
            x = np.hstack([self.positions, [self.sequence_length]])
            x[0] = 0
            left = x[left]
            right = x[right]
        else:
            position=np.arange(tsb.num_sites)
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
        return msprime.load_tables(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            sequence_length=sequence_length)

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
            logger.critical("Exception occured in thread; exiting")
            logger.critical(traceback.format_exc())
            _thread.interrupt_main()


class AncestorMatcher(Matcher):

    def __init__(self, input_file, ancestors_file, **kwargs):
        super().__init__(input_file, **kwargs)
        self.ancestors_file = ancestors_file

    def match_ancestors(self):
        num_ancestors = self.ancestors_file.num_ancestors
        haplotypes = self.ancestors_file.ancestor_haplotypes()

        epoch = self.ancestors_file.time
        focal_sites = self.ancestors_file.focal_sites
        start = self.ancestors_file.start
        end = self.ancestors_file.end

        epoch_info = {}
        current_time = epoch[0]
        num_ancestors_in_epoch = 0
        for j in range(num_ancestors):
            if epoch[j] != current_time:
                epoch_info[current_time] = collections.OrderedDict([
                    ("epoch", str(current_time)),
                    ("nanc", str(num_ancestors_in_epoch))])
                num_ancestors_in_epoch = 0
                current_time = epoch[j]
            num_ancestors_in_epoch += 1
        epoch_info[current_time] = collections.OrderedDict([
            ("epoch", str(current_time)),
            ("nanc", str(num_ancestors_in_epoch))])

        bar_format=(
            "{desc}{percentage:3.0f}%|{bar}"
            "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]")
        progress_monitor = tqdm.tqdm(
            desc="match-ancestors", bar_format=bar_format,
            total=num_ancestors, disable=not self.progress, initial=1, smoothing=0.01,
            postfix=epoch_info[epoch[0]])

        self.tree_sequence_builder.update(1, epoch[0], [], [], [], [], [], [], [])
        a = next(haplotypes)
        assert np.all(a == 0)

        if self.num_threads <= 0:

            matcher = self.ancestor_matcher_class(self.tree_sequence_builder, 0)
            results = ResultBuffer()
            match = np.zeros(self.num_sites, np.uint8)

            current_time = epoch[0]
            num_ancestors_in_epoch = 0
            for j in range(1, num_ancestors):
                if epoch[j] != current_time:
                    # print("Flushing at time ", current_time)
                    self.tree_sequence_builder.update(
                        num_ancestors_in_epoch, current_time,
                        results.left, results.right, results.parent, results.child,
                        results.site, results.node, results.derived_state)
                    results.clear()
                    num_ancestors_in_epoch = 0
                    current_time = epoch[j]
                    progress_monitor.set_postfix(epoch_info[current_time])

                num_ancestors_in_epoch += 1
                a = next(haplotypes)
                self.__find_path_ancestor(
                    a, start[j], end[j], j, focal_sites[j], matcher, results, match)
                progress_monitor.update()
            self.tree_sequence_builder.update(
                num_ancestors_in_epoch, current_time,
                results.left, results.right, results.parent, results.child,
                results.site, results.node, results.derived_state)
        else:

            queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
            match_queue = queue.Queue(queue_depth)
            results = [ResultBuffer() for _ in range(self.num_threads)]
            matcher_memory = np.zeros(self.num_threads)
            mean_traceback_size = np.zeros(self.num_threads)
            num_matches = np.zeros(self.num_threads)

            def match_worker(thread_index):
                with self.catch_thread_error():
                    logger.info(
                        "Ancestor match thread {} starting".format(thread_index))
                    if _prctl_available:
                        prctl.set_name("ancestor-worker-{}".format(thread_index))
                    if _numa_available and numa.available():
                        numa.set_localalloc()
                        logger.debug(
                            "Set NUMA local allocation policy on thread {}."
                            .format(thread_index))
                    match = np.zeros(self.num_sites, np.uint8)
                    matcher = self.ancestor_matcher_class(self.tree_sequence_builder, 0)
                    result_buffer = results[thread_index]
                    while True:
                        work = match_queue.get()
                        if work is None:
                            break
                        node_id, focal_sites, start, end, a = work
                        self.__find_path_ancestor(
                                a, start, end, node_id, focal_sites, matcher,
                                result_buffer, match)
                        progress_monitor.update()
                        mean_traceback_size[thread_index] += matcher.mean_traceback_size
                        num_matches[thread_index] += 1
                        matcher_memory[thread_index] = matcher.total_memory
                        match_queue.task_done()
                    match_queue.task_done()
                    logger.info("Ancestor match thread {} exiting".format(thread_index))

            match_threads = [
                threading.Thread(target=match_worker, args=(j,), daemon=True)
                for j in range(self.num_threads)]
            for j in range(self.num_threads):
                match_threads[j].start()

            num_ancestors_in_epoch = 0
            current_time = epoch[1]
            for j in range(1, num_ancestors):
                if epoch[j] != current_time:
                    # Wait until the match_queue is empty.
                    # TODO Note that these calls to queue.join prevent errors that happen in the
                    # worker process from propagating back here. Might be better to use some
                    # other way of handling threading sync.
                    match_queue.join()
                    epoch_results = ResultBuffer.combine(results)
                    self.tree_sequence_builder.update(
                        num_ancestors_in_epoch, current_time,
                        epoch_results.left, epoch_results.right, epoch_results.parent,
                        epoch_results.child, epoch_results.site, epoch_results.node,
                        epoch_results.derived_state)
                    mean_memory = np.mean(matcher_memory)
                    logger.debug(
                        "Finished epoch {} with {} ancestors; mean_tb_size={:.2f} edges={}; "
                        "mean_matcher_mem={}".format(
                            current_time, num_ancestors_in_epoch,
                            np.sum(mean_traceback_size) / np.sum(num_matches),
                            self.tree_sequence_builder.num_edges,
                            humanize.naturalsize(mean_memory, binary=True)))
                    mean_traceback_size[:] = 0
                    num_matches[:] = 0
                    for k in range(self.num_threads):
                        results[k].clear()
                    num_ancestors_in_epoch = 0
                    progress_monitor.set_postfix(epoch_info[current_time])
                    current_time = epoch[j]

                num_ancestors_in_epoch += 1
                a = next(haplotypes)
                match_queue.put((j, focal_sites[j], start[j], end[j], a))

            for j in range(self.num_threads):
                match_queue.put(None)
            for j in range(self.num_threads):
                match_threads[j].join()

            epoch_results = ResultBuffer.combine(results)
            self.tree_sequence_builder.update(
                num_ancestors_in_epoch, current_time,
                epoch_results.left, epoch_results.right, epoch_results.parent,
                epoch_results.child, epoch_results.site, epoch_results.node,
                epoch_results.derived_state)
            mean_memory = np.mean(matcher_memory)
            logger.debug(
                "Finished epoch {}; mean_tb_size={:.2f} edges={}; "
                "mean_matcher_mem={}".format(
                    current_time, np.sum(mean_traceback_size) / np.sum(num_matches),
                    self.tree_sequence_builder.num_edges,
                    humanize.naturalsize(mean_memory, binary=True)))

            # Signal to the workers to quit and clean up.
            for j in range(self.num_threads):
                match_queue.put(None)
            for j in range(self.num_threads):
                match_threads[j].join()

    def __find_path_ancestor(
            self, ancestor, start, end, node_id, focal_sites, matcher, results, match):
        results.add_mutations(focal_sites, node_id)
        assert np.all(ancestor[0: start] == UNKNOWN_ALLELE)
        assert np.all(ancestor[end:] == UNKNOWN_ALLELE)
        assert np.all(ancestor[focal_sites] == 1)
        ancestor[focal_sites] = 0
        left, right, parent = matcher.find_path(ancestor, start, end, match)
        assert np.all(match == ancestor)
        results.add_edges(left, right, parent, node_id)

        if self.traceback_file_pattern is not None:
            # Write out the traceback debug.
            filename = self.traceback_file_pattern.format(node_id)
            traceback = [matcher.get_traceback(l) for l in range(self.num_sites)]
            with open(filename, "wb") as f:
                debug = {
                    "node_id:": node_id,
                    "focal_sites": focal_sites,
                    "ancestor": ancestor,
                    "start": start,
                    "end": end,
                    "match": match,
                    "traceback": traceback}
                pickle.dump(debug, f)
                logger.debug(
                    "Dumped ancestor traceback debug to {}".format(filename))

class SampleMatcher(Matcher):

    def __init__(self, input_file, ancestors_ts, **kwargs):
        super().__init__(input_file, **kwargs)
        tables = ancestors_ts.dump_tables()
        nodes = tables.nodes
        self.tree_sequence_builder.restore_nodes(nodes.time)
        edges = tables.edges
        self.tree_sequence_builder.restore_edges(
            edges.left.astype(np.int32), edges.right.astype(np.int32),
            edges.parent, edges.child)
        mutations = tables.mutations
        self.tree_sequence_builder.restore_mutations(
            mutations.site, mutations.node, mutations.derived_state - ord('0'),
            mutations.parent)

    def match_samples(self):
        results = ResultBuffer(self.num_samples)
        matcher = self.ancestor_matcher_class(self.tree_sequence_builder, 0)
        match = np.zeros(self.num_sites, np.uint8)
        sample_haplotypes = self.input_file.sample_haplotypes()

        progress_monitor = tqdm.tqdm(
            desc="match-samples",
            total=self.num_samples, disable=not self.progress, smoothing=0.01)

        j = 0
        for a in sample_haplotypes:
            sample_id = self.tree_sequence_builder.num_nodes + j
            j += 1
            self.__process_sample(sample_id, a, matcher, results, match)
            progress_monitor.update()
        assert j == self.num_samples

        self.tree_sequence_builder.update(
            self.num_samples, 0,
            results.left, results.right, results.parent, results.child,
            results.site, results.node, results.derived_state)

    def __process_sample(self, sample_id, haplotype, matcher, results, match):
        left, right, parent = matcher.find_path(haplotype, 0, self.num_sites, match)
        diffs = np.where(haplotype != match)[0]
        derived_state = haplotype[diffs]
        results.add_mutations(diffs, sample_id, derived_state)
        results.add_edges(left, right, parent, sample_id)

    def __process_samples_threads(self, sample_error):
        results = [ResultBuffer() for _ in range(self.num_threads)]
        matchers = [
            self.ancestor_matcher_class(self.tree_sequence_builder, sample_error)
            for _ in range(self.num_threads)]
        work_queue = queue.Queue()

        def worker(thread_index):
            with self.catch_thread_error():
                logger.info("Started sample worker thread {}".format(thread_index))
                if _prctl_available:
                    prctl.set_name("sample-worker-{}".format(thread_index))
                if _numa_available and numa.available():
                    numa.set_localalloc()
                    logger.debug(
                        "Set NUMA local allocation policy on thread {}.".format(
                            thread_index))
                mean_traceback_size = 0
                num_matches = 0
                match = np.zeros(self.num_sites, np.uint8)
                while True:
                    sample_index = work_queue.get()
                    if sample_index is None:
                        break
                    self.__process_sample(
                        sample_index, matchers[thread_index], results[thread_index],
                        match)
                    mean_traceback_size += matchers[thread_index].mean_traceback_size
                    num_matches += 1
                    work_queue.task_done()
                if num_matches > 0:
                    mean_traceback_size /= num_matches
                logger.info(
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
        logger.info("Processing {} samples".format(self.num_samples))
        if self.num_threads <= 0:
            results = ResultBuffer(self.num_samples)
            matcher = self.ancestor_matcher_class(
                self.tree_sequence_builder, sample_error)
            match = np.zeros(self.num_sites, np.uint8)
            for j in range(self.num_samples):
                self.__process_sample(j, matcher, results, match)
        else:
            results = self.__process_samples_threads(sample_error)
        print("updating", results.left)
        self.tree_sequence_builder.update(
            self.num_samples, 0,
            results.left, results.right, results.parent, results.child,
            results.site, results.node, results.derived_state)

    def finalise(self):
        logger.info("Finalising tree sequence")
        ts = self.get_tree_sequence()
        N = ts.num_nodes
        sample_ids = np.arange(N - self.num_samples, N, dtype=np.int32)
        ts_simplified = ts.simplify(
            samples=sample_ids, filter_zero_mutation_sites=False)
        logger.debug("simplified from ({}, {}) to ({}, {}) nodes and edges".format(
            ts.num_nodes, ts.num_edges, ts_simplified.num_nodes,
            ts_simplified.num_edges))
        return ts_simplified


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
        size = max(1, sum(result.num_edges for result in result_buffers))
        combined = cls(size)
        for result in result_buffers:
            combined.add_edges(
                result.left, result.right, result.parent, result.child)
            combined.add_mutations(result.site, result.node, result.derived_state)
        return combined
