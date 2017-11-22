# TODO copyright and license.
"""
TODO module docs.
"""

import collections
import queue
import time
import datetime
import logging

import numpy as np
import tqdm
import humanize
import msprime
import zarr

import _tsinfer
import tsinfer.formats as formats
import tsinfer.algorithm as algorithm
import tsinfer.threads as threads

logger = logging.getLogger(__name__)

UNKNOWN_ALLELE = 255
PHRED_MAX = 255


def proba_to_phred(probability, min_value=1e-10):
    """
    Returns the specfied array of probability values in phred
    encoding, i.e., -10 log(p, 10) rounded to the nearest integer.
    If the input probability is zero then this is encoded as a phred score of 255.
    """
    P = np.array(probability, copy=True)
    scalar_input = False
    if P.ndim == 0:
        P = P[None]  # Makes P 1D
        scalar_input = True
    if np.any(P > 1):
        raise ValueError("Values > 1 not permitted")
    zeros = np.where(P <= min_value)[0]
    P[zeros] = 1  # Avoid division by zero warnings.
    ret = -10 * np.log10(P)
    ret[zeros] = PHRED_MAX
    ret = np.round(ret).astype(np.uint8)
    if scalar_input:
        return np.squeeze(ret)
    return ret


def phred_to_proba(phred_score):
    """
    Returns the specified phred score as a probability, i.e., 10^{-Q / 10}.
    """
    Q = np.asarray(phred_score, dtype=np.float64)
    scalar_input = False
    if Q.ndim == 0:
        Q = Q[None]  # Makes Q 1D
        scalar_input = True
    zeros = np.where(Q >= PHRED_MAX)[0]
    ret = 10**(-Q / 10)
    ret[zeros] = 0
    if scalar_input:
        return np.squeeze(ret)
    return ret



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
    if method == "C":
        logger.debug("Using C AncestorBuilder implementation")
        ancestor_builder = _tsinfer.AncestorBuilder(num_samples, num_sites)
    else:
        logger.debug("Using Python AncestorBuilder implementation")
        ancestor_builder = algorithm.AncestorBuilder(num_samples, num_sites)

    progress_monitor = tqdm.tqdm(total=num_sites, disable=not progress)
    logger.info("Starting site addition")
    for j, v in enumerate(input_file.site_genotypes()):
        frequency = int(np.sum(v))
        ancestor_builder.add_site(j, frequency, v)
        progress_monitor.update()
    progress_monitor.close()
    logger.info("Finished adding sites")

    descriptors = ancestor_builder.ancestor_descriptors()
    num_ancestors = 1 + len(descriptors)
    total_num_focal_sites = sum(len(d[1]) for d in descriptors)
    oldest_time = 1
    if len(descriptors) > 0:
        oldest_time = descriptors[0][0] + 1
    ancestor_file.initialise(
        num_ancestors, oldest_time, total_num_focal_sites, chunk_size=chunk_size,
        compress=compress)

    logger.info("Starting build for {} ancestors".format(num_ancestors))
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
        input_hdf5, ancestors_hdf5, output_path, method="C", progress=False,
        num_threads=0, output_interval=None):
    """
    Runs the copying process of the specified input and ancestors and returns
    the resulting tree sequence.
    """
    input_file = formats.InputFile(input_hdf5)
    ancestors_file = formats.AncestorFile(ancestors_hdf5, input_file, 'r')

    matcher = AncestorMatcher(
        input_file, ancestors_file, output_path=output_path, method=method,
        progress=progress, num_threads=num_threads, output_interval=output_interval)
    matcher.match_ancestors()


def match_samples(input_data, ancestors_ts, method="C", progress=False, num_threads=0):
    input_file = formats.InputFile(input_data)
    manager = SampleMatcher(
        input_file, ancestors_ts, method=method, progress=progress,
        num_threads=num_threads)
    manager.match_samples()
    return manager.finalise()


class Matcher(object):

    # The description for the progress monitor bar.
    progress_bar_description = None

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
        # This is a safe maximum because the max number of ancestors we can have is
        # the number of sites + 1 (for the oldest ancestor).
        max_nodes = self.num_samples + self.num_sites + 1
        self.tree_sequence_builder = self.tree_sequence_builder_class(
            self.sequence_length, self.positions, self.recombination_rate,
            max_nodes=max_nodes, max_edges=max_edges)
        logger.debug("Allocated tree sequence builder with max_nodes={}".format(
            max_nodes))

        # Allocate the matchers and statistics arrays.
        num_threads = max(1, self.num_threads)
        self.match = [np.zeros(self.num_sites, np.uint8) for _ in range(num_threads)]
        self.results = [ResultBuffer() for _ in range(num_threads)]
        self.mean_traceback_size = np.zeros(num_threads)
        self.num_matches = np.zeros(num_threads)
        self.matcher = [
            self.ancestor_matcher_class(self.tree_sequence_builder, 0)
            for _ in range(num_threads)]
        # The progress monitor is allocated later by subclasses.
        self.progress_monitor = None

    def allocate_progress_monitor(self, total, initial=0, postfix=None):
        bar_format = (
            "{desc}{percentage:3.0f}%|{bar}"
            "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]")
        self.progress_monitor = tqdm.tqdm(
            desc=self.progress_bar_description, bar_format=bar_format,
            total=total, disable=not self.progress, initial=initial,
            smoothing=0.01, postfix=postfix, dynamic_ncols=True)

    def _find_path(self, child_id, haplotype, start, end, thread_index=0):
        """
        Finds the path of the specified haplotype and upates the results
        for the specified thread_index.
        """
        matcher = self.matcher[thread_index]
        results = self.results[thread_index]
        match = self.match[thread_index]
        left, right, parent = matcher.find_path(haplotype, start, end, match)
        results.add_edges(left, right, parent, child_id)
        self.progress_monitor.update()
        self.mean_traceback_size[thread_index] += matcher.mean_traceback_size
        self.num_matches[thread_index] += 1

        # TODO add a function to do this output this debug info.
        # if self.traceback_file_pattern is not None:
        #     # Write out the traceback debug.
        #     filename = self.traceback_file_pattern.format(node_id)
        #     traceback = [matcher.get_traceback(l) for l in range(self.num_sites)]
        #     with open(filename, "wb") as f:
        #         debug = {
        #             "node_id:": node_id,
        #             "focal_sites": focal_sites,
        #             "ancestor": ancestor,
        #             "start": start,
        #             "end": end,
        #             "match": match,
        #             "traceback": traceback}
        #         pickle.dump(debug, f)
        #         logger.debug(
        #             "Dumped ancestor traceback debug to {}".format(filename))

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
        return msprime.load_tables(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            sequence_length=sequence_length)


class AncestorMatcher(Matcher):
    progress_bar_description = "match-ancestors"

    def __init__(
            self, input_file, ancestors_file, output_path, output_interval=None,
            **kwargs):
        super().__init__(input_file, **kwargs)
        self.output_interval = 2**32  # Arbitrary very large number of minutes.
        if output_interval is not None:
            self.output_interval = output_interval
        if self.output_interval < 1:
            raise ValueError("Output interval must be at least 1 minute")
        self.output_path = output_path
        self.last_output_time = time.time()
        self.ancestors_file = ancestors_file
        self.num_ancestors = self.ancestors_file.num_ancestors
        self.epoch = self.ancestors_file.time
        self.focal_sites = self.ancestors_file.focal_sites
        self.start = self.ancestors_file.start
        self.end = self.ancestors_file.end
        # This is an iterator over all ancestral haplotypes.
        self.haplotypes = self.ancestors_file.ancestor_haplotypes()

        # Create a list of all ID ranges in each epoch.
        breaks = np.where(self.epoch[1:] != self.epoch[:-1])[0]
        start = np.hstack([[0], breaks + 1])
        end = np.hstack([breaks + 1, [self.num_ancestors]])
        self.epoch_slices = np.vstack([start, end]).T
        self.num_epochs = self.epoch_slices.shape[0]
        self.allocate_progress_monitor(
            self.num_ancestors, initial=1, postfix=self.__epoch_info_dict(0))

    def __epoch_info_dict(self, epoch_index):
        start, end = self.epoch_slices[epoch_index]
        return collections.OrderedDict([
            ("epoch", str(self.epoch[start])),
            ("nanc", str(end - start))
        ])

    def __update_progress_epoch(self, epoch_index):
        """
        Updates the progress monitor to show information about the present epoch
        """
        self.progress_monitor.set_postfix(self.__epoch_info_dict(epoch_index))

    def __ancestor_find_path(self, ancestor_id, haplotype, thread_index=0):
        focal_sites = self.focal_sites[ancestor_id]
        start = self.start[ancestor_id]
        end = self.end[ancestor_id]
        self.results[thread_index].add_mutations(focal_sites, ancestor_id)
        assert np.all(haplotype[0: start] == UNKNOWN_ALLELE)
        assert np.all(haplotype[end:] == UNKNOWN_ALLELE)
        assert np.all(haplotype[focal_sites] == 1)
        haplotype[focal_sites] = 0
        self._find_path(ancestor_id, haplotype, start, end, thread_index)
        assert np.all(self.match[thread_index] == haplotype)

    def __complete_epoch(self, epoch_index):
        start, end = map(int, self.epoch_slices[epoch_index])
        num_ancestors_in_epoch = end - start
        current_time = self.epoch[start]
        epoch_results = ResultBuffer.combine(self.results)
        self.tree_sequence_builder.update(
            num_ancestors_in_epoch, current_time,
            epoch_results.left, epoch_results.right, epoch_results.parent,
            epoch_results.child, epoch_results.site, epoch_results.node,
            epoch_results.derived_state)
        mean_memory = np.mean([matcher.total_memory for matcher in self.matcher])
        logger.debug(
            "Finished epoch {} with {} ancestors; mean_tb_size={:.2f} "
            "edges={}; mean_matcher_mem={}".format(
                current_time, num_ancestors_in_epoch,
                np.sum(self.mean_traceback_size) / np.sum(self.num_matches),
                self.tree_sequence_builder.num_edges,
                humanize.naturalsize(mean_memory, binary=True)))
        self.mean_traceback_size[:] = 0
        self.num_matches[:] = 0
        for results in self.results:
            results.clear()
        # Output the current state if appropriate
        delta = datetime.timedelta(seconds=time.time() - self.last_output_time)
        if delta.total_seconds() >= self.output_interval * 60:
            # TODO We need some way of indicating that the output is incomplete.
            # Probably simplest is to read it back into h5py and stick in an
            # attribute just saying it's a partial read.
            self.store_output()
            self.last_output_time = time.time()
            logger.info("Saved checkpoint {}".format(self.output_path))

    def __match_ancestors_single_threaded(self):
        for j in range(1, self.num_epochs):
            self.__update_progress_epoch(j)
            start, end = map(int, self.epoch_slices[j])
            for ancestor_id in range(start, end):
                a = next(self.haplotypes)
                self.__ancestor_find_path(ancestor_id, a)
            self.__complete_epoch(j)

    def __match_ancestors_multi_threaded(self):
        # See note on match samples multithreaded below. Should combine these
        # into a single function. Possibly when trying to make the thread
        # error handling more robust.

        queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
        match_queue = queue.Queue(queue_depth)

        def match_worker(thread_index):
            while True:
                work = match_queue.get()
                if work is None:
                    break
                ancestor_id, a = work
                self.__ancestor_find_path(ancestor_id, a, thread_index)
                match_queue.task_done()
            match_queue.task_done()

        match_threads = [
            threads.queue_consumer_thread(
                match_worker, match_queue, name="match-worker-{}".format(j),
                index=j)
            for j in range(self.num_threads)]
        logger.info("Started {} match worker threads".format(self.num_threads))

        for j in range(1, self.num_epochs):
            self.__update_progress_epoch(j)
            start, end = map(int, self.epoch_slices[j])
            for ancestor_id in range(start, end):
                a = next(self.haplotypes)
                match_queue.put((ancestor_id, a))
            # Block until all matches have completed.
            match_queue.join()
            self.__complete_epoch(j)

        # Stop the the worker threads.
        for j in range(self.num_threads):
            match_queue.put(None)
        for j in range(self.num_threads):
            match_threads[j].join()

    def match_ancestors(self):
        logger.info("Starting ancestor matching for {} epochs".format(self.num_epochs))
        # Insert the oldest ancestor
        self.tree_sequence_builder.update(1, self.epoch[0], [], [], [], [], [], [], [])
        a = next(self.haplotypes)
        assert np.all(a == 0)
        if self.num_threads <= 0:
            self.__match_ancestors_single_threaded()
        else:
            self.__match_ancestors_multi_threaded()
        self.store_output()
        logger.info("Finished ancestor matching")

    def store_output(self):
        ts = self.get_tree_sequence(rescale_positions=False)
        ts.dump(self.output_path)


class SampleMatcher(Matcher):
    progress_bar_description = "match-samples"

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
        self.sample_haplotypes = self.input_file.sample_haplotypes()
        self.allocate_progress_monitor(self.num_samples)

    def __process_sample(self, sample_id, haplotype, thread_index=0):
        self._find_path(sample_id, haplotype, 0, self.num_sites, thread_index)
        match = self.match[thread_index]
        diffs = np.where(haplotype != match)[0]
        derived_state = haplotype[diffs]
        self.results[thread_index].add_mutations(diffs, sample_id, derived_state)

    def __match_samples_single_threaded(self):
        j = 0
        for a in self.sample_haplotypes:
            sample_id = self.tree_sequence_builder.num_nodes + j
            self.__process_sample(sample_id, a)
            j += 1
        assert j == self.num_samples

    def __match_samples_multi_threaded(self):
        # Note that this function is not almost identical to the match_ancestors
        # multithreaded function above. All we need to do is provide a function
        # to do the matching and some producer for the actual items and we
        # can bring this into a single function.

        queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
        match_queue = queue.Queue(queue_depth)

        def match_worker(thread_index):
            while True:
                work = match_queue.get()
                if work is None:
                    break
                sample_id, a = work
                self.__process_sample(sample_id, a, thread_index)
                match_queue.task_done()
            match_queue.task_done()

        match_threads = [
            threads.queue_consumer_thread(
                match_worker, match_queue, name="match-worker-{}".format(j),
                index=j)
            for j in range(self.num_threads)]
        logger.info("Started {} match worker threads".format(self.num_threads))

        j = 0
        for a in self.sample_haplotypes:
            sample_id = self.tree_sequence_builder.num_nodes + j
            match_queue.put((sample_id, a))
            j += 1
        assert j == self.num_samples

        # Stop the the worker threads.
        for j in range(self.num_threads):
            match_queue.put(None)
        for j in range(self.num_threads):
            match_threads[j].join()

    def match_samples(self):
        logger.info("Started matching for {} samples".format(self.num_samples))
        if self.num_threads <= 0:
            self.__match_samples_single_threaded()
        else:
            self.__match_samples_multi_threaded()
        results = ResultBuffer.combine(self.results)
        self.tree_sequence_builder.update(
            self.num_samples, 0,
            results.left, results.right, results.parent, results.child,
            results.site, results.node, results.derived_state)
        logger.info("Finished sample matching")

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
