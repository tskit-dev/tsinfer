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

import numpy as np
import h5py
import tqdm
import humanize
import daiquiri
import msprime

import _tsinfer

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

FORMAT_NAME_KEY = "format_name"
FORMAT_VERSION_KEY = "format_version"

@contextlib.contextmanager
def open_input(path):
    """
    Loads an input HDF5 file from the specified path and returns the corresponding
    h5py file.
    """
    with h5py.File(path, "r") as root:
        try:
            format_name = root.attrs[FORMAT_NAME_KEY]
        except KeyError:
            raise ValueError("HDF5 file not in tsinfer format: format_name missing")
        if format_name != "tsinfer-input":
            raise ValueError(
                "File must be in tsinfer input format: format '{}' not valid".format(
                    format_name))
        # TODO check format version
        # Also check basic integrity like the shapes of the various arrays.
        yield root


@contextlib.contextmanager
def open_ancestors(path):
    """
    Loads an ancestors HDF5 file from the specified path and returns the corresponding
    h5py file.
    """
    with h5py.File(path, "r") as root:
        try:
            format_name = root.attrs[FORMAT_NAME_KEY]
        except KeyError:
            raise ValueError("HDF5 file not in tsinfer format: format_name missing")
        if format_name != "tsinfer-ancestors":
            raise ValueError(
                "File must be in tsinfer ancestor format: format '{}' not valid".format(
                    format_name))
        # TODO check format version
        # Also check basic integrity like the shapes of the various arrays.
        yield root


def write_version_attrs(hdf5_file, format_name, format_version):
    """
    Writes the version attributes for the specified HDF5 file.
    """
    hdf5_file.attrs[FORMAT_NAME_KEY] = format_name
    hdf5_file.attrs[FORMAT_VERSION_KEY] = format_version


def build_ancestors(
        input_hdf5, output_hdf5, progress=False, method="C", compression=None,
        chunk_size=1024):
    # TODO For some reason we need to take a copy here or we don't get the
    # correct results in C. Not sure why this is.
    samples = input_hdf5["samples/haplotypes"][:]
    position = input_hdf5["sites/position"]

    num_samples, num_sites = samples.shape

    if method == "P":
        ancestor_builder = AncestorBuilder(num_samples, num_sites)
    else:
        ancestor_builder = _tsinfer.AncestorBuilder(num_samples, num_sites)

    progress_monitor = tqdm.tqdm(total=num_sites, disable=not progress)
    for j in range(num_sites):
        v = samples[:, j][:]
        ancestor_builder.add_site(j, int(np.sum(v)), v)
        progress_monitor.update()
    progress_monitor.close()

    descriptors = ancestor_builder.ancestor_descriptors()
    num_ancestors = 1 + len(descriptors)
    total_num_focal_sites = sum(len(d[1]) for d in descriptors)

    # Initialise the output file format.
    write_version_attrs(output_hdf5, "tsinfer-ancestors", (0, 1))
    ancestors_group = output_hdf5.create_group("ancestors")
    chunk_size = min(chunk_size, num_ancestors)
    haplotypes = ancestors_group.create_dataset(
            "haplotypes", (num_ancestors, num_sites), dtype=np.int8,
            chunks=(chunk_size, num_sites), compression=compression)
    # Local buffer for the chunks before we flush them to the HDF5 dataset.
    H = np.empty((chunk_size, num_sites))

    a = np.zeros(num_sites, dtype=np.int8)
    time = np.zeros(num_ancestors, dtype=np.int32)
    start = np.zeros(num_ancestors, dtype=np.int32)
    end = np.zeros(num_ancestors, dtype=np.int32)
    num_focal_sites = np.zeros(num_ancestors, dtype=np.uint32)
    focal_sites = np.zeros(total_num_focal_sites, dtype=np.int32)
    time[0] = descriptors[0][0] + 1
    H[0] = a
    offset = 0
    j = 1
    progress_monitor = tqdm.tqdm(total=num_ancestors, initial=1, disable=not progress)
    for freq, ancestor_focal_sites in descriptors:
        s, e = ancestor_builder.make_ancestor(ancestor_focal_sites, a)
        assert j < num_ancestors
        start[j] = s
        end[j] = e
        time[j] = freq
        H[j % chunk_size] = a
        n = len(ancestor_focal_sites)
        num_focal_sites[j] = n
        focal_sites[offset: offset + n] = ancestor_focal_sites
        offset += n
        j += 1
        if j % chunk_size == 0:
            # Flush this chunk to disk
            # print("Flushing", j - chunk_size, j)
            haplotypes[j - chunk_size: j] = H
        progress_monitor.update()

    last_chunk = num_ancestors % chunk_size
    if last_chunk != 0:
        print("last flush: ", last_chunk)
        print("num_ancestors  ", num_ancestors)
        haplotypes[-last_chunk:] = H[:last_chunk]

    ancestors_group.create_dataset("time", (num_ancestors,), data=time)
    ancestors_group.create_dataset("start", (num_ancestors,), data=start)
    ancestors_group.create_dataset("end", (num_ancestors,), data=end)
    ancestors_group.create_dataset(
        "num_focal_sites", (num_ancestors,), data=num_focal_sites)
    ancestors_group.create_dataset(
        "focal_sites", (total_num_focal_sites,), data=focal_sites)


def match_ancestors(
        input_hdf5, ancestors_hdf5, method="C", progress=False, num_threads=0,
        log_level="WARN"):
    """
    Runs the copying process of the specified input and ancestors and returns
    the resulting tree sequence.
    """
    # TODO remove the log_level argument here.
    manager = InferenceManager(
        input_hdf5, ancestors_hdf5, method=method, progress=progress,
        num_threads=num_threads, log_level=log_level)
    manager.match_ancestors()
    return manager.get_tree_sequence()


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
            self, input_hdf5, ancestors_hdf5, num_threads=1, method="C",
            progress=False, log_level="WARNING",
            ancestor_traceback_file_pattern=None):
        self.ancestors_hdf5 = ancestors_hdf5
        self.input_hdf5 = input_hdf5

        samples_group = input_hdf5["samples"]
        sites_group = input_hdf5["sites"]
        haplotypes = samples_group["haplotypes"]
        self.sequence_length = input_hdf5.attrs["sequence_length"]
        self.num_samples = haplotypes.shape[0]
        self.num_sites = haplotypes.shape[1]
        self.positions = sites_group["position"][:]
        self.recombination_rate = sites_group["recombination_rate"][:]
        self.num_threads = num_threads
        # Debugging. Set this to a file path like "traceback_{}.pkl" to store the
        # the tracebacks for each node ID and other debugging information.
        self.ancestor_traceback_file_pattern = ancestor_traceback_file_pattern
        # Set up logging.
        daiquiri.setup(level=log_level)
        self.logger = daiquiri.getLogger()
        self.progress = progress
        self.progress_monitor_lock = None

        self.tree_sequence_builder_class = TreeSequenceBuilder
        self.ancestor_matcher_class = AncestorMatcher
        if method == "C":
            self.tree_sequence_builder_class = _tsinfer.TreeSequenceBuilder
            self.ancestor_matcher_class = _tsinfer.AncestorMatcher
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

    def match_ancestors(self):
        ancestors_group = self.ancestors_hdf5["ancestors"]
        haplotypes_ds = ancestors_group["haplotypes"]

        epoch = ancestors_group["time"][:]
        start = ancestors_group["start"][:]
        end = ancestors_group["end"][:]
        num_focal_sites = ancestors_group["num_focal_sites"][:]
        focal_sites = ancestors_group["focal_sites"][:]
        focal_sites_offset = np.roll(np.cumsum(num_focal_sites), 1)
        focal_sites_offset[0] = 0
        num_ancestors = epoch.shape[0]

        progress_monitor = tqdm.tqdm(
            total=num_ancestors, disable=not self.progress, smoothing=0.1)

        def haplotypes_iter():
            chunk_size = haplotypes_ds.chunks[0]
            assert haplotypes_ds.chunks[1] == self.num_sites
            num_chunks = num_ancestors // chunk_size
            offset = 0
            for k in range(num_chunks):
                before = time.perf_counter()
                A = haplotypes_ds[offset: offset + chunk_size][:]
                duration = time.perf_counter() - before
                self.logger.debug("Decompressed chunk {} of {} in {:.2f}s".format(
                    k, num_chunks, duration))
                offset += chunk_size
                for a in A:
                    # This is probably not the right place to do this as we'll have a
                    # long pause at the end when we're flushing the queue.
                    progress_monitor.update()
                    yield a
            if offset != num_ancestors:
                A = haplotypes_ds[offset:][:]
                for a in A:
                    progress_monitor.update()
                    yield a

        haplotypes = haplotypes_iter()


        # Allocate 64K edges initially. This will double as needed and will quickly be
        # big enough even for very large instances.
        max_edges = 64 * 1024
        # This is a safe maximum
        max_nodes = self.num_samples + self.num_sites
        self.tree_sequence_builder = self.tree_sequence_builder_class(
            self.sequence_length, self.positions, self.recombination_rate,
            max_nodes=max_nodes, max_edges=max_edges)
        self.tree_sequence_builder.update(1, epoch[0], [], [], [], [], [], [], [])
        a = next(haplotypes)
        assert np.all(a == 0)

        if self.num_threads <= 0:

            matcher = self.ancestor_matcher_class(self.tree_sequence_builder, 0)
            results = ResultBuffer()
            match = np.zeros(self.num_sites, np.int8)

            current_time = epoch[0]
            child = 0
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

                num_ancestors_in_epoch += 1
                a = next(haplotypes)
                focal = focal_sites[
                    focal_sites_offset[j]: focal_sites_offset[j] + num_focal_sites[j]]
                assert len(focal) > 0
                self.__find_path_ancestor(
                    a, start[j], end[j], j, focal, matcher, results, match)

            # print("Flushing", current_time)
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
                    self.logger.info(
                        "Ancestor match thread {} starting".format(thread_index))
                    if _prctl_available:
                        prctl.set_name("ancestor-worker-{}".format(thread_index))
                    if _numa_available and numa.available():
                        numa.set_localalloc()
                        self.logger.debug(
                            "Set NUMA local allocation policy on thread {}."
                            .format(thread_index))
                    match = np.zeros(self.num_sites, np.int8)
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
                        mean_traceback_size[thread_index] += matcher.mean_traceback_size
                        num_matches[thread_index] += 1
                        matcher_memory[thread_index] = matcher.total_memory
                        match_queue.task_done()
                    match_queue.task_done()
                    self.logger.info("Ancestor match thread {} exiting".format(thread_index))

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
                    self.logger.debug(
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
                    current_time = epoch[j]

                num_ancestors_in_epoch += 1
                a = next(haplotypes)
                focal = focal_sites[
                    focal_sites_offset[j]: focal_sites_offset[j] + num_focal_sites[j]]
                assert len(focal) > 0
                match_queue.put((j, focal, start[j], end[j], a))

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
            self.logger.debug(
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

        assert np.all(ancestor[0: start] == -1)
        assert np.all(ancestor[end:] == -1)
        # TODO change the ancestor builder so that we don't need to do this.
        assert np.all(ancestor[focal_sites] == 1)
        ancestor[focal_sites] = 0
        left, right, parent = matcher.find_path(ancestor, start, end, match)
        assert np.all(match == ancestor)
        results.add_edges(left, right, parent, node_id)

        if self.ancestor_traceback_file_pattern is not None:
            # Write out the traceback debug.
            filename = self.ancestor_traceback_file_pattern.format(node_id)
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
                self.logger.debug(
                    "Dumped ancestor traceback debug to {}".format(filename))

    def process_ancestors(self):
        self.logger.info("Copying {} ancestors".format(self.num_ancestors))
        if self.num_threads == 1:
            self.__process_ancestors_single_threaded()
        else:
            self.__process_ancestors_multi_threaded()

    def __process_ancestors_multi_threaded(self):
        match_queue = queue.Queue()
        results = [ResultBuffer() for _ in range(self.num_threads)]
        matcher_memory = np.zeros(self.num_threads)
        mean_traceback_size = np.zeros(self.num_threads)
        num_matches = np.zeros(self.num_threads)

        def match_worker(thread_index):
            with self.catch_thread_error():
                self.logger.info(
                    "Ancestor match thread {} starting".format(thread_index))
                if _prctl_available:
                    prctl.set_name("ancestor-worker-{}".format(thread_index))
                if _numa_available and numa.available():
                    numa.set_localalloc()
                    self.logger.debug(
                        "Set NUMA local allocation policy on thread {}."
                        .format(thread_index))
                match = np.zeros(self.num_sites, np.int8)
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
                    mean_traceback_size[thread_index] += matcher.mean_traceback_size
                    num_matches[thread_index] += 1
                    matcher_memory[thread_index] = matcher.total_memory
                    match_queue.task_done()
                match_queue.task_done()
                self.logger.info("Ancestor match thread {} exiting".format(thread_index))

        match_threads = [
            threading.Thread(target=match_worker, args=(j,), daemon=True)
            for j in range(self.num_threads)]
        for j in range(self.num_threads):
            match_threads[j].start()

        a = np.zeros(self.num_sites, dtype=np.int8)
        for epoch in range(self.num_epochs):
            time = self.epoch_time[epoch]
            ancestor_focal_sites = self.epoch_ancestors[epoch]
            self.logger.debug("Epoch {}; time = {}; {} ancestors to process".format(
                epoch, time, len(ancestor_focal_sites)))
            child = self.tree_sequence_builder.num_nodes
            for focal_sites in ancestor_focal_sites:
                start, end = self.ancestor_builder.make_ancestor(focal_sites, a)
                match_queue.put((child, focal_sites, start, end, a.copy()))
                child += 1
            # Wait until the match_queue is empty.
            # TODO Note that these calls to queue.join prevent errors that happen in the
            # worker process from propagating back here. Might be better to use some
            # other way of handling threading sync.
            match_queue.join()
            epoch_results = ResultBuffer.combine(results)
            self.tree_sequence_builder.update(
                len(ancestor_focal_sites), time,
                epoch_results.left, epoch_results.right, epoch_results.parent,
                epoch_results.child, epoch_results.site, epoch_results.node,
                epoch_results.derived_state)
            mean_memory = np.mean(matcher_memory)
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
        for j in range(self.num_threads):
            match_queue.put(None)
        for j in range(self.num_threads):
            match_threads[j].join()

    def __process_ancestors_single_threaded(self):
        a = np.zeros(self.num_sites, dtype=np.int8)
        matcher = self.ancestor_matcher_class(self.tree_sequence_builder, 0)
        results = ResultBuffer()
        match = np.zeros(self.num_sites, np.int8)

        for epoch in range(self.num_epochs):
            time = self.epoch_time[epoch]
            ancestor_focal_sites = self.epoch_ancestors[epoch]
            self.logger.debug("Epoch {}; time = {}; {} ancestors to process".format(
                epoch, time, len(ancestor_focal_sites)))
            child = self.tree_sequence_builder.num_nodes
            for focal_sites in ancestor_focal_sites:
                start, end = self.ancestor_builder.make_ancestor(focal_sites, a)
                self.__find_path_ancestor(
                    a, start, end, child, focal_sites, matcher, results, match)
                child += 1
            self.tree_sequence_builder.update(
                len(ancestor_focal_sites), time,
                results.left, results.right, results.parent, results.child,
                results.site, results.node, results.derived_state)
            results.clear()

    def __process_sample(self, sample_index, matcher, results, match):
        child = self.sample_ids[sample_index]
        haplotype = self.samples[sample_index]
        left, right, parent = matcher.find_path(haplotype, 0, self.num_sites, match)
        diffs = np.where(haplotype != match)[0]
        derived_state = haplotype[diffs]
        results.add_mutations(diffs, child, derived_state)
        results.add_edges(left, right, parent, child)
        self.__update_progress()

    def __process_samples_threads(self, sample_error):
        results = [ResultBuffer() for _ in range(self.num_threads)]
        matchers = [
            self.ancestor_matcher_class(self.tree_sequence_builder, sample_error)
            for _ in range(self.num_threads)]
        work_queue = queue.Queue()

        def worker(thread_index):
            with self.catch_thread_error():
                self.logger.info("Started sample worker thread {}".format(thread_index))
                if _prctl_available:
                    prctl.set_name("sample-worker-{}".format(thread_index))
                if _numa_available and numa.available():
                    numa.set_localalloc()
                    self.logger.debug(
                        "Set NUMA local allocation policy on thread {}.".format(
                            thread_index))
                mean_traceback_size = 0
                num_matches = 0
                match = np.zeros(self.num_sites, np.int8)
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
                self.tree_sequence_builder, sample_error)
            match = np.zeros(self.num_sites, np.int8)
            for j in range(self.num_samples):
                self.__process_sample(j, matcher, results, match)
        else:
            results = self.__process_samples_threads(sample_error)
        self.tree_sequence_builder.update(
            self.num_samples, 0,
            results.left, results.right, results.parent, results.child,
            results.site, results.node, results.derived_state)

    def finalise(self):
        self.logger.info("Finalising tree sequence")
        ts = self.get_tree_sequence()
        ts_simplified = ts.simplify(
            samples=self.sample_ids, filter_zero_mutation_sites=False)
        self.logger.debug("simplified from ({}, {}) to ({}, {}) nodes and edges".format(
            ts.num_nodes, ts.num_edges, ts_simplified.num_nodes,
            ts_simplified.num_edges))
        return ts_simplified

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
        tsb.dump_mutations(
            site=site, node=node, derived_state=derived_state, parent=parent)
        derived_state += ord('0')
        mutations.set_columns(
            site=site, node=node, derived_state=derived_state,
            derived_state_length=np.ones(tsb.num_mutations, dtype=np.uint32),
            parent=parent)
        msprime.sort_tables(nodes, edges, sites=sites, mutations=mutations)
        return msprime.load_tables(
            nodes=nodes, edges=edges, sites=sites, mutations=mutations,
            sequence_length=self.sequence_length)

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
        method="C", num_threads=1, progress=False, log_level="WARNING"):
    positions_array = np.array(positions)
    # If the input recombination rate is a single number set this value for all sites.
    recombination_rate_array = np.zeros(positions_array.shape[0], dtype=np.float64)
    recombination_rate_array[:] = recombination_rate
    # Primary entry point.
    manager = InferenceManager(
        samples, positions_array, sequence_length, recombination_rate_array,
        num_threads=num_threads, method=method, progress=progress, log_level=log_level)
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
    def __init__(self, num_samples, num_sites):
        self.num_samples = num_samples
        self.num_sites = num_sites
        self.sites = [None for _ in range(self.num_sites)]
        self.frequency_map = [{} for _ in range(self.num_samples)]

    def add_site(self, site_id, frequency, genotypes):
        """
        Adds a new site at the specified ID and allele pattern to the builder.
        """
        self.sites[site_id] = Site(site_id, frequency, genotypes)
        if frequency > 1:
            pattern_map = self.frequency_map[frequency]
            # Each unique pattern gets added to the list
            key = genotypes.tobytes()
            if key not in pattern_map:
                pattern_map[key] = []
            pattern_map[key].append(site_id)
        else:
            # Save some memory as we'll never look at these
            self.sites[site_id].genotypes = None

    def print_state(self):
        print("Ancestor builder")
        print("Sites = ")
        for j in range(self.num_sites):
            site = self.sites[j]
            print(site.frequency, "\t", site.genotypes)
        print("Frequency map")
        for f in range(self.num_samples):
            pattern_map = self.frequency_map[f]
            if len(pattern_map) > 0:
                print("f = ", f, "with ", len(pattern_map), "patterns")
                for pattern, sites in pattern_map.items():
                    print("\t", pattern, ":", sites)

    def ancestor_descriptors(self):
        """
        Returns a list of (frequency, focal_sites) tuples describing the
        ancestors in reverse order of frequency.
        """
        ret = []
        for frequency in reversed(range(self.num_samples)):
            for focal_sites in self.frequency_map[frequency].values():
                ret.append((frequency, focal_sites))
        return ret

    def __build_ancestor_sites(self, focal_site, sites, a):
        samples = set()
        g = self.sites[focal_site].genotypes
        for j in range(self.num_samples):
            if g[j] == 1:
                samples.add(j)
        for l in sites:
            a[l] = 0
            if self.sites[l].frequency > self.sites[focal_site].frequency:
                # print("\texamining:", self.sites[l])
                # print("\tsamples = ", samples)
                num_ones = 0
                num_zeros = 0
                for j in samples:
                    if self.sites[l].genotypes[j] == 1:
                        num_ones += 1
                    else:
                        num_zeros += 1
                # TODO choose a branch uniformly if we have equality.
                if num_ones >= num_zeros:
                    a[l] = 1
                    samples = set(j for j in samples if self.sites[l].genotypes[j] == 1)
                else:
                    samples = set(j for j in samples if self.sites[l].genotypes[j] == 0)
            if len(samples) == 1:
                # print("BREAK")
                break

    def make_ancestor(self, focal_sites, a):
        a[:] = -1
        focal_site = focal_sites[0]
        sites = range(focal_sites[-1] + 1, self.num_sites)
        self.__build_ancestor_sites(focal_site, sites, a)
        focal_site = focal_sites[-1]
        sites = range(focal_sites[0] - 1, -1, -1)
        self.__build_ancestor_sites(focal_site, sites, a)
        for j in range(focal_sites[0], focal_sites[-1] + 1):
            if j in focal_sites:
                a[j] = 1
            else:
                self.__build_ancestor_sites(focal_site, [j], a)
        known = np.where(a != -1)[0]
        start = known[0]
        end = known[-1] + 1
        return start, end


class TreeSequenceBuilder(object):

    def __init__(
            self, sequence_length, positions, recombination_rate,
            max_nodes, max_edges):
        self.num_nodes = 0
        self.sequence_length = sequence_length
        self.positions = positions
        self.recombination_rate = recombination_rate
        self.num_sites = positions.shape[0]
        self.time = []
        self.flags = []
        self.mutations = collections.defaultdict(list)
        self.edges = []
        self.mean_traceback_size = 0

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
        print("Removal order = ", self.removal_order)

        if nodes.num_rows > 1:
            msprime.sort_tables(nodes, edges)
            samples = np.where(nodes.flags == 1)[0].astype(np.int32)
            msprime.simplify_tables(samples, nodes, edges)
            print("edges = ")
            print(edges)

    def update(
            self, num_nodes, time, left, right, parent, child, site, node,
            derived_state):
        for _ in range(num_nodes):
            self.add_node(time)
        for l, r, p, c in zip(left, right, parent, child):
            self.edges.append(Edge(l, r, p, c))

        # print("update at time ", time, "num_edges = ", len(self.edges))
        for s, u, d in zip(site, node, derived_state):
            self.mutations[s].append((u, d))

        # Index the edges
        self.edges.sort(key=lambda e: (e.left, self.time[e.parent]))
        M = len(self.edges)
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
                if d == 0:
                    parent[j] = p
                j += 1


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
    if u < v:
        assert not ret
    # print("IS_DESCENDENT(", u, v, ") = ", ret)
    return ret


class AncestorMatcher(object):

    def __init__(self, tree_sequence_builder, error_rate=0):
        self.tree_sequence_builder = tree_sequence_builder
        self.error_rate = error_rate
        self.num_sites = tree_sequence_builder.num_sites
        self.positions = tree_sequence_builder.positions
        self.parent = None
        self.left_child = None
        self.right_sib = None
        self.traceback = None
        self.likelihood = None
        self.likelihood_nodes = None

    def get_max_likelihood_node(self):
        """
        Returns the node with the maxmimum likelihood from the specified map.
        """
        u = -1
        max_likelihood = -1
        for node in self.likelihood_nodes:
            likelihood = self.likelihood[node]
            if likelihood > max_likelihood:
                u = node
                max_likelihood = likelihood
        assert u != -1
        return u

    def get_max_likelihood_traceback_node(self, L):
        u = -1
        max_likelihood = -1
        for node, likelihood in L.items():
            if likelihood > max_likelihood:
                u = node
                max_likelihood = likelihood
        assert u != -1
        return u

    def check_likelihoods(self):
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

    def store_traceback(self, site):
        self.traceback[site] = {u: self.likelihood[u] for u in self.likelihood_nodes}
        # print("Stored traceback for ", site, self.traceback[site])

    def update_site(self, site, state):
        n = self.tree_sequence_builder.num_nodes
        recombination_rate = self.tree_sequence_builder.recombination_rate
        err = self.error_rate

        r = 1 - np.exp(-recombination_rate[site] / n)
        recomb_proba = r / n
        no_recomb_proba = 1 - r + r / n

        if site not in self.tree_sequence_builder.mutations:
            if err == 0:
                # This is a special case for ancestor matching. Very awkward
                # flow control here. FIXME
                if state == 0:
                    assert len(self.likelihood_nodes) > 0
                self.store_traceback(site)
                if site > 0 and (site - 1) \
                        not in self.tree_sequence_builder.mutations:
                    assert self.traceback[site] == self.traceback[site - 1]
                # NASTY!!!!
                return
            mutation_node = msprime.NULL_NODE
        else:
            mutation_node = self.tree_sequence_builder.mutations[site][0][0]
            # Insert an new L-value for the mutation node if needed.
            if self.likelihood[mutation_node] == -1:
                u = mutation_node
                while self.likelihood[u] == -1:
                    u = self.parent[u]
                self.likelihood[mutation_node] = self.likelihood[u]
                self.likelihood_nodes.append(mutation_node)

        # print("Site ", site, "mutation = ", mutation_node, "state = ", state)
        self.store_traceback(site)

        distance = 1
        if site > 0:
            distance = self.positions[site] - self.positions[site - 1]
        # Update the likelihoods for this site.
        # print("Site ", site, "distance = ", distance)
        max_L = -1
        # print("Computing likelihoods for ", mutation_node)
        path_cache = np.zeros(n, dtype=np.int8) - 1
        for u in self.likelihood_nodes:
            v = u
            while v != -1 and v != mutation_node and path_cache[v] == -1:
                v = self.parent[v]
            d = False
            if v != -1 and path_cache[v] != -1:
                d = path_cache[v]
            else:
                d = v == mutation_node
            assert d == is_descendant(self.parent, u, mutation_node)
            # Insert this path into the cache.
            v = u
            while v != -1 and v != mutation_node and path_cache[v] == -1:
                path_cache[v] = d
                v = self.parent[v]

            # TODO should we remove this parameter here and include it
            # in the recombination rate parameter??? In practise we'll
            # probably be working it out from a recombination map, so
            # there's no point in complicating this further by rescaling
            # it back into physical distance.
            x = self.likelihood[u] * no_recomb_proba * distance
            assert x >= 0
            y = recomb_proba * distance
            if x > y:
                z = x
            else:
                z = y
            if state == 1:
                emission_p = (1 - err) * d + err * (not d)
            else:
                emission_p = err * d + (1 - err) * (not d)
            # print("setting ", v, z, emission_p, mutation_node)
            self.likelihood[u] = z * emission_p
            if self.likelihood[u] > max_L:
                max_L = self.likelihood[u]
        assert max_L > 0

        # Normalise and reset the path cache
        for u in self.likelihood_nodes:
            self.likelihood[u] /= max_L
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

    def approximately_equal(self, a, b):
        # Based on Python is_close, https://www.python.org/dev/peps/pep-0485/
        rel_tol = 1e-9
        abs_tol = 0.0
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def approximately_one(self, a):
        return self.approximately_equal(a, 1.0)

    def find_path(self, h, start, end, match):

        M = len(self.tree_sequence_builder.edges)
        O = self.tree_sequence_builder.removal_order
        n = self.tree_sequence_builder.num_nodes
        m = self.tree_sequence_builder.num_sites
        edges = self.tree_sequence_builder.edges
        self.parent = np.zeros(n, dtype=int) - 1
        self.left_child = np.zeros(n, dtype=int) - 1
        self.right_child = np.zeros(n, dtype=int) - 1
        self.left_sib = np.zeros(n, dtype=int) - 1
        self.right_sib = np.zeros(n, dtype=int) - 1
        self.traceback = [{} for _ in range(m)]
        self.likelihood = np.zeros(n) - 2
        self.likelihood_nodes = []
        L_cache = np.zeros_like(self.likelihood) - 1

        # print("MATCH: start=", start, "end = ", end)
        j = 0
        k = 0
        left = 0
        pos = 0
        right = m
        while j < M and k < M and edges[j].left <= start:
            # print("top of init loop:", left, right)
            while edges[O[k]].right == pos:
                self.remove_edge(edges[O[k]])
                k += 1
            while j < M and edges[j].left == pos:
                self.insert_edge(edges[j])
                j += 1
            left = pos
            right = m
            if j < M:
                right = min(right, edges[j].left)
            if k < M:
                right = min(right, edges[O[k]].right)
            pos = right
        assert left < right

        self.likelihood_nodes.append(0)
        self.likelihood[0] = 1
        for u in range(n):
            if self.parent[u] != -1:
                self.likelihood[u] = -1

        remove_start = k
        while left < end:
            assert left < right

            # print("START OF TREE LOOP", left, right)
            normalisation_required = False
            for l in range(remove_start, k):
                edge = edges[O[l]]
                for u in [edge.parent, edge.child]:
                    if self.is_nonzero_root(u):
                        # print("REMOVING ROOT", edge.child, self.likelihood[edge.child])
                        if self.approximately_one(self.likelihood[u]):
                            normalisation_required = True
                        self.likelihood[u] = -2
                        if u in self.likelihood_nodes:
                            self.likelihood_nodes.remove(u)
            if normalisation_required:
                max_L = max(self.likelihood[u] for u in self.likelihood_nodes)
                for u in self.likelihood_nodes:
                    self.likelihood[u] /= max_L

            self.check_likelihoods()
            for site in range(max(left, start), min(right, end)):
                # print("UPDATE site", site)
                self.update_site(site, h[site])

            # print("UPDATE TREE", left, right)
            remove_start = k
            while k < M and edges[O[k]].right == right:
                edge = edges[O[k]]
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
                edge = edges[O[l]]
                u = edge.parent
                while L_cache[u] != -1:
                    L_cache[u] = -1
                    u = self.parent[u]
            assert np.all(L_cache == -1)

            left = right
            while j < M and edges[j].left == left:
                edge = edges[j]
                self.insert_edge(edge)
                j += 1
                # There's no point in compressing the likelihood tree here as we'll be
                # doing it after we update the first site anyway.
                for u in [edge.parent, edge.child]:
                    if self.likelihood[u] == -2:
                        self.likelihood[u] = 0
                        self.likelihood_nodes.append(u)
            right = m
            if j < M:
                right = min(right, edges[j].left)

            if k < M:
                right = min(right, edges[O[k]].right)

        # print("LIKELIHOODS")
        # for l in range(self.num_sites):
        #     print("\t", l, traceback[l])

        return self.run_traceback(start, end, match)

    def run_traceback(self, start, end, match):

        M = len(self.tree_sequence_builder.edges)
        edges = self.tree_sequence_builder.edges
        u = self.get_max_likelihood_node()
        output_edge = Edge(right=end, parent=u)
        output_edges = [output_edge]

        # Now go back through the trees.
        j = M - 1
        k = M - 1
        I = self.tree_sequence_builder.removal_order
        # Construct the matched haplotype
        match[:] = 0
        match[:start] = -1
        match[end:] = -1
        self.parent[:] = -1
        # print("TB: max_likelihood node = ", u)
        pos = self.tree_sequence_builder.num_sites
        while pos > start:
            # print("Top of loop: pos = ", pos)
            while k >= 0 and edges[k].left == pos:
                self.parent[edges[k].child] = -1
                k -= 1
            while j >= 0 and edges[I[j]].right == pos:
                self.parent[edges[I[j]].child] = edges[I[j]].parent
                j -= 1
            right = pos
            left = 0
            if k >= 0:
                left = max(left, edges[k].left)
            if j >= 0:
                left = max(left, edges[I[j]].right)
            pos = left
            # print("tree:", left, right, "j = ", j, "k = ", k)

            assert left < right
            for l in range(min(right, end) - 1, max(left, start) - 1, -1):
                u = output_edge.parent
                if l in self.tree_sequence_builder.mutations:
                    if is_descendant(
                            self.parent, u,
                            self.tree_sequence_builder.mutations[l][0][0]):
                        match[l] = 1
                L = self.traceback[l]
                # print("TB", l, L)
                v = u
                assert len(L) > 0
                while v not in L:
                    v = self.parent[v]
                    assert v != -1
                x = L[v]
                if not self.approximately_one(x):
                    output_edge.left = l
                    u = self.get_max_likelihood_traceback_node(L)
                    output_edge = Edge(right=l, parent=u)
                    output_edges.append(output_edge)
        output_edge.left = start

        self.mean_traceback_size = sum(len(t) for t in self.traceback) / self.num_sites
        # print("mathc h = ", match)

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
