# TODO copyright and license.
"""
Manage tsinfer's various HDF5 file formats.
"""
import contextlib
import uuid
import logging
import time
import queue

import numpy as np
import h5py
import zarr
import numcodecs.blosc as blosc

import tsinfer.threads as threads

# We don't want blosc to spin up extra threads for compression.
blosc.use_threads = False
logger = logging.getLogger(__name__)


FORMAT_NAME_KEY = "format_name"
FORMAT_VERSION_KEY = "format_version"

DEFAULT_COMPRESSOR = blosc.Blosc(cname='zstd', clevel=9, shuffle=blosc.BITSHUFFLE)


def threaded_row_iterator(array, start=0, queue_size=2):
    """
    Returns an iterator over the rows in the specified 2D array of
    genotypes.
    """
    chunk_size = array.chunks[0]
    num_rows = array.shape[0]
    num_chunks = num_rows // chunk_size
    logger.info("Loading genotypes for {} columns in {} chunks; size={}".format(
        num_rows, num_chunks, array.chunks))

    # Note: got rid of the threaded version of this because it was causing a
    # memory leak. Probably we're better off using a simple double buffered
    # approach here rather than using queues.

    j = 0
    for chunk in range(num_chunks):
        if j + chunk_size >= start:
            before = time.perf_counter()
            A = array[j: j + chunk_size]
            duration = time.perf_counter() - before
            logger.debug("Loaded {:.2f}MiB chunk start={} in {:.2f} seconds".format(
                A.nbytes / 1024**2, j, duration))
            for index in range(chunk_size):
                if j + index >= start:
                    # Yielding a copy here because we end up keeping a second copy
                    # of the matrix when we threads accessing it. Probably not an
                    # issue if we use a simple double buffer though.
                    yield A[index][:]
        else:
            logger.debug("Skipping genotype chunk {}".format(j))
        j += chunk_size
    # TODO this isn't correctly checking for start.
    last_chunk = num_rows % chunk_size
    if last_chunk != 0:
        before = time.perf_counter()
        A = array[-last_chunk:]
        duration = time.perf_counter() - before
        logger.debug(
            "Loaded final genotype chunk in {:.2f} seconds".format(duration))
        for row in A:
            yield row


    # chunk_size = array.chunks[0]
    # num_rows = array.shape[0]
    # num_chunks = num_rows // chunk_size
    # logger.info("Loading genotypes for {} columns in {} chunks; size={}".format(
    #     num_rows, num_chunks, array.chunks))
    # decompressed_queue = queue.Queue(queue_size)

    # def decompression_worker(thread_index):
    #     j = 0
    #     for chunk in range(num_chunks):
    #         if j + chunk_size >= start:
    #             before = time.perf_counter()
    #             A = array[j: j + chunk_size]
    #             duration = time.perf_counter() - before
    #             logger.debug("Loaded {:.2f}MiB chunk start={} in {:.2f} seconds".format(
    #                 A.nbytes / 1024**2, j, duration))
    #             decompressed_queue.put((j, A))
    #         else:
    #             logger.debug("Skipping genotype chunk {}".format(j))
    #         j += chunk_size
    #     last_chunk = num_rows % chunk_size
    #     if last_chunk != 0:
    #         before = time.perf_counter()
    #         A = array[-last_chunk:]
    #         duration = time.perf_counter() - before
    #         logger.debug(
    #             "Loaded final genotype chunk in {:.2f} seconds".format(duration))
    #         decompressed_queue.put((num_rows - last_chunk, A))
    #     decompressed_queue.put(None)

    # decompression_thread = threads.queue_producer_thread(
    #     decompression_worker, decompressed_queue, name="genotype-decompression")

    # while True:
    #     chunk = decompressed_queue.get()
    #     if chunk is None:
    #         break
    #     logger.debug("Got genotype chunk from queue (depth={})".format(
    #         decompressed_queue.qsize()))
    #     start_index, rows = chunk
    #     for index, row in enumerate(rows, start_index):
    #         if index >= start:
    #             yield row
    #     decompressed_queue.task_done()
    # decompression_thread.join()


def transposed_threaded_row_iterator(array, queue_size=4):
    """
    Returns an iterator over the transposed columns in the specified 2D array of
    genotypes.
    """
    chunk_size = array.chunks[1]
    num_cols = array.shape[1]
    num_chunks = num_cols // chunk_size
    logger.info("Loading genotypes for {} columns in {} chunks {}".format(
        num_cols, num_chunks, array.chunks))
    decompressed_queue = queue.Queue(queue_size)

    # NOTE Get rid of this; see notes about memory leak above.

    def decompression_worker(thread_index):
        j = 0
        for chunk in range(num_chunks):
            before = time.perf_counter()
            A = array[:, j: j + chunk_size][:].T
            duration = time.perf_counter() - before
            logger.debug("Loaded genotype chunk in {:.2f} seconds".format(duration))
            decompressed_queue.put(A)
            j += chunk_size
        last_chunk = num_cols % chunk_size
        if last_chunk != 0:
            before = time.perf_counter()
            A = array[:, -last_chunk:][:].T
            duration = time.perf_counter() - before
            logger.debug("Loaded final genotype chunk in {:.2f} seconds".format(
                duration))
            decompressed_queue.put(A)
        decompressed_queue.put(None)

    decompression_thread = threads.queue_producer_thread(
        decompression_worker, decompressed_queue, name="genotype-decompression")

    while True:
        chunk = decompressed_queue.get()
        if chunk is None:
            break
        logger.debug("Got genotype chunk shape={} from queue (depth={})".format(
            chunk.shape, decompressed_queue.qsize()))
        for row in chunk:
            yield row
        decompressed_queue.task_done()
    decompression_thread.join()


@contextlib.contextmanager
def open_input(path):
    """
    Loads an input HDF5 file from the specified path and returns the corresponding
    h5py file.
    """
    logger.info("Opening input file {}".format(path))
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
    logger.info("Opening ancestor file {}".format(path))
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


# TODO change this to Container.
class Hdf5File(object):
    """
    Superclass of all input and intermediate formats used by tsinfer.
    """
    # Must be defined by subclasses.
    format_name = None
    format_version = None

    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def check_format(self):
        format_name = self.hdf5_file.attrs[FORMAT_NAME_KEY]
        format_version = self.hdf5_file.attrs[FORMAT_VERSION_KEY]
        if format_name != self.format_name:
            raise ValueError("Incorrect file format: expected '{}' got '{}'".format(
                self.format_version, format_version))
        if format_version[0] < self.format_version[0]:
            raise ValueError("Format version {} too old. Current version = {}".format(
                format_version, self.format_version))
        if format_version[0] > self.format_version[0]:
            raise ValueError("Format version {} too new. Current version = {}".format(
                format_version, self.format_version))

    @classmethod
    def write_version_attrs(cls, hdf5_file):
        """
        Writes the version attributes for the specified HDF5 file.
        """
        hdf5_file.attrs[FORMAT_NAME_KEY] = cls.format_name
        hdf5_file.attrs[FORMAT_VERSION_KEY] = cls.format_version


class InputFile(Hdf5File):
    """
    The input file for a tsinfer inference session. Stores variant data for a
    set of samples using HDF5.
    """
    format_name = "tsinfer-input"
    format_version = (0, 1)

    def __init__(self, hdf5_file):
        super().__init__(hdf5_file)
        self.check_format()
        self.uuid = self.hdf5_file.attrs["uuid"]
        self.sequence_length = self.hdf5_file.attrs["sequence_length"]
        variants_group = self.hdf5_file["variants"]
        self.genotypes = variants_group["genotypes"]
        self.position = variants_group["position"][:]
        self.recombination_rate = variants_group["recombination_rate"][:]
        self.num_sites = self.genotypes.shape[0]
        self.num_samples = self.genotypes.shape[1]
        logger.info("Loaded input uuid={}; samples={} sites={}".format(
            self.uuid, self.num_samples, self.num_sites))
        # TODO check dimensions.

    def site_genotypes(self):
        """
        Returns an iterator over the genotypes in site-by-site order.
        """
        return threaded_row_iterator(self.genotypes)

    def sample_haplotypes(self):
        """
        Returns an iterator over the sample haplotypes, i.e., the genotype matrix
        in column-by-column order.
        """
        return transposed_threaded_row_iterator(self.genotypes)

    @classmethod
    def build(
            cls, input_hdf5, genotypes, genotype_qualities=None,
            sequence_length=None, position=None,
            recombination_rate=None, chunk_size=None, compress=True):
        """
        Builds a tsinfer InputFile from the specified data.
        """
        num_sites, num_samples = genotypes.shape
        if position is None:
            position = np.arange(num_sites)
        if genotype_qualities is None:
            genotype_qualities = 100 + np.zeros((num_sites, num_samples), dtype=np.uint8)
        if sequence_length is None:
            sequence_length = position[-1] + 1
        if recombination_rate is None:
            recombination_rate = 1
        if chunk_size is None:
            chunk_size = 8 * 1024  # By default chunk in 64MiB squares.

        position = np.array(position)
        if np.any(position[1:] == position[:-1]):
            raise ValueError("All positions must be unique")

        # If the input recombination rate is a single number set this value for all sites.
        recombination_rate_array = np.zeros(position.shape[0], dtype=np.float64)
        recombination_rate_array[:] = recombination_rate

        cls.write_version_attrs(input_hdf5)
        input_hdf5.attrs["sequence_length"] = float(sequence_length)
        input_hdf5.attrs["uuid"] = str(uuid.uuid4())

        compressor = None
        if compress:
            compressor = DEFAULT_COMPRESSOR
        variants_group = input_hdf5.create_group("variants")
        variants_group.create_dataset(
            "position", shape=(num_sites,), data=position, dtype=np.float64,
            compressor=compressor)
        variants_group.create_dataset(
            "recombination_rate", shape=(num_sites,), data=recombination_rate_array,
            dtype=np.float64, compressor=compressor)
        x_chunk = min(chunk_size, num_sites)
        y_chunk = min(chunk_size, num_samples)
        variants_group.create_dataset(
            "genotypes", shape=(num_sites, num_samples), data=genotypes,
            chunks=(x_chunk, y_chunk), dtype=np.uint8, compressor=compressor)
        variants_group.create_dataset(
            "genotype_qualities", shape=(num_sites, num_samples),
            data=genotype_qualities,
            chunks=(min(chunk_size, num_sites), min(chunk_size, num_samples)),
            dtype=np.uint8, compressor=compressor)



class AncestorFile(Hdf5File):
    """
    The intermediate file representing the ancestors that we generate from a
    given input. This is produced by the 'build-ancestors' command and
    used by the 'match-ancestors' command.
    """
    format_name = "tsinfer-ancestors"
    format_version = (0, 1)

    MODE_READ = 'r'
    MODE_WRITE = 'w'

    def __init__(self, hdf5_file, input_file, mode):
        super().__init__(hdf5_file)
        self.input_file = input_file
        self.num_sites = input_file.num_sites
        self.num_samples = input_file.num_samples
        if mode == self.MODE_READ:
            self.check_format()
            self.read_attributes()
        elif mode == self.MODE_WRITE:
            pass
        else:
            raise ValueError("open mode must be 'r' or 'w'")
        self.mode = mode

    #############################
    # Read mode
    #############################

    def read_attributes(self):
        """
        Initialisation for read mode.
        """
        ancestors_group = self.hdf5_file["ancestors"]
        self.haplotypes = ancestors_group["haplotypes"]

        self.num_ancestors = self.haplotypes.shape[0]
        # Take copies of all the small data sets
        self.time = ancestors_group["time"][:]
        self.start = ancestors_group["start"][:]
        self.end = ancestors_group["end"][:]
        self.num_focal_sites = ancestors_group["num_focal_sites"][:]
        # Unflatten the focal sites array
        focal_sites = ancestors_group["focal_sites"][:]
        self.focal_sites = [None for _ in range(self.num_ancestors)]
        offset = 0
        for j in range(self.num_ancestors):
            self.focal_sites[j] = focal_sites[offset: offset + self.num_focal_sites[j]]
            offset += self.num_focal_sites[j]
            if j > 0:
                assert self.focal_sites[j].shape[0] > 0
            else:
                assert self.focal_sites[j].shape[0] == 0
        logger.info("Loaded ancestor matrix; num_ancestors={}".format(
            self.num_ancestors))

    def ancestor_haplotypes(self, start=0):
        """
        Returns an iterator over the ancestor haplotypes.
        """
        return threaded_row_iterator(self.haplotypes, start)

    def site_genotypes(self):
        return transposed_threaded_row_iterator(self.haplotypes)

    #############################
    # Write mode
    #############################

    def write_uuid(self):
        self.hdf5_file.attrs["input-uuid"] = self.input_file.uuid

    def initialise(
            self, num_ancestors, oldest_time, total_num_focal_sites,
            chunk_size=None, compress=True, num_threads=None):
        if self.mode != self.MODE_WRITE:
            raise ValueError("Must open in write mode")
        self.write_version_attrs(self.hdf5_file)
        self.write_uuid()
        self.num_ancestors = num_ancestors
        if chunk_size is None:
            chunk_size = 1024  # Default to 1K ancestors per chunk.
        if chunk_size < 1:
            raise ValueError("Chunk size must be >= 1")
        if num_threads is None:
            num_threads = 4  # Different default?
        num_threads = max(1, num_threads)

        # Create the datasets.
        compressor = None
        if compress:
            compressor = DEFAULT_COMPRESSOR

        x_chunk = min(chunk_size, num_ancestors)
        y_chunk = min(chunk_size, self.num_sites)

        ancestors_group = self.hdf5_file.create_group("ancestors")
        self.haplotypes = ancestors_group.create_dataset(
            "haplotypes", shape=(num_ancestors, self.num_sites), dtype=np.uint8,
            chunks=(x_chunk, y_chunk), compressor=compressor)
        self.time = ancestors_group.create_dataset(
            "time", shape=(num_ancestors,), compressor=compressor, dtype=np.int32)
        self.start = ancestors_group.create_dataset(
            "start", shape=(num_ancestors,), compressor=compressor, dtype=np.int32)
        self.end = ancestors_group.create_dataset(
            "end", shape=(num_ancestors,), compressor=compressor, dtype=np.int32)
        self.num_focal_sites = ancestors_group.create_dataset(
            "num_focal_sites", shape=(num_ancestors,), compressor=compressor,
            dtype=np.int32)
        self.focal_sites = ancestors_group.create_dataset(
            "focal_sites", shape=(total_num_focal_sites,), compressor=compressor,
            dtype=np.int32)

        # Create the data buffers.
        self.__time_buffer = np.zeros(num_ancestors, dtype=np.int32)
        self.__start_buffer = np.zeros(num_ancestors, dtype=np.int32)
        self.__end_buffer = np.zeros(num_ancestors, dtype=np.int32)
        self.__num_focal_sites_buffer = np.zeros(num_ancestors, dtype=np.uint32)
        self.__focal_sites_buffer = np.zeros(total_num_focal_sites, dtype=np.int32)
        self.__focal_sites_offset = 0

        # The haplotypes buffer is more complicated. We allocate an array of
        # num_buffers of these and update the current buffer pointed to by
        # the __haplotypes_buffer_index. These buffers are consumed by the
        # main thread and pushed back onto a queue by the worker threads.
        num_buffers = num_threads * 2
        self.__haplotypes_buffers = [
            np.empty((x_chunk, self.num_sites), dtype=np.uint8)
            for _ in range(num_buffers)]
        self.__haplotypes_buffer_index = 0
        logger.debug("Alloced {} buffers using {}MB each".format(
            num_buffers, self.__haplotypes_buffers[0].nbytes // 1024**2))
        # We consume the zero'th index first. Now push the remaining buffers
        # onto the buffer queue.
        self.__buffer_queue = queue.Queue()
        for j in range(num_buffers):
            self.__buffer_queue.put(j)
        self.__haplotypes_buffer_index = self.__buffer_queue.get()
        # Fill in the oldest ancestor.
        self.__end_buffer[0] = self.num_sites
        self.__time_buffer[0] = oldest_time
        self.__haplotypes_buffers[self.__haplotypes_buffer_index][0, :] = 0
        self.__ancestor_id = 1
        # Start the flush thread.
        self.__flush_queue = queue.Queue(num_buffers)
        self.__flush_threads = [
            threads.queue_consumer_thread(
                self.flush_worker, self.__flush_queue, name="flush-worker-{}".format(j),
                index=j)
            for j in range(num_threads)]
        logger.info("initialised ancestor file for input {}".format(
            self.input_file.uuid))

    def flush_worker(self, thread_index):
        while True:
            work = self.__flush_queue.get()
            if work is None:
                break
            start, end, buffer_index = work
            H = self.__haplotypes_buffers[buffer_index]
            before = time.perf_counter()
            self.haplotypes[start:end, :] = H
            duration = time.perf_counter() - before
            logger.debug("Flushed genotype chunk in {:.2f}s".format(duration))
            self.__flush_queue.task_done()
            self.__buffer_queue.put(buffer_index)
        self.__flush_queue.task_done()

    def add_ancestor(
            self, start=None, end=None, ancestor_time=None, focal_sites=None,
            haplotype=None):
        """
        Inserts the specified ancestor into the file. This must be called sequentially
        for each ancestor in turn.
        """
        j = self.__ancestor_id
        H = self.__haplotypes_buffers[self.__haplotypes_buffer_index]
        chunk_size = H.shape[0]
        H[j % chunk_size] = haplotype
        self.__time_buffer[j] = ancestor_time
        self.__start_buffer[j] = start
        self.__end_buffer[j] = end
        num_focal_sites = focal_sites.shape[0]
        self.__num_focal_sites_buffer[j] = num_focal_sites
        k = self.__focal_sites_offset
        self.__focal_sites_buffer[k: k + num_focal_sites] = focal_sites
        self.__focal_sites_offset += num_focal_sites
        self.__ancestor_id += 1
        j = self.__ancestor_id
        if j % chunk_size == 0:
            start = j - chunk_size
            end = j
            logger.debug("Pushing chunk {} onto queue, depth={}".format(
                j // chunk_size, self.__flush_queue.qsize()))
            self.__buffer_queue.task_done()
            self.__flush_queue.put((start, end, self.__haplotypes_buffer_index))
            self.__haplotypes_buffer_index = self.__buffer_queue.get()
            logger.debug("Got new haplotype buffer {}".format(
                self.__haplotypes_buffer_index))

    def finalise(self):
        """
        Finalises the file by writing the buffers out to the datasets.
        """
        H = self.__haplotypes_buffers[0]
        chunk_size = H.shape[0]
        assert self.num_ancestors == self.__ancestor_id
        last_chunk = self.num_ancestors % chunk_size
        if last_chunk != 0:
            H = self.__haplotypes_buffers[self.__haplotypes_buffer_index][:last_chunk]
            logger.debug("Flushing final chunk of size {}".format(last_chunk))
            self.haplotypes[-last_chunk:] = H
        for _ in self.__flush_threads:
            self.__flush_queue.put(None)
        self.__flush_queue.join()
        self.time[:] = self.__time_buffer
        self.start[:] = self.__start_buffer
        self.end[:] = self.__end_buffer
        self.num_focal_sites[:] = self.__num_focal_sites_buffer
        self.focal_sites[:] = self.__focal_sites_buffer
        for thread in self.__flush_threads:
            thread.join()
