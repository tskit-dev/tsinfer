# TODO copyright and license.
"""
Manage tsinfer's various HDF5 file formats.
"""
import contextlib
import uuid
import logging
import time
import threading
import queue

import numpy as np
import h5py

logger = logging.getLogger(__name__)

# TODO move this into an optional features module that we can pull in
# from various places.

# prctl is an optional extra; it allows us assign meaninful names to threads
# for debugging.
_prctl_available = False
try:
    import prctl
    _prctl_available = True
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
        variants_group = self.hdf5_file["variants"]
        self.genotypes = variants_group["genotypes"]
        self.position = variants_group["position"]
        self.recombination_rate = variants_group["recombination_rate"]
        self.num_sites = self.genotypes.shape[0]
        self.num_samples = self.genotypes.shape[1]
        # TODO check dimensions.

    def site_genotypes(self):
        """
        Returns an iterator over the genotypes in site-by-site order.
        """
        V = self.genotypes[:]
        for v in V:
            yield v

    @classmethod
    def build(
            cls, input_hdf5, genotypes, sequence_length=None, position=None,
            recombination_rate=None, chunk_size=None, compression=None):
        """
        Builds a tsinfer InputFile from the specified data.
        """
        num_sites, num_samples = genotypes.shape
        if position is None:
            position = np.arange(num_sites)
        if sequence_length is None:
            sequence_length = position[-1] + 1
        if recombination_rate is None:
            recombination_rate = 1
        if chunk_size is None:
            chunk_size = 8 * 1024  # By default chunk in 64MiB squares.

        cls.write_version_attrs(input_hdf5)
        input_hdf5.attrs["sequence_length"] = sequence_length
        input_hdf5.attrs["uuid"] = str(uuid.uuid4())

        variants_group = input_hdf5.create_group("variants")
        variants_group.create_dataset(
            "position", (num_sites,), data=position, dtype=np.float64,
            compression=compression)
        variants_group.create_dataset(
            "recombination_rate", (num_sites,), data=recombination_rate, dtype=np.float64,
            compression=compression)
        variants_group.create_dataset(
            "genotypes", (num_sites, num_samples), data=genotypes,
            chunks=(min(chunk_size, num_sites), min(chunk_size, num_samples)),
            dtype=np.uint8, compression=compression)



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
            print("read mode")
        elif mode == self.MODE_WRITE:
            pass
        else:
            raise ValueError("open mode must be 'r' or 'w'")
        self.mode = mode

    def write_uuid(self):
        self.hdf5_file.attrs["input-uuid"] = self.input_file.uuid

    def initialise(
           self, num_ancestors, oldest_time, total_num_focal_sites,
           chunk_size=None, compression=None):
        if self.mode != self.MODE_WRITE:
           raise ValueError("Must open in write mode")
        self.write_version_attrs(self.hdf5_file)
        self.write_uuid()
        self.num_ancestors = num_ancestors
        if chunk_size is None:
            chunk_size = 1024  # Default to 1K ancestors per chunk.
        self.chunk_size = min(chunk_size, num_ancestors)
        if self.chunk_size < 1:
            raise ValueError("Chunk size must be >= 1")

        # Create the datasets.
        ancestors_group = self.hdf5_file.create_group("ancestors")
        self.haplotypes = ancestors_group.create_dataset(
            "haplotypes", (num_ancestors, self.num_sites), dtype=np.int8,
            chunks=(self.chunk_size, self.num_sites), compression=compression)
        self.time = ancestors_group.create_dataset(
            "time", (num_ancestors,), compression=compression, dtype=np.int32)
        self.start = ancestors_group.create_dataset(
            "start", (num_ancestors,), compression=compression, dtype=np.int32)
        self.end = ancestors_group.create_dataset(
            "end", (num_ancestors,), compression=compression, dtype=np.int32)
        self.num_focal_sites = ancestors_group.create_dataset(
            "num_focal_sites", (num_ancestors,), compression=compression, dtype=np.int32)
        self.focal_sites = ancestors_group.create_dataset(
            "focal_sites", (total_num_focal_sites,), compression=compression,
            dtype=np.int32)

        # Create the data buffers.
        self.__haplotypes_buffer = np.empty((self.chunk_size, self.num_sites))
        self.__haplotypes_buffer_lock = threading.Lock()
        self.__time_buffer = np.zeros(num_ancestors, dtype=np.int32)
        self.__start_buffer = np.zeros(num_ancestors, dtype=np.int32)
        self.__end_buffer = np.zeros(num_ancestors, dtype=np.int32)
        self.__num_focal_sites_buffer = np.zeros(num_ancestors, dtype=np.uint32)
        self.__focal_sites_buffer = np.zeros(total_num_focal_sites, dtype=np.int32)
        self.__focal_sites_offset = 0

        # Fill in the oldest ancestor.
        self.__time_buffer[0] = oldest_time
        self.__haplotypes_buffer[0, :] = 0
        self.__ancestor_id = 1
        # Start the flush thread.
        self.__flush_thread = threading.Thread(target=self.flush_worker, daemon=True)
        self.__flush_queue = queue.Queue(16)  # Arbitrary.
        logger.debug("initialised ancestor file for input {}".format(
            self.input_file.uuid))
        self.__flush_thread.start()

    def flush_worker(self):
        logger.info("Ancestor flush worker thread starting")
        if _prctl_available:
            prctl.set_name("ancestor-flush-worker")
        while True:
            work = self.__flush_queue.get()
            if work is None:
                break
            start, end, H = work
            before = time.perf_counter()
            self.haplotypes[start:end,:] = H
            duration = time.perf_counter() - before
            logger.info("Flushed genotype chunk in {:.2f}s".format(duration))
            self.__flush_queue.task_done()
        self.__flush_queue.task_done()
        logger.info("Ancestor flush worker thread exiting")

    def add_ancestor(
            self, start=None, end=None, ancestor_time=None, focal_sites=None,
            haplotype=None):
        """
        Inserts the specified ancestor into the file. This must be called sequentially
        for each ancestor in turn.
        """
        j = self.__ancestor_id
        self.__haplotypes_buffer[j % self.chunk_size] = haplotype
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
        if j % self.chunk_size == 0:
            start = j - self.chunk_size
            end = j
            logger.info("Pushing chunk {} onto queue, depth={}".format(
                j // self.chunk_size, self.__flush_queue.qsize()))
            self.__flush_queue.put((start, end, self.__haplotypes_buffer.copy()))

    def finalise(self):
        """
        Finalises the file by writing the buffers out to the datasets.
        """
        assert self.num_ancestors == self.__ancestor_id
        last_chunk = self.num_ancestors % self.chunk_size
        if last_chunk != 0:
            start = -last_chunk
            end = -1
            self.__flush_queue.put(
                (start, end, self.__haplotypes_buffer[:last_chunk]))
        self.__flush_queue.put(None)
        self.__flush_queue.join()
        self.time[:] = self.__time_buffer
        self.start[:] = self.__start_buffer
        self.end[:] = self.__end_buffer
        self.num_focal_sites[:] = self.__num_focal_sites_buffer
        self.focal_sites[:] = self.__focal_sites_buffer
        self.__flush_thread.join()
