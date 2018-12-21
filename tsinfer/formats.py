#
# Copyright (C) 2018 University of Oxford
#
# This file is part of tsinfer.
#
# tsinfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tsinfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Manage tsinfer's various HDF5 file formats.
"""
import collections.abc as abc
import datetime
import itertools
import logging
import os
import os.path
import queue
import threading
import uuid
import json

import numpy as np
import zarr
import lmdb
import humanize
import numcodecs
import msprime
import attr

import tsinfer.threads as threads
import tsinfer.provenance as provenance
import tsinfer.exceptions as exceptions


# FIXME need some global place to keep these constants
UNKNOWN_ALLELE = 255

logger = logging.getLogger(__name__)


FORMAT_NAME_KEY = "format_name"
FORMAT_VERSION_KEY = "format_version"
FINALISED_KEY = "finalised"

# We use the zstd compressor because it allows for compression of buffers
# bigger than 2GB, which can occur in a larger instances.
DEFAULT_COMPRESSOR = numcodecs.Zstd()


def remove_lmdb_lockfile(lmdb_file):
    lockfile = lmdb_file + "-lock"
    if os.path.exists(lockfile):
        os.unlink(lockfile)


class BufferedItemWriter(object):
    """
    Class that writes items sequentially into a set of zarr arrays,
    buffering writes and flushing them to the destination arrays
    asynchronosly using threads.
    """
    def __init__(self, array_map, num_threads=0):
        self.chunk_size = -1
        for key, array in array_map.items():
            if self.chunk_size == -1:
                self.chunk_size = array.chunks[0]
            else:
                if array.chunks[0] != self.chunk_size:
                    raise ValueError("Chunk sizes must be equal")
        self.arrays = array_map
        if num_threads <= 0:
            # Use a syncronous algorithm.
            self.num_threads = 0
            self.num_buffers = 1
        else:
            # One buffer for each thread. Buffers are referred to by their indexes.
            self.num_buffers = num_threads
            self.num_threads = num_threads
        self.buffers = {}
        self.current_size = 0
        self.total_items = 0
        for key, array in self.arrays.items():
            self.buffers[key] = [None for _ in range(self.num_buffers)]
            np_array = array[:]
            shape = list(array.shape)
            shape[0] = self.chunk_size
            for j in range(self.num_buffers):
                self.buffers[key][j] = np.empty_like(np_array)
                self.buffers[key][j].resize(*shape)
            # Make sure the destination array is zero sized at the start.
            shape[0] = 0
            array.resize(*shape)

        self.start_offset = [0 for _ in range(self.num_buffers)]
        self.num_buffered_items = [0 for _ in range(self.num_buffers)]
        self.write_buffer = 0
        # This lock must be held when resizing the underlying arrays.
        # This is no-op when using a single-threaded algorithm, but it's
        # not worth removing and complicating the logic.
        self.resize_lock = threading.Lock()
        if self.num_threads > 0:
            # Buffer indexes are placed in the queues. The current write buffer
            # is obtained from the write_queue. Flush worker threads pull buffer
            # indexes from the flush queue, and push them back on to the write
            # queue when the buffer has been flushed.
            self.write_queue = queue.Queue()
            self.flush_queue = queue.Queue()
            # The initial write buffer is 0; place the others on the queue.
            for j in range(1, self.num_buffers):
                self.write_queue.put(j)
            # Make the flush threads.
            self.flush_threads = [
                threads.queue_consumer_thread(
                    self._flush_worker, self.flush_queue,
                    name="flush-worker-{}".format(j))
                for j in range(self.num_threads)]
            logger.info("Started {} flush worker threads".format(self.num_threads))

    def _commit_write_buffer(self, write_buffer):
        start = self.start_offset[write_buffer]
        n = self.num_buffered_items[write_buffer]
        end = start + n
        logger.debug("Flushing buffer {}: start={} n={}".format(write_buffer, start, n))
        with self.resize_lock:
            if self.current_size < end:
                self.current_size = end
                for key, array in self.arrays.items():
                    shape = list(array.shape)
                    shape[0] = self.current_size
                    array.resize(*shape)
        for key, array in self.arrays.items():
            buffered = self.buffers[key][write_buffer][:n]
            array[start: end] = buffered
        logger.debug("Buffer {} flush done".format(write_buffer))

    def _flush_worker(self, thread_index):
        """
        Thread worker responsible for flushing buffers. Read a buffer index
        from flush_queue and write it to disk. Push the index back on
        to the write queue to allow it be reused.
        """
        while True:
            buffer_index = self.flush_queue.get()
            if buffer_index is None:
                break
            self._commit_write_buffer(buffer_index)
            self.flush_queue.task_done()
            self.write_queue.put(buffer_index)
        self.flush_queue.task_done()

    def _queue_flush_buffer(self):
        """
        Flushes the buffered ancestors to the data file.
        """
        if self.num_threads > 0:
            logger.debug("Pushing buffer {} to flush queue".format(self.write_buffer))
            self.flush_queue.put(self.write_buffer)
            self.write_buffer = self.write_queue.get()
        else:
            logger.debug("Syncronously flushing buffer")
            self._commit_write_buffer(self.write_buffer)
        self.num_buffered_items[self.write_buffer] = 0
        self.start_offset[self.write_buffer] = self.total_items

    def add(self, **kwargs):
        """
        Add an item to each of the arrays. The keyword arguments for this
        function correspond to the keys in the dictionary of arrays provided
        to the contructor.
        """
        if self.num_buffered_items[self.write_buffer] == self.chunk_size:
            self._queue_flush_buffer()
        offset = self.num_buffered_items[self.write_buffer]
        for key, value in kwargs.items():
            self.buffers[key][self.write_buffer][offset] = value
        self.num_buffered_items[self.write_buffer] += 1
        self.total_items += 1
        return self.total_items - 1

    def flush(self):
        """
        Flush the remaining items to the destination arrays and return all
        items are safely commited.

        It is an error to call ``add`` after ``flush`` has been called.
        """
        self._queue_flush_buffer()
        # Stop the the worker threads.
        for j in range(self.num_threads):
            self.flush_queue.put(None)
        for j in range(self.num_threads):
            self.flush_threads[j].join()
        self.buffers = None


def zarr_summary(array):
    """
    Returns a string with a brief summary of the specified zarr array.
    """
    dtype = str(array.dtype)
    ret = "shape={}; dtype={};".format(array.shape, dtype)
    if dtype != "object":
        # nbytes doesn't work correctly for object arrays.
        ret += "uncompressed size={}".format(humanize.naturalsize(array.nbytes))
    return ret


def chunk_iterator(array):
    """
    Utility to iterate over the rows in the specified array efficiently
    by accessing one chunk at a time.
    """
    chunk_size = array.chunks[0]
    for j in range(array.shape[0]):
        if j % chunk_size == 0:
            chunk = array[j: j + chunk_size][:]
        yield chunk[j % chunk_size]


class DataContainer(object):
    """
    Superclass of objects used to represent a collection of related
    data. Each datacontainer in a wrapper around a zarr group.
    """
    READ_MODE = 0
    BUILD_MODE = 1
    EDIT_MODE = 2

    # Must be defined by subclasses.
    FORMAT_NAME = None
    FORMAT_VERSION = None

    def __init__(
            self, path=None, num_flush_threads=0, compressor=None, chunk_size=1024):
        self._mode = self.BUILD_MODE
        if path is not None and compressor is None:
            compressor = DEFAULT_COMPRESSOR
        self._num_flush_threads = num_flush_threads
        self._chunk_size = max(1, chunk_size)
        self._metadata_codec = numcodecs.JSON()
        self._compressor = compressor
        self.data = zarr.group()
        self.path = path
        if path is not None:
            store = self._new_lmdb_store()
            self.data = zarr.open_group(store=store, mode="w")
        self.data.attrs[FORMAT_NAME_KEY] = self.FORMAT_NAME
        self.data.attrs[FORMAT_VERSION_KEY] = self.FORMAT_VERSION
        self.data.attrs["uuid"] = str(uuid.uuid4())

        chunks = self._chunk_size
        provenances_group = self.data.create_group("provenances")
        provenances_group.create_dataset(
            "timestamp", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=object, object_codec=self._metadata_codec)
        provenances_group.create_dataset(
            "record", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=object, object_codec=self._metadata_codec)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._mode != self.READ_MODE:
            self.finalise()
        else:
            self.close()

    def _open_lmbd_readonly(self):
        # We set the mapsize here because LMBD will map 1TB of virtual memory if
        # we don't, making it hard to figure out how much memory we're actually
        # using.
        map_size = None
        try:
            map_size = os.path.getsize(self.path)
        except OSError as e:
            raise exceptions.FileFormatError(str(e)) from e
        try:
            store = zarr.LMDBStore(
                self.path, map_size=map_size, readonly=True, subdir=False, lock=False)
        except lmdb.InvalidError as e:
            raise exceptions.FileFormatError(
                    "Unknown file format:{}".format(str(e))) from e
        except lmdb.Error as e:
            raise exceptions.FileFormatError(str(e)) from e
        return store

    def _open_readonly(self):
        if self.path is not None:
            store = self._open_lmbd_readonly()
        else:
            # This happens when we finalise an in-memory container.
            store = self.data.store
        self.data = zarr.open(store=store, mode="r")
        self._check_format()
        self._mode = self.READ_MODE

    def _new_lmdb_store(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        # The existence of a lock-file can confuse things, so delete it.
        remove_lmdb_lockfile(self.path)
        return zarr.LMDBStore(self.path, subdir=False)

    @classmethod
    def load(cls, path):
        # Try to read the file. This should raise the correct error if we have a
        # directory, missing file, permissions, etc.
        with open(path, "r"):
            pass
        self = cls.__new__(cls)
        self.mode = self.READ_MODE
        self.path = path
        self._open_readonly()
        logger.info("Loaded {}".format(self.summary()))
        return self

    def close(self):
        """
        Close this DataContainer. Any read or write operations attempted
        after calling this will fail.
        """
        if self._mode != self.READ_MODE:
            self.finalise()
        if self.data.store is not None:
            self.data.store.close()
        self.data = None
        self.mode = -1

    def copy(self, path=None):
        """
        Returns a copy of this DataContainer opened in 'edit' mode. If path
        is specified, this must not be equal to the path of the current
        data container. The new container will have a different UUID to the
        current.
        """
        if self._mode != self.READ_MODE:
            raise ValueError("Cannot copy unless in read mode.")
        if path is not None and self.path is not None:
            if os.path.abspath(path) == os.path.abspath(self.path):
                raise ValueError("Cannot copy to the same file")
        cls = type(self)
        other = cls.__new__(cls)
        other.path = path
        if path is None:
            # Have to work around a fairly weird bug in zarr where if we
            # try to use copy_store on an in-memory array we end up
            # overwriting the original values.
            other.data = zarr.group()
            zarr.copy_all(source=self.data, dest=other.data)
            for key, value in self.data.attrs.items():
                other.data.attrs[key] = value
        else:
            store = other._new_lmdb_store()
            zarr.copy_store(self.data.store, store)
            other.data = zarr.group(store)
        # Set a new UUID
        other.data.attrs["uuid"] = str(uuid.uuid4())
        other.data.attrs[FINALISED_KEY] = False
        other._mode = self.EDIT_MODE
        return other

    def finalise(self):
        """
        Ensures that the state of the data is flushed and writes the
        provenance for the current operation. The specified 'command' is used
        to fill the corresponding entry in the provenance dictionary.
        """
        self._check_write_modes()
        self.data.attrs[FINALISED_KEY] = True
        if self.path is not None:
            store = self.data.store
            store.close()
            logger.debug("Fixing up LMDB file size")
            with lmdb.open(
                    self.path, subdir=False, lock=False, writemap=True) as db:
                # LMDB maps a very large amount of space by default. While this
                # doesn't do any harm, it's annoying because we can't use ls to
                # see the file sizes and the amount of RAM we're mapping can
                # look like it's very large. So, we fix this up so that the
                # map size is equal to the number of pages in use.
                num_pages = db.info()["last_pgno"]
                page_size = db.stat()["psize"]
                db.set_mapsize(num_pages * page_size)
            # Remove the lock file as we don't need it after this point.
            remove_lmdb_lockfile(self.path)
        self._open_readonly()

    def _check_format(self):
        try:
            format_name = self.format_name
            format_version = self.format_version
        except KeyError:
            raise exceptions.FileFormatError("Incorrect file format")
        if format_name != self.FORMAT_NAME:
            raise exceptions.FileFormatError(
                "Incorrect file format: expected '{}' got '{}'".format(
                    self.FORMAT_NAME, format_name))
        if format_version[0] < self.FORMAT_VERSION[0]:
            raise exceptions.FileFormatError(
                "Format version {} too old. Current version = {}".format(
                    format_version, self.FORMAT_VERSION))
        if format_version[0] > self.FORMAT_VERSION[0]:
            raise exceptions.FileFormatError(
                "Format version {} too new. Current version = {}".format(
                    format_version, self.FORMAT_VERSION))

    def _check_build_mode(self):
        if self._mode != self.BUILD_MODE:
            raise ValueError("Invalid opertion: must be in build mode")

    def _check_edit_mode(self):
        if self._mode != self.EDIT_MODE:
            raise ValueError("Invalid opertion: must be in edit mode")

    def _check_write_modes(self):
        if self._mode not in (self.EDIT_MODE, self.BUILD_MODE):
            raise ValueError("Invalid opertion: must be in edit or build mode")

    @property
    def file_size(self):
        """
        Returns the size of the underlying file, or -1 if we do not have a
        file associated.
        """
        ret = -1
        if self.path is not None:
            ret = os.path.getsize(self.path)
        return ret

    def _check_metadata(self, metadata):
        ret = metadata
        if metadata is None:
            ret = {}
        elif not isinstance(metadata, abc.Mapping):
            raise TypeError("Metadata must be a JSON-like dictionary")
        return ret

    def add_provenance(self, timestamp, record):
        """
        Adds a new provenance record with the specified timestamp and record.
        Timestamps should ISO8601 formatted, and record is some JSON encodable
        object.
        """
        if self._mode not in (self.BUILD_MODE, self.EDIT_MODE):
            raise ValueError(
                "Invalid operation: cannot add provenances unless in BUILD "
                "or EDIT mode")
        n = self.num_provenances
        self.provenances_timestamp.resize(n + 1)
        self.provenances_record.resize(n + 1)
        self.provenances_timestamp[n] = timestamp
        self.provenances_record[n] = record

    def record_provenance(self, command=None, **kwargs):
        """
        Records the provenance information for this file using the
        tskit provenances schema.
        """
        timestamp = datetime.datetime.now().isoformat()
        record = provenance.get_provenance_dict(command=command, **kwargs)
        self.add_provenance(timestamp, record)

    @property
    def format_name(self):
        return self.data.attrs[FORMAT_NAME_KEY]

    @property
    def format_version(self):
        return tuple(self.data.attrs[FORMAT_VERSION_KEY])

    @property
    def finalised(self):
        ret = False
        if FINALISED_KEY in self.data.attrs:
            ret = self.data.attrs[FINALISED_KEY]
        return ret

    @property
    def uuid(self):
        return str(self.data.attrs["uuid"])

    @property
    def num_provenances(self):
        return self.provenances_timestamp.shape[0]

    @property
    def provenances_timestamp(self):
        return self.data["provenances/timestamp"]

    @property
    def provenances_record(self):
        return self.data["provenances/record"]

    def _format_str(self, values):
        """
        Helper function for formatting __str__ output.
        """
        s = ""
        # Quick hack to make sure everything lines up.
        max_key = len("provenances/timestamp")
        for k, v in values:
            s += "{:<{}} = {}\n".format(k, max_key, v)
        return s

    def __eq__(self, other):
        ret = NotImplemented
        if isinstance(other, type(self)):
            ret = self.uuid == other.uuid and self.data_equal(other)
        return ret

    def __str__(self):
        values = [
            ("path", self.path),
            ("file_size", humanize.naturalsize(self.file_size, binary=True)),
            ("format_name", self.format_name),
            ("format_version", self.format_version),
            ("finalised", self.finalised),
            ("uuid", self.uuid),
            ("num_provenances", self.num_provenances),
            ("provenances/timestamp", zarr_summary(self.provenances_timestamp)),
            ("provenances/record", zarr_summary(self.provenances_record))]
        return self._format_str(values)

    def arrays(self):
        """
        Returns a list of all the zarr arrays in this DataContainer.
        """
        ret = []

        def visitor(name, obj):
            if isinstance(obj, zarr.Array):
                ret.append((name, obj))
        self.data.visititems(visitor)
        return ret

    @property
    def info(self):
        """
        Returns a string containing the zarr info for each array.
        """
        s = str(self.data.info)
        for _, array in self.arrays():
            s += ("-" * 80) + "\n"
            s += str(array.info)
        return s

    def provenances(self):
        """
        Returns an iterator over the (timestamp, record) pairs representing
        the provenances for this data container.
        """
        timestamp = self.provenances_timestamp[:]
        record = self.provenances_record[:]
        for j in range(self.num_provenances):
            yield timestamp[j], record[j]


@attr.s
class Site(object):
    """
    A single site. Mirrors the definition in msprime.
    """
    # TODO document properly.
    id = attr.ib()
    position = attr.ib()
    ancestral_state = attr.ib()
    inference = attr.ib()
    metadata = attr.ib()


@attr.s
class Variant(object):
    """
    A single variant. Mirrors the definition in msprime but with some extra fields.
    """
    # TODO document properly.
    site = attr.ib()
    genotypes = attr.ib()
    alleles = attr.ib()


@attr.s
class Individual(object):
    """
    An Individual object.
    """
    # TODO document properly.
    id = attr.ib()
    location = attr.ib()
    metadata = attr.ib()


class SampleData(DataContainer):
    """
    SampleData(sequence_length=0, path=None, num_flush_threads=0, \
    compressor=None, chunk_size=1024)

    Class representing input sample data used for inference.
    See sample data file format :ref:`specifications <sec_file_formats_samples>`
    for details on the structure of this file.

    The most common usage for this class will be to import data from some
    external source and save it to file for later use. This will usually
    follow a pattern like:

    .. code-block:: python

        sample_data = tsinfer.SampleData(path="mydata.samples")
        sample_data.add_site(
            position=1234, genotypes=[0, 0, 1, 0], alleles=["G", "C"])
        sample_data.add_site(
            position=5678, genotypes=[1, 1, 1, 0], alleles=["A", "T"])
        sample_data.finalise()

    This creates a sample data file for four haploid samples and two sites, and
    saves it in the file "mydata.samples". Note that the call to
    :meth:`.finalise` is essential here to ensure that all data will be
    correctly flushed to disk. For convenience, a context manager may
    also be used to ensure this is done:

    .. code-block:: python

        with tsinfer.SampleData(path="mydata.samples") as sample_data:
            sample_data.add_site(1234, [0, 0, 1, 0], ["G", "C"])
            sample_data.add_site(5678, [1, 1, 1, 0], ["A", "T"])

    More complex :ref:`data models <sec_inference_data_model>` consisting
    of populations and polyploid individuals can also be specified. For
    example, we might have:

    .. code-block:: python

        with tsinfer.SampleData(path="mydata.samples") as sample_data:
            # Define populations
            sample_data.add_population(metadata={"name": "CEU"})
            sample_data.add_population(metadata={"name": "YRI"})
            # Define individuals
            sample_data.add_individual(
                ploidy=2, population=0, metadata={"name": "NA12878"})
            sample_data.add_individual(
                ploidy=2, population=0, metadata={"name": "NA12891"})
            sample_data.add_individual(
                ploidy=2, population=0, metadata={"name": "NA12892"})
            sample_data.add_individual(
                ploidy=2, population=1, metadata={"name": "NA18484"})
            # Define sites and genotypes
            sample_data.add_site(1234, [0, 1, 1, 1, 0, 0, 0, 0], ["G", "C"])
            sample_data.add_site(5678, [0, 0, 0, 0, 0, 0, 1, 1], ["A", "T"])

    In this example we defined two populations and four diploid individuals,
    and so our genotypes arrays are of length eight. Thus, at first site the
    first individual is heterozygous, the second is homozygous with the derived
    allele and the other two individuals are homozygous with the ancestral
    allele. To illustrate how we can use site and population metadata to link
    up with external data sources we use the 1000 genomes identifiers (although
    of course the genotype data is fake). Here we suppose that we have the
    famous NA12878 trio from the CEU population, and one other individual from
    the YRI population. This metadata is then embedded in the final tree
    sequence that we infer, allowing us to use it conveniently in downstream
    analyses.

    :param float sequence_length: If specified, this is the sequence length
        that will be associated with the tree sequence output by
        :func:`tsinfer.infer` and :func:`tsinfer.match_samples`. If provided
        site coordinates must be less than this value.
    :param str path: The path of the file to store the sample data. If None,
        the information is stored in memory and not persistent.
    :param int num_flush_threads: The number of background threads to use
        for compressing data and flushing to disc. If <= 0, do not spawn
        any threads but use a synchronous algorithm instead. Default=0.
    :param numcodecs.abc.Codec compressor: A :class:`numcodecs.abc.Codec` instance
        to use for compressing data. Any codec may be used, but
        problems may occur with very large datasets on certain codecs as
        they cannot compress buffers >2GB. If None, do not use any compression.
        By default, use the :class:`numcodecs.zstd.Zstd` codec is used
        when data is written to a file, and no compression when data is
        stored in memory.
    :param int chunk_size: The chunk size used for
        `zarr arrays <http://zarr.readthedocs.io/>`_. This affects
        compression level and algorithm performance. Default=1024.
    """
    FORMAT_NAME = "tsinfer-sample-data"
    FORMAT_VERSION = (1, 0)

    # State machine for handling automatic addition of samples.
    ADDING_POPULATIONS = 0
    ADDING_SAMPLES = 1
    ADDING_SITES = 2

    def __init__(self, sequence_length=0, **kwargs):

        super().__init__(**kwargs)
        self.data.attrs["sequence_length"] = float(sequence_length)
        chunks = self._chunk_size,
        populations_group = self.data.create_group("population")
        metadata = populations_group.create_dataset(
            "metadata", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=object, object_codec=self._metadata_codec)
        self._populations_writer = BufferedItemWriter(
            {"metadata": metadata}, num_threads=self._num_flush_threads)

        individuals_group = self.data.create_group("individual")
        metadata = individuals_group.create_dataset(
            "metadata", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=object, object_codec=self._metadata_codec)
        location = individuals_group.create_dataset(
            "location", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype="array:f8")
        self._individuals_writer = BufferedItemWriter(
            {"metadata": metadata, "location": location},
            num_threads=self._num_flush_threads)

        samples_group = self.data.create_group("samples")
        population = samples_group.create_dataset(
            "population", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=np.int32)
        individual = samples_group.create_dataset(
            "individual", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=np.int32)
        metadata = samples_group.create_dataset(
            "metadata", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=object, object_codec=self._metadata_codec)
        self._samples_writer = BufferedItemWriter(
            {"individual": individual, "population": population, "metadata": metadata},
            num_threads=self._num_flush_threads)

        sites_group = self.data.create_group("sites")
        sites_group.create_dataset(
            "position", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=np.float64)
        sites_group.create_dataset(
            "genotypes", shape=(0, 0), chunks=(self._chunk_size, self._chunk_size),
            compressor=self._compressor, dtype=np.uint8)
        sites_group.create_dataset(
            "inference", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=np.uint8)
        sites_group.create_dataset(
            "alleles", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=object, object_codec=self._metadata_codec)
        sites_group.create_dataset(
            "metadata", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=object, object_codec=self._metadata_codec)

        self._last_position = 0
        self._sites_writer = None
        # We are initially in the ADDING_POPULATIONS.
        self._build_state = self.ADDING_POPULATIONS

    def summary(self):
        return "SampleData(num_samples={}, num_sites={})".format(
            self.num_samples, self.num_sites)

    @property
    def sequence_length(self):
        return self.data.attrs["sequence_length"]

    @property
    def num_inference_sites(self):
        if self._mode == self.READ_MODE:
            # Cache the value as it's expensive to compute
            if not hasattr(self, "__num_inference_sites"):
                self.__num_inference_sites = int(np.sum(self.sites_inference[:]))
            return self.__num_inference_sites
        else:
            return int(np.sum(self.sites_inference[:]))

    @property
    def num_non_inference_sites(self):
        return self.num_sites - self.num_inference_sites

    @property
    def num_populations(self):
        return self.populations_metadata.shape[0]

    @property
    def num_samples(self):
        return self.samples_metadata.shape[0]

    @property
    def num_individuals(self):
        return self.individuals_metadata.shape[0]

    @property
    def num_sites(self):
        return self.sites_position.shape[0]

    @property
    def populations_metadata(self):
        return self.data["population/metadata"]

    @property
    def individuals_metadata(self):
        return self.data["individual/metadata"]

    @property
    def individuals_location(self):
        return self.data["individual/location"]

    @property
    def samples_population(self):
        return self.data["samples/population"]

    @property
    def samples_individual(self):
        return self.data["samples/individual"]

    @property
    def samples_metadata(self):
        return self.data["samples/metadata"]

    @property
    def sites_genotypes(self):
        return self.data["sites/genotypes"]

    @property
    def sites_position(self):
        return self.data["sites/position"]

    @property
    def sites_alleles(self):
        return self.data["sites/alleles"]

    @property
    def sites_metadata(self):
        return self.data["sites/metadata"]

    @property
    def sites_inference(self):
        return self.data["sites/inference"]

    @sites_inference.setter
    def sites_inference(self, value):
        self._check_edit_mode()
        new_value = np.array(value, dtype=int)
        if np.any(new_value > 1) or np.any(new_value < 0):
            raise ValueError("Input values must be boolean 0/1")
        self.data["sites/inference"][:] = new_value

    def __str__(self):
        values = [
            ("sequence_length", self.sequence_length),
            ("num_populations", self.num_populations),
            ("num_individuals", self.num_individuals),
            ("num_samples", self.num_samples),
            ("num_sites", self.num_sites),
            ("num_inference_sites", self.num_inference_sites),
            ("populations/metadata", zarr_summary(self.populations_metadata)),
            ("individuals/metadata", zarr_summary(self.individuals_metadata)),
            ("individuals/location", zarr_summary(self.individuals_location)),
            ("samples/individual", zarr_summary(self.samples_individual)),
            ("samples/population", zarr_summary(self.samples_population)),
            ("samples/metadata", zarr_summary(self.samples_metadata)),
            ("sites/position", zarr_summary(self.sites_position)),
            ("sites/alleles", zarr_summary(self.sites_alleles)),
            ("sites/inference", zarr_summary(self.sites_inference)),
            ("sites/genotypes", zarr_summary(self.sites_genotypes)),
            ("sites/metadata", zarr_summary(self.sites_metadata))]
        return super(SampleData, self).__str__() + self._format_str(values)

    def data_equal(self, other):
        """
        Returns True if all the data attributes of this input file and the
        specified input file are equal. This compares every attribute except
        the UUID and provenance.

        To compare two :class:`SampleData`` instances for exact equality of
        all data includeing UUIDs and provenance data, use ``s1 == s2``.

        :param SampleData other: The other :class:`SampleData` instance to
            compare with.
        :return: ``True`` if the data held in this :class:`SampleData`
            instance is identical to the date held in the other instacnce.
        :rtype: bool
        """
        return (
            self.format_name == other.format_name and
            self.format_version == other.format_version and
            self.num_populations == other.num_populations and
            self.num_individuals == other.num_individuals and
            self.num_samples == other.num_samples and
            self.num_sites == other.num_sites and
            self.num_inference_sites == other.num_inference_sites and
            np.all(self.samples_individual[:] == other.samples_individual[:]) and
            np.all(self.samples_population[:] == other.samples_population[:]) and
            np.all(self.sites_position[:] == other.sites_position[:]) and
            np.all(self.sites_inference[:] == other.sites_inference[:]) and
            np.all(self.sites_genotypes[:] == other.sites_genotypes[:]) and
            # Need to take a different approach with np object arrays.
            all(itertools.starmap(np.array_equal, zip(
                self.populations_metadata[:], other.populations_metadata[:]))) and
            all(itertools.starmap(np.array_equal, zip(
                self.individuals_metadata[:], other.individuals_metadata[:]))) and
            all(itertools.starmap(np.array_equal, zip(
                self.individuals_location[:], other.individuals_location[:]))) and
            all(itertools.starmap(np.array_equal, zip(
                self.samples_metadata[:], other.samples_metadata[:]))) and
            all(itertools.starmap(np.array_equal, zip(
                self.sites_metadata[:], other.sites_metadata[:]))) and
            all(itertools.starmap(np.array_equal, zip(
                self.sites_alleles[:], other.sites_alleles[:]))))

    ####################################
    # Write mode
    ####################################

    @classmethod
    def from_tree_sequence(cls, ts, **kwargs):
        self = cls.__new__(cls)
        self.__init__(sequence_length=ts.sequence_length, **kwargs)
        # Assume this is a haploid tree sequence.
        for population in ts.populations():
            self.add_population()
        for u in ts.samples():
            node = ts.node(u)
            self.add_individual(population=node.population, ploidy=1)
        for v in ts.variants():
            self.add_site(v.site.position, v.genotypes, v.alleles)
        # Insert all the provenance from the original tree sequence.
        for prov in ts.provenances():
            self.add_provenance(prov.timestamp, json.loads(prov.record))
        self.record_provenance(command="from-tree-sequence", **kwargs)
        self.finalise()
        return self

    def _alloc_site_writer(self):
        if self.num_samples < 2:
            raise ValueError("Must have at least 2 samples")
        self.sites_genotypes.resize(0, self.num_samples)
        arrays = {
            "position": self.sites_position,
            "genotypes": self.sites_genotypes,
            "alleles": self.sites_alleles,
            "metadata": self.sites_metadata,
            "inference": self.sites_inference,
        }
        self._sites_writer = BufferedItemWriter(
                arrays, num_threads=self._num_flush_threads)

    def add_population(self, metadata=None):
        """
        Adds a new :ref:`sec_inference_data_model_population` to this
        :class:`.SampleData` and returns its ID.

        All calls to this method must be made **before** individuals or sites
        are defined.

        :param dict metadata: A JSON encodable dict-like object containing
            metadata that is to be associated with this population.
        :return: The ID of the newly added population.
        :rtype: int
        """
        self._check_build_mode()
        if self._build_state != self.ADDING_POPULATIONS:
            raise ValueError("Cannot add populations after adding samples or sites")
        return self._populations_writer.add(metadata=self._check_metadata(metadata))

    def add_individual(self, ploidy=1, metadata=None, population=None, location=None):
        """
        Adds a new :ref:`sec_inference_data_model_individual` to this
        :class:`.SampleData` and returns its ID and those of the resulting additional
        samples. Adding an individual with ploidy ``k`` results in ``k`` new samples
        being added, and each of these samples will be associated with the
        new individual. Each new sample will also be associated with the specified
        population ID. It is an error to specify a population ID that does not
        correspond to a population defined using :meth:`.add_population`.

        All calls to this method must be made **after** populations are defined
        using :meth:`.add_population` and **before** sites are defined using
        :meth:`.add_site`.

        :param int ploidy: The ploidy of this individual. This corresponds to the
            number of samples added that refer to this individual. Defaults to 1
            (haploid).
        :param dict metadata: A JSON encodable dict-like object containing
            metadata that is to be associated with this individual.
        :param int population: The ID of the population to assoicate with this
            individual (or more precisely, with the samples for this individual).
            If not specified or None, defaults to the null population (-1).
        :param arraylike location: An array-like object defining n-dimensional
            spatial location of this individual. If not specified or None, the
            empty location is stored.
        :return: The ID of the newly added individual and a list of the sample
            IDs also added.
        :rtype: tuple(int, list(int))
        """
        self._check_build_mode()
        if self._build_state == self.ADDING_POPULATIONS:
            self._populations_writer.flush()
            self._populations_writer = None
            self._build_state = self.ADDING_SAMPLES
        if self._build_state != self.ADDING_SAMPLES:
            raise ValueError("Cannot add individuals after adding sites")

        if population is None:
            population = msprime.NULL_POPULATION
        if population >= self.num_populations:
            raise ValueError("population ID out of bounds")
        if ploidy <= 0:
            raise ValueError("Ploidy must be at least 1")
        if location is None:
            location = []
        location = np.array(location, dtype=np.float64)
        individual_id = self._individuals_writer.add(
            metadata=self._check_metadata(metadata), location=location)
        sample_ids = []
        for _ in range(ploidy):
            # For now default the metadata to the empty dict.
            sid = self._samples_writer.add(
                population=population, individual=individual_id, metadata={})
            sample_ids.append(sid)
        return individual_id, sample_ids

    def add_site(
            self, position, genotypes, alleles=None, metadata=None, inference=None):
        """
        Adds a new site to this :class:`.SampleData` and returns its ID.

        At a minimum, the new site must specify the ``position`` and
        ``genotypes``. Sites must be added in increasing order of position;
        duplicate positions are **not** supported. For each site a list of
        ``alleles`` may be supplied. This list defines the ancestral and
        derived states at the site. For example, if we set ``alleles=["A",
        "T"]`` then the ancestral state is "A" and the derived state is "T".
        The observed state for each sample is then encoded using the
        ``genotypes`` parameter. Thus if we have ``n`` samples then
        this must be a one dimensional array-like object with length ``n``. For
        a given array ``g`` and sample index ``j``, ``g[j]`` should contain
        ``0`` if sample ``j`` carries the ancestral state at this site and
        ``1`` if it carries the derived state. All sites must have genotypes
        for the same number of samples.

        All populations and individuals must be defined **before** this method
        is called. If no individuals have been defined using
        :meth:`.add_individual`, the first call to this method adds ``n``
        haploid individuals, where ``n`` is the length of the ``genotypes``
        array.

        :param float position: The floating point position of this site. Must be
            less than the ``sequence_length`` if provided to the :class:`.SampleData`
            constructor. Must be greater than all previously added sites.
        :param arraylike genotypes: An array-like object defining the sample
            genotypes at this site. The array of genotypes corresponds to the
            observed alleles for each sample, represented by indexes into the
            alleles array. This input is converted to a numpy array with
            dtype ``np.uint8``; therefore, for maximum efficiency ensure
            that the input array is also of this type.
        :param list(str) alleles: A list of strings defining the alleles at this
            site. The zero'th element of this list is the **ancestral state**
            and the one'th element is the **derived state**. Only biallelic
            sites are currently supported. If not specified or None, defaults
            to ["0", "1"].
        :param dict metadata: A JSON encodable dict-like object containing
            metadata that is to be associated with this site.
        :param bool inference: If True, use this site during the inference
            process. If False, do not use this sites for inference; in this
            case, :func:`match_samples` will place mutations on the existing
            tree in a way that encodes the supplied sample genotypes. If
            ``inference=None`` (the default), use any site in which the
            number of samples carrying the derived state is greater than
            1 and less than the number of samples.
        :return: The ID of the newly added site.
        :rtype: int
        """
        genotypes = np.array(genotypes, dtype=np.uint8, copy=False)
        self._check_build_mode()
        if self._build_state == self.ADDING_POPULATIONS:
            if genotypes.shape[0] == 0:
                # We could just raise an error here but we set the state
                # here so that we can raise the same error as other
                # similar conditions.
                self._build_state = self.ADDING_SAMPLES
            else:
                # Add in the default haploid samples.
                for _ in range(genotypes.shape[0]):
                    self.add_individual()
        if self._build_state == self.ADDING_SAMPLES:
            self._individuals_writer.flush()
            self._samples_writer.flush()
            self._alloc_site_writer()
            self._build_state = self.ADDING_SITES
            self._last_position = -1
        assert self._build_state == self.ADDING_SITES

        if alleles is None:
            alleles = ["0", "1"]
        if len(alleles) > 2:
            raise ValueError("Only biallelic sites supported")
        if len(set(alleles)) != len(alleles):
            raise ValueError("Alleles must be distinct")
        if np.any(genotypes >= len(alleles)) or np.any(genotypes < 0):
            raise ValueError("Genotypes values must be between 0 and len(alleles) - 1")
        if genotypes.shape != (self.num_samples,):
            raise ValueError("Must have num_samples genotypes.")
        if position < 0:
            raise ValueError("position must be > 0")
        if self.sequence_length > 0 and position >= self.sequence_length:
            raise ValueError("If sequence_length is set, sites positions must be less.")
        if position <= self._last_position:
            raise ValueError(
                "Sites positions must be unique and added in increasing order")
        count = np.sum(genotypes)
        if count > 1 and count < self.num_samples:
            if inference is None:
                inference = True
        else:
            if inference is None:
                inference = False
            if inference:
                raise ValueError(
                    "Cannot specify singletons or fixed sites for inference")
        site_id = self._sites_writer.add(
            position=position, genotypes=genotypes,
            metadata=self._check_metadata(metadata),
            inference=inference, alleles=alleles)
        self._last_position = position
        return site_id

    def finalise(self):
        if self._mode == self.BUILD_MODE:
            if self._build_state == self.ADDING_POPULATIONS:
                raise ValueError("Must add at least one sample individual")
            elif self._build_state == self.ADDING_SAMPLES:
                self._individuals_writer.flush()
                self._samples_writer.flush()
            elif self._build_state == self.ADDING_SITES:
                self._sites_writer.flush()
            if self.num_sites == 0:
                raise ValueError("Must add at least one site")
            self._build_state = -1
            if self.sequence_length == 0:
                # Need to be careful that sequence_length is JSON serialisable here.
                self.data.attrs["sequence_length"] = float(self._last_position) + 1

        super(SampleData, self).finalise()

    ####################################
    # Read mode
    ####################################

    def genotypes(self, inference_sites=None):
        """
        Returns an iterator over the (site_id, genotypes) pairs.
        If ``inference_sites`` is ``None``, return genotypes for all sites.
        If ``inference_sites`` is ``True``, return only genotypes at sites that have
        been marked for inference; if ``False``, return only genotypes at sites
        that are not marked for inference.

        :param bool inference_sites: Control the sites that we return genotypes
            for.
        """
        inference = self.sites_inference[:]
        for j, a in enumerate(chunk_iterator(self.sites_genotypes)):
            if inference_sites is None or inference[j] == inference_sites:
                yield j, a

    def variants(self, inference_sites=None):
        """
        Returns an iterator over the Variant objects. This is equivalent to
        the TreeSequence.variants iterator.

        If ``inference_sites`` is ``None``, return variants for all sites.
        If ``inference_sites`` is ``True``, return only variants at sites that have
        been marked for inference; if ``False``, return only genotypes at sites
        that are not marked for inference.

        :param bool inference_sites: Control the sites that we return variants
            for.
        """
        position = self.sites_position[:]
        alleles = self.sites_alleles[:]
        inference = self.sites_inference[:]
        metadata = self.sites_metadata[:]
        for j, genotypes in self.genotypes(inference_sites):
            if inference_sites is None or inference[j] == inference_sites:
                site = Site(
                    id=j, position=position[j], ancestral_state=alleles[j][0],
                    inference=inference[j], metadata=metadata[j])
                variant = Variant(
                    site=site, alleles=tuple(alleles[j]), genotypes=genotypes)
                yield variant

    def __all_haplotypes(self, inference_sites=None):
        if inference_sites is not None:
            selection = self.sites_inference[:] == int(inference_sites)
        # We iterate over chunks vertically here, and it's not worth complicating
        # the chunk iterator to handle this.
        chunk_size = self.sites_genotypes.chunks[1]
        for j in range(self.num_samples):
            if j % chunk_size == 0:
                chunk = self.sites_genotypes[:, j: j + chunk_size].T
            a = chunk[j % chunk_size]
            if inference_sites is None:
                yield j, a
            else:
                yield j, a[selection]

    def haplotypes(self, samples=None, inference_sites=None):
        if samples is None:
            samples = np.arange(self.num_samples)
        else:
            samples = np.array(samples, copy=False)
            if np.any(samples[:-1] >= samples[1:]):
                raise ValueError("sample indexes must be in increasing order.")
            if samples.shape[0] > 0 and samples[-1] >= self.num_samples:
                raise ValueError("Sample index too large.")
        j = 0
        for index, a in self.__all_haplotypes(inference_sites):
            if j == len(samples):
                break
            if index == samples[j]:
                yield index, a
                j += 1

    def individuals(self):
        # TODO document
        iterator = zip(self.individuals_location[:], self.individuals_metadata[:])
        for j, (location, metadata) in enumerate(iterator):
            yield Individual(j, location=location, metadata=metadata)


@attr.s
class Ancestor(object):
    """
    An ancestor object.
    """
    # TODO document properly.
    id = attr.ib()
    start = attr.ib()
    end = attr.ib()
    time = attr.ib()
    focal_sites = attr.ib()
    haplotype = attr.ib()


class AncestorData(DataContainer):
    """
    AncestorData(sample_data, path=None, num_flush_threads=0, compressor=None, \
    chunk_size=1024)

    Class representing the stored ancestor data produced by
    :func:`generate_ancestors`. See the samples file format
    :ref:`specifications <sec_file_formats_ancestors>` for details on the structure
    of this file.

    :param SampleData sample_data: The :class:`.SampleData` instance
        that this ancestor data file was generated from.
    :param str path: The path of the file to store the sample data. If None,
        the information is stored in memory and not persistent.
    :param int num_flush_threads: The number of background threads to use
        for compressing data and flushing to disc. If <= 0, do not spawn
        any threads but use a synchronous algorithm instead. Default=0.
    :param numcodecs.abc.Codec compressor: A :class:`numcodecs.abc.Codec` instance
        to use for compressing data. Any codec may be used, but
        problems may occur with very large datasets on certain codecs as
        they cannot compress buffers >2GB. If None, do not use any compression.
        By default, use the :class:`numcodecs.zstd.Zstd` codec is used
        when data is written to a file, and no compression when data is
        stored in memory.
    :param int chunk_size: The chunk size used for
        `zarr arrays <http://zarr.readthedocs.io/>`_. This affects
        compression level and algorithm performance. Default=1024.
    """
    FORMAT_NAME = "tsinfer-ancestor-data"
    FORMAT_VERSION = (1, 0)

    def __init__(self, sample_data, **kwargs):
        super().__init__(**kwargs)
        self.sample_data = sample_data
        # Cache the num_sites value here as it's expensive to compute.
        self._num_sites = self.sample_data.num_inference_sites
        self.data.attrs["sample_data_uuid"] = sample_data.uuid
        if self.sample_data.sequence_length == 0:
            raise ValueError("Bad samples file: sequence_length cannot be zero")
        self.data.attrs["sequence_length"] = self.sample_data.sequence_length

        chunks = self._chunk_size
        # Add in the positions for the sites.
        sites_inference = self.sample_data.sites_inference[:]
        position = self.sample_data.sites_position[:][sites_inference == 1]
        self.data.create_dataset(
            "sites/position", data=position, chunks=chunks, compressor=self._compressor,
            dtype=np.float64)

        self.data.create_dataset(
            "ancestors/start", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=np.int32)
        self.data.create_dataset(
            "ancestors/end", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=np.int32)
        self.data.create_dataset(
            "ancestors/time", shape=(0,), chunks=chunks, compressor=self._compressor,
            dtype=np.uint32)
        self.data.create_dataset(
            "ancestors/focal_sites", shape=(0,), chunks=chunks,
            dtype="array:i4", compressor=self._compressor)
        self.data.create_dataset(
            "ancestors/haplotype", shape=(0,), chunks=chunks,
            dtype="array:u1", compressor=self._compressor)

        self.item_writer = BufferedItemWriter({
            "start": self.ancestors_start,
            "end": self.ancestors_end,
            "time": self.ancestors_time,
            "focal_sites": self.ancestors_focal_sites,
            "haplotype": self.ancestors_haplotype},
            num_threads=self._num_flush_threads)

        # Add in the provenance trail from the sample_data file.
        for timestamp, record in sample_data.provenances():
            self.add_provenance(timestamp, record)

    def summary(self):
        return "AncestorData(num_ancestors={}, num_sites={})".format(
            self.num_ancestors, self.num_sites)

    def __str__(self):
        values = [
            ("sequence_length", self.sequence_length),
            ("sample_data_uuid", self.sample_data_uuid),
            ("num_ancestors", self.num_ancestors),
            ("num_sites", self.num_sites),
            ("sites/position", zarr_summary(self.sites_position)),
            ("ancestors/start", zarr_summary(self.ancestors_start)),
            ("ancestors/end", zarr_summary(self.ancestors_end)),
            ("ancestors/time", zarr_summary(self.ancestors_time)),
            ("ancestors/focal_sites", zarr_summary(self.ancestors_focal_sites)),
            ("ancestors/haplotype", zarr_summary(self.ancestors_haplotype))]
        return super(AncestorData, self).__str__() + self._format_str(values)

    def data_equal(self, other):
        """
        Returns True if all the data attributes of this input file and the
        specified input file are equal. This compares every attribute except
        the UUID.
        """
        return (
            self.sequence_length == other.sequence_length and
            self.sample_data_uuid == other.sample_data_uuid and
            self.format_name == other.format_name and
            self.format_version == other.format_version and
            self.num_ancestors == other.num_ancestors and
            self.num_sites == other.num_sites and
            np.array_equal(self.sites_position[:], other.sites_position[:]) and
            np.array_equal(self.ancestors_start[:], other.ancestors_start[:]) and
            np.array_equal(self.ancestors_end[:], other.ancestors_end[:]) and
            # Need to take a different approach with np object arrays.
            all(itertools.starmap(np.array_equal, zip(
                self.ancestors_focal_sites[:], other.ancestors_focal_sites[:]))) and
            all(itertools.starmap(np.array_equal, zip(
                self.ancestors_haplotype[:], other.ancestors_haplotype[:]))))

    @property
    def sequence_length(self):
        return self.data.attrs["sequence_length"]

    @property
    def sample_data_uuid(self):
        return self.data.attrs["sample_data_uuid"]

    @property
    def num_ancestors(self):
        return self.ancestors_start.shape[0]

    @property
    def num_sites(self):
        return self.sites_position.shape[0]

    @property
    def sites_position(self):
        return self.data["sites/position"]

    @property
    def ancestors_start(self):
        return self.data["ancestors/start"]

    @property
    def ancestors_end(self):
        return self.data["ancestors/end"]

    @property
    def ancestors_time(self):
        return self.data["ancestors/time"]

    @property
    def ancestors_focal_sites(self):
        return self.data["ancestors/focal_sites"]

    @property
    def ancestors_haplotype(self):
        return self.data["ancestors/haplotype"]

    @property
    def ancestors_length(self):
        """
        Returns the length of ancestors in physical coordinates.
        """
        # Ancestor start and end are half-closed. The last site is assumed
        # to cover the region up to sequence length.
        pos = np.hstack([self.sites_position[:], [self.sequence_length]])
        start = self.ancestors_start[:]
        end = self.ancestors_end[:]
        return pos[end] - pos[start]

    ####################################
    # Write mode
    ####################################

    def add_ancestor(self, start, end, time, focal_sites, haplotype):
        """
        Adds an ancestor with the specified haplotype, with ancestral material
        over the interval [start:end], that is associated with the specfied time
        and has new mutations at the specified list of focal sites.
        """
        self._check_build_mode()
        haplotype = np.array(haplotype, dtype=np.uint8)
        focal_sites = np.array(focal_sites, dtype=np.int32)
        if start < 0:
            raise ValueError("Start must be >= 0")
        if end > self._num_sites:
            raise ValueError("end must be <= num_variant_sites")
        if start >= end:
            raise ValueError("start must be < end")
        if haplotype.shape != (end - start,):
            raise ValueError("haplotypes incorrect shape.")
        if time <= 0:
            raise ValueError("time must be > 0")
        if not np.all(haplotype[focal_sites - start] == 1):
            raise ValueError("haplotype[j] must be = 1 for all focal sites")
        if np.any(focal_sites < start) or np.any(focal_sites >= end):
            raise ValueError("focal sites must be between start and end")
        if np.any(haplotype[start: end] > 1):
            raise ValueError("Biallelic sites only supported.")
        self.item_writer.add(
            start=start, end=end, time=time, focal_sites=focal_sites,
            haplotype=haplotype)

    def finalise(self):
        if self._mode == self.BUILD_MODE:
            self.item_writer.flush()
            self.item_writer = None
        super(AncestorData, self).finalise()

    def ancestors(self):
        """
        Returns an iterator over all the ancestors.
        """
        # TODO document properly.
        start = self.ancestors_start[:]
        end = self.ancestors_end[:]
        time = self.ancestors_time[:]
        focal_sites = self.ancestors_focal_sites[:]
        for j, h in enumerate(chunk_iterator(self.ancestors_haplotype)):
            yield Ancestor(
                id=j, start=start[j], end=end[j], time=time[j],
                focal_sites=focal_sites[j], haplotype=h)


def load(path):
    """
    Loads a tsinfer :class:`.SampleData` or :class:`.AncestorData` file from
    the specified path. The correct class will be determined by the content
    of the file. If the file is format not recognised a
    :class:`.FileFormatError` will be thrown.

    :param str path: The path of the file we wish to load.
    :return: The corresponding :class:`.SampleData` or :class:`.AncestorData`
        instance opened in read only mode.
    :rtype: :class:`.AncestorData` or :class:`.SampleData`.
    :raises: :class:`.FileFormatError` if the file cannot be read.
    """
    # TODO This is pretty inelegant, but it works. Really we should call the
    # load on the superclass which can dispatch to the registered subclasses
    # for a given format_name.
    tsinfer_file = None
    try:
        logger.debug("Trying SampleData file")
        tsinfer_file = SampleData.load(path)
        logger.debug("Loaded SampleData file")
    except exceptions.FileFormatError as e:
        logger.debug("SampleData load failed: {}".format(e))
    try:
        logger.debug("Trying AncestorData file")
        tsinfer_file = AncestorData.load(path)
        logger.debug("Loaded AncestorData file")
    except exceptions.FileFormatError as e:
        logger.debug("AncestorData load failed: {}".format(e))
    if tsinfer_file is None:
        raise exceptions.FileFormatError(
            "Unrecognised file format. Try running with -vv and check the log "
            "for more details on what went wrong")
    return tsinfer_file
