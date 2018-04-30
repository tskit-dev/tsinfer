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
import uuid
import logging
import queue
import itertools
import os
import os.path
import threading
import warnings
import collections.abc as abc

import numpy as np
import zarr
import lmdb
import humanize
import numcodecs
import numcodecs.blosc as blosc

import tsinfer.threads as threads
import tsinfer.exceptions as exceptions


####################
# Monkey patch zarr's JSON codec to fix bug.
# https://github.com/zarr-developers/numcodecs/issues/76
def encode(self, buf):
    buf = np.asanyarray(buf)
    items = buf.tolist()
    items.append(buf.dtype.str)
    items.append(buf.shape)
    return self._encoder.encode(items).encode(self._text_encoding)


def decode(self, buf, out=None):
    buf = numcodecs.compat.buffer_tobytes(buf)
    items = self._decoder.decode(buf.decode(self._text_encoding))
    dec = np.empty(items[-1], dtype=items[-2])
    dec[:] = items[:-2]
    if out is not None:
        np.copyto(out, dec)
        return out
    else:
        return dec


numcodecs.json.JSON.encode = encode
numcodecs.json.JSON.decode = decode

####################


# FIXME need some global place to keep these constants
UNKNOWN_ALLELE = 255

# We don't want blosc to spin up extra threads for compression.
blosc.use_threads = False
logger = logging.getLogger(__name__)


FORMAT_NAME_KEY = "format_name"
FORMAT_VERSION_KEY = "format_version"
FINALISED_KEY = "finalised"

DEFAULT_COMPRESSOR = blosc.Blosc(cname='zstd', clevel=9, shuffle=blosc.BITSHUFFLE)


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
            # Make sure the destination array is zero sized at the start.
            shape = list(array.shape)
            shape[0] = 0
            array.resize(*shape)
            self.buffers[key] = [None for _ in range(self.num_buffers)]
            for j in range(self.num_buffers):
                chunks = list(array.shape)
                # For the buffer we must set the chunk size to 1, or we will
                # reencode the buffer each time we update.
                chunks[0] = 1
                # 2018-04-18 Zarr current emits a warning when calling empty_like on
                # object arrays. See https://github.com/zarr-developers/zarr/issues/257
                # Remove this catch_warnings when the bug is fixed.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.buffers[key][j] = zarr.empty_like(
                        array, compressor=None, chunks=chunks)
                chunks[0] = self.chunk_size
                self.buffers[key][j].resize(*chunks)
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
            array[start: end] = self.buffers[key][write_buffer][:n]
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
    return "shape={}; uncompressed size={}".format(
        array.shape, humanize.naturalsize(array.nbytes))


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
    # Must be defined by subclasses.
    FORMAT_NAME = None
    FORMAT_VERSION = None

    def _open_readonly(self, filename):
        # We set the mapsize here because LMBD will map 1TB of virtual memory if
        # we don't, making it hard to figure out how much memory we're actually
        # using.
        map_size = None
        try:
            map_size = os.path.getsize(filename)
        except OSError:
            # Ignore any exceptions here and let LMDB handle them.
            pass
        try:
            self.store = zarr.LMDBStore(
                filename, map_size=map_size, readonly=True, subdir=False, lock=False)
        except lmdb.InvalidError as e:
            raise exceptions.FileFormatError(
                    "Unknown file format:{}".format(str(e))) from e
        except lmdb.Error as e:
            raise exceptions.FileError(str(e)) from e
        self.data = zarr.open(store=self.store, mode="r")
        self.check_format()

    @classmethod
    def load(cls, filename):
        self = cls()
        self._open_readonly(filename)
        return self

    def check_format(self):
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

    def _initialise(self, filename=None, num_flush_threads=0):
        """
        Initialise the basic state of the data container.
        """
        self.store = None
        self._num_flush_threads = num_flush_threads
        self.data = zarr.group()
        if filename is not None:
            self.store = zarr.LMDBStore(filename, subdir=False)
            self.data = zarr.open_group(store=self.store, mode="w")
        self.data.attrs[FORMAT_NAME_KEY] = self.FORMAT_NAME
        self.data.attrs[FORMAT_VERSION_KEY] = self.FORMAT_VERSION
        self.data.attrs["uuid"] = str(uuid.uuid4())

    @property
    def file_size(self):
        """
        Returns the size of the underlying file, or -1 if we do not have a
        file associated.
        """
        ret = -1
        if self.store is not None:
            ret = os.path.getsize(self.store.path)
        return ret

    def finalise(self):
        """
        Ensures that the state of the data is flushed to file if a store
        is present.
        """
        self.data.attrs[FINALISED_KEY] = True
        if self.store is not None:
            filename = self.store.path
            self.store.close()
            logger.debug("Fixing up LMDB file size")
            with lmdb.open(
                    self.store.path, subdir=False, lock=False, writemap=True) as db:
                # LMDB maps a very large amount of space by default. While this
                # doesn't do any harm, it's annoying because we can't use ls to
                # see the file sizes and the amount of RAM we're mapping can
                # look like it's very large. So, we fix this up so that the
                # map size is equal to the number of pages in use.
                num_pages = db.info()["last_pgno"]
                page_size = db.stat()["psize"]
                db.set_mapsize(num_pages * page_size)
            # Remove the lock file as we don't need it after this point.
            lockfile = filename + "-lock"
            if os.path.exists(lockfile):
                os.unlink(lockfile)
            # Reopen the data in read-only mode.
            self.data = None
            self._open_readonly(filename)
        else:
            # Reopen the group in read-only mode.
            self.data = zarr.open(self.data.store, mode='r')

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
            ret = True
        return ret

    @property
    def uuid(self):
        return str(self.data.attrs["uuid"])

    def _format_str(self, values):
        """
        Helper function for formatting __str__ output.
        """
        s = ""
        max_key = max(len(k) for k, _ in values)
        for k, v in values:
            s += "{:<{}} = {}\n".format(k, max_key, v)
        return s

    def __eq__(self, other):
        ret = NotImplemented
        if isinstance(other, type(self)):
            ret = self.uuid == other.uuid and self.data_equal(other)
        return ret

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


class SampleData(DataContainer):
    """
    Class representing the data stored about our input samples.
    """
    FORMAT_NAME = "tsinfer-sample-data"
    FORMAT_VERSION = (0, 3)

    def __init__(self):
        self._num_inference_sites = None

    @property
    def sequence_length(self):
        return self.data.attrs["sequence_length"]

    @property
    def num_inference_sites(self):
        if self._num_inference_sites is None:
            self._num_inference_sites = int(np.sum(self.sites_inference[:]))
        return self._num_inference_sites

    @property
    def num_populations(self):
        return self.populations_metadata.shape[0]

    @property
    def num_samples(self):
        return self.samples_metadata.shape[0]

    @property
    def num_sites(self):
        return self.sites_position.shape[0]

    @property
    def populations_metadata(self):
        return self.data["population/metadata"]

    @property
    def samples_population(self):
        return self.data["samples/population"]

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

    def __str__(self):
        path = None
        if self.store is not None:
            path = self.store.path
        values = [
            ("path", path),
            ("file_size", humanize.naturalsize(self.file_size, binary=True)),
            ("format_name", self.format_name),
            ("format_version", self.format_version),
            ("finalised", self.finalised),
            ("uuid", self.uuid),
            ("sequence_length", self.sequence_length),
            ("num_populations", self.num_populations),
            ("num_samples", self.num_samples),
            ("num_sites", self.num_sites),
            ("num_inference_sites", self.num_inference_sites),
            ("populations/metadata", zarr_summary(self.populations_metadata)),
            ("samples/population", zarr_summary(self.samples_population)),
            ("samples/metadata", zarr_summary(self.samples_metadata)),
            ("sites/position", zarr_summary(self.sites_position)),
            ("sites/alleles", zarr_summary(self.sites_alleles)),
            ("sites/inference", zarr_summary(self.sites_inference)),
            ("sites/genotypes", zarr_summary(self.sites_genotypes)),
            ("sites/metadata", zarr_summary(self.sites_metadata))]
        return self._format_str(values)

    def data_equal(self, other):
        """
        Returns True if all the data attributes of this input file and the
        specified input file are equal. This compares every attribute except
        the UUID.
        """
        return (
            self.format_name == other.format_name and
            self.format_version == other.format_version and
            self.num_populations == other.num_populations and
            self.num_samples == other.num_samples and
            self.num_sites == other.num_sites and
            self.num_inference_sites == other.num_inference_sites and
            np.all(self.samples_population[:] == other.samples_population[:]) and
            np.all(self.sites_position[:] == other.sites_position[:]) and
            np.all(self.sites_inference[:] == other.sites_inference[:]) and
            np.all(self.sites_genotypes[:] == other.sites_genotypes[:]) and
            # Need to take a different approach with np object arrays.
            all(itertools.starmap(np.array_equal, zip(
                self.populations_metadata[:], other.populations_metadata[:]))) and
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
    def initialise(
            cls, sequence_length=0, filename=None, chunk_size=1024,
            compressor=DEFAULT_COMPRESSOR, num_flush_threads=0, num_samples=None):
        """
        Initialises a new SampleData object.
        """
        self = cls()
        super(cls, self)._initialise(filename, num_flush_threads)

        self.data.attrs["sequence_length"] = float(sequence_length)

        chunk_size = max(1, chunk_size)
        chunks = chunk_size,
        metadata_codec = numcodecs.JSON()

        self.populations_group = self.data.create_group("population")
        metadata = self.populations_group.create_dataset(
            "metadata", shape=(0,), chunks=chunks, compressor=compressor,
            dtype=object, object_codec=metadata_codec)
        self._populations_writer = BufferedItemWriter(
            {"metadata": metadata}, num_threads=num_flush_threads)

        self.samples_group = self.data.create_group("samples")
        population = self.samples_group.create_dataset(
            "population", shape=(0,), chunks=chunks, compressor=compressor,
            dtype=np.int32)
        metadata = self.samples_group.create_dataset(
            "metadata", shape=(0,), chunks=chunks, compressor=compressor,
            dtype=object, object_codec=metadata_codec)
        self._samples_writer = BufferedItemWriter(
            {"population": population, "metadata": metadata},
            num_threads=num_flush_threads)

        self.sites_group = self.data.create_group("sites")
        self.sites_group.create_dataset(
            "position", shape=(0,), chunks=chunks, compressor=compressor,
            dtype=np.float64)
        self.sites_group.create_dataset(
            "genotypes", shape=(0, 0), chunks=(chunk_size, chunk_size),
            compressor=compressor, dtype=np.uint8)
        self.sites_group.create_dataset(
            "inference", shape=(0,), chunks=chunks, compressor=compressor,
            dtype=np.uint8)
        self.sites_group.create_dataset(
            "alleles", shape=(0,), chunks=chunks, compressor=compressor,
            dtype=object, object_codec=metadata_codec)
        self.sites_group.create_dataset(
            "metadata", shape=(0,), chunks=chunks, compressor=compressor,
            dtype=object, object_codec=metadata_codec)

        self._sites_writer = None
        if num_samples is not None:
            # Add in the default population and samples.
            self.add_population()
            for _ in range(num_samples):
                self.add_sample()
        return self

    def _alloc_site_writer(self):
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

    def _check_metadata(self, metadata):
        ret = metadata
        if metadata is None:
            ret = {}
        elif not isinstance(metadata, abc.Mapping):
            raise TypeError("Metadata must be a JSON-like dictionary")
        return ret

    def add_population(self, metadata=None):
        self._populations_writer.add(metadata=self._check_metadata(metadata))

    def add_sample(self, population=-1, metadata=None):
        if self._populations_writer is not None:
            self._populations_writer.flush()
            self._populations_writer = None
        if population >= self.num_populations:
            raise ValueError("population ID out of bounds")
        self._samples_writer.add(
            population=population, metadata=self._check_metadata(metadata))

    def add_site(self, position, alleles, genotypes, metadata=None):
        if self._samples_writer is not None:
            self._samples_writer.flush()
            self._samples_writer = None
            self._alloc_site_writer()
        genotypes = np.array(genotypes, dtype=np.uint8, copy=False)
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
        count = np.sum(genotypes)
        inference_site = count > 1 and count < self.num_samples
        self._sites_writer.add(
            position=position, genotypes=genotypes,
            metadata=self._check_metadata(metadata),
            inference=inference_site, alleles=alleles)

    def finalise(self):
        self._sites_writer.flush()
        self._sites_writer = None
        super(SampleData, self).finalise()

    ####################################
    # Read mode
    ####################################

    def genotypes(self, inference_sites=None):
        """
        Returns an iterator over the sample (sites_id, genotypes) pairs.
        If inference_sites is None, return all genotypes. If it is True,
        return only genotypes at sites that have been marked for inference.
        If False, return only genotypes at sites that are not marked for inference.
        """
        inference = self.sites_inference[:]
        for j, a in enumerate(chunk_iterator(self.sites_genotypes)):
            if inference_sites is None or inference[j] == inference_sites:
                yield j, a

    def haplotypes(self, inference_sites=None):
        """
        Returns an iterator over the sample haplotypes. If inference_sites is
        None, return for all sites. If inference_sites is False, return for
        sites that are not selected for inference. If True, return states
        for sites that are selected for inference.
        """
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
                yield a
            else:
                yield a[selection]


class AncestorData(DataContainer):
    """
    Class representing the data stored about our input samples.
    """
    FORMAT_NAME = "tsinfer-ancestor-data"
    FORMAT_VERSION = (0, 2)

    def __str__(self):
        path = None
        if self.store is not None:
            path = self.store.path
        values = [
            ("path", path),
            ("format_name", self.format_name),
            ("format_version", self.format_version),
            ("uuid", self.uuid),
            ("sample_data_uuid", self.sample_data_uuid),
            ("num_ancestors", self.num_ancestors),
            ("num_sites", self.num_sites),
            ("start", zarr_summary(self.start)),
            ("end", zarr_summary(self.end)),
            ("time", zarr_summary(self.time)),
            ("focal_sites", zarr_summary(self.focal_sites)),
            ("ancestor", zarr_summary(self.ancestor))]
        return self._format_str(values)

    def data_equal(self, other):
        """
        Returns True if all the data attributes of this input file and the
        specified input file are equal. This compares every attribute except
        the UUID.
        """
        return (
            self.sample_data_uuid == other.sample_data_uuid and
            self.format_name == other.format_name and
            self.format_version == other.format_version and
            self.num_ancestors == other.num_ancestors and
            self.num_sites == other.num_sites and
            np.array_equal(self.start[:], other.start[:]) and
            np.array_equal(self.end[:], other.end[:]) and
            # Need to take a different approach with np object arrays.
            all(itertools.starmap(np.array_equal, zip(
                self.focal_sites[:], other.focal_sites[:]))) and
            all(itertools.starmap(np.array_equal, zip(
                self.ancestor[:], other.ancestor[:]))))

    @property
    def sample_data_uuid(self):
        return self.data.attrs["sample_data_uuid"]

    @property
    def num_ancestors(self):
        return self.start.shape[0]

    @property
    def num_sites(self):
        return self.data.attrs["num_sites"]

    @property
    def start(self):
        return self.data["start"]

    @property
    def end(self):
        return self.data["end"]

    @property
    def time(self):
        return self.data["time"]

    @property
    def focal_sites(self):
        return self.data["focal_sites"]

    @property
    def ancestor(self):
        return self.data["ancestor"]

    ####################################
    # Write mode
    ####################################

    @classmethod
    def initialise(
            cls, input_data, filename=None, chunk_size=1024,
            num_flush_threads=0, compressor=DEFAULT_COMPRESSOR):
        """
        Initialises a new SampleData object. Data can be added to
        this object using the add_ancestor method.
        """
        self = cls()
        super(cls, self)._initialise(filename, num_flush_threads)
        self.input_data = input_data
        self.compressor = compressor
        self.data.attrs["sample_data_uuid"] = input_data.uuid
        self.data.attrs["num_sites"] = self.input_data.num_inference_sites

        chunks = max(1, chunk_size),
        self.data.create_dataset(
            "start", shape=(0,), chunks=chunks, compressor=self.compressor,
            dtype=np.int32)
        self.data.create_dataset(
            "end", shape=(0,), chunks=chunks, compressor=self.compressor,
            dtype=np.int32)
        self.data.create_dataset(
            "time", shape=(0,), chunks=chunks, compressor=self.compressor,
            dtype=np.uint32)
        self.data.create_dataset(
            "focal_sites", shape=(0,), chunks=chunks,
            dtype="array:i4", compressor=self.compressor)
        self.data.create_dataset(
            "ancestor", shape=(0,), chunks=chunks,
            dtype="array:u1", compressor=self.compressor)

        self.item_writer = BufferedItemWriter({
            "start": self.start, "end": self.end, "time": self.time,
            "focal_sites": self.focal_sites, "ancestor": self.ancestor},
            num_threads=num_flush_threads)
        return self

    def add_ancestor(self, start, end, time, focal_sites, haplotype):
        """
        Adds an ancestor with the specified haplotype, with ancestral material
        over the interval [start:end], that is associated with the specfied time
        and has new mutations at the specified list of focal sites.
        """
        num_sites = self.input_data.num_inference_sites
        haplotype = np.array(haplotype, dtype=np.uint8, copy=False)
        focal_sites = np.array(focal_sites, dtype=np.int32, copy=False)
        if start < 0:
            raise ValueError("Start must be >= 0")
        if end > num_sites:
            raise ValueError("end must be <= num_variant_sites")
        if start >= end:
            raise ValueError("start must be < end")
        if haplotype.shape != (num_sites,):
            raise ValueError("haplotypes incorrect shape.")
        if time <= 0:
            raise ValueError("time must be > 0")
        if not np.all(haplotype[focal_sites] == 1):
            raise ValueError("haplotype[j] must be = 1 for all focal sites")
        if np.any(focal_sites < start) or np.any(focal_sites >= end):
            raise ValueError("focal sites must be between start and end")
        if np.any(haplotype[start: end] > 1):
            raise ValueError("Biallelic sites only supported.")
        ancestor = haplotype[start:end].copy()
        self.item_writer.add(
            start=start, end=end, time=time, focal_sites=focal_sites,
            ancestor=ancestor)

    def finalise(self):
        self.item_writer.flush()
        self.item_writer = None
        super(AncestorData, self).finalise()

    def ancestors(self):
        """
        Returns an iterator over all the ancestors.
        """
        for a in chunk_iterator(self.ancestor):
            yield a
