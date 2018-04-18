# TODO copyright and license.
"""
Manage tsinfer's various HDF5 file formats.
"""
import uuid
import logging
import time
import queue
import dbm

import numpy as np
import zarr
import humanize
import numcodecs.blosc as blosc
import msprime

import tsinfer.threads as threads

# FIXME need some global place to keep these constants
UNKNOWN_ALLELE = 255

# We don't want blosc to spin up extra threads for compression.
blosc.use_threads = False
logger = logging.getLogger(__name__)


FORMAT_NAME_KEY = "format_name"
FORMAT_VERSION_KEY = "format_version"
FINALISED_KEY = "finalised"

DEFAULT_COMPRESSOR = blosc.Blosc(cname='zstd', clevel=9, shuffle=blosc.BITSHUFFLE)

# These functions don't really do what they are supposed to now. Should
# decouple the iteration functions from the data classes below. Also
# need to simplify the logic and introduce a simpler double buffered
# method.


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


class BufferedSite(object):
    """
    Simple container to hold site information while being buffered during
    addition. The frequency is the number of genotypes with the derived
    state.
    """
    def __init__(self, position, frequency, alleles):
        self.position = position
        self.frequency = frequency
        self.alleles = alleles


class BufferedAncestor(object):
    """
    Simple container to hold ancestor information while being buffered during
    addition.
    """
    def __init__(self, start, end, time_, focal_sites):
        self.start = start
        self.end = end
        self.time = time_
        self.focal_sites = focal_sites


def zarr_summary(array):
    """
    Returns a string with a brief summary of the specified zarr array.
    """
    return "shape={};chunks={};size={};dtype={}".format(
        array.shape, array.chunks, humanize.naturalsize(array.nbytes),
        array.dtype)


class DataContainer(object):
    """
    Superclass of objects used to represent a collection of related
    data. Each datacontainer in a wrapper around a zarr group.
    """
    # Must be defined by subclasses.
    FORMAT_NAME = None
    FORMAT_VERSION = None

    @classmethod
    def load(cls, filename):
        self = cls()
        self.store = zarr.LMDBStore(filename, readonly=True, subdir=False, lock=False)
        self.data = zarr.open_group(store=self.store)
        self.check_format()
        return self

    def check_format(self):
        try:
            format_name = self.format_name
            format_version = self.format_version
        except KeyError:
            raise ValueError("Incorrect file format")
        if format_name != self.FORMAT_NAME:
            raise ValueError("Incorrect file format: expected '{}' got '{}'".format(
                self.FORMAT_VERSION, format_version))
        if format_version[0] < self.FORMAT_VERSION[0]:
            raise ValueError("Format version {} too old. Current version = {}".format(
                format_version, self.FORMAT_VERSION))
        if format_version[0] > self.FORMAT_VERSION[0]:
            raise ValueError("Format version {} too new. Current version = {}".format(
                format_version, self.FORMAT_VERSION))

    def _initialise(self, filename=None):
        """
        Initialise the basic state of the data container.
        """
        self.store = None
        self.data = zarr.group()
        if filename is not None:
            self.store = zarr.LMDBStore(filename, lock=False, subdir=False)
            self.data = zarr.open_group(store=self.store)
        self.data.attrs[FORMAT_NAME_KEY] = self.FORMAT_NAME
        self.data.attrs[FORMAT_VERSION_KEY] = self.FORMAT_VERSION
        self.data.attrs["uuid"] = str(uuid.uuid4())

    def finalise(self):
        """
        Ensures that the state of the data is flushed to file if a store
        is present.
        """
        self.data.attrs[FINALISED_KEY] = True

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

    def haplotypes(self, start=0):
        """
        Returns an iterator over the haplotypes starting at the specified
        index..
        """
        iterator = transposed_threaded_row_iterator(self.genotypes)
        for j, h in enumerate(iterator):
            if j >= start:
                yield h


class SampleData(DataContainer):
    """
    Class representing the data stored about our input samples.
    """
    FORMAT_NAME = "tsinfer-sample-data"
    FORMAT_VERSION = (0, 3)

    @property
    def num_samples(self):
        return self.data.attrs["num_samples"]

    @property
    def num_sites(self):
        return self.data.attrs["num_sites"]

    @property
    def num_variant_sites(self):
        return self.data.attrs["num_variant_sites"]

    @property
    def num_singleton_sites(self):
        return self.data.attrs["num_singleton_sites"]

    @property
    def num_invariant_sites(self):
        return self.data.attrs["num_invariant_sites"]

    @property
    def sequence_length(self):
        return self.data.attrs["sequence_length"]

    @property
    def position(self):
        return self.data["sites/position"]

    @property
    def ancestral_state(self):
        return self.data["sites/ancestral_state"]

    @property
    def ancestral_state_offset(self):
        return self.data["sites/ancestral_state_offset"]

    @property
    def derived_state(self):
        return self.data["sites/derived_state"]

    @property
    def derived_state_offset(self):
        return self.data["sites/derived_state_offset"]

    @property
    def frequency(self):
        return self.data["sites/frequency"]

    @property
    def invariant_site(self):
        return self.data["invariants/site"]

    @property
    def singleton_site(self):
        return self.data["singletons/site"]

    @property
    def singleton_sample(self):
        return self.data["singletons/sample"]

    @property
    def genotypes(self):
        return self.data["variants/genotypes"]

    @property
    def variant_site(self):
        return self.data["variants/site"]

    def __str__(self):
        path = None
        if self.store is not None:
            path = self.store.path
        values = [
            ("path", path),
            ("format_name", self.format_name),
            ("format_version", self.format_version),
            ("finalised", self.finalised),
            ("uuid", self.uuid),
            ("num_samples", self.num_samples),
            ("num_sites", self.num_sites),
            ("num_variant_sites", self.num_variant_sites),
            ("num_singleton_sites", self.num_singleton_sites),
            ("num_invariant_sites", self.num_invariant_sites),
            ("sequence_length", self.sequence_length),
            ("position", zarr_summary(self.position)),
            ("frequency", zarr_summary(self.frequency)),
            ("ancestral_state", zarr_summary(self.ancestral_state)),
            ("ancestral_state_offset", zarr_summary(self.ancestral_state_offset)),
            ("derived_state", zarr_summary(self.derived_state)),
            ("derived_state_offset", zarr_summary(self.derived_state_offset)),
            ("variant_site", zarr_summary(self.variant_site)),
            ("singleton_site", zarr_summary(self.singleton_site)),
            ("invariant_site", zarr_summary(self.invariant_site)),
            ("singleton_sample", zarr_summary(self.singleton_sample)),
            ("genotypes", zarr_summary(self.genotypes))]
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
            self.num_samples == other.num_samples and
            self.num_sites == other.num_sites and
            self.num_variant_sites == other.num_variant_sites and
            self.num_singleton_sites == other.num_singleton_sites and
            self.num_invariant_sites == other.num_invariant_sites and
            self.sequence_length == other.sequence_length and
            np.array_equal(self.position[:], other.position[:]) and
            np.array_equal(self.frequency[:], other.frequency[:]) and
            np.array_equal(self.ancestral_state[:], other.ancestral_state[:]) and
            np.array_equal(
                self.ancestral_state_offset[:], other.ancestral_state_offset[:]) and
            np.array_equal(self.derived_state[:], other.derived_state[:]) and
            np.array_equal(
                self.derived_state_offset[:], other.derived_state_offset[:]) and
            np.array_equal(self.variant_site[:], other.variant_site[:]) and
            np.array_equal(self.invariant_site[:], other.invariant_site[:]) and
            np.array_equal(self.singleton_site[:], other.singleton_site[:]) and
            np.array_equal(self.singleton_sample[:], other.singleton_sample[:]) and
            np.array_equal(self.genotypes[:], other.genotypes[:]))

    ####################################
    # Write mode
    ####################################

    @classmethod
    def initialise(
            cls, num_samples=None, sequence_length=0,
            filename=None, chunk_size=8192, compressor=DEFAULT_COMPRESSOR):
        """
        Initialises a new SampleData object. Data can be added to
        this object using the add_variant method.
        """
        self = cls()
        super(cls, self)._initialise(filename)
        self.data.attrs["sequence_length"] = float(sequence_length)
        self.data.attrs["num_samples"] = int(num_samples)

        self.site_buffer = []
        self.genotypes_buffer = np.empty((chunk_size, num_samples), dtype=np.uint8)
        self.genotypes_buffer_offset = 0
        self.singletons_buffer = []
        self.invariants_buffer = []
        self.compressor = compressor
        self.variants_group = self.data.create_group("variants")
        x_chunk = chunk_size
        y_chunk = min(chunk_size, num_samples)
        self.variants_group.create_dataset(
            "genotypes", shape=(0, num_samples), chunks=(x_chunk, y_chunk),
            dtype=np.uint8, compressor=compressor)
        return self

    def add_variant(self, position, alleles, genotypes):
        genotypes = np.array(genotypes, dtype=np.uint8, copy=False)
        if len(alleles) > 2:
            raise ValueError("Only biallelic sites supported")
        if np.any(genotypes >= len(alleles)) or np.any(genotypes < 0):
            raise ValueError("Genotypes values must be between 0 and len(alleles) - 1")
        if genotypes.shape != (self.num_samples,):
            raise ValueError("Must have num_samples genotypes.")
        if position < 0:
            raise ValueError("position must be > 0")
        if self.sequence_length > 0 and position >= self.sequence_length:
            raise ValueError("If sequence_length is set, sites positions must be less.")

        frequency = np.sum(genotypes)
        if frequency == 1:
            sample = np.where(genotypes == 1)[0][0]
            self.singletons_buffer.append((len(self.site_buffer), sample))
        elif 0 < frequency < self.num_samples:
            j = self.genotypes_buffer_offset
            N = self.genotypes_buffer.shape[0]
            self.genotypes_buffer[j] = genotypes
            if j == N - 1:
                self.genotypes.append(self.genotypes_buffer)
                self.genotypes_buffer_offset = -1
            self.genotypes_buffer_offset += 1
        else:
            self.invariants_buffer.append(len(self.site_buffer))
        self.site_buffer.append(BufferedSite(position, frequency, alleles))

    def finalise(self):
        if self.genotypes_buffer is None:
            raise ValueError("Cannot call finalise in read-mode")
        variant_sites = []
        num_samples = self.num_samples
        num_sites = len(self.site_buffer)
        if num_sites == 0:
            raise ValueError("Must have at least one site")
        position = np.empty(num_sites)
        frequency = np.empty(num_sites, dtype=np.uint32)
        ancestral_states = []
        derived_states = []
        for j, site in enumerate(self.site_buffer):
            position[j] = site.position
            frequency[j] = site.frequency
            if site.frequency > 1 and site.frequency < num_samples:
                variant_sites.append(j)
            ancestral_states.append(site.alleles[0])
            derived_states.append("" if len(site.alleles) < 2 else site.alleles[1])
        sites_group = self.data.create_group("sites")
        sites_group.array(
            "position", data=position, chunks=(num_sites,), compressor=self.compressor)
        sites_group.array(
            "frequency", data=frequency, chunks=(num_sites,), compressor=self.compressor)

        ancestral_state, ancestral_state_offset = msprime.pack_strings(ancestral_states)
        sites_group.array(
            "ancestral_state", data=ancestral_state, chunks=(num_sites,),
            compressor=self.compressor)
        sites_group.array(
            "ancestral_state_offset", data=ancestral_state_offset,
            chunks=(num_sites + 1,), compressor=self.compressor)
        derived_state, derived_state_offset = msprime.pack_strings(derived_states)
        sites_group.array(
            "derived_state", data=derived_state, chunks=(num_sites,),
            compressor=self.compressor)
        sites_group.array(
            "derived_state_offset", data=derived_state_offset, chunks=(num_sites + 1,),
            compressor=self.compressor)

        num_singletons = len(self.singletons_buffer)
        singleton_sites = np.array(
            [site for site, _ in self.singletons_buffer], dtype=np.int32)
        singleton_samples = np.array(
            [sample for _, sample in self.singletons_buffer], dtype=np.int32)
        singletons_group = self.data.create_group("singletons")
        chunks = max(num_singletons, 1),
        singletons_group.array(
            "site", data=singleton_sites, chunks=chunks, compressor=self.compressor)
        singletons_group.array(
            "sample", data=singleton_samples, chunks=chunks, compressor=self.compressor)

        num_invariants = len(self.invariants_buffer)
        invariant_sites = np.array(self.invariants_buffer, dtype=np.int32)
        invariants_group = self.data.create_group("invariants")
        chunks = max(num_invariants, 1),
        invariants_group.array(
            "site", data=invariant_sites, chunks=chunks, compressor=self.compressor)

        num_variant_sites = len(variant_sites)
        self.data.attrs["num_sites"] = num_sites
        self.data.attrs["num_variant_sites"] = num_variant_sites
        self.data.attrs["num_singleton_sites"] = num_singletons
        self.data.attrs["num_invariant_sites"] = num_invariants

        chunks = max(num_variant_sites, 1),
        self.variants_group.create_dataset(
            "site", shape=(num_variant_sites,), chunks=chunks,
            dtype=np.int32, data=variant_sites, compressor=self.compressor)

        self.genotypes.append(self.genotypes_buffer[:self.genotypes_buffer_offset])
        self.site_buffer = None
        self.genotypes_buffer = None
        super(SampleData, self).finalise()

    ####################################
    # Read mode
    ####################################

    def variants(self):
        """
        Returns an iterator over the (site_id, genotypes) pairs for all variant
        sites in the input data.
        """
        # TODO add a num_threads or other option to control threading.
        variant_sites = self.variant_site[:]
        for j, genotypes in enumerate(threaded_row_iterator(self.genotypes)):
            yield variant_sites[j], genotypes


class AncestorData(DataContainer):
    """
    Class representing the data stored about our input samples.
    """
    FORMAT_NAME = "tsinfer-ancestor-data"
    FORMAT_VERSION = (0, 2)

    def __str__(self):
        path = None
        store = None
        if self.store is not None:
            path = self.store.path
            store = dbm.whichdb(path)
        values = [
            ("path", path),
            ("store", store),
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
            ("focal_sites_offset", zarr_summary(self.focal_sites_offset)),
            ("genotypes", zarr_summary(self.genotypes))]
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
            np.array_equal(self.focal_sites[:], other.focal_sites[:]) and
            np.array_equal(
                self.focal_sites_offset[:], other.focal_sites_offset[:]) and
            np.array_equal(self.genotypes[:], other.genotypes[:]))

    @property
    def sample_data_uuid(self):
        return self.data.attrs["sample_data_uuid"]

    @property
    def num_ancestors(self):
        return self.data.attrs["num_ancestors"]

    @property
    def num_sites(self):
        return self.data.attrs["num_sites"]

    @property
    def start(self):
        return self.data["ancestors/start"]

    @property
    def end(self):
        return self.data["ancestors/end"]

    @property
    def time(self):
        return self.data["ancestors/time"]

    @property
    def focal_sites(self):
        return self.data["ancestors/focal_sites"]

    @property
    def focal_sites_offset(self):
        return self.data["ancestors/focal_sites_offset"]

    @property
    def genotypes(self):
        return self.data["ancestors/genotypes"]

    ####################################
    # Write mode
    ####################################

    @classmethod
    def initialise(
            cls, input_data, filename=None, chunk_size=8192,
            compressor=DEFAULT_COMPRESSOR):
        """
        Initialises a new SampleData object. Data can be added to
        this object using the add_ancestor method.
        """
        self = cls()
        super(cls, self)._initialise(filename)
        self.input_data = input_data
        self.compressor = compressor
        self.data.attrs["sample_data_uuid"] = input_data.uuid

        num_sites = self.input_data.num_variant_sites
        self.data.attrs["num_sites"] = num_sites
        self.ancestor_buffer = []
        self.genotypes_buffer = np.empty((num_sites, chunk_size), dtype=np.uint8)
        self.genotypes_buffer[:] = UNKNOWN_ALLELE
        self.genotypes_buffer_offset = 0

        self.ancestors_group = self.data.create_group("ancestors")
        x_chunk = max(1, min(chunk_size, num_sites))
        y_chunk = chunk_size
        self.ancestors_group.create_dataset(
            "genotypes", shape=(num_sites, 0), chunks=(x_chunk, y_chunk),
            dtype=np.uint8, compressor=self.compressor)
        return self

    def add_ancestor(self, start, end, time, focal_sites, haplotype):
        """
        Adds an ancestor with the specified haplotype, with ancestral material
        over the interval [start:end], that is associated with the specfied time
        and has new mutations at the specified list of focal sites.
        """
        num_sites = self.input_data.num_variant_sites
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
        j = self.genotypes_buffer_offset
        N = self.genotypes_buffer.shape[1]
        self.genotypes_buffer[start: end, j] = haplotype[start: end]
        if j == N - 1:
            self.genotypes.append(self.genotypes_buffer, axis=1)
            self.genotypes_buffer_offset = -1
            self.genotypes_buffer[:] = UNKNOWN_ALLELE
        self.genotypes_buffer_offset += 1
        self.ancestor_buffer.append(BufferedAncestor(start, end, time, focal_sites))

    def finalise(self):
        if self.ancestor_buffer is None:
            raise ValueError("Cannot call finalise in read-mode")
        total_focal_sites = sum(
            ancestor.focal_sites.shape[0] for ancestor in self.ancestor_buffer)
        num_ancestors = len(self.ancestor_buffer)
        self.data.attrs["num_ancestors"] = num_ancestors
        start = np.empty(num_ancestors, dtype=np.int32)
        end = np.empty(num_ancestors, dtype=np.int32)
        time = np.empty(num_ancestors, dtype=np.uint32)
        focal_sites = np.empty(total_focal_sites, dtype=np.int32)
        focal_sites_offset = np.zeros(num_ancestors + 1, dtype=np.uint32)
        for j, ancestor in enumerate(self.ancestor_buffer):
            start[j] = ancestor.start
            end[j] = ancestor.end
            time[j] = ancestor.time
            focal_sites_offset[j + 1] = (
                focal_sites_offset[j] + ancestor.focal_sites.shape[0])
            focal_sites[
                focal_sites_offset[j]: focal_sites_offset[j + 1]] = ancestor.focal_sites
        chunks = max(num_ancestors, 1),
        self.ancestors_group.create_dataset(
            "start", shape=(num_ancestors,), chunks=chunks, data=start,
            compressor=self.compressor)
        self.ancestors_group.create_dataset(
            "end", shape=(num_ancestors,), chunks=chunks, data=end,
            compressor=self.compressor)
        self.ancestors_group.create_dataset(
            "time", shape=(num_ancestors,), chunks=chunks, data=time,
            compressor=self.compressor)
        self.ancestors_group.create_dataset(
            "focal_sites_offset", shape=(num_ancestors + 1,),
            chunks=(num_ancestors + 1,), data=focal_sites_offset,
            compressor=self.compressor)
        self.ancestors_group.create_dataset(
            "focal_sites", shape=(total_focal_sites,),
            chunks=(max(total_focal_sites, 1)), data=focal_sites,
            compressor=self.compressor)

        self.genotypes.append(
            self.genotypes_buffer[:, :self.genotypes_buffer_offset], axis=1)
        self.ancestor_buffer = None
        self.genotypes_buffer = None
        super(AncestorData, self).finalise()
