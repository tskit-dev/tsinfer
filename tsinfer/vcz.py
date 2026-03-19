#
# Copyright (C) 2018-2026 University of Oxford
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
Low-level utilities for reading and writing VCZ (VCF Zarr) stores.

``open_store``
    Open a VCZ store from a path/URL, or pass through an existing zarr Group.

``open_group``
    Open a writable zarr Group backed by a MemoryStore or filesystem path.

``resolve_field``
    Map a field specification (string, {path,field} dict, scalar, or None)
    to a numpy array.  Used to resolve per-source metadata like sample_time.

``sequence_length``
    Read the contig length from a VCZ store (``contig_length[0]``).

``num_contigs``
    Return the number of contigs declared in a VCZ store.

``AncestorWriter``
    Streams ancestors to a zarr Group, flushing by sample-chunk.
"""

from __future__ import annotations

import collections
import logging
import math
import queue
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from vcztools.retrieval import variant_chunk_iter
from vcztools.samples import parse_samples
from zarr.core.dtype.npy.string import VariableLengthUTF8

logger = logging.getLogger(__name__)

_VLEN_STR = VariableLengthUTF8()
_ZARR_FORMAT = 2

# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------


def open_group(store: str | Path | None = None) -> zarr.Group:
    """
    Open a writable zarr Group.

    Parameters
    ----------
    store : str, Path, or None
        - ``None`` → in-memory store (MemoryStore).
        - ``str`` / ``Path`` → filesystem-backed store at that path.
    """
    if store is None:
        return zarr.open_group(
            zarr.storage.MemoryStore(), mode="w", zarr_format=_ZARR_FORMAT
        )
    return zarr.open_group(str(store), mode="w", zarr_format=_ZARR_FORMAT)


def open_store(path_or_group: str | Path | zarr.Group) -> zarr.Group:
    """
    Return a zarr Group for the given path, URL, or existing Group.

    - zarr.Group  → returned unchanged (in-memory or already-open store)
    - str / Path  → opened with ``zarr.open(..., mode="r")``
    """
    if isinstance(path_or_group, zarr.Group):
        return path_or_group
    return zarr.open(str(path_or_group), mode="r")


# ---------------------------------------------------------------------------
# Field resolution
# ---------------------------------------------------------------------------


def resolve_field(
    store: zarr.Group,
    field_spec: str | dict | int | float | None,
    join_key: str,
    n: int,
    fill_value: Any = None,
) -> np.ndarray | None:
    """
    Resolve a metadata field specification to a 1-D numpy array of length *n*.

    Parameters
    ----------
    store:
        The source VCZ store that the field should be resolved against.
    field_spec:
        One of:
        - ``str``   — name of an array that already lives inside *store*.
        - ``dict``  — ``{"path": ..., "field": ...}``: the field is read from
          a separate VCZ file and joined to *store* via *join_key*.
        - scalar (``int`` / ``float``) — broadcast to all *n* entries.
        - ``None``  — return ``None`` (field not specified).
    join_key:
        The array used as the join key:
        - ``"variant_position"`` for site-dimensioned arrays.
        - ``"sample_id"`` for sample-dimensioned arrays.
    n:
        Expected length of the returned array.
    fill_value:
        Value used for entries that are absent in a dict-style join.
        Design conventions: ``None`` for masks (→ site/sample included),
        ``0`` for sample times.
    """
    if field_spec is None:
        return None

    if isinstance(field_spec, str):
        return np.asarray(store[field_spec][:])

    if isinstance(field_spec, dict):
        return _resolve_joined_field(store, field_spec, join_key, n, fill_value)

    # Scalar — broadcast
    if isinstance(field_spec, (int, float)):
        return np.full(n, field_spec)

    raise TypeError(
        f"field_spec must be str, dict, scalar, or None; got {type(field_spec)}"
    )


def _resolve_joined_field(
    store: zarr.Group,
    spec: dict,
    join_key: str,
    n: int,
    fill_value: Any,
) -> np.ndarray:
    """
    Resolve a ``{path, field}`` spec by opening a second VCZ and joining on
    *join_key*.

    The source store's join key array and the annotation store's join key array
    are both read; the annotation values are then aligned to the source order.
    Entries in the source that have no matching key in the annotation receive
    *fill_value*.
    """
    ann_path = spec.get("path")
    ann_field = spec.get("field")
    if ann_path is None or ann_field is None:
        raise ValueError(
            f"A dict field_spec must contain both 'path' and 'field' keys; got: {spec!r}"
        )

    ann_store = open_store(ann_path)

    source_keys = np.asarray(store[join_key][:])
    ann_keys = np.asarray(ann_store[join_key][:])
    ann_values = np.asarray(ann_store[ann_field][:])

    # Build lookup: annotation key → value
    ann_lookup = {k: v for k, v in zip(ann_keys.tolist(), ann_values.tolist())}

    result = np.array([ann_lookup.get(k, fill_value) for k in source_keys.tolist()])

    if len(result) != n:
        raise ValueError(
            f"resolve_field: expected {n} entries after join on '{join_key}', "
            f"got {len(result)}"
        )
    return result


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def resolve_samples_selection(
    store: zarr.Group,
    samples: str | None,
) -> np.ndarray | None:
    """
    Convert a bcftools-style samples string to a selection index array.

    Returns None if samples is None (keep all samples).
    """
    if samples is None:
        return None
    _, selection = parse_samples(samples, store["sample_id"][:])
    return selection


def iter_variants(
    store: zarr.Group,
    *,
    fields: list[str] | None = None,
    include: str | None = None,
    exclude: str | None = None,
    samples_selection: np.ndarray | None = None,
    regions: str | None = None,
    targets: str | None = None,
):
    """
    Yield per-variant dicts from a VCZ store, with optional filtering.

    Delegates to ``vcztools.retrieval.variant_chunk_iter`` for chunked
    reading with bcftools-style include/exclude expressions, sample
    selection, and region/target filtering.

    Each yielded dict maps zarr array names to per-variant values
    (variants dimension removed).
    """
    for chunk_data in variant_chunk_iter(
        store,
        fields=fields,
        include=include,
        exclude=exclude,
        samples_selection=samples_selection,
        regions=regions,
        targets=targets,
    ):
        first_field = next(iter(chunk_data.values()))
        num_variants = len(first_field)
        for i in range(num_variants):
            yield {name: chunk_data[name][i] for name in chunk_data}


def iter_genotypes(store, positions, ancestral_allele_index, sample_include=None):
    """
    Yield ``(num_haplotypes,)`` int8 derived-genotype rows for the requested
    positions, one row per position in the order given.

    Each yielded row is remapped so that the ancestral allele becomes 0,
    any other allele becomes 1, and missing values become -1.

    Positions are mapped to row indices via ``variant_position``.  Reads
    ``call_genotype`` chunk-by-chunk so that at most one zarr chunk is in
    memory at a time.  Sample selection and ploidy flattening are applied
    transparently.

    Parameters
    ----------
    store : zarr.Group
        VCZ store containing ``call_genotype`` and ``variant_position``.
    positions : array-like of int
        Genomic positions to retrieve (must be non-decreasing and present
        in ``variant_position``).
    ancestral_allele_index : array-like of int
        Per-site index into the allele list identifying the ancestral
        allele.  Must be the same length as *positions*.
    sample_include : array-like of bool or None
        Boolean mask (length ``num_samples``) where True means keep this
        sample.  ``None`` keeps all samples.

    Yields
    ------
    np.ndarray
        ``(num_haplotypes,)`` int8 array where 0 = ancestral, 1 = derived,
        -1 = missing.
    """
    positions = np.asarray(positions, dtype=np.int32)
    ancestral_allele_index = np.asarray(ancestral_allele_index, dtype=np.int8)

    if len(positions) == 0:
        return

    # Map positions to row indices
    all_positions = np.asarray(store["variant_position"][:], dtype=np.int32)
    site_indices = np.searchsorted(all_positions, positions).astype(np.int64)

    if sample_include is not None:
        sample_include = np.asarray(sample_include, dtype=bool)

    call_gt = store["call_genotype"]
    chunk_size = call_gt.chunks[0]

    # Walk through site_indices in order, loading each chunk at most once
    # for a run of consecutive indices that fall inside it.
    n = len(site_indices)
    i = 0
    cached_chunk_id = -1
    cached_flat = None

    while i < n:
        cid = int(site_indices[i]) // chunk_size

        if cid != cached_chunk_id:
            chunk_data = call_gt.blocks[cid]
            if sample_include is not None:
                chunk_data = chunk_data[:, sample_include, :]
            cached_flat = chunk_data.reshape(chunk_data.shape[0], -1)
            cached_chunk_id = cid

        # Yield all indices that fall within the current chunk
        chunk_start = cid * chunk_size
        while i < n and int(site_indices[i]) // chunk_size == cid:
            local = int(site_indices[i]) - chunk_start
            raw = cached_flat[local]
            anc_idx = ancestral_allele_index[i]
            is_missing = raw < 0
            is_ancestral = raw == anc_idx
            yield np.where(
                is_missing, np.int8(-1), np.where(is_ancestral, np.int8(0), np.int8(1))
            ).astype(np.int8)
            i += 1


def sequence_length(store: zarr.Group) -> int:
    """Return the sequence length (first contig length) from a VCZ store."""
    return int(store["contig_length"][0])


def num_contigs(store: zarr.Group) -> int:
    """Return the number of contigs declared in a VCZ store."""
    return int(store["contig_id"].shape[0])


# ---------------------------------------------------------------------------
# Zarr array helpers
# ---------------------------------------------------------------------------


def _arr(group, name, data, dims, chunks=None, compressor=None):
    kw = {}
    if chunks is not None:
        kw["chunks"] = chunks
    if compressor is not None:
        kw["compressor"] = compressor
    a = group.create_array(name, data=data, **kw)
    a.attrs["_ARRAY_DIMENSIONS"] = dims
    return a


def _str_array(group, name, data, dims, compressor=None):
    data = np.asarray(data)
    kw = {}
    if compressor is not None:
        kw["compressor"] = compressor
    a = group.create_array(name, shape=data.shape, dtype=_VLEN_STR, **kw)
    a[:] = data
    a.attrs["_ARRAY_DIMENSIONS"] = dims
    return a


def _build_output_alleles(alleles, anc_indices, num_sites):
    """
    Build [ancestral, derived] allele pairs for output.

    Parameters
    ----------
    alleles : (num_sites, max_alleles) array of str
    anc_indices : (num_sites,) int8 — ancestral allele index per site
    num_sites : int
    """
    out = np.empty((num_sites, 2), dtype=object)
    out[:] = ""
    for i in range(num_sites):
        site_alleles = alleles[i].tolist()
        anc_idx = int(anc_indices[i])
        out[i, 0] = str(site_alleles[anc_idx])
        for j, al in enumerate(site_alleles):
            if j != anc_idx and al is not None and al != "":
                out[i, 1] = str(al)
                break
    return out


# ---------------------------------------------------------------------------
# Ancestor writing — two-queue pipeline
# ---------------------------------------------------------------------------


@dataclass
class _WorkItem:
    """Work item for the worker thread pool."""

    ab: Any  # _tsinfer.AncestorBuilder
    focal_sites: np.ndarray
    anc_time: float
    ancestor_index: int  # global index across all intervals
    expected_count: int  # expected ancestors in this chunk


class ChunkBuffer:
    """
    Thread-safe buffer for one sample-chunk of ancestor data.

    Haplotypes are stored row-wise: ``haplotype_buf[slot]`` is the full
    local-coordinate haplotype array for one ancestor.  This gives optimal
    cache locality during the worker fill (a single contiguous row copy)
    at the cost of a transpose in the writer thread before the zarr write.

    Each worker writes to a distinct ``slot`` (row), so data arrays have
    no races.  Only ``filled_count`` increment + completion check needs a lock.
    """

    __slots__ = (
        "chunk_idx",
        "haplotype_buf",
        "times",
        "starts",
        "ends",
        "focal_positions",
        "expected_count",
        "filled_count",
        "_partial_gt",
        "_partial_count",
        "_lock",
    )

    def __init__(self, n_local: int, chunk_size: int):
        self.chunk_idx: int = -1
        # Row-major: each slot is a contiguous row of n_local sites
        self.haplotype_buf = np.full((chunk_size, n_local), np.int8(-1), dtype=np.int8)
        self.times = np.zeros(chunk_size, dtype=np.float64)
        self.starts = np.zeros(chunk_size, dtype=np.int32)
        self.ends = np.zeros(chunk_size, dtype=np.int32)
        self.focal_positions: list = [None] * chunk_size
        self.expected_count: int = -1  # -1 = unknown
        self.filled_count: int = 0
        # Only set for partial-chunk reload from previous interval
        self._partial_gt: np.ndarray | None = None  # (num_sites, partial_count, 1)
        self._partial_count: int = 0
        self._lock = threading.Lock()

    def record_fill(self) -> bool:
        """Increment filled_count; return True if chunk is complete."""
        with self._lock:
            self.filled_count += 1
            return self.expected_count >= 0 and self.filled_count == self.expected_count

    def seal(self, expected_count: int) -> bool:
        """Set expected_count; return True if already complete."""
        with self._lock:
            self.expected_count = expected_count
            return self.filled_count == self.expected_count

    def reset(self, chunk_size: int):
        """Reset buffer for reuse."""
        self.chunk_idx = -1
        self.haplotype_buf[:] = np.int8(-1)
        self.times[:] = 0
        self.starts[:] = 0
        self.ends[:] = 0
        self.focal_positions = [None] * chunk_size
        self.expected_count = -1
        self.filled_count = 0
        self._partial_gt = None
        self._partial_count = 0


class ChunkBufferPool:
    """Pool of pre-allocated ChunkBuffer objects."""

    def __init__(self, num_buffers: int, n_local: int, chunk_size: int):
        self._chunk_size = chunk_size
        self._free: queue.Queue = queue.Queue()
        for _ in range(num_buffers):
            self._free.put(ChunkBuffer(n_local, chunk_size))

    def acquire(self, chunk_idx: int, expected_count: int) -> ChunkBuffer:
        """Get a buffer from the pool (blocks if empty)."""
        buf = self._free.get()
        buf.chunk_idx = chunk_idx
        if expected_count >= 0:
            buf.expected_count = expected_count
        return buf

    def release(self, buf: ChunkBuffer):
        """Return a buffer to the pool after resetting it."""
        buf.reset(self._chunk_size)
        self._free.put(buf)


class ActiveChunkRegistry:
    """
    Thread-safe registry of in-flight ChunkBuffers keyed by chunk index.

    Handles concurrent access from multiple worker threads that may
    need the same chunk buffer simultaneously.
    """

    def __init__(self, pool: ChunkBufferPool):
        self._pool = pool
        self._active: dict[int, ChunkBuffer] = {}
        self._creating: set[int] = set()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def get_or_create(self, chunk_idx: int, expected_count: int) -> ChunkBuffer:
        with self._cond:
            if chunk_idx in self._active:
                return self._active[chunk_idx]
            # Wait if another thread is creating this chunk
            while chunk_idx in self._creating:
                self._cond.wait()
            # Re-check after waking
            if chunk_idx in self._active:
                return self._active[chunk_idx]
            self._creating.add(chunk_idx)

        # Acquire from pool outside the lock (may block)
        buf = self._pool.acquire(chunk_idx, expected_count)

        with self._cond:
            self._creating.discard(chunk_idx)
            self._active[chunk_idx] = buf
            self._cond.notify_all()
            return buf

    def remove(self, chunk_idx: int) -> ChunkBuffer:
        with self._lock:
            return self._active.pop(chunk_idx)

    def pop_remaining(self) -> list[ChunkBuffer]:
        """Remove and return all remaining active buffers."""
        with self._lock:
            bufs = list(self._active.values())
            self._active.clear()
            return bufs


@dataclass
class PipelineStats:
    """Timing statistics collected across all threads of an AncestorWriter."""

    # Worker-side (summed across all worker threads)
    worker_make_ancestor: float = 0.0  # time in C make_ancestor
    worker_make_min: float = float("inf")
    worker_make_max: float = 0.0
    worker_buf_fill: float = 0.0  # writing fragment + metadata into ChunkBuffer
    worker_registry_wait: float = 0.0  # waiting in get_or_create
    worker_count: int = 0  # total ancestors processed by workers

    # Writer-side (summed across all writer threads)
    writer_scatter: float = 0.0  # transpose + scatter into intermediate array
    writer_zarr: float = 0.0  # time writing to zarr
    writer_release: float = 0.0  # pool.release (reset + enqueue)
    writer_chunks: int = 0  # chunks flushed

    # Main-thread submit
    submit_put: float = 0.0  # blocked on work_queue.put
    submit_count: int = 0

    # Finalize
    finalize_worker_join: float = 0.0
    finalize_seal: float = 0.0
    finalize_writer_join: float = 0.0


class AncestorWriter:
    """
    Two-queue ancestor writing pipeline for a single interval.

    Worker threads pop from work_queue, call make_ancestor, and write
    into ChunkBuffers.  When a chunk is complete, it is pushed to
    write_queue for writer threads to flush to zarr via ``.blocks[]``.

    The zarr group and fixed arrays are created at construction time.
    Per-interval instances resize arrays incrementally and detect
    partial last chunks from previous intervals.

    Parameters
    ----------
    zarr_root : zarr.Group
        The zarr group (already containing fixed arrays).
    num_sites : int
        Number of inference sites.
    chunk_size : int
        Samples chunk size.
    n_ancestors : int
        Number of ancestors to be written in this interval.
    num_threads : int
        Number of worker threads (0 = synchronous).
    write_threads : int
        Number of writer threads.
    compressor : codec or None
        Compressor for zarr arrays.
    """

    def __init__(
        self,
        zarr_root,
        num_sites,
        chunk_size,
        n_ancestors,
        local_mask,
        final_positions,
        num_threads=0,
        write_threads=4,
        compressor=None,
    ):
        self._root = zarr_root
        self._num_sites = num_sites
        self._chunk_size = chunk_size
        self._n_ancestors = n_ancestors
        self._local_mask = local_mask  # maps local site idx → global site idx
        self._final_positions = final_positions  # global site positions
        self._n_local = len(local_mask)
        self._num_threads = num_threads
        self._write_threads = max(1, write_threads)
        self._compressor = compressor

        # Timing stats — aggregated from all threads
        self._stats = PipelineStats()
        self._stats_lock = threading.Lock()

        # Determine existing samples and partial chunk state
        existing_samples = self._root["call_genotype"].shape[1]
        self._base_index = existing_samples
        partial_count = existing_samples % chunk_size if existing_samples > 0 else 0

        # Resize zarr arrays for this interval's ancestors
        new_total = existing_samples + n_ancestors
        if n_ancestors > 0:
            self._root["call_genotype"].resize((num_sites, new_total, 1))
            self._root["sample_time"].resize((new_total,))
            self._root["sample_start_position"].resize((new_total,))
            self._root["sample_end_position"].resize((new_total,))

        # Compute buffer pool size
        max_queued = max(8 * max(num_threads, 1), 1)
        max_active_chunks = (max_queued + max(num_threads, 1)) // chunk_size + 1
        num_buffers = max(4, max_active_chunks + self._write_threads + 3)

        self._pool = ChunkBufferPool(num_buffers, self._n_local, chunk_size)
        self._registry = ActiveChunkRegistry(self._pool)

        # Load partial chunk from previous interval if needed
        if partial_count > 0:
            partial_ci = existing_samples // chunk_size
            buf = self._pool.acquire(partial_ci, expected_count=-1)
            start = partial_ci * chunk_size
            end = existing_samples
            # Store previous interval's data in global coordinates
            # (can't use haplotype_buf — different local_mask)
            buf._partial_gt = np.asarray(
                self._root["call_genotype"][:, start:end, :]
            ).copy()
            buf.times[:partial_count] = np.asarray(self._root["sample_time"][start:end])
            buf.starts[:partial_count] = np.asarray(
                self._root["sample_start_position"][start:end]
            )
            buf.ends[:partial_count] = np.asarray(
                self._root["sample_end_position"][start:end]
            )
            buf.focal_positions[:partial_count] = [None] * partial_count
            buf._partial_count = partial_count
            buf.filled_count = partial_count
            self._registry._active[partial_ci] = buf

        # Queues
        self._work_queue: queue.Queue = queue.Queue(maxsize=max(max_queued, 1))
        self._write_queue: queue.Queue = queue.Queue()

        # Focal positions collected by writer threads
        self._focal_by_chunk: dict[int, list] = {}
        self._focal_lock = threading.Lock()

        # Error propagation
        self._write_error = None
        self._worker_error = None

        # Start threads
        self._worker_threads = []
        n_workers = max(num_threads, 1)
        for _ in range(n_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._worker_threads.append(t)

        self._writer_threads_list = []
        for _ in range(self._write_threads):
            t = threading.Thread(target=self._writer_loop, daemon=True)
            t.start()
            self._writer_threads_list.append(t)

    def _check_errors(self):
        if self._write_error is not None:
            raise self._write_error
        if self._worker_error is not None:
            raise self._worker_error

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    def submit(
        self,
        ab,
        focal_sites,
        anc_time,
        ancestor_local_index,
    ):
        """Submit a work item for ancestor building."""
        self._check_errors()
        ancestor_index = self._base_index + ancestor_local_index
        chunk_idx = ancestor_index // self._chunk_size
        # Determine expected_count for this chunk
        total_samples = self._base_index + self._n_ancestors
        chunk_start = chunk_idx * self._chunk_size
        chunk_end = min(chunk_start + self._chunk_size, total_samples)
        expected_count = chunk_end - chunk_start

        item = _WorkItem(
            ab=ab,
            focal_sites=focal_sites,
            anc_time=anc_time,
            ancestor_index=ancestor_index,
            expected_count=expected_count,
        )
        t0 = _time.monotonic()
        self._work_queue.put(item)
        dt = _time.monotonic() - t0
        with self._stats_lock:
            self._stats.submit_put += dt
            self._stats.submit_count += 1

    def _worker_loop(self):
        """Worker thread: pop work items, build ancestors, fill chunk buffers."""
        local_array = None
        # Thread-local accumulators — merged into shared stats at exit
        t_make = 0.0
        t_make_min = float("inf")
        t_make_max = 0.0
        t_buf_fill = 0.0
        t_registry = 0.0
        n_processed = 0
        local_mask = self._local_mask
        final_positions = self._final_positions
        n_local = self._n_local
        try:
            while True:
                item = self._work_queue.get()
                if item is None:
                    break

                # Get or create thread-local output array
                if local_array is None or len(local_array) != n_local:
                    local_array = np.empty(n_local, dtype=np.int8)

                t0 = _time.monotonic()
                start_local, end_local = item.ab.make_ancestor(
                    item.focal_sites, local_array
                )
                dt = _time.monotonic() - t0
                t_make += dt
                if dt < t_make_min:
                    t_make_min = dt
                if dt > t_make_max:
                    t_make_max = dt

                # Map to global coordinates for metadata only
                start_site_idx = int(local_mask[start_local])
                end_site_idx = int(local_mask[end_local - 1]) + 1
                start_pos = int(final_positions[start_site_idx])
                end_pos = int(final_positions[end_site_idx - 1])
                focal_global = local_mask[item.focal_sites]
                focal_pos = final_positions[focal_global]

                # Get or create chunk buffer
                chunk_idx = item.ancestor_index // self._chunk_size
                slot = item.ancestor_index % self._chunk_size

                t0 = _time.monotonic()
                buf = self._registry.get_or_create(chunk_idx, item.expected_count)
                t_registry += _time.monotonic() - t0

                # Fill buffer slot — contiguous row copy (cache-friendly)
                t0 = _time.monotonic()
                buf.haplotype_buf[slot] = local_array
                buf.times[slot] = item.anc_time
                buf.starts[slot] = start_pos
                buf.ends[slot] = end_pos
                buf.focal_positions[slot] = np.asarray(focal_pos, dtype=np.int32)
                t_buf_fill += _time.monotonic() - t0

                n_processed += 1
                if buf.record_fill():
                    self._registry.remove(chunk_idx)
                    self._write_queue.put(buf)
        except Exception as e:
            self._worker_error = e
        finally:
            with self._stats_lock:
                self._stats.worker_make_ancestor += t_make
                if t_make_min < self._stats.worker_make_min:
                    self._stats.worker_make_min = t_make_min
                if t_make_max > self._stats.worker_make_max:
                    self._stats.worker_make_max = t_make_max
                self._stats.worker_buf_fill += t_buf_fill
                self._stats.worker_registry_wait += t_registry
                self._stats.worker_count += n_processed

    def _writer_loop(self):
        """Writer thread: pop completed chunks, transpose + scatter, write to zarr."""
        # Thread-local accumulators
        t_scatter = 0.0
        t_zarr = 0.0
        t_release = 0.0
        n_chunks = 0
        local_mask = self._local_mask
        num_sites = self._num_sites
        chunk_size = self._chunk_size
        # Pre-allocate intermediate array for full chunks (reused across iterations)
        intermediate = np.empty((num_sites, chunk_size, 1), dtype=np.int8)
        try:
            while True:
                buf = self._write_queue.get()
                if buf is None:
                    break

                ci = buf.chunk_idx
                n = buf.expected_count
                pc = buf._partial_count

                # Use pre-allocated buffer for full chunks, slice for partial
                if n == chunk_size:
                    out = intermediate
                else:
                    out = intermediate[:, :n, :]
                out[:] = np.int8(-1)

                # Scatter into intermediate array
                t0 = _time.monotonic()

                # Copy previous interval's data (already in global coords)
                if buf._partial_gt is not None and pc > 0:
                    out[:, :pc, :] = buf._partial_gt

                # Scatter new ancestors from haplotype_buf via local_mask
                # haplotype_buf[pc:n, :] has shape (n-pc, n_local)
                # Transposed: (n_local, n-pc)
                # out[local_mask, pc:n, 0] has shape (n_local, n-pc)
                if n > pc:
                    out[local_mask, pc:n, 0] = buf.haplotype_buf[pc:n, :].T
                dt_scatter = _time.monotonic() - t0
                t_scatter += dt_scatter

                # Write to zarr
                t0 = _time.monotonic()
                col_start = ci * chunk_size
                col_end = col_start + n
                self._root["call_genotype"][:, col_start:col_end, :] = out
                self._root["sample_time"][col_start:col_end] = buf.times[:n]
                self._root["sample_start_position"][col_start:col_end] = buf.starts[:n]
                self._root["sample_end_position"][col_start:col_end] = buf.ends[:n]
                dt_zarr = _time.monotonic() - t0
                t_zarr += dt_zarr

                # Collect focal positions
                focals = [buf.focal_positions[s] for s in range(n)]
                with self._focal_lock:
                    self._focal_by_chunk[ci] = focals

                t0 = _time.monotonic()
                self._pool.release(buf)
                dt_release = _time.monotonic() - t0
                t_release += dt_release

                n_chunks += 1
                gt_mb = num_sites * n / (1024 * 1024)
                logger.debug(
                    "Writer: flushed chunk %d (%d ancestors, %d partial, %.1fMiB) "
                    "scatter=%.3fs zarr=%.3fs release=%.3fs",
                    ci,
                    n,
                    pc,
                    gt_mb,
                    dt_scatter,
                    dt_zarr,
                    dt_release,
                )
        except Exception as e:
            self._write_error = e
        finally:
            if n_chunks > 0:
                logger.debug(
                    "Writer thread done: %d chunks, scatter=%.3fs zarr=%.3fs "
                    "release=%.3fs",
                    n_chunks,
                    t_scatter,
                    t_zarr,
                    t_release,
                )
            with self._stats_lock:
                self._stats.writer_scatter += t_scatter
                self._stats.writer_zarr += t_zarr
                self._stats.writer_release += t_release
                self._stats.writer_chunks += n_chunks

    def finalize(self):
        """
        Drain all queues, join threads, return focal positions for
        this interval's ancestors.

        Returns list of (np.ndarray | None) — one per ancestor in this
        interval.  None entries correspond to ancestors from a previous
        interval's partial chunk reload.
        """
        # Send sentinel to each worker and join
        t0 = _time.monotonic()
        for _ in self._worker_threads:
            self._work_queue.put(None)
        for t in self._worker_threads:
            t.join()
        self._stats.finalize_worker_join = _time.monotonic() - t0

        self._check_errors()

        # Seal any remaining active chunks (last chunk of interval)
        t0 = _time.monotonic()
        remaining = self._registry.pop_remaining()
        logger.debug("Finalize: %d remaining active chunk(s) to seal", len(remaining))
        for buf in remaining:
            total_samples = self._base_index + self._n_ancestors
            chunk_start = buf.chunk_idx * self._chunk_size
            chunk_end = min(chunk_start + self._chunk_size, total_samples)
            expected = chunk_end - chunk_start
            if buf.seal(expected):
                logger.debug(
                    "Finalize: sealed chunk %d (%d/%d ancestors) → write queue",
                    buf.chunk_idx,
                    buf.filled_count,
                    expected,
                )
                self._write_queue.put(buf)
            else:
                # Not yet complete — shouldn't happen if all workers
                # finished, but handle gracefully
                logger.warning(
                    "Chunk %d not complete at finalize: filled=%d expected=%d",
                    buf.chunk_idx,
                    buf.filled_count,
                    expected,
                )
                # Force it through
                buf.expected_count = buf.filled_count
                self._write_queue.put(buf)
        self._stats.finalize_seal = _time.monotonic() - t0

        # Send sentinel to each writer and join
        t0 = _time.monotonic()
        for _ in self._writer_threads_list:
            self._write_queue.put(None)
        for t in self._writer_threads_list:
            t.join()
        self._stats.finalize_writer_join = _time.monotonic() - t0

        self._check_errors()

        # Reassemble focal positions in order
        focal_list = []
        if self._focal_by_chunk:
            min_ci = min(self._focal_by_chunk)
            max_ci = max(self._focal_by_chunk)
            for ci in range(min_ci, max_ci + 1):
                if ci in self._focal_by_chunk:
                    focal_list.extend(self._focal_by_chunk[ci])

        # Strip out entries from previous interval's partial chunk
        # (those are None placeholders) and entries beyond our range
        result = []
        for i, fp in enumerate(focal_list):
            # Compute the global index of this entry
            ci_start = (
                min(self._focal_by_chunk) * self._chunk_size
                if self._focal_by_chunk
                else self._base_index
            )
            global_idx = ci_start + i
            if (
                global_idx >= self._base_index
                and global_idx < self._base_index + self._n_ancestors
                and fp is not None
            ):
                result.append(fp)

        return result


def setup_ancestor_zarr(
    num_sites,
    positions,
    alleles,
    anc_indices,
    seq_intervals,
    store=None,
    samples_chunk_size=1000,
    variants_chunk_size=1000,
    contig_id="1",
    contig_length=0,
    compressor=None,
):
    """
    Create and return a zarr Group with fixed (site-dimensioned) arrays
    and empty ancestor-dimensioned arrays, ready for AncestorWriter.
    """
    root = open_group(store)
    vchunks = variants_chunk_size
    ckw = {}
    if compressor is not None:
        ckw["compressor"] = compressor

    _arr(
        root,
        "variant_position",
        positions,
        ["variants"],
        chunks=(vchunks,),
        compressor=compressor,
    )

    out_alleles = _build_output_alleles(alleles, anc_indices, num_sites)
    va = root.create_array(
        "variant_allele",
        shape=out_alleles.shape,
        dtype=_VLEN_STR,
        chunks=(vchunks, out_alleles.shape[1]),
        **ckw,
    )
    va[:] = out_alleles
    va.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    _arr(
        root,
        "sequence_intervals",
        seq_intervals,
        ["intervals", "coords"],
        compressor=compressor,
    )

    _str_array(
        root,
        "contig_id",
        np.array([contig_id]),
        ["contigs"],
        compressor=compressor,
    )
    _arr(
        root,
        "contig_length",
        np.array([contig_length], dtype=np.int64),
        ["contigs"],
        compressor=compressor,
    )
    _arr(
        root,
        "variant_contig",
        np.zeros(num_sites, dtype=np.int8),
        ["variants"],
        chunks=(vchunks,),
        compressor=compressor,
    )

    # Ancestor-dimensioned arrays: start empty
    gt = root.create_array(
        "call_genotype",
        shape=(num_sites, 0, 1),
        dtype=np.int8,
        chunks=(variants_chunk_size, samples_chunk_size, 1),
        fill_value=np.int8(-1),
        **ckw,
    )
    gt.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

    for name in ("sample_time", "sample_start_position", "sample_end_position"):
        dt = np.float64 if name == "sample_time" else np.int32
        a = root.create_array(
            name,
            shape=(0,),
            dtype=dt,
            chunks=(samples_chunk_size,),
            **ckw,
        )
        a.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    return root


def finalize_ancestor_zarr(root, focal_positions_acc, compressor=None):
    """
    Write sample_id and sample_focal_positions to a completed ancestor
    zarr group.  Returns the Group.
    """
    num_anc = root["call_genotype"].shape[1]
    ckw = {}
    if compressor is not None:
        ckw["compressor"] = compressor

    ids = np.array([f"a{i}" for i in range(num_anc)])
    id_arr = root.create_array(
        "sample_id",
        shape=ids.shape,
        dtype=_VLEN_STR,
        **ckw,
    )
    id_arr[:] = ids
    id_arr.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    if num_anc > 0 and len(focal_positions_acc) > 0:
        max_focal = max(len(fp) for fp in focal_positions_acc)
        max_focal = max(max_focal, 1)
    else:
        max_focal = 1
    fp_data = np.full((num_anc, max_focal), -2, dtype=np.int32)
    for j, fp in enumerate(focal_positions_acc):
        fp_data[j, : len(fp)] = fp
    _arr(
        root,
        "sample_focal_positions",
        fp_data,
        ["samples", "focal_alleles"],
        compressor=compressor,
    )

    return root


# ---------------------------------------------------------------------------
# Lazy haplotype reading
# ---------------------------------------------------------------------------


class ChunkCache:
    """Unified LRU cache for sample chunks, keyed by (source_name, chunk_index).

    Uses a memory limit in bytes rather than entry count, so that cache
    utilisation adapts to actual chunk sizes.
    """

    def __init__(self, max_bytes: int):
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._cache: collections.OrderedDict[tuple[str, int], np.ndarray] = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, int], threading.Event] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: tuple[str, int]) -> np.ndarray | None:
        """Return cached chunk or None. Thread-safe."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: tuple[str, int], data: np.ndarray) -> None:
        """Insert chunk, evicting LRU entries until it fits. Thread-safe."""
        evicted_info = []
        with self._lock:
            if key in self._cache:
                return
            new_bytes = data.nbytes
            while self._cache and self._current_bytes + new_bytes > self._max_bytes:
                evict_key, evicted = self._cache.popitem(last=False)
                self._current_bytes -= evicted.nbytes
                evicted_info.append((evict_key, evicted.nbytes))
            self._cache[key] = data
            self._current_bytes += new_bytes
            total = self._current_bytes
            count = len(self._cache)
        # Log outside lock
        for ek, eb in evicted_info:
            logger.info(
                "Chunk cache evict: source=%s chunk=%d freed=%.1f MiB",
                ek[0],
                ek[1],
                eb / (1024 * 1024),
            )
        logger.info(
            "Chunk cache insert: source=%s chunk=%d"
            " size=%.1f MiB (total: %.1f MiB, %d entries)",
            key[0],
            key[1],
            data.nbytes / (1024 * 1024),
            total / (1024 * 1024),
            count,
        )

    def get_or_wait(self, key: tuple[str, int]) -> np.ndarray | None:
        """Check cache, coordinate concurrent loads.

        Returns cached data if present. If another thread is already loading
        this key, waits for it and returns the result. Returns None when the
        caller should perform the load and call ``finish_load``.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            if key in self._pending:
                event = self._pending[key]
                self._misses += 1
            else:
                self._pending[key] = threading.Event()
                self._misses += 1
                return None
        # Wait outside lock
        event.wait()
        return self.get(key)

    def finish_load(self, key: tuple[str, int], data: np.ndarray | None):
        """Insert loaded data and wake waiters. Call in finally block."""
        if data is not None:
            self.put(key, data)
        with self._lock:
            event = self._pending.pop(key, None)
        if event is not None:
            event.set()

    def evict_for(self, nbytes: int) -> None:
        """Pre-evict LRU entries to make room for an upcoming insert."""
        evicted_info = []
        with self._lock:
            while self._cache and self._current_bytes + nbytes > self._max_bytes:
                evict_key, evicted = self._cache.popitem(last=False)
                self._current_bytes -= evicted.nbytes
                evicted_info.append((evict_key, evicted.nbytes))
        for ek, eb in evicted_info:
            logger.info(
                "Chunk cache evict: source=%s chunk=%d freed=%.1f MiB",
                ek[0],
                ek[1],
                eb / (1024 * 1024),
            )

    @property
    def total_bytes(self) -> int:
        """Current memory usage of cached arrays in bytes."""
        return self._current_bytes


class _AlleleMap:
    """Fixed allele encoding: code 0 = ancestral, 1.. = derived.

    Built once from all sources' variant_allele arrays. Immutable after
    construction — no locking needed.
    """

    def __init__(self, num_sites: int, allele_table: list[list[str]]):
        self._num_sites = num_sites
        self._allele_table = allele_table  # allele_table[i] = [anc, der1, der2, ...]

    def lookup(self, ref_site_idx: int, allele_string: str) -> int:
        """Return the code for allele_string at site, or -1 if unknown."""
        table = self._allele_table[ref_site_idx]
        try:
            return table.index(allele_string)
        except ValueError:
            return -1

    def num_alleles(self, ref_site_idx: int) -> int:
        return len(self._allele_table[ref_site_idx])

    def num_alleles_array(self) -> np.ndarray:
        """Return (num_sites,) uint64 array of allele counts per site."""
        return np.array([len(t) for t in self._allele_table], dtype=np.uint64)

    def site_alleles_array(self) -> np.ndarray:
        """Return (num_sites, max_alleles) object array for extend_ts."""
        max_a = max(len(t) for t in self._allele_table) if self._allele_table else 2
        max_a = max(max_a, 2)
        result = np.full((self._num_sites, max_a), "", dtype=object)
        for i, table in enumerate(self._allele_table):
            for j, a in enumerate(table):
                result[i, j] = a
        return result


class VCZHaplotypeReader:
    """
    Reads and polarises haplotypes from a single VCZ store.

    Handles both ancestor and sample stores uniformly: positions are
    aligned to a reference coordinate system, and genotypes are polarised
    so that the ancestral allele is always 0.

    Uses an external ``ChunkCache`` shared across all sources to avoid
    repeated zarr reads when successive haplotypes fall in the same chunk.

    Parameters
    ----------
    store : zarr.Group or path
        The VCZ store to read from.
    positions : np.ndarray
        Reference variant positions (the coordinate system to align to).
    ancestral_alleles : np.ndarray
        Authoritative ancestral allele strings for polarisation.
    source_name : str
        Name used as the first element of the cache key.
    cache : ChunkCache
        Shared chunk cache instance.
    samples_selection : np.ndarray or None
        Optional subset of sample indices to use.
    """

    def __init__(
        self,
        store,
        positions: np.ndarray,
        ancestral_alleles: np.ndarray,
        *,
        source_name: str,
        cache: ChunkCache,
        samples_selection: np.ndarray | None = None,
        _allele_map: _AlleleMap | None = None,
    ):
        self._store = open_store(store)
        self._ref_positions = np.asarray(positions, dtype=np.int32)
        self._num_ref_sites = len(self._ref_positions)
        self._samples_selection = samples_selection
        self._source_name = source_name
        self._cache = cache
        self._allele_map = _allele_map

        # Build sample_id → column mapping
        raw_ids = self._store["sample_id"][:]
        if samples_selection is not None:
            raw_ids = raw_ids[samples_selection]
        self._sample_id_to_col = {str(sid): i for i, sid in enumerate(raw_ids.tolist())}

        # Determine sample chunk size from zarr array
        call_gt = self._store["call_genotype"]
        if hasattr(call_gt, "chunks") and call_gt.chunks is not None:
            self._sample_chunk_size = call_gt.chunks[1]
        else:
            self._sample_chunk_size = call_gt.shape[1]
        self._ploidy = call_gt.shape[2]
        self._num_store_samples = call_gt.shape[1]

        # Build position alignment map
        src_positions = np.asarray(self._store["variant_position"][:], dtype=np.int32)
        src_pos_to_idx = {}
        for i, p in enumerate(src_positions.tolist()):
            src_pos_to_idx[p] = i

        anc_idxs = []
        src_idxs = []
        for site_idx, pos in enumerate(self._ref_positions.tolist()):
            if pos in src_pos_to_idx:
                anc_idxs.append(site_idx)
                src_idxs.append(src_pos_to_idx[pos])
        self._position_map = (
            np.array(list(zip(anc_idxs, src_idxs)), dtype=np.int32).reshape(-1, 2)
            if anc_idxs
            else np.empty((0, 2), dtype=np.int32)
        )

        # Build ancestral allele index for polarisation using the
        # authoritative ancestral alleles from the ancestor store, not
        # the local store's variant_ancestral_allele field.
        num_src_sites = len(src_positions)

        # Map ref-site ancestral alleles to a position-keyed lookup
        anc_by_pos = {
            int(p): str(ancestral_alleles[i])
            for i, p in enumerate(self._ref_positions.tolist())
        }

        # Source allele strings (for encoding during load)
        self._src_alleles = np.asarray(self._store["variant_allele"][:])

        self._ancestral_allele_index = np.full(num_src_sites, -1, dtype=np.int8)
        for i in range(num_src_sites):
            pos = int(src_positions[i])
            anc_str = anc_by_pos.get(pos)
            if anc_str is None:
                continue
            for j, a in enumerate(self._src_alleles[i].tolist()):
                if a and str(a) == anc_str:
                    self._ancestral_allele_index[i] = j
                    break

        # Variant selection: which source variants match a reference position
        self._variant_select = np.zeros(num_src_sites, dtype=bool)
        if len(self._position_map) > 0:
            self._variant_select[self._position_map[:, 1]] = True
        self._num_selected = int(np.count_nonzero(self._variant_select))

        # Map: source variant index → row in cached chunk (-1 if not selected)
        self._src_var_to_selected_row = np.full(num_src_sites, -1, dtype=np.int32)
        self._src_var_to_selected_row[self._variant_select] = np.arange(
            self._num_selected, dtype=np.int32
        )

        # Map: selected row index → ref site index
        # In source-variant order (ascending src_site_idx)
        if len(self._position_map) > 0:
            order = np.argsort(self._position_map[:, 1])
            self._selected_to_ref_idx = self._position_map[order, 0]
        else:
            self._selected_to_ref_idx = np.empty(0, dtype=np.int32)

        # Zarr variant chunk size for chunk-wise iteration
        self._variant_chunk_size = (
            call_gt.chunks[0]
            if hasattr(call_gt, "chunks") and call_gt.chunks is not None
            else call_gt.shape[0]
        )

        # Precompute which variant chunks contain selected sites
        vc_size = self._variant_chunk_size
        n_vc = math.ceil(num_src_sites / vc_size) if num_src_sites > 0 else 0
        self._needed_vc = []  # [(vc_idx, local_indices, out_rows), ...]
        for vc_idx in range(n_vc):
            vc_start = vc_idx * vc_size
            vc_end = min(vc_start + vc_size, num_src_sites)
            local_select = self._variant_select[vc_start:vc_end]
            if not np.any(local_select):
                continue
            local_indices = np.where(local_select)[0]
            out_rows = self._src_var_to_selected_row[vc_start + local_indices]
            self._needed_vc.append((vc_idx, local_indices, out_rows))

        # Allele remap table — populated by _precompute_allele_remap()
        self._allele_remap = None

        # Timing accumulators
        self._stats_chunks_loaded = 0
        self._stats_zarr_time = 0.0
        self._stats_encode_time = 0.0
        self._stats_zarr_chunks_read = 0

    @property
    def chunk_bytes(self) -> int:
        """Byte size of one cached sample chunk (selected variants only)."""
        return self._num_selected * self._sample_chunk_size * self._ploidy * 1

    def _precompute_allele_remap(self):
        """Build a (num_selected, max_src_alleles) int8 remap table.

        Each entry maps a source allele index to the unified allele
        code via ``_allele_map.lookup()``.  Called once after
        ``_allele_map`` is assigned; removes all string operations
        from the hot cache-miss path.
        """
        if self._allele_map is None or self._num_selected == 0:
            return
        selected_src = np.where(self._variant_select)[0]
        max_a = self._src_alleles.shape[1]
        remap = np.full((self._num_selected, max_a), -1, dtype=np.int8)
        for row, src_idx in enumerate(selected_src):
            ref_idx = int(self._selected_to_ref_idx[row])
            for j in range(max_a):
                a = str(self._src_alleles[src_idx, j])
                if a:
                    remap[row, j] = self._allele_map.lookup(ref_idx, a)
        self._allele_remap = remap

    def _do_load_chunk(self, chunk_idx: int) -> np.ndarray:
        """Perform zarr I/O and encoding for a single sample chunk.

        Returns array of shape ``(num_selected, chunk_samples, ploidy)``.
        """
        t_start = _time.monotonic()
        call_gt = self._store["call_genotype"]
        sc_start = chunk_idx * self._sample_chunk_size
        chunk_samples = min(
            self._sample_chunk_size,
            self._num_store_samples - sc_start,
        )

        result = np.full(
            (self._num_selected, chunk_samples, self._ploidy),
            -1,
            dtype=np.int8,
        )

        t_zarr = 0.0
        t_encode = 0.0
        n_zarr = 0
        n_sites = 0

        for vc_idx, local_indices, out_rows in self._needed_vc:
            t0 = _time.monotonic()
            block = np.asarray(call_gt.blocks[vc_idx, chunk_idx], dtype=np.int8)
            t_zarr += _time.monotonic() - t0
            n_zarr += 1

            t0 = _time.monotonic()
            raw = block[local_indices]  # (n, chunk_samples, ploidy)
            n = len(out_rows)
            n_sites += n

            if self._allele_remap is not None:
                remap = self._allele_remap[out_rows]
                safe = np.clip(raw, 0, remap.shape[1] - 1)
                encoded = remap[np.arange(n)[:, None, None], safe]
                result[out_rows] = np.where(raw < 0, np.int8(-1), encoded)
            else:
                result[out_rows] = raw
            t_encode += _time.monotonic() - t0

        t_total = _time.monotonic() - t_start

        self._stats_chunks_loaded += 1
        self._stats_zarr_time += t_zarr
        self._stats_encode_time += t_encode
        self._stats_zarr_chunks_read += n_zarr

        logger.info(
            "Cache populate source=%s chunk=%d: %d sites from "
            "%d zarr chunks zarr=%.3fs encode=%.3fs total=%.3fs",
            self._source_name,
            chunk_idx,
            n_sites,
            n_zarr,
            t_zarr,
            t_encode,
            t_total,
        )
        return result

    def _load_sample_chunk(self, chunk_idx: int) -> np.ndarray:
        """Load a sample chunk, selecting only matched variants.

        On cache miss, iterates over zarr variant chunks, skipping chunks
        with no selected variants. When ``_allele_remap`` is set, allele
        encoding is performed at load time via vectorized numpy remap.
        Returns array of shape ``(num_selected, chunk_samples, ploidy)``.

        Thread-safe via the shared ``ChunkCache``.  Concurrent misses for
        the same chunk are deduplicated — only one thread loads from zarr.
        The returned numpy array remains valid even if later evicted from
        the cache, because Python's reference counting keeps it alive.
        """
        key = (self._source_name, chunk_idx)
        data = self._cache.get_or_wait(key)
        if data is not None:
            return data
        # Pre-evict so peak memory stays within the cache limit
        sc_start = chunk_idx * self._sample_chunk_size
        chunk_samples = min(
            self._sample_chunk_size,
            self._num_store_samples - sc_start,
        )
        estimated_bytes = self._num_selected * chunk_samples * self._ploidy
        self._cache.evict_for(estimated_bytes)
        # We are the loader
        result = None
        try:
            result = self._do_load_chunk(chunk_idx)
        finally:
            self._cache.finish_load(key, result)
        return result

    def read_haplotype(self, sample_id: str, ploidy_index: int = 0) -> np.ndarray:
        """
        Read and polarise a single haplotype.

        Returns (num_ref_sites,) int8 array:
        0=ancestral, 1=derived, -1=missing.
        """
        col = self._sample_id_to_col[sample_id]

        # Map to store column (respecting samples_selection)
        if self._samples_selection is not None:
            store_col = int(self._samples_selection[col])
        else:
            store_col = col

        # Determine chunk and offset
        chunk_idx = store_col // self._sample_chunk_size
        chunk_offset = store_col % self._sample_chunk_size

        # Read from cache — shape (num_selected, chunk_samples, ploidy)
        chunk_data = self._load_sample_chunk(chunk_idx)
        encoded_col = chunk_data[:, chunk_offset, ploidy_index]  # (num_selected,)

        hap = np.full(self._num_ref_sites, np.int8(-1), dtype=np.int8)
        if self._num_selected > 0:
            if self._allele_map is not None:
                # Already encoded at load time
                hap[self._selected_to_ref_idx] = encoded_col
            else:
                # Fallback: polarise from raw genotypes
                src_idxs = self._position_map[:, 1]
                anc_idxs = self._position_map[:, 0]
                selected_rows = self._src_var_to_selected_row[src_idxs]
                raw = encoded_col[selected_rows]
                ai = self._ancestral_allele_index[src_idxs]
                valid = ai >= 0
                raw_v, ai_v, anc_v = raw[valid], ai[valid], anc_idxs[valid]
                encoded = np.where(
                    raw_v < 0,
                    np.int8(-1),
                    np.where(raw_v == ai_v, np.int8(0), np.int8(1)),
                )
                hap[anc_v] = encoded
        return hap


class HaplotypeReader:
    """
    Facade that routes haplotype reads to per-source VCZHaplotypeReaders.

    All sources (including ancestors) are treated uniformly as VCZ stores.
    """

    def __init__(
        self,
        sources: dict,
        positions: np.ndarray,
        ancestral_alleles: np.ndarray,
        cache_size_mb: int = 256,
    ):
        """
        Parameters
        ----------
        sources : dict[str, Source]
            All sources to read from (including ancestor sources).
        positions : np.ndarray
            Reference variant positions (the coordinate system).
        ancestral_alleles : np.ndarray
            (num_sites,) array of ancestral allele strings, one per
            reference position.  Authoritative source for polarisation.
        cache_size_mb : int
            Cache memory limit in MiB.
        """
        self._positions = np.asarray(positions, dtype=np.int32)
        self._ancestral_alleles = ancestral_alleles
        self._sources = sources
        max_bytes = cache_size_mb * 1024 * 1024

        # Eagerly create all source readers so we can validate sizes
        # (allele map not yet available — assigned after building it)
        self._readers: dict[str, VCZHaplotypeReader] = {}
        for name, source in sources.items():
            store = open_store(source.path)
            samples_selection = resolve_samples_selection(store, source.samples)
            reader = VCZHaplotypeReader(
                store,
                self._positions,
                self._ancestral_alleles,
                source_name=name,
                cache=None,  # placeholder, set below
                samples_selection=samples_selection,
            )
            self._readers[name] = reader

        # Build fixed allele map from all sources.
        # Iterate sources in config order — lowest-index source wins
        # the lowest derived allele code.
        allele_table: list[list[str]] = [[str(a)] for a in ancestral_alleles]
        for reader in self._readers.values():
            for ref_idx, src_idx in reader._position_map:
                ref_idx = int(ref_idx)
                site_alleles = reader._src_alleles[src_idx]
                allele_list = (
                    site_alleles.tolist()
                    if hasattr(site_alleles, "tolist")
                    else site_alleles
                )
                for a in allele_list:
                    a_str = str(a)
                    if a_str and a_str not in allele_table[ref_idx]:
                        allele_table[ref_idx].append(a_str)

        self._allele_map = _AlleleMap(len(ancestral_alleles), allele_table)

        # Assign the allele map to all readers and precompute remap tables
        for reader in self._readers.values():
            reader._allele_map = self._allele_map
            reader._precompute_allele_remap()

        # Validate that the cache can fit at least one chunk from every source
        max_chunk = 0
        max_chunk_source = None
        for name, reader in self._readers.items():
            cb = reader.chunk_bytes
            if cb > max_chunk:
                max_chunk = cb
                max_chunk_source = name
            chunks_fit = max_bytes // cb if cb > 0 else float("inf")
            logger.info(
                "Cache: source=%s chunk_size=%.1f MiB, fits %d chunks",
                name,
                cb / (1024 * 1024),
                chunks_fit,
            )
            if cb > 0 and chunks_fit < 2:
                import warnings

                warnings.warn(
                    f"Cache can hold fewer than 2 chunks from source "
                    f"'{name}' (chunk={cb / (1024 * 1024):.1f} MiB, "
                    f"cache={cache_size_mb} MiB). "
                    f"Consider increasing --cache-size.",
                    stacklevel=2,
                )

        if max_chunk > 0 and max_bytes < max_chunk:
            raise ValueError(
                f"Cache size {cache_size_mb} MiB cannot fit a single chunk "
                f"from source '{max_chunk_source}' "
                f"({max_chunk / (1024 * 1024):.1f} MiB). "
                f"Increase --cache-size to at least "
                f"{max_chunk / (1024 * 1024):.0f} MiB."
            )

        # Now create the shared cache and assign to all readers
        self._cache = ChunkCache(max_bytes=max_bytes)
        for reader in self._readers.values():
            reader._cache = self._cache

        self._readers_lock = threading.Lock()

    def _get_reader(self, source_name: str) -> VCZHaplotypeReader:
        if source_name in self._readers:
            return self._readers[source_name]

        with self._readers_lock:
            if source_name in self._readers:
                return self._readers[source_name]

            source = self._sources[source_name]
            store = open_store(source.path)
            samples_selection = resolve_samples_selection(store, source.samples)
            reader = VCZHaplotypeReader(
                store,
                self._positions,
                self._ancestral_alleles,
                source_name=source_name,
                cache=self._cache,
                samples_selection=samples_selection,
                _allele_map=self._allele_map,
            )
            self._readers[source_name] = reader
            return reader

    def read_haplotype(self, job) -> np.ndarray:
        """
        Read and encode a single haplotype for the given MatchJob.

        Returns (num_ref_sites,) int8 array:
        0=ancestral, 1=derived, -1=missing.
        """
        reader = self._get_reader(job.source)
        return reader.read_haplotype(job.sample_id, job.ploidy_index)

    def get_num_alleles(self) -> np.ndarray:
        """Return (num_sites,) uint64 array of allele counts per site."""
        return self._allele_map.num_alleles_array()

    def get_site_alleles(self) -> np.ndarray:
        """Return (num_sites, max_alleles) object array of allele strings.

        Allele codes are consistent with the encoding used by
        ``read_haplotype``: index 0 is always the ancestral allele,
        index 1 is the first derived allele seen across all sources, etc.
        """
        return self._allele_map.site_alleles_array()


def write_empty_ancestor_vcz(
    seq_intervals,
    store=None,
    contig_id="1",
    contig_length=0,
    compressor=None,
):
    """Return an ancestor VCZ with zero sites and zero ancestors.

    Parameters
    ----------
    seq_intervals : array-like
        Sequence interval array to write.
    store : str, Path, or None
        Backing store — ``None`` for in-memory, or a filesystem path.
    contig_id : str
        Contig name for vcztools compatibility.
    contig_length : int
        Contig length in base pairs.
    compressor : numcodecs codec or None
        Compressor for zarr arrays.
    """
    root = open_group(store)
    ckw = {}
    if compressor is not None:
        ckw["compressor"] = compressor

    g = root.create_array(
        "call_genotype",
        shape=(0, 0, 1),
        dtype=np.int8,
        **ckw,
    )
    g.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

    _arr(
        root,
        "variant_position",
        np.zeros(0, dtype=np.int32),
        ["variants"],
        compressor=compressor,
    )

    va = root.create_array("variant_allele", shape=(0, 2), dtype=_VLEN_STR, **ckw)
    va.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    ids = root.create_array("sample_id", shape=(0,), dtype=_VLEN_STR, **ckw)
    ids.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    _arr(
        root,
        "sample_time",
        np.zeros(0, dtype=np.float64),
        ["samples"],
        compressor=compressor,
    )
    _arr(
        root,
        "sample_start_position",
        np.zeros(0, dtype=np.int32),
        ["samples"],
        compressor=compressor,
    )
    _arr(
        root,
        "sample_end_position",
        np.zeros(0, dtype=np.int32),
        ["samples"],
        compressor=compressor,
    )
    _arr(
        root,
        "sample_focal_positions",
        np.zeros((0, 1), dtype=np.int32),
        ["samples", "focal_alleles"],
        compressor=compressor,
    )
    _arr(
        root,
        "sequence_intervals",
        seq_intervals,
        ["intervals", "coords"],
        compressor=compressor,
    )

    # Contig metadata — required for vcztools compatibility
    _str_array(
        root, "contig_id", np.array([contig_id]), ["contigs"], compressor=compressor
    )
    _arr(
        root,
        "contig_length",
        np.array([contig_length], dtype=np.int64),
        ["contigs"],
        compressor=compressor,
    )
    _arr(
        root,
        "variant_contig",
        np.zeros(0, dtype=np.int8),
        ["variants"],
        compressor=compressor,
    )

    return root
