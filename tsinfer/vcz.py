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

import logging
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
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
    from vcztools.samples import parse_samples

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
    from vcztools.retrieval import variant_chunk_iter

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
    try:
        chunk_size = call_gt.chunks[0]
    except (AttributeError, TypeError, IndexError):
        chunk_size = call_gt.shape[0]

    # Walk through site_indices in order, loading each chunk at most once
    # for a run of consecutive indices that fall inside it.
    n = len(site_indices)
    i = 0
    cached_chunk_id = -1
    cached_flat = None

    while i < n:
        cid = int(site_indices[i]) // chunk_size

        if cid != cached_chunk_id:
            chunk_start = cid * chunk_size
            chunk_end = min(chunk_start + chunk_size, call_gt.shape[0])
            chunk_data = np.asarray(call_gt[chunk_start:chunk_end], dtype=np.int8)
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


def _arr(group, name, data, dims):
    a = group.create_array(name, data=data)
    a.attrs["_ARRAY_DIMENSIONS"] = dims
    return a


def _str_array(group, name, data, dims):
    data = np.asarray(data)
    a = group.create_array(name, shape=data.shape, dtype=_VLEN_STR)
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
# Ancestor writing
# ---------------------------------------------------------------------------


class AncestorWriter:
    """
    Streams ancestors to a zarr Group, flushing by sample-chunk.

    Fixed (site-dimensioned) arrays are written at construction time.
    Ancestor-dimensioned arrays (``call_genotype``, ``sample_time``,
    ``sample_start_position``, ``sample_end_position``) are appended
    in chunks of *chunk_size* ancestors.  ``sample_focal_positions`` and
    ``sample_id`` are written once at :meth:`finalize` because the
    second dimension of focal_positions is only known after all
    ancestors have been generated.

    Parameters
    ----------
    num_sites : int
        Number of inference sites (variants axis).
    positions, alleles, anc_indices, seq_intervals :
        Site-dimensioned data written once at init.
    store : str, Path, or None
        Backing store — ``None`` for in-memory, or a filesystem path.
    samples_chunk_size : int
        Number of ancestors to buffer before flushing to zarr.
    variants_chunk_size : int
        Chunk size along the variants axis for ``call_genotype``.
    """

    def __init__(
        self,
        num_sites,
        positions,
        alleles,
        anc_indices,
        seq_intervals,
        store=None,
        samples_chunk_size=1000,
        variants_chunk_size=1000,
    ):
        self._num_sites = num_sites
        self._chunk_size = samples_chunk_size
        self._buffer = []
        self._focal_positions_acc = []
        self._num_flushed = 0
        # For out-of-order insertion: pending ancestors keyed by index,
        # drained into _buffer in contiguous order.
        self._pending = {}
        self._next_index = 0

        # --- Build the zarr group and write fixed arrays ---
        self._root = open_group(store)

        _arr(self._root, "variant_position", positions, ["variants"])

        out_alleles = _build_output_alleles(alleles, anc_indices, num_sites)
        va = self._root.create_array(
            "variant_allele", shape=out_alleles.shape, dtype=_VLEN_STR
        )
        va[:] = out_alleles
        va.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

        _arr(
            self._root,
            "sequence_intervals",
            seq_intervals,
            ["intervals", "coords"],
        )

        # --- Ancestor-dimensioned arrays: start empty, append by chunk ---
        gt = self._root.create_array(
            "call_genotype",
            shape=(num_sites, 0, 1),
            dtype=np.int8,
            chunks=(variants_chunk_size, samples_chunk_size, 1),
        )
        gt.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

        for name in ("sample_time", "sample_start_position", "sample_end_position"):
            dt = np.float64 if name == "sample_time" else np.int32
            a = self._root.create_array(
                name,
                shape=(0,),
                dtype=dt,
                chunks=(samples_chunk_size,),
            )
            a.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    # -----------------------------------------------------------------

    def add_ancestor(self, ancestor):
        """
        Add an :class:`~tsinfer.ancestors.Ancestor` to the writer.

        Ancestors may arrive out of order (when built in parallel).
        They are buffered by index and drained into the write buffer
        in sequential order.
        """
        self._pending[ancestor.index] = ancestor
        drained = 0
        while self._next_index in self._pending:
            anc = self._pending.pop(self._next_index)
            self._buffer.append(anc)
            self._focal_positions_acc.append(
                np.asarray(anc.focal_positions, dtype=np.int32)
            )
            self._next_index += 1
            drained += 1
            if len(self._buffer) >= self._chunk_size:
                self._flush()
        logger.debug(
            "add_ancestor idx=%d: drained=%d pending=%d buffer=%d "
            "focal_acc=%d flushed=%d",
            ancestor.index,
            drained,
            len(self._pending),
            len(self._buffer),
            len(self._focal_positions_acc),
            self._num_flushed,
        )

    # -----------------------------------------------------------------

    def _flush(self):
        if not self._buffer:
            return

        t0 = _time.monotonic()
        n = len(self._buffer)
        ns = self._num_sites

        gt_chunk = np.full((ns, n, 1), np.int8(-1), dtype=np.int8)
        times = np.empty(n, dtype=np.float64)
        starts = np.empty(n, dtype=np.int32)
        ends = np.empty(n, dtype=np.int32)

        for j, anc in enumerate(self._buffer):
            gt_chunk[anc.start_site_idx : anc.end_site_idx, j, 0] = anc.haplotype
            times[j] = anc.time
            starts[j] = anc.start_position
            ends[j] = anc.end_position

        self._root["call_genotype"].append(gt_chunk, axis=1)
        self._root["sample_time"].append(times)
        self._root["sample_start_position"].append(starts)
        self._root["sample_end_position"].append(ends)

        self._num_flushed += n
        self._buffer.clear()
        elapsed = _time.monotonic() - t0
        gt_chunk_mb = gt_chunk.nbytes / (1024 * 1024)
        focal_mb = sum(fp.nbytes for fp in self._focal_positions_acc) / (1024 * 1024)
        logger.debug(
            "Flushed chunk: %d ancestors (%d total) in %.3fs; "
            "gt_chunk=%.1fMiB focal_acc=%.1fMiB (%d entries)",
            n,
            self._num_flushed,
            elapsed,
            gt_chunk_mb,
            focal_mb,
            len(self._focal_positions_acc),
        )

    # -----------------------------------------------------------------

    def finalize(self):
        """Flush remaining buffer and write final arrays.  Returns the Group."""
        self._flush()
        num_anc = self._num_flushed
        focal_mb = sum(fp.nbytes for fp in self._focal_positions_acc) / (1024 * 1024)
        logger.debug(
            "Finalizing AncestorWriter: %d ancestors total; "
            "focal_acc=%.1fMiB (%d entries) pending=%d",
            num_anc,
            focal_mb,
            len(self._focal_positions_acc),
            len(self._pending),
        )

        t0 = _time.monotonic()

        # sample_id — trivial, write once
        ids = np.array([f"ancestor_{i}" for i in range(num_anc)])
        id_arr = self._root.create_array(
            "sample_id",
            shape=ids.shape,
            dtype=_VLEN_STR,
        )
        id_arr[:] = ids
        id_arr.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

        # sample_focal_positions — variable-width, write once
        if num_anc > 0:
            max_focal = max(len(fp) for fp in self._focal_positions_acc)
            max_focal = max(max_focal, 1)
        else:
            max_focal = 1
        fp_data = np.full((num_anc, max_focal), -2, dtype=np.int32)
        for j, fp in enumerate(self._focal_positions_acc):
            fp_data[j, : len(fp)] = fp
        _arr(
            self._root,
            "sample_focal_positions",
            fp_data,
            ["samples", "focal_alleles"],
        )

        elapsed = _time.monotonic() - t0
        logger.debug("Finalize complete in %.3fs", elapsed)
        return self._root


# ---------------------------------------------------------------------------
# Lazy haplotype reading
# ---------------------------------------------------------------------------


@dataclass
class _SourceContext:
    """Cached per-source state for HaplotypeReader."""

    store: zarr.Group
    position_map: (
        np.ndarray
    )  # (n_aligned, 2) int32 — (ancestor_site_idx, source_site_idx)
    ancestral_allele_index: np.ndarray  # (n_source_sites,) int8
    sample_id_to_col: dict  # sample_id → column index
    samples_selection: np.ndarray | None


class HaplotypeReader:
    """Lazily reads and encodes individual haplotypes from VCZ stores."""

    def __init__(self, ancestor_store_path, sources, positions):
        """
        Parameters
        ----------
        ancestor_store_path : path or zarr.Group
            Path to ancestor VCZ (or in-memory Group).
        sources : dict[str, Source]
            cfg.sources mapping (name → Source).
        positions : np.ndarray
            Ancestor variant positions (the reference coordinate system).
        """
        self._anc_store = open_store(ancestor_store_path)
        self._positions = np.asarray(positions, dtype=np.int32)
        self._sources = sources
        self._source_contexts: dict[str, _SourceContext] = {}
        self._num_sites = len(positions)

        # Build ancestor sample_id → column mapping
        anc_ids = [str(x) for x in self._anc_store["sample_id"][:].tolist()]
        self._anc_id_to_col = {sid: i for i, sid in enumerate(anc_ids)}

    def _get_source_context(self, source_name: str) -> _SourceContext:
        if source_name in self._source_contexts:
            return self._source_contexts[source_name]

        source = self._sources[source_name]
        store = open_store(source.path)

        # Build sample_id → column mapping (respecting samples_selection)
        samples_selection = resolve_samples_selection(store, source.samples)
        raw_ids = store["sample_id"][:]
        if samples_selection is not None:
            raw_ids = raw_ids[samples_selection]
        sample_id_to_col = {str(sid): i for i, sid in enumerate(raw_ids.tolist())}

        # Build position alignment map
        src_positions = np.asarray(store["variant_position"][:], dtype=np.int32)
        src_pos_to_idx = {}
        for i, p in enumerate(src_positions.tolist()):
            src_pos_to_idx[p] = i

        anc_idxs = []
        src_idxs = []
        for site_idx, pos in enumerate(self._positions.tolist()):
            if pos in src_pos_to_idx:
                anc_idxs.append(site_idx)
                src_idxs.append(src_pos_to_idx[pos])
        position_map = (
            np.array(list(zip(anc_idxs, src_idxs)), dtype=np.int32).reshape(-1, 2)
            if anc_idxs
            else np.empty((0, 2), dtype=np.int32)
        )

        # Build ancestral allele index
        src_alleles = np.asarray(store["variant_allele"][:])
        num_src_sites = len(src_positions)
        if "variant_ancestral_allele" in store:
            src_anc_state = np.asarray(store["variant_ancestral_allele"][:])
        else:
            src_anc_state = np.array([str(a[0]) for a in src_alleles], dtype=object)

        ancestral_allele_index = np.full(num_src_sites, -1, dtype=np.int8)
        for i in range(num_src_sites):
            for j, a in enumerate(src_alleles[i].tolist()):
                if a and str(a) == str(src_anc_state[i]):
                    ancestral_allele_index[i] = j
                    break

        ctx = _SourceContext(
            store=store,
            position_map=position_map,
            ancestral_allele_index=ancestral_allele_index,
            sample_id_to_col=sample_id_to_col,
            samples_selection=samples_selection,
        )
        self._source_contexts[source_name] = ctx
        return ctx

    def read_haplotype(self, job) -> np.ndarray:
        """
        Read and encode a single haplotype for the given MatchJob.

        Returns (num_ancestor_sites,) int8 array:
        0=ancestral, 1=derived, -1=missing.
        """
        if job.source == "ancestors" and job.sample_id == "virtual_root":
            return np.zeros(self._num_sites, dtype=np.int8)

        if job.source == "ancestors":
            anc_col = self._anc_id_to_col[job.sample_id]
            return np.asarray(
                self._anc_store["call_genotype"][:, anc_col, 0], dtype=np.int8
            )

        # Sample source
        ctx = self._get_source_context(job.source)
        col = ctx.sample_id_to_col[job.sample_id]

        # Read from the underlying store, applying samples_selection
        if ctx.samples_selection is not None:
            # col is relative to the filtered set; map back to store column
            store_col = int(ctx.samples_selection[col])
        else:
            store_col = col

        raw_gt = np.asarray(
            ctx.store["call_genotype"][:, store_col, job.ploidy_index], dtype=np.int8
        )

        hap = np.full(self._num_sites, np.int8(-1), dtype=np.int8)

        if len(ctx.position_map) == 0:
            return hap

        anc_idxs = ctx.position_map[:, 0]
        src_idxs = ctx.position_map[:, 1]
        raw = raw_gt[src_idxs]
        ai = ctx.ancestral_allele_index[src_idxs]
        valid = ai >= 0
        raw_v = raw[valid]
        ai_v = ai[valid]
        anc_v = anc_idxs[valid]
        encoded = np.where(
            raw_v < 0,
            np.int8(-1),
            np.where(raw_v == ai_v, np.int8(0), np.int8(1)),
        )
        hap[anc_v] = encoded
        return hap


def write_empty_ancestor_vcz(seq_intervals, store=None):
    """Return an ancestor VCZ with zero sites and zero ancestors.

    Parameters
    ----------
    seq_intervals : array-like
        Sequence interval array to write.
    store : str, Path, or None
        Backing store — ``None`` for in-memory, or a filesystem path.
    """
    root = open_group(store)

    g = root.create_array(
        "call_genotype",
        shape=(0, 0, 1),
        dtype=np.int8,
    )
    g.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

    _arr(root, "variant_position", np.zeros(0, dtype=np.int32), ["variants"])

    va = root.create_array("variant_allele", shape=(0, 2), dtype=_VLEN_STR)
    va.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    ids = root.create_array("sample_id", shape=(0,), dtype=_VLEN_STR)
    ids.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    _arr(root, "sample_time", np.zeros(0, dtype=np.float64), ["samples"])
    _arr(
        root,
        "sample_start_position",
        np.zeros(0, dtype=np.int32),
        ["samples"],
    )
    _arr(
        root,
        "sample_end_position",
        np.zeros(0, dtype=np.int32),
        ["samples"],
    )
    _arr(
        root,
        "sample_focal_positions",
        np.zeros((0, 1), dtype=np.int32),
        ["samples", "focal_alleles"],
    )
    _arr(root, "sequence_intervals", seq_intervals, ["intervals", "coords"])

    return root
