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
    to a numpy array.  Used to resolve per-source metadata like site_mask,
    sample_mask, and sample_time.

``sequence_length``
    Read the contig length from a VCZ store (``contig_length[0]``).

``num_contigs``
    Return the number of contigs declared in a VCZ store.

``AncestorWriter``
    Streams ancestors to a zarr Group, flushing by sample-chunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.core.dtype.npy.string import VariableLengthUTF8

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


def iter_genotypes(store, site_indices, sample_include=None):
    """
    Yield ``(num_haplotypes,)`` int8 genotype rows for the requested site
    indices, one row per index in the order given.

    Reads ``call_genotype`` chunk-by-chunk so that at most one zarr chunk
    is in memory at a time.  Sample selection and ploidy flattening are
    applied transparently.

    Parameters
    ----------
    store : zarr.Group
        VCZ store containing ``call_genotype``.
    site_indices : array-like of int
        Global row indices into ``call_genotype`` (must be non-decreasing).
    sample_include : array-like of bool or None
        Boolean mask (length ``num_samples``) where True means keep this
        sample.  ``None`` keeps all samples.

    Yields
    ------
    np.ndarray
        ``(num_haplotypes,)`` int8 array — one flattened genotype row per
        requested site index.
    """
    call_gt = store["call_genotype"]
    site_indices = np.asarray(site_indices, dtype=np.int64)

    if len(site_indices) == 0:
        return

    if sample_include is not None:
        sample_include = np.asarray(sample_include, dtype=bool)

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
            yield cached_flat[local].copy()
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

    def add_ancestor(self, time, haplotype, focal_positions, start_pos, end_pos):
        self._buffer.append(
            (float(time), haplotype.copy(), int(start_pos), int(end_pos))
        )
        self._focal_positions_acc.append(
            np.asarray(focal_positions, dtype=np.int32).copy()
        )
        if len(self._buffer) >= self._chunk_size:
            self._flush()

    # -----------------------------------------------------------------

    def _flush(self):
        if not self._buffer:
            return

        n = len(self._buffer)
        ns = self._num_sites

        gt_chunk = np.full((ns, n, 1), np.int8(-1), dtype=np.int8)
        times = np.empty(n, dtype=np.float64)
        starts = np.empty(n, dtype=np.int32)
        ends = np.empty(n, dtype=np.int32)

        for j, (t, hap, s, e) in enumerate(self._buffer):
            gt_chunk[:, j, 0] = hap
            times[j] = t
            starts[j] = s
            ends[j] = e

        self._root["call_genotype"].append(gt_chunk, axis=1)
        self._root["sample_time"].append(times)
        self._root["sample_start_position"].append(starts)
        self._root["sample_end_position"].append(ends)

        self._num_flushed += n
        self._buffer.clear()

    # -----------------------------------------------------------------

    def finalize(self):
        """Flush remaining buffer and write final arrays.  Returns the Group."""
        self._flush()
        num_anc = self._num_flushed

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

        return self._root


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
