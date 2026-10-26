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

``resolve_contig``
    Return the single contig shared by several VCZ stores, checking consistency.

``contig_variant_rows``
    Return the slice of variant rows in a VCZ store that lie on a given contig.

``num_contigs``
    Return the number of contigs declared in a VCZ store.

``AncestorWriter``
    Streams ancestors to a zarr Group, flushing by sample-chunk.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
import queue
import threading
import time as _time
import warnings
from typing import Any

import numpy as np
import vcztools
import vcztools.regions
import zarr
from zarr.core.dtype.npy.string import VariableLengthUTF8

from . import config

logger = logging.getLogger(__name__)

_VLEN_STR = VariableLengthUTF8()
_ZARR_FORMAT = 2


@dataclasses.dataclass
class Variant:
    """A single variant yielded by ``MultiSourceView.iter_genotypes``."""

    position: float
    genotypes: np.ndarray  # int8, canonical codes (0=anc, 1+=derived, -1=missing)
    alleles: tuple[str, ...]  # code-to-allele mapping (trimmed of trailing empty)


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------


def open_group(store: str | pathlib.Path | None = None) -> zarr.Group:
    """
    Open a writable zarr Group.

    Parameters
    ----------
    store : str, pathlib.Path, or None
        - ``None`` → in-memory store (MemoryStore).
        - ``str`` / ``pathlib.Path`` → filesystem-backed store at that path.
    """
    if store is None:
        return zarr.open_group(
            zarr.storage.MemoryStore(), mode="w", zarr_format=_ZARR_FORMAT
        )
    return zarr.open_group(str(store), mode="w", zarr_format=_ZARR_FORMAT)


def open_store(path_or_group: str | pathlib.Path | zarr.Group) -> zarr.Group:
    """
    Return a zarr Group for the given path, URL, or existing Group.

    - zarr.Group  → returned unchanged (in-memory or already-open store)
    - str / pathlib.Path  → opened with ``zarr.open(..., mode="r")``
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
) -> np.ndarray:
    """
    Convert a bcftools-style samples string (e.g. ``"s1,s0"`` or ``"^s2"``)
    to an array of sample indexes, in the order given (or in store order
    for a ``^`` complement).

    Returns the full index array if samples is None (keep all samples).
    """
    if samples is None:
        num_samples = store["sample_id"].shape[0]
        return np.arange(num_samples)
    complement = samples.startswith("^")
    if complement:
        samples = samples[1:]
    names = samples.split(",")
    with vcztools.VczReader(store) as reader:
        reader.set_samples(names, complement=complement)
        raw_ids = reader.raw_sample_ids.tolist()
        selected_ids = reader.sample_ids.tolist()
    raw_index = {sample_id: j for j, sample_id in enumerate(raw_ids)}
    return np.array([raw_index[sample_id] for sample_id in selected_ids], dtype=np.int64)


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


def iter_raw_genotypes(store, rows, sample_include):
    """
    Yield ``(num_haplotypes,)`` int8 raw allele-index rows for the specified
    variant rows of a store, in the order given.

    Unlike ``iter_genotypes``, no ancestral-allele polarisation is applied.
    Values are raw allele indices into ``variant_allele``; -1 means missing.

    Parameters
    ----------
    store : zarr.Group
        VCZ store containing ``call_genotype``.
    rows : array-like of int
        Ascending absolute variant row indexes to retrieve.
    sample_include : array-like of bool or None
        Boolean mask (length ``num_samples``) where True means keep this
        sample.  ``None`` keeps all samples.

    Yields
    ------
    np.ndarray
        ``(num_haplotypes,)`` int8 array of raw allele indices.
    """
    site_indices = np.asarray(rows, dtype=np.int64)

    if sample_include is not None:
        sample_include = np.asarray(sample_include, dtype=bool)

    call_gt = store["call_genotype"]
    chunk_size = call_gt.chunks[0]

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

        chunk_start = cid * chunk_size
        while i < n and int(site_indices[i]) // chunk_size == cid:
            local = int(site_indices[i]) - chunk_start
            yield cached_flat[local].astype(np.int8)
            i += 1


@dataclasses.dataclass
class Contig:
    """
    The contig being inferred. Stores may use different names for it:
    ``store_ids`` maps each store's label (a source name, or
    ``"ancestral_state"``) to the name in that store's header.
    """

    id: str  # the name used in outputs
    length: int
    store_ids: dict[str, str]


def _header_contig_ids(store: zarr.Group) -> list[str]:
    return [str(x) for x in store["contig_id"][:].tolist()]


def _contig_index(store: zarr.Group, contig_id: str) -> int:
    """Return the index of the specified contig in a VCZ store's header."""
    header_ids = _header_contig_ids(store)
    if contig_id not in header_ids:
        raise ValueError(
            f"Contig {contig_id!r} is not declared in the store's contig_id "
            f"header: {header_ids}"
        )
    return header_ids.index(contig_id)


def _store_contig_id(
    label: str,
    store: zarr.Group,
    store_contig_id: str | None,
    contig_id: str | None,
) -> str:
    """
    Return the name of the contig being inferred in the store with the
    specified label. This is the first of: the store's own contig option
    (*store_contig_id*); the top-level contig option (*contig_id*), if the
    store declares it; the single contig the store's variants lie on; and
    the store's only declared contig.
    """
    header_ids = _header_contig_ids(store)
    if store_contig_id is not None:
        _contig_index(store, store_contig_id)
        return store_contig_id
    if contig_id is not None and contig_id in header_ids:
        return contig_id
    contig_indexes = np.unique(store["variant_contig"][:])
    if len(contig_indexes) == 1:
        return header_ids[int(contig_indexes[0])]
    if len(contig_indexes) == 0 and len(header_ids) == 1:
        return header_ids[0]
    if len(contig_indexes) > 1:
        names = [header_ids[j] for j in contig_indexes.tolist()]
        reason = f"has variants on multiple contigs {names}"
    else:
        reason = f"has no variants and declares contigs {header_ids}"
    if contig_id is not None:
        reason = f"does not declare contig {contig_id!r} and {reason}"
    raise ValueError(
        f"Cannot determine the contig for {label}: the store {reason}. Set the "
        f"contig option for {label} (or the top-level contig option) to choose "
        "which contig to infer."
    )


def resolve_contig(
    stores: dict[str, tuple[zarr.Group, str | None]], contig_id: str | None
) -> Contig:
    """
    Return the contig to infer from the specified VCZ stores, given as a
    mapping from label to (store, the store's own contig option or None).
    Each store's name for the contig is found by ``_store_contig_id``.

    The contig is named in outputs by *contig_id* (the top-level contig
    option) if given, with a warning if any store uses a different name;
    otherwise all stores must use the same name. At least one store must declare
    the contig length; if the declared lengths differ, a warning is issued and
    the largest is used.
    """
    store_ids = {}
    lengths = set()
    for label, (store, store_contig_id) in stores.items():
        store_id = _store_contig_id(label, store, store_contig_id, contig_id)
        store_ids[label] = store_id
        if "contig_length" in store:
            index = _contig_index(store, store_id)
            lengths.add(int(store["contig_length"][index]))

    names = sorted(set(store_ids.values()))
    names_str = ", ".join(f"{name} ({label})" for label, name in store_ids.items())
    if contig_id is None:
        if len(names) > 1:
            raise ValueError(
                f"Stores use different names for the contig: {names_str}. Set "
                "the top-level contig option to name the inferred contig."
            )
        contig_id = names[0]
    elif names != [contig_id]:
        warnings.warn(
            f"Inferring contig {contig_id} from stores that name it: {names_str}",
            stacklevel=2,
        )

    lengths = sorted(lengths)
    if len(lengths) == 0:
        raise ValueError(
            f"No source declares the length of contig {contig_id}: at "
            "least one source must provide contig_length"
        )
    if len(lengths) > 1:
        warnings.warn(
            f"Sources declare conflicting lengths for contig {contig_id}: "
            f"{lengths}; using the largest",
            stacklevel=2,
        )
    return Contig(id=contig_id, length=lengths[-1], store_ids=store_ids)


def resolve_sources(
    sources: list[config.Source],
    ancestral_state: config.AncestralState | None,
    contig_id: str | None,
):
    """
    Resolve the contig for the specified sources and (optional) ancestral
    state. Returns a ``ResolvedContig`` holding the contig and copies of the
    sources and ancestral state with their ``contig`` set to their store's
    name for it.
    """
    stores = {}
    for source in sources:
        stores[source.name] = (open_store(source.path), source.contig)
    if ancestral_state is not None:
        stores[_ANCESTRAL_STATE_LABEL] = (
            open_store(ancestral_state.path),
            ancestral_state.contig,
        )
    contig = resolve_contig(stores, contig_id)
    resolved_sources = []
    for source in sources:
        store_id = contig.store_ids[source.name]
        resolved_sources.append(dataclasses.replace(source, contig=store_id))
    if ancestral_state is not None:
        store_id = contig.store_ids[_ANCESTRAL_STATE_LABEL]
        ancestral_state = dataclasses.replace(ancestral_state, contig=store_id)
    return ResolvedContig(
        contig=contig, sources=resolved_sources, ancestral_state=ancestral_state
    )


_ANCESTRAL_STATE_LABEL = "ancestral_state"


@dataclasses.dataclass
class ResolvedContig:
    """The result of ``resolve_sources``."""

    contig: Contig
    sources: list[config.Source]
    ancestral_state: config.AncestralState | None


def contig_variant_rows(store: zarr.Group, contig_id: str) -> slice:
    """
    Return the slice of variant rows in a VCZ store that lie on the specified
    contig. Raises a ValueError if these rows are not contiguous.
    """
    index = _contig_index(store, contig_id)
    variant_contig = store["variant_contig"][:]
    rows = np.flatnonzero(variant_contig == index)
    if len(rows) == 0:
        return slice(0, 0)
    start = int(rows[0])
    stop = int(rows[-1]) + 1
    if len(rows) != stop - start:
        raise ValueError(
            f"Variants on contig {contig_id!r} are not contiguous: the store "
            "must be sorted by contig"
        )
    return slice(start, stop)


def num_contigs(store: zarr.Group) -> int:
    """Return the number of contigs declared in a VCZ store."""
    return int(store["contig_id"].shape[0])


# ---------------------------------------------------------------------------
# Source variant selection and ancestral alleles
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SourceVariants:
    """
    The variants of a source on one contig that pass the source's filters,
    excluding those at positions shared by more than one passing variant.
    """

    rows: np.ndarray  # (num_variants,) absolute store row indexes, ascending
    positions: np.ndarray  # (num_variants,) int32
    alleles: np.ndarray  # (num_variants, max_alleles) allele strings


def _source_has_filters(source: config.Source | config.AncestralState) -> bool:
    return (
        source.include is not None
        or source.exclude is not None
        or source.regions is not None
        or source.targets is not None
    )


def _region_chunk_plan(
    reader: vcztools.VczReader, source: config.Source | config.AncestralState
) -> list:
    """
    Return the vcztools chunk plan selecting the variants that pass the
    source's regions and targets filters. Only the variant chunks that
    overlap the regions are read.

    This is the only use of vcztools internals: vcztools has no public API
    for regions and targets.
    """
    regions = None
    if source.regions is not None:
        regions = source.regions.split(",")
    targets = None
    targets_complement = False
    if source.targets is not None:
        targets_str = source.targets
        targets_complement = targets_str.startswith("^")
        if targets_complement:
            targets_str = targets_str[1:]
        targets = targets_str.split(",")
    return vcztools.regions.build_chunk_plan(
        reader,
        regions=regions,
        targets=targets,
        targets_complement=targets_complement,
    )


def _filtered_rows(
    store: zarr.Group,
    source: config.Source | config.AncestralState,
    contig_rows: slice,
) -> np.ndarray:
    """
    Return the ascending row indexes of the variants in a store that pass the
    source's include/exclude/regions/targets filters. Without regions or
    targets, only the variants in *contig_rows* are considered. Filters are
    evaluated over all samples; a per-sample filter keeps a variant if any
    sample passes.
    """
    parts = [np.zeros(0, dtype=np.int64)]
    with vcztools.VczReader(store) as reader:
        if source.regions is not None or source.targets is not None:
            reader.set_variants(_region_chunk_plan(reader, source))
        else:
            reader.set_variants(np.arange(contig_rows.start, contig_rows.stop))
        if source.include is not None or source.exclude is not None:
            variant_filter = vcztools.BcftoolsFilter(
                reader, include=source.include, exclude=source.exclude
            )
            reader.set_variant_filter(variant_filter)
        for chunk in reader.variant_chunks(fields=["variant_index"]):
            parts.append(chunk["variant_index"])
    return np.concatenate(parts)


def _selected_rows(
    store: zarr.Group,
    source: config.Source | config.AncestralState,
    contig_rows: slice,
) -> np.ndarray:
    """
    Return the ascending row indexes of the variants in *contig_rows* that
    pass the source's filters.
    """
    if not _source_has_filters(source):
        return np.arange(contig_rows.start, contig_rows.stop, dtype=np.int64)
    rows = _filtered_rows(store, source, contig_rows)
    in_contig = (rows >= contig_rows.start) & (rows < contig_rows.stop)
    return rows[in_contig]


def select_source_variants(store: zarr.Group, source: config.Source) -> SourceVariants:
    """
    Return the variants of the source on its contig (``source.contig``, as
    set by ``resolve_sources``) that pass its filters. Duplicate positions
    are identified after filtering, so that a filter which removes all but
    one of the variants at a position (e.g. ``TYPE="snp"`` for a SNP and an
    indel) keeps the remaining variant.
    """
    contig_id = source.contig
    contig_rows = contig_variant_rows(store, contig_id)
    num_on_contig = contig_rows.stop - contig_rows.start
    rows = _selected_rows(store, source, contig_rows)
    num_passing = len(rows)

    if num_passing > 0:
        # Read only the span of rows that passed the filters
        span = slice(int(rows[0]), int(rows[-1]) + 1)
        local = rows - span.start
        positions = np.asarray(store["variant_position"][span], dtype=np.int32)[local]
        dup_positions = _find_duplicate_positions(positions)
        keep = ~np.isin(positions, dup_positions)
        rows = rows[keep]
        positions = positions[keep]
        alleles = np.asarray(store["variant_allele"][span])[local[keep]]
    else:
        positions = np.zeros(0, dtype=np.int32)
        alleles = np.zeros((0, store["variant_allele"].shape[1]), dtype=object)
    alleles = alleles.astype(object)

    logger.info(
        "Source %s: %d variants on contig %s, %d pass filters, %d at duplicate "
        "positions, %d used",
        source.name,
        num_on_contig,
        contig_id,
        num_passing,
        num_passing - len(rows),
        len(rows),
    )
    return SourceVariants(rows=rows, positions=positions, alleles=alleles)


# Ancestral state values meaning that the ancestral allele is unknown
MISSING_ANCESTRAL_STATES = {".", "N", ""}


@dataclasses.dataclass
class AncestralAnnotation:
    """The result of ``lookup_ancestral_alleles``."""

    alleles: np.ndarray  # (num_positions,) ancestral allele, "" if unknown
    refs: np.ndarray  # (num_positions,) annotation REF, "" if no annotation


def lookup_ancestral_alleles(
    ancestral_state: config.AncestralState, positions: np.ndarray
) -> AncestralAnnotation:
    """
    Return the ancestral allele and REF allele of the annotation variant at
    each of the specified positions on the ancestral state store's contig
    (``ancestral_state.contig``), using only the variants that pass the
    ancestral state's filters. Raises a ValueError if more than one variant
    remains at any of the positions. Ancestral alleles in
    ``MISSING_ANCESTRAL_STATES`` are returned as "" (unknown).
    """
    ann_store = open_store(ancestral_state.path)
    contig_rows = contig_variant_rows(ann_store, ancestral_state.contig)
    rows = _selected_rows(ann_store, ancestral_state, contig_rows)
    all_positions = np.asarray(
        ann_store["variant_position"][contig_rows], dtype=np.int32
    )
    ann_positions = all_positions[rows - contig_rows.start]
    lo = np.searchsorted(ann_positions, positions, side="left")
    hi = np.searchsorted(ann_positions, positions, side="right")
    counts = hi - lo

    duplicated = counts > 1
    num_duplicated = int(np.sum(duplicated))
    if num_duplicated > 0:
        first = np.flatnonzero(duplicated)[0]
        first_rows = rows[lo[first] : hi[first]]
        first_alleles = ann_store["variant_allele"].oindex[first_rows, :2]
        raise ValueError(
            f"The ancestral state store {ancestral_state.path} has more than one "
            f"variant at {num_duplicated} position(s) used for inference (e.g. "
            f"positions {positions[duplicated][:5].tolist()}; alleles at "
            f"{int(positions[first])}: {first_alleles.tolist()}). Use the "
            "include, exclude, regions or targets options of [ancestral_state] "
            "to select one variant per position, e.g. include = 'TYPE=\"snp\"'."
        )

    found = counts == 1
    needed_rows = rows[lo[found]]
    if ancestral_state.is_reference:
        values = np.asarray(ann_store["variant_allele"].oindex[needed_rows, 0])
    else:
        values = np.asarray(ann_store[ancestral_state.field].oindex[needed_rows])
    ann_refs = np.asarray(ann_store["variant_allele"].oindex[needed_rows, 0])

    alleles = np.full(len(positions), "", dtype=object)
    alleles[found] = values
    refs = np.full(len(positions), "", dtype=object)
    refs[found] = ann_refs
    missing = np.isin(alleles, list(MISSING_ANCESTRAL_STATES))
    alleles[missing] = ""
    logger.info(
        "Ancestral alleles: %d positions with no annotation, %d with a missing "
        "ancestral state",
        int(np.sum(~found)),
        int(np.sum(missing & found)),
    )
    return AncestralAnnotation(alleles=alleles, refs=refs)


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


@dataclasses.dataclass
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


@dataclasses.dataclass
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
        num_threads=None,
        write_threads=None,
        compressor=None,
    ):
        if num_threads is None:
            num_threads = 0
        if write_threads is None:
            write_threads = config.DEFAULT_ANCESTOR_WRITE_THREADS
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
                end_pos = int(final_positions[end_site_idx - 1]) + 1
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

    for name in (
        "sample_time",
        "sample_start_position",
        "sample_end_position",
    ):
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
    # vcztools takes the samples chunk size from sample_id, so chunk it
    # like call_genotype
    samples_chunk_size = root["call_genotype"].chunks[1]
    id_arr = root.create_array(
        "sample_id",
        shape=ids.shape,
        dtype=_VLEN_STR,
        chunks=(samples_chunk_size,),
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


class ScheduledCache:
    """Schedule-driven chunk cache with a pool of read workers.

    Requires a full read schedule at construction: a refcount per chunk
    (how many haplotype reads will use it) and the order chunks will first
    be needed.

    A pool of background worker threads autonomously loads chunks in
    schedule order, blocking when the memory budget is full.  Chunks are
    evicted when their refcount reaches zero, freeing space for workers.
    ``get()`` never triggers a load — it returns cached data or waits
    for a worker to deliver it.
    """

    def __init__(
        self,
        max_bytes: int,
        refcounts: dict[tuple[str, int], int],
        chunk_order: list[tuple[str, int]],
        num_workers: int = 1,
    ):
        self._max_bytes = max_bytes
        self._chunks: dict[tuple[str, int], np.ndarray] = {}
        self._current_bytes = 0
        self._pending_bytes = 0
        self._refcount = dict(refcounts)
        self._chunk_order = chunk_order
        self._cursor = 0
        self._condition = threading.Condition()
        self._loaders: dict[str, Any] = {}
        self._chunk_bytes: dict[str, int] = {}
        self._shutdown = False
        self._num_workers = num_workers
        self._workers: list[threading.Thread] = []
        self._hits = 0
        self._misses = 0
        self._total_load_time = 0.0
        self._total_wait_time = 0.0
        self._total_loads = 0
        self._total_evictions = 0
        self._total_evicted_bytes = 0
        self._t_start = _time.monotonic()

    def register_loader(
        self,
        source_name: str,
        load_fn,
        chunk_bytes: int,
    ) -> None:
        """Register a loader ``load_fn(chunk_idx) -> ndarray`` for a source."""
        self._loaders[source_name] = load_fn
        self._chunk_bytes[source_name] = chunk_bytes

    def start(self) -> None:
        """Start the read worker pool."""
        logger.info("Cache budget: %.1f MiB", self._max_bytes / (1024 * 1024))
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"cache-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def _cache_state_summary(self) -> str:
        """Return a brief snapshot of cached chunks and their refcounts.

        Must be called under ``self._condition``.
        """
        parts = []
        for src, idx in sorted(self._chunks):
            ref = self._refcount.get((src, idx), 0)
            parts.append(f"{src}:{idx}(ref={ref})")
        return "[" + ", ".join(parts) + "]"

    # ----- worker pool -----------------------------------------------------

    def _advance_cursor(self) -> tuple[str, int] | None:
        """Advance cursor past cached/unneeded chunks, return next needed key.

        Must be called under ``self._condition``.
        """
        while self._cursor < len(self._chunk_order):
            k = self._chunk_order[self._cursor]
            if k in self._chunks or k not in self._refcount:
                self._cursor += 1
                continue
            if k[0] not in self._loaders:
                self._cursor += 1
                continue
            return k
        return None

    def _force_evict(self) -> tuple[str, int] | None:
        """Evict one cached chunk to free memory. Must be called under lock.

        Picks the chunk with the fewest remaining references.  The
        chunk's ``_refcount`` entry is preserved so it can be reloaded
        later if needed.  Returns the evicted key, or None if nothing
        can be evicted.
        """
        best_key = None
        best_refs = float("inf")
        for k in list(self._chunks):
            refs = self._refcount.get(k, 0)
            if refs < best_refs:
                best_key = k
                best_refs = refs
        if best_key is None:
            return None
        data = self._chunks.pop(best_key)
        freed = data.nbytes
        self._current_bytes -= freed
        self._total_evictions += 1
        self._total_evicted_bytes += freed
        logger.info(
            "Chunk cache force-evict: source=%s chunk=%d"
            " freed=%.1f MiB (remaining refs=%d)",
            best_key[0],
            best_key[1],
            freed / (1024 * 1024),
            self._refcount.get(best_key, 0),
        )
        self._condition.notify_all()
        return best_key

    def _worker_loop(self) -> None:
        """Worker: pull chunks from the schedule and load them."""
        while True:
            with self._condition:
                while not self._shutdown:
                    key = self._advance_cursor()
                    if key is None:
                        return  # schedule exhausted
                    est = self._chunk_bytes.get(key[0], 0)
                    committed = self._current_bytes + self._pending_bytes + est
                    if committed <= self._max_bytes:
                        self._cursor += 1
                        self._pending_bytes += est
                        break
                    self._condition.wait()
                else:
                    return  # shutdown requested

            t_load = _time.monotonic()
            data = self._loaders[key[0]](key[1])
            load_elapsed = _time.monotonic() - t_load

            with self._condition:
                self._chunks[key] = data
                self._current_bytes += data.nbytes
                self._pending_bytes -= est
                self._total_load_time += load_elapsed
                self._total_loads += 1
                state = self._cache_state_summary()
                self._condition.notify_all()

            logger.info(
                "Chunk cache load: source=%s chunk=%d size=%.1f MiB"
                " load=%.3fs (total: %.1f MiB, %d entries)",
                key[0],
                key[1],
                data.nbytes / (1024 * 1024),
                load_elapsed,
                self._current_bytes / (1024 * 1024),
                len(self._chunks),
            )
            logger.debug("Chunk cache state: %s", state)

    # ----- public interface ------------------------------------------------

    def get(self, key: tuple[str, int]) -> np.ndarray:
        """Return cached chunk, waiting for a worker to load it.

        If workers cannot make progress (e.g. the cursor has passed this
        chunk after a force-eviction), falls back to loading the chunk
        synchronously from the calling thread, evicting another chunk
        first to stay within the memory budget.
        """
        with self._condition:
            if key in self._chunks:
                self._hits += 1
                return self._chunks[key]
            self._misses += 1
            t_wait = _time.monotonic()
            while key not in self._chunks:
                if not self._condition.wait(timeout=1.0):
                    if key[0] in self._loaders and key not in self._chunks:
                        # Evict a chunk to stay within the memory budget
                        self._force_evict()
                        loader = self._loaders[key[0]]
                        logger.warning(
                            "Chunk cache stall: loading source=%s "
                            "chunk=%d synchronously",
                            key[0],
                            key[1],
                        )
                        self._condition.release()
                        try:
                            data = loader(key[1])
                        finally:
                            self._condition.acquire()
                        if key not in self._chunks:
                            self._chunks[key] = data
                            self._current_bytes += data.nbytes
                            self._condition.notify_all()
            self._total_wait_time += _time.monotonic() - t_wait
            return self._chunks[key]

    def record_read(self, key: tuple[str, int]) -> None:
        """Record that one haplotype was read from *key*.

        When the refcount reaches zero the chunk is evicted, freeing
        memory for workers to load the next chunks.
        """
        evicted = False
        freed = 0
        remaining = 0
        with self._condition:
            self._refcount[key] -= 1
            remaining = self._refcount[key]
            if remaining <= 0:
                del self._refcount[key]
                data = self._chunks.pop(key, None)
                if data is not None:
                    freed = data.nbytes
                    self._current_bytes -= freed
                    self._total_evictions += 1
                    self._total_evicted_bytes += freed
                    evicted = True
            state = self._cache_state_summary()
            if evicted:
                self._condition.notify_all()
        if evicted:
            logger.info(
                "Chunk cache evict (refcount=0): source=%s chunk=%d"
                " freed=%.1f MiB (remaining: %.1f MiB, %d entries)",
                key[0],
                key[1],
                freed / (1024 * 1024),
                self._current_bytes / (1024 * 1024),
                len(self._chunks),
            )
            logger.debug("Chunk cache state: %s", state)
        else:
            logger.debug(
                "Chunk cache read: source=%s chunk=%d refcount=%d",
                key[0],
                key[1],
                remaining,
            )

    def log_state(self) -> None:
        """Log an info-level summary of current cache state."""
        with self._condition:
            n_chunks = len(self._chunks)
            cached_mb = self._current_bytes / (1024 * 1024)
            ready_haps = sum(self._refcount.get(k, 0) for k in self._chunks)
        logger.info(
            "Chunk cache state: %.1f MiB, %d chunks cached, %d haplotypes ready",
            cached_mb,
            n_chunks,
            ready_haps,
        )

    def shutdown(self) -> None:
        """Stop workers and log summary stats."""
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        for t in self._workers:
            t.join(timeout=10)
        wall = _time.monotonic() - self._t_start
        logger.info(
            "Chunk cache summary: %d loads"
            " %d evictions, %d hits, %d misses,"
            " load_time=%.1fs wait_time=%.1fs wall=%.1fs",
            self._total_loads,
            self._total_evictions,
            self._hits,
            self._misses,
            self._total_load_time,
            self._total_wait_time,
            wall,
        )

    @property
    def total_bytes(self) -> int:
        """Current memory usage of cached arrays in bytes."""
        return self._current_bytes


def _find_duplicate_positions(positions: np.ndarray) -> np.ndarray:
    """Return sorted array of genomic positions that appear more than once."""
    unique, counts = np.unique(positions, return_counts=True)
    dup_mask = counts > 1
    if not np.any(dup_mask):
        return np.empty(0, dtype=positions.dtype)
    dup_arr = unique[dup_mask]
    n_dup_sites = int(np.sum(counts[dup_mask]))
    logger.info(
        "Found %d duplicate position(s) (%d sites total); "
        "these will be excluded from inference",
        len(dup_arr),
        n_dup_sites,
    )
    return dup_arr


def build_allele_mapper(
    positions: np.ndarray,
    ancestral_alleles: np.ndarray,
    per_source_alleles: list[tuple[np.ndarray, np.ndarray]],
) -> AlleleMapper:
    """
    Build an AlleleMapper from ancestral alleles and
    per-source (positions, variant_allele) arrays.

    per_source_alleles[i] = (src_positions, src_alleles)
    """
    allele_table: list[list[str]] = [[str(a)] for a in ancestral_alleles]
    positions = np.asarray(positions, dtype=np.int32)
    if len(positions) == 0:
        return AlleleMapper(0, allele_table)

    for src_positions, src_alleles in per_source_alleles:
        src_positions = np.asarray(src_positions, dtype=np.int32)
        idx_in_unified = np.searchsorted(positions, src_positions)
        valid = (idx_in_unified < len(positions)) & (
            positions[np.minimum(idx_in_unified, len(positions) - 1)] == src_positions
        )
        matched_unified = idx_in_unified[valid]
        matched_src_rows = np.where(valid)[0]

        for k in range(len(matched_unified)):
            ref_idx = int(matched_unified[k])
            site_alleles = src_alleles[matched_src_rows[k]]
            allele_list = (
                site_alleles.tolist()
                if hasattr(site_alleles, "tolist")
                else site_alleles
            )
            for a in allele_list:
                a_str = str(a)
                if a_str and a_str not in allele_table[ref_idx]:
                    allele_table[ref_idx].append(a_str)

    return AlleleMapper(len(positions), allele_table)


class AlleleMapper:
    """Fixed allele encoding: code 0 = ancestral, 1.. = derived.

    Built once from all sources' variant_allele arrays. Provides
    efficient bulk conversion between allele codes and real strings.
    Immutable after construction — no locking needed.
    """

    def __init__(self, num_sites: int, allele_table: list[list[str]]):
        self._num_sites = num_sites
        self._allele_table = allele_table  # allele_table[i] = [anc, der1, der2, ...]
        # Pre-build the forward map: (num_sites, max_alleles) object array
        max_a = max((len(t) for t in allele_table), default=2)
        max_a = max(max_a, 2)
        self._forward = np.full((num_sites, max_a), "", dtype=object)
        for i, table in enumerate(allele_table):
            for j, a in enumerate(table):
                self._forward[i, j] = a
        # Pre-build the reverse map: dict[(site_id, allele_str)] -> code
        self._reverse: dict[tuple[int, str], int] = {}
        for i, table in enumerate(allele_table):
            for code, a in enumerate(table):
                self._reverse[(i, a)] = code

    def lookup(self, ref_site_idx: int, allele_string: str) -> int:
        """Return the code for allele_string at site, or -1 if unknown."""
        return self._reverse.get((ref_site_idx, allele_string), -1)

    def num_alleles(self, ref_site_idx: int) -> int:
        return len(self._allele_table[ref_site_idx])

    def num_alleles_array(self) -> np.ndarray:
        """Return (num_sites,) uint64 array of allele counts per site."""
        return np.array([len(t) for t in self._allele_table], dtype=np.uint64)

    def site_alleles_array(self) -> np.ndarray:
        """Return (num_sites, max_alleles) object array for extend_ts."""
        return self._forward.copy()

    def forward_map(self) -> np.ndarray:
        """(num_sites, max_alleles) object array: code -> allele string."""
        return self._forward

    def ancestral_alleles(self) -> np.ndarray:
        """(num_sites,) array of ancestral allele strings (code 0)."""
        return self._forward[:, 0]

    def decode_mutations(self, site_ids: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Convert integer allele codes to real allele strings (bulk).

        Parameters
        ----------
        site_ids : (N,) int array
        codes : (N,) int8 array

        Returns
        -------
        (N,) object array of allele strings
        """
        return self._forward[site_ids, codes]

    def batch_lookup(self, site_indices: np.ndarray, allele_strings) -> np.ndarray:
        """Return codes for allele_strings at given sites, or -1 if unknown.

        Parameters
        ----------
        site_indices : (N,) int array
        allele_strings : (N,) array of strings

        Returns
        -------
        (N,) int8 array of codes
        """
        allele_strings = np.asarray(allele_strings)
        result = np.full(len(site_indices), -1, dtype=np.int8)
        for c in range(self._forward.shape[1]):
            match = self._forward[site_indices, c] == allele_strings
            result[match] = c
        return result

    def encode_mutations(
        self, site_ids: np.ndarray, allele_strings: np.ndarray
    ) -> np.ndarray:
        """Convert real allele strings to integer codes (bulk).

        Parameters
        ----------
        site_ids : (N,) int array
        allele_strings : (N,) array of strings

        Returns
        -------
        (N,) int8 array of codes
        """
        allele_strings = np.asarray(allele_strings)
        result = np.full(len(site_ids), -1, dtype=np.int8)
        for c in range(self._forward.shape[1]):
            match = self._forward[site_ids, c] == allele_strings
            result[match] = c
        return result


class VCZHaplotypeReader:
    """
    Reads and polarises haplotypes from a single VCZ store.

    Handles both ancestor and sample stores uniformly: the source's selected
    variants are aligned to a reference coordinate system, and genotypes are
    polarised so that the ancestral allele is always 0. Reference positions
    without a selected variant read as missing.

    Uses an external ``ScheduledCache`` shared across all sources to avoid
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
    cache : ScheduledCache
        Shared chunk cache instance.
    variants : SourceVariants
        The variants of the store to align to the reference positions
        (see ``select_source_variants``).
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
        cache: ScheduledCache,
        variants: SourceVariants,
        samples_selection: np.ndarray | None = None,
        _allele_mapper: AlleleMapper | None = None,
    ):
        self._store = open_store(store)
        self._ref_positions = np.asarray(positions, dtype=np.int32)
        self._num_ref_sites = len(self._ref_positions)
        self._samples_selection = samples_selection
        self._source_name = source_name
        self._cache = cache
        self._allele_mapper = _allele_mapper

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

        # Select the variants at reference positions. Variants are in
        # ascending row (and position) order, so the selection is too.
        ref_idx = np.searchsorted(self._ref_positions, variants.positions)
        clipped = np.minimum(ref_idx, max(self._num_ref_sites - 1, 0))
        found = ref_idx < self._num_ref_sites
        if self._num_ref_sites > 0:
            found &= self._ref_positions[clipped] == variants.positions
        selected_rows = variants.rows[found]
        self._selected_to_ref_idx = ref_idx[found].astype(np.int32)
        self._selected_alleles = variants.alleles[found]
        self._num_selected = len(selected_rows)

        # Index of each selected variant's ancestral allele (-1 if absent),
        # used to polarise when no allele mapper is assigned
        selected_ancestral = np.asarray(ancestral_alleles, dtype=object)[
            self._selected_to_ref_idx
        ]
        is_ancestral = (self._selected_alleles == selected_ancestral[:, None]) & (
            self._selected_alleles != ""
        )
        self._ancestral_allele_index = np.where(
            np.any(is_ancestral, axis=1), np.argmax(is_ancestral, axis=1), -1
        ).astype(np.int8)

        # Zarr variant chunk size for chunk-wise iteration
        self._variant_chunk_size = (
            call_gt.chunks[0]
            if hasattr(call_gt, "chunks") and call_gt.chunks is not None
            else call_gt.shape[0]
        )

        # Precompute which variant chunks contain selected variants:
        # [(vc_idx, row indexes within the block, selected indexes), ...]
        vc_size = self._variant_chunk_size
        selected_chunks = selected_rows // vc_size
        self._needed_vc = []
        for vc_idx in np.unique(selected_chunks).tolist():
            out_rows = np.flatnonzero(selected_chunks == vc_idx)
            block_indices = selected_rows[out_rows] - vc_idx * vc_size
            self._needed_vc.append((vc_idx, block_indices, out_rows))

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
        code via ``_allele_mapper.lookup()``.  Called once after
        ``_allele_mapper`` is assigned; removes all string operations
        from the hot cache-miss path.
        """
        if self._allele_mapper is None or self._num_selected == 0:
            return
        max_a = self._selected_alleles.shape[1]
        remap = np.full((self._num_selected, max_a), -1, dtype=np.int8)
        for row in range(self._num_selected):
            ref_idx = int(self._selected_to_ref_idx[row])
            for j in range(max_a):
                a = str(self._selected_alleles[row, j])
                if a:
                    remap[row, j] = self._allele_mapper.lookup(ref_idx, a)
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

        for vc_idx, block_indices, out_rows in self._needed_vc:
            t0 = _time.monotonic()
            block = np.asarray(call_gt.blocks[vc_idx, chunk_idx], dtype=np.int8)
            t_zarr += _time.monotonic() - t0
            n_zarr += 1

            t0 = _time.monotonic()
            raw = block[block_indices]  # (n, chunk_samples, ploidy)
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

    def _chunk_for_sample(self, sample_id: str) -> int:
        """Map a sample_id to its zarr sample-chunk index."""
        col = self._sample_id_to_col[sample_id]
        if self._samples_selection is not None:
            store_col = int(self._samples_selection[col])
        else:
            store_col = col
        return store_col // self._sample_chunk_size

    def _load_sample_chunk(self, chunk_idx: int) -> np.ndarray:
        """Load a sample chunk via the shared ScheduledCache.

        Returns array of shape ``(num_selected, chunk_samples, ploidy)``.

        Thread-safe: concurrent loads for the same chunk are
        deduplicated by the cache.  The returned numpy array remains
        valid even after eviction because Python ref-counting keeps
        it alive.
        """
        key = (self._source_name, chunk_idx)
        return self._cache.get(key)

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
        key = (self._source_name, chunk_idx)
        chunk_data = self._load_sample_chunk(chunk_idx)
        encoded_col = chunk_data[:, chunk_offset, ploidy_index]  # (num_selected,)

        hap = np.full(self._num_ref_sites, np.int8(-1), dtype=np.int8)
        if self._num_selected > 0:
            if self._allele_mapper is not None:
                # Already encoded at load time
                hap[self._selected_to_ref_idx] = encoded_col
            else:
                # Fallback: polarise from raw genotypes
                ai = self._ancestral_allele_index
                valid = ai >= 0
                raw = encoded_col[valid]
                encoded = np.where(
                    raw < 0,
                    np.int8(-1),
                    np.where(raw == ai[valid], np.int8(0), np.int8(1)),
                )
                hap[self._selected_to_ref_idx[valid]] = encoded

        # Notify the cache that one scheduled read is done
        self._cache.record_read(key)
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
        cache_size_mb: int | None = None,
        schedule: list[tuple[str, str, int]] | None = None,
        read_workers: int | None = None,
        num_threads: int = 0,
        reference_alleles: list[list[str]] | None = None,
    ):
        """
        Parameters
        ----------
        sources : dict[str, config.Source]
            All sources to read from (including ancestor sources), each
            with its ``contig`` set (see ``resolve_sources``).
        positions : np.ndarray
            Reference variant positions (the coordinate system).
        ancestral_alleles : np.ndarray
            (num_sites,) array of ancestral allele strings, one per
            reference position.  Authoritative source for polarisation.
        cache_size_mb : int
            Cache memory limit in MiB.
        schedule : list of (source_name, sample_id, ploidy_index)
            Ordered list of every haplotype read that will be made.
            The cache uses this to compute per-chunk reference counts
            for deterministic eviction and background read-ahead.
        read_workers : int or None
            Number of background threads that load chunks into the cache.
            Defaults to ``max(1, num_threads // 2)``.
        num_threads : int
            Number of matching threads (used to derive ``read_workers``
            default).
        reference_alleles : list of list of str or None
            Extra alleles for each reference position (e.g. the alleles
            already present in a reference tree sequence), which are added
            to the allele map alongside those of the sources.
        """
        if read_workers is None:
            read_workers = max(1, num_threads // 2)
        if cache_size_mb is None:
            cache_size_mb = config.DEFAULT_CACHE_SIZE
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
                variants=select_source_variants(store, source),
            )
            self._readers[name] = reader

        # Build fixed allele map from all sources via build_allele_mapper
        per_source_alleles = []
        for reader in self._readers.values():
            selected_positions = self._positions[reader._selected_to_ref_idx]
            per_source_alleles.append((selected_positions, reader._selected_alleles))
        if reference_alleles is not None:
            per_source_alleles.append((self._positions, reference_alleles))

        self._allele_mapper = build_allele_mapper(
            self._positions, ancestral_alleles, per_source_alleles
        )

        # Assign the allele map to all readers and precompute remap tables
        for reader in self._readers.values():
            reader._allele_mapper = self._allele_mapper
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

        # Build schedule-driven cache from the read schedule
        if schedule is None:
            schedule = []
        refcounts: dict[tuple[str, int], int] = {}
        chunk_order: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for source_name, sample_id, _ploidy_index in schedule:
            reader = self._readers[source_name]
            chunk_idx = reader._chunk_for_sample(sample_id)
            key = (source_name, chunk_idx)
            refcounts[key] = refcounts.get(key, 0) + 1
            if key not in seen:
                chunk_order.append(key)
                seen.add(key)

        self._cache = ScheduledCache(
            max_bytes, refcounts, chunk_order, num_workers=read_workers
        )
        for name, reader in self._readers.items():
            reader._cache = self._cache
            self._cache.register_loader(name, reader._do_load_chunk, reader.chunk_bytes)
        self._cache.start()

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
                variants=select_source_variants(store, source),
                _allele_mapper=self._allele_mapper,
            )
            self._readers[source_name] = reader
            self._cache.register_loader(
                source_name, reader._do_load_chunk, reader.chunk_bytes
            )
            return reader

    def log_cache_state(self) -> None:
        """Log an info-level summary of current cache state."""
        self._cache.log_state()

    def shutdown(self) -> None:
        """Shut down the cache's background read-ahead executor."""
        self._cache.shutdown()

    def read_haplotype(self, job) -> np.ndarray:
        """
        Read and encode a single haplotype for the given MatchJob.

        Returns (num_ref_sites,) int8 array:
        0=ancestral, 1=derived, -1=missing.
        """
        reader = self._get_reader(job.source)
        return reader.read_haplotype(job.sample_id, job.ploidy_index)

    @property
    def allele_mapper(self) -> AlleleMapper:
        return self._allele_mapper

    def get_num_alleles(self) -> np.ndarray:
        """Return (num_sites,) uint64 array of allele counts per site."""
        return self._allele_mapper.num_alleles_array()

    def get_site_alleles(self) -> np.ndarray:
        """Return (num_sites, max_alleles) object array of allele strings.

        Allele codes are consistent with the encoding used by
        ``read_haplotype``: index 0 is always the ancestral allele,
        index 1 is the first derived allele seen across all sources, etc.
        """
        return self._allele_mapper.site_alleles_array()


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
    store : str, pathlib.Path, or None
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


# ---------------------------------------------------------------------------
# MultiSourceView — unified multi-source variant iteration
# ---------------------------------------------------------------------------


class MultiSourceView:
    """
    Unified view over one or more VCZ sample stores for variant iteration.

    The view's sites are the union of the variants of each source on its
    contig that pass the source's filters (see ``select_source_variants``).
    Each site's ancestral allele is looked up in the ``ancestral_state``
    store (see ``lookup_ancestral_alleles``). A ValueError is raised if the
    sources, or the sources and the annotation, have different REF alleles
    at any site; sources may have different ALT alleles. The ancestral
    allele is unknown ("") if it is missing or not among the site's alleles
    in the sources. Allele codes are fixed at construction: code 0 is the
    ancestral allele and codes 1+ are the other alleles.

    Each source and the ancestral state must have their ``contig`` set to
    their store's name for the contig (see ``resolve_sources``).
    """

    def __init__(self, sources, ancestral_state):
        if isinstance(sources, config.Source):
            sources = [sources]

        n_sources = len(sources)
        self._sources = sources
        self._stores = [open_store(source.path) for source in sources]
        self._sample_includes = []
        self._ploidies = []
        self._num_haplotypes_per_source = []
        self._sample_ids_per_source = []
        self._variants = []

        # --- Per-source initialization ---
        for source, store in zip(sources, self._stores):
            samples_selection = resolve_samples_selection(store, source.samples)
            gt_shape = store["call_genotype"].shape
            num_samples = gt_shape[1]
            ploidy = gt_shape[2]
            self._ploidies.append(ploidy)

            sample_include = np.zeros(num_samples, dtype=bool)
            sample_include[samples_selection] = True
            self._sample_includes.append(sample_include)
            self._num_haplotypes_per_source.append(len(samples_selection) * ploidy)

            raw_ids = store["sample_id"][:][samples_selection]
            self._sample_ids_per_source.append([str(x) for x in raw_ids.tolist()])

            self._variants.append(select_source_variants(store, source))

        # --- Unified site set ---
        all_positions = functools.reduce(
            np.union1d,
            [variants.positions for variants in self._variants],
            np.zeros(0, dtype=np.int32),
        ).astype(np.int32)
        # source_row[j, k] is the index of site j in source k's variants, or -1
        source_row = np.full((len(all_positions), n_sources), -1, dtype=np.intp)
        for src_idx, variants in enumerate(self._variants):
            site_idx = np.searchsorted(all_positions, variants.positions)
            source_row[site_idx, src_idx] = np.arange(len(variants.positions))

        self._positions = all_positions
        self._source_row = source_row

        # --- REF alleles must agree across the sources ---
        # The first source with a site defines its REF allele
        first_src = np.argmax(source_row >= 0, axis=1)
        refs = np.full(len(all_positions), "", dtype=object)
        for src_idx, variants in enumerate(self._variants):
            is_first = first_src == src_idx
            rows = source_row[is_first, src_idx]
            refs[is_first] = variants.alleles[rows, 0]
        for src_idx, variants in enumerate(self._variants):
            has_site = source_row[:, src_idx] >= 0
            src_refs = variants.alleles[source_row[has_site, src_idx], 0]
            mismatch = np.flatnonzero(src_refs != refs[has_site])
            if len(mismatch) > 0:
                site = np.flatnonzero(has_site)[mismatch[0]]
                first_name = sources[first_src[site]].name
                raise ValueError(
                    f"REF allele mismatch at {len(mismatch)} site(s), e.g. "
                    f"position {int(all_positions[site])}: source {first_name!r} "
                    f"has REF {refs[site]!r} but source {sources[src_idx].name!r} "
                    f"has REF {src_refs[mismatch[0]]!r}"
                )

        # --- REF alleles must agree with the ancestral state annotation ---
        annotation = lookup_ancestral_alleles(ancestral_state, all_positions)
        annotated = annotation.refs != ""
        mismatch = np.flatnonzero(annotated & (annotation.refs != refs))
        if len(mismatch) > 0:
            site = mismatch[0]
            source_name = sources[first_src[site]].name
            raise ValueError(
                f"REF allele mismatch at {len(mismatch)} site(s), e.g. position "
                f"{int(all_positions[site])}: source {source_name!r} has REF "
                f"{refs[site]!r} but the ancestral state store "
                f"{ancestral_state.path} has REF {annotation.refs[site]!r}"
            )

        # --- Ancestral alleles: unknown unless among the site's alleles ---
        ancestral = annotation.alleles
        among_alleles = np.zeros(len(self._positions), dtype=bool)
        for src_idx, variants in enumerate(self._variants):
            has_site = self._source_row[:, src_idx] >= 0
            site_alleles = variants.alleles[self._source_row[has_site, src_idx]]
            matches = site_alleles == ancestral[has_site, None]
            among_alleles[has_site] |= np.any(matches, axis=1)
        known = ancestral != ""
        not_among = known & ~among_alleles
        logger.info(
            "Ancestral alleles: %d site(s) with an ancestral allele that is not "
            "among the site's alleles, treated as unknown",
            int(np.sum(not_among)),
        )
        ancestral[not_among] = ""
        self._ancestral_alleles = ancestral

        # --- Allele codes and per-source remap tables ---
        per_source_allele_pairs = []
        for variants in self._variants:
            per_source_allele_pairs.append((variants.positions, variants.alleles))
        self._allele_mapper = build_allele_mapper(
            self._positions, self._ancestral_alleles, per_source_allele_pairs
        )
        # remap_tables[k][i, a] is the code of allele a of source k's variant i
        self._remap_tables = []
        for src_idx, variants in enumerate(self._variants):
            has_site = self._source_row[:, src_idx] >= 0
            site_idx = np.flatnonzero(has_site)
            src_rows = self._source_row[has_site, src_idx]
            max_a = variants.alleles.shape[1]
            remap = np.full((len(variants.positions), max_a), -1, dtype=np.int8)
            for j in range(max_a):
                col_alleles = np.array(
                    [str(x) for x in variants.alleles[src_rows, j].tolist()]
                )
                nonempty = col_alleles != ""
                codes = self._allele_mapper.batch_lookup(
                    site_idx[nonempty], col_alleles[nonempty]
                )
                remap[src_rows[nonempty], j] = codes
            self._remap_tables.append(remap)

    @property
    def num_sites(self) -> int:
        return len(self._positions)

    @property
    def num_haplotypes(self) -> int:
        return sum(self._num_haplotypes_per_source)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def ancestral_alleles(self) -> np.ndarray:
        """Ancestral allele strings aligned with ``positions`` ("" if unknown)."""
        return self._ancestral_alleles

    @property
    def alleles(self) -> np.ndarray:
        """(num_sites, max_alleles) array mapping allele codes to strings."""
        return self._allele_mapper.forward_map()

    def source_store(self, src_idx: int):
        return self._stores[src_idx]

    def haplotypes(self) -> list[tuple[str, str, int]]:
        """(source_name, sample_id, ploidy_index) per haplotype."""
        result = []
        for src_idx in range(len(self._sources)):
            source_name = self._sources[src_idx].name
            ploidy = self._ploidies[src_idx]
            for sid in self._sample_ids_per_source[src_idx]:
                for p in range(ploidy):
                    result.append((source_name, sid, p))
        return result

    def positions_in_intervals(self, intervals) -> np.ndarray:
        """
        Return the subset of ``self.positions`` that fall within the
        given ``[start, end)`` intervals.

        Parameters
        ----------
        intervals : list of [start, end) pairs
            Half-open intervals.
        """
        iv_arr = np.asarray(intervals, dtype=np.float64)
        iv_starts = iv_arr[:, 0]
        iv_ends = iv_arr[:, 1]
        positions = self._positions.astype(np.float64)
        idx = np.searchsorted(iv_starts, positions, side="right") - 1
        valid = idx >= 0
        in_interval = valid & (positions < iv_ends[np.maximum(idx, 0)])
        return self._positions[in_interval]

    def filter_positions(self, exclude_positions, positions=None) -> np.ndarray:
        """
        Return positions with *exclude_positions* removed.

        Parameters
        ----------
        exclude_positions : array-like
            Positions to exclude.
        positions : array-like or None
            Positions to filter. Defaults to ``self.positions``.
        """
        if positions is None:
            positions = self._positions
        else:
            positions = np.asarray(positions, dtype=np.int32)
        exclude = np.asarray(exclude_positions, dtype=np.int32)
        return positions[~np.isin(positions, exclude)]

    def _build_permutation(self, sample_identifiers):
        """
        Build an int array mapping output column index to natural
        haplotype column index (or -1 for identifiers not in the view).

        Matches first on (source_name, sample_id, ploidy_index), then
        falls back to (sample_id, ploidy_index) to handle cases where
        the source names differ (e.g. augment_sites).
        """
        natural = self.haplotypes()
        key_to_col = {key: i for i, key in enumerate(natural)}
        # Fallback: (sample_id, ploidy_index) → col index
        sid_ploidy_to_col: dict[tuple[str, int], int] = {}
        for src_name, sid, p in natural:
            short_key = (sid, p)
            if short_key not in sid_ploidy_to_col:
                sid_ploidy_to_col[short_key] = key_to_col[(src_name, sid, p)]

        result = np.full(len(sample_identifiers), -1, dtype=np.intp)
        for i, ident in enumerate(sample_identifiers):
            ident = tuple(ident)
            if ident in key_to_col:
                result[i] = key_to_col[ident]
            else:
                short_key = (ident[1], ident[2])
                if short_key in sid_ploidy_to_col:
                    result[i] = sid_ploidy_to_col[short_key]
        return result

    def iter_genotypes(
        self,
        positions: np.ndarray | None = None,
        *,
        sample_identifiers=None,
    ):
        """
        Yield ``Variant`` instances with canonical allele codes
        (0=ancestral, 1+=derived, -1=missing), concatenated across
        sources. Sources without a variant at a site (including those whose
        filters exclude it) contribute -1 arrays.

        Parameters
        ----------
        positions : array-like or None
            Subset of ``self.positions`` to iterate. None = all.
            Use ``positions_in_intervals`` and ``filter_positions``
            to narrow the set before calling.
        sample_identifiers : list of (source_name, sample_id, ploidy_index) or None
            Reorder output columns to match; unmatched identifiers get -1.
            When None, columns stay in natural source order.
        """
        if positions is None:
            positions = self._positions
        else:
            positions = np.asarray(positions, dtype=np.int32)

        if len(positions) == 0:
            return

        perm = None
        if sample_identifiers is not None:
            perm = self._build_permutation(sample_identifiers)

        site_indexes = np.searchsorted(self._positions, positions)
        source_row = self._source_row[site_indexes]
        alleles_array = self._allele_mapper.forward_map()

        n_sources = len(self._sources)
        source_iters = []
        for src_idx in range(n_sources):
            has_site = source_row[:, src_idx] >= 0
            src_rows = self._variants[src_idx].rows[source_row[has_site, src_idx]]
            source_iters.append(
                iter_raw_genotypes(
                    self._stores[src_idx], src_rows, self._sample_includes[src_idx]
                )
            )

        for j in range(len(positions)):
            parts = []
            for src_idx in range(n_sources):
                row = source_row[j, src_idx]
                if row >= 0:
                    raw = next(source_iters[src_idx])
                    remap = self._remap_tables[src_idx][row]
                    safe = np.clip(raw, 0, len(remap) - 1)
                    encoded = np.where(raw < 0, np.int8(-1), remap[safe])
                    parts.append(encoded.astype(np.int8))
                else:
                    num_haps = self._num_haplotypes_per_source[src_idx]
                    parts.append(np.full(num_haps, -1, dtype=np.int8))

            genotypes = np.concatenate(parts)

            if perm is not None:
                natural_gt = genotypes
                genotypes = np.full(len(perm), np.int8(-1), dtype=np.int8)
                valid = perm >= 0
                genotypes[valid] = natural_gt[perm[valid]]

            # Alleles tuple, trimmed of trailing empty entries
            site_alleles = alleles_array[site_indexes[j]]
            allele_list = [str(a) if a is not None else "" for a in site_alleles]
            while allele_list and allele_list[-1] == "":
                allele_list.pop()

            yield Variant(
                position=float(positions[j]),
                genotypes=genotypes,
                alleles=tuple(allele_list),
            )
