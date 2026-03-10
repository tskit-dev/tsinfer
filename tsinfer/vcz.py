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
Low-level utilities for reading VCZ (VCF Zarr) stores.

``open_store``
    Open a VCZ store from a path/URL, or pass through an existing zarr Group.

``resolve_field``
    Map a field specification (string, {path,field} dict, scalar, or None)
    to a numpy array.  Used to resolve per-source metadata like site_mask,
    sample_mask, and sample_time.

``sequence_length``
    Read the contig length from a VCZ store (``contig_length[0]``).

``num_contigs``
    Return the number of contigs declared in a VCZ store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------


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


def sequence_length(store: zarr.Group) -> int:
    """Return the sequence length (first contig length) from a VCZ store."""
    return int(store["contig_length"][0])


def num_contigs(store: zarr.Group) -> int:
    """Return the number of contigs declared in a VCZ store."""
    return int(store["contig_id"].shape[0])
