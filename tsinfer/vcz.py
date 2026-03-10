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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr


def open_store(path_or_group: str | Path | zarr.Group) -> zarr.Group:
    """
    Return a zarr Group for the given path or pass through an existing Group.
    Accepts local paths, remote URLs, or in-memory zarr Groups (for testing).
    """
    raise NotImplementedError


def resolve_field(
    store: zarr.Group,
    field_spec: str | dict | float | None,
    join_key: str,
    n: int,
    fill_value: Any = None,
) -> np.ndarray | None:
    """
    Resolve a metadata field specification to a numpy array of length n.

    field_spec may be:
      - str: name of an array within store
      - dict with {path, field}: array from a separate VCZ, joined by join_key
      - scalar: broadcast to all n entries
      - None: return None

    join_key is "variant_position" (site arrays) or "sample_id" (sample arrays).
    Missing entries in a join get fill_value (None → site/sample included; 0 for time).
    """
    raise NotImplementedError
