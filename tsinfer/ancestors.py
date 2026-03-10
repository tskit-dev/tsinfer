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
Ancestor generation: builds the ancestor VCZ store from a samples VCZ store.
"""

from __future__ import annotations

import zarr

from .config import AncestorsConfig, AncestralState, Source


def compute_inference_sites(
    store: zarr.Group,
    site_mask: object,
    ancestral_state: AncestralState | None,
) -> tuple:
    """
    Return (positions, alleles, anc_state_indices) for sites that pass the mask
    and have a valid ancestral state.

    positions: (n_sites,) int32
    alleles:   (n_sites, n_alleles) str
    anc_index: (n_sites,) int8 — index of ancestral allele within alleles
    """
    raise NotImplementedError


def compute_sequence_intervals(
    positions: object,
    sequence_length: int,
    max_gap_length: int,
) -> object:
    """
    Return (n_intervals, 2) int32 array of [start, end) pairs for regions
    containing inference sites, splitting on gaps longer than max_gap_length.
    """
    raise NotImplementedError


def infer_ancestors(
    source: Source | zarr.Group,
    cfg: AncestorsConfig,
    ancestral_state: AncestralState | None = None,
) -> zarr.Group:
    """
    Build the ancestor VCZ store from a samples VCZ store.

    Reads genotypes from source, computes inference sites, runs the C
    AncestorBuilder, applies gap clipping, and writes the ancestor VCZ format.
    Returns the ancestor zarr Group (backed by MemoryStore if cfg.path is None,
    otherwise written to disk and returned).

    Steps:
      1. Open source store, apply site_mask and sample_mask.
      2. Compute inference sites (compute_inference_sites).
      3. Compute sequence_intervals (compute_sequence_intervals).
      4. Run _tsinfer.AncestorBuilder to get raw ancestor haplotypes.
      5. Clip ancestors to their containing interval; split any that span a gap.
      6. Write and return the ancestor VCZ store.
    """
    raise NotImplementedError
