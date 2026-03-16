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
Test data helpers.

All helpers return in-memory zarr Groups (zarr v2 format) so tests require no
filesystem access. String arrays use zarr's VariableLengthUTF8 dtype to match
the convention used by bio2zarr.

Array dimension names (``_ARRAY_DIMENSIONS`` attrs) are set on every array so
that vcztools can interpret the stores as valid VCZ. ``vcz_to_vcf`` converts an
in-memory group to VCF text by calling vcztools internals directly.

Sample VCZ format
-----------------
Follows the bio2zarr VCF Zarr convention:

  call_genotype            (num_sites, num_samples, ploidy)  int8; -1 = missing
  variant_position         (num_sites,)                    int32
  variant_allele           (num_sites, n_alleles)           str
  variant_contig           (num_sites,)                    int8  -- index into contig_id
  variant_ancestral_allele (num_sites,)                    str
  contig_id                (n_contigs,)                  str
  contig_length            (n_contigs,)                  int64
  sample_id                (num_samples,)                  str

Any extra kwargs are stored as additional arrays under their keyword name.
Missing genotypes are encoded as -1 in call_genotype; call_genotype_mask is
not used.

Ancestor VCZ format
-------------------
Follows the format defined in design.md:

  call_genotype          (num_sites, num_ancestors, 1)            int8
  variant_position       (num_sites,)                           int32
  variant_allele         (num_sites, n_alleles)                 str
  sample_id              (num_ancestors,)                       str
  sample_time            (num_ancestors,)                       float64
  sample_start_position  (num_ancestors,)                       int32
  sample_end_position    (num_ancestors,)                       int32
  sample_focal_positions (num_ancestors, max_focal_positions)   int32
  sequence_intervals     (n_intervals, 2)                     int32
"""

from __future__ import annotations

import io

import numpy as np
import zarr
from zarr.core.dtype.npy.string import VariableLengthUTF8

_VLEN_STR = VariableLengthUTF8()
_ZARR_FORMAT = 2


def _open_memory_group() -> zarr.Group:
    store = zarr.storage.MemoryStore()
    return zarr.open_group(store, mode="w", zarr_format=_ZARR_FORMAT)


def _arr(group: zarr.Group, name: str, data: np.ndarray, dims: list[str]) -> zarr.Array:
    """Create a numeric array and set its _ARRAY_DIMENSIONS attribute."""
    a = group.create_array(name, data=data)
    a.attrs["_ARRAY_DIMENSIONS"] = dims
    return a


def _str_array(
    group: zarr.Group, name: str, data: np.ndarray, dims: list[str]
) -> zarr.Array:
    """Create a variable-length string array and set its _ARRAY_DIMENSIONS attribute."""
    data = np.asarray(data)
    a = group.create_array(name, shape=data.shape, dtype=_VLEN_STR)
    a[:] = data
    a.attrs["_ARRAY_DIMENSIONS"] = dims
    return a


def make_sample_vcz(
    genotypes: np.ndarray,
    positions: np.ndarray,
    alleles: np.ndarray,
    ancestral_state: np.ndarray,
    sequence_length: int,
    contig_id: str = "1",
    sample_ids: np.ndarray | None = None,
    **kwargs,
) -> zarr.Group:
    """
    Build an in-memory sample VCZ store.

    Parameters
    ----------
    genotypes:
        Integer array of shape (num_sites, num_samples, ploidy). Use -1 for missing.
    positions:
        Integer array of shape (num_sites,) giving genomic positions.
    alleles:
        String array of shape (num_sites, n_alleles). Allele 0 should be the
        reference allele; ancestral allele is specified separately.
    ancestral_state:
        String array of shape (num_sites,). The ancestral allele at each site.
        Stored as ``variant_ancestral_allele``.
    sequence_length:
        Length of the contig in base pairs.
    contig_id:
        Name of the single contig (default ``"1"``).
    sample_ids:
        String array of shape (num_samples,). Defaults to ``sample_0``, ``sample_1``, …
    **kwargs:
        Additional arrays to store verbatim (e.g. ``sample_time=…``).
        Each is written as a zarr array under its keyword name.
        Dimension names are inferred from shape: first axis matching num_sites gets
        ``"variants"``; first axis matching num_samples gets ``"samples"``.
    """
    genotypes = np.asarray(genotypes, dtype=np.int8)
    positions = np.asarray(positions, dtype=np.int32)
    alleles = np.asarray(alleles)
    ancestral_state = np.asarray(ancestral_state)

    num_sites, num_samples, ploidy = genotypes.shape

    if sample_ids is None:
        sample_ids = np.array([f"sample_{i}" for i in range(num_samples)])

    root = _open_memory_group()

    _arr(root, "call_genotype", genotypes, ["variants", "samples", "ploidy"])
    _arr(root, "variant_position", positions, ["variants"])
    _arr(root, "variant_contig", np.zeros(num_sites, dtype=np.int8), ["variants"])
    _str_array(root, "variant_allele", alleles, ["variants", "alleles"])
    _str_array(root, "variant_ancestral_allele", ancestral_state, ["variants"])
    _str_array(root, "contig_id", np.array([contig_id]), ["contigs"])
    _arr(
        root,
        "contig_length",
        np.array([sequence_length], dtype=np.int64),
        ["contigs"],
    )
    _str_array(root, "sample_id", sample_ids, ["samples"])

    for name, value in kwargs.items():
        arr = np.asarray(value)
        # Infer leading dimension name from shape
        if arr.shape and arr.shape[0] == num_sites:
            leading = ["variants"]
        elif arr.shape and arr.shape[0] == num_samples:
            leading = ["samples"]
        else:
            leading = ["unknown"]
        dims = leading + [f"dim_{i}" for i in range(1, arr.ndim)]
        if arr.dtype.kind in ("U", "S", "O"):
            _str_array(root, name, arr, dims)
        else:
            _arr(root, name, arr, dims)

    return root


def make_ancestor_vcz(
    genotypes: np.ndarray,
    positions: np.ndarray,
    alleles: np.ndarray,
    times: np.ndarray,
    focal_positions: np.ndarray,
    sequence_intervals: np.ndarray,
    sample_ids: np.ndarray | None = None,
    contig_id: str = "1",
    contig_length: int = 1000,
) -> zarr.Group:
    """
    Build an in-memory ancestor VCZ store.

    Parameters
    ----------
    genotypes:
        Integer array of shape (num_sites, num_ancestors, 1). Use -1 for missing
        flanks; 0 for ancestral, 1 for derived within the ancestor's span.
    positions:
        Integer array of shape (num_sites,) — inference site positions.
    alleles:
        String array of shape (num_sites, n_alleles). Allele 0 is ancestral,
        allele 1 is derived.
    times:
        Float array of shape (num_ancestors,). Drives epoch ordering (older = larger).
    focal_positions:
        Integer array of shape (num_ancestors, max_focal_positions). Genomic
        positions of focal sites. Unused slots padded with -2.
    sequence_intervals:
        Integer array of shape (n_intervals, 2) giving [start, end) coordinate
        pairs for regions containing inference sites.
    sample_ids:
        String array of shape (num_ancestors,). Defaults to
        ``ancestor_0``, ``ancestor_1``, …
    """
    genotypes = np.asarray(genotypes, dtype=np.int8)
    positions = np.asarray(positions, dtype=np.int32)
    alleles = np.asarray(alleles)
    times = np.asarray(times, dtype=np.float64)
    focal_positions = np.asarray(focal_positions, dtype=np.int32)
    sequence_intervals = np.asarray(sequence_intervals, dtype=np.int32)

    num_sites, num_ancestors, _ = genotypes.shape

    if sample_ids is None:
        sample_ids = np.array([f"ancestor_{i}" for i in range(num_ancestors)])

    # Derive start/end positions from missing data pattern in genotypes.
    # Within an ancestor's span, values are 0 or 1; flanks are -1.
    start_positions = np.empty(num_ancestors, dtype=np.int32)
    end_positions = np.empty(num_ancestors, dtype=np.int32)
    for i in range(num_ancestors):
        hap = genotypes[:, i, 0]
        non_missing = np.where(hap != -1)[0]
        if len(non_missing) == 0:
            # All-missing ancestor: use first and last position
            start_positions[i] = positions[0]
            end_positions[i] = positions[-1]
        else:
            start_positions[i] = positions[non_missing[0]]
            end_positions[i] = positions[non_missing[-1]]

    root = _open_memory_group()

    _arr(root, "call_genotype", genotypes, ["variants", "samples", "ploidy"])
    _arr(root, "variant_position", positions, ["variants"])
    _str_array(root, "variant_allele", alleles, ["variants", "alleles"])
    _str_array(root, "sample_id", sample_ids, ["samples"])
    _arr(root, "sample_time", times, ["samples"])
    _arr(root, "sample_start_position", start_positions, ["samples"])
    _arr(root, "sample_end_position", end_positions, ["samples"])
    _arr(root, "sample_focal_positions", focal_positions, ["samples", "focal_alleles"])
    _arr(root, "sequence_intervals", sequence_intervals, ["intervals", "coords"])

    # Contig metadata — required for vcztools compatibility
    _str_array(root, "contig_id", np.array([contig_id]), ["contigs"])
    _arr(root, "contig_length", np.array([contig_length], dtype=np.int64), ["contigs"])
    _arr(root, "variant_contig", np.zeros(num_sites, dtype=np.int8), ["variants"])

    return root


def vcz_to_vcf(root: zarr.Group, *, no_version: bool = True) -> str:
    """
    Convert an in-memory VCZ group to VCF text using vcztools internals.

    Returns the full VCF as a string. ``no_version`` suppresses the
    ``##bcftools_view`` header line for deterministic output in tests.
    """
    from vcztools import retrieval
    from vcztools.samples import parse_samples
    from vcztools.utils import _as_fixed_length_string
    from vcztools.vcf_writer import (
        _generate_header,
        c_chunk_to_vcf,
        get_filter_ids,
    )

    all_samples = root["sample_id"][:]
    sample_ids, samples_selection = parse_samples(None, all_samples)

    header = _generate_header(root, sample_ids, no_version=no_version)

    contigs = _as_fixed_length_string(root["contig_id"][:])
    filters = get_filter_ids(root)

    body = io.StringIO()
    for chunk_data in retrieval.variant_chunk_iter(
        root, samples_selection=samples_selection
    ):
        c_chunk_to_vcf(
            chunk_data,
            samples_selection,
            contigs,
            filters,
            body,
            drop_genotypes=False,
            no_update=None,
        )

    return header + body.getvalue()


def ts_to_sample_vcz(
    ts,
    ancestral_allele: str = "REF",
    contig_id: str | None = None,
) -> zarr.Group:
    """
    Convert a tskit TreeSequence to an in-memory sample VCZ store.

    Each tskit individual becomes one sample; ploidy is inferred from the
    number of nodes per individual (falls back to 1 if no individuals).

    Parameters
    ----------
    ts:
        A ``tskit.TreeSequence``.
    ancestral_allele:
        Controls which allele is used as the ancestral state.
        ``"REF"`` (default) uses allele 0 at each site; ``"ANCESTRAL"`` uses
        ``site.ancestral_state`` where available, falling back to allele 0.
    contig_id:
        Name to use for the single contig. Defaults to ``"1"``.
    """
    if ts.num_sites == 0:
        raise ValueError("Tree sequence has no sites")

    # Determine ploidy and sample layout
    if ts.num_individuals > 0:
        ploidy = len(list(ts.individuals())[0].nodes)
        num_samples = ts.num_individuals
        sample_nodes = np.array(
            [node for ind in ts.individuals() for node in ind.nodes], dtype=np.int32
        )
        sample_ids = np.array(
            [f"tsk_{i}" for i in range(num_samples)],
        )
    else:
        ploidy = 1
        num_samples = ts.num_samples
        sample_nodes = np.array(ts.samples(), dtype=np.int32)
        sample_ids = np.array([f"tsk_{i}" for i in range(num_samples)])

    num_sites = ts.num_sites

    # Collect variants
    positions = np.empty(num_sites, dtype=np.int32)
    # Determine max alleles across all sites
    max_alleles = max(len(v.alleles) for v in ts.variants())
    all_alleles = np.empty((num_sites, max_alleles), dtype=object)
    all_alleles[:] = ""
    anc_state = np.empty(num_sites, dtype=object)
    genotypes = np.empty((num_sites, num_samples, ploidy), dtype=np.int8)

    for s_idx, variant in enumerate(ts.variants(samples=sample_nodes)):
        positions[s_idx] = int(variant.site.position)
        site_alleles = variant.alleles  # tuple of str, None for missing
        for a_idx, allele in enumerate(site_alleles):
            all_alleles[s_idx, a_idx] = allele if allele is not None else ""

        if ancestral_allele == "ANCESTRAL":
            anc = variant.site.ancestral_state
            anc_state[s_idx] = anc if anc else site_alleles[0]
        else:
            anc_state[s_idx] = site_alleles[0]

        # Reshape flat haplotype genotypes to (num_samples, ploidy)
        gt_flat = variant.genotypes  # (num_samples * ploidy,)
        genotypes[s_idx] = gt_flat.reshape(num_samples, ploidy)

    sequence_length = int(ts.sequence_length)
    if contig_id is None:
        contig_id = "1"

    return make_sample_vcz(
        genotypes=genotypes,
        positions=positions,
        alleles=all_alleles,
        ancestral_state=anc_state,
        sequence_length=sequence_length,
        contig_id=contig_id,
        sample_ids=sample_ids,
    )
