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
Configuration dataclasses and TOML loading for the tsinfer pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Source:
    """
    A named, configured view over a VCZ store.

    Metadata arrays (site_mask, sample_mask, sample_time) can be specified as:
      - str: field name within the source store
      - dict with {path, field}: field from a separate VCZ joined by position/sample_id
      - scalar: constant applied to all sites or samples
      - None: not specified
    """

    path: str | Path
    name: str = ""
    site_mask: str | dict | None = None
    sample_mask: str | dict | None = None
    sample_time: str | dict | float | None = None


@dataclass
class AncestralState:
    """Specifies where to read the ancestral allele for each variant position."""

    path: str | Path
    field: str


@dataclass
class AncestorsConfig:
    """Configuration for the infer_ancestors step."""

    path: str | Path
    sources: list[str]
    max_gap_length: int = 500_000


@dataclass
class MatchConfig:
    """Configuration for a match step."""

    sources: list[str]
    output: str | Path
    recombination_rate: float | Any  # float or msprime.RateMap
    mismatch_ratio: float = 1.0
    path_compression: bool = True
    num_threads: int = 1


@dataclass
class IndividualMetadataConfig:
    """
    Declares how to populate tskit individual metadata from VCZ arrays.

    fields: mapping from tskit metadata field name → VCZ sample-dimensioned array name.
    population: VCZ array name whose unique values become tskit populations.
    """

    fields: dict[str, str] = field(default_factory=dict)
    population: str | None = None


@dataclass
class Config:
    """
    Full pipeline configuration. Constructed directly or via Config.from_toml().

    sources: mapping from source name → Source
    """

    sources: dict[str, Source]
    ancestors: AncestorsConfig
    match: MatchConfig
    ancestral_state: AncestralState | None = None
    individual_metadata: IndividualMetadataConfig | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load and parse a TOML config file; paths resolve relative to its location."""
        raise NotImplementedError
