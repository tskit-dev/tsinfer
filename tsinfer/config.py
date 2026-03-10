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

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Field specification type
# ---------------------------------------------------------------------------

# A metadata field spec is one of:
#   str              — field name within the source store
#   dict             — {"path": ..., "field": ...} from a separate VCZ
#   int | float      — scalar constant broadcast to all sites/samples
#   None             — not specified
FieldSpec = str | dict | int | float | None


def _resolve_path(p: str | Path | None, base: Path) -> str | Path | None:
    """Resolve p relative to base if it looks like a local path.

    Remote URLs (containing ``://``) are returned unchanged as strings.
    Local paths are resolved to absolute Path objects.
    """
    if p is None:
        return None
    s = str(p)
    # Leave remote URLs (e.g. s3://, https://) unchanged
    if "://" in s:
        return s
    return (base / s).resolve()


def _resolve_field_spec(spec: FieldSpec, base: Path) -> FieldSpec:
    """Resolve any path embedded in a field spec dict."""
    if isinstance(spec, dict):
        resolved = dict(spec)
        if "path" in resolved:
            resolved["path"] = _resolve_path(resolved["path"], base)
        return resolved
    return spec


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """
    A named, configured view over a VCZ store.

    Metadata field specs (site_mask, sample_mask, sample_time) can be:
      - str:        field name within the source store
      - dict:       {"path": ..., "field": ...} from a separate VCZ
      - int/float:  scalar constant applied to all sites or samples
      - None:       not specified
    """

    path: str | Path
    name: str = ""
    site_mask: FieldSpec = None
    sample_mask: FieldSpec = None
    sample_time: FieldSpec = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Source must have a non-empty name")


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
    """Configuration for the match step."""

    sources: list[str]
    output: str | Path
    recombination_rate: float | Any  # float or msprime.RateMap
    mismatch_ratio: float = 1.0
    path_compression: bool = True
    num_threads: int = 1
    reference_ts: str | Path | None = None


@dataclass
class PostProcessConfig:
    """Configuration for the post_process step."""

    split_ultimate: bool = True
    erase_flanks: bool = True


@dataclass
class IndividualMetadataConfig:
    """
    Declares how to populate tskit individual metadata from VCZ arrays.

    fields:     tskit metadata field name → VCZ sample-dimensioned array name.
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
    ancestors: AncestorsConfig | None
    match: MatchConfig
    ancestral_state: AncestralState | None = None
    individual_metadata: IndividualMetadataConfig | None = None
    post_process: PostProcessConfig | None = None

    def __post_init__(self):
        if self.ancestors is None and self.match.reference_ts is None:
            raise ValueError(
                "Config must contain either an [ancestors] section "
                "or [match].reference_ts"
            )

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load and parse a TOML config file; paths resolve relative to its location."""
        path = Path(path).resolve()
        base = path.parent

        with open(path, "rb") as f:
            raw = tomllib.load(f)

        return _parse_config(raw, base)


# ---------------------------------------------------------------------------
# TOML parsing helpers
# ---------------------------------------------------------------------------


def _parse_sources(raw: dict, base: Path) -> dict[str, Source]:
    """Parse [[source]] entries into a name-keyed dict."""
    source_list = raw.get("source", [])
    sources = {}
    for entry in source_list:
        entry = dict(entry)
        name = entry.pop("name", None)
        if not name:
            raise ValueError("Every [[source]] entry must have a 'name' field")
        raw_path = entry.pop("path", None)
        if raw_path is None:
            raise ValueError(f"Source '{name}' must have a 'path' field")
        src = Source(
            path=_resolve_path(raw_path, base),
            name=name,
            site_mask=_resolve_field_spec(entry.get("site_mask"), base),
            sample_mask=_resolve_field_spec(entry.get("sample_mask"), base),
            sample_time=_resolve_field_spec(entry.get("sample_time"), base),
        )
        if name in sources:
            raise ValueError(f"Duplicate source name: '{name}'")
        sources[name] = src
    return sources


def _parse_ancestral_state(raw: dict, base: Path) -> AncestralState | None:
    entry = raw.get("ancestral_state")
    if entry is None:
        return None
    try:
        return AncestralState(
            path=_resolve_path(entry["path"], base),
            field=entry["field"],
        )
    except KeyError as e:
        raise ValueError(f"[ancestral_state] missing required key: {e}") from e


def _parse_ancestors(raw: dict, base: Path) -> AncestorsConfig | None:
    entry = raw.get("ancestors")
    if entry is None:
        return None
    try:
        return AncestorsConfig(
            path=_resolve_path(entry["path"], base),
            sources=list(entry["sources"]),
            max_gap_length=int(entry.get("max_gap_length", 500_000)),
        )
    except KeyError as e:
        raise ValueError(f"[ancestors] missing required key: {e}") from e


def _parse_match(raw: dict, base: Path) -> MatchConfig:
    entry = raw.get("match")
    if entry is None:
        raise ValueError("Config must contain a [match] section")
    try:
        ref_ts = entry.get("reference_ts")
        return MatchConfig(
            sources=list(entry["sources"]),
            output=_resolve_path(entry["output"], base),
            recombination_rate=float(entry["recombination_rate"]),
            mismatch_ratio=float(entry.get("mismatch_ratio", 1.0)),
            path_compression=bool(entry.get("path_compression", True)),
            num_threads=int(entry.get("num_threads", 1)),
            reference_ts=_resolve_path(ref_ts, base),
        )
    except KeyError as e:
        raise ValueError(f"[match] missing required key: {e}") from e


def _parse_individual_metadata(raw: dict) -> IndividualMetadataConfig | None:
    entry = raw.get("individual_metadata")
    if entry is None:
        return None
    return IndividualMetadataConfig(
        fields=dict(entry.get("fields", {})),
        population=entry.get("population"),
    )


def _parse_post_process(raw: dict) -> PostProcessConfig | None:
    entry = raw.get("post_process")
    if entry is None:
        return None
    return PostProcessConfig(
        split_ultimate=bool(entry.get("split_ultimate", True)),
        erase_flanks=bool(entry.get("erase_flanks", True)),
    )


def _parse_config(raw: dict, base: Path) -> Config:
    sources = _parse_sources(raw, base)
    ancestors = _parse_ancestors(raw, base)
    match = _parse_match(raw, base)
    return Config(
        sources=sources,
        ancestors=ancestors,
        match=match,
        ancestral_state=_parse_ancestral_state(raw, base),
        individual_metadata=_parse_individual_metadata(raw),
        post_process=_parse_post_process(raw),
    )
