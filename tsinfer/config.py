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

import zarr

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

    Variant filtering uses bcftools-style expressions (delegated to vcztools):
      include/exclude:  e.g. ``"QUAL > 30"``, ``"TYPE='snp'"``
      samples:          e.g. ``"sample_0,sample_1"`` or ``"^sample_2"``
      regions/targets:  e.g. ``"chr20:1000-50000"``

    sample_time is metadata (not filtering) and uses a FieldSpec.
    """

    path: str | Path
    name: str = ""
    include: str | None = None
    exclude: str | None = None
    samples: str | None = None
    regions: str | None = None
    targets: str | None = None
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
    samples_chunk_size: int = 1000
    variants_chunk_size: int = 1000
    genotype_encoding: int = 0  # 0 = eight-bit, 1 = one-bit
    compressor: str = "zstd"  # blosc cname: zstd, lz4, lz4hc, etc.
    compression_level: int = 7
    write_threads: int = 4


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
    groups: str | Path | None = None
    # Path pattern, e.g. "intermediates/group_{group}.trees"
    intermediate_ts: str | None = None


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

    def format(self) -> str:
        """Return the resolved config as a human-readable string."""
        lines = []
        for name, src in self.sources.items():
            lines.append(f"[source.{name}]")
            lines.append(f"  path = {src.path}")
            if src.include is not None:
                lines.append(f"  include = {src.include}")
            if src.exclude is not None:
                lines.append(f"  exclude = {src.exclude}")
            if src.samples is not None:
                lines.append(f"  samples = {src.samples}")
            if src.regions is not None:
                lines.append(f"  regions = {src.regions}")
            if src.targets is not None:
                lines.append(f"  targets = {src.targets}")
            if src.sample_time is not None:
                lines.append(f"  sample_time = {src.sample_time}")
            lines.append("")

        if self.ancestral_state is not None:
            lines.append("[ancestral_state]")
            lines.append(f"  path = {self.ancestral_state.path}")
            lines.append(f"  field = {self.ancestral_state.field}")
            lines.append("")

        if self.ancestors is not None:
            lines.append("[ancestors]")
            lines.append(f"  path = {self.ancestors.path}")
            lines.append(f"  sources = {self.ancestors.sources}")
            lines.append(f"  max_gap_length = {self.ancestors.max_gap_length}")
            enc = "one_bit" if self.ancestors.genotype_encoding == 1 else "eight_bit"
            lines.append(f"  genotype_encoding = {enc}")
            lines.append("")

        lines.append("[match]")
        lines.append(f"  sources = {self.match.sources}")
        lines.append(f"  output = {self.match.output}")
        lines.append(f"  recombination_rate = {self.match.recombination_rate}")
        lines.append(f"  mismatch_ratio = {self.match.mismatch_ratio}")
        lines.append(f"  path_compression = {self.match.path_compression}")
        lines.append(f"  num_threads = {self.match.num_threads}")
        if self.match.reference_ts is not None:
            lines.append(f"  reference_ts = {self.match.reference_ts}")
        if self.match.groups is not None:
            lines.append(f"  groups = {self.match.groups}")
        lines.append("")

        if self.post_process is not None:
            lines.append("[post_process]")
            lines.append(f"  split_ultimate = {self.post_process.split_ultimate}")
            lines.append(f"  erase_flanks = {self.post_process.erase_flanks}")
            lines.append("")

        if self.individual_metadata is not None:
            lines.append("[individual_metadata]")
            lines.append(f"  fields = {self.individual_metadata.fields}")
            if self.individual_metadata.population is not None:
                lines.append(f"  population = {self.individual_metadata.population}")
            lines.append("")

        return "\n".join(lines)

    def validate(self) -> list[str]:
        """Check that all input paths exist. Return list of error strings."""
        errors = []

        for name, src in self.sources.items():
            if src.path is not None:
                p = Path(str(src.path))
                if not p.exists():
                    errors.append(f"Source '{name}' path does not exist: {src.path}")
            if isinstance(src.sample_time, dict) and "path" in src.sample_time:
                if not Path(str(src.sample_time["path"])).exists():
                    errors.append(
                        f"Source '{name}' sample_time path does not "
                        f"exist: {src.sample_time['path']}"
                    )

        # ancestors.path is an output — don't check for existence.
        # But check that ancestral state info is available.
        if self.ancestors is not None and self.ancestral_state is None:
            for src_name in self.ancestors.sources:
                src = self.sources.get(src_name)
                if src is None:
                    errors.append(f"Ancestors references unknown source: '{src_name}'")
                    continue
                if src.path is None:
                    continue
                p = Path(str(src.path))
                if p.exists():
                    try:
                        store = zarr.open(str(p), mode="r")
                        if "variant_ancestral_allele" not in store:
                            errors.append(
                                f"Source '{src_name}' has no "
                                f"'variant_ancestral_allele' array and no "
                                f"[ancestral_state] section is configured"
                            )
                    except Exception:
                        pass  # path existence errors reported above

        if self.match.reference_ts is not None:
            p = Path(str(self.match.reference_ts))
            if not p.exists():
                errors.append(
                    f"Match reference_ts path does not exist: {self.match.reference_ts}"
                )

        if self.match.groups is not None:
            p = Path(str(self.match.groups))
            if not p.exists():
                errors.append(f"Match groups path does not exist: {self.match.groups}")

        if self.ancestral_state is not None:
            p = Path(str(self.ancestral_state.path))
            if not p.exists():
                errors.append(
                    f"Ancestral state path does not exist: {self.ancestral_state.path}"
                )

        return errors

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load and parse a TOML config file; paths resolve relative to cwd."""
        path = Path(path).resolve()
        base = Path.cwd()

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
            include=entry.get("include"),
            exclude=entry.get("exclude"),
            samples=entry.get("samples"),
            regions=entry.get("regions"),
            targets=entry.get("targets"),
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
        genotype_encoding = entry.get("genotype_encoding", 0)
        if isinstance(genotype_encoding, str):
            _encoding_names = {"eight_bit": 0, "one_bit": 1}
            if genotype_encoding.lower() not in _encoding_names:
                raise ValueError(
                    f"[ancestors] genotype_encoding must be 'eight_bit' or "
                    f"'one_bit'; got '{genotype_encoding}'"
                )
            genotype_encoding = _encoding_names[genotype_encoding.lower()]
        return AncestorsConfig(
            path=_resolve_path(entry["path"], base),
            sources=list(entry["sources"]),
            max_gap_length=int(entry.get("max_gap_length", 500_000)),
            samples_chunk_size=int(entry.get("samples_chunk_size", 1000)),
            variants_chunk_size=int(entry.get("variants_chunk_size", 1000)),
            genotype_encoding=int(genotype_encoding),
            compressor=str(entry.get("compressor", "zstd")),
            compression_level=int(entry.get("compression_level", 7)),
            write_threads=int(entry.get("write_threads", 4)),
        )
    except KeyError as e:
        raise ValueError(f"[ancestors] missing required key: {e}") from e


def _parse_match(raw: dict, base: Path) -> MatchConfig:
    entry = raw.get("match")
    if entry is None:
        raise ValueError("Config must contain a [match] section")
    try:
        ref_ts = entry.get("reference_ts")
        groups = entry.get("groups")
        intermediate_ts = entry.get("intermediate_ts")
        if intermediate_ts is not None:
            intermediate_ts = str(Path(base / intermediate_ts))
        return MatchConfig(
            sources=list(entry["sources"]),
            output=_resolve_path(entry["output"], base),
            recombination_rate=float(entry["recombination_rate"]),
            mismatch_ratio=float(entry.get("mismatch_ratio", 1.0)),
            path_compression=bool(entry.get("path_compression", True)),
            num_threads=int(entry.get("num_threads", 1)),
            reference_ts=_resolve_path(ref_ts, base),
            groups=_resolve_path(groups, base),
            intermediate_ts=intermediate_ts,
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
