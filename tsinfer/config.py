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


def _resolve_path(p: str | Path | None) -> str | Path | None:
    """Return *p* as-is (string or Path).

    Remote URLs (containing ``://``) are returned unchanged as strings.
    Local paths are returned as strings without resolution — the caller
    is responsible for interpreting them relative to the working directory.
    """
    if p is None:
        return None
    s = str(p)
    if "://" in s:
        return s
    return s


def _resolve_field_spec(spec: FieldSpec) -> FieldSpec:
    """Pass through a field spec, resolving any embedded path."""
    if isinstance(spec, dict):
        resolved = dict(spec)
        if "path" in resolved:
            resolved["path"] = _resolve_path(resolved["path"])
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

    name: str
    path: str | Path
    sources: list[str]
    max_gap_length: int = 500_000
    samples_chunk_size: int = 1000
    variants_chunk_size: int = 1000
    genotype_encoding: int = 0  # 0 = eight-bit, 1 = one-bit
    compressor: str = "zstd"  # blosc cname: zstd, lz4, lz4hc, etc.
    compression_level: int = 7


@dataclass
class MatchSourceConfig:
    """Per-source parameters for the match step."""

    node_flags: int = 1  # tskit.NODE_IS_SAMPLE
    create_individuals: bool = True


@dataclass
class MatchConfig:
    """Configuration for the match step."""

    sources: dict[str, MatchSourceConfig]
    output: str | Path
    path_compression: bool = True
    reference_ts: str | Path | None = None
    workdir: str | Path | None = None
    keep_intermediates: bool = False


@dataclass
class PostProcessConfig:
    """Configuration for the post_process step."""

    split_ultimate: bool = True
    erase_flanks: bool = True


@dataclass
class AugmentSitesConfig:
    """Configuration for the augment_sites step."""

    sources: list[str]


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
    ancestors: list[AncestorsConfig]
    match: MatchConfig
    ancestral_state: AncestralState | None = None
    individual_metadata: IndividualMetadataConfig | None = None
    post_process: PostProcessConfig | None = None
    augment_sites: AugmentSitesConfig | None = None

    def __post_init__(self):
        if len(self.ancestors) == 0 and self.match.reference_ts is None:
            raise ValueError(
                "Config must contain either an [[ancestors]] section "
                "or [match].reference_ts"
            )
        if self.match.keep_intermediates and self.match.workdir is None:
            raise ValueError("keep_intermediates requires workdir to be set")
        for anc in self.ancestors:
            if anc.name not in self.match.sources:
                raise ValueError(f"Ancestor '{anc.name}' must appear in [match.sources]")

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

        for anc in self.ancestors:
            lines.append(f"[[ancestors]] (name={anc.name})")
            lines.append(f"  path = {anc.path}")
            lines.append(f"  sources = {anc.sources}")
            lines.append(f"  max_gap_length = {anc.max_gap_length}")
            enc = "one_bit" if anc.genotype_encoding == 1 else "eight_bit"
            lines.append(f"  genotype_encoding = {enc}")
            lines.append("")

        lines.append("[match]")
        lines.append(f"  output = {self.match.output}")
        for src_name, src_cfg in self.match.sources.items():
            lines.append(f"  [match.sources.{src_name}]")
            lines.append(f"    node_flags = {src_cfg.node_flags}")
            lines.append(f"    create_individuals = {src_cfg.create_individuals}")
        lines.append(f"  path_compression = {self.match.path_compression}")
        if self.match.reference_ts is not None:
            lines.append(f"  reference_ts = {self.match.reference_ts}")
        if self.match.workdir is not None:
            lines.append(f"  workdir = {self.match.workdir}")
        if self.match.keep_intermediates:
            lines.append(f"  keep_intermediates = {self.match.keep_intermediates}")
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

        if self.augment_sites is not None:
            lines.append("[augment_sites]")
            lines.append(f"  sources = {self.augment_sites.sources}")
            lines.append("")

        return "\n".join(lines)

    def validate(self) -> list[str]:
        """Check that all input paths exist. Return list of error strings."""
        errors = []

        # Ancestor source paths are outputs — skip existence checks for them
        ancestor_names = {anc.name for anc in self.ancestors}

        for name, src in self.sources.items():
            if name in ancestor_names:
                continue  # ancestor paths are outputs, not inputs
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
        for anc in self.ancestors:
            if self.ancestral_state is None:
                for src_name in anc.sources:
                    src = self.sources.get(src_name)
                    if src is None:
                        errors.append(
                            f"Ancestors '{anc.name}' references unknown "
                            f"source: '{src_name}'"
                        )
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

        if self.ancestral_state is not None:
            p = Path(str(self.ancestral_state.path))
            if not p.exists():
                errors.append(
                    f"Ancestral state path does not exist: {self.ancestral_state.path}"
                )

        return errors

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load and parse a TOML config file."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        return _parse_config(raw)


# ---------------------------------------------------------------------------
# TOML parsing helpers
# ---------------------------------------------------------------------------


_KNOWN_TOP_LEVEL_KEYS = {
    "source",
    "ancestral_state",
    "ancestors",
    "match",
    "individual_metadata",
    "post_process",
    "augment_sites",
}

_KNOWN_SOURCE_KEYS = {
    "name",
    "path",
    "include",
    "exclude",
    "samples",
    "regions",
    "targets",
    "sample_time",
}

_KNOWN_ANCESTRAL_STATE_KEYS = {"path", "field"}

_KNOWN_ANCESTORS_KEYS = {
    "name",
    "path",
    "sources",
    "max_gap_length",
    "samples_chunk_size",
    "variants_chunk_size",
    "genotype_encoding",
    "compressor",
    "compression_level",
}

_KNOWN_MATCH_KEYS = {
    "sources",
    "output",
    "path_compression",
    "reference_ts",
    "workdir",
    "keep_intermediates",
}

_KNOWN_MATCH_SOURCE_KEYS = {"node_flags", "create_individuals"}

_KNOWN_INDIVIDUAL_METADATA_KEYS = {"fields", "population"}

_KNOWN_POST_PROCESS_KEYS = {"split_ultimate", "erase_flanks"}

_KNOWN_AUGMENT_SITES_KEYS = {"sources"}


def _check_unknown_keys(section_name, entry, known):
    """Raise ValueError if *entry* contains keys not in *known*."""
    unknown = set(entry.keys()) - known
    if unknown:
        unknown_str = ", ".join(sorted(unknown))
        raise ValueError(f"Unrecognised key(s) in [{section_name}]: {unknown_str}")


def _parse_sources(raw: dict) -> dict[str, Source]:
    """Parse [[source]] entries into a name-keyed dict."""
    source_list = raw.get("source", [])
    sources = {}
    for entry in source_list:
        entry = dict(entry)
        _check_unknown_keys("source", entry, _KNOWN_SOURCE_KEYS)
        name = entry.pop("name", None)
        if not name:
            raise ValueError("Every [[source]] entry must have a 'name' field")
        raw_path = entry.pop("path", None)
        if raw_path is None:
            raise ValueError(f"Source '{name}' must have a 'path' field")
        src = Source(
            path=_resolve_path(raw_path),
            name=name,
            include=entry.get("include"),
            exclude=entry.get("exclude"),
            samples=entry.get("samples"),
            regions=entry.get("regions"),
            targets=entry.get("targets"),
            sample_time=_resolve_field_spec(entry.get("sample_time")),
        )
        if name in sources:
            raise ValueError(f"Duplicate source name: '{name}'")
        sources[name] = src
    return sources


def _parse_ancestral_state(raw: dict) -> AncestralState | None:
    entry = raw.get("ancestral_state")
    if entry is None:
        return None
    _check_unknown_keys("ancestral_state", entry, _KNOWN_ANCESTRAL_STATE_KEYS)
    try:
        return AncestralState(
            path=_resolve_path(entry["path"]),
            field=entry["field"],
        )
    except KeyError as e:
        raise ValueError(f"[ancestral_state] missing required key: {e}") from e


def _parse_one_ancestor(entry: dict) -> AncestorsConfig:
    """Parse a single ancestor config entry."""
    _check_unknown_keys("ancestors", entry, _KNOWN_ANCESTORS_KEYS)
    try:
        name = entry.get("name", "ancestors")
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
            name=str(name),
            path=_resolve_path(entry["path"]),
            sources=list(entry["sources"]),
            max_gap_length=int(entry.get("max_gap_length", 500_000)),
            samples_chunk_size=int(entry.get("samples_chunk_size", 1000)),
            variants_chunk_size=int(entry.get("variants_chunk_size", 1000)),
            genotype_encoding=int(genotype_encoding),
            compressor=str(entry.get("compressor", "zstd")),
            compression_level=int(entry.get("compression_level", 7)),
        )
    except KeyError as e:
        raise ValueError(f"[ancestors] missing required key: {e}") from e


def _parse_ancestors(raw: dict) -> list[AncestorsConfig]:
    entry = raw.get("ancestors")
    if entry is None:
        return []
    # TOML [[ancestors]] produces a list; [ancestors] produces a dict
    if isinstance(entry, dict):
        return [_parse_one_ancestor(entry)]
    if isinstance(entry, list):
        return [_parse_one_ancestor(e) for e in entry]
    raise ValueError("[ancestors] must be a table or array of tables")


def _parse_match(raw: dict) -> MatchConfig:
    entry = raw.get("match")
    if entry is None:
        raise ValueError("Config must contain a [match] section")
    _check_unknown_keys("match", entry, _KNOWN_MATCH_KEYS)
    try:
        raw_sources = entry["sources"]
        if not isinstance(raw_sources, dict):
            raise ValueError(
                "[match] sources must be a table of tables, e.g. [match.sources.cohort]"
            )
        sources: dict[str, MatchSourceConfig] = {}
        for src_name, src_val in raw_sources.items():
            if isinstance(src_val, dict):
                _check_unknown_keys(
                    f"match.sources.{src_name}",
                    src_val,
                    _KNOWN_MATCH_SOURCE_KEYS,
                )
                sources[src_name] = MatchSourceConfig(
                    node_flags=int(src_val.get("node_flags", 1)),
                    create_individuals=bool(src_val.get("create_individuals", True)),
                )
            else:
                sources[src_name] = MatchSourceConfig()
        return MatchConfig(
            sources=sources,
            output=_resolve_path(entry["output"]),
            path_compression=bool(entry.get("path_compression", True)),
            reference_ts=_resolve_path(entry.get("reference_ts")),
            workdir=_resolve_path(entry.get("workdir")),
            keep_intermediates=bool(entry.get("keep_intermediates", False)),
        )
    except KeyError as e:
        raise ValueError(f"[match] missing required key: {e}") from e


def _parse_individual_metadata(raw: dict) -> IndividualMetadataConfig | None:
    entry = raw.get("individual_metadata")
    if entry is None:
        return None
    _check_unknown_keys("individual_metadata", entry, _KNOWN_INDIVIDUAL_METADATA_KEYS)
    return IndividualMetadataConfig(
        fields=dict(entry.get("fields", {})),
        population=entry.get("population"),
    )


def _parse_post_process(raw: dict) -> PostProcessConfig | None:
    entry = raw.get("post_process")
    if entry is None:
        return None
    _check_unknown_keys("post_process", entry, _KNOWN_POST_PROCESS_KEYS)
    return PostProcessConfig(
        split_ultimate=bool(entry.get("split_ultimate", True)),
        erase_flanks=bool(entry.get("erase_flanks", True)),
    )


def _parse_augment_sites(raw: dict) -> AugmentSitesConfig | None:
    entry = raw.get("augment_sites")
    if entry is None:
        return None
    _check_unknown_keys("augment_sites", entry, _KNOWN_AUGMENT_SITES_KEYS)
    sources = entry.get("sources")
    if sources is None:
        raise ValueError("[augment_sites] missing required key: 'sources'")
    if not isinstance(sources, list):
        raise ValueError("[augment_sites] sources must be a list of source names")
    return AugmentSitesConfig(sources=list(sources))


def _parse_config(raw: dict) -> Config:
    _check_unknown_keys("top-level", raw, _KNOWN_TOP_LEVEL_KEYS)
    sources = _parse_sources(raw)
    ancestors = _parse_ancestors(raw)
    match = _parse_match(raw)

    # Auto-create a Source for each ancestor config
    for anc in ancestors:
        if anc.name in sources:
            raise ValueError(
                f"Ancestor name '{anc.name}' conflicts with a [[source]] name"
            )
        sources[anc.name] = Source(
            path=anc.path,
            name=anc.name,
            sample_time="sample_time",
        )

    return Config(
        sources=sources,
        ancestors=ancestors,
        match=match,
        ancestral_state=_parse_ancestral_state(raw),
        individual_metadata=_parse_individual_metadata(raw),
        post_process=_parse_post_process(raw),
        augment_sites=_parse_augment_sites(raw),
    )
