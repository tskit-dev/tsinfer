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
Tests for the Config system: direct construction, TOML parsing, path
resolution, field specs, and validation errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tsinfer.config import (
    AncestorsConfig,
    AncestralState,
    Config,
    IndividualMetadataConfig,
    MatchConfig,
    PostProcessConfig,
    Source,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "inference.toml"
    p.write_text(content)
    return p


def _minimal_match_cfg(**kwargs):
    defaults = dict(
        sources=["ancestors", "cohort"],
        output="out.trees",
        recombination_rate=1e-8,
    )
    defaults.update(kwargs)
    return MatchConfig(**defaults)


def _minimal_ancestors_cfg(**kwargs):
    defaults = dict(path="ancestors.vcz", sources=["cohort"])
    defaults.update(kwargs)
    return AncestorsConfig(**defaults)


def _minimal_config(**kwargs):
    defaults = dict(
        sources={"cohort": Source(path="samples.vcz", name="cohort")},
        ancestors=_minimal_ancestors_cfg(),
        match=_minimal_match_cfg(),
    )
    defaults.update(kwargs)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# Direct construction
# ---------------------------------------------------------------------------


class TestSourceConstruction:
    def test_basic(self):
        s = Source(path="samples.vcz", name="cohort")
        assert s.name == "cohort"
        assert str(s.path) == "samples.vcz"
        assert s.site_mask is None
        assert s.sample_mask is None
        assert s.sample_time is None

    def test_site_mask_string(self):
        s = Source(path="s.vcz", name="s", site_mask="variant_filter")
        assert s.site_mask == "variant_filter"

    def test_site_mask_dict(self):
        spec = {"path": "ann.vcz", "field": "filter"}
        s = Source(path="s.vcz", name="s", site_mask=spec)
        assert s.site_mask == spec

    def test_sample_time_scalar(self):
        s = Source(path="s.vcz", name="ancient", sample_time=1.5)
        assert s.sample_time == 1.5

    def test_sample_time_string(self):
        s = Source(path="s.vcz", name="s", sample_time="sample_age")
        assert s.sample_time == "sample_age"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            Source(path="s.vcz", name="")

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            Source(path="s.vcz")


class TestAncestralStateConstruction:
    def test_basic(self):
        a = AncestralState(path="ann.vcz", field="variant_ancestral_allele")
        assert str(a.path) == "ann.vcz"
        assert a.field == "variant_ancestral_allele"


class TestAncestorsConfigConstruction:
    def test_basic(self):
        a = AncestorsConfig(path="anc.vcz", sources=["cohort"])
        assert a.sources == ["cohort"]
        assert a.max_gap_length == 500_000

    def test_custom_gap_length(self):
        a = AncestorsConfig(path="anc.vcz", sources=["cohort"], max_gap_length=1_000)
        assert a.max_gap_length == 1_000


class TestMatchConfigConstruction:
    def test_basic(self):
        m = MatchConfig(
            sources=["ancestors", "cohort"],
            output="out.trees",
            recombination_rate=1e-8,
        )
        assert m.mismatch_ratio == 1.0
        assert m.path_compression is True
        assert m.num_threads == 1
        assert m.reference_ts is None

    def test_with_reference_ts(self):
        m = MatchConfig(
            sources=["cohort"],
            output="out.trees",
            recombination_rate=1e-8,
            reference_ts="ref.trees",
        )
        assert str(m.reference_ts) == "ref.trees"


class TestPostProcessConfigConstruction:
    def test_defaults(self):
        p = PostProcessConfig()
        assert p.split_ultimate is True
        assert p.erase_flanks is True

    def test_custom(self):
        p = PostProcessConfig(split_ultimate=False, erase_flanks=False)
        assert p.split_ultimate is False


class TestIndividualMetadataConfigConstruction:
    def test_defaults(self):
        m = IndividualMetadataConfig()
        assert m.fields == {}
        assert m.population is None

    def test_with_fields(self):
        m = IndividualMetadataConfig(
            fields={"sample_id": "sample_id", "sex": "sample_sex"},
            population="sample_population",
        )
        assert m.fields["sex"] == "sample_sex"
        assert m.population == "sample_population"


class TestConfigConstruction:
    def test_basic(self):
        cfg = _minimal_config()
        assert "cohort" in cfg.sources
        assert cfg.ancestral_state is None
        assert cfg.individual_metadata is None
        assert cfg.post_process is None

    def test_no_ancestors_no_reference_ts_raises(self):
        with pytest.raises(ValueError, match="ancestors"):
            Config(
                sources={},
                ancestors=None,
                match=MatchConfig(
                    sources=["cohort"],
                    output="out.trees",
                    recombination_rate=1e-8,
                ),
            )

    def test_reference_ts_without_ancestors_ok(self):
        cfg = Config(
            sources={"cohort": Source(path="s.vcz", name="cohort")},
            ancestors=None,
            match=MatchConfig(
                sources=["cohort"],
                output="out.trees",
                recombination_rate=1e-8,
                reference_ts="ref.trees",
            ),
        )
        assert cfg.ancestors is None
        assert cfg.match.reference_ts is not None


# ---------------------------------------------------------------------------
# TOML parsing — standard case
# ---------------------------------------------------------------------------

_STANDARD_TOML = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name      = "cohort"
path      = "samples.vcz"
site_mask = "variant_filter"

[ancestors]
path           = "ancestors.vcz"
sources        = ["cohort"]
max_gap_length = 500000

[match]
sources            = ["ancestors", "cohort"]
output             = "final.trees"
recombination_rate = 1e-8
mismatch_ratio     = 1.0

[individual_metadata]
fields     = {sample_id = "sample_id", sex = "sample_sex"}
population = "sample_population"

[post_process]
split_ultimate = true
erase_flanks   = true
"""


class TestFromTomlStandard:
    def test_loads_without_error(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert isinstance(cfg, Config)

    def test_sources(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert "cohort" in cfg.sources
        assert cfg.sources["cohort"].site_mask == "variant_filter"

    def test_ancestral_state(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.ancestral_state is not None
        assert cfg.ancestral_state.field == "variant_ancestral_allele"

    def test_ancestors(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.ancestors is not None
        assert cfg.ancestors.sources == ["cohort"]
        assert cfg.ancestors.max_gap_length == 500_000

    def test_match(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.match.sources == ["ancestors", "cohort"]
        assert cfg.match.recombination_rate == pytest.approx(1e-8)
        assert cfg.match.mismatch_ratio == pytest.approx(1.0)

    def test_individual_metadata(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.individual_metadata is not None
        assert cfg.individual_metadata.fields["sex"] == "sample_sex"
        assert cfg.individual_metadata.population == "sample_population"

    def test_post_process(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.post_process is not None
        assert cfg.post_process.split_ultimate is True
        assert cfg.post_process.erase_flanks is True


# ---------------------------------------------------------------------------
# TOML parsing — path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_source_path_is_absolute(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.sources["cohort"].path.is_absolute()

    def test_source_path_resolved_relative_to_config(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.sources["cohort"].path == tmp_path / "samples.vcz"

    def test_ancestors_path_resolved(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.ancestors.path == tmp_path / "ancestors.vcz"

    def test_match_output_resolved(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.match.output == tmp_path / "final.trees"

    def test_ancestral_state_path_resolved(self, tmp_path):
        cfg = Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.ancestral_state.path == tmp_path / "annotations.vcz"

    def test_config_in_subdirectory(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        toml_path = subdir / "inference.toml"
        toml_path.write_text(_STANDARD_TOML)
        cfg = Config.from_toml(toml_path)
        assert cfg.sources["cohort"].path == subdir / "samples.vcz"

    def test_remote_url_unchanged(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "s3://bucket/samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
sources            = ["ancestors", "cohort"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        cfg = Config.from_toml(_write_toml(tmp_path, toml))
        assert str(cfg.sources["cohort"].path) == "s3://bucket/samples.vcz"

    def test_reference_ts_path_resolved(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "samples.vcz"

[match]
sources            = ["cohort"]
output             = "out.trees"
recombination_rate = 1e-8
reference_ts       = "ref.trees"
"""
        cfg = Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.match.reference_ts == tmp_path / "ref.trees"


# ---------------------------------------------------------------------------
# TOML parsing — field specs
# ---------------------------------------------------------------------------


class TestFieldSpecs:
    def test_site_mask_string(self, tmp_path):
        toml = """\
[[source]]
name      = "cohort"
path      = "samples.vcz"
site_mask = "variant_filter"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
sources            = ["ancestors", "cohort"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        cfg = Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["cohort"].site_mask == "variant_filter"

    def test_site_mask_dict(self, tmp_path):
        toml = """\
[[source]]
name      = "cohort"
path      = "samples.vcz"
site_mask = {path = "ann.vcz", field = "variant_filter"}

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
sources            = ["ancestors", "cohort"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        cfg = Config.from_toml(_write_toml(tmp_path, toml))
        spec = cfg.sources["cohort"].site_mask
        assert isinstance(spec, dict)
        assert spec["field"] == "variant_filter"
        assert spec["path"] == tmp_path / "ann.vcz"

    def test_sample_time_scalar(self, tmp_path):
        toml = """\
[[source]]
name        = "parents"
path        = "parents.vcz"
sample_time = 1

[ancestors]
path    = "ancestors.vcz"
sources = ["parents"]

[match]
sources            = ["ancestors", "parents"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        cfg = Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["parents"].sample_time == 1

    def test_sample_time_string(self, tmp_path):
        toml = """\
[[source]]
name        = "ancient"
path        = "ancient.vcz"
sample_time = "sample_age_generations"

[ancestors]
path    = "ancestors.vcz"
sources = ["ancient"]

[match]
sources            = ["ancestors", "ancient"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        cfg = Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["ancient"].sample_time == "sample_age_generations"

    def test_multiple_sources(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "samples.vcz"

[[source]]
name        = "ancient"
path        = "ancient.vcz"
sample_time = "sample_age"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
sources            = ["ancestors", "cohort", "ancient"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        cfg = Config.from_toml(_write_toml(tmp_path, toml))
        assert set(cfg.sources) == {"cohort", "ancient"}
        assert cfg.match.sources == ["ancestors", "cohort", "ancient"]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_missing_match_section(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "samples.vcz"
"""
        with pytest.raises(ValueError, match=r"\[match\]"):
            Config.from_toml(_write_toml(tmp_path, toml))

    def test_match_missing_sources(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"
recombination_rate = 1e-8
"""
        with pytest.raises((ValueError, KeyError)):
            Config.from_toml(_write_toml(tmp_path, toml))

    def test_match_missing_recombination_rate(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
sources = ["ancestors", "cohort"]
output  = "out.trees"
"""
        with pytest.raises((ValueError, KeyError)):
            Config.from_toml(_write_toml(tmp_path, toml))

    def test_source_missing_name(self, tmp_path):
        toml = """\
[[source]]
path = "samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
sources            = ["ancestors", "cohort"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        with pytest.raises(ValueError, match="name"):
            Config.from_toml(_write_toml(tmp_path, toml))

    def test_source_missing_path(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
sources            = ["ancestors", "cohort"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        with pytest.raises(ValueError, match="path"):
            Config.from_toml(_write_toml(tmp_path, toml))

    def test_duplicate_source_names(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "a.vcz"

[[source]]
name = "cohort"
path = "b.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
sources            = ["ancestors", "cohort"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        with pytest.raises(ValueError, match="Duplicate"):
            Config.from_toml(_write_toml(tmp_path, toml))

    def test_no_ancestors_no_reference_ts(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "samples.vcz"

[match]
sources            = ["cohort"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        with pytest.raises(ValueError, match="ancestors"):
            Config.from_toml(_write_toml(tmp_path, toml))

    def test_ancestors_missing_path(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "samples.vcz"

[ancestors]
sources = ["cohort"]

[match]
sources            = ["ancestors", "cohort"]
output             = "out.trees"
recombination_rate = 1e-8
"""
        with pytest.raises((ValueError, KeyError)):
            Config.from_toml(_write_toml(tmp_path, toml))
