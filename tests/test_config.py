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

import pathlib

import helpers
import numpy as np
import pytest
import zarr

from tsinfer import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    p = tmp_path / "inference.toml"
    p.write_text(content)
    return p


def _minimal_match_cfg(**kwargs):
    defaults = dict(
        sources={
            "ancestors": config.MatchSourceConfig(
                node_flags=0, create_individuals=False
            ),
            "cohort": config.MatchSourceConfig(),
        },
        output="out.trees",
    )
    defaults.update(kwargs)
    return config.MatchConfig(**defaults)


def _minimal_ancestors_cfg(**kwargs):
    defaults = dict(name="ancestors", path="ancestors.vcz", sources=["cohort"])
    defaults.update(kwargs)
    return config.AncestorsConfig(**defaults)


def _minimal_config(**kwargs):
    defaults = dict(
        sources={
            "cohort": config.Source(path="samples.vcz", name="cohort"),
            "ancestors": config.Source(
                path="ancestors.vcz", name="ancestors", sample_time="sample_time"
            ),
        },
        ancestors=[_minimal_ancestors_cfg()],
        match=_minimal_match_cfg(),
        ancestral_state=config.AncestralState(
            path="annotations.vcz", field="variant_ancestral_allele"
        ),
    )
    defaults.update(kwargs)
    return config.Config(**defaults)


# ---------------------------------------------------------------------------
# Direct construction
# ---------------------------------------------------------------------------


class TestSourceConstruction:
    def test_basic(self):
        s = config.Source(path="samples.vcz", name="cohort")
        assert s.name == "cohort"
        assert str(s.path) == "samples.vcz"
        assert s.include is None
        assert s.exclude is None
        assert s.samples is None
        assert s.regions is None
        assert s.targets is None
        assert s.sample_time is None

    def test_include_expression(self):
        s = config.Source(path="s.vcz", name="s", include="QUAL > 30")
        assert s.include == "QUAL > 30"

    def test_exclude_expression(self):
        s = config.Source(path="s.vcz", name="s", exclude="AC == 0")
        assert s.exclude == "AC == 0"

    def test_samples_string(self):
        s = config.Source(path="s.vcz", name="s", samples="^sample_2")
        assert s.samples == "^sample_2"

    def test_sample_time_scalar(self):
        s = config.Source(path="s.vcz", name="ancient", sample_time=1.5)
        assert s.sample_time == 1.5

    def test_sample_time_string(self):
        s = config.Source(path="s.vcz", name="s", sample_time="sample_age")
        assert s.sample_time == "sample_age"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            config.Source(path="s.vcz", name="")

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            config.Source(path="s.vcz")


class TestAncestralStateConstruction:
    def test_basic(self):
        a = config.AncestralState(path="ann.vcz", field="variant_ancestral_allele")
        assert str(a.path) == "ann.vcz"
        assert a.field == "variant_ancestral_allele"


class TestAncestorsConfigConstruction:
    def test_basic(self):
        a = config.AncestorsConfig(name="anc", path="anc.vcz", sources=["cohort"])
        assert a.name == "anc"
        assert a.sources == ["cohort"]
        assert a.max_gap_length == 500_000

    def test_custom_gap_length(self):
        a = config.AncestorsConfig(
            name="anc", path="anc.vcz", sources=["cohort"], max_gap_length=1_000
        )
        assert a.max_gap_length == 1_000


class TestMatchConfigConstruction:
    def test_basic(self):
        m = config.MatchConfig(
            sources={
                "ancestors": config.MatchSourceConfig(
                    node_flags=0, create_individuals=False
                ),
                "cohort": config.MatchSourceConfig(),
            },
            output="out.trees",
        )
        assert m.path_compression is True
        assert m.reference_ts is None
        assert m.sources["cohort"].node_flags == 1
        assert m.sources["cohort"].create_individuals is True
        assert m.sources["ancestors"].node_flags == 0
        assert m.sources["ancestors"].create_individuals is False

    def test_with_reference_ts(self):
        m = config.MatchConfig(
            sources={"cohort": config.MatchSourceConfig()},
            output="out.trees",
            reference_ts="ref.trees",
        )
        assert str(m.reference_ts) == "ref.trees"


class TestPostProcessConfigConstruction:
    def test_defaults(self):
        p = config.PostProcessConfig()
        assert p.split_ultimate is True
        assert p.erase_flanks is True

    def test_custom(self):
        p = config.PostProcessConfig(split_ultimate=False, erase_flanks=False)
        assert p.split_ultimate is False


class TestIndividualMetadataConfigConstruction:
    def test_defaults(self):
        m = config.IndividualMetadataConfig()
        assert m.fields == {}
        assert m.population is None

    def test_with_fields(self):
        m = config.IndividualMetadataConfig(
            fields={"sample_id": "sample_id", "sex": "sample_sex"},
            population="sample_population",
        )
        assert m.fields["sex"] == "sample_sex"
        assert m.population == "sample_population"


class TestConfigConstruction:
    def test_basic(self):
        cfg = _minimal_config()
        assert "cohort" in cfg.sources
        assert cfg.ancestral_state is not None
        assert cfg.individual_metadata is None
        assert cfg.post_process is None

    def test_no_ancestors_no_reference_ts_raises(self):
        with pytest.raises(ValueError, match="ancestors"):
            config.Config(
                sources={},
                ancestors=[],
                match=config.MatchConfig(
                    sources={"cohort": config.MatchSourceConfig()},
                    output="out.trees",
                ),
                ancestral_state=config.AncestralState(
                    path="dummy", field="variant_ancestral_allele"
                ),
            )

    def test_reference_ts_without_ancestors_ok(self):
        cfg = config.Config(
            sources={"cohort": config.Source(path="s.vcz", name="cohort")},
            ancestors=[],
            match=config.MatchConfig(
                sources={"cohort": config.MatchSourceConfig()},
                output="out.trees",
                reference_ts="ref.trees",
            ),
            ancestral_state=config.AncestralState(
                path="dummy", field="variant_ancestral_allele"
            ),
        )
        assert cfg.ancestors == []
        assert cfg.match.reference_ts is not None


# ---------------------------------------------------------------------------
# TOML parsing — standard case
# ---------------------------------------------------------------------------

_STANDARD_TOML = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name    = "cohort"
path    = "samples.vcz"
include = "QUAL > 20"

[[ancestors]]
name           = "ancestors"
path           = "ancestors.vcz"
sources        = ["cohort"]
max_gap_length = 500000

[match]
output             = "final.trees"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.cohort]

[individual_metadata]
fields     = {sample_id = "sample_id", sex = "sample_sex"}
population = "sample_population"

[post_process]
split_ultimate = true
erase_flanks   = true
"""


class TestFromTomlStandard:
    def test_loads_without_error(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert isinstance(cfg, config.Config)

    def test_sources(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert "cohort" in cfg.sources
        assert cfg.sources["cohort"].include == "QUAL > 20"

    def test_ancestral_state(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.ancestral_state is not None
        assert cfg.ancestral_state.field == "variant_ancestral_allele"

    def test_ancestors(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert len(cfg.ancestors) == 1
        assert cfg.ancestors[0].sources == ["cohort"]
        assert cfg.ancestors[0].max_gap_length == 500_000

    def test_match(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert set(cfg.match.sources.keys()) == {"ancestors", "cohort"}
        assert cfg.match.sources["ancestors"].node_flags == 0
        assert cfg.match.sources["ancestors"].create_individuals is False
        assert cfg.match.sources["cohort"].node_flags == 1
        assert cfg.match.sources["cohort"].create_individuals is True

    def test_individual_metadata(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.individual_metadata is not None
        assert cfg.individual_metadata.fields["sex"] == "sample_sex"
        assert cfg.individual_metadata.population == "sample_population"

    def test_post_process(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.post_process is not None
        assert cfg.post_process.split_ultimate is True
        assert cfg.post_process.erase_flanks is True


# ---------------------------------------------------------------------------
# TOML parsing — path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_source_path_as_is(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.sources["cohort"].path == "samples.vcz"

    def test_ancestors_path_as_is(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.ancestors[0].path == "ancestors.vcz"

    def test_match_output_as_is(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.match.output == "final.trees"

    def test_ancestral_state_path_as_is(self, tmp_path):
        cfg = config.Config.from_toml(_write_toml(tmp_path, _STANDARD_TOML))
        assert cfg.ancestral_state.path == "annotations.vcz"

    def test_absolute_path_preserved(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "/data/samples.vcz"

[ancestors]
path    = "/data/ancestors.vcz"
sources = ["cohort"]

[match]
output             = "/data/final.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["cohort"].path == "/data/samples.vcz"
        assert cfg.ancestors[0].path == "/data/ancestors.vcz"
        assert cfg.match.output == "/data/final.trees"

    def test_remote_url_unchanged(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "s3://bucket/samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert str(cfg.sources["cohort"].path) == "s3://bucket/samples.vcz"

    def test_reference_ts_path_as_is(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "samples.vcz"

[match]
output             = "out.trees"
reference_ts       = "ref.trees"

[match.sources.cohort]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.match.reference_ts == "ref.trees"


# ---------------------------------------------------------------------------
# TOML parsing — field specs
# ---------------------------------------------------------------------------


class TestFieldSpecs:
    def test_include_expression(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name    = "cohort"
path    = "samples.vcz"
include = "QUAL > 30"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["cohort"].include == "QUAL > 30"

    def test_exclude_expression(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name    = "cohort"
path    = "samples.vcz"
exclude = "AC == 0"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["cohort"].exclude == "AC == 0"

    def test_samples_string(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name    = "cohort"
path    = "samples.vcz"
samples = "^sample_2,sample_3"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["cohort"].samples == "^sample_2,sample_3"

    def test_sample_time_scalar(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name        = "parents"
path        = "parents.vcz"
sample_time = 1

[ancestors]
path    = "ancestors.vcz"
sources = ["parents"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.parents]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["parents"].sample_time == 1

    def test_sample_time_string(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name        = "ancient"
path        = "ancient.vcz"
sample_time = "sample_age_generations"

[ancestors]
path    = "ancestors.vcz"
sources = ["ancient"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.ancient]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.sources["ancient"].sample_time == "sample_age_generations"

    def test_multiple_sources(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

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
output             = "out.trees"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.cohort]

[match.sources.ancient]
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert set(cfg.sources) == {"cohort", "ancient", "ancestors"}
        assert set(cfg.match.sources.keys()) == {"ancestors", "cohort", "ancient"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_missing_match_section(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "samples.vcz"
"""
        with pytest.raises(ValueError, match=r"\[match\]"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_match_missing_sources(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"
"""
        with pytest.raises((ValueError, KeyError)):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_source_missing_name(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
path = "samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        with pytest.raises(ValueError, match="name"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_source_missing_path(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        with pytest.raises(ValueError, match="path"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_duplicate_source_names(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

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
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        with pytest.raises(ValueError, match="Duplicate"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_no_ancestors_no_reference_ts(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "samples.vcz"

[match]
output             = "out.trees"

[match.sources.cohort]
"""
        with pytest.raises(ValueError, match="ancestors"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_ancestors_missing_path(self, tmp_path):
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "samples.vcz"

[ancestors]
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        with pytest.raises((ValueError, KeyError)):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_toml_missing_ancestral_state_raises(self, tmp_path):
        toml = """\
[[source]]
name = "cohort"
path = "samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        with pytest.raises(ValueError, match="ancestral_state"):
            config.Config.from_toml(_write_toml(tmp_path, toml))


# ---------------------------------------------------------------------------
# config.Config.format()
# ---------------------------------------------------------------------------


class TestConfigFormat:
    def test_includes_source(self):
        cfg = _minimal_config()
        text = cfg.format()
        assert "[source.cohort]" in text
        assert "samples.vcz" in text

    def test_includes_match(self):
        cfg = _minimal_config()
        text = cfg.format()
        assert "[match]" in text
        assert "path_compression" in text

    def test_includes_ancestors(self):
        cfg = _minimal_config()
        text = cfg.format()
        assert "[ancestors]" in text
        assert "max_gap_length" in text

    def test_includes_ancestral_state(self):
        cfg = _minimal_config(
            ancestral_state=config.AncestralState(path="ann.vcz", field="anc_allele"),
        )
        text = cfg.format()
        assert "[ancestral_state]" in text
        assert "ann.vcz" in text
        assert "anc_allele" in text

    def test_includes_post_process(self):
        cfg = _minimal_config(post_process=config.PostProcessConfig())
        text = cfg.format()
        assert "[post_process]" in text
        assert "split_ultimate" in text

    def test_includes_individual_metadata(self):
        cfg = _minimal_config(
            individual_metadata=config.IndividualMetadataConfig(
                fields={"sample_id": "sample_id"},
                population="pop",
            ),
        )
        text = cfg.format()
        assert "[individual_metadata]" in text
        assert "population = pop" in text

    def test_optional_source_fields(self):
        cfg = _minimal_config(
            sources={
                "s": config.Source(
                    path="s.vcz",
                    name="s",
                    include="QUAL > 30",
                    exclude="AC == 0",
                    samples="sample_0,sample_1",
                    sample_time=1.5,
                ),
            },
        )
        text = cfg.format()
        assert "include = QUAL > 30" in text
        assert "exclude = AC == 0" in text
        assert "samples = sample_0,sample_1" in text
        assert "sample_time = 1.5" in text

    def test_no_optional_sections(self):
        cfg = config.Config(
            sources={"s": config.Source(path="s.vcz", name="s")},
            ancestors=[],
            match=config.MatchConfig(
                sources={"s": config.MatchSourceConfig()},
                output="out.trees",
                reference_ts="ref.trees",
            ),
            ancestral_state=config.AncestralState(
                path="dummy", field="variant_ancestral_allele"
            ),
        )
        text = cfg.format()
        assert "[[ancestors]]" not in text
        assert "[post_process]" not in text
        assert "[individual_metadata]" not in text
        assert "[ancestral_state]" in text
        assert "reference_ts" in text


# ---------------------------------------------------------------------------
# config.Config.validate()
# ---------------------------------------------------------------------------


def _write_sample_vcz(tmp_path):
    """Create a sample VCZ on disk and return its path."""
    store = helpers.make_sample_vcz(
        genotypes=np.array([[[0], [1]], [[1], [0]]], dtype=np.int8),
        positions=np.array([100, 200], dtype=np.int32),
        alleles=np.array([["A", "T"], ["A", "T"]]),
        ancestral_state=np.array(["A", "A"]),
        sequence_length=1000,
    )
    vcz_path = tmp_path / "samples.vcz"
    zarr.save(str(vcz_path), **{k: store[k][:] for k in store})
    return vcz_path


class TestConfigValidate:
    def _anc_src(self, path=None):
        return config.Source(path=path, name="ancestors", sample_time="sample_time")

    def test_valid_config(self, tmp_path):
        vcz_path = _write_sample_vcz(tmp_path)
        cfg = config.Config(
            sources={
                "test": config.Source(path=vcz_path, name="test"),
                "ancestors": self._anc_src(tmp_path / "ancestors.vcz"),
            },
            ancestors=[
                config.AncestorsConfig(
                    name="ancestors", path=tmp_path / "ancestors.vcz", sources=["test"]
                )
            ],
            match=_minimal_match_cfg(),
            ancestral_state=config.AncestralState(
                path=vcz_path, field="variant_ancestral_allele"
            ),
        )
        errors = cfg.validate()
        assert errors == []

    def test_missing_source_path(self, tmp_path):
        cfg = config.Config(
            sources={
                "test": config.Source(path=tmp_path / "nonexistent.vcz", name="test"),
                "ancestors": self._anc_src(),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=_minimal_match_cfg(),
            ancestral_state=config.AncestralState(
                path="dummy", field="variant_ancestral_allele"
            ),
        )
        errors = cfg.validate()
        assert any("does not exist" in e for e in errors)

    def test_ancestors_path_not_checked(self, tmp_path):
        """ancestors.path is an output — should not be checked."""
        vcz_path = _write_sample_vcz(tmp_path)
        cfg = config.Config(
            sources={
                "test": config.Source(path=vcz_path, name="test"),
                "ancestors": self._anc_src(tmp_path / "not_yet_created.vcz"),
            },
            ancestors=[
                config.AncestorsConfig(
                    name="ancestors",
                    path=tmp_path / "not_yet_created.vcz",
                    sources=["test"],
                )
            ],
            match=_minimal_match_cfg(),
            ancestral_state=config.AncestralState(
                path=vcz_path, field="variant_ancestral_allele"
            ),
        )
        errors = cfg.validate()
        assert errors == []

    def test_unknown_ancestor_source(self, tmp_path):
        vcz_path = _write_sample_vcz(tmp_path)
        cfg = config.Config(
            sources={
                "test": config.Source(path=vcz_path, name="test"),
                "ancestors": self._anc_src(),
            },
            ancestors=[
                config.AncestorsConfig(
                    name="ancestors", path=None, sources=["nonexistent"]
                )
            ],
            match=_minimal_match_cfg(),
            ancestral_state=config.AncestralState(
                path=vcz_path, field="variant_ancestral_allele"
            ),
        )
        errors = cfg.validate()
        assert any("unknown source" in e.lower() for e in errors)

    def test_missing_sample_time_path(self, tmp_path):
        """Error when sample_time dict spec references nonexistent path."""
        vcz_path = _write_sample_vcz(tmp_path)
        cfg = config.Config(
            sources={
                "test": config.Source(
                    path=vcz_path,
                    name="test",
                    sample_time={
                        "path": str(tmp_path / "missing.vcz"),
                        "field": "time",
                    },
                ),
                "ancestors": self._anc_src(),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=_minimal_match_cfg(),
            ancestral_state=config.AncestralState(
                path=vcz_path, field="variant_ancestral_allele"
            ),
        )
        errors = cfg.validate()
        assert any("sample_time" in e and "does not exist" in e for e in errors)

    def test_missing_reference_ts(self):
        cfg = config.Config(
            sources={"s": config.Source(path="s.vcz", name="s")},
            ancestors=[],
            match=config.MatchConfig(
                sources={"s": config.MatchSourceConfig()},
                output="out.trees",
                reference_ts="/nonexistent/ref.trees",
            ),
            ancestral_state=config.AncestralState(
                path="dummy", field="variant_ancestral_allele"
            ),
        )
        errors = cfg.validate()
        assert any("reference_ts" in e and "does not exist" in e for e in errors)

    def test_missing_ancestral_state_path(self):
        cfg = _minimal_config(
            ancestral_state=config.AncestralState(
                path="/nonexistent/ann.vcz", field="anc"
            ),
        )
        errors = cfg.validate()
        assert any("Ancestral state" in e and "does not exist" in e for e in errors)

    def test_source_with_none_path(self, tmp_path):
        """Sources with path=None (in-memory) should not error."""
        vcz_path = _write_sample_vcz(tmp_path)
        cfg = config.Config(
            sources={
                "test": config.Source(path=None, name="test"),
                "ancestors": self._anc_src(),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path=None, sources=["test"])
            ],
            match=_minimal_match_cfg(),
            ancestral_state=config.AncestralState(
                path=vcz_path, field="variant_ancestral_allele"
            ),
        )
        errors = cfg.validate()
        assert errors == []


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------

_MINIMAL_TOML = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "samples.vcz"

[[ancestors]]
name    = "ancestors"
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output             = "out.trees"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.cohort]
"""


class TestUnknownKeys:
    def test_unknown_top_level_key(self, tmp_path):
        toml = _MINIMAL_TOML + '\n[bogus]\nfoo = "bar"\n'
        with pytest.raises(ValueError, match="Unrecognised.*top-level.*bogus"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_unknown_source_key(self, tmp_path):
        toml = _MINIMAL_TOML.replace(
            'path = "samples.vcz"',
            'path = "samples.vcz"\nflavour = "vanilla"',
        )
        with pytest.raises(ValueError, match="Unrecognised.*source.*flavour"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_unknown_ancestors_key(self, tmp_path):
        toml = _MINIMAL_TOML.replace(
            'sources = ["cohort"]',
            'sources = ["cohort"]\nfoo = 1',
            1,  # only replace first occurrence (ancestors section)
        )
        with pytest.raises(ValueError, match="Unrecognised.*ancestors.*foo"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_unknown_match_key(self, tmp_path):
        toml = _MINIMAL_TOML.replace(
            'output             = "out.trees"',
            'output             = "out.trees"\nmagic = true',
        )
        with pytest.raises(ValueError, match="Unrecognised.*match.*magic"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_unknown_post_process_key(self, tmp_path):
        toml = _MINIMAL_TOML + "\n[post_process]\nturbo = true\n"
        with pytest.raises(ValueError, match="Unrecognised.*post_process.*turbo"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_valid_config_no_error(self, tmp_path):
        config.Config.from_toml(_write_toml(tmp_path, _MINIMAL_TOML))


class TestWorkdirConfig:
    def test_workdir_parsed_from_toml(self, tmp_path):
        toml = _MINIMAL_TOML.replace(
            'output             = "out.trees"',
            'output             = "out.trees"\nworkdir = "checkpoints"',
        )
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.match.workdir == "checkpoints"

    def test_keep_intermediates_parsed_from_toml(self, tmp_path):
        toml = _MINIMAL_TOML.replace(
            'output             = "out.trees"',
            'output             = "out.trees"\n'
            'workdir = "checkpoints"\nkeep_intermediates = true',
        )
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert cfg.match.keep_intermediates is True
        assert cfg.match.workdir == "checkpoints"

    def test_keep_intermediates_without_workdir_errors(self):
        with pytest.raises(ValueError, match="keep_intermediates requires workdir"):
            config.Config(
                sources={
                    "t": config.Source(path="x", name="t"),
                    "ancestors": config.Source(
                        path="a", name="ancestors", sample_time="sample_time"
                    ),
                },
                ancestors=[
                    config.AncestorsConfig(name="ancestors", path="a", sources=["t"])
                ],
                match=config.MatchConfig(
                    sources={
                        "ancestors": config.MatchSourceConfig(
                            node_flags=0, create_individuals=False
                        ),
                        "t": config.MatchSourceConfig(),
                    },
                    output="o.trees",
                    keep_intermediates=True,
                ),
                ancestral_state=config.AncestralState(
                    path="dummy", field="variant_ancestral_allele"
                ),
            )

    def test_workdir_defaults_to_none(self):
        cfg = config.Config(
            sources={
                "t": config.Source(path="x", name="t"),
                "ancestors": config.Source(
                    path="a", name="ancestors", sample_time="sample_time"
                ),
            },
            ancestors=[
                config.AncestorsConfig(name="ancestors", path="a", sources=["t"])
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(
                        node_flags=0, create_individuals=False
                    ),
                    "t": config.MatchSourceConfig(),
                },
                output="o.trees",
            ),
            ancestral_state=config.AncestralState(
                path="dummy", field="variant_ancestral_allele"
            ),
        )
        assert cfg.match.workdir is None
        assert cfg.match.keep_intermediates is False

    def test_intermediate_ts_rejected(self, tmp_path):
        """Old intermediate_ts key is rejected as unknown."""
        toml = _MINIMAL_TOML.replace(
            'output             = "out.trees"',
            'output             = "out.trees"\nintermediate_ts = "foo_{group}.trees"',
        )
        with pytest.raises(ValueError, match="Unrecognised.*match.*intermediate_ts"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_groups_rejected(self, tmp_path):
        """Old groups key is rejected as unknown."""
        toml = _MINIMAL_TOML.replace(
            'output             = "out.trees"',
            'output             = "out.trees"\ngroups = "groups.json"',
        )
        with pytest.raises(ValueError, match="Unrecognised.*match.*groups"):
            config.Config.from_toml(_write_toml(tmp_path, toml))


class TestConfigCoverageEdgeCases:
    def test_resolve_field_spec_dict_with_path(self):
        spec = {"path": "some/path", "field": "x"}
        result = config._resolve_field_spec(spec)
        assert result["path"] == "some/path"
        assert result["field"] == "x"

    def test_resolve_field_spec_string(self):
        result = config._resolve_field_spec("simple_field")
        assert result == "simple_field"

    def test_ancestral_state_missing_path(self, tmp_path):
        toml = """\
[ancestral_state]
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output = "out.trees"

[match.sources.ancestors]
[match.sources.cohort]
"""
        with pytest.raises(ValueError, match="missing required key"):
            config.Config.from_toml(_write_toml(tmp_path, toml))

    def test_ancestors_invalid_format(self):
        raw = {
            "ancestral_state": {"path": "x", "field": "y"},
            "ancestors": "not_a_table",
            "match": {"sources": {"cohort": {}}, "output": "out.trees"},
            "source": [{"name": "cohort", "path": "s.vcz"}],
        }
        with pytest.raises(ValueError, match="must be a table"):
            config._parse_ancestors(raw)

    def test_augment_sites_missing_sources(self, tmp_path):
        raw = {"augment_sites": {}}
        with pytest.raises(ValueError, match="missing required key.*sources"):
            config._parse_augment_sites(raw)

    def test_augment_sites_sources_not_list(self, tmp_path):
        raw = {"augment_sites": {"sources": "not_a_list"}}
        with pytest.raises(ValueError, match="sources must be a list"):
            config._parse_augment_sites(raw)

    def test_ancestor_not_in_match_sources(self):
        with pytest.raises(ValueError, match="must appear in"):
            config.Config(
                sources={"cohort": config.Source(name="cohort", path="s.vcz")},
                ancestors=[
                    config.AncestorsConfig(
                        name="ancestors",
                        path="a.vcz",
                        sources=["cohort"],
                    )
                ],
                match=config.MatchConfig(
                    sources={"cohort": config.MatchSourceConfig()},
                    output="out.trees",
                ),
                ancestral_state=config.AncestralState(path="ann.vcz", field="x"),
            )

    def test_match_source_simple_value(self, tmp_path):
        """A match source with a bare value (not a table) gets default config."""
        toml = """\
[ancestral_state]
path  = "annotations.vcz"
field = "variant_ancestral_allele"

[[source]]
name = "cohort"
path = "samples.vcz"

[ancestors]
path    = "ancestors.vcz"
sources = ["cohort"]

[match]
output = "out.trees"

[match.sources]
ancestors = true
cohort = true
"""
        cfg = config.Config.from_toml(_write_toml(tmp_path, toml))
        assert isinstance(cfg.match.sources["cohort"], config.MatchSourceConfig)

    def test_config_format_optional_fields(self):
        """Config.format() includes optional source fields when set."""
        cfg = config.Config(
            sources={
                "cohort": config.Source(
                    name="cohort",
                    path="s.vcz",
                    regions="chr1:1-100",
                    targets="targets.bed",
                )
            },
            ancestors=[
                config.AncestorsConfig(
                    name="ancestors", path="a.vcz", sources=["cohort"]
                )
            ],
            match=config.MatchConfig(
                sources={
                    "ancestors": config.MatchSourceConfig(),
                    "cohort": config.MatchSourceConfig(),
                },
                output="out.trees",
            ),
            ancestral_state=config.AncestralState(path="ann.vcz", field="x"),
        )
        text = cfg.format()
        assert "regions" in text
        assert "targets" in text
