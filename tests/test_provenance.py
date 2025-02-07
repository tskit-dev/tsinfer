#
# Copyright (C) 2018 University of Oxford
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
Tests for the provenance stored in the output tree sequences.
"""
import json
import math
import time

import pytest
import tskit

import tsinfer
import tsinfer.provenance as provenance


class TestProvenanceValid:
    """
    Checks that the provenance information on the tree sequences is
    valid.
    """

    def validate_ts(self, ts):
        for prov in ts.provenances():
            p_doc = json.loads(prov.record)
            tskit.validate_provenance(p_doc)

    def validate_file(self, data):
        for _timestamp, record in data.provenances():
            tskit.validate_provenance(record)

    def test_infer(self, small_sd_fixture):
        inferred_ts = tsinfer.infer(small_sd_fixture)
        self.validate_ts(inferred_ts)

    def test_ancestors_ts(self, small_sd_fixture):
        ancestor_data = tsinfer.generate_ancestors(small_sd_fixture)
        ancestors_ts = tsinfer.match_ancestors(small_sd_fixture, ancestor_data)
        self.validate_ts(ancestors_ts)

    def test_sample_data(self, small_ts_fixture):
        with tsinfer.SampleData() as sample_data:
            for var in small_ts_fixture.variants():
                sample_data.add_site(var.site.position, genotypes=var.genotypes)
            sample_data.record_provenance("test", arg1=1, arg2=2)
        self.validate_file(sample_data)

    def test_from_tree_sequence(self, small_ts_fixture):
        sample_data = tsinfer.SampleData.from_tree_sequence(small_ts_fixture)
        self.validate_file(sample_data)

    def test_ancestors_file(self, small_sd_fixture):
        ancestor_data = tsinfer.generate_ancestors(small_sd_fixture)
        self.validate_file(ancestor_data)


class TestResourceMetrics:
    """
    Tests for the ResourceMetrics dataclass.
    """

    def test_create_and_asdict(self):
        metrics = provenance.ResourceMetrics(
            elapsed_time=1.5, user_time=1.0, sys_time=0.5, max_memory=1000
        )
        d = metrics.asdict()
        assert d == {
            "elapsed_time": 1.5,
            "user_time": 1.0,
            "sys_time": 0.5,
            "max_memory": 1000,
        }

    def test_combine_metrics(self):
        m1 = provenance.ResourceMetrics(
            elapsed_time=1.0, user_time=0.5, sys_time=0.2, max_memory=1000
        )
        m2 = provenance.ResourceMetrics(
            elapsed_time=2.0, user_time=1.5, sys_time=0.3, max_memory=2000
        )
        combined = provenance.ResourceMetrics.combine([m1, m2])
        assert combined.elapsed_time == 3.0
        assert combined.user_time == 2.0
        assert combined.sys_time == 0.5
        assert combined.max_memory == 2000

    def test_combine_empty_list(self):
        with pytest.raises(ValueError):
            provenance.ResourceMetrics.combine([])

    def test_combine_single_metric(self):
        m = provenance.ResourceMetrics(
            elapsed_time=1.0, user_time=0.5, sys_time=0.2, max_memory=1000
        )
        combined = provenance.ResourceMetrics.combine([m])
        assert combined.elapsed_time == 1.0
        assert combined.user_time == 0.5
        assert combined.sys_time == 0.2
        assert combined.max_memory == 1000


class TestIncludeProvenance:
    """
    Test that we can include or exclude provenances
    """

    def test_no_provenance_infer(self, small_sd_fixture):
        ts = tsinfer.infer(small_sd_fixture, record_provenance=False)
        assert ts.num_provenances == small_sd_fixture.num_provenances

    def test_no_provenance_generate_ancestors(self, small_sd_fixture):
        ancestors = tsinfer.generate_ancestors(
            small_sd_fixture, record_provenance=False
        )
        assert ancestors.num_provenances == small_sd_fixture.num_provenances

    def test_no_provenance_match_ancestors(self, small_sd_fixture):
        ancestors = tsinfer.generate_ancestors(
            small_sd_fixture, record_provenance=False
        )
        anc_ts = tsinfer.match_ancestors(
            small_sd_fixture, ancestors, record_provenance=False
        )
        assert anc_ts.num_provenances == small_sd_fixture.num_provenances

    def test_no_provenance_match_samples(self, small_sd_fixture):
        ancestors = tsinfer.generate_ancestors(
            small_sd_fixture, record_provenance=False
        )
        anc_ts = tsinfer.match_ancestors(
            small_sd_fixture, ancestors, record_provenance=False
        )
        ts = tsinfer.match_samples(small_sd_fixture, anc_ts, record_provenance=False)
        assert ts.num_provenances == small_sd_fixture.num_provenances

    @pytest.mark.parametrize("mmr", [None, 0.1])
    @pytest.mark.parametrize("pc", [True, False])
    @pytest.mark.parametrize("post", [True, False])
    @pytest.mark.parametrize("precision", [4, 5])
    def test_provenance_infer(self, small_sd_fixture, mmr, pc, post, precision):
        ts = tsinfer.infer(
            small_sd_fixture,
            path_compression=pc,
            post_process=post,
            precision=precision,
            mismatch_ratio=mmr,
            recombination_rate=1e-8,
        )
        assert ts.num_provenances == small_sd_fixture.num_provenances + 1
        record = json.loads(ts.provenance(-1).record)
        params = record["parameters"]
        assert params["command"] == "infer"
        assert params["post_process"] == post
        assert params["precision"] == precision
        assert params["mismatch_ratio"] == mmr
        assert params["path_compression"] == pc
        assert "simplify" not in params
        assert "resources" in record

    def test_provenance_generate_ancestors(self, small_sd_fixture):
        ancestors = tsinfer.generate_ancestors(small_sd_fixture)
        assert ancestors.num_provenances == small_sd_fixture.num_provenances + 1
        for p in ancestors.provenances():
            timestamp, record = p
        params = record["parameters"]
        assert params["command"] == "generate_ancestors"
        assert "resources" in record

    @pytest.mark.parametrize("mmr", [None, 0.1])
    @pytest.mark.parametrize("pc", [True, False])
    @pytest.mark.parametrize("precision", [4, 5])
    def test_provenance_match_ancestors(self, small_sd_fixture, mmr, pc, precision):
        ancestors = tsinfer.generate_ancestors(small_sd_fixture)
        anc_ts = tsinfer.match_ancestors(
            small_sd_fixture,
            ancestors,
            mismatch_ratio=mmr,
            recombination_rate=1e-8,
            path_compression=pc,
            precision=precision,
        )
        assert anc_ts.num_provenances == small_sd_fixture.num_provenances + 2
        params = json.loads(anc_ts.provenance(-2).record)["parameters"]
        assert params["command"] == "generate_ancestors"
        params = json.loads(anc_ts.provenance(-1).record)["parameters"]
        assert params["command"] == "match_ancestors"
        assert params["mismatch_ratio"] == mmr
        assert params["path_compression"] == pc
        assert params["precision"] == precision
        for provenance_index in [-2, -1]:
            record = json.loads(anc_ts.provenance(provenance_index).record)
            assert "resources" in record

    @pytest.mark.parametrize("mmr", [None, 0.1])
    @pytest.mark.parametrize("pc", [True, False])
    @pytest.mark.parametrize("post", [True, False])
    @pytest.mark.parametrize("precision", [4, 5])
    def test_provenance_match_samples(self, small_sd_fixture, mmr, pc, precision, post):
        ancestors = tsinfer.generate_ancestors(small_sd_fixture)
        anc_ts = tsinfer.match_ancestors(small_sd_fixture, ancestors)
        ts = tsinfer.match_samples(
            small_sd_fixture,
            anc_ts,
            mismatch_ratio=mmr,
            path_compression=pc,
            precision=precision,
            post_process=post,
            recombination_rate=1e-8,
        )
        assert ts.num_provenances == small_sd_fixture.num_provenances + 3
        params = json.loads(ts.provenance(-3).record)["parameters"]
        assert params["command"] == "generate_ancestors"
        params = json.loads(ts.provenance(-2).record)["parameters"]
        assert params["command"] == "match_ancestors"
        params = json.loads(ts.provenance(-1).record)["parameters"]
        assert params["command"] == "match_samples"
        assert params["mismatch_ratio"] == mmr
        assert params["path_compression"] == pc
        assert params["precision"] == precision
        assert params["post_process"] == post
        assert "simplify" not in params  # deprecated
        for provenance_index in [-3, -2, -1]:
            record = json.loads(ts.provenance(provenance_index).record)
            assert "resources" in record

    @pytest.mark.parametrize("simp", [True, False])
    def test_deprecated_simplify(self, small_sd_fixture, simp):
        # Included for completeness, but this is deprecated
        ancestors = tsinfer.generate_ancestors(small_sd_fixture)
        anc_ts = tsinfer.match_ancestors(small_sd_fixture, ancestors)
        ts1 = tsinfer.match_samples(small_sd_fixture, anc_ts, simplify=simp)
        ts2 = tsinfer.infer(small_sd_fixture, simplify=simp)
        for ts in [ts1, ts2]:
            record = json.loads(ts.provenance(-1).record)
            params = record["parameters"]
            assert params["simplify"] == simp
            assert "post_process" not in params


class TestGetProvenance:
    """
    Check the get_provenance_dict function.
    """

    def test_no_command(self):
        with pytest.raises(ValueError):
            provenance.get_provenance_dict()

    def validate_encoding(self, params, resources=None):
        pdict = provenance.get_provenance_dict("test", resources=resources, **params)
        encoded = pdict["parameters"]
        assert encoded["command"] == "test"
        del encoded["command"]
        assert encoded == params
        if resources is not None:
            assert "resources" in pdict
            assert pdict["resources"] == resources
        else:
            assert "resources" not in pdict

    def test_empty_params(self):
        self.validate_encoding({})

    def test_non_empty_params(self):
        self.validate_encoding({"a": 1, "b": "b", "c": 12345})

    def test_with_resources(self):
        self.validate_encoding(
            {}, resources={"elapsed_time": 1.23, "max_memory": 567.89}
        )


def test_timing_and_memory_context_manager():
    with provenance.TimingAndMemory() as timing:
        # Do some work to ensure measurable changes
        time.sleep(0.1)
        for i in range(1000000):
            math.sqrt(i)
        _ = [0] * 1000000

    assert timing.metrics is not None
    assert timing.metrics.elapsed_time > 0.1
    # Check we have highres timing
    assert timing.metrics.elapsed_time < 1
    assert timing.metrics.user_time > 0
    assert timing.metrics.sys_time >= 0
    assert timing.metrics.max_memory > 100_000_000

    # Test metrics are not available during context
    with provenance.TimingAndMemory() as timing2:
        assert timing2.metrics is None
