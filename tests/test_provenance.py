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
    def test_provenance_infer(self, small_sd_fixture, mmr):
        ts = tsinfer.infer(
            small_sd_fixture, mismatch_ratio=mmr, recombination_rate=1e-8
        )
        assert ts.num_provenances == small_sd_fixture.num_provenances + 1
        record = json.loads(ts.provenance(-1).record)
        params = record["parameters"]
        assert params["command"] == "infer"
        assert params["mismatch_ratio"] == mmr

    def test_provenance_generate_ancestors(self, small_sd_fixture):
        ancestors = tsinfer.generate_ancestors(small_sd_fixture)
        assert ancestors.num_provenances == small_sd_fixture.num_provenances + 1
        for p in ancestors.provenances():
            timestamp, record = p
        params = record["parameters"]
        assert params["command"] == "generate_ancestors"

    @pytest.mark.parametrize("mmr", [None, 0.1])
    def test_provenance_match_ancestors(self, small_sd_fixture, mmr):
        ancestors = tsinfer.generate_ancestors(small_sd_fixture)
        anc_ts = tsinfer.match_ancestors(
            small_sd_fixture, ancestors, mismatch_ratio=mmr, recombination_rate=1e-8
        )
        assert anc_ts.num_provenances == small_sd_fixture.num_provenances + 2
        params = json.loads(anc_ts.provenance(-2).record)["parameters"]
        assert params["command"] == "generate_ancestors"
        params = json.loads(anc_ts.provenance(-1).record)["parameters"]
        assert params["command"] == "match_ancestors"
        assert params["mismatch_ratio"] == mmr

    @pytest.mark.parametrize("mmr", [None, 0.1])
    def test_provenance_match_samples(self, small_sd_fixture, mmr):
        ancestors = tsinfer.generate_ancestors(small_sd_fixture)
        anc_ts = tsinfer.match_ancestors(small_sd_fixture, ancestors)
        ts = tsinfer.match_samples(
            small_sd_fixture, anc_ts, mismatch_ratio=mmr, recombination_rate=1e-8
        )
        assert ts.num_provenances == small_sd_fixture.num_provenances + 3
        params = json.loads(ts.provenance(-3).record)["parameters"]
        assert params["command"] == "generate_ancestors"
        params = json.loads(ts.provenance(-2).record)["parameters"]
        assert params["command"] == "match_ancestors"
        params = json.loads(ts.provenance(-1).record)["parameters"]
        assert params["command"] == "match_samples"
        assert params["mismatch_ratio"] == mmr


class TestGetProvenance:
    """
    Check the get_provenance_dict function.
    """

    def test_no_command(self):
        with pytest.raises(ValueError):
            provenance.get_provenance_dict()

    def validate_encoding(self, params):
        pdict = provenance.get_provenance_dict("test", **params)
        encoded = pdict["parameters"]
        assert encoded["command"] == "test"
        del encoded["command"]
        assert encoded == params

    def test_empty_params(self):
        self.validate_encoding({})

    def test_non_empty_params(self):
        self.validate_encoding({"a": 1, "b": "b", "c": 12345})
