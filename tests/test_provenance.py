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
import unittest
import json

import msprime

import tsinfer
import tsinfer.provenance as provenance


class TestProvenanceValid(unittest.TestCase):
    """
    Checks that the provenance information on the tree sequences is
    valid.
    """
    def validate_ts(self, ts):
        for prov in ts.provenances():
            p_doc = json.loads(prov.record)
            msprime.validate_provenance(p_doc)

    def validate_file(self, data):
        for timestamp, record in data.provenances():
            msprime.validate_provenance(record)

    def test_infer(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred_ts = tsinfer.infer(samples)
        self.validate_ts(inferred_ts)

    def test_ancestors_ts(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ancestor_data = tsinfer.generate_ancestors(samples)
        ancestors_ts = tsinfer.match_ancestors(samples, ancestor_data)
        self.validate_ts(ancestors_ts)

    def test_sample_data(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        with tsinfer.SampleData() as sample_data:
            for var in ts.variants():
                sample_data.add_site(var.site.position, genotypes=var.genotypes)
            sample_data.record_provenance("test", arg1=1, arg2=2)
        self.validate_file(sample_data)

    def test_from_tree_sequence(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.validate_file(sample_data)

    def test_ancestors_file(self):
        ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
        self.assertGreater(ts.num_sites, 1)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        self.validate_file(ancestor_data)


class TestGetProvenance(unittest.TestCase):
    """
    Check the get_provenance_dict function.
    """

    def test_no_command(self):
        with self.assertRaises(ValueError):
            provenance.get_provenance_dict()

    def validate_encoding(self, params):
        pdict = provenance.get_provenance_dict("test", **params)
        encoded = pdict["parameters"]
        self.assertEqual(encoded["command"], "test")
        del encoded["command"]
        self.assertEqual(encoded, params)

    def test_empty_params(self):
        self.validate_encoding({})

    def test_non_empty_params(self):
        self.validate_encoding({"a": 1, "b": "b", "c": 12345})
