#
# Copyright (C) 2020 University of Oxford
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
Configuration and fixtures for pytest. Only put test-suite wide fixtures in here. Module
specific fixtures should live in their modules.

To use a fixture in a test simply refer to it by name as an argument. This is called
dependancy injection. Note that all fixtures should have the suffix "_fixture" to make
it clear in test code.

For example to use the `ts` fixture (a tree sequence with data in all fields) in a test:

class TestClass:
    def test_something(self, ts_fixture):
        assert ts_fixture.some_method() == expected

Fixtures can be parameterised etc. see https://docs.pytest.org/en/stable/fixture.html

Note that fixtures have a "scope" for example `ts_fixture` below is only created once
per test session and re-used for subsequent tests.
"""
import msprime
import numpy as np
import pytest
from pytest import fixture
from tsutil import mark_mutation_times_unknown

import tsinfer


def pytest_addoption(parser):
    """
    Add an option to skip tests marked with `@pytest.mark.slow`
    """
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="Skip slow tests"
    )


def pytest_configure(config):
    """
    Add docs on the "slow" marker
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow specified")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def num_nonsample_muts(ts):
    return np.sum(np.logical_not(np.isin(ts.tables.mutations.node, ts.samples())))


@fixture(scope="session")
def small_ts_fixture():
    """
    A simple 1-tree sequence with at least 2 inference sites
    (i.e. mutations above a non-sample node), and no mutation times
    """
    ts = msprime.simulate(10, mutation_rate=1, random_seed=1)
    assert num_nonsample_muts(ts) > 1
    return mark_mutation_times_unknown(ts)


@fixture(scope="session")
def small_sd_fixture(small_ts_fixture):
    """
    A sample data instance from the small 1-tree sequence
    """
    return tsinfer.SampleData.from_tree_sequence(small_ts_fixture, use_sites_time=False)


@fixture(scope="session")
def small_sd_anc_fixture(small_ts_fixture):
    """
    A sample data and an ancestors instance from the small 1-tree sequence
    """
    sd = tsinfer.SampleData.from_tree_sequence(small_ts_fixture, use_sites_time=False)
    return sd, tsinfer.generate_ancestors(sd)


@fixture(scope="session")
def medium_ts_fixture():
    """
    A medium sized tree sequence with a good number of trees and inference mutations
    (i.e. mutations above a non-sample node), and no mutation
    """
    ts = msprime.simulate(10, recombination_rate=2, mutation_rate=10, random_seed=3)
    assert ts.num_trees > 10
    assert num_nonsample_muts(ts) > 50
    return mark_mutation_times_unknown(ts)


@fixture(scope="session")
def medium_sd_fixture(medium_ts_fixture):
    """
    A sample data instance from the medium-sized tree sequence
    """
    return tsinfer.SampleData.from_tree_sequence(
        medium_ts_fixture, use_sites_time=False
    )
