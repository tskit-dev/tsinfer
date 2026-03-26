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
Tests for the provenance module.
"""

import time

import numba
import pytest
import tskit
import zarr

import tsinfer
from tsinfer import provenance


class TestGetEnvironment:
    def test_returns_dict(self):
        env = provenance.get_environment()
        assert isinstance(env, dict)

    def test_has_libraries(self):
        env = provenance.get_environment()
        libs = env["libraries"]
        assert "tskit" in libs
        assert "zarr" in libs
        assert "numba" in libs

    def test_library_versions_are_strings(self):
        env = provenance.get_environment()
        for lib_info in env["libraries"].values():
            assert isinstance(lib_info["version"], str)

    def test_library_versions_match(self):
        env = provenance.get_environment()
        assert env["libraries"]["tskit"]["version"] == tskit.__version__
        assert env["libraries"]["zarr"]["version"] == zarr.__version__
        assert env["libraries"]["numba"]["version"] == numba.__version__

    def test_has_os_info(self):
        env = provenance.get_environment()
        os_info = env["os"]
        assert "system" in os_info
        assert "node" in os_info
        assert "release" in os_info
        assert "machine" in os_info

    def test_has_python_info(self):
        env = provenance.get_environment()
        py = env["python"]
        assert "implementation" in py
        assert "version" in py


class TestGetProvenanceDict:
    def test_requires_command(self):
        with pytest.raises(ValueError, match="Command must be provided"):
            provenance.get_provenance_dict()

    def test_basic_structure(self):
        d = provenance.get_provenance_dict(command="test_cmd")
        assert d["schema_version"] == "1.0.0"
        assert d["software"]["name"] == "tsinfer"
        assert d["software"]["version"] == tsinfer.__version__
        assert d["parameters"]["command"] == "test_cmd"
        assert "environment" in d

    def test_extra_kwargs_in_parameters(self):
        d = provenance.get_provenance_dict(
            command="match", num_threads=4, path="/tmp/foo"
        )
        assert d["parameters"]["command"] == "match"
        assert d["parameters"]["num_threads"] == 4
        assert d["parameters"]["path"] == "/tmp/foo"

    def test_resources_included_when_provided(self):
        resources = {"elapsed_time": 1.5, "max_memory": 1024}
        d = provenance.get_provenance_dict(command="run", resources=resources)
        assert d["resources"] == resources

    def test_resources_absent_when_none(self):
        d = provenance.get_provenance_dict(command="run")
        assert "resources" not in d


class TestResourceMetrics:
    def test_asdict(self):
        m = provenance.ResourceMetrics(
            elapsed_time=1.0, user_time=0.5, sys_time=0.3, max_memory=1024
        )
        d = m.asdict()
        assert d == {
            "elapsed_time": 1.0,
            "user_time": 0.5,
            "sys_time": 0.3,
            "max_memory": 1024,
        }

    def test_combine(self):
        m1 = provenance.ResourceMetrics(
            elapsed_time=1.0, user_time=0.5, sys_time=0.2, max_memory=100
        )
        m2 = provenance.ResourceMetrics(
            elapsed_time=2.0, user_time=1.0, sys_time=0.3, max_memory=200
        )
        combined = provenance.ResourceMetrics.combine([m1, m2])
        assert combined.elapsed_time == 3.0
        assert combined.user_time == 1.5
        assert combined.sys_time == 0.5
        assert combined.max_memory == 200

    def test_combine_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            provenance.ResourceMetrics.combine([])


class TestGetPeakMemoryBytes:
    def test_returns_positive_int(self):
        mem = provenance.get_peak_memory_bytes()
        assert mem is not None
        assert isinstance(mem, int)
        assert mem > 0


class TestTimingAndMemory:
    def test_metrics_recorded(self):
        with provenance.TimingAndMemory() as tm:
            time.sleep(0.01)
        assert tm.metrics is not None
        assert tm.metrics.elapsed_time > 0
        assert tm.metrics.max_memory > 0

    def test_user_time_nonnegative(self):
        with provenance.TimingAndMemory() as tm:
            pass
        assert tm.metrics.user_time >= 0
        assert tm.metrics.sys_time >= 0

    def test_elapsed_time_reasonable(self):
        with provenance.TimingAndMemory() as tm:
            time.sleep(0.05)
        assert tm.metrics.elapsed_time >= 0.04
