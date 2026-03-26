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
Provenance methods for recording the state and versions of
dependencies, the OS, and resource usage.
"""

import dataclasses
import platform
import sys
import time

import numba
import psutil
import tskit
import zarr

from . import __version__

if sys.platform != "win32":
    import resource


@dataclasses.dataclass
class ResourceMetrics:
    elapsed_time: float
    user_time: float
    sys_time: float
    max_memory: int

    def asdict(self):
        return dataclasses.asdict(self)

    @classmethod
    def combine(cls, metrics_list):
        if not metrics_list:
            raise ValueError("Cannot combine empty list of metrics")
        return cls(
            elapsed_time=sum(m.elapsed_time for m in metrics_list),
            user_time=sum(m.user_time for m in metrics_list),
            sys_time=sum(m.sys_time for m in metrics_list),
            max_memory=max(m.max_memory for m in metrics_list),
        )


def get_environment():
    """
    Returns a dictionary describing the environment in which tsinfer
    is currently running.
    """
    env = {
        "libraries": {
            "tskit": {"version": tskit.__version__},
            "zarr": {"version": zarr.__version__},
            "numba": {"version": numba.__version__},
        },
        "os": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version_tuple(),
        },
    }
    return env


def get_provenance_dict(command=None, resources=None, **kwargs):
    """
    Returns a dictionary encoding an execution of tsinfer following the
    tskit provenance schema.

    https://tskit.readthedocs.io/en/stable/provenance.html
    """
    if command is None:
        raise ValueError("Command must be provided")
    parameters = dict(kwargs)
    parameters["command"] = command
    document = {
        "schema_version": "1.0.0",
        "software": {"name": "tsinfer", "version": __version__},
        "parameters": parameters,
        "environment": get_environment(),
    }
    if resources is not None:
        document["resources"] = resources
    return document


def get_peak_memory_bytes():
    """Return peak memory usage in bytes, or None if unavailable."""
    if sys.platform in ("linux", "darwin"):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss = usage.ru_maxrss
        if sys.platform == "linux":
            return max_rss * 1024
        return max_rss
    elif sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset
    else:
        return None


class TimingAndMemory:
    """Context manager for tracking timing and memory usage."""

    def __init__(self):
        self.metrics = None

    def __enter__(self):
        self._process = psutil.Process()
        self._start_elapsed = time.perf_counter()
        self._start_times = self._process.cpu_times()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_times = self._process.cpu_times()
        self.metrics = ResourceMetrics(
            elapsed_time=time.perf_counter() - self._start_elapsed,
            user_time=end_times.user - self._start_times.user,
            sys_time=end_times.system - self._start_times.system,
            max_memory=get_peak_memory_bytes(),
        )
