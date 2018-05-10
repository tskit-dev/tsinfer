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
Common provenance methods used to determine the state and versions
of various dependencies and the OS.
"""

import platform

import zarr
import numcodecs
import lmdb
import msprime


__version__ = "undefined"
try:
    from . import _version
    __version__ = _version.version
except ImportError:
    pass


def get_environment():
    """
    Returns a dictionary describing the environment in which tsinfer
    is currently running.
    """
    env = {
        "libraries": {
            "zarr": {
                "version": zarr.__version__
            },
            "numcodecs": {
                "version": numcodecs.__version__
            },
            "lmdb": {
                "version": lmdb.__version__
            },
            "msprime": {
                "version": msprime.__version__
            },
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
        }
    }
    return env


def get_provenance_dict(command, parameters=None, source=None):
    """
    Returns a dictionary encoding an execution of tsinfer.

    Note: this format is incomplete and provisional.
    """
    document = {
        "software": "tsinfer",
        "version": __version__,
        "command": command,
        "parameters": parameters,
        "source": source,
        "environment": get_environment()
    }
    return document
