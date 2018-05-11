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
Tree sequence inference.

Python 3 only.
"""

import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 only")

__version__ = "undefined"
try:
    from . import _version
    __version__ = _version.version
except ImportError:
    pass

from .inference import *  # NOQA
from .formats import *  # NOQA
from .evaluation import *  # NOQA
from .exceptions import *  # NOQA
from .cli import get_cli_parser  # NOQA
