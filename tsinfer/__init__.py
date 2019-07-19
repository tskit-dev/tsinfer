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
# Start temporary monkey patch to allow use of functions / constants in tskit v2.0.
# The following lines can be deleted once the master tskit version has been updated
import numpy as np
import tskit
tskit.MISSING_DATA = -1


class util():
    @staticmethod
    def safe_np_int_cast(int_array, dtype, copy=False):  # Copied from v2.0 tskit/util.py
        if not isinstance(int_array, np.ndarray):
            int_array = np.array(int_array)
            copy = False
        if int_array.size == 0:
            return int_array.astype(dtype, copy=copy)
        try:
            return int_array.astype(dtype, casting='safe', copy=copy)
        except TypeError:
            bounds = np.iinfo(dtype)
            if np.any(int_array < bounds.min) or np.any(int_array > bounds.max):
                raise OverflowError("Cannot convert safely to {} type".format(dtype))
            if int_array.dtype.kind == 'i' and np.dtype(dtype).kind == 'u':
                casting = 'unsafe'
            else:
                casting = 'same_kind'
            return int_array.astype(dtype, casting=casting, copy=copy)


tskit.util = util
# End temporary monkey patch

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
from .eval_util import *  # NOQA
from .exceptions import *  # NOQA
from .constants import *  # NOQA
from .cli import get_cli_parser  # NOQA
