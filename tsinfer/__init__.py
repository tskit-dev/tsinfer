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
"""
import sys
import warnings

# tsinfer #957. This warning pops up as a result of using fast-math. It bascially means
# that tiny tiny values are being rounded to zero, which should be fine for our purposes.
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal for <class 'numpy.float64'> "
    "type is zero",
)


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
