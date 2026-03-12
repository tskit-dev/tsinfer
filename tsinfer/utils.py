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
Utility classes for tsinfer.
"""

from __future__ import annotations

import concurrent.futures as cf


class SynchronousExecutor(cf.Executor):
    """
    An executor that runs tasks synchronously in the calling thread.

    Implements the :class:`concurrent.futures.Executor` interface so that
    threaded and non-threaded code paths share the same API.  Used when
    ``num_threads <= 0`` for testing and debugging.
    """

    def submit(self, fn, /, *args, **kwargs):
        future = cf.Future()
        future.set_result(fn(*args, **kwargs))
        return future
