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
Exceptions raised by tsinfer.
"""


class TsinferException(Exception):
    """
    Superclass of all exceptions thrown by tsinfer.
    """


class FileError(TsinferException):
    """
    Exception raised when some non-specific error happens during file handling.
    """


class FileFormatError(FileError):
    """
    Exception raised when a malformed file is encountered.
    """
