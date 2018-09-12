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
Collection of constants used in tsinfer.
"""

UNKNOWN_ALLELE = 255

C_ENGINE = "C"
PY_ENGINE = "P"

# Bit 16 is set in node flags when they have been created by path compression.
NODE_IS_PC_ANCESTOR = 1 << 16
# Bit 17 is set in node flags when they have been created by shared recombination
# breakpoint
NODE_IS_SRB_ANCESTOR = 1 << 17
# Bit 18 is set in node flags when they are samples inserted to augment existing
# ancestors.
NODE_IS_SAMPLE_ANCESTOR = 1 << 18
