#
# Copyright (C) 2018-2020 University of Oxford
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
Collection of constants used in tsinfer. We also make use of constants defined in tskit.
"""
import numpy as np


C_ENGINE = "C"
PY_ENGINE = "P"


# TODO Change these to use the enum.IntFlag class

# Bit 16 is set in node flags when they have been created by path compression.
NODE_IS_PC_ANCESTOR = 1 << 16
# Bit 17 is set in node flags when they have been created by shared recombination
# breakpoint
NODE_IS_SRB_ANCESTOR = 1 << 17
# Bits 18 and 19 should be mutually exclusive. Bit 18 is set in node flags when a node
# corresponds to an ancestor that is very like a sample genome but is allowed to differ,
# for example, at sites not used in full inference. Bit 19 is set when a node
# corresponds to an ancestor that actually *is* a sampled genome (and which might, for
# example, be associated with an individual in the individuals table). If bit 19 is set
# it will lead to the tskit.NODE_IS_SAMPLE flag being set on this node by match_samples.
NODE_IS_PROXY_SAMPLE_ANCESTOR = 1 << 18
NODE_IS_TRUE_SAMPLE_ANCESTOR = 1 << 19

# Marker constants for node & site time values
TIME_UNSPECIFIED = -np.inf

# What type of inference have we done at a site?
INFERENCE_FULL = "full"
INFERENCE_FITCH_PARSIMONY = "fitch_parsimony"
