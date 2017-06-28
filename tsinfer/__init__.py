"""
Implementation of the Li and Stephens algorithm for inferring a tree
sequence.

Python 3 only.
"""

import sys

__version__ = "0.4.a1"

if sys.version_info[0] < 3:
    raise Exception("Python 3 only")

# TODO remove old code and change file names.
from .inference import ReferencePanel
from .inference import Illustrator

# TODO make this more specific
from .new_inference import *
