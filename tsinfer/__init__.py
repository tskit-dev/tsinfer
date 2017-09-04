"""
Implementation of the Li and Stephens algorithm for inferring a tree
sequence.

Python 3 only.
"""

import sys

__version__ = "0.5.0"

if sys.version_info[0] < 3:
    raise Exception("Python 3 only")


from .inference import *
