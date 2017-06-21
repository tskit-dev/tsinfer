from __future__ import division
from __future__ import print_function

import subprocess
# TODO this will need to get imported somewhere else so that we can add
# numpy as a setup-requires.
import numpy as np

# First, we try to use setuptools. If it's not available locally,
# we fall back on ez_setup.
try:
    from setuptools import setup, Extension
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, Extension

long_description = "TODO"

d = "lib/"
_tsinfer_module = Extension(
    '_tsinfer',
    sources=[
        "_tsinfermodule.c", d + "ls.c", d + "ancestor_matcher.c",
        d + "ancestor_store.c", d + "ancestor_builder.c", d + "object_heap.c",
        d + "ancestor_sorter.c", d + "ancestor_store_builder.c", d + "traceback.c"],
    # Enable asserts by default.
    undef_macros=["NDEBUG"],
    libraries=["m"],
    include_dirs = [np.get_include()],
)

setup(
    name="tsinfer",
    description="Infer tree sequences from genetic variation data.",
    long_description=long_description,
    packages=["tsinfer"],
    author="Jerome Kelleher",
    author_email="jerome.kelleher@well.ox.ac.uk",
    url="http://pypi.python.org/pypi/tsinfer",
    ext_modules=[_tsinfer_module],
)
