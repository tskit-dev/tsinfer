import os
import platform

import numpy
from setuptools import Extension
from setuptools import setup


IS_WINDOWS = platform.system() == "Windows"

libdir = "lib"
tskroot = os.path.join(libdir, "subprojects", "tskit")
tskdir = os.path.join(tskroot, "tskit")
kasdir = os.path.join(tskroot, "subprojects", "kastore")
includes = [libdir, tskroot, tskdir, kasdir]

tsi_source_files = [
    "ancestor_matcher.c",
    "ancestor_builder.c",
    "object_heap.c",
    "tree_sequence_builder.c",
    "err.c",
    "avl.c",
]
# We're not actually using very much of tskit at the moment, so
# just build the stuff we need.
tsk_source_files = ["core.c"]
kas_source_files = ["kastore.c"]

sources = (
    ["_tsinfermodule.c"]
    + [os.path.join(libdir, f) for f in tsi_source_files]
    + [os.path.join(tskdir, f) for f in tsk_source_files]
    + [os.path.join(kasdir, f) for f in kas_source_files]
)

libraries = []
if IS_WINDOWS:
    # Needed for generating UUIDs in tskit
    libraries.append("Advapi32")

_tsinfer_module = Extension(
    "_tsinfer",
    sources=sources,
    extra_compile_args=["-std=c99"],
    libraries=libraries,
    # Enable asserts by default.
    undef_macros=["NDEBUG"],
    include_dirs=includes + [numpy.get_include()],
)

setup(
    # The package name along with all the other metadata is specified in setup.cfg
    # However, GitHub's dependency graph can't see the package unless we put this here.
    name="tsinfer",
    ext_modules=[_tsinfer_module],
)
