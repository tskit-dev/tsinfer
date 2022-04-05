import os
import platform

from setuptools import Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

IS_WINDOWS = platform.system() == "Windows"


# Obscure magic required to allow numpy be used as an 'setup_requires'.
class build_ext(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


libdir = "lib"
tskroot = os.path.join(libdir, "subprojects", "tskit")
tskdir = os.path.join(tskroot, "tskit")
kasdir = os.path.join(libdir, "subprojects", "kastore")
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
    # Enable asserts by default.
    undef_macros=["NDEBUG"],
    extra_compile_args=["-std=c99"],
    include_dirs=includes,
    libraries=libraries,
)

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="tsinfer",
    description="Infer tree sequences from genetic variation data.",
    long_description=long_description,
    packages=["tsinfer"],
    author="Jerome Kelleher",
    author_email="jerome.kelleher@bdi.ox.ac.uk",
    url="http://pypi.python.org/pypi/tsinfer",
    python_requires=">=3.7",
    entry_points={"console_scripts": ["tsinfer=tsinfer.__main__:main"]},
    setup_requires=["setuptools_scm", "numpy"],
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy",
        "six",
        "tqdm",
        "humanize",
        "daiquiri",
        "tskit>=0.3.7",
        "numcodecs>=0.6",
        # issues 965 and 967 at zarr-python prevent usage of 2.11.0 and 2.11.1
        "zarr>=2.2,!=2.11.0,!=2.11.1,!=2.11.2",
        "lmdb",
        "sortedcontainers",
        "attrs>=19.2.0",
    ],
    ext_modules=[_tsinfer_module],
    keywords=[],
    license="GNU GPLv3+",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    use_scm_version={"write_to": "tsinfer/_version.py"},
)
