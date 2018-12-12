from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


# Obscure magic required to allow numpy be used as an 'setup_requires'.
class build_ext(_build_ext):
    def finalize_options(self):
        super(build_ext, self).finalize_options()
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


d = "lib/"
_tsinfer_module = Extension(
    '_tsinfer',
    sources=[
        "_tsinfermodule.c", d + "ancestor_matcher.c",
        d + "ancestor_builder.c", d + "object_heap.c",
        d + "tree_sequence_builder.c", d + "block_allocator.c",
        d + "avl.c"],
    # Enable asserts by default.
    undef_macros=["NDEBUG"],
    extra_compile_args=["-std=c99"],
)

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="tsinfer",
    description="Infer tree sequences from genetic variation data.",
    long_description=long_description,
    packages=["tsinfer"],
    author="Jerome Kelleher",
    author_email="jerome.kelleher@well.ox.ac.uk",
    url="http://pypi.python.org/pypi/tsinfer",
    entry_points={
        'console_scripts': [
            'tsinfer=tsinfer.__main__:main',
        ]
    },
    setup_requires=['setuptools_scm', 'numpy'],
    cmdclass={'build_ext': build_ext},
    install_requires=[
        "numpy",
        "six",
        "tqdm",
        "humanize",
        "daiquiri",
        "msprime>=0.6.1",
        "numcodecs>=0.6",
        "zarr>=2.2",
        "lmdb",
        "sortedcontainers",
        "attrs",
    ],
    ext_modules=[_tsinfer_module],
    keywords=[],
    license="GNU GPLv3+",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
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
