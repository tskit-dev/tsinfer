from setuptools import setup, Extension


def get_numpy_includes():
    import numpy as np
    return np.get_include()


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
    libraries=["m"],
    extra_compile_args=["-std=c99"],
    include_dirs=[get_numpy_includes()],
)

long_description = "TODO"
# with open("README.txt") as f:
#     long_description = f.read()

setup(
    name="tsinfer",
    description="Infer tree sequences from genetic variation data.",
    long_description=long_description,
    packages=["tsinfer"],
    author="Jerome Kelleher",
    author_email="jerome.kelleher@well.ox.ac.uk",
    url="http://pypi.python.org/pypi/tsinfer",
    # entry_points={
    #     'console_scripts': [
    #         'htsget=htsget.cli:htsget_main',
    #     ]
    # },
    install_requires=[],
    ext_modules=[_tsinfer_module],
    keywords=[],
    license="GNU GPLv3+",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    setup_requires=['setuptools_scm', 'numpy'],
    use_scm_version={"write_to": "tsinfer/_version.py"},
)
