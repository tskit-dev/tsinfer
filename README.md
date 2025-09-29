# tsinfer <img align="right" width="145" height="90" src="https://raw.githubusercontent.com/tskit-dev/tsinfer/main/docs/tsinfer_logo.svg">

[![License](https://img.shields.io/github/license/tskit-dev/tsinfer)](https://github.com/tskit-dev/tsinfer/blob/main/LICENSE) [![PyPI version](https://img.shields.io/pypi/v/tsinfer.svg)](https://pypi.org/project/tsinfer/) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/tsinfer.svg)](https://pypi.org/project/tsinfer/) [![Docs Build](https://github.com/tskit-dev/tsinfer/actions/workflows/docs.yml/badge.svg)](https://github.com/tskit-dev/tsinfer/actions/workflows/docs.yml) [![Binary wheels](https://github.com/tskit-dev/tsinfer/actions/workflows/wheels.yml/badge.svg)](https://github.com/tskit-dev/tsinfer/actions/workflows/wheels.yml) [![Tests](https://github.com/tskit-dev/tsinfer/actions/workflows/tests.yml/badge.svg)](https://github.com/tskit-dev/tsinfer/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/tskit-dev/tsinfer/branch/main/graph/badge.svg)](https://codecov.io/gh/tskit-dev/tsinfer)


Infer whole-genome tree sequences from genetic variation data. Tsinfer implements efficient algorithms to reconstruct ancestral haplotypes and recombination breakpoints, producing succinct tree sequences that capture shared ancestry across the genome. It scales to large cohorts and integrates cleanly with the broader tskit ecosystem for downstream statistics and analysis.

The documentation ([stable](https://tskit.dev/tsinfer/docs/stable/) • [latest](https://tskit.dev/tsinfer/docs/latest/)) contains details of how to use this software, including [installation instructions](https://tskit.dev/tsinfer/docs/stable/installation.html).

## Installation

```bash
python -m pip install tsinfer
# or
conda install -c conda-forge tsinfer
```

The initial algorithm, its rationale, and results from testing on simulated and real data are described in the following [Nature Genetics paper](https://doi.org/10.1038/s41588-019-0483-y):

> Jerome Kelleher, Yan Wong, Anthony W Wohns, Chaimaa Fadil, Patrick K Albers and Gil McVean (2019) *Inferring whole-genome histories in large population datasets*. Nature Genetics **51**: 1330-1338

_Tsinfer_ versions [0.2.0](https://github.com/tskit-dev/tsinfer/releases/tag/0.2.0) onwards allow missing data and provide a fully parameterised Li & Stephens matching algorithm (i.e. which allows mismatch). These improvements are described in the
following [Science paper](https://doi.org/10.1126/science.abi8264):

> Anthony Wilder Wohns, Yan Wong, Ben Jeffery, Ali Akbari, Swapan Mallick, Ron Pinhasi, Nick Patterson, David Reich, Jerome Kelleher, and Gil McVean (2022) A unified genealogy of modern and ancient genomes. Science 375: eabi8264

Please cite either or both of these if you use ``tsinfer`` in your work. Code to reproduce the results in the first paper is present in a [separate GitHub repository](https://github.com/mcveanlab/treeseq-inference).

Note that `tsinfer` does not attempt to infer node times (i.e. branch lengths of the
inferred trees). If you require a tree sequence where the dates of common ancestors
are expressed in calendar or generation times, you should post-process the ``tsinfer``
output using software such as [``tsdate``](https://github.com/tskit-dev/tsdate).
