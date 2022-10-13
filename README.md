# tsinfer <img align="right" width="145" height="90" src="https://raw.githubusercontent.com/tskit-dev/tsinfer/main/docs/tsinfer_logo.svg">

[![CircleCI](https://circleci.com/gh/tskit-dev/tsinfer.svg?style=svg)](https://circleci.com/gh/tskit-dev/tsinfer) [![Build Status](https://travis-ci.org/tskit-dev/tsinfer.svg?branch=main)](https://travis-ci.org/tskit-dev/tsinfer) [![Docs Build](https://github.com/tskit-dev/tsinfer/actions/workflows/docs.yml/badge.svg)](https://tskit.dev/tsinfer/docs/stable/introduction.html) [![codecov](https://codecov.io/gh/tskit-dev/tsinfer/branch/main/graph/badge.svg)](https://codecov.io/gh/tskit-dev/tsinfer)


Infer a tree sequence from genetic variation data

The [documentation](https://tskit.dev/tsinfer/docs/stable) contains details of how to use this software, including [installation instructions](https://tskit.dev/tsinfer/docs/stable/installation.html).

The algorithm, its rationale, and results from testing on simulated and real data are described in the following [Nature Genetics paper](https://doi.org/10.1038/s41588-019-0483-y):

> Jerome Kelleher, Yan Wong, Anthony W Wohns, Chaimaa Fadil, Patrick K Albers and Gil McVean (2019) *Inferring whole-genome histories in large population datasets*. Nature Genetics **51**: 1330-1338

Please cite this if you use ``tsinfer`` in your work. Code to reproduce the results in the paper is present in a [separate GitHub repository](https://github.com/mcveanlab/treeseq-inference).

Note that `tsinfer` does not attempt to infer node times (i.e. branch lengths of the
inferred trees). If you require a tree sequence where the dates of common ancestors
are expressed in calendar or generation times, you should post-process the ``tsinfer``
output using software such as [``tsdate``](https://github.com/tskit-dev/tsdate).
