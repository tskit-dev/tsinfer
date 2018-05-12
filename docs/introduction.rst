.. _sec_introduction:

============
Introduction
============

The goal of ``tsinfer`` is to infer *succinct tree sequences* from observed
genetic variation data. A succinct tree sequence (or tree sequence, for short)
is an efficient way of representing the correlated genealogies that
describe the ancestry of many species. By inferring these tree sequences, we
make two very important gains:

1. We obtain an approximation of the true history of our sampled data, which
   may be useful for other inferential tasks.

2. The data structure itself is an extremely concise and efficient means of
   storing and processing the data that we have.

The output of ``tsinfer`` is an :class:`msprime.TreeSequence` and so the
full `msprime API <https://msprime.readthedocs.io/>`_ can be used to
analyse real data, in precisely the same way that it is currently used
to analyse simulation data.
