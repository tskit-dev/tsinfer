.. _sec_introduction:

============
Introduction
============

The goal of ``tsinfer`` is to infer *succinct tree sequences* from observed
genetic variation data. A succinct tree sequence (or
:ref:`tree sequence<tutorials:sec_what_is>`, for short)
is an efficient way of representing the correlated genealogies that
describe the ancestry of many species. By inferring these tree sequences, we
make two very important gains:

1. We obtain an approximation of the true history of our sampled data, which
   may be useful for other inferential tasks.

2. The data structure itself is an extremely concise and efficient means of
   storing and processing the data that we have.

The output of ``tsinfer`` is a :class:`tskit.TreeSequence` and so the
full `tskit API <https://tskit.dev/tskit/docs/stable>`_ can be used to
analyse real data, in precisely the same way that it is commonly used
to analyse simulation data, for example, from `msprime <https://tskit.dev/msprime/docs/stable/>`_.

.. note::

  ``Tsinfer`` infers the genetic relationships between sampled genomes, but does not
  attempt to infer the *times* of most recent common ancestors (tMRCAs) in the genealogy.
  If you are using the output of ``tsinfer`` in downstream analysis that relies on
  node times, you are advised not to use the inferred tree sequences directly; instead,
  you should post-process the ``tsinfer`` output using software such as
  `tsdate <https://tsdate.readthedocs.io>`_ that attempts to assign calendar or
  generation times to the tree sequence nodes.