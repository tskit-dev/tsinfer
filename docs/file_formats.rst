.. _sec_file_formats:

============
File formats
============

``tsinfer`` uses the excellent `zarr library <http://zarr.readthedocs.io/>`_
to encode data in a form that is both compact and efficient to process.
See the :ref:`API documentation <sec_api_file_formats>` for details on
how to construct and manipulate these files using Python. The
:ref:`tsinfer list <sec_cli>` command provides a way to print out a
summary of these files.


.. _sec_file_formats_ancestors:

**************
Ancestors File
**************

The ancestors file contains the ancestral haplotype data inferred from the
sample data in the :ref:`sec_inference_generate_ancestors` step.

.. todo:: Document the structure of the ancestors file.


.. _sec_file_formats_tree_sequences:

**************
Tree sequences
**************

The goal of ``tsinfer`` is to infer correlated genealogies from variation
data, and it uses the very efficient `succinct tree sequence
<https://tskit.dev/tskit/docs/stable/data-model.html>`_ data structure
to encode this output. Please see the `tskit documentation
<https://tskit.dev/tskit/docs/stable/>`_ for details on how to
process and manipulate such tree sequences.

The intermediate ``.ancestors.trees`` file produced by the
:ref:`sec_inference_match_ancestors` step is also a
tree sequence and can be loaded and analysed using the
`tskit API <https://tskit.dev/tskit/docs/stable/python-api.html>`_.
