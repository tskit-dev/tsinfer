.. _sec_api:

=================
API Documentation
=================

.. _sec_api_file_formats:

************
File formats
************

.. autofunction:: tsinfer.load

+++++++++++
Sample data
+++++++++++

.. autoclass:: tsinfer.VariantData
    :members:
    :inherited-members:


.. autoclass:: tsinfer.SampleData
    :members:
    :inherited-members:

.. todo::

    1. Add documentation for the data attributes in read-mode.
    2. Define copy mode.
    3. Provide example of updating inference_sites

+++++++++++++
Ancestor data
+++++++++++++

.. autoclass:: tsinfer.AncestorData
    :members:
    :inherited-members:

.. todo::

    1. Add documentation for the data attributes in read-mode.


.. _sec_api_file_inference:

*****************
Running inference
*****************

.. autofunction:: tsinfer.infer

.. autofunction:: tsinfer.generate_ancestors

.. autoclass:: tsinfer.GenotypeEncoding
   :members:

.. autofunction:: tsinfer.match_ancestors

.. autofunction:: tsinfer.match_samples

.. autofunction:: tsinfer.augment_ancestors

.. autofunction:: tsinfer.post_process

*****************
Batched inference
*****************

.. autofunction:: tsinfer.match_ancestors_batch_init

.. autofunction:: tsinfer.match_ancestors_batch_groups

.. autofunction:: tsinfer.match_ancestors_batch_group_partition

.. autofunction:: tsinfer.match_ancestors_batch_group_finalise

.. autofunction:: tsinfer.match_ancestors_batch_finalise

.. autofunction:: tsinfer.match_samples_batch_init

.. autofunction:: tsinfer.match_samples_batch_partition

.. autofunction:: tsinfer.match_samples_batch_finalise


*****************
Container classes
*****************

.. autoclass:: tsinfer.Variant

.. autoclass:: tsinfer.Site


**********
Exceptions
**********

.. autoexception:: tsinfer.FileFormatError

