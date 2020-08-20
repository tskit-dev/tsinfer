.. _sec_api:

=================
API Documentation
=================

.. _sec_api_file_formats:

++++++++++++
File formats
++++++++++++

.. autoclass:: tsinfer.SampleData
    :members:
    :inherited-members:

.. todo::

    1. Add documentation for the data attributes in read-mode.
    2. Define copy mode.
    3. Provide example of updating inference_sites

.. autoclass:: tsinfer.AncestorData
    :members:
    :inherited-members:

.. todo::

    1. Add documentation for the data attributes in read-mode.


.. autofunction:: tsinfer.load

.. _sec_api_file_inference:

+++++++++++++++++
Running inference
+++++++++++++++++

.. autofunction:: tsinfer.infer

.. autofunction:: tsinfer.generate_ancestors

.. autofunction:: tsinfer.match_ancestors

.. autofunction:: tsinfer.match_samples

.. autofunction:: tsinfer.augment_ancestors

++++++++++
Exceptions
++++++++++

.. autoexception:: tsinfer.FileFormatError

