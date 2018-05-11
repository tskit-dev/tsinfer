.. _sec_cli:

======================
Command line interface
======================

The command line interface in ``tsinfer`` is intended to provide a convenient
interface to the high-level :ref:`API functionality <sec_api>`. There are two
equivalent ways to invoke this program:

.. code-block:: bash

    $ tsinfer

or

.. code-block:: bash

    $ python3 -m tsinfer

The first form is more intuitive and works well most of the time. The second
form is useful when multiple versions of Python are installed or if the
:command:`tsinfer` executable is not installed on your path.

The :command:`tsinfer` program has five subcommands: :command:`list` prints a
summary of the data held in one of tsinfer's :ref:`file formats <sec_file_formats>`;
:command:`infer` runs the complete :ref:`inference process <sec_inference>` for a given
input :ref:`samples file <sec_file_formats_samples>`; and
:command:`generate-ancestors`, :command:`match-ancestors` and
:command:`match-samples` run the three parts of this inference
process as separate steps. Running the inference as separate steps like this
is recommended for large inferences as it allows for greater control over
the inference process.

++++++++++++++++
Argument details
++++++++++++++++

.. argparse::
    :module: tsinfer
    :func: get_cli_parser
    :prog: tsinfer
    :nodefault:

