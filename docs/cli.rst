.. _sec_cli_reference:

======================
Command line interface
======================

The ``tsinfer`` command line interface runs the inference pipeline using a
TOML configuration file. See the :ref:`quickstart <sec_quickstart>` for an
introduction and the :ref:`config reference <sec_config_reference>` for all
available options.

.. code-block:: bash

    $ tsinfer run config.toml --threads 4 -v

.. click:: tsinfer.cli:main
   :prog: tsinfer
   :nested: full
