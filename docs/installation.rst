.. _sec_installation:

############
Installation
############

Python 3.7 or newer is required for ``tsinfer``. Any Unix-like platform should
work (``tsinfer`` is tested on Linux, OS X, and Windows).

***************
Binary packages
***************

The most reliable way to install ``tsinfer`` is to install the binary conda package:
e.g.::

    $ conda install tsinfer -c conda-forge

you can then ``import tsinfer`` in python or use the ``tsinfer`` executable directly::

    $ tsinfer --help

**********************
Installing from source
**********************

It is also possible to install from source via ``pip`` (although see the issues below):

    $ python3 -m pip install tsinfer --user

which will install ``tsinfer`` to the Python installation corresponding to your
``python3`` executable. All requirements should be installed automatically.

To run the command line interface to ``tsinfer`` you can then use::

    $ python3 -m tsinfer --help


If your ``PATH`` is set up to point at the corresponding ``bin`` directory
you can also use the ``tsinfer`` executable directly::

    $ tsinfer --help

You may wish to install into a virtual environment
first using `venv <https://docs.python.org/3/library/venv.html>`_::

    $ python3 -m venv tsinfer-venv
    $ source tsinfer-venv/bin/activate
    (tsinfer-venv) $ python3 -m pip install tsinfer
    (tsinfer-venv) $ tsinfer --help

****************
Potential issues
****************

#. One of the dependencies of ``tsinfer``,
   `numcodecs <https://numcodecs.readthedocs.io/>`_, is compiled to
   use AVX2 instructions (where available) when installed using pip. This can lead to
   issues when ``numcodecs`` is compiled on a machine that supports AVX2
   and subsequently run on older machines that do not. To resolve this, ``numcodecs``
   has a ``DISABLE_NUMCODECS_AVX2`` variable which can be turned on before calling
   ``pip install``, see
   `these instructions <https://numcodecs.readthedocs.io/en/stable/#installation>`_
   for details.

#. There can be problems compiling from source using the default compilers under Mac OS
   (see https://github.com/tskit-dev/tsinfer/issues/376). The current workaround is
   either to compile from source by installing alternative python and C compilers via
   conda (``conda install -c conda-forge c-compiler``) or to install the binary
   packages via conda as recommended at the top of this page.
