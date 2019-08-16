.. _sec_installation:

############
Installation
############

Python 3.3 or newer is required for ``tsinfer``. Any Unix-like platform should work
(``tsinfer`` is tested on Linux and OSX).

Please use ``pip`` to install,
e.g.::

    $ python3 -m pip install tsinfer --user

will install ``tsinfer`` to the Python installation corresponding to your
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
    (tsinfer-venv) $ pip install tsinfer
    (tsinfer-venv) $ tsinfer --help

.. _sec_installation_installation_problems:

+++++++++++++++++++++
Installation problems
+++++++++++++++++++++


There are situations (usually where the GSL libraries are not in the
default locations) where ``tskit`` installation can fail. The same issue can 
occur when installing ``msprime``, so you can consult the 
`msprime installation documentation <https://msprime.readthedocs.io/en/stable/installation.html>`_
for details on the various ways to address this problem.

Note that one of the dependencies of ``tsinfer``, ``numcodecs``, can be compiled to
use AVX2 instructions if those are available on your hardware. This has led to
issues when installing on hardware with AVX2, then trying to use ``tsinfer``
from a cluster compute node without AVX2. To resolve this, ``numcodecs`` has a
``DISABLE_NUMCODECS_AVX2`` variable which can be turned on before calling
``pip install``, see 
`these instructions <https://numcodecs.readthedocs.io/en/stable/#installation>`_
for details.
