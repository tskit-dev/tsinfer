.. _sec_installation:

############
Installation
############

Python 3.3 or newer is required for ``tsinfer``. Please use ``pip`` to install,
e.g.::

    $ python3 -m pip install tsinfer --user

will install ``tsinfer`` to the Python installation corresponding to your
``python3`` executable. To run the command line interface to
``tsinfer`` you can then use::

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
