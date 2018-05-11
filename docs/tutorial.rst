.. _sec_tutorial:

=======================
Tutorial
=======================

+++++++++++
Toy example
+++++++++++

Suppose that we have observed the following data::

    sample  haplotype
    0       AGCGAT
    1       TGACAG
    2       AGACAT
    3       ACCGCT
    4       ACCGCT

Here we have phased haplotype data for five samples at six sites. We wish
to infer the genealogies that gave rise to this data set. To import the
data into ``tsinfer`` we must know the *ancestral state* for each site; there
are many methods for achieving this, which are outside the scope of this manual.
Assuming that we know the ancestral state, we can then import our into a
``tsinfer`` :ref:`Sample data <sec_file_formats_samples>` file using the
Python :ref:`API <sec_api_file_formats>`:

.. code-block:: python

    import tsinfer

    with tsinfer.SampleData(num_samples=5) as sample_data:
        sample_data.add_site(0, ["A", "T"], [0, 1, 0, 0, 0])
        sample_data.add_site(1, ["G", "C"], [0, 0, 0, 1, 1])
        sample_data.add_site(2, ["C", "A"], [0, 1, 1, 0, 0])
        sample_data.add_site(3, ["G", "C"], [0, 1, 1, 0, 0])
        sample_data.add_site(4, ["A", "C"], [0, 0, 0, 1, 1])
        sample_data.add_site(5, ["T", "G"], [0, 1, 0, 0, 0])

Here we create a new :class:`.SampleData` object for five samples.
We then sequentially add the data for each site one-by-one using the
:func:`add_site` method. The first argument for ``add_site`` is the
position of the site in genome coordinates. This can be any
positive value (even floating point), but site positions must be unique
and sites must be added in increasing order of positions. For convenience
we've given the sites position 0 to 5 here, but they could be any values.

The second argument to ``add_site`` is the list of *alleles* for the
site. The first element of this list must be the ancestral state and
the second the derived state (currently only biallelic sites are
supported). The third argument for ``add_site`` is a list of
*genotypes* for the site. Each value in a genotypes array ``g`` is
an index into the list of alleles. Thus, each call to ``add_site``
stores a single column of the original haplotype data above. For
example, the ancestral and derived states for the site at position
0 are "A"and "T" and the genotypes are 01000; together, these encode
encode the first column, ATAAA.

Once we have stored our data in a :class:`.SampleData` object, we
can easily infer a tree sequence using the Python API:

.. code-block:: python

    inferred_ts = tsinfer.infer(sample_data)

And that's it: we now have a fully functional msprime TreeSequence object
that we can interrogate in the usual ways. For example, we can look at the
inferred topology and the stored haplotypes:

.. code-block:: python

    for tree in inferred_ts.trees():
        print(tree.draw(format="unicode"))

    for sample_id, h in enumerate(inferred_ts.haplotypes()):
        print(sample_id, h, sep="\t")

Which gives us the output::

        7
    ┏━━┳┻━━┓
    ┃  5   6
    ┃ ┏┻┓ ┏┻┓
    0 3 4 1 2

    0       AGCGAT
    1       TGACAG
    2       AGACAT
    3       ACCGCT
    4       ACCGCT

Note here that the inferred tree contains a *polytomy* at the root.
This is a common feature of trees inferred by ``tsinfer`` and signals
that there was not sufficient information to resolve the tree at
this node.

Note also that we exactly recover the input haplotype data: ``tsinfer``
is guaranteed to losslessly encode any give input data, regardless of
the inferred topology.

++++++++++++++++++++++++++++++++
Using the command line interface
++++++++++++++++++++++++++++++++

The previous example showed how we can infer a tree sequence using the
Python API. However, there is also a very useful :ref:`command
line interface <sec_cli>` to ``tsinfer``. The only difference is that
we must save our sample data to a file when importing:

.. code-block:: python

    with tsinfer.SampleData(num_samples=5, path="toy.samples") as sample_data:
        sample_data.add_site(0, ["A", "T"], [0, 1, 0, 0, 0])
        sample_data.add_site(1, ["G", "C"], [0, 0, 0, 1, 1])
        sample_data.add_site(2, ["C", "A"], [0, 1, 1, 0, 0])
        sample_data.add_site(3, ["G", "C"], [0, 1, 1, 0, 0])
        sample_data.add_site(4, ["A", "C"], [0, 0, 0, 1, 1])
        sample_data.add_site(5, ["T", "G"], [0, 1, 0, 0, 0])

This code is identical to the code above except we provide a filename to the
``path`` argument. Running the inference is then simple:

.. code-block:: bash

    $ tsinfer infer toy.samples

Running this command will infer the same tree sequence as above and store
it in the file ``toy.trees``.

.. todo::

    1. Add documentation links for msprime above so we can explain tree
       sequences there.

    2. Add some more less trivial from an msprime simulation, where
       we show how it works at scale.
