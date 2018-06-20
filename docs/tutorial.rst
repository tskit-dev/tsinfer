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

Here we have phased haplotype data for five samples at six sites. We wish to infer the
genealogies that gave rise to this data set. To import the data into ``tsinfer`` we must know the
*ancestral state* for each site; there are many methods for achieving this, which are outside the
scope of this manual. Assuming that we know the ancestral state, we can then import our into a
``tsinfer`` :ref:`Sample data <sec_file_formats_samples>` file using the Python :ref:`API
<sec_api_file_formats>`:

.. code-block:: python

    import tsinfer

    with tsinfer.SampleData() as sample_data:
        sample_data.add_site(0, [0, 1, 0, 0, 0], ["A", "T"])
        sample_data.add_site(1, [0, 0, 0, 1, 1], ["G", "C"])
        sample_data.add_site(2, [0, 1, 1, 0, 0], ["C", "A"])
        sample_data.add_site(3, [0, 1, 1, 0, 0], ["G", "C"])
        sample_data.add_site(4, [0, 0, 0, 1, 1], ["A", "C"])
        sample_data.add_site(5, [0, 1, 0, 0, 0], ["T", "G"])

Here we create a new :class:`.SampleData` object for five samples. We then
sequentially add the data for each site one-by-one using the
:meth:`.Sample.add_site` method. The first argument for ``add_site`` is the
position of the site in genome coordinates. This can be any positive value
(even floating point), but site positions must be unique and sites must be
added in increasing order of position. For convenience we've given the sites
position 0 to 5 here, but they could be any values. The second argument for
``add_site`` is a list of *genotypes* for the site. Each value in a genotypes
array ``g`` is an integer: 0 represents the ancestral state for a site, and 1
the derived state. The third argument to ``add_site`` is the list of *alleles*
for the site. The first element of this list is the ancestral state and the
second the derived state (currently only biallelic sites are supported). Thus,
each call to ``add_site`` stores a single column of the original haplotype data
above. For example, the ancestral and derived states for the site at position 0
are "A"and "T" and the genotypes are 01000; together, these encode encode the
first column, ATAAA.


Once we have stored our data in a :class:`.SampleData` object, we can easily infer a tree
sequence using the Python API:

.. code-block:: python

    inferred_ts = tsinfer.infer(sample_data)

And that's it: we now have a fully functional msprime TreeSequence object that we can interrogate
in the usual ways. For example, we can look at the inferred topology and the stored haplotypes:

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

Note here that the inferred tree contains a *polytomy* at the root. This is a common feature of
trees inferred by ``tsinfer`` and signals that there was not sufficient information to resolve
the tree at this node.

Note also that we exactly recover the input haplotype data: ``tsinfer`` is guaranteed to
losslessly encode any give input data, regardless of the inferred topology.

++++++++++++++++++++
A simulation example
++++++++++++++++++++

The previous example showed how we can infer a tree sequence using the Python API for a trivial
toy example. However, for real data we will not prepare our data and infer the tree sequence all
in one go; rather, we will usually split the process into at least two distinct steps.

The first step in any inference is to prepare your data and import it into a :ref:`sample data
<sec_file_formats_samples>` file. For simplicity here we'll simulate some data under the
coalescent with recombination using `msprime
<https://msprime.readthedocs.io/en/stable/api.html#msprime.simulate>`_:

.. code-block:: python

    import tqdm
    import msprime
    import tsinfer

    ts = msprime.simulate(
        sample_size=10000, Ne=10**4, recombination_rate=1e-8,
        mutation_rate=1e-8, length=10*10**6, random_seed=42)
    ts.dump("simulation-source.trees")
    print("simulation done:", ts.num_trees, "trees and", ts.num_sites,  "sites")

    progress = tqdm.tqdm(total=ts.num_sites)
    with tsinfer.SampleData(
            path="simulation.samples", sequence_length=ts.sequence_length,
            num_flush_threads=2) as sample_data:
        for var in ts.variants():
            sample_data.add_site(var.site.position, var.genotypes, var.alleles)
            progress.update()
        progress.close()

Running the code we get::

    $ python3 simulation-example.py
    Simulation done: 36734 trees and 39001 sites
    100%|████████████████████████████████| 39001/39001 [00:51<00:00, 762.26it/s]

In this script we first run a simulation of a sample of 10 thousand 10 megabase chromosomes with
human-like parameters, which results in about 37K distinct trees and 39K segregating sites. We
then create a :class:`.SampleData` instance to store the data we have simulated as before, but
providing a few more parameters in this case. Firstly, we pass a ``path`` argument to provide a
filename in which to permanently store the information. We also provide a ``sequence_length``
argument (which defines the overall coordinate space for site positions) so that this value can
be recovered in the final tree sequence that we output later. Finally, we set
``num_flush_threads=2``, which tells ``tsinfer`` to use two background threads for compressing
data and flushing it to disk.

To allow us to keep track of how this process of compressing and storing the sample data is
progressing, we also set up a progress meter using `tqdm <https://github.com/tqdm/tqdm>`_. The
script output above shows the state of the progress meter at the end of this process, and shows
that it took about 50 seconds to import the data for this simulation into ``tsinfer``'s sample
data format.

Examining the files, we then see the following::

    $ ls -lh simulation*
    -rw-r--r-- 1 jk jk  22M May 12 11:06 simulation.samples
    -rw-r--r-- 1 jk jk 4.8M May 12 11:06 simulation-source.trees

The ``simulation.samples`` file is quite small, being only about four times the size of the
original the ``msprime`` tree sequence file. The :ref:`tsinfer command line interface <sec_cli>`
provides a useful way to examine files in more detail using the ``list`` (or ``ls``) command::

    $ tsinfer ls simulate.samples
    path                  = simulation.samples
    file_size             = 21.8 MiB
    format_name           = tsinfer-sample-data
    format_version        = (1, 0)
    finalised             = True
    uuid                  = ab667d05-06bc-4a15-ab85-ab5a0ac39c36
    num_provenances       = 1
    provenances/timestamp = shape=(1,); dtype=object;
    provenances/record    = shape=(1,); dtype=object;
    sequence_length       = 10000000.0
    num_populations       = 0
    num_individuals       = 10000
    num_samples           = 10000
    num_sites             = 39001
    num_inference_sites   = 35166
    populations/metadata  = shape=(0,); dtype=object;
    individuals/metadata  = shape=(10000,); dtype=object;
    individuals/location  = shape=(10000,); dtype=object;
    samples/individual    = shape=(10000,); dtype=int32;uncompressed size=40.0 kB
    samples/population    = shape=(10000,); dtype=int32;uncompressed size=40.0 kB
    samples/metadata      = shape=(10000,); dtype=object;
    sites/position        = shape=(39001,); dtype=float64;uncompressed size=312.0 kB
    sites/alleles         = shape=(39001,); dtype=object;
    sites/inference       = shape=(39001,); dtype=uint8;uncompressed size=39.0 kB
    sites/genotypes       = shape=(39001, 10000); dtype=uint8;uncompressed size=390.0 MB
    sites/metadata        = shape=(39001,); dtype=object;

Most of this output is not particularly interesting here, but we can see that the
``sites/genotypes`` array which holds all of the sample genotypes (and thus the vast bulk of the
actual data) requires about 390MB uncompressed. The ``tsinfer`` sample data format is therefore
achieving a roughly 20X compression in this case. In practise this means we can keep such files
lying around without taking up too much space.

Once we have our ``.samples`` file created, running the inference is straightforward::

    $ tsinfer infer simulation.samples -p -t 4
    ga-add   (1/6): 100%|███████████████████████| 35.2K/35.2K [00:02, 15.3Kit/s]
    ga-gen   (2/6): 100%|███████████████████████| 26.5K/26.5K [00:30,   862it/s]
    ma-match (3/6): 100%|██████████████████████▉| 26.5K/26.5K [01:02,   160it/s]
    ms-match (4/6): 100%|███████████████████████| 10.0K/10.0K [02:27,  67.9it/s]
    ms-paths (5/6): 100%|███████████████████████| 10.0K/10.0K [00:00, 26.0Kit/s]
    ms-sites (6/6): 100%|███████████████████████| 39.0K/39.0K [00:02, 15.5Kit/s]

Running the ``infer`` command runs the full inference pipeline in one go (the individual steps
are explained :ref:`here <sec_inference>`), writing the output, by default, to the tree sequence
file ``simulation.trees``. We provided two extra arguments to ``infer``: the ``-p`` flag
(``--progress``) gives us the progress bars show above, and ``-t 4`` (``--num-threads=4``) tells
``tsinfer`` to use four worker threads whenever it can use them.

This inference was run on a Core i3-530 processor (launched 2010) with 4GiB of RAM, and took
about four minutes. The maximum memory usage was about 600MiB.

Looking at our output files, we see::

    $ ls -lh simulation*
    -rw-r--r-- 1 jk jk  22M May 12 11:06 simulation.samples
    -rw-r--r-- 1 jk jk 4.8M May 12 11:06 simulation-source.trees
    -rw-r--r-- 1 jk jk 4.4M May 12 11:27 simulation.trees

Therefore our output tree sequence file that we have just inferred in less than five minutes is
*even smaller* than the original ``msprime`` simulated tree sequence! Because the output file is
also an :class:`msprime.TreeSequence`, we can use the same API to work with both.

.. code-block:: python

    import msprime

    source = msprime.load("simulation-source.trees")
    inferred = msprime.load("simulation.trees")

    subset = range(0, 6)
    source_subset = source.simplify(subset)
    inferred_subset = inferred.simplify(subset)

    tree = source_subset.first()
    print("True tree: interval=", tree.interval)
    print(tree.draw(format="unicode"))

    tree = inferred_subset.first()
    print("Inferred tree: interval=", tree.interval)
    print(tree.draw(format="unicode"))

Here we first load up our source and inferred tree sequences from their corresponding
``.trees`` files. Each of the trees in these tree sequences has 10 thousand samples
which is much too large to easily visualise. Therefore, to make things simple here
we subset both tree sequences down to their minimal representations for six
samples using :meth:`msprime.TreeSequence.simplify`.
(Using this tiny subset of the overall data allows us to get an informal
feel for the trees that are inferred by ``tsinfer``, but this is certainly
not a recommended approach for validating the inference!)

Once we've subsetted the tree sequences down to something that we can
comfortably look at, we then get the **first** tree from each tree sequence
and print it out. Note again that we are looking at only the first tree here;
there will be thousands more trees in each sequence. The output we get is::

    True tree: interval= (0.0, 488.1131463889296)
        4546
     ┏━━┻━┓
     ┃    900
     ┃  ┏━┻━┓
     ┃  ┃   854
     ┃  ┃ ┏━┻┓
     309┃ ┃  ┃
    ┏┻┓ ┃ ┃  ┃
    ┃ ┃ ┃ ┃  41
    ┃ ┃ ┃ ┃ ┏┻┓
    0 1 2 3 4 5

    Inferred tree: interval= (0.0, 3080.7017155601206)
      3493
    ┏━┻━┓
    ┃   3440
    ┃ ┏━┻━┓
    ┃ ┃   2290
    ┃ ┃ ┏━╋━━┓
    ┃ ┃ ┃ ┃  667
    ┃ ┃ ┃ ┃ ┏┻┓
    1 0 2 3 4 5

There are a number of things to note about these two trees. Firstly, it
is important to note that the intervals over which these trees apply are
quite different: the true tree covers the interval up to coordinate
488, but the inferred tree covers a much longer interval, up to 3080.
Our inference depends on the mutational information that is present.
If no mutations fall on a particular an edge in the tree sequence, then
we have no way of inferring that this edge existed. As a result, there
will be tree transitions that we cannot pick up. In the simulation that we
performed the mutation rate is equal to the recombination rate, and so
we expect that many recombinations will be invisible to us in the
output data.

For similar reasons, there will be many nodes in the tree at which
polytomies occur. Here we correctly infer that 4 and 5 coalesce
first and that 4 is a sibling of this node. However, we were not
able to distinguish the order in which 2 and 3 coalesced with
the ancestors of 4 and 5, and so we have three children of node 2290
in the inferred tree. (Note that, other than the samples, there is
no correspondence between the node IDs in the source tree and the
inferred tree.)

The final point to make here is that there will be incorrect inferences in some
trees. In this example we incorrectly inferred that 0 coalesces with the
ancestor of nodes 2, 3, 4 and 5 before 1.


.. todo::

    1. Add documentation links for msprime above so we can explain tree
       sequences there.

++++++++++++
Data example
++++++++++++

.. todo:: Worked example where we process a VCF to get some data.

.. todo:: Also add metadata and populations to this example, showing
     how we retrieve the metadata from the tree sequence.

