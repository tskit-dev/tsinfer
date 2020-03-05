.. _sec_inference:

==================
Inference overview
==================

The process of inferring a tree sequence from variation data is split into a
number of different steps. We first review the
:ref:`requirements <sec_inference_data_requirements>` for input data
and discuss how such data is :ref:`imported <sec_inference_import_samples>`
into ``tsinfer``'s sample data format. We then outline the three
basic steps for inference:
:ref:`generating ancestors <sec_inference_generate_ancestors>`,
:ref:`matching ancestors <sec_inference_match_ancestors>` and
:ref:`matching samples <sec_inference_match_samples>`.

.. _sec_inference_data_requirements:

*****************
Data requirements
*****************

Input haplotype data for tskit must satisfy the following requirements:

- Data must be *phased*.
- For each site used for inference we must know the *ancestral state*. Note that this is
  not necessarily the same as the REF column from VCF, and ancestral
  states must be computed using some other method.
- Only biallelic sites can be used for inference.


.. _sec_inference_data_model:

**********
Data model
**********

The data model for ``tsinfer`` is tightly integrated with
``tskit``'s `data model <https://tskit.readthedocs.io/en/latest/data-model.html>`_
and uses the same concepts throughout. The intermediate file formats and APIs
described here provide a bridge between this model and existing data sources. For
convenience, we provide a brief description of concepts needed for importing
data into ``tsinfer`` here. Please see the `tskit documentation
<https://tskit.readthedocs.io/>`_ for more detailed information.

.. _sec_inference_data_model_individual:

++++++++++
Individual
++++++++++

In ``tsinfer`` an individual defines one of the subjects for which we have
genotype data. Individuals may have arbitrary ploidy levels (i.e. haploid,
diploid, tetraploid, etc.). Different individuals within a dataset can have
different ploidy levels. You can add an individual, and specify arbitrary
:ref:`sec_inference_data_model_metadata` for it, using the
:meth:`.SampleData.add_individual` method; if you do not do so ``tsinfer``
will create a default set of haploid individuals, one for each sampled genome.

The tree sequence that you infer from your data will contain sample nodes
(i.e. genomes) that are linked to these individuals: the ``tskit``
documentation provides more detail on the
:ref:`distinction between individuals and the genomes they contain<sec_nodes_or_individuals>`.

.. _sec_inference_data_model_sample:

++++++
Sample
++++++

Each individual is composed of one or more ``sample`` genomes, depending on their
ploidy. If an individual is diploid they have two samples, one each for the
maternal and paternal chromosome copies. More generally, a ``node`` refers
to a maternal or paternal chromosome that is either a sample, or an
ancestor of our samples.

When we add an individual with ploidy ``k`` using
:meth:`.SampleData.add_individual`, ``k`` new samples are also added
which refer to this new individual. When adding genotype information using the
:meth:`.SampleData.add_site` method, the user must ensure that the observed
genotypes are in this same order.

.. _sec_inference_data_model_population:

++++++++++
Population
++++++++++

A population is some grouping of individuals. Populations principally
exist to allow us define metadata for groups of individuals (although
technically, population IDs are associated with samples).
Populations are added using the :meth:`.SampleData.add_population`
method.

.. _sec_inference_data_model_metadata:

++++++++
Metadata
++++++++

Metadata can be associated with populations and individuals in ``tsinfer``,
which results in this information being available in the final tree
sequence. Metadata allows us to incorporate extra information
that is not part of the basic data model; for example, we can record
the upstream ID of a given individual and their family relationships
with other individuals.

In ``tsinfer``, metadata can be stored by providing a JSON encodable
mapping. This information is then stored as JSON, and embedded in the
final tree sequence object and can be recovered using the ``msprime``
APIs.

.. _sec_inference_import_samples:

*******************
Import samples data
*******************

In ``tsinfer`` we make several passes through the input sample haplotypes
in order to construct ancestors and to find copying paths for samples. To
do this efficiently we store the data using the `zarr library
<http://zarr.readthedocs.io>`_, which provides very fast access to
large arrays of numerical data compressed using cutting-edge
`compression methods <http://numcodecs.readthedocs.io>`_. As a result, we
can store the input sample haplotypes and related metadata in a
fraction of the size of a compressed VCF as well as process it efficiently.

Rather than require the user to understand the internal structure of this
file format, we provide a simple :ref:`Python API <sec_api_file_formats>`
to allow the user to efficiently construct it from their own data.
An example of how to use this API is given in the :ref:`sec_tutorial`.

We do not provide an automatic means of importing data from VCF (or any
other format) intentionally, as we believe that this would be extremely difficult to do.
As there is no universally accepted way of encoding ancestral state
information in VCF, in practise the user would most often have to write
a new VCF file with ancestral state and metadata information in a specific
form that we would require. Thus, it is more efficient to skip this intermediate
step and to directly produce a :ref:`format <sec_file_formats_samples>`
that is both compact and very efficient to process.

.. _sec_inference_generate_ancestors:

******************
Generate ancestors
******************

The first step in a ``tsinfer`` inference process is to generate a large
number of potential ancestors and to store these in an
:ref:`ancestors file <sec_file_formats_ancestors>`. The ancestors
file conventionally ends with ``.ancestors``.

.. todo:: Describe the ancestor generation algorithm.


.. _sec_inference_match_ancestors:

***************
Match ancestors
***************

After we have generated a set of potential ancestors and stored them in
and :ref:`ancestors file <sec_file_formats_ancestors>`, we then
run a matching process on these ancestors. Each ancestor occurs at a
given time, and an ancestor can copy from any older ancestor. For each
ancestor, we find a path through older ancestors that minimises the
number of recombination events.

As well as minimising recombination events by finding the best path, we can also
minimise events by looking for *shared recombination breakpoints*. A shared
breakpoint exists if a set of children share a breakpoint in the same position,
and they also have identical parents to the left of the breakpoint and identical
parents to the right of the breakpoint. Rather than supposing that these
children experienced multiple identical recombination events in parallel, we can
reduce the number of ancestral recombination events by postulating a "synthetic
ancestor" with this breakpoint, existing at a slightly older point
in time, from whom all the children are descended at this genomic position. We
call the algorithm used to implement this addition to the ancestral copying
paths, "path compression".


.. todo:: Schematic of the ancestors copying process.

The copying path for each ancestor then describes its ancestry at every
point in the sequence: from a genealogical perspective, we know its
parent node. This information is encoded precisely as an `edge
<https://tskit.readthedocs.io/en/latest/data-model.html#edge-table>`_ in a
`tree sequence
<https://tskit.readthedocs.io/en/latest/data-model.html>`_.
Thus, we refer to the output of this step as the "ancestors tree sequence",
which is conventionally stored in a file ending with ``.ancestors.trees``.

.. _sec_inference_match_samples:

*************
Match samples
*************

The final phase of a ``tsinfer`` inference consists of a number steps:

1. The first (and usually most time-consuming) is to find copying paths
   for our sample haplotypes through the ancestors. Each copying path
   corresponds to a set of tree sequence edges in precisely the same
   way as for ancestors, and the path compression algorithm can be equally
   applied here.


2. As we only use a subset of the available sites for inference
   (excluding by default any sites that are fixed or singletons)
   we then place mutations on the inferred trees in order to
   represent the information at these sites. This is done using the tskit
   `map_mutations <https://tskit.readthedocs.io/en/latest/python-api.html#tskit.Tree.map_mutations>`_.
   method.
   

3. Remove ancestral paths that do not lead to any of the samples by
   `simplifying
   <https://tskit.readthedocs.io/en/latest/python-api.html#tskit.TreeSequence.simplify>`_
   the final tree sequence. When simplifying, we keep non-branching ("unary")
   nodes, as they indicate ancestors which we have actively inferred, and
   for technical reasons keeping unary ancestors can also lead to better
   compression. Note that this means that not every internal node in the
   inferred tree sequence will correspond to a coalescent event.

.. todo::
    1. Describe path compression here and above in the ancestors
       section
    2. Describe the structure of the output tree sequences; how the
       nodes are mapped, what the time values mean, etc.

