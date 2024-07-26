---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

:::{currentmodule} tsinfer
:::


(sec_inference)=

# Inference overview

The process of inferring a tree sequence from variation data is split into a
number of different steps. We first review the
{ref}`requirements <sec_inference_data_requirements>` for input data
and discuss how such data is {ref}`imported <sec_inference_import_samples>`
into `tsinfer`'s sample data format. We then outline the three
basic steps for inference:
{ref}`generating ancestors <sec_inference_generate_ancestors>`,
{ref}`matching ancestors <sec_inference_match_ancestors>` and
{ref}`matching samples <sec_inference_match_samples>`.


(sec_inference_data_requirements)=

## Data requirements

Input haplotype data for tskit must satisfy the following requirements:

- Data must be *phased*.
- For each site used for inference we must know the *ancestral state*. Note that
  this is not necessarily the same as the REF column from VCF, and ancestral states
  must be computed using some other method. Sites with unknown ancestral states can be
  specified, but are not used for inference; instead, the ancestral state is imputed.
- Only biallelic sites can be used for inference.
- Missing data can be included in the haplotypes using the value
  {data}`tskit.MISSING_DATA` (-1). The inferred tree sequence will have values
  imputed for the haplotypes at these sites.


(sec_inference_data_model)=

## Data model

The data model for `tsinfer` is tightly integrated with
`tskit`'s {ref}`data model <sec_data_model>`
and uses the same concepts throughout. The intermediate file formats and APIs
described here provide a bridge between this model and existing data sources. For
convenience, we provide a brief description of concepts needed for importing
data into `tsinfer` here. Please see the
[tskit documentation](https://tskit.dev/tskit/docs/stable/)
for more detailed information.


(sec_inference_data_model_individual)=

### Individual

In `tsinfer` an individual defines one of the subjects for which we have
genotype data. Individuals may have arbitrary ploidy levels (i.e. haploid,
diploid, tetraploid, etc.). Different individuals within a dataset can have
different ploidy levels. You can add an individual, and specify arbitrary
{ref}`sec_inference_data_model_metadata` for it, using the
{meth}`.SampleData.add_individual` method; if you do not do so `tsinfer`
will create a default set of haploid individuals, one for each sampled genome.

The tree sequence that you infer from your data will contain sample nodes
(i.e. genomes) that are linked to these individuals: the `tskit`
documentation provides more detail on the
{ref}`distinction between individuals and the genomes they contain<sec_nodes_or_individuals>`.


(sec_inference_data_model_sample)=

### Sample

Each individual is composed of one or more ``sample`` genomes, depending on their
ploidy. If an individual is diploid they have two samples, one each for the
maternal and paternal chromosome copies. More generally, a ``node`` refers
to a maternal or paternal chromosome that is either a sample, or an
ancestor of our samples.

When we add an individual with ploidy `k` using
{meth}`.SampleData.add_individual`, `k` new samples are also added
which refer to this new individual. When adding genotype information using the
{meth}`.SampleData.add_site` method, the user must ensure that the observed
genotypes are in this same order.


(sec_inference_data_model_population)=

### Population

A population is some grouping of individuals. Populations principally
exist to allow us define metadata for groups of individuals (although
technically, population IDs are associated with samples).
Populations are added using the {meth}`.SampleData.add_population`
method.


(sec_inference_data_model_metadata)=

### Metadata

Metadata can be associated with populations and individuals in ``tsinfer``,
which results in this information being available in the final tree
sequence. Metadata allows us to incorporate extra information
that is not part of the basic data model; for example, we can record
the upstream ID of a given individual and their family relationships
with other individuals.

In `tsinfer`, metadata can be stored by providing a JSON encodable
mapping. This information is then stored as JSON, and embedded in the
final tree sequence object and can be recovered using the `tskit`
APIs.


(sec_inference_import_samples)=

## Import samples data

In `tsinfer` we make several passes through the input sample haplotypes
in order to construct ancestors and to find copying paths for samples. To
do this efficiently we store the data using the
[zarr library](http://zarr.readthedocs.io), which provides very fast access to
large arrays of numerical data compressed using cutting-edge
[compression methods](http://numcodecs.readthedocs.io). As a result, we
can store the input sample haplotypes and related metadata in a
fraction of the size of a compressed VCF and can process it efficiently  (although
still not as efficently as it is possible to analyse an equivalent tree sequence)

Rather than require the user to understand the internal structure of this
file format, we provide a simple {ref}`Python API <sec_api_file_formats>`
to allow the user to efficiently construct it from their own data.
An example of how to use this API is given in the {ref}`sec_usage` documentation.

We do not provide an automatic means of importing data from VCF (or any
other format) intentionally, as we believe that this would be extremely difficult to do.
As there is no universally accepted way of encoding ancestral state
information in VCF, in practise the user would most often have to write
a new VCF file with ancestral state and metadata information in a specific
form that we would require. Thus, it is more efficient to skip this intermediate
step and to directly produce a {ref}`format <sec_file_formats_samples>`
that is both compact and very efficient to process.


(sec_inference_generate_ancestors)=

## Generate ancestors

The first step in a `tsinfer` inference process is to generate a large
number of potential ancestors and to store these in an
{ref}`ancestors file <sec_file_formats_ancestors>`. The ancestors
file conventionally ends with `.ancestors`.

The ancestor generation algorithm is described in the Methods section of
[the original tsinfer paper](https://doi.org/10.1038/s41588-019-0483-y).

:::{todo}
Describe the ancestor generation algorithm in more detail here
:::

(sec_inference_match_ancestors_and_samples)=

## Matching ancestors & samples

After we have generated a set of potential ancestors and stored them in
an {ref}`ancestors file <sec_file_formats_ancestors>`, we then
run two matching steps. First we match the ancestors against each
other to generate an "ancestors tree sequence", then we match the samples
against this ancestors tree sequence to generate the final result.

In both matching stages, we can set parameters that adjust the
behaviour of the matching algorithm, in particular the `path_compression`
setting, and the `recombination_rate` and `mismatch_ratio` parameters.
The latter two only need to be specified if you wish to allow multiple
mutations to occur at a single site (i.e. breaking the infinite sites model
of mutation). Note, however, that multiple mutations are useful not only to
show true recurrent or back mutations in the evolutionary history of a
site, but also to represent errors in sequencing etc. which cause the
distribution of variation to fit poorly to the marginal tree at that site.
Hence, if there is error in your dataset, you may wish to experiment with
these settings to obtain optimal results.

(sec_inference_recombination_and_mismatch)=

### Recombination and mismatch

The `recombination_rate` parameter is either a floating point value giving a
single rate ({math}`\rho`) per unit length of genome, used to calculate the
genetic distance between adjacent sites, or an {class}`msprime.RateMap` object
(whose `.get_cumulative_mass` method provides the genetic distances instead).
The genetic distances are then used to derive an array of probabilities of
recombination between adjacent sites, ({math}`r`) used when assessing the relative
likelihood that a mismatch (and hence an extra mutation) may be responsible
for some patterns of variation at a site.

The `mismatch_ratio` parameter is only relevant if a recombination rate has
been provided. It is used to adjust the balance of recombination to multiple
mutations at a site. More specifically, a single probability of mismatch is
used for all sites, which is calculated from the median genetic distance between
inference sites, such that for conventional (small) distances, the probability of
a mismatch is approximately equal to the probability of recombination multiplied
by the `mismatch_ratio`. In other words, a mismatch ratio of 2 makes a recurrent
mutation two times more likely than a recombination event to explain why an
otherwise closely matching ancestral haplotype does not match at a particular site.
Setting a high `mismatch_ratio` therefore results in tree sequences with more
recurrent mutations and fewer recombinations (and edges). Setting a low value
results in tree sequences with more recombination events and edges, and fewer
mutations. In the limit, as the mismatch_ratio tends to zero, only one mutation
will be inferred per variable site. This is the default behaviour if no
`recombination_rate` is given or if there is only one inference site.
Alternatively, if `recombination_rate` is set, `mismatch_ratio` defaults to
1, which has been shown to give reasonable results in simulated inference of
human-like data with error.  As a rough guide, such simulations recommend
mismatch ratios between 1e-3 and 1e3.

(sec_inference_path_compression)=

### Path compression

The `path_compression` setting is used to further minimise recombination events
by looking for *shared recombination breakpoints*. A shared
breakpoint exists if a set of children share a breakpoint in the same position,
and they also have identical parents to the left of the breakpoint and identical
parents to the right of the breakpoint. Rather than supposing that these
children experienced multiple identical recombination events in parallel, we can
reduce the number of ancestral recombination events by postulating a "synthetic
ancestor" with this breakpoint, existing at a slightly older point
in time, from whom all the children are descended at this genomic position. We
call the algorithm used to implement this addition to the ancestral copying
paths, "path compression".

(sec_inference_match_ancestors)=

## Match ancestors

Matching ancestors is dependent on the time allocated to each ancestor; an
ancestor can only copy from any older ancestor. For each ancestor,
we find the most likely path through older ancestors: that is the path that
maximises the product of the probabilities of recombination and mismatch
over all sites.

:::{todo}
Schematic of the ancestors copying process.
:::

The copying path for each ancestor then describes its ancestry at every
point in the sequence: from a genealogical perspective, we know its
parent node. This information is encoded precisely as an
{ref}`edge <sec_edge_table_definition>` in a {ref}`tree sequence <sec_data_model>`.
Thus, we refer to the output of this step as the "ancestors tree sequence",
which is conventionally stored in a file ending with `.ancestors.trees`.

(sec_inference_match_samples)=

## Match samples

The final phase of a `tsinfer` inference consists of a number steps:

1. The first (and usually most time-consuming) is to find copying paths
   for our sample haplotypes through the ancestors. Each copying path
   corresponds to a set of tree sequence edges in precisely the same
   way as for ancestors, and the path compression algorithm can be equally
   applied here.

2. As we only use a subset of the available sites for inference
   (excluding by default any sites that are fixed, singletons, or where the ancestral
   state is unknown) we then place mutations on the inferred trees in order to
   represent the information at these sites. This is done using
   {meth}`tskit.Tree.map_mutations`.

3. Post-process the final tree sequence to
    a. modify the oldest ancestors in the tree sequence, in particular, by
       taking the single "grand MRCA" containing entirely ancestral states
       that `tsinfer` uses to provide a baseline ancestor for the matching
       algorithm, and splitting it into separate ancestral genomic fragments.
    b. remove topology at positions before the first site and one genomic unit
       after the last site, to avoid extrapolating genealogical patterns into
       regions for which we have no data.
    c. remove ancestral paths that do not lead to any of the samples by
       {meth}`simplifying <tskit.TreeSequence.simplify>` the final output. When
       simplifying, we keep non-branching ("unary") nodes, as they indicate
       ancestors which we have actively inferred, and
       for technical reasons keeping unary ancestors can also lead to better
       compression. Note that this means that not every internal node in the
       inferred tree sequence will correspond to a coalescent event.

:::{todo}
    1. Describe path compression here and above in the ancestors
       section
    2. Describe the structure of the output tree sequences; how the
       nodes are mapped, what the time values mean, etc.
:::
