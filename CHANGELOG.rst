********************
[0.2.3] - 2022-04-08
********************

**Features**

- Added ``ancestor(id)`` to ``AncestorData`` class.
  (:pr:`570`, :issue:`569`, :user:`hyanwong`)

**Fixes**

- Mark zarr 2.11.0, 2.11.1 and 2.11.2 as incompatible due to ``zarr-python``
  bugs #965 and #967.
  (:issue:`643`, :pr:`657`, :user:`benjeffery`)

********************
[0.2.2] - 2022-02-23
********************

**Bugfixes**:

- Mutations at non-inference sites are now guaranteed to be fully parsimonious.
  Previous versions required a mutation above the root when the input ancestral state
  disagreed with the ancestral state produced by the parsimony algorithm. Now fixed by
  using the new map_mutations code from tskit 0.3.7 (:pr:`557`, :user:`hyanwong`)

**New Features**:

**Breaking changes**:

- Oldest nodes in a standard inferred tree sequence are no longer set to frequencies ~2
  and ~3 (i.e. 2 or 3 times as old as all the other nodes), but are spaced above the
  others by the mean time between unique ancestor ages (:pr:`485`, :user:`hyanwong`)
  
- The ``tsinfer.SampleData.from_tree_sequence()`` function now defaults to setting
  ``use_sites_time`` and ``use_individuals_time`` to ``False`` rather than ``True``
  (:pr:`599`, :user:`hyanwong`)

********************
[0.2.1] - 2021-05-26
********************

Bugfix release

**Bugfixes**:

- Fix a bug in the core LS matching algorithm in which the rate of recombination
  was being incorrectly computed (:issue:`493`, :pr:`514`, :user:`jeromekelleher`,
  :user:`hyanwong`).

- ``tsinfer.verify()`` no longer requires that non-ancestral alleles in a SampleData
  and Tree Sequence file are in the same order (:issue:`490`, :pr:`492`,
  :user:`hyanwong`).

**New Features**:

- Inferred ancestral haplotypes may be truncated via
  ``AncestorData.truncate_ancestors()`` to improve performance when inferring large
  datasets (:issue:`276`, :pr:`467`, :user:`awohns`).

**Breaking changes**:

- tsinfer now requires Python 3.7


********************
[0.2.0] - 2020-12-18
********************

Major feature release, including some incompatible file format and API updates.

**New features**:

- Mismatch and recombination parameters can now be specified via the
  recombination_rate and mismatch_ratio arguments in the Python API.

- Missing data can be accomodated in SampleData using the tskit.MISSING_DATA
  value in input genotypes. Missing data will be imputed in the output
  tree sequence.

- Metadata schemas for population, individual, site and tree sequence metadata
  can now we be specified in the SampleData format. These will be included
  in the final tree sequence and allow for automatic decoding of JSON metadata.

- Map non-inference sites onto the tree by using the tskit ``map_mutations``
  parsimony method. This allows us to support sites with > 2 alleles.

- Historical (non-contemporaneous) samples can now be accommodated in inference,
  assuming that the true dates of ancestors have been set, by using the concept
  of "proxy samples". This is done via the new function
  ``AncestorData.insert_proxy_samples()``, then setting the new
  parameter ``force_sample_times=True`` when matching samples.

- The default tree sequence returned after inference when ``simplify=True`` retains
  unary nodes (i.e. simplify is done with ``keep_unary=True``.


**Breaking changes**:

- The ancestors tree sequence now contains the real alleles and not
  0/1 values as before.

- Times for undated sites now use frequencies (0..1), not as counts (1..num_samples),
  and are now stored as ``tskit.UNKNOWN_TIME``, then calculated on the fly in the
  variants() iterator.

- The SampleData file no longer accepts the ``inference`` argument to add_site.
  This functionality has been replaced by the ``exclude_positions`` argument
  to the ``infer`` and ``generate_ancestors`` functions.

- The SampleData format is now at version 5, and older versions cannot be read.
  Users should rerun their data ingest pipelines.

- Users can specify variant ages, via ``sample_data.add_sites(... , time=user_time)``.
  If not ``None``, this overrides the default time position of an ancestor, otherwise
  ancestors are ordered in time by using the frequency of the derived variant (#143).

- Change "age" to "time" to match tskit/msprime notation, and to avoid confusion
  with the age since birth of an individual (#149). Together with the 2 changes below,
  this addition bumped the file format to 3.0.

- Add the ability to record user-specified times for individuals, and therefore
  the samples contained in them (currently ignored during inference). Times are
  added using ``sample_data.add_individual(... , time=user_time)`` (#190).

- Change ``tsinfer.UNKNOWN_ALLELE`` to ``tskit.MISSING_DATA`` for marking unknown regions
  of ancestral haplotypes (#188) . This also involves changing the allele storage to a
  signed int from ``np.uint8`` which matches the tskit v0.2 format for allele storage
  (see https://github.com/tskit-dev/tskit/issues/144).

**Bugfixes**:

- Individuals and populations in the SampleData file are kept in the returned tree
  sequence, even if they are not referenced by any sample. The individual and population
  ids are therefore guaranteed to stay the same between the sample data file and the
  inferred tree sequence. (:pr:`348`)

********************
[0.1.4] - 2018-12-12
********************

Bugfix release.

- Fix issue caused by upstream changes in numcodecs (#136).

********************
[0.1.3] - 2018-11-02
********************

Release corresponding to code used in the preprint.

********************
[0.1.2] - 2018-06-18
********************

Minor update to take advantage of msprime 0.6.0's Population and Individual
objects and fix various bugs.


**Breaking changes**:

- Bumped SampleData file format version to 1.0 because of the addition
  of individuals and populations. Older SampleData files will not be
  readable and must be regenerated.

- Changed the order of the ``alleles`` and ``genotypes`` arguments to
  SampleData.add_site.

**New features**:

- Sample and individual metadata now handled correctly.

- Added --no-simplify option to CLI and simplify=True option to infer function.

- Better handling of missing files (raises correct exceptions).

- tsinfer list now presents basic information for .trees files.

**Bug fixes**:

- Degenerate examples with zero inference sites are now rooted (#44)

- Simplify=False results in tree sequence with correct sample nodes.
