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
