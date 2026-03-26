- Use vcz_validator to validate the ancestors zarr.

- Add the number of new edges along with final ts.num_trees to the extend_ts INFO output.

- Get ruff to enforce no imports except at the start of the file, as Claude just loves
using them

- It's not clear why we're using kwargs.get in pipeline.match. This is an antipattern,
avoid unless there's a very strong reason for it.

- In extend_ts we could save some sorting time by keeping track of the oldest parent
and then finding the first edge for that parent. We can then use this as the start_index
for sorting, which would likely improve things.

- Consider how pedigree data may be associated with individuals, and how this could be
passed through to the tree sequence.

- Switch the ``_erase_flanks`` call to use the ``delete_intervals`` function in tskit
directly on the sequence intervals from the metadata.

- Add top-level sequence_length key which should be the length of the reference sequence
for the contig in question.

- Add tests for missing data across the stack. Is missing data tolerated in the
infer-ancestors stage? If we have missing data at an inference site is it imputed
in the expected way in the final trees? Is missing data imputed at augemented
sites correctly?
