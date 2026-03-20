- Use vcz_validator to validate the ancestors zarr.

- Add the number of new edges along with final ts.num_trees to the extend_ts INFO output.

- It's not clear that HaplotypeReader is behaving correctly on chunk reading with samples_selection.
  This needs careful consideration.

- Get ruff to enforce no imports except at the start of the file, as Claude just loves
using them

- It's not clear why we're using kwargs.get in pipeline.match. This is an antipattern,
avoid unless there's a very strong reason for it.

- In extend_ts we could save some sorting time by keeping track of the oldest parent
and then finding the first edge for that parent. We can then use this as the start_index
for sorting, which would likely improve things.


- Consider how pedigree data may be associated with individuals, and how this could be
passed through to the tree sequence.

- Enforce that sample IDs are unique across input sources for main inference and
augment sites. You can't used the same sample twice.

- Switch the ``_erase_flanks`` call to use the ``delete_intervals`` function in tskit
directly on the sequence intervals from the metadata.

- Add top-level sequence_length key which should be the length of the reference sequence
for the contig in question.

