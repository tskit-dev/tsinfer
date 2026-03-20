- in resolve_samples_selection return the full list of samples when the selection 
is None, and percolate the effects through downstream code.

- In ``_find_duplicate_positions`` pass the actual positions as a parameter not the store
to avoid rereading the Zarr array.

- Use set operations to find if there's any duplicate sample IDs instead of loops

- Don't apply the filters until we do the actual iter_genotypes. No iter_variants in 
the init of MultiSourceView

- What is ``require_ancestral_match``? This very complicated for no real benefit

- Generall, the MultiSourceView init is doing too much work. It should find a minimal subset 
of potential sites across the sources. Client code can then ask to iterate over a subset 
of these. In the case of the ancestor inference, it can iterate over all sites first, 
and then compute its preferred set of inference site positions. 
The second pass (per interval) then passes the list of required positions to iter_genotypes.


DONE - delete

---

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

- Add tests for missing data across the stack. Is missing data tolerated in the
infer-ancestors stage? If we have missing data at an inference site is it imputed
in the expected way in the final trees? Is missing data imputed at augemented
sites correctly?
