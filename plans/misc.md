- Use vcz_validator to validate the ancestors zarr.

- Add the number of new edges along with final ts.num_trees to the extend_ts INFO output.

- It's not clear that HaplotypeReader is behaving correctly on chunk reading with samples_selection.
  This needs careful consideration.

- Get ruff to enforce no imports except at the start of the file, as Claude just loves
using them

- It's not clear why we're using kwargs.get in pipeline.match. This is an antipattern,
avoid unless there's a very strong reason for it.

- We shouldn't be creating MatchJob instances in grouping.py, as there's a lot
of details about metadata etc there. We should just be thinking about how to assign
them to groups in that module.

- In extend_ts we could save some sorting time by keeping track of the oldest parent
and then finding the first edge for that parent. We can then use this as the start_index
for sorting, which would likely improve things.

- It's not clear why we need site_alleles in extend_ts, and where this is being derived.
It would be better to pass the actual alleles through in the mutations.

- It would be better to deal with all the individual metadata at once, and add it
to the match job rather than derive it after the fact.

- Consider how pedigree data may be associated with individuals, and how this could be 
passed through to the tree sequence.

- Rename groups.json to match_jobs.json
