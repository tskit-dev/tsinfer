- Use vcz_validator to validate the ancestors zarr.

- Improve tests for match resume, we need to validate that we correctly resume when the
match is stopped at different times. Add a parameter "group_stop" which is 1+ the index
of the last group to be processed. This will help with testing.

- Add the number of new edges along with final ts.num_trees to the extend_ts INFO output.

- It's not clear that HaplotypeReader is behaving correctly on chunk reading with samples_selection.
  This needs careful consideration.
