We need to implement threading based parallelism for the ancestor generation 
phase. Currently, where we iterate over the time, `focal_sites` elements in
`ancestor_descriptors`` we want to use a concurrent.futures ThreadPoolExecutor
to run the current loop body. So, ancestors are always generated in a background
worker thread. When we push the work dataclass onto the list of futures to be executed,
we also include the index of the ancestor. This is included in the returned 
dataclass.

For testing and debugging we should provide a dummy SynchronousExecutor like is 
used in bio2zarr core.py, which we use when num_threads<=0.

In the main thread then, we consume the futures as they are executed and 
call writer.add_ancestor. The ancestor index is also passed to add_ancestor.
Then, the additional complexity that we need to handle in the AncestorWriter is that 
we deterministically write the ancestors in index order. To do this we map the 
ancestor index to the corresponding chunk index and then we only flush chunks 
when all of the ancestors for that chunk have been added. This will add complexity
to the final chunk, but we can verify that when finalize is called that the 
we partially fill the chunk and there's no gaps.
