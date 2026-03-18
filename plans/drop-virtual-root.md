# Dropping virtual root

We should not be generating the virtual root when considering ancestors.
Only ancestors that are actually stored should be taken into account
when generating groups. The virtual root idea should be taken care of
when we generate the initial tree sequence for matching in make_root_ts.
This means we add a two nodes, 0 and 1 with an edge between them.


# Updates to the match function in pipeline.py

We should not be using the ``_tsinfer.TreeSequencBuilder`` here, we should
just be using a standard tskit tree sequence. We start with the root_ts
produced by make_root_ts and append to this as we go.

I don't understand why we are perturbing the match_times. Remove this
unless there's a compelling reason for keeping it.

The match function is also not using the code in matching.py. No low-level
about matching should occur in pipeline.py.

Aim for maximium simplicity at this stage so that we can reason more easily
about parallelism later.
