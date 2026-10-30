# tsls issues found while integrating it into tsinfer

These were found while adding tsls as an optional match engine (against
tsls commit `d557198`). tsls is now tsinfer's only engine. This records how
each issue was resolved; `tests/test_tsls_contract.py` pins the behaviour
tsinfer relies on.

1. **No per-site allele lists.** Fixed in tsls: `tsls.TreeSequence` takes a
   list of alleles for each site, including a 2D array padded with `""`.
   tsinfer passes `AlleleMapper.forward_map()` directly, and the
   translation (`tsls_allele_codes`) is gone.
2. **num_alleles can't be set, and unseen alleles are errors.** Fixed by
   (1): the number of alleles at a site is the length of its list (the VCZ
   count), and an allele that isn't in the tree sequence at a site is a
   mismatch. tsinfer no longer hides such alleles as missing data.
3. **No way to exclude nodes from copying.** Not changed. tsinfer keeps its
   ultimate root, node 0, and copies node 1 wherever tsls copies node 0
   (`matching.check_ultimate_root`, `matching.copy_node1_for_node0`). The
   contract tests check that this is exact. Removing the ultimate root
   needs another way to keep the virtual root copyable where it has no
   children.
4. **Allele numbering isn't exposed.** Fixed: `TreeSequence.num_alleles`.
   The numbering is no longer needed, given (1).
5. **`find_path` leaves `match` uninitialised outside `[start, end)`.**
   Fixed: it is -1 (missing) there.
6. **Ties are broken differently from tsinfer.** Unchanged, and documented
   in tsls: which of equally likely paths is returned isn't specified. The
   tests pin paths only where they are unique.
7. **Packaging.** tsls now has the standard tskit-dev layout and installs
   from GitHub (with its tskit submodule). It isn't on PyPI yet, so tsinfer
   depends on it from GitHub. Bad inputs raise `ValueError`, and
   `MatchImpossible` is raised when no path is possible.

## Observations (not problems)

- On a 400-sample, 5.4k-site msprime simulation (`tsinfer run`, 8
  threads), tsls gave the same mutation count as the previous built-in
  engine with 0.2% more edges, both reproduced the genotypes exactly, and
  the pipeline took 5.7 s vs 9.9 s. On 1000 samples and 15k sites it took
  24.5 s vs 31.2 s.
