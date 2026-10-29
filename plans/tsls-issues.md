# tsls problems found while integrating it into tsinfer

Found while adding the optional tsls match engine (`[match] engine = "tsls"`,
`tsinfer/matching.py:TslsEngine`), against tsls commit `d557198`. None of
these are fixed in tsls. The workarounds are in tsinfer.

## 1. No per-site allele lists

tsls numbers each site's alleles itself: ancestral = 0, then derived
states in order of first appearance in the mutation table. The only
alternative is one global `alleles` list. tsinfer has fixed per-site codes
(the VCZ `AlleleMapper`, ancestral = 0), so it has to:

- copy tsls's private numbering (`matching.tsls_allele_codes`) and
  translate every haplotype into it and every matched haplotype back;
- break if tsls ever changes that numbering. Nothing would detect the
  change, apart from tsinfer's test comparing against tskit's
  `Variant.alleles`.

The global list can't stand in: it is capped at `TSLS_MAX_ALLELES` (127)
in total, which indels quickly exceed, and it sets every site's A_j to the
length of the list (see 2).

Suggested fix: let `tsls.TreeSequence` take per-site allele lists
(e.g. a list of lists, or a `(num_sites, max_alleles)` array), with A_j
the length of each site's list.

## 2. num_alleles can't be set, and unseen alleles are errors

A_j is the number of distinct states in the tree sequence at site j
(minimum 2). This causes two problems:

- The mismatch odds μ/(1 − (A−1)μ) differ from tsinfer's engine, where A
  is the number of alleles in the VCZ. This doesn't matter for the
  default μ = 1e-20, but it does for user-set mismatch values.
- A haplotype allele that is valid but not yet in the tree sequence at
  that site (e.g. the third allele of a multiallelic site, matched
  before any node carries it) raises `LibraryError: Haplotype allele
  must be missing (-1) or >= 0 and < num_alleles`, not a mismatch.
  tsinfer hides such alleles as missing. That is exact for Viterbi
  (every node mismatches, so the emission cancels in normalisation), but
  callers shouldn't need to know that.

Suggested fix: per-site allele lists (1) fix both. Failing that, accept an
optional `num_alleles` array.

## 3. No way to exclude nodes from copying

tsinfer never copies its ultimate root (node 0). tsls copies any node in
the tree. The README's workaround (delete node 0's edges) doesn't work
for tsinfer's root-only tree sequence (nodes 0 and 1, edge 0→1), which
is the reference for the first ancestor group: deleting the edge isolates
node 1, so nothing can be copied. Confirmed: on that tree sequence tsls
returns a path copying node 0.

tsinfer's workaround: give tsls the unchanged tree sequence and rewrite
parent 0 → 1 in the path (`matching.copy_node1_for_node0`). This is exact
only while node 0's sole child is node 1 and neither has mutations, which
`TslsEngine` checks.

Suggested fix: an option to exclude nodes from copying (e.g. an
`exclude_nodes` / copyable-node mask), or a documented convention for an
ultimate root that is never a state.

## 4. Allele numbering isn't exposed

`tsls.TreeSequence` only exposes `num_nodes` and `num_sites`. Callers
can't read the per-site allele numbering or A_j to check their
translation against tsls. Not needed if (1) is fixed.

## 5. `find_path` leaves `match` uninitialised outside [start, end)

`Hmm_find_path` allocates the matched-haplotype array with
`PyArray_SimpleNew` and the C code only writes sites in [start, end). The
rest is uninitialised memory, which reaches the caller through
`match_out[:] = matched`. tsinfer only reads the in-range part. Suggest
zero-filling (`PyArray_Zeros`) or documenting it.

## 6. Ties are broken differently from tsinfer

As the README says, paths match tsinfer's engine except for ties. So
exact regression comparisons between the engines only work on tie-free
inputs. tsinfer's tests compare paths exactly on hand-built tie-free
panels and otherwise check that genotypes are reproduced and that perfect
inference recovers the topology.

## 7. Packaging / building

- `setup.py` builds against `../git-submodules/tskit`, and
  `lwt_interface` is a symlink into it. So tsls only builds from a git
  checkout with submodules. There is no standalone sdist/wheel, and no
  PyPI release.
- At integration time the tsls working tree had uncommitted C edits that
  didn't compile (`traceback_delta_t` has no member `node` /
  `previous_flag` in `tsls_hmm_print_state`). Building from the working
  tree (an editable path dependency) breaks whenever tsls is mid-edit. So
  tsinfer pins a commit through a local git source in `pyproject.toml`:
  `tsls = { git = "file:///…/tsls", rev = "d557198", subdirectory = "tsls" }`.
  This absolute `file://` URL only works on this machine, and `uv lock`
  resolves every extra, so a checkout without it (e.g. CI) can't lock.
  Both go away once tsls is released.
- The out-of-range allele error is a generic `LibraryError`, not a
  `ValueError`/`MatchImpossible`, so callers can't tell input errors from
  internal failures.

## Observations (not problems)

- On a 400-sample, 5.4k-site msprime simulation (`tsinfer run`, 8
  threads), tsls gave the same mutation count as the built-in engine
  with 0.2% more edges, both reproduced the genotypes exactly, and the
  pipeline took 5.7 s vs 9.9 s. On 1000 samples and 15k sites it took
  24.5 s vs 31.2 s.
