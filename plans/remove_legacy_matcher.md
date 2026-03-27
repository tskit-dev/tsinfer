## Context

The new `AncestorMatcher2`/`MatcherIndexes` implementation has been validated: 100% identical edges and mutations vs legacy, ~5% faster. The legacy `TreeSequenceBuilder` + `AncestorMatcher` code path is dead weight — remove it entirely.

## Files to modify

### 1. Delete C source file
- **`lib/tree_sequence_builder.c`** — entire file (1101 lines)
- **`lib/object_heap.c`** + **`lib/object_heap.h`** — only used by TSB

### 2. `lib/tsinfer.h` — remove legacy structs and declarations
- Remove `#include "object_heap.h"`
- Remove `tree_sequence_builder_t` struct (~lines 172-210)
- Remove `ancestor_matcher_t` struct (~lines 212-250)
- Remove all `tree_sequence_builder_*` function declarations (~lines 324-357)
- Remove all `ancestor_matcher_*` (non-`2`) function declarations (~lines 313-322)
- Keep: `matcher_indexes_t`, `ancestor_matcher2_t`, and their function declarations

### 3. `lib/ancestor_matcher.c` — remove legacy matcher functions
- Remove lines 29-1020 (all `ancestor_matcher_*` functions + helpers like `is_nonzero_root`)
- Keep: `matcher_indexes_*` functions (lines ~1022-1205) and `ancestor_matcher2_*` functions (lines ~1207-2110)
- `is_nonzero_root` at line 29 — check if `ancestor_matcher2` also uses it before removing

### 4. `_tsinfermodule.c` — remove legacy Python wrappers
- Remove `TreeSequenceBuilder` struct definition (lines 29-32)
- Remove `AncestorMatcher` struct definition (lines 34-38)
- Remove all `TreeSequenceBuilder_*` functions + type object (lines ~466-1317)
- Remove all `AncestorMatcher_*` (non-`2`) functions + type object (lines ~1326-1661)
- Remove module registration of both types (~lines 2123-2138)
- Keep: `MatcherIndexes`, `AncestorMatcher2`, `LightweightTableCollection`, `AncestorBuilder`

### 5. `setup.py` — remove from build
- Remove `"tree_sequence_builder.c"` from `tsi_source_files`
- Remove `"object_heap.c"` from `tsi_source_files`

### 6. `tsinfer/matching.py` — simplify to AncestorMatcher2 only
- Remove `_USE_MATCHER2` flag (line 41)
- Remove `_tsb_from_ts()` function (lines 75-157)
- Remove `_LegacyMatcherWrapper` class (lines 528-551)
- Simplify `Matcher.__init__()` — remove conditional, always use `MatcherIndexes`
- Simplify `Matcher._match_one()` — remove conditional, always use `AncestorMatcher2`
- Can also remove the `allele_mapper` parameter from `Matcher.__init__` if it was only used by `_tsb_from_ts`

### 7. Tests
- **`tests/test_python_c.py`** — remove `TestAncestorMatcher` class (lines 62-130) and `TestTreeSequenceBuilder` class (lines 133-240). Also remove `TestMaxMemoryUsage` (lines 35-60) if it only tests TSB.
- **`tests/algorithm.py`** — remove `TreeSequenceBuilder` class (lines 334-659) and the `sortedcontainers` import. Keep `AncestorMatcher` (the Python reference implementation used by test_lshmm.py).
- **`tests/test_matching.py`**, **`tests/test_matcher_fixtures.py`** — no changes needed (they're path-agnostic via `Matcher`)

### 8. `lib/tests/tests.c` — remove legacy C tests
- This file (1694 lines) exclusively tests TSB + legacy matcher. Delete entirely or gut the legacy parts.

## Dependency chain (what's safe to remove)

```
tree_sequence_builder.c  →  only used by ancestor_matcher.c (legacy part) and _tsinfermodule.c
object_heap.c/h          →  only used by tree_sequence_builder.c
avl.c                    →  used by both tree_sequence_builder.c AND ancestor_builder.c — KEEP
tsi_blkalloc.c           →  used by ancestor_matcher2 — KEEP
```

## Order of operations

1. Remove Python-level legacy code (`matching.py` simplification)
2. Remove C extension wrappers (`_tsinfermodule.c`)
3. Remove C implementation (`tsinfer.h`, `ancestor_matcher.c` legacy part, `tree_sequence_builder.c`, `object_heap.*`)
4. Update `setup.py`
5. Remove legacy tests
6. Rebuild and run tests

## Verification

```bash
uv sync --reinstall-package tsinfer
uv run pytest tests/test_matching.py tests/test_matcher_fixtures.py tests/test_lshmm.py tests/test_python_c.py -v
uv run python leak_test.py  # confirm no regression
```
