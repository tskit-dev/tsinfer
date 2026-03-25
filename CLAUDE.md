# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Setup

```bash
uv sync                # Install all dependencies (including dev groups)
```

The package includes a C extension (`_tsinfer`) built from `lib/` sources via setuptools.
`uv sync` compiles it automatically. If you modify C code, re-run `uv sync` to rebuild.

## Common Commands

```bash
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/test_matching.py  # Run a single test file
uv run pytest tests/test_matching.py::TestFoo::test_bar -v  # Run a single test
uv run pytest --skip-slow            # Skip slow tests

uv run ruff check --fix              # Lint Python code (auto-fix)
uv run ruff format                   # Format Python code
```

Line length is 89 characters (configured in pyproject.toml for both ruff and clang-format).

## Architecture

**tsinfer** infers tree sequences from genetic variation data stored in VCZ (Variant Call Zarr) format.

### Pipeline (three-stage inference)

The public API is in `tsinfer/__init__.py`, exposing three main functions from `pipeline.py`:

1. **`infer_ancestors`** (`ancestors.py`) — Generates ancestral haplotypes from sample genotypes. Two-pass chunk-aware approach: first computes per-site statistics, then builds ancestors using the C `_tsinfer.AncestorBuilder`. Output is an ancestor VCZ store.

2. **`match`** (`pipeline.py` → `matching.py`) — Matches ancestors against each other, then matches samples against the ancestor tree sequence. Uses the C `_tsinfer.AncestorMatcher` (Li & Stephens HMM) and `_tsinfer.TreeSequenceBuilder`. Ancestors are grouped for parallel matching via `grouping.py`.

3. **`post_process`** (`pipeline.py`) — Cleans up the raw inferred tree sequence (edge extension, parsimony-based refinement).

`run()` in `pipeline.py` chains all three stages.

### Key modules

- **`config.py`** — Dataclass-based configuration (`Config`, `AncestorsConfig`, `MatchConfig`, `PostProcessConfig`, `Source`)
- **`vcz.py`** — VCZ/Zarr I/O layer; chunk-aware genotype loading (`get_genotypes_for_sites`)
- **`grouping.py`** — Ancestor grouping and match job computation for parallel processing
- **`matching.py`** — Core matching logic; `_ts_from_tsb` converts `TreeSequenceBuilder` to tskit tree sequence
- **`ancestors.py`** — Ancestor generation with `InferenceSites` and `AncestorWriter`
- **`tests/algorithm.py`** — Pure Python reference implementations of `AncestorBuilder`, `AncestorMatcher`, `TreeSequenceBuilder` (used for testing correctness against C)

### C extension (`_tsinfer`)

Source in `lib/`. Three main classes exposed to Python:
- `AncestorBuilder` — builds inferred ancestors from genotype data
- `AncestorMatcher` — Li & Stephens HMM matching algorithm
- `TreeSequenceBuilder` — constructs tree sequences incrementally

Vendored dependencies in `lib/subprojects/`: tskit C library and kastore.

### Data flow

Sample VCZ → `infer_ancestors` → Ancestor VCZ → `match` → raw `tskit.TreeSequence` → `post_process` → final tree sequence

## Git usage

- Don't include Co-authored-By lines in git commits.

## Code Style

- Do not be overly defensive - defend only against circumstances that can
  occur within the current codebase.
- Prefer dataclasses over tuples when returning multiple values.
- Use explicit `None` comparisons: `if x is not None` not `if x`.
- Zarr v3 is now used (dependency: `zarr>=3`).
- Import all modules at the top of the file unless compelling reason otherwise.
- Prefer importing a module and using module.function instead of
  using ``from module import function``. This applies to intra-package
  imports too: use ``from . import config`` then ``config.X``, not
  ``from .config import X``. Exceptions: ``from typing import ...`` is
  acceptable; ``from .X import Y`` is acceptable in ``__init__.py`` for
  defining the public API. Use ``import concurrent.futures as cf``.
- Use idiomatic pathlib.Path operations instead of os.path operations.
- When a parameter has a computed default derived from another parameter,
  compute it once at the point of use (the leaf function), not at every
  layer in the call chain. Pass `None` through intermediate layers.
- Use PEP 604 union syntax: `int | None`, not `Optional[int]`.
- One `logger = logging.getLogger(__name__)` per module at top level.

## Tool use

- Use `uv run` for all Python tooling (never bare `python -m`)

## Testing

- Organise tests in classes, not flat functions. Use pytest fixtures for setup.
- Test helpers are in `tests/helpers.py` (e.g., `make_sample_vcz`, `make_ancestor_vcz`)
- `tests/algorithm.py` contains Python reference implementations used to verify C code
- `msprime` is used to simulate test data
