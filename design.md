# tsinfer v1.0 design

## Ancestor data format

Ancestors are stored as a VCF Zarr (VCZ) store, making the format natively
interoperable with the wider VCZ ecosystem. Each ancestor is represented as a
haploid sample (ploidy = 1). The store contains only the inference sites — the
subset of sites from the input samples VCZ that pass the site mask applied
during `infer_ancestors`.

### Arrays

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `call_genotype` | `(n_sites, n_ancestors, 1)` | `int8` | Genotype calls: 0 = ancestral, 1 = derived, -1 = missing |
| `variant_position` | `(n_sites,)` | `int32` | Genomic position of each inference site |
| `variant_alleles` | `(n_sites, n_alleles)` | `str` | ancestral and derived alleles carried through from the samples VCZ |
| `sample_id` | `(n_ancestors,)` | `str` | Ancestor identifiers, e.g. `ancestor_0`, `ancestor_1` |
| `sample_time` | `(n_ancestors,)` | `float64` | Ancestor time, used to define epoch ordering during matching |
| `sample_focal_positions` | `(n_ancestors, max_focal_positions)` | `int32` | Positions of focal sites for each ancestor; padded with -2 |

### Ancestor haplotypes

Each ancestor spans a contiguous range of inference sites. Outside that range,
the state of the ancestor is unknown; these flanking positions are encoded as
missing (`-1`) in `call_genotype`. Within the span, each site takes the value 0
(ancestral allele) or 1 (derived allele). There is no separate start/end
coordinate array — the extent of each ancestor is fully determined by the
positions of its non-missing genotype calls.

### Focal sites

Focal sites are the inference sites at which a particular ancestor was
generated. An ancestor is defined by carrying the derived allele at all of its
focal sites. Most ancestors have a single focal site; the `max_focal_positions`
dimension accommodates the minority with more. Unused slots are padded with
`-2` following standard VCZ conventions. Focal site values are genomic
positions, consistent with `variant_position`.

### Relationship to the samples VCZ

The ancestor VCZ is a derived store produced by `infer_ancestors` from a
samples VCZ. Its `variant_position` array is a strict subset of the positions
in the samples VCZ, containing only the inference sites. The `variant_alleles`
values are copied through unchanged from the samples VCZ so that the ancestor
store is interpretable as a standalone file.

## API and configuration

### Design philosophy

The primary interface to tsinfer is the command-line tool driven by a TOML
configuration file. The configuration file fully describes a dataset and the
parameters needed to run inference, so that results are reproducible and the
inference pipeline can be re-run or resumed from any step without re-specifying
arguments. The Python API is a secondary interface, intended mainly to support
the CLI implementation and programmatic use in scripts and notebooks.

### Configuration file

A single TOML file describes the entire pipeline for a given dataset. An
example:

```toml
[data]
samples = "path/to/samples.vcz"
ancestral_state = "variant_ancestral_allele"  # field name in the samples VCZ
site_mask = "variant_filter"                  # optional boolean field in samples VCZ
sample_mask = "sample_mask"                   # optional boolean field in samples VCZ
sites_time = "variant_time"                   # optional field overriding inferred times

[ancestors]
path = "ancestors.vcz"

[match]
recombination_rate = 1e-8
mismatch_ratio = 1.0
path_compression = true

[output]
ancestors_ts = "ancestors.trees"
samples_ts = "final.trees"

[post_process]
split_ultimate = true
erase_flanks = true
```

All paths are resolved relative to the config file's location. The
`[data]` section is shared across all pipeline steps. Step-specific sections
(`[ancestors]`, `[match]`, `[output]`, `[post_process]`) supply the parameters
for each stage.

### CLI

The CLI exposes one command per pipeline step, plus a convenience command that
runs all steps in sequence:

```
tsinfer infer-ancestors  config.toml
tsinfer match-ancestors  config.toml
tsinfer match-samples    config.toml
tsinfer post-process     config.toml

tsinfer run              config.toml   # runs all four steps
```

Runtime options that are not part of the reproducible configuration (parallelism,
verbosity, whether to overwrite existing outputs) are passed as CLI flags rather
than in the config file:

```
tsinfer run config.toml --threads 8 --force --verbose
```

### Python API

Each pipeline step has a corresponding Python function. Functions accept a
`Config` object as their primary argument, which can be loaded from a TOML
file or constructed programmatically:

```python
cfg = tsinfer.Config.from_toml("inference.toml")

tsinfer.infer_ancestors(cfg)
ancestors_ts = tsinfer.match_ancestors(cfg)
ts = tsinfer.match_samples(cfg, ancestors_ts)
final_ts = tsinfer.post_process(ts)
```

Individual parameters can be overridden by passing keyword arguments, which
take precedence over values in the config:

```python
cfg = tsinfer.Config.from_toml("inference.toml")
ancestors_ts = tsinfer.match_ancestors(cfg, recombination_rate=2e-8)
```

The `Config` class can also be constructed directly without a TOML file for
fully programmatic use:

```python
cfg = tsinfer.Config(
    samples="samples.vcz",
    ancestral_state="variant_ancestral_allele",
    ancestors="ancestors.vcz",
    recombination_rate=1e-8,
    ...
)
```

## Testing

### Synthetic data helpers

Since VCZ is a zarr store, test data can be constructed directly in memory
using `zarr.store.MemoryStore` — no files, no VCF parsing, no simulation
required for simple cases. Two helper functions form the foundation of the test
suite:

```python
def make_sample_vcz(
    genotypes,        # (n_sites, n_samples, ploidy) int8
    positions,        # (n_sites,) int
    alleles,          # (n_sites, n_alleles) str
    ancestral_state,  # (n_sites,) str
    sequence_length,
    **kwargs,         # site_mask, sample_mask, sites_time, ...
) -> zarr.Group:      # backed by MemoryStore

def make_ancestor_vcz(
    genotypes,          # (n_sites, n_ancestors, 1) int8; -1 for missing flanks
    positions,          # (n_sites,) int
    alleles,            # (n_sites, n_alleles) str
    times,              # (n_ancestors,) float
    focal_positions,    # (n_ancestors, max_focal_positions) int; -2 padded
) -> zarr.Group:        # backed by MemoryStore
```

All test data construction goes through these helpers. Test functions themselves
contain only the call under test and assertions, with no setup logic.

### Three tiers of test data

**Tier 1: Hand-constructed minimal cases.** Small numpy arrays built directly
in the test, where the correct output is known by inspection. Used for corner
cases including:

- Single site, single ancestor, single sample
- Ancestor with missing left flank only, right flank only, or both
- `max_focal_positions > 1` (ancestor with multiple focal sites)
- Site at position 0 or at `sequence_length`
- All-missing site in the samples
- Site mask and sample mask excluding various subsets
- Diploid vs haploid samples
- Multiple ancestors at the same time (single-epoch matching)
- All samples identical (no variation)

**Tier 2: Simulation-based cases.** For larger, statistically realistic inputs,
msprime is used to simulate a tree sequence which is then converted to VCZ via
a single utility:

```python
def ts_to_sample_vcz(ts, ancestral_state="REF") -> zarr.Group:
```

These are used for end-to-end pipeline tests where the goal is round-trip
fidelity. Cases are parametrised over sample count, ploidy, sequence length,
and mutation/recombination rate.

**Tier 3: Config and CLI tests.** The `Config` class can be constructed
programmatically using in-memory zarr stores, so CLI tests do not need to write
TOML files or touch the filesystem. TOML-specific tests (path resolution,
missing fields, invalid values) use a small `tmp_path` fixture.

### Parametric coverage

Rather than writing individual tests for each combination of inputs, a pytest
fixture provides a set of named canonical cases that are reused across multiple
test functions:

```python
@pytest.fixture(params=["single_site", "two_epochs", "diploid", "with_missing"])
def sample_vcz(request):
    return CANONICAL_CASES[request.param]
```

Named corner cases are written as individual tests so their intent is clear.
