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

## Distributed inference

This section is not required for the MVP but must be fully supported in v1.0.

### Parallelism characteristics of the pipeline

- **`infer_ancestors`** — single job.
- **`match_ancestors`** — has a two-level structure. Ancestors are grouped by
  time into *groups* (collections of one or more epochs). Groups must be
  processed strictly in order, since each group's matching depends on the tree
  sequence produced by all previous groups. Within a large group, ancestors are
  subdivided into *partitions* that can be matched independently in parallel.
  Small groups are never partitioned. The number of partitions in a group is
  determined by `min_work_per_job`: ancestors are greedily bin-packed into
  partitions so that each partition has approximately `min_work_per_job`
  genotypes of work.
- **`match_samples`** — embarrassingly parallel. Samples are partitioned into
  independent batches, each matched against the same fixed `ancestors_ts`.
  Partition count is again controlled by `min_work_per_job`.
- **`post_process`** — single job.

### Configuration

The `[distributed]` section of the config file controls batch granularity:

```toml
[distributed]
work_dir = "inference_work"    # directory for intermediate state
min_work_per_job = 1_000_000  # genotypes per job, controls partition count
max_num_partitions = 1000      # upper bound on partitions per group
```

All intermediate files are written under `work_dir`. A `metadata.json` file
written by each init command records the full job structure (group count,
partition counts per group, parameters) so that subsequent commands and
workflow managers can determine what jobs need to be run.

### CLI — match_ancestors

```
# 1. Initialise: creates work_dir and writes metadata.json
tsinfer match-ancestors-init config.toml

# 2a. For small groups (no partitioning): run one or more groups in a single job
tsinfer match-ancestors-group config.toml GROUP_START GROUP_END

# 2b. For large groups: run each partition independently (parallel)
tsinfer match-ancestors-partition config.toml GROUP_INDEX PARTITION_INDEX

# 3. Finalise each large group after all its partitions complete
tsinfer match-ancestors-finalise-group config.toml GROUP_INDEX

# 4. Finalise the full match_ancestors step once all groups are done
tsinfer match-ancestors-finalise config.toml
```

Steps 2a and 2b are not mutually exclusive. Small groups that require no
partitioning are run with `match-ancestors-group`; large groups that are
partitioned use `match-ancestors-partition` followed by
`match-ancestors-finalise-group`. The workflow manager determines which
command applies for each group by reading `metadata.json`.

### CLI — match_samples

```
# 1. Initialise: creates work_dir, computes partitions, writes metadata.json
tsinfer match-samples-init config.toml

# 2. Match each partition independently (fully parallel)
tsinfer match-samples-partition config.toml PARTITION_INDEX

# 3. Merge all partitions into the final tree sequence
tsinfer match-samples-finalise config.toml
```

### Snakemake integration

The init commands are natural Snakemake checkpoints: they run first, write
`metadata.json`, and the downstream rules read it to determine how many jobs
to submit. A sketch of the Snakemake rules for `match_samples`:

```python
checkpoint match_samples_init:
    input:  "ancestors.trees"
    output: "inference_work/samples/metadata.json"
    shell:  "tsinfer match-samples-init config.toml"

def sample_partitions(wildcards):
    meta = json.load(checkpoints.match_samples_init.get().output[0])
    n = meta["num_partitions"]
    return expand("inference_work/samples/partition_{i}.pkl", i=range(n))

rule match_samples_partition:
    input:  "inference_work/samples/metadata.json"
    output: "inference_work/samples/partition_{i}.pkl"
    shell:  "tsinfer match-samples-partition config.toml {wildcards.i}"

rule match_samples_finalise:
    input:  sample_partitions
    output: "final.trees"
    shell:  "tsinfer match-samples-finalise config.toml"
```

The `match_ancestors` Snakemake rules follow the same checkpoint pattern but
also encode the sequential dependency between groups.

### Python API

The batch functions are available directly for programmatic use:

```python
cfg = tsinfer.Config.from_toml("inference.toml")

# match_ancestors
meta = tsinfer.match_ancestors_init(cfg)
for group_index, group in enumerate(meta["ancestor_grouping"]):
    if group["partitions"] is None:
        tsinfer.match_ancestors_group(cfg, group_index, group_index + 1)
    else:
        for partition_index in range(len(group["partitions"])):
            tsinfer.match_ancestors_partition(cfg, group_index, partition_index)
        tsinfer.match_ancestors_finalise_group(cfg, group_index)
ancestors_ts = tsinfer.match_ancestors_finalise(cfg)

# match_samples
meta = tsinfer.match_samples_init(cfg)
for partition_index in range(meta["num_partitions"]):
    tsinfer.match_samples_partition(cfg, partition_index)
ts = tsinfer.match_samples_finalise(cfg)
```
