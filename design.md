# tsinfer v1.0 design

## MVP

The MVP delivers end-to-end inference — equivalent to the current `tsinfer.infer` —
on the new architecture, running on a single machine. Backwards compatibility
with existing file formats and APIs is not a goal.

### In scope

- **Zarr v3** throughout. Drop the Zarr v2 dependency.
- **VCZ-native input.** Accept a samples VCZ store directly. Drop `SampleData`
  and the legacy tsinfer-specific sample format.
- **VCZ ancestor data.** `infer_ancestors` writes the ancestor VCZ format
  defined in this document. Drop `AncestorData` and the legacy tsinfer-specific
  ancestor format.
- **Simplified API.** Four functions: `infer_ancestors`, `match_ancestors`,
  `match_samples`, `post_process`. Drop the monolithic `infer` wrapper and the
  `generate_ancestors` alias.
- **Explicit `post_process`.** Remove the `post_process=True` default from
  `match_samples`. Post-processing is always a separate, explicit call.
- **Config file and CLI.** A TOML config file drives the pipeline. The CLI
  exposes `infer-ancestors`, `match-ancestors`, `match-samples`, `post-process`,
  and `run` (all steps in sequence).
- **C engine only.** The Python engine is removed from production code. It moves
  to the test suite, where it is used only for validating `infer_ancestors`
  output.
- **New test suite foundations.** In-memory VCZ test helpers (`make_sample_vcz`,
  `make_ancestor_vcz`, `ts_to_sample_vcz`) and the three-tier test structure
  described in the testing section.

### Out of scope (post-MVP)

- **Distributed inference.** The batch CLI commands (`match-ancestors-init`,
  `match-samples-partition`, etc.) and Snakemake integration are fully specified
  in this document but not implemented in the MVP. Single-machine threading via
  `--threads` is sufficient for the MVP.
- **External HMM engine.** Integration with `tsls` as a pluggable matcher is
  deferred. The existing C matching engine is retained unchanged.
- **Full Python engine removal.** The Python engine remains in the test suite
  for the MVP; complete removal is deferred.

## Matching engine

### Design

The matching engine is abstracted behind a `Matcher` class that takes a
**fixed** `tskit.TreeSequence` as its reference panel and matches one or more
haplotypes against it. This separates two concerns that are currently entangled
in the existing code:

- **Matching** — running the HMM to find the most likely copying path and
  mutations for each haplotype given a reference panel. This is stateless once
  the reference tree sequence is fixed.
- **Tree building** — extending the accumulated tree sequence with the match
  results from one group before the next group can be matched. This is where
  state between groups lives, and it is now expressed explicitly as a data
  transformation on a tree sequence rather than as internal engine state.

```python
@dataclass
class MatchResult:
    path_left:       np.ndarray  # (n_edges,) int32 — left breakpoints (site indices)
    path_right:      np.ndarray  # (n_edges,) int32 — right breakpoints (site indices)
    path_parent:     np.ndarray  # (n_edges,) int32 — parent node ids
    mutation_sites:  np.ndarray  # (n_mutations,) int32 — site indices
    mutation_state:  np.ndarray  # (n_mutations,) int8  — derived allele

class Matcher:
    def __init__(
        self,
        ts: tskit.TreeSequence,
        positions: np.ndarray,       # inference site positions
        recombination_rate,          # float or msprime.RateMap
        mismatch_ratio: float = 1.0,
        path_compression: bool = True,
        num_threads: int = 1,
    ): ...

    def match(
        self,
        haplotypes: np.ndarray,      # (n_haplotypes, n_sites) int8
        start_index: np.ndarray,     # (n_haplotypes,) first active site per haplotype
        end_index: np.ndarray,       # (n_haplotypes,) last active site per haplotype
    ) -> list[MatchResult]: ...
```

`start_index` and `end_index` are derived from the missing data pattern in
the ancestor VCZ (or for samples, cover all sites). They tell the HMM the
active range for each haplotype, replacing the role that the stored
`sample_start_position`/`sample_end_position` serve in the grouping step.

Results are plain data objects with no dependency on the engine internals, so
they can be serialised to disk for the distributed case.

The tree sequence is extended by a standalone `extend_ts` function after all
haplotypes in a group have been matched:

```python
def extend_ts(
    ts: tskit.TreeSequence,
    node_times: np.ndarray,          # time for each new node (one per haplotype)
    results: list[MatchResult],
    path_compression: bool = True,
) -> tskit.TreeSequence: ...
```

`extend_ts` adds the new nodes, inserts their edges (with optional path
compression), and records mutations. The returned tree sequence becomes the
reference panel for the next group.

### match_ancestors loop

```python
current_ts = make_root_ts(sequence_length, inference_positions)

for group_index, ancestor_ids in enumerate(groups):
    haplotypes, start_idx, end_idx = get_ancestor_haplotypes(ancestor_vcz, ancestor_ids)
    matcher = Matcher(current_ts, positions, recombination_rate, mismatch_ratio)
    results = matcher.match(haplotypes, start_idx, end_idx)
    times = ancestor_vcz["sample_time"].oindex[ancestor_ids]
    current_ts = extend_ts(current_ts, times, results, path_compression)

ancestors_ts = current_ts
```

### match_samples

Sample matching is a single call with no looping:

```python
haplotypes, start_idx, end_idx = get_sample_haplotypes(sample_vcz)
matcher = Matcher(ancestors_ts, positions, recombination_rate, mismatch_ratio)
results = matcher.match(haplotypes, start_idx, end_idx)
ts = extend_ts(ancestors_ts, sample_times, results, path_compression)
```

### MVP implementation

For the MVP, `Matcher` wraps the existing C engine. The existing code already
has a `restore_tree_sequence_builder` function that reconstructs the internal
`TreeSequenceBuilder` state from a `tskit.TreeSequence`, used in the current
batch workflow when resuming from an intermediate result. The MVP `Matcher`
uses exactly this mechanism: on construction it calls
`restore_tree_sequence_builder(ts)` to initialise the C engine, then delegates
`match()` to the existing `_tsinfer.AncestorMatcher`. The `MatchResult`
objects and `extend_ts` are thin wrappers over the existing `add_path` and
`add_mutations` calls on the `TreeSequenceBuilder`.

This means the MVP requires no changes to the C extension and no algorithmic
changes — only the Python-level structure changes.

### Future: pluggable engines

Any alternative implementation (e.g. `tsls`) that satisfies the `Matcher`
interface can be dropped in without changes to the pipeline. The config file
would gain an `engine` key:

```toml
[match]
engine = "tsls"   # or "tsinfer" (default)
```

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
| `sample_start_position` | `(n_ancestors,)` | `int32` | Genomic position of the first non-missing site for each ancestor |
| `sample_end_position` | `(n_ancestors,)` | `int32` | Genomic position of the last non-missing site for each ancestor |
| `sample_focal_positions` | `(n_ancestors, max_focal_positions)` | `int32` | Positions of focal sites for each ancestor; padded with -2 |

### Ancestor haplotypes

Each ancestor spans a contiguous range of inference sites. Outside that range,
the state of the ancestor is unknown; these flanking positions are encoded as
missing (`-1`) in `call_genotype`. Within the span, each site takes the value 0
(ancestral allele) or 1 (derived allele). The genomic extent of each ancestor is also recorded explicitly in
`sample_start_position` and `sample_end_position`. These values are derivable
from the missing data pattern in `call_genotype`, but `call_genotype` is chunked
along the sites axis, so deriving them would require reading the entire genotype
array. The grouping algorithm needs start and end for all ancestors
simultaneously before matching begins, so storing them as small 1D arrays avoids
what could otherwise be a full scan of gigabytes of genotype data.

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

## CLI

### Overview

All CLI commands take a config file as their primary argument. Parameters that
describe the dataset and inference (data paths, field names, rates, flags) live
in the config file. Parameters that are runtime or environment concerns
(parallelism, verbosity, whether to overwrite outputs) are CLI flags.

```
tsinfer <command> config.toml [flags]
```

### Pipeline commands

| Command | Reads | Writes | Notes |
|---|---|---|---|
| `infer-ancestors` | `data.samples` (VCZ) | `ancestors.path` (VCZ) | Builds ancestor haplotypes |
| `match-ancestors` | samples VCZ, ancestors VCZ | `output.ancestors_ts` | Single-machine matching |
| `match-samples` | samples VCZ, ancestors `.trees` | `output.samples_ts` | Single-machine matching |
| `post-process` | samples `.trees` | `output.samples_ts` (overwrite) | Simplify, erase flanks |
| `run` | samples VCZ | `output.samples_ts` | Runs all four steps in sequence |

Each command exits with code 0 on success and non-zero on failure. Errors are
written to stderr; no output is written to stdout except by `config show`.

### Runtime flags

These flags are accepted by all pipeline commands:

| Flag | Default | Description |
|---|---|---|
| `--threads N` | `1` | Worker threads for matching |
| `--force` | off | Overwrite existing output files |
| `--progress` | off | Show per-step progress bars |
| `-v / --verbose` | off | Increase log verbosity (`-vv` for debug) |

### The `config` utility command

```
tsinfer config show   config.toml    # print fully resolved config (with defaults filled)
tsinfer config check  config.toml    # validate config and check all input paths exist
```

`config check` is useful as the first step in a workflow script to catch
missing files or invalid parameter values before any compute begins.
`config show` prints the resolved config as TOML, which is useful for
recording exactly what parameters were used.

### Distributed pipeline commands

When `[distributed]` is present in the config, the following commands replace
`match-ancestors` and `match-samples`. See the distributed inference section
for full details.

```
tsinfer match-ancestors-init           config.toml
tsinfer match-ancestors-group          config.toml GROUP_START GROUP_END
tsinfer match-ancestors-partition      config.toml GROUP_INDEX PARTITION_INDEX
tsinfer match-ancestors-finalise-group config.toml GROUP_INDEX
tsinfer match-ancestors-finalise       config.toml

tsinfer match-samples-init             config.toml
tsinfer match-samples-partition        config.toml PARTITION_INDEX
tsinfer match-samples-finalise         config.toml
```

### Logging

Log messages are written to stderr using structured logging. Default level is
`WARNING`; `-v` raises it to `INFO`, `-vv` to `DEBUG`. At `INFO` level each
command logs the step being run, input/output paths, key parameter values, and
a wall-time and peak-memory summary on exit. At `DEBUG` level the matching
engine emits per-group statistics (group index, number of ancestors, epoch
regime used, partition count).

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
- **`match_ancestors`** — has a two-level structure. Ancestors are arranged
  into *groups* by a dependency analysis described below. Groups must be
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

### Ancestor grouping algorithm

Before matching begins, ancestors are arranged into an ordered sequence of
groups. The correct grouping must satisfy two constraints: (1) an ancestor that
overlaps genomically with an older ancestor must be placed in a later group, so
that when it is matched the older ancestor is already present in the tree
sequence; (2) same-time ancestors that overlap genomically must be placed in the
same group, so that they cannot match to each other.

The grouping is computed by `group_by_linesweep` using `sample_time`,
`sample_start_position`, and `sample_end_position` from the ancestor VCZ.
Two regimes are used depending on epoch size:

**Linesweep regime (earlier ancestors).** For ancestors in epochs up to the
first *large epoch* (defined below), a dependency graph is built by a
genomic-coordinate linesweep:

1. Overlapping same-time ancestors are temporarily merged into combined
   intervals.
2. A linesweep over genomic positions identifies all pairs of ancestors where
   one is older and their spans overlap, adding a directed dependency edge.
3. A topological sort of the dependency graph assigns each ancestor a group
   index. Ancestors with no inter-group dependencies may end up in the same
   group even if they come from different time epochs, maximising the
   parallel work available per group.
4. Same-time ancestors are un-merged, restoring the original ancestor
   identities within each group.

**Epoch regime (later ancestors).** For large epochs the linesweep dependency
graph becomes computationally intractable. The cutoff is the first epoch — in
the second half of the time distribution — whose size exceeds 500× the median
epoch size across all epochs. From that point onward each epoch forms its own
group, with ancestors grouped by time. These groups are typically large and
benefit most from partitioning.

The two regimes are combined into a single ordered group list. The first group
always contains only the virtual root ancestor and is handled separately.

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
