# tsinfer v1.0 design

## Development phasing

- **Phase 1** is structural: getting data sources working in the correct
  generalised way. The exact names of parameters are not critical; the goal is
  to establish the new shape of the codebase. Multi-source support is not in
  scope, but the configuration and code structure must make it a natural later
  extension. The VCZ ancestor format, unified match loop, config-file-first API,
  node/individual metadata, and new test foundations are all Phase 1.
- **Phase 2** is refinement: multi-source ancestor generation and matching,
  reparametrisation of the HMM, external HMM engine, and detailed consideration
  of naming and finer API details.

## Phase 1 MVP

The MVP delivers end-to-end inference — equivalent to the current `tsinfer.infer`
— on the new architecture, running on a single machine. Backwards compatibility
with existing file formats and APIs is not a goal.

### In scope

- **Zarr v3** throughout. Drop the Zarr v2 dependency.
- **VCZ-native input.** Accept a samples VCZ store directly. Drop `SampleData`
  and the legacy tsinfer-specific sample format.
- **VCZ ancestor data.** `infer_ancestors` writes the ancestor VCZ format
  defined in this document. Drop `AncestorData` and the legacy ancestor format.
- **Simplified API.** Three functions: `infer_ancestors`, `match`, `post_process`.
  The distinction between `match_ancestors` and `match_samples` is dropped —
  both are instances of the unified match loop.
- **Explicit `post_process`.** Post-processing is always a separate, explicit
  call; there is no implicit post-processing step.
- **Config file and CLI.** A TOML config file drives the pipeline. The CLI
  exposes `infer-ancestors`, `match`, `post-process`, and `run` (all steps in
  sequence).
- **C engine only.** The Python engine moves to the test suite, where it is
  used only for validating `infer_ancestors` output.
- **Node and individual metadata.** Each node carries JSON provenance metadata
  (`source`, `sample_id`, `ploidy_index`). Individual metadata is populated from
  user-declared VCZ field mappings.
- **New test suite foundations.** In-memory VCZ helpers (`make_sample_vcz`,
  `make_ancestor_vcz`, `ts_to_sample_vcz`) and the three-tier test structure
  described in the testing section.

### Out of scope (Phase 1)

- **Precise naming**. Phase 1 is about the structure, and should not be bogged
  down by trying to get the name of everything right in the first pass.
- **Multi-source inference.** Single source for ancestor generation and
  matching. The configuration structure uses lists throughout so multi-source
  is a pure Phase 2 extension.
- **Distributed inference.** The batch CLI commands and Snakemake integration
  are fully specified in this document but not implemented. Single-machine
  threading via `--threads` is sufficient for Phase 1.
- **HMM reparametrisation.** Current HMM parameters are kept as-is.
- **External HMM engine.** The existing C matching engine is retained unchanged.
- **Interfacing with HMM** to run matches for single samples programatically
  is not in scope
- **Benchmarking against simulations** is not initially in scope, but may be
  handled explicitly as a convenience interface in later phases.

## Data sources and configuration

### Design philosophy

The primary interface to tsinfer is the command-line tool driven by a TOML
configuration file. The configuration file fully describes a dataset and the
parameters needed to run inference, so that results are reproducible and the
pipeline can be re-run or resumed from any step without re-specifying arguments.
The Python API is a secondary interface, mainly supporting the CLI
implementation. The TOML config interface should support all necessary use-cases,
and out-of-band scripting with Python is regarded as an anti-pattern.

### Data sources

The fundamental input unit is a **source** — a named, configured view over a
VCZ store. A source specifies the store path (local or remote) and how to
resolve each metadata array. Metadata can come from a field within the store
itself, from a field in a separate annotation VCZ (joined by `variant_position`
or `sample_id`), or as a scalar constant. This makes fully remote, read-only
stores first-class inputs: none of the metadata needs to live in the remote
store itself.

tsinfer inference always operates on a single contig. After applying all site
masks, the remaining inference positions across all sources in a step must
belong to a single contig; this is checked at runtime.

```toml
[[source]]
name = "ukb"
path = "s3://bucket/ukb-chr20.vcz"          # remote, read-only
ancestral_state = {path = "annotations.vcz", field = "variant_ancestral_allele"}
site_mask       = {path = "annotations.vcz", field = "variant_filter"}
# sample_mask and sample_time can also be specified here
```

For the common case where all metadata lives in the store, the field name is a
string:

```toml
[[source]]
name            = "local"
path            = "samples.vcz"
ancestral_state = "variant_ancestral_allele"
site_mask       = "variant_filter"
```

Metadata resolution rules:

- **String** — field name within the source store itself
- **`{path, field}`** — field from a separate VCZ, joined by `variant_position`
  (site arrays) or `sample_id` (sample arrays). Missing entries get fill values:
  no ancestral state → site excluded; no mask → site included; no time → time=0
- **Scalar** — constant applied to all sites or samples

### Use cases

**Standard inference (modern samples only).** Sources have no `sample_time`;
all haplotypes match at time=0.

**Ancient samples.** Set `sample_time` on the source to a field in the VCZ
holding per-sample ages (in generations). Ancient haplotypes
are interleaved with ancestral haplotypes inferred from modern data in the
match step by the time-ordering
grouping: they copy from whatever is already in the reference tree sequence at
their time, then become part of the panel for younger samples. They produce
individuals just like modern samples.

```toml
[[source]]
name        = "ancient"
path        = "ancient_dna.vcz"
sample_time = "sample_age_generations"
ancestral_state = "variant_ancestral_allele"
```

**Known pedigree.** Parents or grandparents with known relationships can be
included as sources with `sample_time = 1` or `sample_time = 2` (in
generations). The pipeline handles them identically to ancient samples —
they copy from the existing tree sequence and then become part of the panel
for the next, younger group.

```toml
[[source]]
name        = "parents"
path        = "parents.vcz"
sample_time = 1          # scalar: all samples in this source at time=1
ancestral_state = "variant_ancestral_allele"
```

### Configuration file

A single TOML file describes the entire pipeline. The standard case — a single
cohort, with ancestor VCZ written to disk, then samples matched against it:

```toml
[[source]]
name            = "cohort"
path            = "samples.vcz"
ancestral_state = "variant_ancestral_allele"
site_mask       = "variant_filter"

[ancestors]
path           = "ancestors.vcz"
sources        = ["cohort"]
max_gap_length = 500_000

[[match]]
name               = "ancestors"
sources            = ["ancestors"]
reference_ts       = null
output             = "ancestors.trees"
create_individuals = false
recombination_rate = 1e-8
mismatch_ratio     = 1.0

[[match]]
name               = "samples"
sources            = ["cohort"]
reference_ts       = "ancestors.trees"
output             = "final.trees"
recombination_rate = 1e-8
mismatch_ratio     = 1.0

[individual_metadata]
fields     = {sample_id = "sample_id", sex = "sample_sex"}
population = "sample_population"

[post_process]
split_ultimate = true
erase_flanks   = true
```

**Adding ancient samples.** To include ancient DNA alongside modern samples,
add the ancient source and list it in the samples match step. The grouping
algorithm handles the ordering automatically:

```toml
[[source]]
name        = "ancient"
path        = "ancient.vcz"
sample_time = "sample_age_generations"
ancestral_state = "variant_ancestral_allele"

[[match]]
name               = "samples"
sources            = ["cohort", "ancient"]   # ancient haplotypes ordered before modern
reference_ts       = "ancestors.trees"
output             = "final.trees"
recombination_rate = 1e-8
mismatch_ratio     = 1.0
```

No other changes are needed. The ancient samples are matched before time=0
samples, produce individuals, and appear in the output tree sequence with their
correct times.

**Multi-source (Phase 2).** When `[ancestors]` lists multiple sources,
`infer_ancestors` operates on the union of their inference sites. Genotypes
at sites absent from a source are treated as missing for that source's samples.
A site is included in inference only if it passes the mask in at least one
source and has a consistent ancestral state across all sources that have it.
When a `[[match]]` step lists multiple sources, the combined genotype data is
presented to the matching engine as a single virtual dataset; sources need not
have the same samples.

All paths are resolved relative to the config file's location. A `reference_ts`
of `null` or omitted means start from a trivial root tree sequence.

### Python API

```python
cfg = tsinfer.Config.from_toml("inference.toml")

tsinfer.infer_ancestors(cfg)
tsinfer.match(cfg, "ancestors")
tsinfer.match(cfg, "samples")
tsinfer.post_process(cfg)
```

Individual parameters can be overridden by keyword argument:

```python
tsinfer.match(cfg, "ancestors", recombination_rate=2e-8)
```

The `Config` class can also be constructed directly:

```python
cohort = tsinfer.Source("samples.vcz",
                        ancestral_state="variant_ancestral_allele",
                        site_mask="variant_filter")
ancient = tsinfer.Source("ancient.vcz",
                         sample_time="sample_age_generations",
                         ancestral_state="variant_ancestral_allele")

cfg = tsinfer.Config(
    sources={"cohort": cohort, "ancient": ancient},
    ancestors=tsinfer.AncestorsConfig(
        path="ancestors.vcz", sources=["cohort"], max_gap_length=500_000),
    match=[
        tsinfer.MatchConfig(name="ancestors", sources=["ancestors"],
                            reference_ts=None, output="ancestors.trees",
                            create_individuals=False,
                            recombination_rate=1e-8),
        tsinfer.MatchConfig(name="samples", sources=["cohort", "ancient"],
                            reference_ts="ancestors.trees", output="final.trees",
                            recombination_rate=1e-8),
    ],
    individual_metadata=tsinfer.IndividualMetadataConfig(
        fields={"sample_id": "sample_id", "sex": "sample_sex"},
        population="sample_population",
    ),
)
```

## Ancestor data format

Ancestors are stored as a VCF Zarr (VCZ) store. Each ancestor is represented
as a haploid sample (ploidy = 1). The store contains only the inference sites
— the subset of sites from the input samples VCZ that pass the site mask.

### Arrays

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `call_genotype` | `(n_sites, n_ancestors, 1)` | `int8` | 0 = ancestral, 1 = derived, -1 = missing |
| `variant_position` | `(n_sites,)` | `int32` | Genomic position of each inference site |
| `variant_alleles` | `(n_sites, n_alleles)` | `str` | Ancestral and derived alleles |
| `sample_id` | `(n_ancestors,)` | `str` | e.g. `ancestor_0`, `ancestor_1` |
| `sample_time` | `(n_ancestors,)` | `float64` | Ancestor time, drives epoch ordering |
| `sample_start_position` | `(n_ancestors,)` | `int32` | Genomic position of first non-missing site |
| `sample_end_position` | `(n_ancestors,)` | `int32` | Genomic position of last non-missing site |
| `sample_focal_positions` | `(n_ancestors, max_focal_positions)` | `int32` | Focal site positions; padded with -2 |
| `sequence_intervals` | `(n_intervals, 2)` | `int32` | `[start, end)` pairs for regions containing inference sites |

### Ancestor haplotypes

Each ancestor spans a contiguous range of inference sites. Outside that range
the state is unknown; flanking positions are encoded as missing (`-1`) in
`call_genotype`. Within the span each site is 0 (ancestral) or 1 (derived).

The genomic extent is recorded explicitly in `sample_start_position` and
`sample_end_position`. These values are derivable from the missing data pattern
in `call_genotype`, but `call_genotype` is chunked along the sites axis, so
deriving them would require reading the entire genotype array. The grouping
algorithm needs start and end for all ancestors simultaneously before matching
begins, so storing them as small 1D arrays avoids a full scan of potentially
gigabytes of genotype data.

### Focal sites

Focal sites are the inference sites at which a particular ancestor was
generated. An ancestor carries the derived allele at all its focal sites. Most
ancestors have a single focal site; the `max_focal_positions` dimension
accommodates the minority with more. Unused slots are padded with `-2` following
standard VCZ conventions. Values are genomic positions, consistent with
`variant_position`.

### Gap intervals

Long genomic regions with no inference sites are recorded in
`sequence_intervals`, which lists the `[start, end)` coordinate pairs for
regions *containing* inference sites. Its complement within `[0, sequence_length)`
is the set of gaps. `sequence_intervals` is computed at `infer_ancestors` time
from the union of inference site positions across all sources, using the
`max_gap_length` threshold from `[ancestors]` in the config. (This
parametrisation is crude and will likely be refined in Phase 2.)

Ancestors are clipped to the interval containing their focal site(s) before
being written to the store. Focal sites grouped together that span gap
boundaries are split into separate ancestors per interval. Any sites in
`call_genotype` that fall outside the ancestor's interval are set to missing
(`-1`), and `sample_start_position` / `sample_end_position` are clamped to
interval boundaries. This is a pure Python transformation; no changes to the C
ancestor-building code are required.

`sequence_intervals` is transferred to the tree sequence metadata when the root
tree sequence is auto-created at the start of the ancestor matching phase.
Downstream steps (`post_process`) read it from there and need not re-open the
ancestor VCZ.

### Multi-source ancestor generation (Phase 2)

When `[ancestors]` lists multiple sources, `infer_ancestors` constructs the
ancestor haplotypes from the union of inference sites across all sources:

- Genotypes at sites absent from a given source are treated as missing for that
  source's samples.
- A site is included in inference if it passes the site mask in at least one
  source and has a consistent ancestral state across all sources that have it.
- `sequence_intervals` and `max_gap_length` are applied to the union of all
  inference site positions.
- The single-contig constraint is enforced across all sources jointly.

### Relationship to the samples VCZ

The ancestor VCZ is produced by `infer_ancestors` from one or more samples
VCZs. Its `variant_position` is a strict subset of the input positions,
containing only inference sites. `variant_alleles` is copied through,
but resolved so that allele 0 is the ancestral state and 1 is the
derived state. The ancestor store is therefore interpretable as a
standalone file.

## Matching engine

### Design

The matching engine is abstracted behind a `Matcher` class that takes a
**fixed** `tskit.TreeSequence` as its reference panel and matches one or more
haplotypes against it. This separates two concerns currently entangled in the
existing code:

- **Matching** — running the HMM to find the most likely copying path and
  mutations for each haplotype. Stateless once the reference tree sequence
  is fixed.
- **Tree building** — extending the accumulated tree sequence with match
  results before the next group can be matched. State between groups is now
  expressed explicitly as a data transformation rather than internal engine
  state.

```python
# Note: preliminary design — a list of structs storable as JSON may be
# preferable for debugging.

@dataclass
class MatchResult:
    path_left:      np.ndarray  # (n_edges,) int32 — left breakpoints (site indices)
    path_right:     np.ndarray  # (n_edges,) int32 — right breakpoints (site indices)
    path_parent:    np.ndarray  # (n_edges,) int32 — parent node ids
    mutation_sites: np.ndarray  # (n_mutations,) int32 — site indices
    mutation_state: np.ndarray  # (n_mutations,) int8  — derived allele

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
    ) -> list[MatchResult]: ...
```

The active matching range for each haplotype is derived from its missing data
pattern (scanning flanks for the first and last non-missing site). For ancestor
haplotypes this is redundant with the stored `sample_start_position` /
`sample_end_position`, but scanning is cheap and avoids a separate lookup.

Results are plain data objects with no dependency on the engine internals, so
they can be serialised to disk for the distributed case.

The tree sequence is extended after all haplotypes in a group have been matched:

```python
def extend_ts(
    ts: tskit.TreeSequence,
    node_times: np.ndarray,          # (n_haplotypes,) time for each new node
    results: list[MatchResult],
    node_metadata: list[dict],       # provenance metadata per haplotype
    create_individuals: bool = True, # whether to create tskit individuals
    ploidy: int = 1,                 # nodes per individual (from VCZ call_genotype shape)
    path_compression: bool = True,
) -> tskit.TreeSequence: ...
```

`make_root_ts` copies `sequence_intervals` from the ancestor VCZ array into
the tree sequence's top-level metadata. Downstream steps that require this
(in particular `post_process`) read `sequence_intervals` from the tree
sequence metadata, so do not need to re-open the ancestor VCZ.

`extend_ts` adds nodes, edges (clipped at gap boundaries using
`sequence_intervals` from the ts metadata), and mutations. When
`create_individuals` is true, nodes are grouped into tskit individuals at the
rate of one individual per `ploidy` nodes. The returned tree sequence becomes
the reference panel for the next group.

### The match loop

All match steps — whether matching generated ancestors, modern samples, ancient
samples, or pedigree samples — use the same loop. The only variation is in how
haplotypes are grouped before matching begins.

```python
def match(step_config, sources, reference_ts):
    positions, all_haplotypes, all_times, node_metadata = collect_haplotypes(sources)

    if not step_config.create_individuals:
        # Ancestor step: linesweep grouping accounts for genomic overlap
        groups = linesweep_groups(sources)
    else:
        # Sample step (modern, ancient, or pedigree): group strictly by time.
        # All haplotypes at the same time form one group; time=0 is last.
        groups = time_ordered_groups(all_times)

    current_ts = reference_ts or make_root_ts(
        sequence_length, positions,
        sequence_intervals=sources[0]["sequence_intervals"][:],
    )

    for haplotype_ids in groups:
        matcher = Matcher(current_ts, positions,
                          step_config.recombination_rate,
                          step_config.mismatch_ratio)
        results = matcher.match(all_haplotypes[haplotype_ids])
        current_ts = extend_ts(
            current_ts,
            node_times=all_times[haplotype_ids],
            results=results,
            node_metadata=[node_metadata[i] for i in haplotype_ids],
            create_individuals=step_config.create_individuals,
            path_compression=step_config.path_compression,
        )

    return current_ts
```

**Linesweep grouping (ancestor step).** Generated ancestors have complex
within-epoch overlap patterns. The full linesweep algorithm (see distributed
inference section) builds a dependency graph and topologically sorts ancestors
into groups. This is only correct and necessary for the ancestor VCZ; applying
it to real sample haplotypes would be unnecessary overhead.

**Time-ordered grouping (sample step).** For real sample haplotypes —
including ancient DNA and pedigree members — simple time ordering is sufficient.
Haplotypes with identical times form a group; groups are processed from oldest
to youngest, with time=0 forming the final group. No linesweep is needed
because real samples do not have the same within-epoch overlap constraints as
generated ancestors.

**Ancient and pedigree samples.** Adding sources with `sample_time > 0` to a
samples match step requires no special handling. The time-ordered grouping
automatically places ancient haplotypes before modern samples. They copy from
whatever is already in the reference tree sequence (which already contains all
generated ancestors), then become part of the panel for younger groups. They
produce individuals and appear in the output tree sequence with their correct
times.

### MVP implementation

For the MVP, `Matcher` wraps the existing C engine. The existing code has a
`restore_tree_sequence_builder` function that reconstructs the internal
`TreeSequenceBuilder` state from a `tskit.TreeSequence`. The MVP `Matcher`
calls `restore_tree_sequence_builder(ts)` on construction, then delegates
`match()` to the existing `_tsinfer.AncestorMatcher`. No state is maintained
between groups; a fresh `Matcher` is instantiated for each group's tree
sequence.

No changes to the C extension are required.

### Future: external HMM matching engine

Phase 2 will delegate HMM matching to an external package. Any implementation
satisfying the `Matcher` interface can be dropped in without changes to the
pipeline. The config would gain an `engine` key in `[[match]]`.

## Node and individual metadata

### Node metadata

Every node added by tsinfer carries JSON metadata recording its provenance.
The tskit node metadata schema is set automatically:

| Field | Present for | Description |
|---|---|---|
| `source` | all nodes | Name of the `[[source]]` the haplotype came from |
| `sample_id` | all nodes | Value of `sample_id` in the source VCZ |
| `ploidy_index` | sample nodes only | 0-based index within the individual (0 or 1 for diploid) |

Ancestor nodes are haploid and have no associated individual, so `ploidy_index`
is omitted. Example node metadata:

```json
{"source": "ancestors",  "sample_id": "ancestor_42"}
{"source": "cohort",     "sample_id": "NA12878", "ploidy_index": 0}
{"source": "cohort",     "sample_id": "NA12878", "ploidy_index": 1}
{"source": "ancient",    "sample_id": "VK2013",  "ploidy_index": 0}
```

Note that there is some redundancy here with the individual metadata ---
this is intentional as the node metadata is primarily a debugging tool and
can be dropped automatically in postprocessing for production builds.

### Individual metadata

Sample nodes are grouped into tskit individuals at the rate of one individual
per `ploidy` nodes (derived from the `call_genotype` shape in the source VCZ).
Individual metadata is populated from user-declared VCZ field mappings in
`[individual_metadata]`:

```toml
[individual_metadata]
# Maps tskit individual metadata field name → VCZ sample-dimensioned array name.
# The MetadataSchema is derived from VCZ array dtypes (str → string,
# int → integer) and stored in the output tree sequence.
fields     = {sample_id = "sample_id", sex = "sample_sex"}

# Optional: maps a VCZ array to tskit populations.
# Unique values in the array become population entries; each individual is
# assigned to its population by index. If omitted, all individuals share one
# unnamed population.
population = "sample_population"
```

JSON metadata encoding is used during inference. Conversion to more efficient
struct-based tskit metadata can be performed in post-processing if desired.

With multiple sources in a `[[match]]` step, fields are resolved per source.
For a sample present in multiple sources, the first source that provides a
value for a given field wins.

Ancestor nodes have no individuals and carry no individual metadata.

## CLI

### Overview

All CLI commands take a config file as their primary argument. Parameters
describing the dataset and inference live in the config file; runtime concerns
(parallelism, verbosity, whether to overwrite outputs) are CLI flags.

```
tsinfer <command> config.toml [flags]
```

### Pipeline commands

| Command | Reads | Writes | Notes |
|---|---|---|---|
| `infer-ancestors` | sources VCZ | `ancestors.path` VCZ | Builds ancestor haplotypes |
| `match NAME` | source VCZ(s), reference `.trees` | output `.trees` | Runs one `[[match]]` step |
| `post-process` | `.trees` | `.trees` (overwrite) | Simplify, erase gaps |
| `run` | sources VCZ | final `.trees` | All steps in sequence |

`match NAME` executes the `[[match]]` entry whose `name` field matches. Steps
are also addressable by zero-based index (`match 0`, `match 1`).

Each command exits 0 on success and non-zero on failure. Errors go to stderr;
nothing goes to stdout except `config show`.

### Runtime flags

| Flag | Default | Description |
|---|---|---|
| `--threads N` | `1` | Worker threads for matching |
| `--force` | off | Overwrite existing output files |
| `--progress` | off | Show per-step progress bars |
| `-v / --verbose` | off | Increase log verbosity (`-vv` for debug) |

### The `config` utility

```
tsinfer config show   config.toml    # print resolved config with defaults filled
tsinfer config check  config.toml    # validate config and verify all input paths
```

`config check` is useful as the first step in a workflow to catch missing files
or invalid values before compute begins. `config show` prints the resolved
config as TOML, useful for recording exactly what parameters were used.

### Distributed pipeline commands

When `[distributed]` is present in the config, each `match` step can be
decomposed into init / group / partition / finalise commands. See the
distributed inference section for full details.

```
tsinfer match-init           config.toml NAME
tsinfer match-group          config.toml NAME GROUP_START GROUP_END
tsinfer match-partition      config.toml NAME GROUP_INDEX PARTITION_INDEX
tsinfer match-finalise-group config.toml NAME GROUP_INDEX
tsinfer match-finalise       config.toml NAME
```

### Logging

Log messages go to stderr. Default level is `WARNING`; `-v` raises to `INFO`,
`-vv` to `DEBUG`. At `INFO` level each command logs the step, input/output
paths, key parameter values, and a wall-time and peak-memory summary on exit.
At `DEBUG` level the matching engine emits per-group statistics (group index,
haplotype count, grouping regime, partition count).

## Testing

### Synthetic data helpers

VCZ stores are zarr groups, so test data can be constructed directly in memory
using `zarr.store.MemoryStore` — no files, no VCF parsing, no simulation
required for simple cases:

```python
def make_sample_vcz(
    genotypes,        # (n_sites, n_samples, ploidy) int8
    positions,        # (n_sites,) int
    alleles,          # (n_sites, n_alleles) str
    ancestral_state,  # (n_sites,) str
    sequence_length,
    **kwargs,         # site_mask, sample_mask, sample_time, ...
) -> zarr.Group:      # backed by MemoryStore

def make_ancestor_vcz(
    genotypes,          # (n_sites, n_ancestors, 1) int8; -1 for missing flanks
    positions,          # (n_sites,) int
    alleles,            # (n_sites, n_alleles) str
    times,              # (n_ancestors,) float
    focal_positions,    # (n_ancestors, max_focal_positions) int; -2 padded
    sequence_intervals, # (n_intervals, 2) int
) -> zarr.Group:        # backed by MemoryStore
```

All test data construction goes through these helpers. Test functions contain
only the call under test and assertions.

### Three tiers of test data

**Tier 1: Hand-constructed minimal cases.** Small numpy arrays where the correct
output is known by inspection. Cases include:

- Single site, single ancestor, single sample
- Ancestor with missing left flank only, right flank only, or both
- `max_focal_positions > 1` (ancestor with multiple focal sites)
- Ancestor clipped by gap interval
- Site at position 0 or at `sequence_length`
- All-missing site in the samples
- Site mask and sample mask excluding various subsets
- Diploid vs haploid samples
- Multiple ancestors at the same time (single-epoch matching)
- Ancient samples interleaved with modern samples
- Pedigree samples at time=1 and time=2
- All samples identical (no variation)

**Tier 2: Simulation-based cases.** msprime is used to simulate a tree sequence
which is converted to VCZ via:

```python
def ts_to_sample_vcz(ts, ancestral_state="REF") -> zarr.Group:
```

Used for end-to-end pipeline tests checking round-trip fidelity. Cases are
parametrised over sample count, ploidy, sequence length, and
mutation/recombination rate.

**Tier 3: Config and CLI tests.** `Config` can be constructed programmatically
using in-memory stores, so CLI tests need not write TOML files. TOML-specific
tests (path resolution, missing fields, invalid values) use a `tmp_path`
fixture.

### Parametric coverage

```python
@pytest.fixture(params=["single_site", "two_epochs", "diploid",
                         "with_missing", "with_ancient", "with_pedigree"])
def sample_vcz(request):
    return CANONICAL_CASES[request.param]
```

Named corner cases are written as individual tests so their intent is clear.

## Distributed inference

Not required for the Phase 1 MVP but must be fully supported in v1.0.

### Parallelism characteristics

- **`infer_ancestors`** — single job.
- **Ancestor `match` step** — two-level structure. Groups (from the linesweep
  algorithm) must be processed strictly in order. Within a large group, ancestors
  are subdivided into *partitions* that can be matched independently in parallel.
  Small groups are never partitioned. Partition count is controlled by
  `min_work_per_job`: ancestors are greedily bin-packed so each partition has
  approximately `min_work_per_job` genotypes of work.
- **Samples `match` step** — embarrassingly parallel for time=0 samples (all
  copy from the same fixed reference ts). Ancient/pedigree groups must be
  processed in time order before the final modern-sample group, but each group
  is independently parallelisable within itself.
- **`post_process`** — single job.

### Ancestor grouping algorithm

Before matching begins, haplotypes in a `[[match]]` step are arranged into an
ordered sequence of groups satisfying two constraints: (1) a haplotype that
overlaps genomically with an older haplotype must be placed in a later group,
so that when it is matched the older haplotype is already in the tree sequence;
(2) same-time haplotypes that overlap genomically must be placed in the same
group, so they cannot match each other.

For the ancestor `[[match]]` step, the grouping is computed by
`group_by_linesweep` using `sample_time`, `sample_start_position`, and
`sample_end_position` from the ancestor VCZ. Two regimes are used:

**Linesweep regime (earlier ancestors).** For epochs up to the first *large
epoch*:

1. Overlapping same-time ancestors are temporarily merged into combined intervals.
2. A linesweep identifies all pairs where one ancestor is older and their spans
   overlap, adding a directed dependency edge.
3. A topological sort assigns each ancestor a group index. Ancestors with no
   inter-group dependencies may end up in the same group across different time
   epochs, maximising parallel work per group.
4. Same-time ancestors are un-merged, restoring original identities.

**Epoch regime (later ancestors).** For large epochs the dependency graph
becomes intractable. The cutoff is the first epoch — in the second half of the
time distribution — whose size exceeds 500× the median epoch size. From that
point, each epoch forms its own group. These groups are typically large and
benefit most from partitioning.

The two regimes combine into a single ordered group list. The first group
always contains only the virtual root ancestor.

For sample `[[match]]` steps, groups are formed strictly by time (oldest first,
time=0 last). No linesweep is needed.

### Configuration

```toml
[distributed]
work_dir           = "inference_work"  # directory for intermediate state
min_work_per_job   = 1_000_000        # genotypes per job, controls partition count
max_num_partitions = 1000             # upper bound on partitions per group
```

A `metadata.json` file written by each init command records the full job
structure (group count, partition counts per group, parameters) so that
subsequent commands and workflow managers can determine what jobs to run.

### CLI

```
# 1. Initialise: creates work_dir/NAME/, writes metadata.json
tsinfer match-init           config.toml NAME

# 2a. Small groups (no partitioning): run one or more groups in a single job
tsinfer match-group          config.toml NAME GROUP_START GROUP_END

# 2b. Large groups: run each partition independently (parallel)
tsinfer match-partition      config.toml NAME GROUP_INDEX PARTITION_INDEX

# 3. Finalise a large group once all partitions complete
tsinfer match-finalise-group config.toml NAME GROUP_INDEX

# 4. Finalise the step once all groups done; writes output .trees file
tsinfer match-finalise       config.toml NAME
```

Steps 2a and 2b are not mutually exclusive; the workflow manager reads
`metadata.json` to determine which applies to each group. The same commands
handle both the ancestor and samples match steps — only the name differs.

### Snakemake integration

```python
checkpoint match_init:
    input:  lambda wc: get_reference_ts(wc.name)
    output: "inference_work/{name}/metadata.json"
    shell:  "tsinfer match-init config.toml {wildcards.name}"

def match_partitions(wildcards):
    meta = json.load(checkpoints.match_init.get(name=wildcards.name).output[0])
    partitions = []
    for g, group in enumerate(meta["groups"]):
        if group["partitions"]:
            partitions += expand(
                "inference_work/{name}/group_{g}/partition_{p}.pkl",
                name=wildcards.name, g=g, p=range(len(group["partitions"])))
    return partitions

rule match_partition:
    input:  "inference_work/{name}/metadata.json"
    output: "inference_work/{name}/group_{g}/partition_{p}.pkl"
    shell:  "tsinfer match-partition config.toml {wildcards.name} {wildcards.g} {wildcards.p}"

rule match_finalise:
    input:  match_partitions
    output: lambda wc: get_output_path(wc.name)
    shell:  "tsinfer match-finalise config.toml {wildcards.name}"
```

### Python API

```python
cfg = tsinfer.Config.from_toml("inference.toml")

for name in ["ancestors", "samples"]:
    meta = tsinfer.match_init(cfg, name)
    for g, group in enumerate(meta["groups"]):
        if group["partitions"] is None:
            tsinfer.match_group(cfg, name, g, g + 1)
        else:
            for p in range(len(group["partitions"])):
                tsinfer.match_partition(cfg, name, g, p)
            tsinfer.match_finalise_group(cfg, name, g)
    tsinfer.match_finalise(cfg, name)
```
