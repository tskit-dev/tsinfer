(sec_config_reference)=

# Configuration reference

_Tsinfer_ is configured via a TOML file passed to the CLI. Paths in the config
are resolved relative to the config file's directory.

A complete annotated example is in
[example_config.toml](https://github.com/tskit-dev/tsinfer/blob/main/example_config.toml).


## Top-level fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `contig` | string | — | Contig to infer (e.g. `"chr20"`). Required when a store holds variants on more than one contig. |

Inference works on one contig at a time. Every store (sources, ancestral state,
ancestors) is restricted to the variants on `contig`, which must be declared in
each store's `contig_id` header under the same name. Each store must be sorted
by contig. If `contig` is not set, the variants in all stores must lie on a
single contig, which is used.

The sequence length is taken from the contig's `contig_length` in the stores,
at least one of which must declare it. If stores declare different lengths, a
warning is issued and the largest is used.


## `[[source]]`

Each `[[source]]` block defines a named view over a VCZ store. The same store
can appear multiple times with different filters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | (required) | Unique name for this source |
| `path` | string | (required) | Path to VCZ store |
| `include` | string | — | bcftools include expression (e.g. `"TYPE='snp'"`) |
| `exclude` | string | — | bcftools exclude expression |
| `samples` | string | — | Sample filter (comma-separated; prefix `^` to exclude) |
| `regions` | string | — | Genomic region, half-open (e.g. `"chr20:1000-50000"`) |
| `targets` | string | — | Exact target positions |
| `sample_time` | various | — | Per-sample times: constant, field name, or `{path, field}` dict |


## `[ancestral_state]`

Specifies where to read the ancestral allele for each variant position.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | (required) | Path to VCZ containing ancestral alleles |
| `field` | string | — | Array name in the store (e.g. `"variant_AA"`). Required unless `is_reference` is set. |
| `is_reference` | bool | `false` | Use the REF allele (`variant_allele[:, 0]`) as the ancestral state. Useful for simulations. `field` must not be set when this is `true`. |
| `contig` | string | — | Name of the contig in this store's header (see the top-level `contig` option) |
| `include` | string | — | bcftools-style filter expression selecting annotation variants (e.g. `'TYPE="snp"'`) |
| `exclude` | string | — | bcftools-style filter expression excluding annotation variants |
| `regions` | string | — | Genomic region, half-open (e.g. `"chr20:1000-50000"`) |
| `targets` | string | — | Exact target positions |

The annotation is joined to the sources by position. After filtering, there
must be at most one annotation variant at each site, and its REF allele must
equal the site's REF allele in the sources; otherwise an error is raised. All
sources with a variant at a site must also have the same REF allele there,
although their ALT alleles may differ. If
the annotation contains, for example, both a SNP and an indel at the same
position, use `include = 'TYPE="snp"'` to keep only the SNP. An ancestral
state of `.`, `N` or the empty string means the ancestral allele is unknown,
as does one that is not among the site's alleles; such sites are excluded from
inference and from `augment_sites`.


## `[[ancestors]]`

Controls the ancestor-generation step (`infer-ancestors`). At least one
`[[ancestors]]` block is required unless `[match]` specifies a `reference_ts`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | (required) | Unique ancestor set name |
| `path` | string | (required) | Output VCZ path |
| `sources` | list[str] | (required) | Source names to build ancestors from |
| `max_gap_length` | int | 500,000 | Split intervals at gaps wider than this (bp) |
| `samples_chunk_size` | int | 100 | Zarr chunk size (ancestor dimension) |
| `variants_chunk_size` | int | 50,000 | Zarr chunk size (site dimension) |
| `compressor` | string | `"zstd"` | Blosc compressor name |
| `compression_level` | int | 7 | Compression level (0–9) |
| `genotype_encoding` | string | `"eight_bit"` | `"one_bit"` uses ~8x less memory (biallelic only) |


## `[match]`

Controls the HMM matching step.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output` | string | (required) | Output `.trees` file path |
| `path_compression` | bool | `true` | Enable Viterbi path compression |
| `reference_ts` | string | — | Tree sequence to match against instead of inferred ancestors, e.g. the output of matching ancestors only. Haplotypes are matched at its sites, and all of its nodes must be older than the haplotypes being matched. |
| `workdir` | string | — | Checkpoint directory (enables resume) |
| `keep_intermediates` | bool | `false` | Keep per-group checkpoint files |


## `[match.sources.<name>]`

Per-source parameters. Every source that should appear in the output tree
sequence needs an entry here.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_flags` | int | 1 | tskit node flags (`1` = `NODE_IS_SAMPLE`, `0` for ancestors) |
| `create_individuals` | bool | `true` | Group sample nodes into tskit individuals |
| `recombination_rate` | float or string | — | Recombination rate per bp per generation, or the path to a HapMap genetic map (see below). Cannot be combined with `recombination`. |
| `recombination` | float | `0.01` | Probability of recombination between each pair of adjacent inference sites |
| `mismatch` | float | `1e-20` | Probability of a mismatch (an extra mutation) at each inference site |
| `likelihood_threshold` | float | smallest positive double | Lower bound on the normalised HMM likelihoods. Larger values (e.g. `1e-13`) let more likelihoods be merged, making matching faster at some cost in accuracy |

These are the parameters of the Li & Stephens copying HMM used to match each
haplotype from the source. The defaults (a high recombination probability and a
negligible mismatch probability) mean that haplotypes are copied exactly, using
recombination rather than extra mutations to explain differences.

When `recombination_rate` is given, the genetic distance *d* (in Morgans)
between adjacent inference sites is converted to a recombination probability
using Haldane's map function, (1 - exp(-2*d)) / 2. A genetic map file is in
HapMap format: a single header line followed by whitespace-separated columns
giving the chromosome, position (bp), rate (cM/Mb) and cumulative map
position (cM). Only the position and map columns are used. The map is
linearly interpolated between entries, starts at 0 cM at position 0, and
has zero recombination rate beyond the last entry.

The mismatch probability is not derived from the recombination rate, so
setting `recombination_rate` on its own still allows no mismatches; set
`mismatch` as well to allow them. For example:

```toml
[match.sources.ancestors]
node_flags = 0
create_individuals = false
recombination_rate = "maps/chr20.txt"
mismatch = 1e-10

[match.sources.cohort]
recombination_rate = "maps/chr20.txt"
mismatch = 1e-4
```


## `[post_process]`

Optional cleanup applied after matching.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `split_ultimate` | bool | `true` | Split virtual root into per-tree roots |
| `erase_flanks` | bool | `true` | Erase ancestry outside informative sites |


## `[augment_sites]`

Place non-inference sites via parsimony.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sources` | list[str] | (required) | Source names for parsimony placement |


## `[individual_metadata]`

Map VCZ sample-dimensioned arrays into tskit individual metadata.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `population` | string | — | VCZ array whose unique values become tskit populations |

### `[individual_metadata.fields]`

Each key becomes a tskit metadata field; the value names the VCZ array.

```toml
[individual_metadata.fields]
name = "sample_id"
sex = "sample_sex"
```
