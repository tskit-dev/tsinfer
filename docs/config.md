(sec_config_reference)=

# Configuration reference

_Tsinfer_ is configured via a TOML file passed to the CLI. Paths in the config
are resolved relative to the config file's directory.

A complete annotated example is in
[example_config.toml](https://github.com/tskit-dev/tsinfer/blob/main/example_config.toml).


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
| `field` | string | (required) | Array name in the store (e.g. `"variant_AA"`) |


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
| `reference_ts` | string | — | Reference tree sequence (skip ancestor generation) |
| `workdir` | string | — | Checkpoint directory (enables resume) |
| `keep_intermediates` | bool | `false` | Keep per-group checkpoint files |


## `[match.sources.<name>]`

Per-source parameters. Every source that should appear in the output tree
sequence needs an entry here.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_flags` | int | 1 | tskit node flags (`1` = `NODE_IS_SAMPLE`, `0` for ancestors) |
| `create_individuals` | bool | `true` | Group sample nodes into tskit individuals |


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
