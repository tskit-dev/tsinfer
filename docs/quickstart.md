(sec_quickstart)=

# Quickstart

_Tsinfer_ infers [tree sequences](https://tskit.dev/tutorials/what_is.html)
from phased genetic variation data. Input data is stored in
[VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/) (.vcz) format, and
the pipeline is controlled by a TOML configuration file.

The typical workflow is:

1. Convert a bgzipped VCF to VCZ format
2. Write a TOML config describing inputs, outputs, and parameters
3. Run the pipeline via the `tsinfer` CLI
4. Analyse the resulting tree sequence with [tskit](https://tskit.dev/)


(sec_quickstart_preparing_data)=

## Preparing input data

_Tsinfer_ reads phased genotype data from `.vcz` stores. If you have a
bgzipped, indexed VCF, convert it using
[vcf2zarr](https://sgkit-dev.github.io/bio2zarr/vcf2zarr/overview.html):

```bash
vcf2zarr convert mydata.vcf.gz mydata.vcz
```

### Ancestral states

Each site used for inference requires a known **ancestral allele**. There are
several ways to provide this:

- **AA INFO field in the VCF.** If your VCF has an `AA` (ancestral allele) INFO
  field, `vcf2zarr` will store it as `variant_AA` in the `.vcz` store. You can
  then reference it directly in the config:
  ```toml
  [ancestral_state]
  path = "mydata.vcz"
  field = "variant_AA"
  ```

- **Separate ancestral allele VCZ.** If ancestral alleles come from a different
  source (e.g. an Ensembl ancestral allele VCF), convert that to `.vcz` too and
  point to it:
  ```toml
  [ancestral_state]
  path = "ancestral_alleles.vcz"
  field = "variant_AA"
  ```

Sites where the ancestral allele is unknown or does not match any allele in the
data are automatically treated as _non-inference_ sites (see
{ref}`sec_quickstart_inference_sites`).


(sec_quickstart_config)=

## Writing the config

The TOML config tells _tsinfer_ where to find inputs, where to write outputs,
and what parameters to use. Here is a minimal example:

```toml
# -- Sources: one or more named VCZ stores ---------------------------------
[[source]]
name = "example"
path = "example_data.vcz"

# -- Ancestral state --------------------------------------------------------
[ancestral_state]
path = "example_data.vcz"
field = "variant_AA"

# -- Ancestors: output store for inferred ancestors -------------------------
[[ancestors]]
name = "ancestors"
path = "example_ancestors.vcz"
sources = ["example"]

# -- Match: HMM matching and tree sequence output --------------------------
[match]
output = "example_output.trees"

[match.sources.ancestors]
node_flags = 0              # ancestors are not samples
create_individuals = false

[match.sources.example]
# node_flags = 1            # default: mark as samples
# create_individuals = true # default: group into individuals
```

### Sources

Each `[[source]]` block names a VCZ store. You can filter variants and samples
using bcftools-style expressions:

```toml
[[source]]
name = "1kgp_chr20"
path = "data/1kgp_chr20.vcz"
include = "TYPE='snp' && N_ALT=1"   # biallelic SNPs only
exclude = "FILTER != 'PASS'"
regions = "chr20:1000000-50000000"   # restrict to a region
samples = "^NA12878"                 # exclude specific samples
```

### Ancestors

The `[[ancestors]]` block configures ancestor generation. Key options:

| Field | Default | Description |
|-------|---------|-------------|
| `name` | (required) | Unique name for this ancestor set |
| `path` | (required) | Output path for the ancestor `.vcz` store |
| `sources` | (required) | List of source names to build ancestors from |
| `max_gap_length` | 500,000 | Split intervals at gaps wider than this (bp) |
| `genotype_encoding` | `"eight_bit"` | `"one_bit"` uses ~8x less memory (biallelic only) |

### Match

The `[match]` section controls HMM matching. Each source that should appear
in the output tree sequence needs a `[match.sources.<name>]` sub-table.
Ancestors should have `node_flags = 0` and `create_individuals = false`.

| Field | Default | Description |
|-------|---------|-------------|
| `output` | (required) | Output `.trees` file path |
| `path_compression` | `true` | Viterbi path compression |
| `workdir` | — | Directory for checkpoints (enables resume) |
| `keep_intermediates` | `false` | Keep per-group checkpoint files |

### Post-processing (optional)

```toml
[post_process]
split_ultimate = true    # split virtual root into per-tree roots
erase_flanks = true      # erase ancestry outside informative sites
```


(sec_quickstart_running)=

## Running the pipeline

### Full pipeline

Run all steps (infer ancestors, match, post-process) in one command:

```bash
tsinfer run config.toml --threads 4 -v
```

### Individual steps

For large datasets, you may want to run steps separately:

```bash
# Step 1: Build ancestors
tsinfer infer-ancestors config.toml --threads 4 -v

# Step 2: Match ancestors and samples
tsinfer match config.toml --threads 4 -v
```

Post-processing and site augmentation can also be run separately:

```bash
tsinfer post-process config.toml --input raw.trees -v
tsinfer augment-sites config.toml --input output.trees --output final.trees
```

### Validating the config

Before running, check that your config is valid and all paths resolve:

```bash
tsinfer config check config.toml
```

To see the fully resolved config with all defaults filled in:

```bash
tsinfer config show config.toml
```

### Common CLI options

| Flag | Description |
|------|-------------|
| `-t, --threads N` | Number of worker threads (default: 1) |
| `-f, --force` | Overwrite existing output files |
| `-p, --progress` | Show progress bars |
| `-v` | Verbose logging (repeat for more: `-vv` for debug) |
| `-l, --log-file FILE` | Write logs to a file |


(sec_quickstart_inspecting)=

## Inspecting the result

The output is a standard [tskit](https://tskit.dev/) tree sequence. Load it in
Python to explore:

```python
import tskit

ts = tskit.load("example_output.trees")
print(f"{ts.num_trees} trees, {ts.num_samples} samples, {ts.num_sites} sites")

# Draw the trees
ts.draw_svg(size=(600, 300), y_axis=True)
```

Each sample in the original VCZ file corresponds to an _individual_ in the tree
sequence. Since diploid individuals have two haploid genomes,
`ts.num_samples` will be twice the number of diploid individuals.

:::{note}
By default, internal node times in the inferred tree sequence are
_not_ in years or generations — they reflect allele frequencies. To get
meaningful dates, use [tsdate](https://tskit.dev/software/tsdate.html).
Calculating branch-length statistics on uncalibrated trees will raise an error.
:::


(sec_quickstart_inference_sites)=

## Inference sites

Not all sites are used by _tsinfer_ for inferring the genealogy. These
_non-inference_ sites are still included in the final tree sequence, but their
mutations are placed by
[parsimony](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.Tree.map_mutations).
Non-inference sites include:

- **Fixed sites** — no variation between samples
- **Singletons** — only one genome carries the derived allele
- **Unknown ancestral state** — the ancestral allele does not match any allele
  at the site
- **Multiallelic sites** — more than two alleles

Additional sites can be excluded from inference using the `exclude` filter
in the `[[source]]` config, or by adding an `[augment_sites]` section to
place them separately via parsimony.


(sec_quickstart_config_reference)=

## Config reference

### `[[source]]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | (required) | Unique name for this source |
| `path` | string | (required) | Path to VCZ store |
| `include` | string | — | bcftools include expression |
| `exclude` | string | — | bcftools exclude expression |
| `samples` | string | — | Sample filter (comma-separated, `^` to exclude) |
| `regions` | string | — | Genomic region (half-open) |
| `targets` | string | — | Exact target positions |
| `sample_time` | various | — | Per-sample times: constant, field name, or `{path, field}` |

### `[ancestral_state]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | (required) | Path to VCZ containing ancestral alleles |
| `field` | string | (required) | Array name (e.g. `"variant_AA"`) |

### `[[ancestors]]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | (required) | Unique ancestor set name |
| `path` | string | (required) | Output VCZ path |
| `sources` | list | (required) | Source names to build from |
| `max_gap_length` | int | 500,000 | Split at gaps wider than this (bp) |
| `samples_chunk_size` | int | 100 | Zarr chunk size (ancestor dim) |
| `variants_chunk_size` | int | 50,000 | Zarr chunk size (site dim) |
| `compressor` | string | `"zstd"` | Blosc compressor name |
| `compression_level` | int | 7 | Compression level (0–9) |

### `[match]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output` | string | (required) | Output `.trees` path |
| `path_compression` | bool | `true` | Enable Viterbi path compression |
| `reference_ts` | string | — | Reference tree sequence path |
| `workdir` | string | — | Checkpoint directory (enables resume) |
| `keep_intermediates` | bool | `false` | Keep per-group checkpoints |

### `[match.sources.<name>]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_flags` | int | 1 | tskit node flags (1 = `NODE_IS_SAMPLE`) |
| `create_individuals` | bool | `true` | Group sample nodes into individuals |

### `[post_process]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `split_ultimate` | bool | `true` | Split virtual root into per-tree roots |
| `erase_flanks` | bool | `true` | Erase ancestry outside informative sites |

### `[augment_sites]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sources` | list | (required) | Source names for parsimony placement |

### `[individual_metadata]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `population` | string | — | VCZ array whose unique values become populations |
| `fields.<key>` | string | — | Map tskit metadata keys to VCZ arrays |


(sec_quickstart_cli_reference)=

## CLI reference

| Command | Description |
|---------|-------------|
| `tsinfer run CONFIG` | Run the full pipeline |
| `tsinfer infer-ancestors CONFIG` | Build ancestor VCZ from sample data |
| `tsinfer match CONFIG` | Match ancestors and samples against the tree |
| `tsinfer post-process CONFIG --input FILE` | Post-process the matched tree sequence |
| `tsinfer augment-sites CONFIG --input FILE --output FILE` | Place non-inference sites by parsimony |
| `tsinfer config show CONFIG` | Print resolved config with defaults |
| `tsinfer config check CONFIG` | Validate config and verify paths |
