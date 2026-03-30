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


## Preparing input data

_Tsinfer_ reads phased genotype data from `.vcz` stores. If you have a
bgzipped, indexed VCF, convert it using
[vcf2zarr](https://sgkit-dev.github.io/bio2zarr/vcf2zarr/overview.html):

```bash
vcf2zarr convert mydata.vcf.gz mydata.vcz
```

Each site used for inference requires a known **ancestral allele**. If your VCF
has an `AA` INFO field, `vcf2zarr` stores it as `variant_AA` in the `.vcz`
store and you can reference it directly in the config. Alternatively, ancestral
alleles can come from a separate VCZ store. See the
{ref}`config reference <sec_config_reference>` for details.


## Writing the config

The TOML config tells _tsinfer_ where to find inputs, where to write outputs,
and what parameters to use. Here is a minimal example:

```toml
[[source]]
name = "mydata"
path = "mydata.vcz"

[ancestral_state]
path = "mydata.vcz"
field = "variant_AA"

[[ancestors]]
name = "ancestors"
path = "ancestors.vcz"
sources = ["mydata"]

[match]
output = "output.trees"

[match.sources.ancestors]
node_flags = 0
create_individuals = false

[match.sources.mydata]
```

The `[[source]]` block names a VCZ store. `[ancestral_state]` says where to
find ancestral alleles. `[[ancestors]]` configures ancestor generation.
`[match]` controls HMM matching and output. Each source that should appear in
the output needs a `[match.sources.<name>]` entry — ancestors use
`node_flags = 0` (not samples). For the full set of options see the
{ref}`config reference <sec_config_reference>`.


## Running the pipeline

Run all steps in one command:

```bash
tsinfer run config.toml --threads 4 -v
```

Or run steps individually:

```bash
tsinfer infer-ancestors config.toml --threads 4 -v
tsinfer match config.toml --threads 4 -v
```

Validate a config before running:

```bash
tsinfer config check config.toml
```

See the {ref}`CLI reference <sec_cli_reference>` for all commands and options.


## Inspecting the result

The output is a standard [tskit](https://tskit.dev/) tree sequence:

```python
import tskit

ts = tskit.load("output.trees")
print(f"{ts.num_trees} trees, {ts.num_samples} samples, {ts.num_sites} sites")
ts.draw_svg(size=(600, 300), y_axis=True)
```

Each diploid individual in the VCZ file corresponds to an _individual_ in the
tree sequence with two haploid sample nodes, so `ts.num_samples` is twice the
number of diploid individuals.

:::{note}
Internal node times are allele frequencies, not years or generations. Use
[tsdate](https://tskit.dev/software/tsdate.html) to add meaningful dates.
Branch-length statistics on uncalibrated trees will raise an error.
:::


(sec_quickstart_inference_sites)=

## Inference sites

Not all sites are used for inferring the genealogy. _Non-inference_ sites are
included in the final tree sequence with mutations placed by
[parsimony](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.Tree.map_mutations).
These include:

- **Fixed sites** — no variation between samples
- **Singletons** — only one genome carries the derived allele
- **Unknown ancestral state** — ancestral allele does not match any allele
- **Multiallelic sites** — more than two alleles
