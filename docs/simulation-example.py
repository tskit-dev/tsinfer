import builtins
import subprocess
import sys

import msprime
import numpy as np
from Bio import bgzf

if getattr(builtins, "__IPYTHON__", False):  # if running IPython: e.g. in a notebook
    num_diploids, seq_len = 100, 10_000
    name = "notebook-simulation"
else:  # Take parameters from the command-line
    num_diploids, seq_len = int(sys.argv[1]), float(sys.argv[2])
    name = "cli-simulation"

ts = msprime.sim_ancestry(
    num_diploids,
    population_size=10**4,
    recombination_rate=1e-8,
    sequence_length=seq_len,
    random_seed=6,
)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=7)
ts.dump(name + "-source.trees")
print(
    f"Simulated {ts.num_samples} samples over {seq_len/1e6} Mb:",
    f"{ts.num_trees} trees and {ts.num_sites} sites",
)

# Convert to a zarr file: this should be easier once a tskit2zarr utility is made, see
# https://github.com/sgkit-dev/bio2zarr/issues/232
np.save(f"{name}-AA.npy", [s.ancestral_state for s in ts.sites()])
vcf_name = f"{name}.vcf.gz"
with bgzf.open(vcf_name, "wt") as f:
    ts.write_vcf(f)
subprocess.run(["tabix", vcf_name])
ret = subprocess.run(
    "python -m bio2zarr vcf2zarr convert --force".split() + [vcf_name, name + ".vcz"],
    stderr=subprocess.DEVNULL if name == "notebook-simulation" else None,
)
if ret.returncode == 0:
    print(f"Converted to {name}.vcz")
