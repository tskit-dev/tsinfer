import os
import msprime
import sys
sys.path.insert(0, os.path.abspath('..'))
import tsinfer


ts = msprime.simulate(5, mutation_rate=0.7, random_seed=10)
tree = ts.first()
print(ts.num_sites)
print(tree.draw(format="unicode"))

sample_data = tsinfer.SampleData.initialise(num_samples=5, path="toy.samples")
# for var in ts.variants():
#     print(var.genotypes)
#     sample_data.add_site(var.site.id, var.alleles, var.genotypes)

sample_data.add_site(10, ["A", "T"], [0, 1, 0, 0, 0])
sample_data.add_site(12, ["G", "C"], [0, 0, 0, 1, 1])
sample_data.add_site(23, ["C", "A"], [0, 1, 1, 0, 0])
sample_data.add_site(37, ["G", "C"], [0, 1, 1, 0, 0])
sample_data.add_site(40, ["A", "C"], [0, 0, 0, 1, 1])
sample_data.add_site(50, ["T", "G"], [0, 1, 0, 0, 0])
sample_data.finalise()

# print(sample_data)

inferred_ts = tsinfer.infer(sample_data)
for tree in inferred_ts.trees():
    print(tree.draw(format="unicode"))

for sample_id, h in enumerate(inferred_ts.haplotypes()):
    print(sample_id, h, sep="\t")
