import os
import msprime
import sys
sys.path.insert(0, os.path.abspath('..'))
import tsinfer


ts = msprime.simulate(5, mutation_rate=0.7, random_seed=10)
tree = ts.first()
print(ts.num_sites)
print(tree.draw(format="unicode"))

with tsinfer.SampleData(path="toy.samples") as sample_data:
    sample_data.add_site(10, [0, 1, 0, 0, 0], ["A", "T"])
    sample_data.add_site(12, [0, 0, 0, 1, 1], ["G", "C"])
    sample_data.add_site(23, [0, 1, 1, 0, 0], ["C", "A"])
    sample_data.add_site(37, [0, 1, 1, 0, 0], ["G", "C"])
    sample_data.add_site(40, [0, 0, 0, 1, 1], ["A", "C"])
    sample_data.add_site(50, [0, 1, 0, 0, 0], ["T", "G"])

print(sample_data)

inferred_ts = tsinfer.infer(sample_data)
for tree in inferred_ts.trees():
    print(tree.draw(format="unicode"))

for sample_id, h in enumerate(inferred_ts.haplotypes()):
    print(sample_id, h, sep="\t")
