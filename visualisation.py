"""
Visualisation of the copying process and ancestor generation using PIL
"""
import os
import sys

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor

import zarr

import tsinfer
import msprime


class Visualiser(object):

    def __init__(self, samples, ancestors_root, inferred_ts, box_size=8):
        self.box_size = box_size
        self.samples = samples
        self.inferred_ts = inferred_ts
        self.num_samples, self.num_sites = samples.shape

        self.top_padding = box_size
        self.left_padding = box_size
        self.bottom_padding = box_size
        self.mid_padding = 2 * box_size
        self.right_padding = box_size
        self.background_colour = ImageColor.getrgb("white")
        self.copying_outline_colour = ImageColor.getrgb("black")
        self.colours = {
            255: ImageColor.getrgb("white"),
            0: ImageColor.getrgb("blue"),
            1: ImageColor.getrgb("red")}
        self.copy_colours = {
            0: ImageColor.getrgb("black"),
            1: ImageColor.getrgb("green")}

        # Make the haplotype box
        frequency = np.sum(samples, axis=0)
        num_haplotype_rows = 1
        self.row_map = {0:0}
        last_frequency = 0
        self.ancestors = ancestors_root["/ancestors/haplotypes"][:]
        self.num_ancestors = self.ancestors.shape[0]

        num_haplotype_rows += 1
        for j in range(self.num_ancestors):
            self.row_map[j] = num_haplotype_rows
            num_haplotype_rows += 1
        num_haplotype_rows += 1
        for j in range(samples.shape[0]):
            self.row_map[self.num_ancestors + j] = num_haplotype_rows
            num_haplotype_rows += 1

        # Make the tree sequence box.
        num_ancestor_rows = self.num_ancestors

        self.width = box_size * self.num_sites + self.left_padding + self.right_padding
        self.height = (
            self.top_padding + self.bottom_padding + self.mid_padding
            + num_haplotype_rows * box_size)
        self.ts_origin = (self.left_padding, self.top_padding)
        self.haplotype_origin = (
                self.left_padding,
                self.top_padding + self.mid_padding)
        self.base_image = Image.new(
            "RGB", (self.width, self.height), color=self.background_colour)
        self.draw_base()

    def draw_base(self):
        self.draw_base_haplotypes()

    def draw_base_haplotypes(self):
        b = self.box_size
        draw = ImageDraw.Draw(self.base_image)
        origin = self.haplotype_origin
        # Draw the ancestors
        for j in range(self.ancestors.shape[0]):
            a = self.ancestors[j]
            row = self.row_map[j]
            y = row * b + origin[1]
            for k in range(self.num_sites):
                x = k * b + origin[0]
                if a[k] != -1:
                    draw.rectangle([(x, y), (x + b, y + b)], fill=self.colours[a[k]])
        # Draw the samples
        for j in range(self.samples.shape[0]):
            a = self.samples[j]
            row = self.row_map[self.num_ancestors + j]
            y = row * b + origin[1]
            for k in range(self.num_sites):
                x = k * b + origin[0]
                draw.rectangle([(x, y), (x + b, y + b)], fill=self.colours[a[k]])

    def draw_haplotypes(self, filename):
        self.base_image.save(filename)

    def draw_copying_path(self, filename, child_row, parents, breakpoints):
        origin = self.haplotype_origin
        b = self.box_size
        m = self.num_sites
        image = self.base_image.copy()
        draw = ImageDraw.Draw(image)
        y = self.row_map[child_row] * b + origin[1]
        x = origin[0]
        draw.text((x - b, y), str(child_row), fill="black")
        draw.rectangle([(x, y), (x + m * b, y + b)], outline=self.copying_outline_colour)
        for k in range(m):
            if parents[k] != -1:
                row = self.row_map[parents[k]]
                y = row * b + origin[1]
                x = k * b + origin[0]
                a = self.ancestors[parents[k], k]
                draw.rectangle([(x, y), (x + b, y + b)], fill=self.copy_colours[a])

        for k, position in breakpoints.items():
            x = origin[0] + k * b
            y = origin[1] - b
            draw.text((x, y), "{}".format(position), fill="black")
            x = origin[0] + (k + 1) * b
            y1 = origin[0] + self.row_map[0] * b
            y2 = origin[1] + (self.row_map[len(self.row_map) - 1] + 1) * b
            draw.line([(x, y1), (x, y2)], fill="black")

        print("Saving", filename)
        image.save(filename)

    def draw_copying_paths(self, pattern):
        N = self.num_ancestors + self.samples.shape[0]
        P = np.zeros((N, self.num_sites), dtype=int) - 1
        C = np.zeros((self.num_ancestors, self.num_sites), dtype=int)
        breakpoints = []
        ts = self.inferred_ts
        site_index = {}
        sites = list(ts.sites())
        for site in ts.sites():
            site_index[site.position] = site.id
        site_index[ts.sequence_length] = ts.num_sites
        site_index[0] = 0
        for e in ts.edges():
            left = site_index[e.left]
            right = site_index[e.right]
            assert left < right
            P[e.child, left:right] = e.parent
        index = np.arange(self.num_sites, dtype=int)
        n = self.samples.shape[0]
        breakpoints = {}
        for j in range(1, self.num_ancestors + n):
            for k in np.where(P[j][1:] != P[j][:-1])[0]:
                breakpoints[k] = sites[k].position
            self.draw_copying_path(pattern.format(j - 1), j, P[j], breakpoints)


def visualise(ts, recombination_rate, error_rate, method="C", box_size=8):

    samples = ts.genotype_matrix()
    input_root = zarr.group()
    tsinfer.InputFile.build(
        input_root, genotypes=samples,
        recombination_rate=recombination_rate,
        position=[site.position for site in ts.sites()],
        sequence_length=ts.sequence_length,
        compress=False)
    ancestors_root = zarr.group()
    # tsinfer.build_ancestors(
    #     input_root, ancestors_root, method=method, compress=False)
    tsinfer.build_simulated_ancestors(input_root, ancestors_root, ts, guess_unknown=True)
    ancestors_ts = tsinfer.match_ancestors(
        input_root, ancestors_root, method=method, path_compression=False)
    inferred_ts = tsinfer.match_samples(input_root, ancestors_ts, method=method,
            simplify=False, path_compression=False)
    visualiser = Visualiser(samples.T, ancestors_root, inferred_ts, box_size=box_size)
    prefix = "tmp__NOBACKUP__/"
    # visualiser.draw_haplotypes(os.path.join(prefix, "haplotypes.png"))
    visualiser.draw_copying_paths(os.path.join(prefix, "copying_{}.png"))

    inferred_ts = tsinfer.match_samples(input_root, ancestors_ts, method=method,
            simplify=True, path_compression=False)

    for (left, right), tree1, tree2 in tsinfer.tree_pairs(ts, inferred_ts):
        distance = tsinfer.kc_distance(tree1, tree2)
        trailer = ""
        if distance != 0:
            trailer = "[MISMATCH]"
        print("-" * 20)
        print("Interval          =", left, "--", right)
        print("Source interval   =", tree1.interval)
        print("Inferred interval =", tree2.interval)
        print("KC distance       =", distance, trailer)
        print()
        d1 = tree1.draw(format="unicode").splitlines()
        d2 = tree2.draw(format="unicode").splitlines()
        # This won't work when the trees have different structures and therefore
        # different numbers of lines.
        for row1, row2 in zip(d1, d2):
            print(row1, " | ", row2)
        print()

def run_viz(n, L, rate, seed):
    recomb_map = msprime.RecombinationMap.uniform_map(
            length=L, rate=rate, num_loci=L)
    ts = msprime.simulate(
        n, recombination_map=recomb_map, random_seed=seed,
        model="smc_prime")
    # print(ts.num_sites)
    # if ts.num_sites == 0:
    #     print("zero sites; skipping")
    #     return
    ts = tsinfer.insert_perfect_mutations(ts)
    visualise(ts, 1e-9, 0, method="P", box_size=16)


def main():
    run_viz(8, 100, 0.1, 3)


if __name__ == "__main__":
    main()
