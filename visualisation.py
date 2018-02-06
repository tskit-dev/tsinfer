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

# script_path = __file__ if "__file__" in locals() else "./dummy.py"
# sys.path.insert(1,os.path.join(os.path.dirname(os.path.abspath(script_path)),'..','msprime')) # use the local copy of msprime in preference to the global one
# sys.path.insert(1,os.path.join(os.path.dirname(os.path.abspath(script_path)),'..','tsinfer')) # use the local copy of tsinfer in preference to the global one

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

        # for j in range(1, len(self.ancestors)):
        #     jj
        #     if frequency[focal_sites[0]] != last_frequency:
        #         last_frequency = frequency[focal_sites[0]]
        #         num_haplotype_rows += 1
        #     self.row_map[j] = num_haplotype_rows
        #     num_haplotype_rows += 1
        # num_ancestor_rows = num_haplotype_rows

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
        num_ts_rows = num_ancestor_rows

        self.width = box_size * self.num_sites + self.left_padding + self.right_padding
        self.height = (
            self.top_padding + self.bottom_padding + self.mid_padding
            + num_haplotype_rows * box_size + num_ts_rows * box_size)
        self.ts_origin = (self.left_padding, self.top_padding)
        self.haplotype_origin = (
                self.left_padding,
                self.top_padding + self.mid_padding + num_ts_rows * box_size)
        self.base_image = Image.new(
            "RGB", (self.width, self.height), color=self.background_colour)
        self.draw_base()

    def draw_base(self):
        self.draw_base_tree_sequence()
        self.draw_base_haplotypes()

    def draw_base_tree_sequence(self):
        draw = ImageDraw.Draw(self.base_image)
        I = np.zeros((self.num_ancestors, self.num_sites))
        self.__draw_ts_intensity(draw, I)

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

    def __draw_ts_intensity(self, draw, child_intensity):
        b = self.box_size
        origin = self.ts_origin
        for j in range(self.ancestors.shape[0]):
            a = self.ancestors[j]
            row = self.row_map[j]
            y = row * b + origin[1]
            for k in range(self.num_sites):
                x = k * b + origin[0]
                # hsl = "hsl(240, {}%, 50%)".format(int(child_intensity[j, k] * 100))
                # fill = ImageColor.getrgb(hsl)
                v = 255 - int(child_intensity[j, k] * 255)
                fill = (v, v, v)
                draw.rectangle(
                    [(x, y), (x + b, y + b)], fill=fill, outline="black")


    def draw_copying_path(self, filename, child_row, parents, child_intensity):
        origin = self.haplotype_origin
        b = self.box_size
        m = self.num_sites
        image = self.base_image.copy()
        draw = ImageDraw.Draw(image)
        y = self.row_map[child_row] * b + origin[1]
        x = origin[0]
        draw.rectangle([(x, y), (x + m * b, y + b)], outline=self.copying_outline_colour)
        for k in range(m):
            if parents[k] != -1:
                row = self.row_map[parents[k]]
                y = row * b + origin[1]
                x = k * b + origin[0]
                a = self.ancestors[parents[k], k]
                draw.rectangle([(x, y), (x + b, y + b)], fill=self.copy_colours[a])
        print("Saving", filename)
        self.__draw_ts_intensity(draw, child_intensity)
        image.save(filename)

    def draw_copying_paths(self, pattern):
        N = self.num_ancestors + self.samples.shape[0]
        P = np.zeros((N, self.num_sites), dtype=int) - 1
        C = np.zeros((self.num_ancestors, self.num_sites), dtype=int)
        ts = self.inferred_ts
        site_index = {}
        for site in ts.sites():
            site_index[site.position] = site.id
        site_index[ts.sequence_length] = ts.num_sites
        site_index[0] = 0
        max_num_children = 0
        for e in ts.edgesets():
            if e.parent != 0:
                max_num_children = max(max_num_children, len(e.children))
        for e in ts.edges():
            left = site_index[e.left]
            right = site_index[e.right]
            assert left < right
            P[e.child, left:right] = e.parent
        index = np.arange(self.num_sites, dtype=int)
        n = self.samples.shape[0]
        for j in range(n):
            k = self.num_ancestors + j
            C[P[k], index] += 1
            I = C / max_num_children
            self.draw_copying_path(pattern.format(j), k, P[k], I)
        for j in reversed(range(1, self.num_ancestors)):
            picture_index = self.num_ancestors - j + n - 1
            C[P[j],index] += 1
            I = C / max_num_children
            self.draw_copying_path(pattern.format(picture_index), j, P[j], I)


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


def run_viz(n, L, seed):
    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    print(ts.num_sites)
    if ts.num_sites == 0:
        print("zero sites; skipping")
        return
    visualise(ts, 1e-9, 0, method="P", box_size=16)


def main():
    run_viz(5, 8, 1)


if __name__ == "__main__":
    main()
