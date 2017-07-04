"""
Visualisation of the copying process and ancestor generation using PIL
"""
import os

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor

import tsinfer
import msprime


class Visualiser(object):

    def __init__(self, store, samples, box_size=8):
        self.store = store
        self.box_size = box_size
        self.samples = samples
        frequency = np.sum(samples, axis=0)
        num_rows = 0
        self.row_map = {0:0}
        last_frequency = 0
        self.ancestors = np.zeros((store.num_ancestors, store.num_sites), dtype=np.int8)
        for j in range(1, store.num_ancestors):
            start, focal, end, num_older_ancestors = store.get_ancestor(
                    j, self.ancestors[j, :])
            if frequency[focal] != last_frequency:
                last_frequency = frequency[focal]
                num_rows += 1
            self.row_map[j] = num_rows
            num_rows += 1
        num_rows += 1
        for j in range(samples.shape[0]):
            self.row_map[store.num_ancestors + j] = num_rows
            num_rows += 1

        self.top_padding = box_size
        self.left_padding = box_size
        self.bottom_padding = box_size
        self.right_padding = box_size
        self.background_colour = ImageColor.getrgb("white")
        self.copying_outline_colour = ImageColor.getrgb("black")
        self.copying_opacity = 64
        self.colours = {
            0: ImageColor.getrgb("blue"),
            1: ImageColor.getrgb("red")}
        self.width = box_size * store.num_sites + self.left_padding + self.right_padding
        self.height = box_size * num_rows + self.top_padding + self.bottom_padding
        self.base_image = Image.new("RGBA", (self.width, self.height))
        self.draw_base()

    def draw_base(self):
        # Draw a white background over everything.
        b = self.box_size
        draw = ImageDraw.Draw(self.base_image)
        draw.rectangle([(0, 0), (self.width, self.height)], fill=self.background_colour)
        # Draw the ancestors
        for j in range(self.ancestors.shape[0]):
            a = self.ancestors[j]
            row = self.row_map[j]
            y = row * b + self.top_padding
            for k in range(self.store.num_sites):
                x = k * b + self.left_padding
                if a[k] != -1:
                    draw.rectangle([(x, y), (x + b, y + b)], fill=self.colours[a[k]])
        # Draw the samples
        for j in range(self.samples.shape[0]):
            a = self.samples[j]
            row = self.row_map[self.store.num_ancestors + j]
            y = row * b + self.top_padding
            for k in range(self.store.num_sites):
                x = k * b + self.left_padding
                draw.rectangle([(x, y), (x + b, y + b)], fill=self.colours[a[k]])

    def draw_haplotypes(self, filename):
        self.base_image.save(filename)

    def draw_copying_path(self, filename, child_row, parents):
        # print("copying path", filename, child_row, parents)
        b = self.box_size
        m = self.store.num_sites
        image = self.base_image.copy()
        draw = ImageDraw.Draw(image)
        y = self.row_map[child_row] * b + self.top_padding
        x = self.left_padding
        draw.rectangle([(x, y), (x + m * b, y + b)], outline=self.copying_outline_colour)
        for k in range(m):
            row = self.row_map[parents[k]]
            y = row * b + self.top_padding
            x = k * b + self.left_padding
            a = self.ancestors[parents[k], k]
            # fill = tuple(list(self.colours[a]) + [self.copying_opacity])
            fill = "black"
            draw.rectangle([(x, y), (x + b, y + b)], fill=fill)
        print("Saving", filename)
        image.save(filename)

    def draw_copying_paths(self, tree_sequence, pattern):
        N = self.store.num_ancestors + self.samples.shape[0]
        P = np.zeros((N, self.store.num_sites), dtype=int) - 1
        site_index = {}
        for site in tree_sequence.sites():
            site_index[site.position] = site.index
        site_index[tree_sequence.sequence_length] = tree_sequence.num_sites
        site_index[0] = 0
        for e in tree_sequence.edgesets():
            for c in e.children:
                left = site_index[e.left]
                right = site_index[e.right]
                assert left < right
                P[c, left:right] = e.parent
        for j in range(1, self.store.num_ancestors):
            self.draw_copying_path(pattern.format(j), j, P[j])
        for j in range(self.samples.shape[0]):
            k = self.store.num_ancestors + j
            self.draw_copying_path(pattern.format(k), k, P[k])


def visualise(
        samples, positions, length, recombination_rate, error_rate, method="C",
        box_size=8):

    store = tsinfer.build_ancestors(samples, positions, method=method)
    inferred_ts = tsinfer.infer(
        samples, positions, length, recombination_rate, error_rate, method=method)
    visualiser = Visualiser(store, samples, box_size=box_size)
    prefix = "tmp__NOBACKUP__/"
    visualiser.draw_haplotypes(os.path.join(prefix, "haplotypes.png"))
    visualiser.draw_copying_paths(inferred_ts, os.path.join(prefix, "copying_{}.png"))


def run_viz(n, L, seed):
    ts = msprime.simulate(
        n, length=L, recombination_rate=0.5, mutation_rate=1, random_seed=seed)
    print(ts.num_sites)
    if ts.num_sites == 0:
        print("zero sites; skipping")
        return
    positions = np.array([site.position for site in ts.sites()])
    S = np.zeros((ts.sample_size, ts.num_sites), dtype="i1")
    for variant in ts.variants():
        S[:, variant.index] = variant.genotypes
    visualise(S, positions, L, 1e-6, 1e-200, method="C", box_size=8)


def main():
    run_viz(10, 10, 1)


if __name__ == "__main__":
    main()
