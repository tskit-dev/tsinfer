"""
Visualisation of the copying process and ancestor generation using PIL
"""
import os
import sys

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.ImageFont as ImageFont
import svgwrite

import tsinfer
import msprime


def draw_edges(ts, width=800, height=600):
    """
    Returns an SVG depiction of the edges in the specified tree sequence.
    """
    dwg = svgwrite.Drawing(size=(width, height), debug=True)
    x_pad = 20
    y_pad = 20
    x_unit = (width - 2 * x_pad) / ts.sequence_length
    y_unit = (height - 2 * y_pad) / (ts.num_nodes + 1)

    def x_trans(v):
        return x_pad + v * x_unit

    def y_trans(v):
        return height - (y_pad + v * y_unit)

    lines = dwg.add(dwg.g(id='lines', stroke='black', stroke_width=3))
    left_labels = dwg.add(dwg.g(font_size=14, text_anchor="start"))
    mid_labels = dwg.add(dwg.g(font_size=14, text_anchor="middle"))
    for u in range(ts.num_nodes):
        left_labels.add(dwg.text(str(u), (0, y_trans(u))))
    for x in ts.breakpoints():
        dwg.add(dwg.line(
            (x_trans(x), 2 * y_pad), (x_trans(x), height), stroke="grey",
            stroke_width=1))
        dwg.add(dwg.text(str(x), (x_trans(x), y_pad),  writing_mode="tb"))

    for edge in ts.edges():
        a = x_trans(edge.left), y_trans(edge.child)
        b = x_trans(edge.right), y_trans(edge.child)
        c = x_trans(edge.left + (edge.right - edge.left) / 2), y_trans(edge.child) - 5
        mid_labels.add(dwg.text(str(edge.parent), c))
        dwg.add(dwg.circle(center=a, r=3, fill="black"))
        dwg.add(dwg.circle(center=b, r=3, fill="black"))
        lines.add(dwg.line(a, b))

    for site in ts.sites():
        assert len(site.mutations) == 1
        mutation = site.mutations[0]
        a = x_trans(site.position), y_trans(mutation.node)
        dwg.add(dwg.circle(center=a, r=1, fill="red"))


    return dwg.tostring()


def draw_ancestors(ts, width=800, height=600):
    """
    Returns an SVG depiction of the ancestors in the specified tree sequence.
    """
    dwg = svgwrite.Drawing(size=(width, height), debug=True)
    x_pad = 20
    y_pad = 20
    x_unit = (width - 2 * x_pad) / ts.sequence_length
    y_unit = (height - 2 * y_pad) / (ts.num_nodes + 1)

    def x_trans(v):
        return x_pad + v * x_unit

    def y_trans(v):
        return height - (y_pad + v * y_unit)

    lines = dwg.add(dwg.g(id='lines', stroke='black', stroke_width=3))
    left_labels = dwg.add(dwg.g(font_size=14, text_anchor="start"))
    mid_labels = dwg.add(dwg.g(font_size=14, text_anchor="middle"))
    for u in range(ts.num_nodes):
        left_labels.add(dwg.text(str(u), (0, y_trans(u))))
    for x in ts.breakpoints():
        dwg.add(dwg.line(
            (x_trans(x), 2 * y_pad), (x_trans(x), height), stroke="grey",
            stroke_width=1))
        dwg.add(dwg.text("{}".format(x), (x_trans(x), y_pad),  writing_mode="tb"))

    for e in ts.edgesets():
        a = x_trans(e.left), y_trans(e.parent)
        b = x_trans(e.right), y_trans(e.parent)
        c = x_trans(e.left + (e.right - e.left) / 2), y_trans(e.parent) - 5
        mid_labels.add(dwg.text(str(e.children), c))
        dwg.add(dwg.circle(center=a, r=3, fill="black"))
        dwg.add(dwg.circle(center=b, r=3, fill="black"))
        lines.add(dwg.line(a, b))

    for site in ts.sites():
        assert len(site.mutations) == 1
        mutation = site.mutations[0]
        a = x_trans(site.position), y_trans(mutation.node)
        dwg.add(dwg.circle(center=a, r=1, fill="red"))
    return dwg.tostring()



class Visualiser(object):

    def __init__(self, original_ts, sample_data, ancestor_data, inferred_ts, box_size=8):
        self.box_size = box_size
        self.sample_data = sample_data
        self.ancestor_data = ancestor_data
        self.samples = original_ts.genotype_matrix().T
        self.ancestors = np.zeros(
            (ancestor_data.num_ancestors, original_ts.num_sites), dtype=np.uint8)
        start = ancestor_data.start[:]
        end = ancestor_data.end[:]
        variant_sites = np.hstack(
            [sample_data.variant_site, [sample_data.num_sites]])
        A = ancestor_data.genotypes[:]
        for j in range(ancestor_data.num_ancestors):
            self.ancestors[j, variant_sites[:-1]] = A[:, j]
            self.ancestors[j, :variant_sites[start[j]]] = tsinfer.UNKNOWN_ALLELE
            self.ancestors[j, variant_sites[end[j]]:] = tsinfer.UNKNOWN_ALLELE
        self.original_ts = original_ts
        self.inferred_ts = inferred_ts
        self.num_samples = self.original_ts.num_samples
        self.num_sites = self.ancestors.shape[1]

        # Find the site indexes for the true breakpoints
        breakpoints = list(original_ts.breakpoints())
        self.true_breakpoints = breakpoints[1:-1]

        self.top_padding = box_size
        self.left_padding = box_size
        self.bottom_padding = box_size
        self.mid_padding = 2 * box_size
        self.right_padding = box_size
        self.background_colour = ImageColor.getrgb("white")
        self.copying_outline_colour = ImageColor.getrgb("white")
        self.colours = {
            255: ImageColor.getrgb("pink"),
            0: ImageColor.getrgb("blue"),
            1: ImageColor.getrgb("red")}
        self.copy_colours = {
            255: ImageColor.getrgb("white"),
            0: ImageColor.getrgb("black"),
            1: ImageColor.getrgb("green")}

        # Make the haplotype box
        num_haplotype_rows = 1
        self.row_map = {0: 0}
        self.num_ancestors = self.ancestors.shape[0]

        num_haplotype_rows += 1
        for j in range(self.num_ancestors):
            self.row_map[j] = num_haplotype_rows
            num_haplotype_rows += 1
        num_haplotype_rows += 1
        for j in range(self.num_samples):
            self.row_map[self.num_ancestors + j] = num_haplotype_rows
            num_haplotype_rows += 1

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

        b = self.box_size
        origin = self.haplotype_origin
        self.x_coordinate_map = {
            site.position: origin[0] + site.id * b for site in original_ts.sites()}
        self.draw_base()

    def draw_base(self):
        draw = ImageDraw.Draw(self.base_image)
        self.draw_base_haplotypes(draw)
        self.draw_true_breakpoints(draw)

    def draw_true_breakpoints(self, draw):
        b = self.box_size
        origin = self.haplotype_origin
        coordinates = sorted(self.x_coordinate_map.keys())
        for bp in self.true_breakpoints:
            # Find the smallest coordinate > position
            for position in coordinates:
                if position >= bp:
                    break
            x = self.x_coordinate_map[position]
            y1 = origin[0] + self.row_map[0] * b
            y2 = origin[1] + (self.row_map[len(self.row_map) - 1] + 1) * b
            draw.line([(x, y1), (x, y2)], fill="purple", width=3)

    def draw_base_haplotypes(self, draw):
        b = self.box_size
        origin = self.haplotype_origin
        for node in self.row_map.keys():
            y = self.row_map[node] * b + origin[1] + b / 2
            x = origin[0]
            draw.text((x - b, y), str(node), fill="black")
            x = self.width - self.right_padding
            mapped = (node - len(self.row_map) + 1) * -1
            if mapped < self.num_samples:
                mapped = (mapped - self.num_samples + 1) * -1
            draw.text((x + b / 4, y), str(mapped), fill="black")

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
        draw.rectangle([(x, y), (x + m * b, y + b)], outline=self.copying_outline_colour)
        for k in range(m):
            if parents[k] != -1:
                row = self.row_map[parents[k]]
                y = row * b + origin[1]
                x = k * b + origin[0]
                a = self.ancestors[parents[k], k]
                draw.rectangle([(x, y), (x + b, y + b)], fill=self.copy_colours[a])

        for position in breakpoints:
            x = self.x_coordinate_map[position]
            y1 = origin[0] + self.row_map[0] * b
            y2 = origin[1] + (self.row_map[len(self.row_map) - 1] + 1) * b
            draw.line([(x, y1), (x, y2)], fill="black")

        # Draw the positions of the sites.
        font = ImageFont.load_default()
        for site in self.original_ts.sites():
            label = "{} {:.6f}".format(site.id, site.position)
            img_txt = Image.new('L', font.getsize(label), color="white")
            draw_txt = ImageDraw.Draw(img_txt)
            draw_txt.text((0, 0), label, font=font)
            t = img_txt.rotate(90, expand=1)
            x = origin[0] + site.id * b
            y = origin[1] - b
            image.paste(t, (x, y))
        # print("Saving", filename)
        image.save(filename)

    def draw_copying_paths(self, pattern):
        N = self.num_ancestors + self.samples.shape[0]
        P = np.zeros((N, self.num_sites), dtype=int) - 1
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
        n = self.samples.shape[0]
        breakpoints = []
        for j in range(1, self.num_ancestors + n):
            for k in np.where(P[j][1:] != P[j][:-1])[0]:
                breakpoints.append(sites[k + 1].position)
            self.draw_copying_path(pattern.format(j - 1), j, P[j], breakpoints)


def visualise(
        ts, recombination_rate, error_rate, method="C", box_size=8,
        perfect_ancestors=False, path_compression=False, time_chunking=False):

    sample_data = tsinfer.SampleData.initialise(
        num_samples=ts.num_samples, sequence_length=ts.sequence_length,
        compressor=None)
    for v in ts.variants():
        sample_data.add_variant(v.site.position, v.alleles, v.genotypes)
    sample_data.finalise()

    ancestor_data = tsinfer.AncestorData.initialise(sample_data, compressor=None)
    if perfect_ancestors:
        tsinfer.build_simulated_ancestors(
            sample_data, ancestor_data, ts, time_chunking=time_chunking)
    else:
        tsinfer.build_ancestors(sample_data, ancestor_data, method=method)
    ancestor_data.finalise()

    ancestors_ts = tsinfer.match_ancestors(
        sample_data, ancestor_data, method=method, path_compression=path_compression,
        extended_checks=True)
    inferred_ts = tsinfer.match_samples(
        sample_data, ancestors_ts, method=method, simplify=False,
        path_compression=path_compression, extended_checks=True)

    prefix = "tmp__NOBACKUP__/"
    visualiser = Visualiser(
        ts, sample_data, ancestor_data, inferred_ts, box_size=box_size)
    visualiser.draw_copying_paths(os.path.join(prefix, "copying_{}.png"))

    # tsinfer.print_tree_pairs(ts, inferred_ts, compute_distances=False)
    inferred_ts = tsinfer.match_samples(
        sample_data, ancestors_ts, method=method, simplify=True,
        path_compression=False, stabilise_node_ordering=True)

    tsinfer.print_tree_pairs(ts, inferred_ts, compute_distances=True)
    sys.stdout.flush()


def run_viz(
        n, L, rate, seed, mutation_rate=0, method="C",
        perfect_ancestors=True, perfect_mutations=True, path_compression=False,
        time_chunking=True):
    recomb_map = msprime.RecombinationMap.uniform_map(
            length=L, rate=rate, num_loci=L)
    ts = msprime.simulate(
        n, recombination_map=recomb_map, random_seed=seed,
        model="smc_prime", mutation_rate=mutation_rate)
    if perfect_mutations:
        ts = tsinfer.insert_perfect_mutations(ts, delta=1/512)
    else:
        ts = tsinfer.strip_singletons(ts)
    print("num_sites = ", ts.num_sites)

    with open("tmp__NOBACKUP__/edges.svg", "w") as f:
        f.write(draw_edges(ts))
    with open("tmp__NOBACKUP__/ancestors.svg", "w") as f:
        f.write(draw_ancestors(ts))
    visualise(
        ts, 1e-9, 0, method=method, box_size=26, perfect_ancestors=perfect_ancestors,
        path_compression=path_compression, time_chunking=time_chunking)



def main():

    np.set_printoptions(linewidth=20000)
    np.set_printoptions(threshold=20000000)

    # import daiquiri
    # import sys
    # daiquiri.setup(level="DEBUG", outputs=(daiquiri.output.Stream(sys.stdout),))

    run_viz(
        6, 1000, 0.0005, 7, mutation_rate=0.06, perfect_ancestors=False,
        perfect_mutations=False, time_chunking=True, method="C", path_compression=False)

    # run_viz(15, 1000, 0.002, 2, method="C", perfect_ancestors=True)
    # check_inference(500, 1000000, 0.00002, 1, 100000, method="C")

if __name__ == "__main__":
    main()
