import itertools

import msprime
import numpy as np
import tskit

import tsinfer


class Sequence:
    def __init__(self, haplotype):
        self.full_haplotype = haplotype


def run_matches(ts, positions, sequences):
    match_indexes = tsinfer.MatcherIndexes(ts)
    recombination = np.zeros(ts.num_sites) + 1e-9
    mismatch = np.zeros(ts.num_sites)
    matcher = tsinfer.AncestorMatcher2(
        match_indexes, recombination=recombination, mismatch=mismatch
    )
    sites_index = np.searchsorted(positions, ts.sites_position)
    assert np.all(positions[sites_index] == ts.sites_position)
    sites_in_ts = np.zeros(len(positions), dtype=bool)
    sites_in_ts[sites_index] = True
    results = []
    for seq in sequences:
        m = matcher.find_match(seq.full_haplotype[sites_in_ts])
        h = seq.full_haplotype.copy()
        h[sites_in_ts] = 0
        focal_sites = np.where(h != 0)[0]
        results.append((m, focal_sites))
    return results


def insert_matches(tables, time, all_positions, matches):
    ts_sites_position = tables.sites.position
    added_sites = {}
    for m, new_sites in matches:
        u = tables.nodes.add_row(time=time, flags=0)
        for left, right, parent in m.path:
            tables.edges.add_row(left, right, parent, u)
        for site_index in new_sites:
            if site_index not in added_sites:
                s = tables.sites.add_row(all_positions[site_index], "0")
                added_sites[site_index] = s
            tables.mutations.add_row(
                site=added_sites[site_index], node=u, derived_state="1"
            )
        # print(tables)
        # TODO check the matched haplotype for any mutations too.
    # print(tables)
    tables.sort()
    ts = tables.tree_sequence()
    return ts


def match_ancestors(ancestor_data):
    tables = tskit.TableCollection(ancestor_data.sequence_length)

    all_positions = ancestor_data.sites_position[:]

    ancestors = ancestor_data.ancestors()
    # Discard the "ultimate-ultimate ancestor"
    next(ancestors)
    ultimate_ancestor = next(ancestors)
    assert np.all(ultimate_ancestor.full_haplotype == 0)
    tables.nodes.add_row(time=ultimate_ancestor.time + 1)
    tables.nodes.add_row(time=ultimate_ancestor.time)
    tables.edges.add_row(0, tables.sequence_length, 0, 1)
    ts = tables.tree_sequence()

    # TODO We don't want to use the focal sites, so we need to keep track
    # of when each site gets new variation, or at least keep track of all
    # the sites that are entirely ancestral, so we only add sites into the
    # ts as we see variation at them.

    for time, group in itertools.groupby(ancestors, key=lambda a: a.time):
        # print("EPOCH", time)
        group = list(group)
        matches = run_matches(ts, all_positions, group)
        ts = insert_matches(tables, time, all_positions, matches)
        # print(ts.draw_text())
    return ts


def match_samples(ts, sample_data):
    all_positions = sample_data.sites_position[:]
    sequences = [Sequence(h) for _, h in sample_data.haplotypes()]
    matches = run_matches(ts, all_positions, sequences)
    ts = insert_matches(ts.dump_tables(), 0, all_positions, matches)
    tables = ts.dump_tables()
    # We can have sites that are monomorphic for the ancestral state.
    missing_sites = set(all_positions) - set(ts.sites_position)
    for pos in missing_sites:
        tables.sites.add_row(pos, ancestral_state="0")
    tables.sort()
    flags = tables.nodes.flags
    flags[-len(sequences) :] = 1
    tables.nodes.flags = flags
    print(tables)
    return tables.tree_sequence()


if __name__ == "__main__":
    for seed in range(1, 100):
        ts = msprime.sim_ancestry(
            15,
            population_size=1e4,  # recombination_rate=1e-10,
            sequence_length=1_000_000,
            random_seed=seed,
        )
        print(seed)
        ts_orig = msprime.sim_mutations(
            ts, rate=1e-8, random_seed=seed, model=msprime.BinaryMutationModel()
        )
        print(ts_orig)
        print(ts_orig.num_sites, ts_orig.num_mutations)
        # assert ts_orig.num_sites == ts_orig.num_mutations

        # with tsinfer.SampleData(sequence_length=7, path="tmp.samples") as sample_data:
        #     for _ in range(5):
        #         sample_data.add_individual(time=0, ploidy=1)
        #     sample_data.add_site(0, [0, 1, 0, 0, 0], ["A", "T"])
        #     sample_data.add_site(1, [0, 0, 0, 1, 1], ["G", "C"])
        #     sample_data.add_site(2, [0, 1, 1, 0, 0], ["C", "A"])
        #     sample_data.add_site(3, [0, 1, 1, 0, 0], ["G", "C"])
        #     sample_data.add_site(4, [0, 0, 0, 1, 1], ["A", "C"])
        #     sample_data.add_site(5, [0, 1, 0, 0, 0], ["T", "G"])
        #     sample_data.add_site(6, [1, 1, 1, 1, 0], ["T", "G"])
        sample_data = tsinfer.SampleData.from_tree_sequence(ts_orig)
        # print(sample_data)

        ad = tsinfer.generate_ancestors(sample_data)

        ts = match_ancestors(ad)
        # print(sample_data)
        ts = match_samples(ts, sample_data)
        print(ts.draw_text())
        # print(ts.genotype_matrix())
        # print(ts_orig.genotype_matrix())
        np.testing.assert_array_equal(ts.genotype_matrix(), ts_orig.genotype_matrix())
