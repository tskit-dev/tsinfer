import numpy as np
import pytest
import tskit

import tsinfer


def assert_variants_equal(vars1, vars2):
    assert vars1.num_sites == vars2.num_sites
    assert vars1.num_samples == vars2.num_samples
    for var1, var2 in zip(vars1.variants(), vars2.variants()):
        assert var1.alleles == var2.alleles
        assert np.all(var1.genotypes == var2.genotypes)


class TestExtend:
    @pytest.mark.parametrize("num_samples", range(1, 5))
    @pytest.mark.parametrize("num_sites", range(1, 5))
    def test_single_binary_haplotype_one_generation(self, num_samples, num_sites):
        with tsinfer.SampleData(sequence_length=num_sites) as sd:
            for j in range(num_sites):
                sd.add_site(j, [1] * num_samples)
        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend(np.arange(num_samples))
        assert_variants_equal(ts, sd)

    @pytest.mark.parametrize("num_samples", range(1, 5))
    @pytest.mark.parametrize("num_sites", range(1, 5))
    def test_single_binary_haplotype_two_epochs(self, num_samples, num_sites):
        with tsinfer.SampleData(sequence_length=num_sites) as sd:
            for j in range(num_sites):
                sd.add_site(j, [1] * num_samples)
        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend(np.arange(num_samples))

        extender = tsinfer.SequentialExtender(sd, ancestors_ts=ts)
        ts = extender.extend(np.arange(num_samples))
        assert ts.num_samples == 2 * num_samples
        assert np.all(ts.genotype_matrix() == 1)

    @pytest.mark.parametrize("k", range(1, 5))
    def test_single_binary_haplotype_k_generations(self, k):
        num_sites = 5
        num_samples = 4
        with tsinfer.SampleData(sequence_length=num_sites) as sd:
            for j in range(num_sites):
                sd.add_site(j, [1] * (num_samples * k))

        extender = tsinfer.SequentialExtender(sd)
        for _ in range(k):
            ts = extender.extend(np.arange(num_samples) * k)
        assert_variants_equal(ts, sd)

    @pytest.mark.parametrize("k", range(1, 5))
    def test_single_binary_haplotype_k_generations_two_epochs(self, k):
        num_sites = 5
        num_samples = 4
        with tsinfer.SampleData(sequence_length=num_sites) as sd:
            for j in range(num_sites):
                sd.add_site(j, [1] * (num_samples * k))

        extender = tsinfer.SequentialExtender(sd)
        for _ in range(k):
            ts = extender.extend(np.arange(num_samples) * k)
            # last num_samples should all have time 0
            assert np.all(ts.tables.nodes.time[ts.samples()[-num_samples:]] == 0)
        extender = tsinfer.SequentialExtender(sd, ts)
        for _ in range(k):
            ts = extender.extend(np.arange(num_samples) * k)
            assert np.all(ts.tables.nodes.time[ts.samples()[-num_samples:]] == 0)
        assert ts.num_samples == 2 * num_samples * k
        assert np.all(ts.genotype_matrix() == 1)

    def test_single_haplotype_4_alleles(self):
        num_sites = 3
        with tsinfer.SampleData(sequence_length=num_sites) as sd:
            for j in range(num_sites):
                sd.add_site(j, [0, 1, 2, 3], alleles="ACGT")

        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend(np.arange(4))
        assert_variants_equal(ts, sd)

    @pytest.mark.parametrize("k", range(4, 9))
    def test_single_site_4_alleles_rotating(self, k):
        genotypes = np.zeros(k, dtype=int)
        for j in range(k):
            genotypes[j] = j % 4
        with tsinfer.SampleData(sequence_length=1) as sd:
            sd.add_site(0, genotypes, alleles="ACGT")

        extender = tsinfer.SequentialExtender(sd)
        for j in range(k):
            ts = extender.extend([j])
        assert ts.num_mutations == 3
        assert_variants_equal(ts, sd)

    @pytest.mark.parametrize("num_generations", range(1, 5))
    @pytest.mark.parametrize("samples_per_generation", [1, 2, 13])
    @pytest.mark.parametrize("num_sites", [1, 4, 10, 100])
    def test_random_data(self, num_generations, samples_per_generation, num_sites):
        rng = np.random.default_rng(42)
        num_samples = num_generations * samples_per_generation
        with tsinfer.SampleData(sequence_length=num_sites) as sd:
            for j in range(num_samples):
                sd.add_individual(ploidy=1, metadata={"ind_id": j})
            for j in range(num_sites):
                genotypes = rng.integers(0, 4, size=num_samples)
                sd.add_site(j, genotypes, alleles="ACGT")

        extender = tsinfer.SequentialExtender(sd)
        offset = 0
        for _ in range(num_generations):
            next_offset = offset + samples_per_generation
            ts = extender.extend(np.arange(offset, next_offset))
            assert ts.num_samples == next_offset
            offset = next_offset

        assert ts.num_sites == sd.num_sites
        assert ts.num_samples == sd.num_samples
        for var1, var2 in zip(ts.variants(alleles=("A", "C", "G", "T")), sd.variants()):
            assert var1.alleles == var2.alleles
            assert np.all(var1.genotypes == var2.genotypes)
        for j, u in enumerate(ts.samples()):
            assert ts.node(u).metadata == {"ind_id": j}

    @pytest.mark.parametrize("num_generations", [1, 2, 5])
    @pytest.mark.parametrize("samples_per_generation", [1, 2, 13])
    @pytest.mark.parametrize("num_epochs", range(1, 4))
    def test_random_data_multi_epoch_fixed_sites(
        self, num_generations, samples_per_generation, num_epochs
    ):
        rng = np.random.default_rng(142)
        num_sites = 10
        ancestors_ts = None
        total_samples = num_epochs * num_generations * samples_per_generation
        G = rng.integers(0, 4, size=(total_samples, num_sites))
        for epoch in range(num_epochs):
            num_samples = num_generations * samples_per_generation
            epoch_start = epoch * num_samples
            genotypes = G[epoch_start : epoch_start + num_samples]
            with tsinfer.SampleData(sequence_length=num_sites) as sd:
                # Store the genotypes with the individual metadata so
                # we can compare later.
                for j in range(num_samples):
                    sd.add_individual(
                        ploidy=1,
                        metadata={
                            "epoch": epoch,
                            "ind_id": (epoch, j),
                            "genotypes": list(map(int, genotypes[j])),
                        },
                    )
                for j in range(num_sites):
                    sd.add_site(j, genotypes[:, j], alleles="ACGT")
            extender = tsinfer.SequentialExtender(sd, ancestors_ts)
            offset = 0
            for _ in range(num_generations):
                next_offset = offset + samples_per_generation
                ts = extender.extend(np.arange(offset, next_offset))
                offset = next_offset
            assert ts.num_sites == num_sites
            assert ts.num_samples == num_samples * (1 + epoch)
            ancestors_ts = ts
        # Do we round-trip all the data?
        for j, u in enumerate(ts.samples()):
            node = ts.node(u)
            md = node.metadata
            assert md["ind_id"] == [j // num_samples, j % num_samples]
            assert np.array_equal(md["genotypes"], G[j])

    def test_single_sample_metadata(self):
        with tsinfer.SampleData(sequence_length=1) as sd:
            sd.add_individual(ploidy=1, metadata={"x": 1})
            sd.add_site(0, [1])
        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend([0])
        assert_variants_equal(ts, sd)
        assert ts.node(ts.samples()[0]).metadata == {"x": 1}

    @pytest.mark.parametrize("num_generations", range(1, 5))
    def test_stick(self, num_generations):
        # We have a stick tree where the single mutation for a given site
        # happens on one branch and they accumulate over time.
        H = np.zeros((num_generations, num_generations), dtype=int)
        a = np.zeros(num_generations, dtype=int)
        for j in range(num_generations):
            a[j] = 1
            H[j] = a
        with tsinfer.SampleData(sequence_length=num_generations) as sd:
            for j in range(num_generations):
                sd.add_site(j, H[:, j])
        extender = tsinfer.SequentialExtender(sd)
        for j in range(num_generations):
            ts = extender.extend([j])
            assert ts.num_samples == j + 1
        assert ts.num_mutations == num_generations
        assert ts.num_edges == num_generations + 1

    @pytest.mark.parametrize("num_generations", range(1, 5))
    def test_all_zeros(self, num_generations):
        # all the haplotypes are 0s and should just copy directly from
        # the same root.
        a = np.zeros(2 * num_generations, dtype=int)
        with tsinfer.SampleData(sequence_length=num_generations) as sd:
            sd.add_site(0, a)
        extender = tsinfer.SequentialExtender(sd)
        for j in range(num_generations):
            ts = extender.extend([2 * j, 2 * j + 1])
            # assert ts.num_samples == 2 * j + 1
        assert ts.num_mutations == 0
        assert ts.num_trees == 1
        tree = ts.first()
        parents = {tree.parent(u) for u in ts.samples()}
        assert len(parents) == 1

    def test_all_zeros_time_increment(self):
        a = np.zeros(2 * 2, dtype=int)
        with tsinfer.SampleData(sequence_length=2) as sd:
            sd.add_site(0, a)
        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend([0, 1], time_increment=5)
        np.testing.assert_array_equal(ts.nodes_time, [6, 5, 0, 0])
        ts = extender.extend([2, 3], time_increment=2)
        np.testing.assert_array_equal(ts.nodes_time, [8, 7, 2, 2, 0, 0])

        extender = tsinfer.SequentialExtender(sd, ancestors_ts=ts)
        ts = extender.extend([0, 1], time_increment=3)
        np.testing.assert_array_equal(ts.nodes_time, [11, 10, 5, 5, 3, 3, 0, 0])
        assert ts.time_units == tskit.TIME_UNITS_UNCALIBRATED

        ts = extender.extend([0, 1], time_increment=0.1)
        np.testing.assert_array_equal(
            ts.nodes_time, [11.1, 10.1, 5.1, 5.1, 3.1, 3.1, 0.1, 0.1, 0, 0]
        )
        assert ts.time_units == tskit.TIME_UNITS_UNCALIBRATED

    def test_all_zeros_time_units(self):
        a = np.zeros(2 * 2, dtype=int)
        with tsinfer.SampleData(sequence_length=2) as sd:
            sd.add_site(0, a)
        time_units = "days_ago"
        extender = tsinfer.SequentialExtender(sd, time_units=time_units)
        ts = extender.extend([0, 1])
        assert ts.time_units == time_units
        ts = extender.extend([2, 3])
        assert ts.time_units == time_units

        # Specifying different time_units gives an error
        with pytest.raises(ValueError, match="time_units"):
            extender = tsinfer.SequentialExtender(sd, ancestors_ts=ts)
        with pytest.raises(ValueError, match="time_units"):
            extender = tsinfer.SequentialExtender(
                sd, ancestors_ts=ts, time_units="stuff"
            )

        extender = tsinfer.SequentialExtender(
            sd, ancestors_ts=ts, time_units=time_units
        )
        ts = extender.extend([0, 1])
        assert ts.time_units == time_units
        ts = extender.extend([2, 3])
        assert ts.time_units == time_units


class TestExtendPathCompression:
    def example(self):
        with tsinfer.SampleData(sequence_length=4) as sd:
            sd.add_site(0, [0, 1, 1, 1])
            sd.add_site(1, [0, 1, 1, 1])
            sd.add_site(2, [1, 0, 1, 1])
            sd.add_site(3, [1, 0, 2, 1], alleles=("0", "1", "2"))
            return sd

    def test_simple_path_compression_case(self):
        sd = self.example()
        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend([0, 1])
        # NOTE we'd really like to get rid of this vestigial node 0 but
        # the low-level code won't work without it, so until it's
        # gone it's simplest to just live with it and update the test
        # cases later.

        # 2.00┊  0  ┊
        #     ┊  ┃  ┊
        # 1.00┊  1  ┊
        #     ┊ ┏┻┓ ┊
        # 0.00┊ 2 3 ┊
        #     0     4
        assert ts.num_trees == 1
        assert ts.num_nodes == 4
        assert ts.first().parent_dict == {2: 1, 3: 1, 1: 0}

        ts = extender.extend([2, 3])
        # 3.00┊   0   ┊   0   ┊
        #     ┊   ┃   ┊   ┃   ┊
        # 2.00┊   1   ┊   1   ┊
        #     ┊ ┏━┻┓  ┊ ┏━┻┓  ┊
        # 1.00┊ 2  3  ┊ 3  2  ┊
        #     ┊    ┃  ┊    ┃  ┊
        # 1.00┊    6  ┊    6  ┊
        #     ┊   ┏┻┓ ┊   ┏┻┓ ┊
        # 0.00┊   4 5 ┊   4 5 ┊
        #     0       2       4

        assert ts.num_trees == 2
        assert ts.num_nodes == 7
        assert ts.node(6).flags == tsinfer.NODE_IS_PC_ANCESTOR
        assert ts.first().parent_dict == {2: 1, 3: 1, 1: 0, 6: 3, 4: 6, 5: 6}
        assert ts.last().parent_dict == {2: 1, 3: 1, 1: 0, 6: 2, 4: 6, 5: 6}
        assert_variants_equal(ts, sd)

    def test_simple_path_compression_case_no_pc(self):
        sd = self.example()

        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend([0, 1])
        assert ts.num_trees == 1
        assert ts.num_nodes == 4
        assert ts.first().parent_dict == {2: 1, 3: 1, 1: 0}

        ts = extender.extend([2, 3], path_compression=False)
        # 3.00┊   0   ┊   0   ┊
        #     ┊   ┃   ┊   ┃   ┊
        # 2.00┊   1   ┊   1   ┊
        #     ┊ ┏━┻┓  ┊ ┏━┻┓  ┊
        # 1.00┊ 2  3  ┊ 3  2  ┊
        #     ┊   ┏┻┓ ┊   ┏┻┓ ┊
        # 0.00┊   4 5 ┊   4 5 ┊
        #     0       2       4
        assert ts.num_trees == 2
        assert ts.num_nodes == 6
        assert ts.first().parent_dict == {2: 1, 3: 1, 1: 0, 4: 3, 5: 3}
        assert ts.last().parent_dict == {2: 1, 3: 1, 1: 0, 4: 2, 5: 2}
        assert_variants_equal(ts, sd)


class TestExtendIdenticalSequences:
    def test_single_site_one_generation(self):
        with tsinfer.SampleData(sequence_length=1) as sd:
            sd.add_site(0, [1, 1])
        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend([0, 1])

        # 2.00┊  0  ┊
        #     ┊  ┃  ┊
        # 1.00┊  1  ┊
        #     ┊  ┃  ┊
        # 0.01┊  4  ┊
        #     ┊ ┏┻┓ ┊
        # 0.00┊ 2 3 ┊
        #     0     1
        assert ts.num_trees == 1
        assert ts.num_nodes == 5
        assert ts.first().parent_dict == {2: 4, 3: 4, 4: 1, 1: 0}
        assert ts.node(4).flags == tsinfer.NODE_IS_IDENTICAL_SAMPLE_ANCESTOR

        assert_variants_equal(ts, sd)

    def test_two_haplotypes_one_generation(self):
        alleles = ("A", "C", "G")
        with tsinfer.SampleData(sequence_length=2) as sd:
            sd.add_site(0, [1, 1, 2, 2], alleles=alleles)
            sd.add_site(1, [1, 1, 2, 2], alleles=alleles)
        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend([0, 1, 2, 3])

        # 2.00┊    0    ┊
        #     ┊    ┃    ┊
        # 1.00┊    1    ┊
        #     ┊  ┏━┻━┓  ┊
        # 0.00┊  6   7  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 2 3 4 5 ┊
        #     0         2
        assert ts.num_trees == 1
        assert ts.num_nodes == 8

        assert ts.first().parent_dict == {2: 6, 3: 6, 4: 7, 5: 7, 6: 1, 7: 1, 1: 0}
        assert ts.node(6).flags == tsinfer.NODE_IS_IDENTICAL_SAMPLE_ANCESTOR
        assert ts.node(7).flags == tsinfer.NODE_IS_IDENTICAL_SAMPLE_ANCESTOR

        assert_variants_equal(ts, sd)

    def test_two_haplotypes_two_generations(self):
        alleles = ("A", "C", "G")
        with tsinfer.SampleData(sequence_length=2) as sd:
            sd.add_site(0, [1, 1, 2, 2, 2, 2], alleles=alleles)
            sd.add_site(1, [1, 1, 2, 2, 2, 2], alleles=alleles)
        extender = tsinfer.SequentialExtender(sd)
        ts = extender.extend([0, 1, 2, 3])
        ts = extender.extend([4, 5])
        # We correctly see that there was a pre-existing exact match for
        # this haplotype and match against it.
        assert_variants_equal(ts, sd)
        # 3.00┊     0       ┊
        #     ┊     ┃       ┊
        # 2.00┊     1       ┊
        #     ┊  ┏━━┻━━┓    ┊
        # 1.00┊  6     7    ┊
        #     ┊ ┏┻┓ ┏━┳┻┳━┓ ┊
        # 1.00┊ 2 3 4 5 ┃ ┃ ┊
        #     ┊         ┃ ┃ ┊
        # 0.00┊         8 9 ┊
        #     0             2
        assert ts.first().parent_dict == {
            2: 6,
            3: 6,
            4: 7,
            5: 7,
            6: 1,
            7: 1,
            1: 0,
            8: 7,
            9: 7,
        }
        assert ts.node(6).flags == tsinfer.NODE_IS_IDENTICAL_SAMPLE_ANCESTOR
        assert ts.node(7).flags == tsinfer.NODE_IS_IDENTICAL_SAMPLE_ANCESTOR


class TestExtendLsParameters:
    def run(self, num_mismatches=None):

        with tsinfer.SampleData(sequence_length=6) as sd:
            sd.add_site(0, [0, 1, 1])
            sd.add_site(1, [1, 0, 1])
            sd.add_site(2, [0, 1, 1])
            sd.add_site(3, [1, 0, 1])
            sd.add_site(4, [0, 1, 1])
            sd.add_site(5, [1, 0, 1])

        extender = tsinfer.SequentialExtender(sd)
        for j in range(3):
            ts = extender.extend(
                [j],
                num_mismatches=num_mismatches,
                num_threads=0,
                engine=tsinfer.C_ENGINE,
            )
        return ts

    @pytest.mark.parametrize("mismatches", [None, 0, 0.5])
    def test_all_recombination(self, mismatches):
        ts = self.run(mismatches)
        # We have a recombination at every site and exactly one mutation per site.
        assert ts.num_trees == ts.num_sites
        assert ts.num_mutations == ts.num_sites

    @pytest.mark.parametrize("mismatches", [3, 3.1, 4, 100, 1000])
    def test_no_recombination(self, mismatches):
        ts = self.run(mismatches)
        assert ts.num_trees == 1
        assert ts.num_mutations == 9

    def test_one_mismatch(self):
        ts = self.run(1)
        # This is all quite tricky - not quite sure what to expect. Keep
        # lint happy for now
        assert ts is not None
        # print(ts.tables)
        # print()
        # print(ts.draw_text())
        # print(ts.tables.mutations[ts.tables.mutations.node == 4])
        # print(ts.tables.mutations)
