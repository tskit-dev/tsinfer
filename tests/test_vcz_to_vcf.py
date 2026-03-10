#
# Copyright (C) 2018-2026 University of Oxford
#
# This file is part of tsinfer.
#
# tsinfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tsinfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Validate that make_sample_vcz and ts_to_sample_vcz produce correct VCZ by
converting to VCF text via vcztools and comparing against expected output.
"""

import msprime
import numpy as np
from helpers import make_sample_vcz, ts_to_sample_vcz, vcz_to_vcf


def _parse_vcf_records(vcf_text: str) -> list[dict]:
    """Parse VCF data lines into a list of dicts keyed by column name."""
    records = []
    for line in vcf_text.splitlines():
        if line.startswith("#"):
            if line.startswith("#CHROM"):
                cols = line.lstrip("#").split("\t")
            continue
        fields = line.split("\t")
        rec = dict(zip(cols, fields))
        records.append(rec)
    return records


def _parse_gt(gt_str: str) -> list[int]:
    """Parse a GT field like '0/1' or '1|0' into a list of allele indices."""
    sep = "|" if "|" in gt_str else "/"
    return [int(x) for x in gt_str.split(sep)]


# ---------------------------------------------------------------------------
# Hand-constructed cases — expected VCF is known by inspection
# ---------------------------------------------------------------------------


class TestMakeSampleVczVcf:
    """VCF correctness tests for make_sample_vcz outputs."""

    def test_single_site_haploid(self):
        gt = np.array([[[0], [1]]], dtype=np.int8)
        vcz = make_sample_vcz(
            gt, [42], [["A", "T"]], ["A"], 100, sample_ids=np.array(["s0", "s1"])
        )
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        assert len(records) == 1
        r = records[0]
        assert r["CHROM"] == "1"
        assert r["POS"] == "42"
        assert r["REF"] == "A"
        assert r["ALT"] == "T"
        assert _parse_gt(r["s0"]) == [0]
        assert _parse_gt(r["s1"]) == [1]

    def test_two_sites_diploid(self):
        gt = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=np.int8)
        vcz = make_sample_vcz(
            gt,
            [100, 500],
            [["A", "T"], ["C", "G"]],
            ["A", "C"],
            1000,
            sample_ids=np.array(["NA001", "NA002"]),
        )
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)

        assert len(records) == 2

        r0 = records[0]
        assert r0["POS"] == "100"
        assert r0["REF"] == "A"
        assert r0["ALT"] == "T"
        assert _parse_gt(r0["NA001"]) == [0, 1]
        assert _parse_gt(r0["NA002"]) == [1, 0]

        r1 = records[1]
        assert r1["POS"] == "500"
        assert r1["REF"] == "C"
        assert r1["ALT"] == "G"
        assert _parse_gt(r1["NA001"]) == [1, 1]
        assert _parse_gt(r1["NA002"]) == [0, 0]

    def test_ancestral_allele_in_info(self):
        gt = np.array([[[0, 0]]], dtype=np.int8)
        vcz = make_sample_vcz(gt, [10], [["G", "A"]], ["G"], 100)
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        assert "ancestral_allele=G" in records[0]["INFO"]

    def test_three_alleles(self):
        # Triallelic site: REF=A, ALT=T,C
        gt = np.array([[[0, 2], [1, 2]]], dtype=np.int8)
        vcz = make_sample_vcz(
            gt, [200], [["A", "T", "C"]], ["A"], 1000, sample_ids=np.array(["s0", "s1"])
        )
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        assert records[0]["ALT"] == "T,C"
        assert _parse_gt(records[0]["s0"]) == [0, 2]
        assert _parse_gt(records[0]["s1"]) == [1, 2]

    def test_custom_contig_id(self):
        gt = np.array([[[0, 1]]], dtype=np.int8)
        vcz = make_sample_vcz(gt, [50], [["A", "T"]], ["A"], 500, contig_id="chr20")
        vcf = vcz_to_vcf(vcz)
        assert "##contig=<ID=chr20" in vcf
        records = _parse_vcf_records(vcf)
        assert records[0]["CHROM"] == "chr20"

    def test_header_contains_sample_ids(self):
        gt = np.array([[[0, 1], [0, 0]]], dtype=np.int8)
        vcz = make_sample_vcz(
            gt,
            [10],
            [["A", "T"]],
            ["A"],
            100,
            sample_ids=np.array(["SAMPLE_X", "SAMPLE_Y"]),
        )
        vcf = vcz_to_vcf(vcz)
        chrom_line = next(line for line in vcf.splitlines() if line.startswith("#CHROM"))
        assert "SAMPLE_X" in chrom_line
        assert "SAMPLE_Y" in chrom_line

    def test_missing_genotype_encoded_as_dot(self):
        # -1 in call_genotype should appear as '.' in VCF GT field
        gt = np.array([[[-1, 0]]], dtype=np.int8)
        vcz = make_sample_vcz(
            gt, [10], [["A", "T"]], ["A"], 100, sample_ids=np.array(["s0"])
        )
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        gt_parts = records[0]["s0"].split("/")
        assert "." in gt_parts


# ---------------------------------------------------------------------------
# ts_to_sample_vcz round-trip against tskit
# ---------------------------------------------------------------------------


class TestTsToSampleVczVcf:
    """VCF round-trip tests: convert ts → VCZ → VCF and compare with tskit."""

    def _sim(self, n=3, seq_len=10_000, seed=7):
        ts = msprime.sim_ancestry(
            n, sequence_length=seq_len, recombination_rate=1e-4, random_seed=seed
        )
        return msprime.sim_mutations(ts, rate=1e-3, random_seed=seed)

    def test_position_order(self):
        ts = self._sim()
        vcz = ts_to_sample_vcz(ts)
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        positions = [int(r["POS"]) for r in records]
        assert positions == sorted(positions)

    def test_record_count_matches_ts(self):
        ts = self._sim()
        vcz = ts_to_sample_vcz(ts)
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        assert len(records) == ts.num_sites

    def test_positions_match_ts(self):
        ts = self._sim()
        vcz = ts_to_sample_vcz(ts)
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        vcf_positions = [int(r["POS"]) for r in records]
        ts_positions = [int(s.position) for s in ts.sites()]
        assert vcf_positions == ts_positions

    def test_ref_alleles_match_ts(self):
        ts = self._sim()
        vcz = ts_to_sample_vcz(ts)
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        for rec, variant in zip(records, ts.variants()):
            assert rec["REF"] == variant.alleles[0]

    def test_genotypes_match_ts(self):
        ts = self._sim(n=2)
        vcz = ts_to_sample_vcz(ts)
        vcf = vcz_to_vcf(vcz)
        records = _parse_vcf_records(vcf)
        sample_cols = [f"tsk_{i}" for i in range(ts.num_individuals)]

        for rec, variant in zip(records, ts.variants()):
            # tskit gives flat haploid genotypes; reshape to (n_ind, 2)
            expected = variant.genotypes.reshape(ts.num_individuals, 2)
            for i, col in enumerate(sample_cols):
                observed = _parse_gt(rec[col])
                assert observed == list(expected[i]), (
                    f"site {rec['POS']}, sample {col}: {observed} != {list(expected[i])}"
                )

    def test_sample_count_in_header(self):
        ts = self._sim(n=4)
        vcz = ts_to_sample_vcz(ts)
        vcf = vcz_to_vcf(vcz)
        chrom_line = next(line for line in vcf.splitlines() if line.startswith("#CHROM"))
        sample_cols = chrom_line.split("\t")[9:]
        assert len(sample_cols) == ts.num_individuals
