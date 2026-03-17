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

import sys

import bio2zarr.tskit as bzt
import msprime
import numpy as np
import zarr
from helpers import make_ancestor_vcz, make_sample_vcz, ts_to_sample_vcz, vcz_to_vcf


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


# ---------------------------------------------------------------------------
# bio2zarr equivalence
# ---------------------------------------------------------------------------


class TestTsToSampleVczMatchesBio2zarr:
    """
    Verify that ts_to_sample_vcz produces the same core arrays as bio2zarr's
    tskit convert. bio2zarr is written to a temporary directory; we compare
    against the in-memory group from ts_to_sample_vcz.

    Arrays compared: call_genotype, variant_position, variant_allele,
    sample_id, contig_id, contig_length, variant_contig.

    Arrays intentionally not compared: call_genotype_mask, call_genotype_phased,
    region_index, variant_length (all bio2zarr-specific extras we don't produce).
    """

    def _sim(self, n=3, seq_len=5_000, seed=42):
        ts = msprime.sim_ancestry(
            n, sequence_length=seq_len, recombination_rate=1e-4, random_seed=seed
        )
        return msprime.sim_mutations(ts, rate=1e-3, random_seed=seed)

    def test_call_genotype(self, tmp_path):
        ts = self._sim()
        bzt.convert(ts, tmp_path / "ref.vcz")
        ref = zarr.open(str(tmp_path / "ref.vcz"), mode="r")
        ours = ts_to_sample_vcz(ts)
        np.testing.assert_array_equal(ours["call_genotype"][:], ref["call_genotype"][:])

    def test_variant_position(self, tmp_path):
        ts = self._sim()
        bzt.convert(ts, tmp_path / "ref.vcz")
        ref = zarr.open(str(tmp_path / "ref.vcz"), mode="r")
        ours = ts_to_sample_vcz(ts)
        np.testing.assert_array_equal(
            ours["variant_position"][:], ref["variant_position"][:]
        )

    def test_variant_allele(self, tmp_path):
        ts = self._sim()
        bzt.convert(ts, tmp_path / "ref.vcz")
        ref = zarr.open(str(tmp_path / "ref.vcz"), mode="r")
        ours = ts_to_sample_vcz(ts)
        # Both should have the same shape and values; empty strings pad unused alleles
        assert ours["variant_allele"].shape == ref["variant_allele"].shape
        np.testing.assert_array_equal(
            ours["variant_allele"][:], ref["variant_allele"][:]
        )

    def test_sample_id(self, tmp_path):
        ts = self._sim()
        bzt.convert(ts, tmp_path / "ref.vcz")
        ref = zarr.open(str(tmp_path / "ref.vcz"), mode="r")
        ours = ts_to_sample_vcz(ts)
        np.testing.assert_array_equal(ours["sample_id"][:], ref["sample_id"][:])

    def test_contig_id(self, tmp_path):
        ts = self._sim()
        bzt.convert(ts, tmp_path / "ref.vcz")
        ref = zarr.open(str(tmp_path / "ref.vcz"), mode="r")
        ours = ts_to_sample_vcz(ts)
        np.testing.assert_array_equal(ours["contig_id"][:], ref["contig_id"][:])

    def test_contig_length(self, tmp_path):
        ts = self._sim()
        bzt.convert(ts, tmp_path / "ref.vcz")
        ref = zarr.open(str(tmp_path / "ref.vcz"), mode="r")
        ours = ts_to_sample_vcz(ts)
        np.testing.assert_array_equal(ours["contig_length"][:], ref["contig_length"][:])

    def test_variant_contig(self, tmp_path):
        ts = self._sim()
        bzt.convert(ts, tmp_path / "ref.vcz")
        ref = zarr.open(str(tmp_path / "ref.vcz"), mode="r")
        ours = ts_to_sample_vcz(ts)
        np.testing.assert_array_equal(
            ours["variant_contig"][:], ref["variant_contig"][:]
        )

    def test_multiple_seeds(self, tmp_path):
        """Spot-check genotypes across several random tree sequences."""
        for seed in [1, 7, 13, 99]:
            ts = self._sim(seed=seed)
            vcz_path = tmp_path / f"ref_{seed}.vcz"
            bzt.convert(ts, vcz_path)
            ref = zarr.open(str(vcz_path), mode="r")
            ours = ts_to_sample_vcz(ts)
            np.testing.assert_array_equal(
                ours["call_genotype"][:],
                ref["call_genotype"][:],
                err_msg=f"call_genotype mismatch for seed={seed}",
            )


# ---------------------------------------------------------------------------
# Ancestor VCZ → VCF round-trip
# ---------------------------------------------------------------------------


class TestAncestorVczToVcf:
    """Verify that ancestor VCZ stores with contig arrays produce valid VCF."""

    def test_ancestor_vcz_to_vcf(self):
        gt = np.array([[[0, 1]], [[1, 0]]], dtype=np.int8)
        vcz = make_ancestor_vcz(
            genotypes=gt,
            positions=np.array([100, 200]),
            alleles=np.array([["A", "T"], ["C", "G"]]),
            times=np.array([0.5, 0.3]),
            focal_positions=np.array([[100], [200]], dtype=np.int32),
            sequence_intervals=np.array([[100, 201]], dtype=np.int32),
            contig_id="chr1",
            contig_length=5000,
        )
        vcf = vcz_to_vcf(vcz)
        assert "##contig=<ID=chr1" in vcf
        records = _parse_vcf_records(vcf)
        assert len(records) == 2
        assert records[0]["CHROM"] == "chr1"
        assert records[1]["CHROM"] == "chr1"
        assert records[0]["POS"] == "100"
        assert records[1]["POS"] == "200"

    def test_infer_ancestors_vcztools_view(self, tmp_path):
        """vcztools view on a filesystem-backed ancestor VCZ produces valid VCF."""
        import subprocess

        from tsinfer.ancestors import infer_ancestors
        from tsinfer.config import AncestorsConfig, Source

        n_sites = 20
        n_samples = 6
        rng = np.random.RandomState(42)
        gt = rng.randint(0, 2, size=(n_sites, n_samples, 1)).astype(np.int8)
        positions = np.arange(100, 100 + n_sites * 100, 100, dtype=np.int32)
        store = make_sample_vcz(
            gt,
            positions,
            np.array([["A", "T"]] * n_sites),
            np.array(["A"] * n_sites),
            100_000,
            contig_id="chr2",
        )
        anc_path = tmp_path / "ancestors.vcz"
        cfg = AncestorsConfig(
            path=anc_path,
            sources=["test"],
            variants_chunk_size=5,
            samples_chunk_size=3,
        )
        infer_ancestors(Source(path=store, name="test"), cfg)

        result = subprocess.run(
            [sys.executable, "-m", "vcztools", "view", str(anc_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "##contig=<ID=chr2" in result.stdout
        records = _parse_vcf_records(result.stdout)
        assert len(records) > 0
        assert all(r["CHROM"] == "chr2" for r in records)
