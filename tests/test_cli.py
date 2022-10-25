#
# Copyright (C) 2018 University of Oxford
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
Tests for the tsinfer CLI.
"""
import io
import json
import os.path
import pathlib
import sys
import tempfile
import unittest
import unittest.mock as mock

import msprime
import numpy as np
import pytest
import tskit

import tsinfer
import tsinfer.__main__ as main
import tsinfer.cli as cli
import tsinfer.exceptions as exceptions


def capture_output(func, *args, **kwargs):
    """
    Runs the specified function and arguments, and returns the
    tuple (stdout, stderr) as strings.
    """
    buffer_class = io.BytesIO
    if sys.version_info[0] == 3:
        buffer_class = io.StringIO
    stdout = sys.stdout
    sys.stdout = buffer_class()
    stderr = sys.stderr
    sys.stderr = buffer_class()

    try:
        func(*args, **kwargs)
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
    finally:
        sys.stdout.close()
        sys.stdout = stdout
        sys.stderr.close()
        sys.stderr = stderr
    return stdout_output, stderr_output


class TestMain:
    """
    Simple tests for the main function.
    """

    def test_cli_main(self):
        with mock.patch("argparse.ArgumentParser.parse_args") as mocked_parse:
            cli.tsinfer_main()
            mocked_parse.assert_called_once_with(None)

    def test_main(self):
        with mock.patch("argparse.ArgumentParser.parse_args") as mocked_parse:
            main.main()
            mocked_parse.assert_called_once_with(None)


class TestDefaultPaths:
    """
    Tests for the default path creation routines.
    """

    def test_get_default_path(self):
        # The second argument is ignored if the input path is specified.
        for path in ["a", "a/b/c", "a.stuff"]:
            assert cli.get_default_path(path, "a", "b") == path
        assert cli.get_default_path(None, "a", ".x") == "a.x"
        assert cli.get_default_path(None, "a.y", ".z") == "a.z"
        assert cli.get_default_path(None, "a/b/c/a.y", ".z") == "a/b/c/a.z"

    def test_get_ancestors_path(self):
        assert cli.get_ancestors_path(None, "a") == "a.ancestors"

    def test_get_ancestors_trees_path(self):
        assert cli.get_ancestors_trees_path(None, "a") == "a.ancestors.trees"

    def test_get_output_trees_path(self):
        assert cli.get_output_trees_path(None, "a") == "a.trees"


class TestCli(unittest.TestCase):
    """
    Parent of all CLI test cases.
    """

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="tsinfer_cli_test")
        self.sample_file = str(pathlib.Path(self.tempdir.name, "input-data.samples"))
        self.ancestor_file = str(
            pathlib.Path(self.tempdir.name, "input-data.ancestors")
        )
        self.ancestor_trees = str(
            pathlib.Path(self.tempdir.name, "input-data.ancestors.trees")
        )
        self.output_trees = str(pathlib.Path(self.tempdir.name, "input-data.trees"))
        self.input_ts = msprime.simulate(
            10, length=10, mutation_rate=1, recombination_rate=1, random_seed=10
        )
        sample_data = tsinfer.SampleData(
            sequence_length=self.input_ts.sequence_length, path=self.sample_file
        )
        for var in self.input_ts.variants():
            sample_data.add_site(var.site.position, var.genotypes, var.alleles)
        sample_data.finalise()
        tsinfer.generate_ancestors(sample_data, path=self.ancestor_file, chunk_size=10)
        ancestor_data = tsinfer.load(self.ancestor_file)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        ancestors_ts.dump(self.ancestor_trees)
        ts = tsinfer.match_samples(sample_data, ancestors_ts)
        ts.dump(self.output_trees)
        sample_data.close()

    # Need to mock out setup_logging here or we spew logging to the console
    # in later tests.
    @mock.patch("tsinfer.cli.setup_logging")
    def run_command(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tsinfer_main, command)
        assert stderr == ""
        assert stdout == ""
        assert mock_setup_logging.called


class TestCommandsDefaults(TestCli):
    """
    Tests that the basic commands work if we provide the default arguments.
    """

    def verify_output(self, output_path):
        output_trees = tskit.load(output_path)
        assert output_trees.num_samples == self.input_ts.num_samples
        assert output_trees.sequence_length == self.input_ts.sequence_length
        assert output_trees.num_sites == self.input_ts.num_sites
        assert output_trees.num_sites > 1
        assert np.array_equal(
            output_trees.genotype_matrix(), self.input_ts.genotype_matrix()
        )

    def test_infer(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        self.run_command(["infer", self.sample_file, "-O", output_trees])
        self.verify_output(output_trees)

    def test_infer_from_ts(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        self.run_command(["infer", self.sample_file, "-O", output_trees])
        self.assertRaisesRegex(
            exceptions.FileFormatError,
            "from_tree_sequence",
            self.run_command,
            ["infer", output_trees],
        )

    def test_infer_bad_file(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        with open(output_trees, "w") as bad_file:
            bad_file.write("xxx")
        self.assertRaisesRegex(
            exceptions.FileFormatError,
            "Unknown file format",
            self.run_command,
            ["infer", output_trees],
        )

    @pytest.mark.skipif(
        sys.platform == "win32", reason="windows simultaneous file permissions issue"
    )
    def test_nominal_chain(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        self.run_command(["generate-ancestors", self.sample_file])
        self.run_command(["match-ancestors", self.sample_file])
        self.run_command(["match-samples", self.sample_file, "-O", output_trees])
        self.verify_output(output_trees)

    def test_verify(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        self.run_command(["infer", self.sample_file, "-O", output_trees])
        self.run_command(["verify", self.sample_file, output_trees])

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="windows simultaneous file access permissions issue",
    )
    def test_augment_ancestors(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        augmented_ancestors = os.path.join(
            self.tempdir.name, "augmented_ancestors.trees"
        )
        self.run_command(["generate-ancestors", self.sample_file])
        self.run_command(["match-ancestors", self.sample_file])
        self.run_command(["augment-ancestors", self.sample_file, augmented_ancestors])
        self.run_command(
            [
                "match-samples",
                self.sample_file,
                "-O",
                output_trees,
                "-A",
                augmented_ancestors,
            ]
        )
        self.verify_output(output_trees)


class TestCommandsExtra(TestCli):
    """
    Test miscellaneous extra options for standard commands
    """

    def test_filenames_without_keeping_intermediates(self):
        output_anc = os.path.join(self.tempdir.name, "test1")
        output_anc_ts = os.path.join(self.tempdir.name, "test2")
        with pytest.raises(ValueError, match="--keep-intermediates"):
            self.run_command(["infer", self.sample_file, "-a", output_anc])
        with pytest.raises(ValueError, match="--keep-intermediates"):
            self.run_command(["infer", self.sample_file, "-A", output_anc_ts])

    def test_keep_intermediates(self):
        output_anc = os.path.join(self.tempdir.name, "test1")
        output_anc_ts = os.path.join(self.tempdir.name, "test2")
        self.run_command(
            [
                "infer",
                self.sample_file,
                "--keep-intermediates",
                "-a",
                output_anc,
                "-A",
                output_anc_ts,
            ]
        )
        assert os.path.exists(output_anc)
        ancestors = tsinfer.load(output_anc)
        assert ancestors.num_ancestors > 0

        assert os.path.exists(output_anc_ts)
        anc_ts = tskit.load(output_anc_ts)
        assert anc_ts.num_samples > 0


class TestProgress(TestCli):
    """
    Tests that we get some output when we use the progress bar.
    """

    # Need to mock out setup_logging here or we spew logging to the console
    # in later tests.
    @mock.patch("tsinfer.cli.setup_logging")
    def run_command(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tsinfer_main, command + ["--progress"])
        assert len(stderr) > 0
        assert stdout == ""
        assert mock_setup_logging.called

    def test_infer(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        self.run_command(["infer", self.sample_file, "-O", output_trees])

    @pytest.mark.skipif(
        sys.platform == "win32", reason="windows simultaneous file permissions issue"
    )
    def test_nominal_chain(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        self.run_command(["generate-ancestors", self.sample_file])
        self.run_command(["match-ancestors", self.sample_file])
        self.run_command(["match-samples", self.sample_file, "-O", output_trees])

    def test_verify(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        self.run_command(["infer", self.sample_file, "-O", output_trees])
        self.run_command(["verify", self.sample_file, output_trees])

    @pytest.mark.skipif(
        sys.platform == "win32", reason="windows simultaneous file permissions issue"
    )
    def test_augment_ancestors(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        augmented_ancestors = os.path.join(
            self.tempdir.name, "augmented_ancestors.trees"
        )
        self.run_command(["generate-ancestors", self.sample_file])
        self.run_command(["match-ancestors", self.sample_file])
        self.run_command(["augment-ancestors", self.sample_file, augmented_ancestors])
        self.run_command(
            [
                "match-samples",
                self.sample_file,
                "-O",
                output_trees,
                "-A",
                augmented_ancestors,
            ]
        )


class TestProvenance(TestCli):
    """
    Tests that we get provenance in the output trees
    """

    # Need to mock out setup_logging here or we spew logging to the console
    # in later tests.
    @mock.patch("tsinfer.cli.setup_logging")
    def run_command(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tsinfer_main, command)
        assert stderr == ""
        assert stdout == ""

    def verify_ts_provenance(self, treefile, expected_num_provenances):
        ts = tskit.load(treefile)
        prov = json.loads(ts.provenance(-1).record)
        assert ts.num_provenances == expected_num_provenances
        # Getting actual values out of the JSON is problematic here because
        # we're getting the pytest command line.
        assert isinstance(prov["parameters"]["command"], str)
        assert isinstance(prov["parameters"]["args"], list)

    def test_infer(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        sd = tsinfer.load(self.sample_file)
        self.run_command(["infer", self.sample_file, "-O", output_trees])
        self.verify_ts_provenance(output_trees, sd.num_provenances + 1)

    @pytest.mark.skip(
        reason="Ancestors not saving provenance:"
        "see https://github.com/tskit-dev/tsinfer/issues/753"
    )
    def test_ancestors(self):
        sd = tsinfer.load(self.sample_file)
        self.run_command(["generate-ancestors", self.sample_file])
        ancestors = tsinfer.load(self.ancestor_file)
        assert ancestors.num_provenances == sd.num_provenances + 1

    @pytest.mark.skipif(
        sys.platform == "win32", reason="windows simultaneous file permissions issue"
    )
    def test_chain(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        ancestors_trees = os.path.join(self.tempdir.name, "ancestors.trees")
        self.run_command(["generate-ancestors", self.sample_file])
        num_provenances_ancestors = tsinfer.load(self.ancestor_file).num_provenances
        self.run_command(["match-ancestors", self.sample_file, "-A", ancestors_trees])
        self.verify_ts_provenance(ancestors_trees, num_provenances_ancestors + 1)
        self.run_command(
            [
                "match-samples",
                self.sample_file,
                "-A",
                ancestors_trees,
                "-O",
                output_trees,
            ]
        )
        self.verify_ts_provenance(output_trees, num_provenances_ancestors + 2)


class TestRecombinationAndMismatch(TestCli):
    """
    Test that we correctly parse and use recombination and mismatch arguments
    """

    @pytest.mark.skipif(
        sys.platform == "win32", reason="windows simultaneous file permissions issue"
    )
    def test_separate_calls(self):
        self.run_command(["generate-ancestors", self.sample_file])
        with mock.patch("tsinfer.match_ancestors") as ma:
            self.run_command(
                [
                    "match-ancestors",
                    self.sample_file,
                    "--recombination-rate",
                    "0.001",
                    "--mismatch-ratio",
                    "0.01",
                ]
            )
            args, kwargs = ma.call_args
            assert kwargs["recombination_rate"] == 0.001
            assert kwargs["mismatch_ratio"] == 0.01

        with mock.patch("tsinfer.match_samples") as ms:
            self.run_command(
                [
                    "match-samples",
                    self.sample_file,
                    "--recombination-rate",
                    "10",
                    "--mismatch-ratio",
                    "100",
                ]
            )
            args, kwargs = ms.call_args
            assert kwargs["recombination_rate"] == 10
            assert kwargs["mismatch_ratio"] == 100

    def test_infer(self):
        command = [
            "infer",
            self.sample_file,
            "--recombination-rate",
            "0.1",
            "--mismatch-ratio",
            "10",
        ]
        with mock.patch("tsinfer.infer") as infer:
            self.run_command(command)
            args, kwargs = infer.call_args
            assert kwargs["recombination_rate"] == 0.1
            assert kwargs["mismatch_ratio"] == 10

    @pytest.mark.skip(reason="https://github.com/tskit-dev/tsinfer/issues/753")
    @pytest.mark.skipif(
        sys.platform == "win32", reason="windows simultaneous file permissions issue"
    )
    def test_map(self):
        ratemap = os.path.join(self.tempdir.name, "ratemap.txt")
        with open(ratemap, "w") as map:
            print("Chromosome  Position(bp)  Rate(cM/Mb)  Map(cM)", file=map)
            print("chr1 0 0.1 0", file=map)
            print("chr1 1 0.2 0.002", file=map)
        command = [
            "infer",
            self.sample_file,
            "--recombination-map",
            ratemap,
        ]
        with mock.patch("tsinfer.infer") as infer:
            self.run_command(command)
            args, kwargs = infer.call_args
            assert isinstance(kwargs["recombination_rate"], msprime.RateMap)

    @pytest.mark.skip(reason="https://github.com/tskit-dev/tsinfer/issues/753")
    @pytest.mark.skipif(
        sys.platform == "win32", reason="windows simultaneous file permissions issue"
    )
    def test_fails_on_bad_map(self):
        output_trees = os.path.join(self.tempdir.name, "output_test_map.trees")
        ratemap = os.path.join(self.tempdir.name, "ratemap.txt")
        sd = tsinfer.load(self.sample_file)
        last_pos = sd.sites_position[-1]
        assert last_pos > 2
        with open(ratemap, "w") as map:
            print("Chromosome  Position(bp)  Rate(cM/Mb)  Map(cM)", file=map)
            print("chr1 0 0.1 0.0", file=map)
            print(f"chr1 {int(last_pos) - 1} 0.2 0.001", file=map)
        command = [
            "infer",
            self.sample_file,
            "--recombination-map",
            ratemap,
            "-O",
            output_trees,
        ]
        with pytest.raises(ValueError, match="Cannot have positions"):
            self.run_command(command)


class TestMatchSamples(TestCli):
    """
    Tests for the match samples options.
    """

    @pytest.mark.skipif(
        sys.platform == "win32", reason="windows simultaneous file permissions issue"
    )
    def test_no_simplify(self):
        output_trees = os.path.join(self.tempdir.name, "output.trees")
        output_trees_no_simplify = os.path.join(
            self.tempdir.name, "output-nosimplify.trees"
        )
        output_trees_no_post_process = os.path.join(
            self.tempdir.name, "output-nopostprocess.trees"
        )
        self.run_command(["generate-ancestors", self.sample_file])
        self.run_command(["match-ancestors", self.sample_file])
        self.run_command(["match-samples", self.sample_file, "-O", output_trees])
        self.run_command(
            [
                "match-samples",
                self.sample_file,
                "--no-post-process",
                "-O",
                output_trees_no_post_process,
            ]
        )
        t1 = tskit.load(output_trees).tables
        t2 = tskit.load(output_trees_no_post_process).tables
        assert t1.nodes != t2.nodes
        # --no-simplify is an alias
        self.run_command(
            [
                "match-samples",
                self.sample_file,
                "--no-simplify",
                "-O",
                output_trees_no_simplify,
            ]
        )
        t3 = tskit.load(output_trees_no_simplify).tables
        assert t2.nodes == t3.nodes


class TestList(TestCli):
    """
    Tests cases for the list command.
    """

    # Need to mock out setup_logging here or we spew logging to the console
    # in later tests.
    @mock.patch("tsinfer.cli.setup_logging")
    def run_command(self, command, mock_setup_logging):
        stdout, stderr = capture_output(cli.tsinfer_main, command)
        assert stderr == ""
        assert mock_setup_logging.called
        return stdout

    def test_list_samples(self):
        output1 = self.run_command(["list", self.sample_file])
        assert len(output1) > 0
        output2 = self.run_command(["ls", self.sample_file])
        assert output1 == output2

    def test_list_samples_storage(self):
        output1 = self.run_command(["list", "-s", self.sample_file])
        assert len(output1) > 0
        output2 = self.run_command(["list", "--storage", self.sample_file])
        assert output1 == output2

    def test_list_ancestors(self):
        output1 = self.run_command(["list", self.ancestor_file])
        assert len(output1) > 0
        output2 = self.run_command(["ls", self.ancestor_file])
        assert output1 == output2

    def test_list_ancestors_storage(self):
        output1 = self.run_command(["list", "-s", self.ancestor_file])
        assert len(output1) > 0
        output2 = self.run_command(["list", "--storage", self.ancestor_file])
        assert output1 == output2

    def test_list_trees(self):
        output1 = self.run_command(["list", self.output_trees])
        assert len(output1) > 0
        output2 = self.run_command(["ls", self.output_trees])
        assert output1 == output2

    def test_list_ancestor_trees(self):
        output1 = self.run_command(["list", self.ancestor_trees])
        assert len(output1) > 0
        output2 = self.run_command(["ls", self.ancestor_trees])
        assert output1 == output2

    def test_list_unknown_files(self):
        zero_file = os.path.join(self.tempdir.name, "zeros")
        with open(zero_file, "wb") as f:
            f.write(bytearray(100))
        for bad_file in [zero_file]:
            with pytest.raises(exceptions.FileFormatError):
                self.run_command(["list", bad_file])
        if sys.platform == "win32":
            # Windows raises a PermissionError not IsADirectoryError when opening a dir
            with pytest.raises(PermissionError):
                self.run_command(["list", "/"])
        else:
            with pytest.raises(IsADirectoryError):
                self.run_command(["list", "/"])
