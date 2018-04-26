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
import os.path
import sys
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
import msprime

import tsinfer
import tsinfer.cli as cli


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


class TestCommandsDefaults(unittest.TestCase):
    """
    Tests that the basic commands work if we provide the default arguments.
    """
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="tsinfer_cli_test")
        self.sample_file = os.path.join(self.tempdir.name, "samples")
        self.input_ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=10, random_seed=10)
        sample_data = tsinfer.SampleData.initialise(
            num_samples=self.input_ts.num_samples,
            sequence_length=self.input_ts.sequence_length,
            compressor=None, filename=self.sample_file)
        for var in self.input_ts.variants():
            sample_data.add_site(var.site.position, var.alleles, var.genotypes)
        sample_data.finalise()

    def verify_output(self, output_path):
        output_ts = msprime.load(output_path)
        self.assertEqual(output_ts.num_samples, self.input_ts.num_samples)
        self.assertEqual(output_ts.sequence_length, self.input_ts.sequence_length)
        self.assertEqual(output_ts.num_sites, self.input_ts.num_sites)
        self.assertGreater(output_ts.num_sites, 1)
        self.assertTrue(np.array_equal(
            output_ts.genotype_matrix(), self.input_ts.genotype_matrix()))

    # Need to mock out setup_logging here or we spew logging to the console
    # in later tests.
    @mock.patch("tsinfer.cli.setup_logging")
    def run_command(self, command, mock_setup_logging):
        stderr, stdout = capture_output(cli.tsinfer_main, command)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "")
        self.assertTrue(mock_setup_logging.called)

    def test_infer(self):
        output_ts = os.path.join(self.tempdir.name, "output.ts")
        self.run_command(["infer", self.sample_file, "-O", output_ts])
        self.verify_output(output_ts)

    def test_nominal_chain(self):
        output_ts = os.path.join(self.tempdir.name, "output.ts")
        self.run_command(["build-ancestors", self.sample_file])
        self.run_command(["match-ancestors", self.sample_file])
        self.run_command(["match-samples", self.sample_file, "-O", output_ts])
        self.verify_output(output_ts)
