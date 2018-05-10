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
Command line interfaces to tsinfer.
"""
import argparse
import os.path
import logging

import daiquiri
import msprime
import tqdm

import tsinfer
import tsinfer.formats as formats
import tsinfer.exceptions as exceptions


logger = logging.getLogger(__name__)


class ProgressMonitor(object):
    """
    Class responsible for managing in the tqdm progress monitors.
    """
    def __init__(
            self, enabled=True, generate_ancestors=False, match_ancestors=False,
            match_samples=False):
        self.enabled = enabled
        self.num_bars = 0
        if generate_ancestors:
            self.num_bars += 2
        if match_ancestors:
            self.num_bars += 1
        if match_samples:
            self.num_bars += 3

        self.current_bar = 0
        self.descriptions = {
            "ga_add_sites": "ga-add",
            "ga_generate": "ga-gen",
            "ma_match": "ma-match",
            "ms_match": "ms-match",
            "ms_paths": "ms-paths",
            "ms_sites": "ms-sites",
        }

    def get(self, key, total):
        self.current_bar += 1
        desc = "{:<8} ({}/{})".format(
            self.descriptions[key], self.current_bar, self.num_bars)
        bar_format = (
            "{desc}{percentage:3.0f}%|{bar}"
            "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]")
        progress = tqdm.tqdm(
            desc=desc, total=total, disable=not self.enabled,
            bar_format=bar_format, dynamic_ncols=True, smoothing=0.01)
        return progress


def get_default_path(path, input_path, extension):
    if path is None:
        path = os.path.splitext(input_path)[0] + extension
    return path


def get_ancestors_path(path, input_path):
    return get_default_path(path, input_path, ".ancestors")


def get_ancestors_trees_path(path, input_path):
    return get_default_path(path, input_path, ".ancestors.trees")


def get_output_trees_path(path, input_path):
    return get_default_path(path, input_path, ".trees")


def setup_logging(args):
    log_level = "WARN"
    if args.verbosity > 0:
        log_level = "INFO"
    if args.verbosity > 1:
        log_level = "DEBUG"
    if args.log_section is None:
        daiquiri.setup(level=log_level)
    else:
        daiquiri.setup(level="WARN")
        logger = logging.getLogger(args.log_section)
        logger.setLevel(log_level)


def run_list(args):
    setup_logging(args)
    tsinfer_file = None
    try:
        logger.debug("Trying SampleData file")
        tsinfer_file = formats.SampleData.load(args.path)
        logger.debug("Loaded SampleData file")
    except exceptions.FileFormatError as e:
        logger.debug("SampleData load failed: {}".format(e))
    try:
        logger.debug("Trying AncestorData file")
        tsinfer_file = formats.AncestorData.load(args.path)
        logger.debug("Loaded AncestorData file")
    except exceptions.FileFormatError as e:
        logger.debug("SampleData load failed: {}".format(e))

    if tsinfer_file is None:
        raise exceptions.FileFormatError(
            "Unrecognised file format. Try running with -vv and check the log "
            "for more details on what went wrong")

    if args.storage:
        print(tsinfer_file.info)
    else:
        print(tsinfer_file)


def run_infer(args):
    setup_logging(args)
    progress_monitor = ProgressMonitor(
        enabled=args.progress, generate_ancestors=True, match_ancestors=True,
        match_samples=True)
    sample_data = tsinfer.SampleData.load(args.input)
    ts = tsinfer.infer(sample_data, progress_monitor=progress_monitor)
    output_trees = get_output_trees_path(args.output_trees, args.input)
    logger.info("Writing output tree sequence to {}".format(output_trees))
    ts.dump(output_trees)


def run_generate_ancestors(args):
    setup_logging(args)
    ancestors_path = get_ancestors_path(args.ancestors, args.input)
    progress_monitor = ProgressMonitor(enabled=args.progress, generate_ancestors=True)
    sample_data = tsinfer.SampleData.load(args.input)
    tsinfer.generate_ancestors(
        sample_data, progress_monitor=progress_monitor, path=ancestors_path,
        num_flush_threads=args.num_threads)


def run_match_ancestors(args):
    setup_logging(args)
    ancestors_path = get_ancestors_path(args.ancestors, args.input)
    logger.info("Loading ancestral haplotypes from {}".format(ancestors_path))
    ancestors_trees = get_ancestors_trees_path(args.ancestors_trees, args.input)
    sample_data = tsinfer.SampleData.load(args.input)
    ancestor_data = tsinfer.AncestorData.load(ancestors_path)
    progress_monitor = ProgressMonitor(enabled=args.progress, match_ancestors=True)
    ts = tsinfer.match_ancestors(
        sample_data, ancestor_data,
        num_threads=args.num_threads, progress_monitor=progress_monitor,
        path_compression=not args.no_path_compression)
    logger.info("Writing ancestors tree sequence to {}".format(ancestors_trees))
    ts.dump(ancestors_trees)


def run_match_samples(args):
    setup_logging(args)

    sample_data = tsinfer.SampleData.load(args.input)
    ancestors_trees = get_ancestors_trees_path(args.ancestors_trees, args.input)
    output_trees = get_output_trees_path(args.output_trees, args.input)
    logger.info("Loading ancestral genealogies from {}".format(ancestors_trees))
    ancestors_trees = msprime.load(ancestors_trees)
    progress_monitor = ProgressMonitor(enabled=args.progress, match_samples=True)
    ts = tsinfer.match_samples(
        sample_data, ancestors_trees, num_threads=args.num_threads,
        path_compression=not args.no_path_compression,
        progress_monitor=progress_monitor)
    logger.info("Writing output tree sequence to {}".format(output_trees))
    ts.dump(output_trees)


# def run_verify(args):
#     setup_logging(args)
#     print("FIXME!!!")
#     sys.exit(1)


def add_input_file_argument(parser):
    parser.add_argument(
        "input",
        help=(
            "The input sample data in tsinfer 'samples' format. Please see the "
            "documentation at http://tsinfer.readthedocs.io/ for information on "
            "how to import data into this format."))


def add_ancestors_file_argument(parser):
    parser.add_argument(
        "-a", "--ancestors", default=None,
        help=(
            "The path to the ancestor data file in tsinfer 'ancestors' format. "
            "If not specified, this defaults to the input samples file stem "
            "with the extension '.ancestors'. For example, if '1kg-chr1.samples' "
            "is the input file then the default ancestors file would be "
            "'1kg-chr1.ancestors'"))


def add_ancestors_trees_argument(parser):
    parser.add_argument(
        "-A", "--ancestors-trees", default=None,
        help=(
            "The path to the ancestor trees file in tskit '.trees' format. "
            "If not specified, this defaults to the input samples file stem "
            "with the extension '.ancestors.trees'. For example, if '1kg-chr1.samples' "
            "is the input file then the default ancestors file would be "
            "'1kg-chr1.ancestors.trees'"))


def add_output_trees_argument(parser):
    parser.add_argument(
        "-O", "--output-trees", default=None,
        help=(
            "The path to the output trees file in tskit '.trees' format. "
            "If not specified, this defaults to the input samples file stem "
            "with the extension '.trees'. For example, if '1kg-chr1.samples' "
            "is the input file then the default output file would be "
            "'1kg-chr1.trees'"))


def add_progress_argument(parser):
    parser.add_argument(
        "--progress", "-p", action="store_true",
        help="Show a progress monitor.")


def add_path_compression_argument(parser):
    parser.add_argument(
        "--no-path-compression", action="store_true",
        help="Disable path compression")


def add_logging_arguments(parser):
    log_sections = ["tsinfer.inference", "tsinfer.formats", "tsinfer.threads"]
    parser.add_argument(
        "-v", "--verbosity", action='count', default=0,
        help="Increase the verbosity")
    parser.add_argument(
        "--log-section", "-L", choices=log_sections, default=None,
        help=("Log messages only for the specified module"))


def add_num_threads_argument(parser):
    parser.add_argument(
        "--num-threads", "-t", type=int, default=0,
        help=(
            "The number of worker threads to use. If < 1, use a simpler unthreaded "
            "algorithm (default)."))


def get_tsinfer_parser():
    top_parser = argparse.ArgumentParser(
        description="Command line interface for tsinfer.")
    top_parser.add_argument(
        "-V", "--version", action='version',
        version='%(prog)s {}'.format(tsinfer.__version__))

    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser = subparsers.add_parser(
        "generate-ancestors",
        aliases=["ga"],
        help=(
            "Builds a set of ancestors from the input sample data and stores "
            "the results in a tsinfer ancestors file."))
    add_input_file_argument(parser)
    add_ancestors_file_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    add_logging_arguments(parser)
    parser.set_defaults(runner=run_generate_ancestors)

    parser = subparsers.add_parser(
        "match-ancestors",
        aliases=["ma"],
        help=(
            "Matches the ancestors built by the 'build-ancestors' command against "
            "each other using the model information specified in the input file "
            "and writes the output to a tskit .trees file."))
    add_input_file_argument(parser)
    add_logging_arguments(parser)
    add_ancestors_file_argument(parser)
    add_ancestors_trees_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    add_path_compression_argument(parser)
    parser.set_defaults(runner=run_match_ancestors)

    parser = subparsers.add_parser(
        "match-samples",
        aliases=["ms"],
        help=(
            "Matches the samples against the tree sequence structure built "
            "by the match-ancestors command"))
    add_input_file_argument(parser)
    add_logging_arguments(parser)
    add_ancestors_trees_argument(parser)
    add_path_compression_argument(parser)
    add_output_trees_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    parser.set_defaults(runner=run_match_samples)

    # parser = subparsers.add_parser(
    #     "verify",
    #     help=(
    #         "Verifies the integrity of the files associated with a build."))
    # add_input_file_argument(parser)
    # add_logging_arguments(parser)
    # add_ancestors_file_argument(parser)
    # add_ancestors_trees_argument(parser)
    # add_output_trees_argument(parser)
    # add_progress_argument(parser)
    # parser.set_defaults(runner=run_verify)

    parser = subparsers.add_parser(
        "infer",
        help=(
            "TODO: document"))
    add_input_file_argument(parser)
    add_logging_arguments(parser)
    add_output_trees_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    parser.set_defaults(runner=run_infer)

    parser = subparsers.add_parser(
        "list",
        aliases=["ls"],
        help=(
            "Show a summary of the specified tsinfer related file."))
    add_logging_arguments(parser)
    parser.add_argument(
        "path", help="The tsinfer file to show information about.")
    parser.add_argument(
        "--storage", "-s", action="store_true",
        help="Show detailed information about data storage.")
    parser.set_defaults(runner=run_list)

    return top_parser


def tsinfer_main(arg_list=None):
    parser = get_tsinfer_parser()
    args = parser.parse_args(arg_list)
    args.runner(args)
