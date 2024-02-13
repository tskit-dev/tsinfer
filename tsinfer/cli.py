#
# Copyright (C) 2018-2022 University of Oxford
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
import json
import logging
import math
import os.path
import sys

try:
    import resource
except ImportError:
    resource = None  # resource.getrusage absent on windows, so skip outputting max mem

import daiquiri

import tskit
import humanize
import time
import numpy as np

import tsinfer
import tsinfer.exceptions as exceptions
import tsinfer.provenance as provenance


logger = logging.getLogger(__name__)


__before = time.time()


def summarise_usage():
    wall_time = humanize.naturaldelta(time.time() - __before)
    user_time = humanize.naturaldelta(os.times().user)
    sys_time = os.times().system
    if resource is None:
        # Don't report max memory on Windows. We could do this using the psutil lib, via
        # psutil.Process(os.getpid()).get_ext_memory_info().peak_wset if demand exists
        maxmem_str = ""
    else:
        max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            max_mem *= 1024  # Linux and other OSs (e.g. freeBSD) report maxrss in kb
        maxmem_str = "; max memory={}".format(
            humanize.naturalsize(max_mem, binary=True)
        )
    logger.info(f"wall time = {wall_time}")
    logger.info(f"rusage: user={user_time}; sys={sys_time:.2f}s" + maxmem_str)


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


def get_recombination_map(args):
    if args.recombination_rate is not None:
        return args.recombination_rate
    # uncomment below when https://github.com/tskit-dev/tsinfer/issues/753 fixed
    # if args.recombination_map is not None:
    #     return msprime.RateMap.read_hapmap(args.recombination_map)
    return None


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


def summarise_tree_sequence(ts):
    print(ts)
    # TODO Add optional tree statistics like mean degree, etc.


def run_list(args):
    setup_logging(args)
    # First try to load with tskit.
    ts = None
    try:
        ts = tskit.load(args.path)
    except tskit.FileFormatError:
        pass
    if ts is None:
        tsinfer_file = tsinfer.load(args.path)
        if args.storage:
            print(tsinfer_file.info)
        else:
            print(tsinfer_file)
    else:
        summarise_tree_sequence(ts)


def write_ts(ts, path):
    logger.info(f"Writing output tree sequence to {path}")
    tables = ts.dump_tables()
    # Following guidance at
    # https://tskit.dev/tskit/docs/stable/provenance.html#cli-invocations
    record = provenance.get_provenance_dict(
        command=sys.argv[0],
        args=sys.argv[1:],
    )
    tables.provenances.add_row(json.dumps(record))
    # Avoid creating a new TS object by writing tables.
    assert tables.has_index()
    tables.dump(path)


def run_infer(args):
    setup_logging(args)
    try:
        sample_data = tsinfer.SampleData.load(args.samples)
    except exceptions.FileFormatError as e:
        # Check if the user has tried to infer a tree sequence, a common basic mistake
        try:
            tskit.load(args.samples)
        except tskit.FileFormatError:
            raise e  # Re-raise the original error
        raise exceptions.FileFormatError(
            "Expecting a sample data file, not a tree sequence (you can create one "
            "via the Python function `tsinfer.SampleData.from_tree_sequence()`)."
        )
    sample_data = tsinfer.SampleData.load(args.samples)
    if args.keep_intermediates:
        run_generate_ancestors(args, usage_summary=False)
        run_match_ancestors(args, usage_summary=False)
        run_match_samples(args, usage_summary=False)
    else:
        if args.ancestors is not None:
            raise ValueError(
                "Must specify --keep-intermediates to save an ancestors file"
            )
        if args.ancestors_trees is not None:
            raise ValueError(
                "Must specify --keep-intermediates to save an ancestors tree sequence"
            )

        ts = tsinfer.infer(
            sample_data,
            progress_monitor=args.progress,
            num_threads=args.num_threads,
            recombination_rate=get_recombination_map(args),
            mismatch_ratio=args.mismatch_ratio,
            path_compression=not args.no_path_compression,
            record_provenance=False,
        )
        output_trees = get_output_trees_path(args.output_trees, args.samples)
        write_ts(ts, output_trees)
    summarise_usage()


def run_generate_ancestors(args, usage_summary=True):
    setup_logging(args)
    ancestors_path = get_ancestors_path(args.ancestors, args.samples)
    sample_data = tsinfer.SampleData.load(args.samples)
    tsinfer.generate_ancestors(
        sample_data,
        progress_monitor=args.progress,
        num_flush_threads=getattr(args, "num_flush_threads", 0),
        num_threads=args.num_threads,
        path=ancestors_path,
        record_provenance=False,
    )
    # NB: ideally we should store the cli provenance in here, but this creates
    # perf issues - see https://github.com/tskit-dev/tsinfer/issues/743
    if usage_summary:
        summarise_usage()


def run_match_ancestors(args, usage_summary=True):
    setup_logging(args)
    ancestors_path = get_ancestors_path(args.ancestors, args.samples)
    logger.info(f"Loading ancestral haplotypes from {ancestors_path}")
    ancestors_trees = get_ancestors_trees_path(args.ancestors_trees, args.samples)
    sample_data = tsinfer.SampleData.load(args.samples)
    ancestor_data = tsinfer.AncestorData.load(ancestors_path)
    ts = tsinfer.match_ancestors(
        sample_data,
        ancestor_data,
        num_threads=args.num_threads,
        progress_monitor=args.progress,
        recombination_rate=get_recombination_map(args),
        mismatch_ratio=args.mismatch_ratio,
        path_compression=not args.no_path_compression,
        record_provenance=False,
    )
    write_ts(ts, ancestors_trees)
    if usage_summary:
        summarise_usage()


def run_augment_ancestors(args, usage_summary=True):
    setup_logging(args)

    sample_data = tsinfer.SampleData.load(args.samples)
    ancestors_trees = get_ancestors_trees_path(args.ancestors_trees, args.samples)
    output_path = args.augmented_ancestors
    logger.info(f"Loading ancestral genealogies from {ancestors_trees}")
    ancestors_trees = tskit.load(ancestors_trees)
    # TODO Need some error checking on these values
    n = args.num_samples
    N = sample_data.num_samples
    if n is None:
        n = int(math.ceil(10 * N / 100))

    sample_indexes = np.linspace(0, N - 1, num=n).astype(int)
    ts = tsinfer.augment_ancestors(
        sample_data,
        ancestors_trees,
        sample_indexes,
        num_threads=args.num_threads,
        path_compression=not args.no_path_compression,
        progress_monitor=args.progress,
        recombination_rate=get_recombination_map(args),
        mismatch_ratio=args.mismatch_ratio,
        record_provenance=False,
    )
    logger.info(f"Writing output tree sequence to {output_path}")
    ts.dump(output_path)
    if usage_summary:
        summarise_usage()


def run_match_samples(args, usage_summary=True):
    setup_logging(args)

    sample_data = tsinfer.SampleData.load(args.samples)
    ancestors_trees = get_ancestors_trees_path(args.ancestors_trees, args.samples)
    output_trees = get_output_trees_path(args.output_trees, args.samples)
    logger.info(f"Loading ancestral genealogies from {ancestors_trees}")
    ancestors_trees = tskit.load(ancestors_trees)
    ts = tsinfer.match_samples(
        sample_data,
        ancestors_trees,
        num_threads=args.num_threads,
        path_compression=not args.no_path_compression,
        post_process=not args.no_post_process,
        progress_monitor=args.progress,
        recombination_rate=get_recombination_map(args),
        mismatch_ratio=args.mismatch_ratio,
        record_provenance=False,
    )
    write_ts(ts, output_trees)
    if usage_summary:
        summarise_usage()


def run_verify(args):
    setup_logging(args)
    samples = tsinfer.SampleData.load(args.samples)
    ts = tskit.load(args.tree_sequence)
    tsinfer.verify(samples, ts, progress_monitor=args.progress)
    summarise_usage()


def add_samples_file_argument(parser):
    parser.add_argument(
        "samples",
        help=(
            "The input sample data in tsinfer 'samples' format. Please see the "
            "documentation at https://tskit.dev/tsinfer/docs/ for information on "
            "how to import data into this format."
        ),
    )


def add_ancestors_file_argument(parser):
    parser.add_argument(
        "-a",
        "--ancestors",
        default=None,
        help=(
            "The path to the ancestor data file in tsinfer 'ancestors' format. "
            "If not specified, this defaults to the input samples file stem "
            "with the extension '.ancestors'. For example, if '1kg-chr1.samples' "
            "is the input file then the default ancestors file would be "
            "'1kg-chr1.ancestors'"
        ),
    )


def add_ancestors_trees_argument(parser):
    parser.add_argument(
        "-A",
        "--ancestors-trees",
        default=None,
        help=(
            "The path to the ancestor trees file in tskit '.trees' format. "
            "If not specified, this defaults to the input samples file stem "
            "with the extension '.ancestors.trees'. For example, if '1kg-chr1.samples' "
            "is the input file then the default ancestors file would be "
            "'1kg-chr1.ancestors.trees'"
        ),
    )


def add_output_trees_argument(parser):
    parser.add_argument(
        "-O",
        "--output-trees",
        default=None,
        help=(
            "The path to the output trees file in tskit '.trees' format. "
            "If not specified, this defaults to the input samples file stem "
            "with the extension '.trees'. For example, if '1kg-chr1.samples' "
            "is the input file then the default output file would be "
            "'1kg-chr1.trees'"
        ),
    )


def add_progress_argument(parser):
    parser.add_argument(
        "--progress", "-p", action="store_true", help="Show a progress monitor."
    )


def add_path_compression_argument(parser):
    parser.add_argument(
        "--no-path-compression", action="store_true", help="Disable path compression"
    )


def add_postprocess_argument(parser):
    parser.add_argument(
        "--no-post-process",
        "--no-simplify",  # Deprecated alias
        action="store_true",
        help="Do not post process the output tree sequence",
    )


def add_recombination_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--recombination-rate",
        default=None,
        type=float,
        help="The recombination rate per unit genome",
    )
    # Uncomment below when https://github.com/tskit-dev/tsinfer/issues/753 fixed
    # group.add_argument(
    #     "--recombination-map",
    #     default=None,
    #     help=(
    #         "The path to a file containing recombination rates along the chromosome "
    #         "in HapMap format (see https://tskit.dev/msprime/docs/latest/api.html"
    #         "#msprime.RateMap.read_hapmap for details of the format)"
    #     ),
    # )


def add_mismatch_argument(parser):
    parser.add_argument(
        "--mismatch-ratio",
        type=float,
        default=None,
        help=(
            "The mismatch ratio: measures the relative importance of multiple "
            "mutation/error versus recombination during inference. This defaults "
            "to unity if a recombination rate or map are specified."
        ),
    )


def add_logging_arguments(parser):
    log_sections = ["tsinfer.inference", "tsinfer.formats", "tsinfer.threads"]
    parser.add_argument(
        "-v", "--verbosity", action="count", default=0, help="Increase the verbosity"
    )
    parser.add_argument(
        "--log-section",
        "-L",
        choices=log_sections,
        default=None,
        help=("Log messages only for the specified module"),
    )


def add_num_threads_argument(parser):
    parser.add_argument(
        "--num-threads",
        "-t",
        type=int,
        default=0,
        help=(
            "The number of worker threads to use. If < 1, use a simpler unthreaded "
            "algorithm (default)."
        ),
    )


def add_num_flush_threads_argument(parser):
    parser.add_argument(
        "--num-flush-threads",
        "-F",
        type=int,
        default=2,
        help=(
            "The number of data flush threads to use. If < 1, all data is flushed "
            "synchronously in the main thread (default=2)"
        ),
    )


def add_keep_intermediates_argument(parser):
    parser.add_argument(
        "--keep-intermediates",
        "-k",
        action="store_true",
        help=(
            "Keep the intermediate ancestors and ancestors-tree-sequence files. "
            "To override the default locations where these files are saved, use the "
            "--ancestors and --ancestors-trees options"
        ),
    )


def get_cli_parser():
    top_parser = argparse.ArgumentParser(
        description="Command line interface for tsinfer."
    )
    top_parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {tsinfer.__version__}",
    )

    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser = subparsers.add_parser(
        "generate-ancestors",
        aliases=["ga"],
        help=(
            "Generates a set of ancestors from the input sample data and stores "
            "the results in a tsinfer ancestors file."
        ),
    )
    add_samples_file_argument(parser)
    add_ancestors_file_argument(parser)
    add_num_threads_argument(parser)
    add_num_flush_threads_argument(parser)
    add_progress_argument(parser)
    add_logging_arguments(parser)
    parser.set_defaults(runner=run_generate_ancestors)

    parser = subparsers.add_parser(
        "match-ancestors",
        aliases=["ma"],
        help=(
            "Matches the ancestors built by the 'generate-ancestors' command against "
            "each other using the model information specified in the input file "
            "and writes the output to a tskit .trees file."
        ),
    )
    add_samples_file_argument(parser)
    add_logging_arguments(parser)
    add_ancestors_file_argument(parser)
    add_ancestors_trees_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    add_path_compression_argument(parser)
    add_recombination_arguments(parser)
    add_mismatch_argument(parser)
    parser.set_defaults(runner=run_match_ancestors)

    parser = subparsers.add_parser(
        "augment-ancestors",
        aliases=["aa"],
        help="Augments the ancestors tree sequence by adding a subset of the samples",
    )
    add_samples_file_argument(parser)
    parser.add_argument(
        "augmented_ancestors",
        help="The path to write the augmented ancestors tree sequence to",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="The number of samples to use. Defaults to 10%% of the total.",
    )
    add_ancestors_trees_argument(parser)
    add_logging_arguments(parser)
    add_path_compression_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    add_recombination_arguments(parser)
    add_mismatch_argument(parser)
    parser.set_defaults(runner=run_augment_ancestors)

    parser = subparsers.add_parser(
        "match-samples",
        aliases=["ms"],
        help=(
            "Matches the samples against the tree sequence structure built "
            "by the match-ancestors command"
        ),
    )
    add_samples_file_argument(parser)
    add_logging_arguments(parser)
    add_ancestors_trees_argument(parser)
    add_path_compression_argument(parser)
    add_postprocess_argument(parser)
    add_output_trees_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    add_recombination_arguments(parser)
    add_mismatch_argument(parser)
    parser.set_defaults(runner=run_match_samples)

    parser = subparsers.add_parser(
        "infer",
        help=(
            "Runs the generate-ancestors, match-ancestors and match-samples "
            "steps in one go. Not recommended for large inferences."
        ),
    )
    add_samples_file_argument(parser)
    add_logging_arguments(parser)
    add_output_trees_argument(parser)
    add_path_compression_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    add_postprocess_argument(parser)
    add_recombination_arguments(parser)
    add_mismatch_argument(parser)
    add_keep_intermediates_argument(parser)
    add_ancestors_file_argument(parser)  # Only used if keep-intermediates
    add_ancestors_trees_argument(parser)  # Only used if keep-intermediates
    parser.set_defaults(runner=run_infer)

    parser = subparsers.add_parser(
        "list",
        aliases=["ls"],
        help=("Show a summary of the specified tsinfer related file."),
    )
    add_logging_arguments(parser)
    parser.add_argument("path", help="The tsinfer file to show information about.")
    parser.add_argument(
        "--storage",
        "-s",
        action="store_true",
        help="Show detailed information about data storage.",
    )
    parser.set_defaults(runner=run_list)

    parser = subparsers.add_parser(
        "verify",
        help=(
            "Verify that the specified tree sequence and samples files represent "
            "the same data"
        ),
    )
    add_logging_arguments(parser)
    add_samples_file_argument(parser)
    parser.add_argument(
        "tree_sequence", help="The tree sequence to compare with in .trees format."
    )
    add_progress_argument(parser)
    parser.set_defaults(runner=run_verify)

    return top_parser


def tsinfer_main(arg_list=None):
    parser = get_cli_parser()
    args = parser.parse_args(arg_list)
    args.runner(args)
