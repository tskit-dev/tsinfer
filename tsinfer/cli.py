# TODO copyright headers.
"""
Command line interfaces to tsinfer.
"""
import argparse
import sys
import os.path

import h5py

import tsinfer


def get_ancestors_path(path, input_path):
    if path is None:
        path = os.path.splitext(input_path)[0] + ".tsinf"
    return path


def run_build_ancestors(args):
    if args.compression == "none":
        args.compression = None
    ancestors_path = get_ancestors_path(args.ancestors, args.input)
    with tsinfer.open_input(args.input) as input_hdf5, \
            h5py.File(ancestors_path, "w") as ancestors_hdf5:
        tsinfer.build_ancestors(
            input_hdf5, ancestors_hdf5, progress=args.progress,
            compression=args.compression)


def run_match_ancestors(args):
    with tsinfer.open_input(args.input) as input_hdf5, \
            tsinfer.open_ancestors(args.ancestors) as ancestors_hdf5:
        ts = tsinfer.match_ancestors(
            input_hdf5, ancestors_hdf5, num_threads=args.num_threads,
            progress=args.progress)
        ts.dump(args.output)


def add_input_file_argument(parser):
    parser.add_argument(
        "input", help="The input data in tsinfer input HDF5 format.")


def add_ancestors_file_argument(parser):
    parser.add_argument(
        "-a", "--ancestors", default=None,
        help=(
            "The path to the ancestors HDF5 file. If not specified, this "
            "defaults to the input file stem with the extension '.tsanc'. "
            "For example, if '1kg-chr1.tsinf' is the input file then the "
            "default ancestors file would be 1kg-chr1.tsanc"))


def add_progress_argument(parser):
    parser.add_argument(
        "--progress", "-p", action="store_true",
        help="Show a progress monitor.")


def add_num_threads_argument(parser):
    parser.add_argument(
        "--num-threads", "-t", type=int, default=0,
        help=(
            "The number of worker threads to use. If < 1, use a simpler unthreaded "
            "algorithm (default)."))


def add_compression_argument(parser):
    parser.add_argument(
        "--compression", "-z", choices=["gzip", "lzf", "none"], default="gzip",
        help="Enable HDF5 compression on datasets.")


def get_tsinfer_parser():
    top_parser = argparse.ArgumentParser(
        description="Command line interface for tsinfer.")
    top_parser.add_argument(
        "-V", "--version", action='version',
        version='%(prog)s {}'.format(tsinfer.__version__))
    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser = subparsers.add_parser(
        "build-ancestors",
        aliases=["ba"],
        help=(
            "Builds a set of ancestors from the input haplotype data and stores "
            "the results in an output HDF5 file."))
    add_input_file_argument(parser)
    add_ancestors_file_argument(parser)
    add_compression_argument(parser)
    add_progress_argument(parser)
    parser.set_defaults(runner=run_build_ancestors)

    parser = subparsers.add_parser(
        "match-ancestors",
        aliases=["ma"],
        help=(
            "Matches the ancestors built by the 'build-ancestors' command against "
            "each other using the model information specified in the input file "
            "and writes the output to a tree sequence HDF5 file."))
    add_input_file_argument(parser)
    parser.add_argument(
        "ancestors", help="The set of ancestors to match against each other. ")
    parser.add_argument(
        "output", help="The output tree sequence file.")
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    parser.set_defaults(runner=run_match_ancestors)

    return top_parser


def tsinfer_main(arg_list=None):
    parser = get_tsinfer_parser()
    args = parser.parse_args(arg_list)
    args.runner(args)
