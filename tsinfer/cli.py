# TODO copyright headers.
"""
Command line interfaces to tsinfer.
"""
import argparse
import sys

import tsinfer


def run_build_ancestors(args):
    print("Running build ancestors.")


def add_input_file_argument(parser):
    parser.add_argument(
        "input", help="The input data in tsinfer input HDF5 format.")


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
        help=(
            "Builds a set of ancestors from the input haplotype data and stores "
            "the results in an outout HDF5 file."))
    add_input_file_argument(parser)
    parser.add_argument(
        "ancestors", help="The path to store the output ancestors.")
    parser.add_argument(
        "--compress", "-z", action="store_true",
        help="Enable HDF5's transparent zlib compression")
    parser.set_defaults(runner=run_build_ancestors)

    return top_parser


def tsinfer_main(arg_list=None):
    parser = get_tsinfer_parser()
    args = parser.parse_args(arg_list)
    args.runner(args)
