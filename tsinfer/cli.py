# TODO copyright headers.
"""
Command line interfaces to tsinfer.
"""
import argparse
import sys
import os.path
import shutil

import h5py
import zarr
import daiquiri

import tsinfer


def get_ancestors_path(path, input_path):
    if path is None:
        path = os.path.splitext(input_path)[0] + ".tsanc"
    return path


def setup_logging(verbosity):
    log_level = "WARN"
    if verbosity > 0:
        log_level = "INFO"
    if verbosity > 1:
        log_level = "DEBUG"
    daiquiri.setup(level=log_level)


def run_build_ancestors(args):
    setup_logging(args.verbosity)
    if args.compression == "none":
        args.compression = None
    ancestors_path = get_ancestors_path(args.ancestors, args.input)
    if os.path.exists(ancestors_path):
        # TODO add error and only do this on --force
        # shutil.rmtree(ancestors_path)
        os.unlink(ancestors_path)
    # with tsinfer.open_input(args.input) as input_hdf5, \
    #         h5py.File(ancestors_path, "w") as ancestors_hdf5:
    # input_container = zarr.DirectoryStore(args.input)
    # ancestors_container = zarr.DirectoryStore(ancestors_path)
    input_container = zarr.ZipStore(args.input)
    ancestors_container = zarr.ZipStore(ancestors_path)

    input_root = zarr.open_group(store=input_container)
    ancestors_root = zarr.group(store=ancestors_container, overwrite=True)

    tsinfer.build_ancestors(
        input_root, ancestors_root, progress=args.progress,
        compression=args.compression)

    input_container.close()
    ancestors_container.close()


def run_match_ancestors(args):
    log_level = "WARN"
    if args.verbosity > 0:
        log_level = "INFO"
    if args.verbosity > 1:
        log_level = "DEBUG"
    ancestors_path = get_ancestors_path(args.ancestors, args.input)
    with tsinfer.open_input(args.input) as input_hdf5, \
            tsinfer.open_ancestors(ancestors_path) as ancestors_hdf5:
        ts = tsinfer.match_ancestors(
            input_hdf5, ancestors_hdf5, num_threads=args.num_threads,
            progress=args.progress, log_level=log_level)
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


def add_verbosity_argument(parser):
    parser.add_argument(
        "-v", "--verbosity", action='count', default=0,
        help="Increase the verbosity")


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
    add_verbosity_argument(parser)
    parser.set_defaults(runner=run_build_ancestors)

    parser = subparsers.add_parser(
        "match-ancestors",
        aliases=["ma"],
        help=(
            "Matches the ancestors built by the 'build-ancestors' command against "
            "each other using the model information specified in the input file "
            "and writes the output to a tree sequence HDF5 file."))
    add_input_file_argument(parser)
    add_verbosity_argument(parser)
    add_ancestors_file_argument(parser)
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
