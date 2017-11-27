# TODO copyright headers.
"""
Command line interfaces to tsinfer.
"""
import argparse
import sys
import os.path
import shutil
import logging
import bsddb3

import h5py
import zarr
import daiquiri
import msprime

import tsinfer

logger = logging.getLogger(__name__)


# TODO Need better names/extensions for these files.

def get_ancestors_path(path, input_path):
    if path is None:
        path = os.path.splitext(input_path)[0] + ".tsanc"
    return path


def get_ancestors_ts(path, input_path):
    # FIXME!!
    if path is None:
        path = os.path.splitext(input_path)[0] + ".tsancts"
    return path


def get_output_ts(path, input_path):
    # FIXME!!
    if path is None:
        path = os.path.splitext(input_path)[0] + ".ts"
    return path


def setup_logging(args):
    log_level = "WARN"
    if args.verbosity > 0:
        log_level = "INFO"
    if args.verbosity > 1:
        log_level = "DEBUG"
    daiquiri.setup(level=log_level)


def run_infer(args):
    setup_logging(args)
    # if args.compression == "none":
        # args.compression = None
    input_container = zarr.DBMStore(args.input, open=bsddb3.btopen)
    input_root = zarr.open_group(store=input_container)
    ancestors_root = zarr.group()
    tsinfer.build_ancestors(
        input_root, ancestors_root, progress=args.progress,
        num_threads=args.num_threads)
    ancestors_ts = tsinfer.match_ancestors(
        input_root, ancestors_root,
        num_threads=args.num_threads, progress=args.progress)
    output_ts = get_output_ts(args.output_ts, args.input)
    ts = tsinfer.match_samples(
        input_root, ancestors_ts, num_threads=args.num_threads,
        progress=args.progress)
    logger.info("Writing output tree sequence to {}".format(output_ts))
    ts.dump(output_ts)

def run_build_ancestors(args):
    setup_logging(args)
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
    # input_container = zarr.ZipStore(args.input, mode='r')
    # ancestors_container = zarr.ZipStore(ancestors_path)

    input_container = zarr.DBMStore(args.input, open=bsddb3.btopen)
    ancestors_container = zarr.DBMStore(ancestors_path, open=bsddb3.btopen)

    input_root = zarr.open_group(store=input_container)
    ancestors_root = zarr.group(store=ancestors_container, overwrite=True)

    tsinfer.build_ancestors(
        input_root, ancestors_root, progress=args.progress,
        num_threads=args.num_threads)

    input_container.close()
    ancestors_container.close()


def run_match_ancestors(args):
    setup_logging(args)

    ancestors_path = get_ancestors_path(args.ancestors, args.input)
    logger.info("Loading ancestral haplotypes from {}".format(ancestors_path))
    ancestors_ts = get_ancestors_ts(args.ancestors_ts, args.input)
    # with tsinfer.open_input(args.input) as input_hdf5, \
    #         tsinfer.open_ancestors(ancestors_path) as ancestors_hdf5:
    # input_container = zarr.ZipStore(args.input, mode='r')
    # ancestors_container = zarr.ZipStore(ancestors_path)

    input_container = zarr.DBMStore(args.input, open=bsddb3.btopen)
    ancestors_container = zarr.DBMStore(ancestors_path, open=bsddb3.btopen)

    input_root = zarr.open_group(store=input_container)
    ancestors_root = zarr.open_group(store=ancestors_container)
    tsinfer.match_ancestors(
        input_root, ancestors_root, output_path=ancestors_ts,
        num_threads=args.num_threads, progress=args.progress,
        output_interval=args.output_interval, resume=args.resume)


def run_match_samples(args):
    setup_logging(args)
    # input_container = zarr.ZipStore(args.input, mode='r')

    input_container = zarr.DBMStore(args.input, open=bsddb3.btopen)

    input_root = zarr.open_group(store=input_container)
    ancestors_ts = get_ancestors_ts(args.ancestors_ts, args.input)
    output_ts = get_output_ts(args.output_ts, args.input)
    logger.info("Loading ancestral genealogies from {}".format(ancestors_ts))
    ancestors_ts = msprime.load(ancestors_ts)
    ts = tsinfer.match_samples(
        input_root, ancestors_ts, num_threads=args.num_threads,
        progress=args.progress)
    logger.info("Writing output tree sequence to {}".format(output_ts))
    ts.dump(output_ts)


def run_verify(args):
    setup_logging(args)

    input_container = zarr.DBMStore(args.input, open=bsddb3.btopen)
    input_root = zarr.open_group(store=input_container)
    ancestors_path = get_ancestors_path(args.ancestors, args.input)
    ancestors_container = zarr.DBMStore(ancestors_path, open=bsddb3.btopen)

    ancestors_root = zarr.open_group(store=ancestors_container)
    ancestors_ts = get_ancestors_ts(args.ancestors_ts, args.input)
    output_ts = get_output_ts(args.output_ts, args.input)
    logger.info("Loading ancestral genealogies from {}".format(ancestors_ts))
    ancestors_ts = msprime.load(ancestors_ts)
    tsinfer.verify(input_root, ancestors_root, ancestors_ts, progress=args.progress)


def add_input_file_argument(parser):
    parser.add_argument(
        "input", help="The input data in tsinfer input HDF5 format.")


# TODO there are very poor names. Need to think of something more descriptive than
# 'file' and 'ts'. One is the ancestor-source and the ancestor-genealogies or
# something?
def add_ancestors_file_argument(parser):
    parser.add_argument(
        "-a", "--ancestors", default=None,
        help=(
            "The path to the ancestors HDF5 file. If not specified, this "
            "defaults to the input file stem with the extension '.tsanc'. "
            "For example, if '1kg-chr1.tsinf' is the input file then the "
            "default ancestors file would be 1kg-chr1.tsanc"))


def add_ancestors_ts_argument(parser):
    parser.add_argument(
        # Again, this is really bad. Need a much better name.
        "-A", "--ancestors-ts", default=None,
        help=("TODO DOCUMENT"))


def add_output_ts_argument(parser):
    parser.add_argument(
        # Again, this is really bad. Need a much better name.
        "-O", "--output-ts", default=None,
        help=("TODO DOCUMENT"))


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


def add_output_interval_argument(parser):
    parser.add_argument(
        "--output-interval", "-I", type=int, default=None,
        help=(
            "The interval in minutes between output auto-saves. By default the "
            "output is only saved at the end of the run"))


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
    add_num_threads_argument(parser)
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
    add_ancestors_ts_argument(parser)
    add_output_interval_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    parser.add_argument(
        "--resume", "-r", default=False, action="store_true",
        help="Resume an existing build")
    parser.set_defaults(runner=run_match_ancestors)

    parser = subparsers.add_parser(
        "match-samples",
        aliases=["ms"],
        help=(
            "Matches the samples against the tree sequence structure built "
            "by the match-ancestors command"))
    add_input_file_argument(parser)
    add_verbosity_argument(parser)
    add_ancestors_ts_argument(parser)
    add_output_ts_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    parser.set_defaults(runner=run_match_samples)

    parser = subparsers.add_parser(
        "verify",
        help=(
            "Verifies the integrity of the files associated with a build."))
    add_input_file_argument(parser)
    add_verbosity_argument(parser)
    add_ancestors_file_argument(parser)
    add_ancestors_ts_argument(parser)
    add_output_ts_argument(parser)
    add_progress_argument(parser)
    parser.set_defaults(runner=run_verify)

    parser = subparsers.add_parser(
        "infer",
        help=(
            "TODO: document"))
    add_input_file_argument(parser)
    add_verbosity_argument(parser)
    add_output_ts_argument(parser)
    add_num_threads_argument(parser)
    add_progress_argument(parser)
    parser.set_defaults(runner=run_infer)

    return top_parser


def tsinfer_main(arg_list=None):
    parser = get_tsinfer_parser()
    args = parser.parse_args(arg_list)
    args.runner(args)
