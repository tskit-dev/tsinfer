#!/usr/bin/env python3
description="""
"""

import argparse
import os

import zarr
import bsddb3
import tsinfer

parser = argparse.ArgumentParser(description=description)
parser.add_argument('infiles', nargs="+",
                    help='One or more .tsinf files produced e.g. by vcf2tsinfer.py.')
parser.add_argument('-o', '--outfile',
                    help='The output file name. If not given, will save under same name as infile with a .ts extension' \
                    'if more than one input file, will append ".0", ".1", ".2" etc to the name.')
parser.add_argument('-P', '--progress', action='store_true',
                    help='Show a progress bar.')
args = parser.parse_args()

method, path_compression, simplify = "C", True, True #set defaults

for i, fn in enumerate(args.infiles):
    ext = ('.' + str(i)) if len(args.infiles) > 1 else ''
    if args.outfile:
        out_fn = args.outfile + ext
    else:
        out_fn = os.path.splitext(fn)[0] + '.hdf5'
    if not os.path.isfile(fn):
        raise FileNotFoundError
    input_hdf5 = zarr.DBMStore(fn, open=bsddb3.btopen)
    input_root = zarr.group(store=input_hdf5)
    
    ancestors_root = zarr.group()
    tsinfer.build_ancestors(
        input_root, ancestors_root, method=method, chunk_size=16, compress=False,
        progress = args.progress)
    ancestors_ts = tsinfer.match_ancestors(
        input_root, ancestors_root, method=method, path_compression=path_compression,
        progress = args.progress)
    full_inferred_ts = tsinfer.match_samples(
        input_root, ancestors_ts, method=method, path_compression=path_compression,
        simplify=simplify, progress = args.progress)
    full_inferred_ts.dump(out_fn)
