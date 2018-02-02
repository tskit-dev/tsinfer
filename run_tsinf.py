#!/usr/bin/env python3
description="""
"""

import argparse
import os

import zarr
import bsddb3
import tsinfer

parser = argparse.ArgumentParser(description=description)
parser.add_argument('infile', 
                    help='a .tsinf file produced e.g. by vcf2tsinfer.py')
parser.add_argument('-o', '--outfile',
                    help='the output file. If not given, will save under same name as infile with a .ts extension')
args = parser.parse_args()

method="C"
input_hdf5 = zarr.DBMStore(args.infile, open=bsddb3.btopen)
input_root = zarr.group(store=input_hdf5)

ancestors_root = zarr.group()
tsinfer.build_ancestors(
    input_root, ancestors_root, method=method, chunk_size=16, compress=False)
ancestors_ts = tsinfer.match_ancestors(
    input_root, ancestors_root, method=method, path_compression=path_compression)
full_inferred_ts = tsinfer.match_samples(
    input_root, ancestors_ts, method=method, path_compression=path_compression,
    simplify=simplify)
full_inferred_ts.dump(args.outfile or (os.path.splitext(infile) + '.ts'))
