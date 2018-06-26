"""
Quick script to patch up the sequence length attribute in version
1.0 sample files to make sure they are not 0. Older version supported
this but new versions will not.
"""

import tsinfer
import zarr
import sys
import os.path

filename = sys.argv[1]
sample_data = tsinfer.load(filename)
sequence_length = sample_data.sites_position[-1] + 1
sample_data.close()

# Add a megabyte to the map size in the file size goes up.
map_size = os.path.getsize(filename) + 1024**2
store = zarr.LMDBStore(filename, subdir=False, map_size=map_size)
data = zarr.open(store=store, mode="w+")
data.attrs["sequence_length"] = sequence_length
store.close()

sample_data = tsinfer.load(filename)
print("patched up sequence length")
print(sample_data)
