#
# Copyright (C) 2018-2024 University of Oxford
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
Custom LMDB store for zarr v3.

zarr v3 removed its built-in LMDBStore. This module provides a replacement
that implements the zarr v3 async Store ABC while keeping the same on-disk
LMDB layout as zarr v2's LMDBStore, so existing .samples and .ancestors files
can continue to be used (subject to the object-dtype encoding changes in
formats.py — see ON-DISK COMPATIBILITY note there).

The store maps zarr path-style keys (e.g. "provenances/timestamp/.zarray")
directly to LMDB byte-string keys. There is no directory indirection: all
keys live in a flat LMDB namespace, exactly as zarr v2's LMDBStore did.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator, Iterable

import lmdb

from zarr.abc.store import Store

if TYPE_CHECKING:
    from zarr.buffer import Buffer, BufferPrototype
    from zarr.core.buffer import ByteRequest


class LMDBStore(Store):
    """
    Zarr v3 store backed by LMDB.

    Implements the zarr.abc.store.Store async interface using synchronous LMDB
    calls (no true async I/O; LMDB is memory-mapped and latency is negligible).

    Parameters
    ----------
    path:
        Path to the LMDB environment file.
    map_size:
        Maximum size of the LMDB environment in bytes.
    readonly:
        Open the store in read-only mode.
    subdir:
        Whether LMDB uses a sub-directory layout (False = single-file).
    lock:
        Whether LMDB should use locking.
    """

    def __init__(
        self,
        path: str,
        map_size: int = 2**40,
        readonly: bool = False,
        subdir: bool = False,
        lock: bool = True,
    ) -> None:
        super().__init__(read_only=readonly)
        self._path = path
        self._map_size = map_size
        self._subdir = subdir
        self._lock = lock
        self._env = lmdb.open(
            path,
            map_size=map_size,
            readonly=readonly,
            subdir=subdir,
            lock=lock,
        )

    # ------------------------------------------------------------------
    # zarr v3 Store ABC — required abstract methods (non-async)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LMDBStore):
            return self._path == other._path
        return NotImplemented

    async def get_partial_values(
        self,
        prototype: "BufferPrototype",
        key_ranges: "Iterable[tuple[str, ByteRequest | None]]",
    ) -> "list[Buffer | None]":
        """Read multiple key ranges in one call (no real partial I/O in LMDB)."""
        results = []
        for key, byte_range in key_ranges:
            results.append(await self.get(key, prototype, byte_range))
        return results

    # ------------------------------------------------------------------
    # zarr v3 Store ABC — required properties
    # ------------------------------------------------------------------

    @property
    def supports_writes(self) -> bool:
        return not self._read_only

    @property
    def supports_deletes(self) -> bool:
        return not self._read_only

    @property
    def supports_listing(self) -> bool:
        return True

    @property
    def supports_partial_writes(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # zarr v3 Store ABC — required async methods
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        with self._env.begin() as txn:
            value = txn.get(key.encode())
        if value is None:
            return None
        if byte_range is not None:
            # Apply byte range slicing.  zarr v3 ByteRequest subtypes:
            #   RangeByteRequest(start, end), SuffixByteRequest(suffix_length),
            #   FullByteRequest / None.
            # We handle the common cases; fall back to full read for unknowns.
            if hasattr(byte_range, "start") and hasattr(byte_range, "end"):
                value = value[byte_range.start : byte_range.end]
            elif hasattr(byte_range, "suffix_length"):
                value = value[-byte_range.suffix_length :]
        return prototype.buffer.from_bytes(value)

    async def set(self, key: str, value: Buffer) -> None:
        if self._read_only:
            raise ValueError("Store is read-only")
        data = value.as_numpy_array().tobytes()
        with self._env.begin(write=True) as txn:
            txn.put(key.encode(), data)

    async def delete(self, key: str) -> None:
        if self._read_only:
            raise ValueError("Store is read-only")
        with self._env.begin(write=True) as txn:
            txn.delete(key.encode())

    async def exists(self, key: str) -> bool:
        with self._env.begin() as txn:
            return txn.get(key.encode()) is not None

    async def list(self) -> AsyncGenerator[str, None]:
        with self._env.begin() as txn:
            cursor = txn.cursor()
            if cursor.first():
                for key in cursor.iternext(keys=True, values=False):
                    yield key.decode()

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        prefix_bytes = prefix.encode()
        with self._env.begin() as txn:
            cursor = txn.cursor()
            if not cursor.set_range(prefix_bytes):
                return
            for key in cursor.iternext(keys=True, values=False):
                if not key.startswith(prefix_bytes):
                    break
                yield key.decode()

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        """Yield names of immediate children under *prefix*."""
        # LMDB keys are flat strings like "a/b/c". We simulate a directory
        # listing by collecting the first path component after the prefix.
        prefix_with_sep = (prefix + "/") if prefix else ""
        prefix_bytes = prefix_with_sep.encode()
        seen: set[str] = set()
        with self._env.begin() as txn:
            cursor = txn.cursor()
            if prefix_bytes:
                if not cursor.set_range(prefix_bytes):
                    return
            else:
                if not cursor.first():
                    return
            for key in cursor.iternext(keys=True, values=False):
                key_str = key.decode()
                if not key_str.startswith(prefix_with_sep):
                    break
                rest = key_str[len(prefix_with_sep) :]
                if not rest:
                    continue
                child = rest.split("/")[0]
                if child and child not in seen:
                    seen.add(child)
                    yield child

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _open(self) -> None:
        """Called by zarr v3 factory functions after construction; no-op here."""
        pass

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Convenience helpers used by formats.py
    # ------------------------------------------------------------------

    def set_mapsize(self, size: int) -> None:
        """Resize the LMDB map. Used in DataContainer.finalise() to shrink."""
        self._env.set_mapsize(size)

    def info(self) -> dict:
        """Return the LMDB info dict (page count, etc.)."""
        return self._env.info()

    def stat(self) -> dict:
        """Return the LMDB statistics dict (page size, etc.)."""
        return self._env.stat()
