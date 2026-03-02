import pathlib
from collections.abc import AsyncIterator, Iterable

import lmdb
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype


class LMDBStore(Store):
    """
    Minimal LMDB-backed Zarr store for zarr-python v3.

    This store implements only the subset of Store behavior needed by tsinfer.
    """

    def __init__(
        self,
        path,
        *,
        map_size=None,
        readonly=False,
        subdir=False,
        lock=True,
        read_only=None,
    ):
        # zarr v2 LMDBStore used `readonly`; normalize to Store's `read_only`.
        if read_only is None:
            read_only = readonly
        super().__init__(read_only=read_only)
        self.path = str(pathlib.Path(path))
        self.subdir = subdir
        self.lock = lock
        self.map_size = map_size
        self._env = None

    def __eq__(self, value):
        return (
            isinstance(value, LMDBStore)
            and self.path == value.path
            and self.read_only == value.read_only
            and self.subdir == value.subdir
            and self.lock == value.lock
        )

    async def _open(self):
        await super()._open()
        kwargs = {
            "subdir": self.subdir,
            "readonly": self.read_only,
            "lock": self.lock,
            "create": not self.read_only,
            "readahead": False,
            "max_dbs": 1,
        }
        if self.map_size is not None:
            kwargs["map_size"] = int(self.map_size)
        try:
            self._env = lmdb.open(self.path, **kwargs)
        except lmdb.Error:
            self._is_open = False
            raise

    def with_read_only(self, read_only=False):
        return type(self)(
            self.path,
            map_size=self.map_size,
            read_only=read_only,
            subdir=self.subdir,
            lock=self.lock,
        )

    @property
    def supports_writes(self):
        return not self.read_only

    @property
    def supports_deletes(self):
        return not self.read_only

    @property
    def supports_listing(self):
        return True

    async def get(
        self,
        key: str,
        prototype: BufferPrototype = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        await self._ensure_open()
        with self._env.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
        if value is None:
            return None
        if byte_range is not None:
            if isinstance(byte_range, RangeByteRequest):
                value = value[byte_range.start : byte_range.end]
            elif isinstance(byte_range, OffsetByteRequest):
                value = value[byte_range.offset :]
            elif isinstance(byte_range, SuffixByteRequest):
                value = value[-byte_range.suffix :]
            else:
                raise TypeError(f"Unexpected byte_range {byte_range}")
        return prototype.buffer.from_bytes(value)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [await self.get(key, prototype, byte_range) for key, byte_range in key_ranges]

    async def exists(self, key: str) -> bool:
        await self._ensure_open()
        with self._env.begin(write=False) as txn:
            return txn.get(key.encode("utf-8")) is not None

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        await self._ensure_open()
        with self._env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), bytes(value.as_buffer_like()))

    async def delete(self, key: str) -> None:
        self._check_writable()
        await self._ensure_open()
        with self._env.begin(write=True) as txn:
            txn.delete(key.encode("utf-8"))

    async def list(self) -> AsyncIterator[str]:
        await self._ensure_open()
        with self._env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    yield key.decode("utf-8")

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        await self._ensure_open()
        prefix_b = prefix.encode("utf-8")
        with self._env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                found = cursor.set_range(prefix_b)
                while found:
                    key = cursor.key()
                    if not key.startswith(prefix_b):
                        break
                    yield key.decode("utf-8")
                    found = cursor.next()

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        seen = set()
        async for key in self.list_prefix(prefix):
            rel = key[len(prefix) :] if prefix else key
            part = rel.split("/", 1)[0]
            if part and part not in seen:
                seen.add(part)
                yield part

    def close(self):
        super().close()
        if self._env is not None:
            self._env.close()
            self._env = None
