"""Read the large local protocol-5 pickle without copying its 25 GB array.

Only load trusted project files: this is a pickle loader, not a security boundary.
The Python unpickler is used because its BYTEARRAY8 handler can retain a read-only
view of a memory-mapped file. Small metadata reads retain ordinary bytes semantics.
"""

import hashlib
import mmap
import pickle
import struct
from pathlib import Path


class MappedReader:
    def __init__(self, path):
        self.file = open(path, "rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.position = 0

    def read(self, size=-1):
        if size < 0:
            size = len(self.mapping) - self.position
        start = self.position
        self.position = min(start + size, len(self.mapping))
        if size > 1024 * 1024:
            return memoryview(self.mapping)[start:self.position]
        return self.mapping[start:self.position]

    def readline(self):
        end = self.mapping.find(b"\n", self.position)
        end = len(self.mapping) if end < 0 else end + 1
        start, self.position = self.position, end
        return self.mapping[start:end]


class MappedUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def load_bytearray8(self):
        size, = struct.unpack("<Q", self.read(8))
        value = self.read(size)
        if len(value) != size:
            raise pickle.UnpicklingError("Truncated BYTEARRAY8 payload")
        self.append(value)

    dispatch[pickle.BYTEARRAY8[0]] = load_bytearray8


def load_modes(path):
    reader = MappedReader(path)
    modes = MappedUnpickler(reader).load()
    # Retain reader for the lifetime of any NumPy view of the mapped data.
    return modes, reader


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
