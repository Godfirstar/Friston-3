"""Read CPU Safetensors slices without mapping the full checkpoint on Windows."""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import prod
import os
import struct
import sys
from threading import RLock


# Names and storage widths from the Safetensors dtype table. Torch stays lazy.
_DTYPES = {
    "F64": ("float64", 8), "F32": ("float32", 4),
    "F16": ("float16", 2), "BF16": ("bfloat16", 2),
    "I64": ("int64", 8), "I32": ("int32", 4),
    "I16": ("int16", 2), "I8": ("int8", 1),
    "U64": ("uint64", 8), "U32": ("uint32", 4),
    "U16": ("uint16", 2), "U8": ("uint8", 1),
    "BOOL": ("bool", 1), "C64": ("complex64", 8),
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E4M3FNUZ": ("float8_e4m3fnuz", 1),
    "F8_E5M2": ("float8_e5m2", 1),
    "F8_E5M2FNUZ": ("float8_e5m2fnuz", 1),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Safetensors header key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _TensorInfo:
    shape: tuple[int, ...]
    dtype: str
    start: int
    end: int


class _TensorSlice:
    def __init__(self, reader, info):
        self._reader = reader
        self._info = info

    def get_shape(self):
        return list(self._info.shape)

    def get_dtype(self):
        return self._info.dtype

    def __getitem__(self, index):
        """The streaming loader requests only contiguous first-dimension rows."""
        shape = self._info.shape
        if not shape:
            raise IndexError("A scalar Safetensors value has no row dimension")
        if not isinstance(index, slice):
            raise TypeError("Checkpoint slices require a first-dimension slice")
        start, stop, step = index.indices(shape[0])
        if step != 1:
            raise ValueError("Checkpoint slices require contiguous rows (step=1)")
        row_bytes = prod(shape[1:]) * _DTYPES[self._info.dtype][1]
        result_shape = (max(stop - start, 0), *shape[1:])
        return self._reader._tensor(self._info, result_shape, start * row_bytes)


class BoundedSafetensors:
    """Internal CPU/PyTorch reader; data buffers cover only requested slices.

    The legacy loader already bounds each request by its row chunk size.
    Unlike safe_open(backend='pread'), construction never maps the data file.
    """

    def __init__(self, filename):
        if sys.byteorder != "little":
            raise ValueError("This checkpoint reader requires a little-endian host")
        self._lock = RLock()
        self._file = open(filename, "rb", buffering=0)
        try:
            self._read_header()
        except BaseException:
            self._file.close()
            raise

    def _read_header(self):
        file_bytes = os.fstat(self._file.fileno()).st_size
        prefix = self._file.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated Safetensors header length")
        header_bytes = struct.unpack("<Q", prefix)[0]
        # Match the native reader's header bound; the data section is unbounded.
        if header_bytes > 100_000_000 or header_bytes > file_bytes - 8:
            raise ValueError("Invalid Safetensors header length")
        raw = self._file.read(header_bytes)
        if len(raw) != header_bytes:
            raise ValueError("Truncated Safetensors header")
        header = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(header, dict):
            raise ValueError("Safetensors header must be an object")
        self._data_start = 8 + header_bytes
        data_bytes = file_bytes - self._data_start
        metadata = header.pop("__metadata__", None)
        if metadata is not None and (not isinstance(metadata, dict) or
                any(not isinstance(k, str) or not isinstance(v, str)
                    for k, v in metadata.items())):
            raise ValueError("Safetensors metadata must contain string values")
        self._metadata = metadata
        self._tensors = {}
        for name, entry in header.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid tensor header: {name!r}")
            dtype, shape, offsets = (entry.get(k) for k in ("dtype", "shape", "data_offsets"))
            if not isinstance(dtype, str) or dtype not in _DTYPES:
                raise ValueError(f"Unsupported Safetensors dtype for {name!r}: {dtype!r}")
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError(f"Invalid tensor shape: {name!r}")
            if (not isinstance(offsets, list) or len(offsets) != 2 or
                    any(type(d) is not int or d < 0 for d in offsets)):
                raise ValueError(f"Invalid tensor offsets: {name!r}")
            start, end = offsets
            if end < start or end > data_bytes or end - start != prod(shape) * _DTYPES[dtype][1]:
                raise ValueError(f"Tensor byte range does not match its shape/dtype: {name!r}")
            self._tensors[name] = _TensorInfo(tuple(shape), dtype, start, end)
        self._offset_keys = sorted(self._tensors, key=lambda name: (
            self._tensors[name].start, self._tensors[name].end, name))
        cursor = 0
        for name in self._offset_keys:
            info = self._tensors[name]
            if info.start != cursor:
                raise ValueError(f"Non-contiguous or overlapping tensor data: {name!r}")
            cursor = info.end
        if cursor != data_bytes:
            raise ValueError("Safetensors data section contains unclaimed bytes")

    def _ensure_open(self):
        if self._file.closed:
            raise ValueError("Checkpoint reader is closed")

    def _tensor(self, info, shape, relative_offset=0):
        import torch

        dtype_name, item_bytes = _DTYPES[info.dtype]
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Installed PyTorch does not support {info.dtype}")
        nbytes = prod(shape) * item_bytes
        with self._lock:
            self._ensure_open()
            if nbytes == 0:
                return torch.empty(shape, dtype=dtype, device="cpu")
            buffer = bytearray(nbytes)
            view = memoryview(buffer)
            self._file.seek(self._data_start + info.start + relative_offset)
            done = 0
            while done < nbytes:
                count = self._file.readinto(view[done:])
                if not count:
                    raise ValueError("Checkpoint data was truncated while reading")
                done += count
        # frombuffer retains the writable buffer, so close() does not invalidate it.
        return torch.frombuffer(buffer, dtype=dtype).reshape(shape)

    def keys(self):
        self._ensure_open()
        return sorted(self._tensors)

    def offset_keys(self):
        self._ensure_open()
        return list(self._offset_keys)

    def metadata(self):
        self._ensure_open()
        return None if self._metadata is None else dict(self._metadata)

    def get_slice(self, name):
        self._ensure_open()
        return _TensorSlice(self, self._tensors[name])

    def get_tensor(self, name):
        self._ensure_open()
        info = self._tensors[name]
        return self._tensor(info, info.shape)

    def close(self):
        with self._lock:
            self._file.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        handle = getattr(self, "_file", None)
        if handle is not None:
            handle.close()
