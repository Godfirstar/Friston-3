"""按所需切片读取 Safetensors，降低 Windows 加载大模型时的暂存内存。

由本模块 checkpoint_reader 临时接到 loader 的 safe_open 调用处；不是训练算法。
读文件头确定 tensor 的形状/类型/偏移，再只申请本次切片的 bytearray。
通过读取接口替换控制临时内存；loader.py 将实际权重与模型参数对应。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from math import prod
import os
import struct
import sys
from pathlib import Path
from threading import RLock, get_ident
from typing import Any
# 此模块实现loader所需的一小部分Safetensors读取接口；不会创建π₀.₅模型或执行训练。
# PyTorch在真正_tensor读取时才导入，查看其他轻量功能不必提前加载torch。


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
    # 【调用方】json.loads的object_pairs_hook；pairs是按原文本顺序排列的(key,value)列表。
    # 返回普通dict；若JSON重复声明同一键则抛错，避免后一个值静默覆盖前一个。
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Safetensors header key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _TensorInfo:
    # dataclass自动生成初始化等方法；frozen=True禁止给字段重新赋值，保存只读元数据。
    # shape是各轴长度tuple，start/end是相对数据区的字节偏移，不是tensor的数值。
    shape: tuple[int, ...]
    dtype: str
    start: int
    end: int


class _TensorSlice:
    # 【创建方】BoundedSafetensors.get_slice；保存读取器与元数据，直到[]切片才读字节。
    def __init__(self, reader, info):
        self._reader = reader
        self._info = info

    def get_shape(self):
        return list(self._info.shape)

    def get_dtype(self):
        return self._info.dtype

    def __getitem__(self, index):
        """The streaming loader requests only contiguous first-dimension rows."""
        # Python看到对象[start:stop]会调用__getitem__(slice(start,stop,None))。
        # 输入仅接受第一轴连续切片；返回这段的CPU Torch Tensor，不是整个权重文件。
        shape = self._info.shape
        if not shape:
            raise IndexError("A scalar Safetensors value has no row dimension")
        if not isinstance(index, slice):
            raise TypeError("Checkpoint slices require a first-dimension slice")
        start, stop, step = index.indices(shape[0])
        # slice.indices按第一轴长度规范化省略/负下标；返回可实际使用的三元组。
        if step != 1:
            raise ValueError("Checkpoint slices require contiguous rows (step=1)")
        # 例如 [out,in] 权重每一行有 in 个元素；元素字节数由 dtype 决定。
        row_bytes = prod(shape[1:]) * _DTYPES[self._info.dtype][1]
        result_shape = (max(stop - start, 0), *shape[1:])
        # tuple里的*解包剩余维度：原[out,in]切若干行后变成[行数,in]。
        return self._reader._tensor(self._info, result_shape, start * row_bytes)


class BoundedSafetensors:
    """Internal CPU/PyTorch reader; data buffers cover only requested slices.

    The legacy loader already bounds each request by its row chunk size.
    Unlike safe_open(backend='pread'), construction never maps the data file.
    """

    def __init__(self, filename):
        # 【创建方】checkpoint_reader.scoped_open；filename是safetensors路径。
        # self是当前读取器；构造只开文件并解析头，尚未把所有模型权重读入内存。
        if sys.byteorder != "little":
            raise ValueError("This checkpoint reader requires a little-endian host")
        self._lock = RLock()
        self._file = open(filename, "rb", buffering=0)
        # Python open的rb为只读二进制；buffering=0绕过Python文件缓冲，按请求读入。
        try:
            self._read_header()
        except BaseException:
            self._file.close()
            raise

    def _read_header(self):
        # 【调用方】__init__；填充self._tensors等元数据，无业务返回值。
        # 文件布局：8字节头长度、JSON头、tensor数据区；偏移都相对数据区解释。
        file_bytes = os.fstat(self._file.fileno()).st_size
        prefix = self._file.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated Safetensors header length")
        header_bytes = struct.unpack("<Q", prefix)[0]
        # struct将二进制解释为数值；<表示小端，Q为8字节无符号整数；[0]取唯一返回值。
        # Match the native reader's header bound; the data section is unbounded.
        if header_bytes > 100_000_000 or header_bytes > file_bytes - 8:
            raise ValueError("Invalid Safetensors header length")
        raw = self._file.read(header_bytes)
        if len(raw) != header_bytes:
            raise ValueError("Truncated Safetensors header")
        header = json.loads(raw, object_pairs_hook=_unique_object)
        # JSON解码调用上面的_unique_object检查重复键；raw是头部bytes，可由json读取。
        if not isinstance(header, dict):
            raise ValueError("Safetensors header must be an object")
        self._data_start = 8 + header_bytes
        # 以后读某tensor时：真实文件偏移=数据区起点+tensor起点+切片相对偏移。
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
            # 把JSON的shape list转tuple后装进只读dataclass；字典按tensor名字索引。
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
        # 【调用方】keys/metadata/get_slice/get_tensor等；提前说明读取器已关闭的错误。
        if self._file.closed:
            raise ValueError("Checkpoint reader is closed")

    def _tensor(self, info, shape, relative_offset=0):
        # 【调用方】get_tensor或_TensorSlice.__getitem__；info为该tensor头，shape为本次切片。
        # 返回CPU tensor；relative_offset是切片相对该tensor首字节的位移，默认完整读取。
        import torch

        dtype_name, item_bytes = _DTYPES[info.dtype]
        # 字典值为(对应torch类型名,每元素字节数)，例如F32→("float32",4)。
        dtype = getattr(torch, dtype_name, None)
        # getattr动态取得torch.float32等对象；当前PyTorch不支持时返回None。
        if dtype is None:
            raise ValueError(f"Installed PyTorch does not support {info.dtype}")
        nbytes = prod(shape) * item_bytes
        # math.prod把各轴长度相乘得到元素数量，再乘元素字节宽度得到要分配的bytes。
        with self._lock:
            self._ensure_open()
            if nbytes == 0:
                return torch.empty(shape, dtype=dtype, device="cpu")
            buffer = bytearray(nbytes)  # 本次切片大小，不是整个 checkpoint 大小。
            view = memoryview(buffer)
            # memoryview创建同一bytearray的视图，不再复制整块；下面可直接把文件读进这块内存。
            self._file.seek(self._data_start + info.start + relative_offset)
            # seek移动文件指针到本次切片起点；它本身不读取/分配tensor。
            done = 0
            while done < nbytes:
                count = self._file.readinto(view[done:])
                # readinto把文件字节写入未填满区，返回本次实际读取量；短读时继续while。
                if not count:
                    raise ValueError("Checkpoint data was truncated while reading")
                done += count
        # frombuffer retains the writable buffer, so close() does not invalidate it.
        # frombuffer 共享这块 CPU 存储；reshape 恢复维度，后续 loader 再复制到目标设备。
        return torch.frombuffer(buffer, dtype=dtype).reshape(shape)
        # frombuffer让CPU Tensor共享buffer；reshape只恢复约定轴长，通常无需再复制。
        # 后续loader会再把这份CPU切片复制到目标参数的CPU/GPU存储。

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
        # 【调用方】loader.stream_load；返回可被[start:stop]访问的懒读取对象。
        self._ensure_open()
        return _TensorSlice(self, self._tensors[name])

    def get_tensor(self, name):
        # 【调用方】loader需要完整tensor时；按name取得元数据，再走统一_tensor实现。
        self._ensure_open()
        info = self._tensors[name]
        return self._tensor(info, info.shape)

    def close(self):
        # 关闭文件句柄；已返回Tensor仍持有自己的buffer引用，不因文件关闭而失效。
        with self._lock:
            self._file.close()

    def __enter__(self):
        # 支持with reader as handle：进入时验证文件仍开着，as绑定此处return的self。
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # with主体结束或抛异常都会调用；三个参数描述异常，本实现统一关闭、不吞异常。
        self.close()

    def __del__(self):
        # 对象销毁时的兜底；正常调用仍应使用with/close，不依赖垃圾回收时机。
        handle = getattr(self, "_file", None)
        if handle is not None:
            handle.close()


def read_json(path: Path) -> Any:
    # 【调用方】trainer/本模块的配置读取；Path来自pathlib，-> Any说明JSON结果类型可变。
    # read_text读成str；json.loads把JSON文本解析成dict/list等；utf-8-sig兼容文件头BOM。
    return json.loads(path.read_text(encoding='utf-8-sig'))


_checkpoint_reader_lock = RLock()
# 模块级对象，同一进程导入本包的调用者共用；可重入锁协调临时替换函数的生命周期。


@contextmanager
# 装饰器把下面含yield的生成器函数转换成支持with的上下文管理器。
def checkpoint_reader():
    """Use bounded CPU file reads through the loader's existing reader interface.

    Native safe_open can map the whole checkpoint even with backend='pread'.
    Scope the replacement to the loading thread; other frameworks, devices,
    and explicitly requested mmap calls keep their native implementation.
    """
    # 【调用方】训练load_training_model、部署Pi05Policy；用with checkpoint_reader():包加载段。
    # 没有业务返回值；作用是临时改变safetensors.safe_open，再保证恢复原函数。
    import safetensors

    with _checkpoint_reader_lock:
        # with进入时获取锁，退出时释放；异常也会退出，不必手写acquire/release配对。
        original = safetensors.safe_open
        # 没有括号：保存函数对象引用，不是在这里读文件。
        owner = get_ident()
        # get_ident来自threading，记录此次加载线程ID；其他线程不采用本替换读取实现。
        active = True
        unspecified = object()
        # 创建唯一哨兵对象，用is区分“backend没传”和“backend显式传了其他值”。

        def scoped_open(filename, framework, device="cpu", *, backend=unspecified):
            # 内部函数是闭包，能读到外层original/owner/active/unspecified。
            # *后的backend只能按名传；默认哨兵表示调用方省略了这个参数。
            if (active and get_ident() == owner and framework == "pt" and device == "cpu"
                    and (backend is unspecified or backend == "pread")):
                return BoundedSafetensors(filename)
                # 类定义在weights.py；只在本线程、CPU/PyTorch、适用backend时替换。
            if backend is unspecified:
                return original(filename, framework, device)
            return original(filename, framework, device, backend=backend)

        safetensors.safe_open = scoped_open
        # 临时替换模块属性（monkey patch）；loader内部再取safe_open时会得到包装函数。
        try:
            # @contextmanager：with 的主体在 yield 期间执行；返回后继续做 finally 清理。
            yield
        finally:
            # 即使装载抛异常，也恢复原函数；不把本项目的读取方式永久留给其他调用者。
            active = False
            # 即使有别处保留了scoped_open引用，active=False也使它回退到原读取函数。
            safetensors.safe_open = original
