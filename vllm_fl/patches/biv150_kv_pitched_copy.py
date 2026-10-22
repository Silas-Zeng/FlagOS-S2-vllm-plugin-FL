"""仅normal raw页SAVE的正等差规划/显式SDK调用；导入不载库、不导入框架或GPU。

R/原逻辑页/字节全部保留，不排序源页、不生成packing buffer。PreparedSave持有
Tensor及descriptor强引用；提交不是完成，调用者须保留它直到同stream完成事件。
"""
import ctypes
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

SIZE_MAX = (1 << 64) - 1
PAGE_BYTES = 16384
PAYLOAD_CAP = 1 << 30


def _integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= SIZE_MAX:
        raise ValueError("需要范围内Python整数")
    return value


def _multiply(a, b):
    a, b = _integer(a), _integer(b)
    if a and b > SIZE_MAX // a:
        raise ValueError("size_t乘法overflow")
    return a * b


@dataclass(frozen=True)
class Segment:
    source_start: int
    destination_start: int
    height: int
    source_step: int


@dataclass(frozen=True)
class SavePlan:
    source_pages: int
    page_bytes: int
    source_ids: tuple[int, ...]
    segments: tuple[Segment, ...]
    original_c95f_runs: int

    @property
    def logical_pages(self):
        return len(self.source_ids)

    def __post_init__(self):
        _integer(self.source_pages, 1)
        if self.page_bytes != PAGE_BYTES or type(self.source_ids) is not tuple or not self.source_ids:
            raise ValueError("仅nonempty normal16KiB页")
        _multiply(self.source_pages, self.page_bytes)
        _multiply(self.logical_pages, self.page_bytes)
        if (any(type(i) is not int or not 0 <= i < self.source_pages for i in self.source_ids)
                or len(set(self.source_ids)) != self.logical_pages
                or type(self.segments) is not tuple):
            raise ValueError("源页越界/重复或计划类型不符")
        expanded, next_destination = [], 0
        for segment in self.segments:
            if type(segment) is not Segment:
                raise ValueError("只接受Segment")
            start, destination, height, step = (segment.source_start, segment.destination_start,
                                                  segment.height, segment.source_step)
            for value in (start, destination): _integer(value)
            _integer(height, 1)
            _integer(step, 1)
            if destination != next_destination or height > self.logical_pages - destination:
                raise ValueError("目的页必须原序完整无洞/无重叠")
            last = start + _multiply(height - 1, step)
            if last >= self.source_pages or last > SIZE_MAX:
                raise ValueError("正等差源越界")
            expanded.extend(start + j * step for j in range(height))
            next_destination += height
        if tuple(expanded) != self.source_ids or next_destination != self.logical_pages:
            raise ValueError("计划未精确保留全部原逻辑页")
        expected_runs = 1 + sum(b != a + 1 for a, b in zip(self.source_ids, self.source_ids[1:]))
        if type(self.original_c95f_runs) is not int or self.original_c95f_runs != expected_runs:
            raise ValueError("原连续段计数不符")


def make_plan(source_ids, source_pages, page_bytes=PAGE_BYTES):
    """保原c95f长连续段；仅对相邻singleton区左到右取最长正恒差，不排序。"""
    ids = tuple(source_ids)
    if not ids or any(type(i) is not int for i in ids):
        raise ValueError("nonempty整数页号")
    runs = []
    for index, page in enumerate(ids):
        if runs and page == ids[index - 1] + 1:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((index, 1))
    segments, r = [], 0
    while r < len(runs):
        start, count = runs[r]
        if count > 1:
            segments.append(Segment(ids[start], start, count, 1))
            r += 1
            continue
        end = r + 1
        while end < len(runs) and runs[end][1] == 1:
            end += 1
        limit, i = runs[end - 1][0] + 1, start
        while i < limit:
            step = ids[i + 1] - ids[i] if i + 1 < limit else 0
            height = 1
            if step > 0:
                height = 2
                while i + height < limit and ids[i + height] - ids[i + height - 1] == step:
                    height += 1
            segments.append(Segment(ids[i], i, height, step if height > 1 else 1))
            i += height
        r = end
    return SavePlan(source_pages, page_bytes, ids, tuple(segments), len(runs))


class _Descriptor(ctypes.Structure):
    _fields_ = [("source", ctypes.c_void_p), ("destination", ctypes.c_void_p),
                ("source_available", ctypes.c_size_t), ("destination_available", ctypes.c_size_t),
                ("width", ctypes.c_size_t), ("height", ctypes.c_size_t),
                ("source_pitch", ctypes.c_size_t), ("destination_pitch", ctypes.c_size_t),
                ("kind", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]


@dataclass(frozen=True)
class PreparedSave:
    descriptors: object
    keepalive: tuple
    submitted_bytes: int
    linear_calls: int
    pitched_calls: int
    fallback_pitched_segments: int
    original_c95f_calls: int


def _tensor_range(torch, tensor, shape, device):
    if (type(tensor) is not torch.Tensor or str(tensor.dtype) != "torch.uint8"
            or str(tensor.device) != device or tuple(tensor.shape) != shape
            or tuple(tensor.stride()) != (PAGE_BYTES, 1) or not tensor.is_contiguous()
            or device == "cpu" and not tensor.is_pinned()):
        raise ValueError("只支持normal连续rawuint8/cuda0与pinnedCPU目的")
    storage = tensor.untyped_storage()
    base, size, pointer = _integer(storage.data_ptr(), 1), _integer(storage.nbytes()), _integer(tensor.data_ptr(), 1)
    offset = _integer(tensor.storage_offset())  # uint8元素即字节；data_ptr须已含storage offset。
    size_view = _multiply(shape[0], shape[1])
    if base > SIZE_MAX - size or pointer != base + offset or offset > size or size_view > size - offset:
        raise ValueError("Tensor真实storage/view界或offset不符")
    return pointer, size_view, base


def prepare_save(sources, destinations, plan, *, max_pitch):
    """只读全部Tensor描述符并强持；任何copy前完成全batch验证，无GPU分配/值读取。"""
    if type(plan) is not SavePlan:
        raise TypeError("需要已验证SavePlan")
    _integer(max_pitch, PAGE_BYTES)
    torch = sys.modules.get("torch")
    if torch is None:
        raise RuntimeError("只供已加载torch的设备worker，不主动import框架")
    sources, destinations = tuple(sources), tuple(destinations)
    if len(sources) not in (1, 42) or len(destinations) != len(sources):
        raise ValueError("仅单层门或实际normal42层")
    nbytes = _multiply(_multiply(plan.logical_pages, PAGE_BYTES), len(sources))
    if nbytes > PAYLOAD_CAP:
        raise ValueError("本候选SAVE payload不超过1GiB")
    descriptors, source_storages, destination_storages = [], set(), set()
    linear = pitched = fallbacks = 0
    for source, destination in zip(sources, destinations):
        src, src_bytes, src_storage = _tensor_range(torch, source, (plan.source_pages, PAGE_BYTES), "cuda:0")
        dst, dst_bytes, dst_storage = _tensor_range(torch, destination, (plan.logical_pages, PAGE_BYTES), "cpu")
        if src_storage in source_storages or dst_storage in destination_storages:
            raise ValueError("不支持跨层source/host storage alias")
        source_storages.add(src_storage)
        destination_storages.add(dst_storage)
        for segment in plan.segments:
            s, d, height, step = segment.source_start, segment.destination_start, segment.height, segment.source_step
            pitch = _multiply(step, PAGE_BYTES)
            # 超真实设备pitch限制在任何提交前拆回原单页线性；不绕SDK报错后重试。
            fallback = height > 1 and step > 1 and pitch > max_pitch
            if fallback:
                fallbacks += 1
            pieces = ((s + j * step, d + j, 1, 1) for j in range(height)) if fallback else ((s, d, height, step),)
            for start, target, rows, stride in pieces:
                source_offset, destination_offset = _multiply(start, PAGE_BYTES), _multiply(target, PAGE_BYTES)
                kind = 2 if rows > 1 and stride > 1 else 1
                width = PAGE_BYTES if kind == 2 else _multiply(rows, PAGE_BYTES)
                h = rows if kind == 2 else 1
                sp = _multiply(stride, PAGE_BYTES) if kind == 2 else width
                dp = PAGE_BYTES if kind == 2 else width
                descriptors.append(_Descriptor(src + source_offset, dst + destination_offset,
                    src_bytes - source_offset, dst_bytes - destination_offset, width, h, sp, dp, kind, 0))
                linear += kind == 1
                pitched += kind == 2
    if len(descriptors) > 65536:
        raise ValueError("batch descriptor数超独立桥界")
    array = (_Descriptor * len(descriptors))(*descriptors)
    return PreparedSave(array, sources + destinations, nbytes, linear, pitched, fallbacks,
                        plan.original_c95f_runs * len(sources))


class CopySubmissionError(RuntimeError):
    def __init__(self, code, submitted_calls, total_calls):
        self.code, self.submitted_calls, self.total_calls = code, submitted_calls, total_calls
        self.partial = submitted_calls > 0
        super().__init__(f"native SAVE提交失败code={code}, submitted={submitted_calls}/{total_calls}; 必须settle，禁止旧路静默重试")


class Bridge:
    """仅显式设备worker构造时加载已pin.so；完成事件/所有keepalive由调用者持有。"""
    def __init__(self, path, *, expected_sha256):
        path = Path(path)
        if (path.is_symlink() or not path.is_file() or type(expected_sha256) is not str
                or len(expected_sha256) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256):
            raise RuntimeError("只加载控制器编译/归档/固定SHA的普通bridge.so")
        if ctypes.sizeof(ctypes.c_size_t) != 8 or ctypes.sizeof(_Descriptor) != 72:
            raise RuntimeError("需要已核64位size_t/descriptor ABI")
        self._library = ctypes.CDLL(str(path.resolve()))
        self._library.biv150_pitched_copy_abi.restype = ctypes.c_uint32
        self._library.biv150_pitched_copy_abi.argtypes = []
        self._library.biv150_pitched_copy_descriptor_size.restype = ctypes.c_size_t
        self._library.biv150_pitched_copy_descriptor_size.argtypes = []
        if self._library.biv150_pitched_copy_abi() != 1 or self._library.biv150_pitched_copy_descriptor_size() != 72:
            raise RuntimeError("桥版本/descriptor大小不符")
        self._pitch = self._library.biv150_pitched_copy_max_pitch
        self._pitch.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        self._pitch.restype = ctypes.c_int
        self._submit = self._library.biv150_pitched_save_batch
        self._submit.argtypes = [ctypes.POINTER(_Descriptor), ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self._submit.restype = ctypes.c_int

    def max_pitch(self, device=0):
        if type(device) is not int or device != 0:
            raise ValueError("仅cuda:0")
        value = ctypes.c_size_t()
        code = self._pitch(device, ctypes.byref(value))
        if code or value.value < PAGE_BYTES:
            raise CopySubmissionError(code or -1002, 0, 0)
        return value.value

    def submit(self, prepared, *, stream_handle):
        if type(prepared) is not PreparedSave:
            raise TypeError("需要首提交前完成的PreparedSave")
        _integer(stream_handle)
        submitted = ctypes.c_size_t()
        code = self._submit(prepared.descriptors, len(prepared.descriptors), ctypes.c_void_p(stream_handle), ctypes.byref(submitted))
        if code or submitted.value != len(prepared.descriptors):
            raise CopySubmissionError(code or -1006, submitted.value, len(prepared.descriptors))
        return {"submitted_bytes": prepared.submitted_bytes, "copy_calls": submitted.value,
                "linear_calls": prepared.linear_calls, "pitched_calls": prepared.pitched_calls,
                "fallback_pitched_segments": prepared.fallback_pitched_segments,
                "original_c95f_calls": prepared.original_c95f_calls, "completion_ack": False}
