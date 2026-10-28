"""BI-V150 BF16 N64/S2 与 N32/S4 原生页循环，其余保留原路。"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from vllm_fl.compile_semantics import native_gqa_library_path, native_model_matches

_ENABLE = "VLLM_FL_BIV150_NATIVE_GQA"
_LOADED = False


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Runtime:
    def __init__(self, torch, library):
        self.torch = torch
        self.device = torch.cuda.current_device()
        self.library_sha256 = _sha(library)
        self.selected = set()
        self.captured = set()

    def _eligible(self, grid, kw):
        if tuple(grid) not in ((64, 2, 2), (32, 2, 4)):
            return False
        rows, _, segments = grid
        values = {"queries_per_kv": 8, "ROWS": 8, "HEAD_SIZE": 128,
                  "PAGE_SIZE": 16, "SEGMENTS": segments, "num_warps": 1,
                  "num_stages": 1, "k_stride_block": 8192,
                  "k_stride_slot": 256, "k_stride_head": 128,
                  "v_stride_block": 8192, "v_stride_slot": 256,
                  "v_stride_head": 128}
        if any(kw.get(key) != value for key, value in values.items()):
            return False
        q, k, v = (kw.get(key) for key in ("query_ptr", "key_cache_ptr", "value_cache_ptr"))
        if q is None or k is None or v is None:
            return False
        if (q.dtype != self.torch.bfloat16 or tuple(q.shape) != (rows, 16, 128)
                or tuple(q.stride()) not in ((2048, 128, 1), (2560, 128, 1))
                or not q.is_cuda or q.device.index != self.device
                or q.requires_grad):
            return False
        for tensor in (k, v):
            if (tensor.dtype != q.dtype or tensor.device != q.device
                    or len(tensor.shape) != 4 or tuple(tensor.shape[1:]) != (16, 2, 128)
                    or tuple(tensor.stride()) != (8192, 256, 128, 1)):
                return False
        if k.shape[0] != v.shape[0]:
            return False
        if k.data_ptr() % 64 or v.data_ptr() % 64:
            return False
        lengths, table = kw.get("seq_lens_ptr"), kw.get("block_table_ptr")
        if (lengths is None or table is None or lengths.dtype != self.torch.int32
                or tuple(lengths.shape) != (rows,) or not lengths.is_contiguous()
                or table.dtype != self.torch.int32 or len(table.shape) != 2
                or table.shape[0] != rows or table.stride(1) != 1):
            return False
        partial = kw.get("segm_output_ptr")
        if (partial is None or partial.dtype != self.torch.float32
                or len(partial.shape) != 4 or partial.shape[0] < rows
                or tuple(partial.shape[1:]) != (16, segments, 128)
                or partial.stride(3) != 1 or partial.stride(2) != 128):
            return False
        maximum, summation = kw.get("segm_max_ptr"), kw.get("segm_expsum_ptr")
        for tensor in (maximum, summation):
            if (tensor is None or tensor.dtype != self.torch.float32
                    or len(tensor.shape) != 3 or tensor.shape[0] < rows
                    or tuple(tensor.shape[1:]) != (16, segments) or tensor.stride(2) != 1):
                return False
        if maximum.stride() != summation.stride():
            return False
        if lengths.device != q.device or table.device != q.device:
            return False
        for pointer, dim, key in (
                ("query_ptr", 0, "query_stride_token"), ("query_ptr", 1, "query_stride_head"),
                ("block_table_ptr", 0, "block_table_stride"),
                ("segm_output_ptr", 0, "segment_stride_token"),
                ("segm_output_ptr", 1, "segment_stride_head"),
                ("segm_output_ptr", 2, "segment_stride_segment"),
                ("segm_max_ptr", 0, "max_stride_token"), ("segm_max_ptr", 1, "max_stride_head"),
                ("segm_expsum_ptr", 0, "max_stride_token"), ("segm_expsum_ptr", 1, "max_stride_head")):
            tensor = kw.get(pointer)
            if tensor is None or tensor.device != q.device or tensor.stride(dim) != kw.get(key):
                return False
        return True

    def try_produce(self, grid, kw):
        try:
            eligible = self._eligible(grid, kw)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            eligible = False
        if not eligible:
            return False
        # Once an eligible native launch starts, never hide its error by running
        # the old kernel over partially written scratch.
        rows, _, segments = grid
        operator = (self.torch.ops.corex_native_gqa_s2_probe.produce if segments == 2
                    else self.torch.ops.corex_native_gqa_s4_probe.produce)
        operator(
            kw["query_ptr"], kw["key_cache_ptr"], kw["value_cache_ptr"],
            kw["seq_lens_ptr"], kw["block_table_ptr"], kw["segm_output_ptr"],
            kw["segm_max_ptr"], kw["segm_expsum_ptr"], kw["softmax_scale_log2e"])
        shape = (rows, segments)
        if not (shape in self.selected and shape in self.captured):
            capturing = self.torch.cuda.is_current_stream_capturing()
            observed = self.captured if capturing else self.selected
            if shape not in observed:
                record = dict(rows=rows, segments=segments, grid=list(grid), threads=64,
                              q_stride=list(kw["query_ptr"].stride()),
                              k_stride=list(kw["key_cache_ptr"].stride()),
                              v_stride=list(kw["value_cache_ptr"].stride()),
                              library_sha256=self.library_sha256,
                              overlap_v=True, extra_KV_bytes=0,
                              proof="successful native launch; not a request coverage ratio")
                if capturing:
                    self.captured.add(shape)
                    print("BIV150_NATIVE_GQA_CAPTURED", json.dumps(record, sort_keys=True), flush=True)
                else:
                    self.selected.add(shape)
                    print("BIV150_NATIVE_GQA_SELECTED", json.dumps(record, sort_keys=True), flush=True)
        return True


def create_runtime():
    global _LOADED
    if os.environ.get(_ENABLE, "1") != "1":
        return None
    from vllm.platforms import current_platform
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        return None
    from vllm.config import get_current_vllm_config
    import torch
    try:
        model = get_current_vllm_config().model_config
        if not native_model_matches(model.hf_config) or model.dtype != torch.bfloat16:
            return None
    except (AttributeError, AssertionError, RuntimeError):
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if "BI-V150" not in properties.name or properties.multi_processor_count != 16:
        return None
    library = native_gqa_library_path()
    if not library.is_file():
        raise RuntimeError("请先运行 python tools/build_biv150_native_gqa.py 构建原生 GQA")
    if not _LOADED:
        torch.ops.load_library(str(library))

        def fake(q, k, v, lengths, table, partial, maximum, summation, scale):
            return None
        for name in ("corex_native_gqa_s2_probe::produce", "corex_native_gqa_s4_probe::produce"):
            torch.library.register_fake(name)(fake)
        _LOADED = True
    runtime = _Runtime(torch, library)
    print("BIV150_NATIVE_GQA_INSTALL", json.dumps(dict(
        installed=True, supported_shapes=[[64, 2], [32, 4]], q_strides=[[2048, 128, 1], [2560, 128, 1]],
        library_sha256=runtime.library_sha256, compile_at_runtime=False,
        overlap_v=True, extra_KV_bytes=0), sort_keys=True), flush=True)
    return runtime
