"""Audited in-memory prefill normalization clone; serving registration is separate."""

import ast
import functools
import hashlib
import importlib
import inspect
import linecache
import re
import types
from pathlib import Path

_AUDITED = {
    "vllm.v1.attention.ops.triton_unified_attention": "54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f",
    "vllm.v1.attention.ops.triton_attention_helpers": "59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1",
}


_STRIDES = ('block_table_stride', 'query_stride_0', 'query_stride_1',
    'output_stride_0', 'output_stride_1', 'stride_k_cache_0', 'stride_k_cache_1',
    'stride_k_cache_2', 'stride_v_cache_0', 'stride_v_cache_1', 'stride_v_cache_2')


def _specialize_strides(source):
    header, body = source.split('\n):\n', 1)
    original_header, original_body = header, body
    for name in _STRIDES:
        old = name + ': tl.int64'
        if header.count(old) != 1:
            raise ValueError('未识别唯一stride签名：' + name)
        header = header.replace(old, name + ': tl.constexpr', 1)
        body = re.sub(r'\b' + name + r'\b', name + '_i64', body)
    aliases = ''.join('    ' + name + '_i64 = tl.full((), ' + name + ', tl.int64)\n' for name in _STRIDES)
    header = header.replace('def kernel_biv150_prefill_normalized(', 'def kernel_biv150_prefill_stride(', 1)
    candidate = header + '\n):\n' + aliases + body
    restored_header = header.replace('def kernel_biv150_prefill_stride(', 'def kernel_biv150_prefill_normalized(', 1)
    restored_body = body
    for name in _STRIDES:
        restored_header = restored_header.replace(name + ': tl.constexpr', name + ': tl.int64', 1)
        restored_body = re.sub(r'\b' + name + r'_i64\b', name, restored_body)
    if (restored_header, restored_body) != (original_header, original_body):
        raise AssertionError('stride变换不能逐字逆回')
    ast.parse(candidate)
    return candidate


def make_normalized(attention):
    audited = {}
    for name, expected in _AUDITED.items():
        module = importlib.import_module(name)
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Unaudited attention source: " + name)
        audited[name] = actual
    kernel = attention.kernel_unified_attention
    if not hasattr(kernel, "fn"):
        raise RuntimeError("Unrecognized prefill kernel wrapper")
    source = inspect.getsource(kernel.fn)
    old = "L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)"
    if source.count(old) != 1:
        raise RuntimeError("Unexpected cumulative softmax initialization")
    source = source.replace(old, "L = tl.zeros([BLOCK_M], dtype=tl.float32)")
    source = source.replace(
        "def kernel_unified_attention(", "def kernel_biv150_prefill_normalized(", 1
    )
    source = _specialize_strides(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    filename = "<biv150-prefill-normalized-" + digest + ".py>"
    # Standard inspect/linecache support lets Triton read the exact pinned clone
    # without writing generated source into the installed package or checkout.
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = dict(vars(attention))
    namespace["__name__"] = __name__
    exec(compile(source, filename, "exec"), namespace)
    normalized_kernel = namespace["kernel_biv150_prefill_stride"]
    raw = inspect.unwrap(attention.unified_attention)
    if raw.__name__ != "unified_attention":
        raise RuntimeError("Unexpected attention wrapper")
    globals_copy = dict(raw.__globals__)
    globals_copy["kernel_unified_attention"] = normalized_kernel
    function = types.FunctionType(
        raw.__code__, globals_copy, raw.__name__, raw.__defaults__, raw.__closure__
    )
    function.__kwdefaults__ = raw.__kwdefaults__
    functools.update_wrapper(function, raw)
    return function, dict(audited=audited, kernel_sha256=digest, initial_mass=0)


_INSTALLED = "_biv150_prefill_normalization_installed"
_state = {"calls": 0, "fallbacks": 0, "selected_shapes": set()}


def _eligible(bound):
    import torch

    q = bound.get("q")
    k = bound.get("k")
    v = bound.get("v")
    out = bound.get("out")
    if any(value is None for value in (q, k, v, out)):
        return False
    if not (
        q.dtype == k.dtype == v.dtype == out.dtype == torch.bfloat16
        and q.ndim == out.ndim == 3
        and q.shape[1:] == (16, 128)
        and out.shape == q.shape
        and k.ndim == v.ndim == 4
        and k.shape[1:] == v.shape[1:] == (16, 2, 128)
        and bound.get("max_seqlen_q", 0) > 1
        and bound.get("causal") is True
    ):
        return False
    if any(
        bound.get(name) is not None
        for name in (
            "sinks",
            "alibi_slopes",
            "qq_bias",
            "mm_prefix_range",
            "output_scale",
            "k_scale_cache",
            "v_scale_cache",
        )
    ):
        return False
    return (
        bound.get("softcap", 0) == 0
        and not bound.get("use_alibi_sqrt", False)
        and bound.get("window_size", (-1, -1)) == (-1, -1)
        and bound.get("chunk_lookback", -1) == -1
        and not bound.get("use_td", False)
        and str(bound.get("kv_quant_mode", 0)) in ("0", "KVQuantMode.NONE")
    )


def install():
    import os

    if os.environ.get("VLLM_FL_BIV150_PREFILL_NORMALIZE", "1") != "1":
        return {"installed": False, "reason": "disabled"}
    attention = importlib.import_module(
        "vllm.v1.attention.ops.triton_unified_attention"
    )
    if getattr(attention, _INSTALLED, False):
        return {"installed": True, "reason": "already installed"}
    if (
        attention.current_platform.vendor_name != "iluvatar"
        or attention.is_batch_invariant
    ):
        return {
            "installed": False,
            "reason": "unsupported platform or batch-invariant mode",
        }
    backend = importlib.import_module("vllm.v1.attention.backends.triton_attn")
    original = attention.unified_attention
    if backend.unified_attention is not original:
        return {"installed": False, "reason": "unrecognized backend binding"}
    try:
        normalized, metadata = make_normalized(attention)
    except (RuntimeError, OSError, TypeError) as error:
        return {"installed": False, "reason": str(error)}
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        bound = kwargs
        if args:
            try:
                bound = signature.bind(*args, **kwargs).arguments
            except TypeError:
                return original(*args, **kwargs)
        if _eligible(bound):
            _state["calls"] += 1
            shape = (
                bound["q"].shape[0],
                bound["max_seqlen_q"],
                bound["seqused_k"].shape[0],
            )
            if shape not in _state["selected_shapes"]:
                _state["selected_shapes"].add(shape)
                print("BIV150_PREFILL_NORMALIZATION_SELECTED", shape, flush=True)
            return normalized(*args, **kwargs)
        _state["fallbacks"] += 1
        return original(*args, **kwargs)

    attention.unified_attention = wrapper
    backend.unified_attention = wrapper
    setattr(attention, _INSTALLED, True)
    return dict(
        installed=True,
        scope="audited BF16 causal MiniCPM prefill without sinks",
        backend_rebound=True,
        **metadata,
    )
