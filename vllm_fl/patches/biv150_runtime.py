# Copyright (c) 2026 BAAI. All rights reserved.

"""Shape-aware BI-V150 runtime patches for MiniCPM5-2B inference."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pathlib
import shutil
import sys

import torch

logger = logging.getLogger(__name__)

_ATTENTION_MODULE = "vllm.v1.attention.ops.triton_unified_attention"
_ATTENTION_BACKEND_MODULE = "vllm.v1.attention.backends.triton_attn"
_ATTENTION_MARKER = "# _biv150_minicpm_attention_patched_v2"

_LAUNCH_DEFAULTS = """    launch_num_warps: int | None = None
    launch_num_stages: int | None = None
"""

_LAUNCH_TUNED = """    launch_num_warps: int | None = None
    launch_num_stages: int | None = None

    # MiniCPM5-2B prefill: 16 query heads / 2 KV heads / head size 128.
    # Wider query and KV tiles amortize paged-KV address generation on BI-V150.
    biv150_prefill_tile: int | None = None
    if (
        current_platform.vendor_name == "iluvatar"
        and head_size == 128
        and num_queries_per_kv == 8
        and max_seqlen_q > 1
    ):
        BLOCK_M = 128
        BLOCK_Q = 16
        biv150_prefill_tile = 64
        launch_num_warps = 8
        launch_num_stages = 1
"""

_DECODE_TILE = """    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )
"""

_DECODE_TILE_TUNED = """    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )
    if biv150_prefill_tile is not None:
        TILE_SIZE_PREFILL = biv150_prefill_tile
"""


def _patch_top_p_sampler() -> None:
    """Use the bounded histogram top-p kernel for top-p-only sampling."""
    try:
        from vllm.v1.sample.ops import topk_topp_sampler
        from vllm.v1.sample.ops import topk_topp_triton

        from .biv150_top_p_triton import apply_top_p_hist_triton

        current = topk_topp_triton.apply_top_k_top_p_triton
        if getattr(current, "_biv150_top_p_dispatch", False):
            return
        original = current

        def apply_top_k_top_p_triton(
            logits,
            k,
            p,
            mask_value: float = float("-inf"),
        ):
            if (
                k is None
                and p is not None
                and logits.ndim == 2
                and logits.shape[1] == 130560
                and logits.dtype == torch.float32
            ):
                if logits.stride(1) != 1:
                    logits = logits.contiguous()
                return apply_top_p_hist_triton(
                    logits, p.to(torch.float32), mask_value
                )
            return original(logits, k, p, mask_value)

        apply_top_k_top_p_triton._biv150_top_p_dispatch = True
        topk_topp_triton.apply_top_k_top_p_triton = apply_top_k_top_p_triton
        topk_topp_sampler.apply_top_k_top_p_triton = apply_top_k_top_p_triton
        logger.info("Installed BI-V150 histogram top-p sampler")
    except Exception as exc:
        logger.warning("Failed to install BI-V150 top-p sampler: %s", exc)


def _patch_generic_attention() -> None:
    """Patch the fixed vLLM 0.24 generic Triton attention launch policy.

    The pinned vLLM source has no public launch-config hook. Patch the small
    launch-policy section before workers import it, then refresh the imported
    function reference in the TRITON_ATTN backend.
    """
    try:
        spec = importlib.util.find_spec(_ATTENTION_MODULE)
        if spec is None or spec.origin is None:
            raise RuntimeError("generic Triton attention module not found")
        path = pathlib.Path(spec.origin)
        source = path.read_text(encoding="utf-8")
        if _ATTENTION_MARKER not in source:
            if source.count(_LAUNCH_DEFAULTS) != 1:
                raise RuntimeError("unexpected launch-default block")
            source = source.replace(_LAUNCH_DEFAULTS, _LAUNCH_TUNED, 1)
            if source.count(_DECODE_TILE) != 1:
                raise RuntimeError("unexpected decode-tile block")
            source = source.replace(_DECODE_TILE, _DECODE_TILE_TUNED, 1)
            source += f"\n{_ATTENTION_MARKER}\n"
            path.write_text(source, encoding="utf-8")
            pycache = path.parent / "__pycache__"
            if pycache.exists():
                shutil.rmtree(pycache, ignore_errors=True)

        sys.modules.pop(_ATTENTION_MODULE, None)
        attention = importlib.import_module(_ATTENTION_MODULE)
        backend = importlib.import_module(_ATTENTION_BACKEND_MODULE)
        backend.unified_attention = attention.unified_attention
        logger.info(
            "Installed BI-V150 MiniCPM prefill attention policy "
            "(BM128/TILE64/8w/1s)"
        )
    except Exception as exc:
        logger.warning("Failed to install BI-V150 attention policy: %s", exc)


def apply_biv150_runtime_patches() -> None:
    """Install all idempotent runtime patches for the target device/model."""
    _patch_top_p_sampler()
    _patch_generic_attention()
