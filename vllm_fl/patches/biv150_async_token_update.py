"""Exact integer gather/scatter for the audited async input-ID update seam.

No autotuning, generated source per shape, floating arithmetic or RNG work.
Only the pinned runner transformer calls this; unsupported contracts return
False before launching so its unchanged original gather/scatter runs instead.
"""

from numbers import Integral
from pathlib import Path

import torch
import triton
import triton.language as tl
from vllm.platforms import current_platform


selected_launches = 0


@triton.jit(do_not_specialize=["n", "src_rows", "dst_rows", "src_stride", "dst_stride"])
def _gather_scatter_i32(src, dst, prev_idx, dst_idx,
                        n, src_rows, dst_rows, src_stride, dst_stride,
                        BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < n
    old_row = tl.load(prev_idx + offset, mask=active, other=0)
    new_row = tl.load(dst_idx + offset, mask=active, other=0)
    valid = active & (old_row >= 0) & (old_row < src_rows)
    valid = valid & (new_row >= 0) & (new_row < dst_rows)
    token = tl.load(src + old_row * src_stride, mask=valid, other=0)
    tl.store(dst + new_row * dst_stride, token, mask=valid)


def _mapping_proof(previous, destinations, src_rows, dst_rows):
    """CPU lists already used for the two H2D copies; no tensor inspection.

    Duplicate reads are legal. Duplicate destinations fall back because
    parallel stores cannot preserve an arbitrary last-writer contract.
    """
    if not previous or len(previous) != len(destinations):
        return False
    if any(not isinstance(x, Integral) or isinstance(x, bool) for x in (*previous, *destinations)):
        return False
    return (len(set(destinations)) == len(destinations)
            and all(0 <= x < src_rows for x in previous)
            and all(0 <= x < dst_rows for x in destinations))


def _runner_eligible(runner):
    config = getattr(runner, "vllm_config", None)
    if config is None:
        return False
    model, parallel = config.model_config, config.parallel_config
    text = model.hf_text_config
    return (
        current_platform.vendor_name == "iluvatar"
        and Path(str(model.model).rstrip("/\\")).name == "MiniCPM5-2B"
        and model.dtype == torch.bfloat16
        and (text.hidden_size, text.num_hidden_layers,
             text.num_attention_heads, text.num_key_value_heads) == (2048, 42, 16, 2)
        and all(getattr(parallel, name, None) == 1 for name in (
            "tensor_parallel_size", "pipeline_parallel_size",
            "decode_context_parallel_size", "data_parallel_size"))
        and parallel.use_ubatching is False
        and config.speculative_config is None and config.lora_config is None
        and runner.use_async_scheduling is True
        and runner.use_async_spec_decode is False
        and runner.num_spec_tokens == 0 and runner.prev_num_spec_tokens == 0
        and runner.enable_prompt_embeds is False
        and runner.use_aux_hidden_state_outputs is False
        and runner._draft_token_ids is None
        and model.is_encoder_decoder is False
        and runner.uses_mrope is False and runner.uses_xdrope_dim == 0
    )


def _tensor_contract(src, dst, prev_idx, dst_idx, n, total_tokens):
    values = (src, dst, prev_idx, dst_idx)
    if not all(isinstance(value, torch.Tensor) for value in values):
        return False
    device = src.device
    return (
        device.type == "cuda" and all(value.device == device for value in values)
        and src.dtype == dst.dtype == torch.int32
        and prev_idx.dtype == dst_idx.dtype == torch.int64
        and src.ndim == 2 and src.shape[1] == 1 and src.stride(0) > 0
        and dst.ndim == 1 and dst.stride(0) > 0
        and 0 < total_tokens <= dst.shape[0] and n > 0
        and prev_idx.ndim == dst_idx.ndim == 1
        and prev_idx.shape[0] == dst_idx.shape[0] == n
        and prev_idx.stride(0) == dst_idx.stride(0) == 1
        and len({value.untyped_storage().data_ptr() for value in values}) == 4
    )


def fused_sampled_token_update(runner, src, dst, prev_idx, dst_idx,
                               previous, destinations, total_tokens):
    global selected_launches
    n = len(previous)
    if (not _runner_eligible(runner)
            or not _tensor_contract(src, dst, prev_idx, dst_idx, n, total_tokens)
            or not _mapping_proof(previous, destinations, src.shape[0], total_tokens)
            or torch.cuda.is_current_stream_capturing()):
        return False
    _gather_scatter_i32[(triton.cdiv(n, 128),)](
        src, dst, prev_idx, dst_idx, n, src.shape[0], total_tokens,
        src.stride(0), dst.stride(0), BLOCK=128, num_warps=2, num_stages=1)
    # Calls occur on the original current stream after the original two H2Ds.
    # Register allocator lifetime independently of existing stream readiness.
    stream = torch.cuda.current_stream(device=src.device)
    for value in (src, dst, prev_idx, dst_idx):
        value.record_stream(stream)
    if selected_launches == 0:
        print("BIV150_ASYNC_TOKEN_UPDATE_FIRST_SELECTED integer_gather_scatter=1", flush=True)
    selected_launches += 1
    return True
