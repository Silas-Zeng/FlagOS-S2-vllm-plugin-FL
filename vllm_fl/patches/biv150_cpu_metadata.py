"""Reuse already-computed CPU metadata in the exact non-spec text runner.

The byte transformer is the only integration seam. Unsupported descriptions
return False before allocation/copy, preserving the original GPU expression.
No default installer, model import, new stream, synchronization or RNG work.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys


EXPECTED_INPUT_BATCH = "de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65"
EXPECTED_UTILS = "4e4cda209e10d4477d251003779dec26ea5ef47541c3310995c007c7ee5fdcdb"
_STATE = "_biv150_cpu_metadata_staging"


def _scope(runner, torch):
    config = runner.vllm_config
    model, parallel = config.model_config, config.parallel_config
    text = model.hf_text_config
    return (
        Path(str(model.model).rstrip("/\\")).name == "MiniCPM5-2B"
        and model.dtype == torch.bfloat16 and model.quantization is None
        and (text.hidden_size, text.num_hidden_layers,
             text.num_attention_heads, text.num_key_value_heads) == (2048, 42, 16, 2)
        and all(getattr(parallel, name, None) == 1 for name in (
            "tensor_parallel_size", "pipeline_parallel_size",
            "decode_context_parallel_size", "data_parallel_size"))
        and parallel.use_ubatching is False
        and config.speculative_config is None and config.lora_config is None
        and config.cache_config.kv_sharing_fast_prefill is False
        and runner.is_pooling_model is False and runner.supports_mm_inputs is False
        and runner.enable_prompt_embeds is False
        and runner.use_aux_hidden_state_outputs is False
        and runner.uses_mrope is False and runner.uses_xdrope_dim == 0
        and runner.use_async_spec_decode is False
        and runner.use_async_scheduling is True and runner.prepare_inputs_event is not None
        and runner.num_spec_tokens == 0 and runner.prev_num_spec_tokens == 0
        and runner._draft_token_ids is None
        and runner.mamba_prev_last_scheduled_idx is None
        and runner.pin_memory is True
    )


def _require_dependencies():
    for name, expected in (("vllm.v1.worker.gpu_input_batch", EXPECTED_INPUT_BATCH),
                           ("vllm.v1.utils", EXPECTED_UTILS)):
        module = sys.modules.get(name)
        if module is None or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("unaudited CPU-metadata dependency: " + name)


def _descriptors(runner, positions_np, req_indices, scheduled, torch, np):
    number, total = len(scheduled), len(positions_np)
    computed = runner.input_batch.num_computed_tokens_cpu_tensor
    computed_np = runner.input_batch.num_computed_tokens_cpu
    optimistic = runner.optimistic_seq_lens_cpu
    positions, seq_lens = runner.positions, runner.seq_lens
    return (
        isinstance(positions_np, np.ndarray) and positions_np.dtype == np.int64
        and positions_np.ndim == 1 and positions_np.flags.c_contiguous
        and isinstance(req_indices, np.ndarray) and req_indices.dtype == np.int64
        and req_indices.shape == positions_np.shape
        and isinstance(scheduled, np.ndarray) and scheduled.dtype == np.int32
        and scheduled.ndim == 1 and 0 < number == runner.input_batch.num_reqs
        and 0 < total <= runner.max_num_tokens
        and computed.device.type == "cpu" and computed.dtype == torch.int32
        and computed.ndim == 1 and computed.stride(0) == 1
        and computed_np.dtype == np.int32 and computed_np.ndim == 1
        and computed_np.shape == tuple(computed.shape)
        and computed_np.strides == (4,)
        and computed_np.__array_interface__["data"][0] == computed.data_ptr()
        and optimistic.device.type == "cpu" and optimistic.dtype == torch.int32
        and optimistic.ndim == 1 and optimistic.is_contiguous() and optimistic.is_pinned()
        and tuple(optimistic.shape) == tuple(seq_lens.shape)
        and positions.device.type == seq_lens.device.type == "cuda"
        and positions.device == seq_lens.device == runner.device
        and positions.dtype == torch.int64 and seq_lens.dtype == torch.int32
        and positions.ndim == seq_lens.ndim == 1
        and positions.is_contiguous() and seq_lens.is_contiguous()
        and positions.shape[0] == runner.max_num_tokens and number <= seq_lens.shape[0]
        and len({positions.untyped_storage().data_ptr(), seq_lens.untyped_storage().data_ptr(),
                 runner.num_computed_tokens.untyped_storage().data_ptr()}) == 3
    )


def try_cpu_metadata(runner, positions_np, req_indices, num_scheduled_tokens):
    """Called inside the original synchronize_input_prep event lifetime.

    The exact source guarantees optimistic_seq_lens_cpu already contains the
    original int32 sum plus a zero tail, and positions_np the int64 position
    expression. Its private req/query/scheduled GPU inputs have no consumer in
    this scope; their CPU mirrors remain refreshed. Positions/seq GPU pointers
    are never rebound. Failure after a submitted copy propagates; no fallback
    is attempted after mutation.
    """
    import numpy as np
    import torch

    try:
        eligible = (_scope(runner, torch)
                    and _descriptors(runner, positions_np, req_indices, num_scheduled_tokens, torch, np)
                    and not torch.cuda.is_current_stream_capturing())
    except (AttributeError, TypeError, ValueError):
        return False
    if not eligible:
        return False
    state = getattr(runner, _STATE, None)
    if state is None:
        _require_dependencies()
        staging = torch.empty(runner.max_num_tokens, dtype=torch.int64, pin_memory=True)
        state = {"tensor": staging, "numpy": staging.numpy(), "selected": 0}
        setattr(runner, _STATE, state)
    total, number = len(positions_np), len(num_scheduled_tokens)
    # Plain CPU copies of values already calculated by the original method.
    state["numpy"][:total] = positions_np
    runner.req_indices.np[:total] = req_indices
    runner.num_scheduled_tokens.np[:number] = num_scheduled_tokens
    runner.positions[:total].copy_(state["tensor"][:total], non_blocking=True)
    runner.seq_lens.copy_(runner.optimistic_seq_lens_cpu, non_blocking=True)
    state["selected"] += 1
    if state["selected"] == 1:
        print("BIV150_CPU_METADATA_FIRST_SELECTED positions_seq_h2d=2", flush=True)
    return True
