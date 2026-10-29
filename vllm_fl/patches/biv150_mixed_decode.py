"""Experimental natural-prefix Attention split; no sorting or model split.

Install after the audited w2s1 runtime, before creating the model runner.
Only builder.build and Impl.forward are wrapped. Pure decode is unchanged;
ordinary mixed dummy runs may split, but an active CUDA capture never does.
Mixed 2D -> 3D changes reduction order and still requires an accuracy gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import functools
import hashlib
import importlib
from pathlib import Path
import sys


EXPECTED_BACKEND_SHA256 = "1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605"
EXPECTED_ATTENTION_SHA256 = "54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f"
EXPECTED_HELPERS_SHA256 = "59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1"
EXPECTED_RUNTIME_SHA256 = "e5f68a303f991f1e5ce1145941d44350d280a85090db94045b653e3828c3ed5f"
_PLAN = "_biv150_natural_prefix_plan"
_MARKER = "_biv150_natural_prefix_wrapper"
_NATIVE_MIXED = None # Cold-installed default-OFF, independently qualified exact-row adapter.


def _require_source(module, expected):
    path = Path(module.__file__)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"unaudited mixed-decode dependency: {path}: {actual}")


def _require_fast_runtime(attention):
    proxy = attention.kernel_unified_attention
    cls = type(proxy)
    runtime = sys.modules.get(cls.__module__)
    if runtime is None or getattr(runtime, "DecodeLaunch", None) is not cls:
        raise RuntimeError("mixed decode requires the installed audited w2s1 DecodeLaunch")
    _require_source(runtime, EXPECTED_RUNTIME_SHA256)
    if not hasattr(proxy, "original") or isinstance(proxy.original, cls):
        raise RuntimeError("unexpected w2s1 proxy nesting")


def _cuda_tensors(*tensors):
    """Only inspect device descriptors; never transfer tensor contents."""
    device = tensors[0].device
    return device.type == "cuda" and all(value.device == device for value in tensors)


def _builder_eligible(builder, backend, torch):
    config = builder.vllm_config
    model, parallel = config.model_config, config.parallel_config
    text = model.hf_text_config
    return (
        Path(str(model.model).rstrip("/\\")).name == "MiniCPM5-2B"
        and model.dtype == torch.bfloat16
        and (text.hidden_size, text.num_hidden_layers,
             text.num_attention_heads, text.num_key_value_heads) == (2048, 42, 16, 2)
        and (builder.num_heads_q, builder.num_heads_kv, builder.headdim,
             builder.block_size) == (16, 2, 128, 16)
        and all(getattr(parallel, name, None) == 1 for name in (
            "tensor_parallel_size", "pipeline_parallel_size",
            "decode_context_parallel_size", "data_parallel_size"))
        and not parallel.use_ubatching
        and config.speculative_config is None
        and config.lora_config is None
        and not config.cache_config.kv_sharing_fast_prefill
        and config.compilation_config.cudagraph_mode == backend.CUDAGraphMode.FULL_DECODE_ONLY
        and not backend.envs.VLLM_BATCH_INVARIANT
    )


def _metadata_eligible(metadata):
    return (
        metadata.causal is True
        and metadata.use_cascade is False
        and metadata.common_prefix_len == 0
        and metadata.mm_prefix_range is None
        and metadata.mm_prefix_range_tensor is None
        and metadata.scheduler_metadata is None
        and metadata.prefix_scheduler_metadata is None
    )


def _prefix(common, metadata):
    """CPU-only offsets proof. Return zero for unsupported/padded shapes."""
    cpu = common.query_start_loc_cpu
    count, total = common.num_reqs, metadata.num_actual_tokens
    if (cpu.device.type != "cpu" or cpu.ndim != 1
            or cpu.shape[0] != count + 1
            or metadata.query_start_loc.shape != cpu.shape
            or metadata.seq_lens.shape[0] != count
            or metadata.block_table.shape[0] != count
            or metadata.slot_mapping.shape[0] < total):
        return 0
    offsets = cpu.tolist()
    if offsets[0] != 0 or offsets[-1] != total:
        return 0
    lengths = [right - left for left, right in zip(offsets, offsets[1:])]
    if not lengths or min(lengths) <= 0 or max(lengths) != metadata.max_query_len:
        return 0
    prefix = 0
    for length in lengths:
        if length != 1:
            break
        prefix += 1
    if not 0 < prefix < count or prefix > metadata.seq_threshold_3D:
        return 0
    # All prefix lengths are exactly one, so token and row boundaries coincide.
    return prefix


@dataclass
class _SplitPlan:
    prefix_tokens: int
    total_tokens: int
    decode: object
    prefill: object
    producer_stream: object
    consumer_streams: set = field(default_factory=set)

    def prepare_consumer(self, torch):
        stream = torch.cuda.current_stream(device=self.prefill.query_start_loc.device)
        if stream != self.producer_stream and stream not in self.consumer_streams:
            # wait_stream establishes readiness; record_stream independently
            # prevents allocator reuse after the Python metadata is released.
            stream.wait_stream(self.producer_stream)
            self.prefill.query_start_loc.record_stream(stream)
            self.consumer_streams.add(stream)


def _impl_eligible(impl, backend, torch, query, key, value, cache, metadata,
                   output, output_scale, output_block_scale):
    return (
        impl.attn_type == backend.AttentionType.DECODER
        and (impl.num_heads, impl.num_kv_heads, impl.head_size) == (16, 2, 128)
        and impl._kv_quant_mode == backend.KVQuantMode.NONE
        and impl.kv_cache_dtype in ("auto", "bfloat16")
        and impl.kv_sharing_target_layer_name is None
        and not impl._is_per_token_head_quant
        and impl.alibi_slopes is None and impl.sinks is None
        and not impl.use_alibi_sqrt and not impl.use_td
        and impl.sliding_window == (-1, -1)
        and impl.logits_soft_cap == 0 and impl.chunk_lookback == -1
        and output_scale is None and output_block_scale is None
        and _metadata_eligible(metadata)
        and query is not None and key is not None and value is not None
        and query.dtype == key.dtype == value.dtype == cache.dtype == output.dtype == torch.bfloat16
        and query.ndim == output.ndim == 3
        and tuple(query.shape[1:]) == tuple(output.shape[1:]) == (16, 128)
        and query.shape[0] >= metadata.num_actual_tokens
        and output.shape[0] >= metadata.num_actual_tokens
        and key.shape[0] >= metadata.num_actual_tokens
        and value.shape[0] >= metadata.num_actual_tokens
        and cache.ndim == 5 and tuple(cache.shape[1:]) == (2, 16, 2, 128)
        and _cuda_tensors(query, key, value, cache, output, metadata.query_start_loc)
    )


def _wrap_backend(backend, torch):
    """Internal seam for the small CPU metadata/alias tests; install guards it."""
    builder_cls, impl_cls = backend.TritonAttentionMetadataBuilder, backend.TritonAttentionImpl
    original_build, original_forward = builder_cls.build, impl_cls.forward
    if getattr(original_build, _MARKER, False) or getattr(original_forward, _MARKER, False):
        if not (getattr(original_build, _MARKER, False)
                and getattr(original_forward, _MARKER, False)):
            raise RuntimeError("partially installed mixed-decode wrappers")
        return

    @functools.wraps(original_build)
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original_build(self, common_prefix_len, common_attn_metadata,
                                  fast_build=fast_build)
        if (metadata.max_query_len <= 1 or fast_build or common_prefix_len != 0
                or not _builder_eligible(self, backend, torch)
                or not _metadata_eligible(metadata)):
            return metadata
        prefix = _prefix(common_attn_metadata, metadata)
        if (not prefix or not _cuda_tensors(metadata.query_start_loc, metadata.seq_lens,
                                           metadata.block_table, metadata.slot_mapping)
                or torch.cuda.is_current_stream_capturing()):
            return metadata
        if (metadata.num_par_softmax_segments != 16
                or metadata.softmax_segm_output is None
                or metadata.softmax_segm_max is None
                or metadata.softmax_segm_expsum is None
                or metadata.softmax_segm_output.shape[0] < prefix
                or metadata.softmax_segm_max.shape[0] < prefix
                or metadata.softmax_segm_expsum.shape[0] < prefix):
            return metadata
        count, total = common_attn_metadata.num_reqs, metadata.num_actual_tokens
        producer = torch.cuda.current_stream(device=metadata.query_start_loc.device)
        suffix_cu = metadata.query_start_loc[prefix:count + 1] - prefix
        decode = replace(metadata, num_actual_tokens=prefix, max_query_len=1,
                         query_start_loc=metadata.query_start_loc[:prefix + 1],
                         seq_lens=metadata.seq_lens[:prefix],
                         block_table=metadata.block_table[:prefix],
                         slot_mapping=metadata.slot_mapping[:prefix])
        prefill = replace(metadata, num_actual_tokens=total - prefix,
                          query_start_loc=suffix_cu,
                          seq_lens=metadata.seq_lens[prefix:count],
                          block_table=metadata.block_table[prefix:count],
                          slot_mapping=metadata.slot_mapping[prefix:total])
        setattr(metadata, _PLAN, _SplitPlan(prefix, total, decode, prefill, producer))
        return metadata

    @functools.wraps(original_forward)
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        plan = getattr(attn_metadata, _PLAN, None)
        if (plan is None
                or not _impl_eligible(self, backend, torch, query, key, value, kv_cache,
                                      attn_metadata, output, output_scale, output_block_scale)
                or torch.cuda.is_current_stream_capturing()):
            return original_forward(self, layer, query, key, value, kv_cache,
                                    attn_metadata, output, output_scale=output_scale,
                                    output_block_scale=output_block_scale)
        plan.prepare_consumer(torch)
        boundary, total = plan.prefix_tokens, plan.total_tokens
        if _NATIVE_MIXED is None:
            original_forward(self, layer, query[:boundary], key[:boundary], value[:boundary],
                             kv_cache, plan.decode, output[:boundary],
                             output_scale=output_scale, output_block_scale=output_block_scale)
        else:
            _NATIVE_MIXED.forward(original_forward, self, layer, query, key, value,
                                  kv_cache, plan, output, output_scale, output_block_scale)
        original_forward(self, layer, query[boundary:total], key[boundary:total],
                         value[boundary:total], kv_cache, plan.prefill, output[boundary:total],
                         output_scale=output_scale, output_block_scale=output_block_scale)
        return output

    setattr(build, _MARKER, True)
    setattr(forward, _MARKER, True)
    builder_cls.build, impl_cls.forward = build, forward


def install():
    """Fail closed before wrapping either class; performs no file mutation."""
    backend = importlib.import_module("vllm.v1.attention.backends.triton_attn")
    attention = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    helpers = importlib.import_module("vllm.v1.attention.ops.triton_attention_helpers")
    for module, expected in ((backend, EXPECTED_BACKEND_SHA256),
                             (attention, EXPECTED_ATTENTION_SHA256),
                             (helpers, EXPECTED_HELPERS_SHA256)):
        _require_source(module, expected)
    _require_fast_runtime(attention)
    if (attention.current_platform.vendor_name != "iluvatar"
            or attention.is_batch_invariant
            or backend.unified_attention is not attention.unified_attention
            or attention.unified_attention.__globals__ is not vars(attention)):
        raise RuntimeError("unsupported mixed-decode platform/function binding")
    torch = importlib.import_module("torch")
    _wrap_backend(backend, torch)
    return {"installed": True, "scope": "BF16 MiniCPM natural-prefix mixed Attention only",
            "ordinary_mixed_dummy_may_split": True,
            "accuracy": "mixed 2D to 3D remains unverified"}
