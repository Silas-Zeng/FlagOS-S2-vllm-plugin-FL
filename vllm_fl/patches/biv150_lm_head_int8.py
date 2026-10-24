"""Audited untied MiniCPM LM-head operator candidate; model validation pending."""

import torch

from vllm_fl.patches.biv150_native_w8a8 import (
    _eligible_model,
    load_library,
    quantize_weight,
)

_STATE = {"heads": 0, "eager_calls": 0}


def install():
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )

    config = get_current_vllm_config().model_config.hf_config
    if (
        not _eligible_model()
        or getattr(config, "tie_word_embeddings", True)
        or get_tensor_model_parallel_world_size() != 1
    ):
        return {"installed": False, "reason": "requires audited untied MiniCPM TP1"}
    if getattr(UnquantizedEmbeddingMethod, "_biv150_lm_head_int8", False):
        return {"installed": True, "already": True}
    load_library()
    original_after = UnquantizedEmbeddingMethod.process_weights_after_loading
    original_apply = UnquantizedEmbeddingMethod.apply

    def after(self, layer):
        original_after(self, layer)
        if not isinstance(layer, ParallelLMHead):
            return
        weight = layer.weight
        if (
            weight.dtype != torch.bfloat16
            or not weight.is_cuda
            or tuple(weight.shape) != (130560, 2048)
            or getattr(layer, "bias", None) is not None
        ):
            raise RuntimeError("Unexpected audited LM head geometry/dtype/bias")
        q, scales = quantize_weight(weight.data)
        layer.register_buffer("_biv150_head_q", q, persistent=False)
        layer.register_buffer("_biv150_head_scales", scales, persistent=False)
        marker = torch.zeros((), device=weight.device, dtype=weight.dtype)
        layer.weight = torch.nn.Parameter(marker.expand_as(weight), requires_grad=False)
        _STATE["heads"] += 1
        if _STATE["heads"] != 1:
            raise RuntimeError("Expected exactly one untied LM head")
        print(
            "BIV150_LM_HEAD_WEIGHT",
            dict(
                prefix=getattr(layer, "prefix", type(layer).__name__),
                shape=list(weight.shape),
                q_stride=list(q.stride()),
                bytes_saved=weight.numel() * weight.element_size()
                - q.numel()
                - scales.numel() * scales.element_size(),
                compressed_only=True,
            ),
            flush=True,
        )

    def apply(self, layer, x, bias=None):
        if not hasattr(layer, "_biv150_head_q"):
            return original_apply(self, layer, x, bias)
        if x.dtype != torch.bfloat16 or x.shape[-1] != 2048 or bias is not None:
            raise RuntimeError("Unsupported input for compressed audited LM head")
        # Keep Python telemetry outside compilation and capture; no tensor data
        # is copied to the host. Log powers of two to bound serving log volume.
        if (
            not torch.compiler.is_compiling()
            and not torch.cuda.is_current_stream_capturing()
        ):
            _STATE["eager_calls"] += 1
            calls = _STATE["eager_calls"]
            if calls & (calls - 1) == 0:
                print(
                    "BIV150_LM_HEAD_SELECTED",
                    dict(calls=calls, shape=list(x.shape)),
                    flush=True,
                )
        y = torch.ops.biv150_native.linear(
            x.reshape(-1, 2048),
            layer._biv150_head_q,
            layer._biv150_head_scales,
            layer.weight,
            None,
            17,
        )
        return y.reshape(*x.shape[:-1], 130560)

    UnquantizedEmbeddingMethod.process_weights_after_loading = after
    UnquantizedEmbeddingMethod.apply = apply
    UnquantizedEmbeddingMethod._biv150_lm_head_int8 = True
    return dict(
        installed=True,
        scope="single untied BF16-input MiniCPM LM head, TP1",
        weight_shape=[130560, 2048],
        compressed_only=True,
        hidden_layer_quantization=False,
        input_embedding_quantization=False,
    )
