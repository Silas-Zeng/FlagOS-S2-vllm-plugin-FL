# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import os

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor, MinPLogitsProcessor, MinTokensLogitsProcessor,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import InputBatch

from flag_gems.runtime.backend._metax.ops.bf16_top_p_sample import sample_bf16_top_p

_original_forward = Sampler.forward
_original_make_sampling_metadata = InputBatch._make_sampling_metadata
logger = logging.getLogger(__name__)


def _processors_are_noops(processors):
    for processor in processors.all:
        kind = type(processor)
        if kind is MinPLogitsProcessor and processor.min_p_count == 0:
            continue
        if kind is LogitBiasLogitsProcessor and not processor.biases:
            continue
        if kind is MinTokensLogitsProcessor and not processor.min_toks:
            continue
        return False
    return True


def _eligible(self, logits, metadata, predict_bonus_token, override):
    return (
        not predict_bonus_token and override is None
        and not self.use_fp64_gumbel
        and self.logprobs_mode not in ("processed_logits", "processed_logprobs")
        and os.environ.get("VLLM_FL_TOP_P_KERNEL", "1").lower() not in ("0", "false")
        and logits.ndim == 2 and logits.dtype == torch.bfloat16
        and logits.device.type == "cuda"
        and logits.is_contiguous() and not logits.requires_grad
        and logits.shape[0] >= 8 and logits.shape[1] >= 32768
        and metadata.all_random and not metadata.all_greedy
        and metadata.temperature is not None and metadata.top_p is not None
        and metadata.top_k is None and not metadata.generators
        and metadata.max_num_logprobs is None and not metadata.logprob_token_ids
        and metadata.no_penalties and not metadata.bad_words_token_ids
        and metadata.allowed_token_ids_mask is None
        and not any(metadata.spec_token_ids or ())
        and (
            metadata.thinking_budget_state_holder is None
            or not metadata.thinking_budget_state_holder.has_tracked_requests()
        )
        and _processors_are_noops(metadata.logitsprocs)
    )


@functools.wraps(_original_forward)
def _forward(self, logits, sampling_metadata, predict_bonus_token=False,
             logprobs_mode_override=None):
    if _eligible(self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override):
        sampled = sample_bf16_top_p(
            logits, sampling_metadata.temperature, sampling_metadata.top_p,
            unit_temperature=getattr(
                sampling_metadata, "_metax_unit_temperature", False
            ),
        )
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)
    return _original_forward(
        self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
    )


def _make_sampling_metadata(self):
    metadata = _original_make_sampling_metadata(self)
    # CPU request parameters are already available here. Cache this fact when
    # the batch changes; never read a GPU tensor back on the sampling path.
    metadata._metax_unit_temperature = bool(
        metadata.temperature is not None
        and (self.temperature_cpu[:self.num_reqs] == 1.0).all()
    )
    return metadata


def warmup(sampler, logits):
    """Compile the common path during memory profiling without advancing RNG."""
    if (
        getattr(sampler, "_metax_top_p_warmed", False)
        or not getattr(sampler, "_metax_fused_bf16_installed", False)
        or getattr(sampler, "use_fp64_gumbel", False)
        or getattr(sampler, "logprobs_mode", "raw_logprobs") in (
            "processed_logits", "processed_logprobs"
        )
        or os.environ.get("VLLM_FL_TOP_P_KERNEL", "1").lower() in ("0", "false")
        or logits.ndim != 2 or logits.shape[0] < 8 or logits.shape[1] < 32768
        or logits.dtype != torch.bfloat16 or logits.device.type != "cuda"
        or not logits.is_contiguous()
    ):
        return False
    with torch.random.fork_rng(devices=[logits.device.index]):
        temperature = torch.ones(8, device=logits.device, dtype=torch.float32)
        top_p = torch.full_like(temperature, 0.95)
        sample_bf16_top_p(logits[:8], temperature, top_p, unit_temperature=True)
        # The large-batch path has a runtime batch stride. Warm its single
        # specialization before requests arrive, preserving the caller's RNG.
        if logits.shape[0] >= 64:
            temperature = torch.ones(64, device=logits.device, dtype=torch.float32)
            top_p = torch.full_like(temperature, 0.95)
            sample_bf16_top_p(logits[:64], temperature, top_p, unit_temperature=True)
    sampler._metax_top_p_warmed = True
    logger.info("Warmed MetaX BF16 top-p sampler during model memory profiling")
    return True


def install():
    if os.environ.get("VLLM_FL_METAX_FUSED_SAMPLER", "1") != "1":
        return False
    if not getattr(Sampler, "_metax_fused_bf16_installed", False):
        Sampler.forward = _forward
        InputBatch._make_sampling_metadata = _make_sampling_metadata
        diagnostic_path = os.environ.get("VLLM_FL_METAX_SAMPLER_DIAGNOSTIC")
        if diagnostic_path:
            # Bounded, host-metadata-only diagnostics for profiling sessions.
            import json
            from pathlib import Path

            seen = set()
            capture_path = os.environ.get("VLLM_FL_METAX_CAPTURE_LOGITS")
            captured = False

            @functools.wraps(_forward)
            def diagnostic_forward(self, logits, sampling_metadata, predict_bonus_token=False,
                                   logprobs_mode_override=None):
                nonlocal captured
                metadata = sampling_metadata
                state = dict(
                    batch=logits.shape[0], dtype=str(logits.dtype),
                    contiguous=logits.is_contiguous(), requires_grad=logits.requires_grad,
                    all_random=metadata.all_random, all_greedy=metadata.all_greedy,
                    has_top_k=metadata.top_k is not None,
                    has_top_p=metadata.top_p is not None,
                    max_num_logprobs=metadata.max_num_logprobs,
                    generators=len(metadata.generators),
                    no_penalties=metadata.no_penalties,
                    has_spec_tokens=any(metadata.spec_token_ids or ()),
                    processors_noop=_processors_are_noops(metadata.logitsprocs),
                    eligible=_eligible(self, logits, metadata, predict_bonus_token,
                                       logprobs_mode_override),
                )
                key = json.dumps(state, sort_keys=True)
                if key not in seen and len(seen) < 128:
                    seen.add(key)
                    with Path(diagnostic_path).open("a") as stream:
                        stream.write(key + "\n")
                if (capture_path and not captured and state["eligible"]
                        and logits.shape[0] >= 64
                        and not torch.cuda.is_current_stream_capturing()):
                    torch.save(dict(
                        logits=logits.detach().cpu(),
                        temperature=metadata.temperature.detach().cpu(),
                        top_p=metadata.top_p.detach().cpu(),
                        discard_mask=getattr(metadata, "_metax_discard_mask_cpu", None),
                    ), capture_path)
                    captured = True
                return _forward(self, logits, metadata, predict_bonus_token,
                                logprobs_mode_override)

            Sampler.forward = diagnostic_forward
        Sampler._metax_fused_bf16_installed = True
    return True
