# SPDX-License-Identifier: Apache-2.0
"""Fuse compatible projection/RMSNorm pairs without changing model forwards."""

import logging
import os

import torch
from vllm.compilation.passes.inductor_pass import InductorPass

logger = logging.getLogger(__name__)


@torch.library.custom_op("vllm_fl::metax_linear_residual_rms", mutates_args=(),
                         device_types="cuda")
def _linear_norm(a: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor,
                 residual: torch.Tensor, norm_weight: torch.Tensor, epsilon: float,
                 down: bool, split_k: int, large_prefill: bool
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    from flag_gems.runtime.backend._metax.ops.linear_residual_rms import linear_residual_rms

    return linear_residual_rms(a, weight, packed, residual, norm_weight, epsilon,
                               down=down, split_k=split_k, large_prefill=large_prefill)


@_linear_norm.register_fake
def _linear_norm_fake(a, weight, packed, residual, norm_weight, epsilon,
                      down, split_k, large_prefill):
    return torch.empty_like(residual), torch.empty_like(residual)


class LinearNormFusionPass(InductorPass):
    def __init__(self, previous=None):
        self.previous = previous
        self.matched_count = 0
        self.down_split = 6 if os.environ.get("VLLM_FL_METAX_ODD_SPLITK", "1") == "1" else 4
        self.o_split = 3 if self.down_split == 6 else 0
        self.large_prefill = os.environ.get("VLLM_FL_METAX_LARGE_PREFILL_DOWN", "1") == "1"

    def uuid(self):
        return self.hash_source(LinearNormFusionPass) + self.hash_dict({
            "previous": self.previous.uuid() if self.previous else None,
            "down_split": self.down_split, "o_split": self.o_split,
            "large_prefill": self.large_prefill,
        })

    def __call__(self, graph):
        if self.previous is not None:
            self.previous(graph)
        count = 0
        for norm in list(graph.nodes):
            if (norm.op != "call_function"
                    or norm.target != torch.ops.vllm_fl.metax_residual_rms.default):
                continue
            linear = norm.args[0]
            if (not isinstance(linear, torch.fx.Node) or linear.op != "call_function"
                    or len(linear.users) != 1):
                continue
            down = linear.target == torch.ops.vllm_fl.metax_packed_down.default
            projection = linear.target == torch.ops.vllm_fl.metax_prefill_linear_nn.default
            if not (down or projection) or len(linear.args) != 3:
                continue
            # Restrict projection fusion to the packed O projection, not QKV or
            # legacy transposed caches. Tensor metadata contains static weights.
            weight = linear.args[1].meta.get("val")
            packed = linear.args[2].meta.get("val")
            if (not isinstance(weight, torch.Tensor) or not isinstance(packed, torch.Tensor)
                    or weight.shape != (2048, 6144 if down else 2048)
                    or packed.shape != (16, 96 if down else 32, 128, 64)):
                continue
            with graph.inserting_before(norm):
                fused = graph.call_function(
                    torch.ops.vllm_fl.metax_linear_residual_rms.default,
                    args=(*linear.args, *norm.args[1:], down,
                          self.down_split if down else self.o_split, self.large_prefill),
                )
            fused.meta = norm.meta.copy()
            # The original operator's eager arguments have a different schema.
            # Inductor must infer constraints for the newly inserted operator.
            fused.meta.pop("eager_input_vals", None)
            norm.replace_all_uses_with(fused)
            graph.erase_node(norm)
            graph.erase_node(linear)
            count += 1
        if count:
            self.matched_count += count
            graph.lint()
            logger.info("MetaX fused %d projection/residual RMSNorm pairs", count)


def install(compilation_config):
    if os.environ.get("VLLM_FL_METAX_LINEAR_NORM_FUSION", "1") != "1":
        return False
    key = "post_grad_custom_post_pass"
    previous = compilation_config.inductor_compile_config.get(key)
    if not isinstance(previous, LinearNormFusionPass):
        compilation_config.inductor_compile_config[key] = LinearNormFusionPass(previous)
    return True
