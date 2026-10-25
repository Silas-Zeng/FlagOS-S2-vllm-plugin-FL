# SPDX-License-Identifier: Apache-2.0

import os

import torch


_ENV = "VLLM_FL_METAX_FUSED_GATE_UP"
_INSTALLED_ATTR = "_metax_fused_gate_up_silu_installed"


def _fused_gate_up_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from flag_gems.runtime.backend._metax.ops.fused_gate_up_silu import (
        fused_gate_up_silu,
    )

    return fused_gate_up_silu(x, weight)


def _eligible(layer, x: torch.Tensor) -> bool:
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
    )

    gate_up = layer.gate_up_proj
    return (
        isinstance(gate_up, MergedColumnParallelLinear)
        and isinstance(layer.act_fn, SiluAndMul)
        and x.ndim == 2
        and x.shape[0] <= 64
        and x.shape[1] == 2048
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and getattr(gate_up, "bias", None) is None
        and getattr(gate_up, "tp_size", 1) == 1
        and gate_up.weight.dtype == torch.bfloat16
        and gate_up.weight.shape == (12288, 2048)
        and gate_up.weight.is_contiguous()
    )


def install() -> dict[str, object]:
    if os.getenv(_ENV, "0") != "1":
        return {"installed": False, "reason": "disabled"}

    from vllm.model_executor.models.llama import LlamaMLP

    if getattr(LlamaMLP, _INSTALLED_ATTR, False):
        return {"installed": True, "reason": "already installed"}

    original_forward = LlamaMLP.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _eligible(self, x):
            x = _fused_gate_up_silu(x, self.gate_up_proj.weight)
        else:
            x, _ = self.gate_up_proj(x)
            x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x

    setattr(forward, _INSTALLED_ATTR, True)
    LlamaMLP.forward = forward
    return {
        "installed": True,
        "scope": "MiniCPM BF16 M1..64 fused Gate/Up GEMM + SwiGLU",
    }
