# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone fusion of audited native IR BF16 residual RMSNorm and quantization.

RMSNorm math follows inspected Inductor native IR output (Apache2.0).
The source-level pre-weight BF16 cast was eliminated by that compiler.
Inputs are immutable; residual and optional normalized outputs are distinct.
This module is not installed in a model or dispatcher.
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


@triton.jit
def _norm_quant(
    X,
    R,
    R_OUT,
    W,
    Q,
    S,
    NORM_OUT,
    M: tl.constexpr,
    EPS,
    RESIDUAL: tl.constexpr,
    WRITE_NORM: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, 2048)
    if row < M:
        x = tl.load(X + row * 2048 + cols).to(tl.float32)
        r = tl.load(R + row * 2048 + cols).to(tl.float32)
        x += r
        # Preserve BF16 residual storage but normalize the unrounded FP32 sum.
        tl.store(R_OUT + row * 2048 + cols, x)
        var = tl.sum(x * x, axis=0) / 2048
        rrms = libdevice.rsqrt(var + EPS)
        w = tl.load(W + cols)
        # The selected generated native kernel keeps FP32 until after weight.
        unweighted = x * rrms
        y = (unweighted * w.to(tl.float32)).to(tl.bfloat16)
        if WRITE_NORM:
            tl.store(NORM_OUT + row * 2048 + cols, y)
        y32 = y.to(tl.float32)
        peak = tl.max(tl.abs(y32), axis=0)
        scale = tl.div_rn(tl.maximum(peak, 1e-6), 127.0)
        f = tl.div_rn(y32, scale)
        rounded = tl.where(f >= 0, tl.floor(f + 0.5), tl.ceil(f - 0.5))
        first = tl.minimum(127.0, tl.maximum(-127.0, rounded)).to(tl.int8)
        tl.store(Q + row * 2048 + cols, first)
        tl.store(S + row, scale)
        if RESIDUAL:
            second_scale = tl.div_rn(scale, 254.0)
            residual = tl.div_rn(y32 - first.to(tl.float32) * scale, second_scale)
            second = tl.where(
                residual >= 0, tl.floor(residual + 0.5), tl.ceil(residual - 0.5)
            )
            second = tl.minimum(127.0, tl.maximum(-127.0, second)).to(tl.int8)
            tl.store(Q + (row + 16) * 2048 + cols, second)
            tl.store(S + row + 16, second_scale)
    else:
        tl.store(Q + row * 2048 + cols, 0)
        tl.store(S + row, 1.0)
        if RESIDUAL:
            tl.store(Q + (row + 16) * 2048 + cols, 0)
            tl.store(S + row + 16, 1.0)



SELECTED_ROWS = (1, 8, 16, 17, 32, 64, 256)


def norm_quant(x, residual, weight, eps, write_norm=False):
    """Non-mutating component; unknown row counts are not an eligible route."""
    assert x.is_cuda and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert x.shape == residual.shape and x.ndim == 2 and x.shape[1] == 2048
    assert x.device == residual.device == weight.device and weight.shape == (2048,)
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    rows = x.shape[0]
    assert rows in SELECTED_ROWS
    padded = max(32, rows)
    q = torch.empty((padded, 2048), device=x.device, dtype=torch.int8)
    scales = torch.empty((padded,), device=x.device, dtype=torch.float32)
    residual_out = torch.empty_like(residual)
    normalized_out = torch.empty_like(x) if write_norm else x
    # WRITE_NORM writes to a separate output too: inputs and aliases stay intact.
    kernel = _norm_quant[(16 if rows <= 16 else padded,)](
        x, residual, residual_out, weight, q, scales, normalized_out, rows, eps,
        RESIDUAL=rows <= 16, WRITE_NORM=write_norm, num_warps=4,
    )
    return q, scales, residual_out, normalized_out, kernel
