"""Pure CPU oracles, extracted unchanged from the validated residual audit.

Inputs must already be on CPU before FP64 conversion. No GPU dependencies.
"""

import torch


def cpu_quantize(x):
    """CPU FP32, native round-away and contracted residual subtraction.

    FP64 exactly represents each product of these FP32 operands before its
    final FP32 rounding, emulating the assembly's single-round FMA.
    """
    assert x.device.type == 'cpu'
    f = x.float()
    s1 = f.abs().amax(1).clamp_min(1e-6) / 127
    s2 = s1 / 254
    def away(v):
        return torch.where(v >= 0, (v + 0.5).floor(), (v - 0.5).ceil()).clamp(-127,127)
    first = away(f / s1[:, None])
    residual = (f.double() - first.double() * s1.double()[:, None]).float()
    second = away(residual / s2[:, None])
    return first.to(torch.int8), second.to(torch.int8), s1, s2


def cpu_output(first, second, s1, s2, weights, scales):
    # INT64 dot is exact; assembly rounds the residual product, then FMA.
    a = (first.long() @ weights.long()).float()
    b = (second.long() @ weights.long()).float()
    residual = b * s2[:,None]
    linear = (a.double() * s1.double()[:,None] + residual.double()).float() * scales
    gate, up = linear.to(torch.bfloat16).float().chunk(2,-1)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
