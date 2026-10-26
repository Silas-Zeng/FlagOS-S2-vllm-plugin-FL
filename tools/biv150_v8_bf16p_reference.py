"""CPU-only scalar references for V8 storage and BF16 weighted P.

``closed`` is the independent FP64 global-softmax oracle: it never quantizes P,
uses no page/segment recurrence, and multiplies V codes by per-token/head scales.
``streaming`` models the unchanged PAGE16 recurrence for algebra checks, with an
optional BF16 rounding of P*scale. It does not emulate a GPU dot or compiler.
"""
import math
import struct


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bfloat16(value):
    """Round FP32 to BF16 with round-to-nearest, ties-to-even, using stdlib."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7F800000 == 0x7F800000:
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]


def quantize_values(values):
    maximum = max((abs(v) for v in values), default=0.0)
    assert math.isfinite(maximum)
    scale = float32(1.0 if maximum == 0 else max(maximum / 127, 2**-126))
    codes = []
    for value in values:
        ratio = value / scale
        rounded = math.copysign(math.floor(abs(ratio) + 0.5), ratio)
        codes.append(max(-127, min(127, int(rounded))))
    return codes, scale


def closed(scores, values, scales, dimensions=None):
    if not scores:
        return [0.0] * (dimensions or 0)
    assert len(scores) == len(values) == len(scales)
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    denominator = math.fsum(weights)
    return [math.fsum(p * row[d] * scale for p, row, scale in
                      zip(weights, values, scales)) / denominator
            for d in range(len(values[0]))]


def streaming(scores, values, scales, segments, *, round_weighted_p=False,
              dimensions=None):
    if not scores:
        return [0.0] * (dimensions or 0)
    span = math.ceil(len(scores) / (segments * 16)) * 16
    partials = []
    for lower in range(0, len(scores), span):
        upper = min(lower + span, len(scores))
        maximum, total = -math.inf, 0.0
        accumulator = [0.0] * len(values[0])
        for start in range(lower, upper, 16):
            end = min(start + 16, upper)
            new_maximum = max(maximum, max(scores[start:end]))
            rescale = math.exp(maximum - new_maximum)
            probability = [math.exp(scores[i] - new_maximum) for i in range(start, end)]
            total = total * rescale + math.fsum(probability)
            weighted = [p * scales[i] for p, i in zip(probability, range(start, end))]
            if round_weighted_p:
                weighted = [bfloat16(p) for p in weighted]
            accumulator = [a * rescale + math.fsum(p * values[i][d]
                           for p, i in zip(weighted, range(start, end)))
                           for d, a in enumerate(accumulator)]
            maximum = new_maximum
        partials.append((maximum, total, accumulator))
    maximum = max(p[0] for p in partials)
    denominator = math.fsum(total * math.exp(peak - maximum)
                            for peak, total, _ in partials)
    return [math.fsum(acc[d] * math.exp(peak - maximum)
                      for peak, _, acc in partials) / denominator
            for d in range(len(values[0]))]
