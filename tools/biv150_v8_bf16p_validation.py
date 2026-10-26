"""Independent CPU validation; no torch, Triton or device-cache-derived oracle."""
import math
import struct

from biv150_v8_bf16p_reference import bfloat16, closed, float32, quantize_values, streaming


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _ulp(value):
    bits = _bits(abs(float32(value)))
    following = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    return following - abs(float32(value))


def validate_codec_vector(original_bf16, actual_codes, actual_scale):
    """Check actual data against *original* CPU BF16 values, not actual scale.

    FP32 maxabs/127 is the independent scale. One FP32 scale ULP permits the
    producer's constant-reciprocal multiplication. A one-code difference is
    allowed only within four FP32 ratio ULPs of a mathematical half tie: one
    scale ULP plus division/reciprocal rounding can straddle that boundary.
    Neither the expected code nor its tie margin uses the device scale.
    """
    assert len(original_bf16) == len(actual_codes) == 128
    assert all(math.isfinite(v) and bfloat16(v) == v for v in original_bf16)
    expected_codes, expected_scale = quantize_values(original_bf16)
    assert math.isfinite(actual_scale) and actual_scale > 0, "Invalid V scale"
    scale_ulps = abs(_bits(actual_scale) - _bits(expected_scale))
    assert scale_ulps <= 1, ("Independent V-scale mismatch", actual_scale, expected_scale, scale_ulps)
    if max(abs(v) for v in original_bf16) == 0 or expected_scale == 2**-126:
        assert scale_ulps == 0, "Zero/min-normal scale must be exact"
    differences, max_tie_distance_ulps, examples = 0, 0.0, []
    for value, actual, expected in zip(original_bf16, actual_codes, expected_codes):
        assert isinstance(actual, int) and -127 <= actual <= 127, "Invalid signed code"
        if actual == expected:
            continue
        ratio = abs(value / expected_scale)
        distance = abs(ratio - math.floor(ratio) - 0.5)
        assert abs(actual - expected) == 1 and distance <= 4 * _ulp(ratio), (
            "Independent V-code mismatch away from half boundary", value, actual, expected, distance)
        differences += 1
        distance_ulps = distance/_ulp(ratio)
        max_tie_distance_ulps = max(max_tie_distance_ulps, distance_ulps)
        if len(examples) < 4:
            examples.append(dict(original=value, expected_code=expected, actual_code=actual,
                                 expected_cpu_scale=expected_scale, ratio=ratio,
                                 half_tie_distance_fp32_ulps=distance_ulps))
    return dict(scale_ulps=scale_ulps, half_boundary_code_differences=differences,
                max_half_tie_distance_fp32_ulps=max_tie_distance_ulps,
                half_boundary_examples=examples)


def mapped_byte_offsets(slots, pages, encode):
    """All and only bytes this diagnostic writer may change for a slot map."""
    valid = [slot for slot in slots if 0 <= slot < pages * 16]
    assert len(valid) == len(set(valid)), "Overlapping destination slots"
    offsets = []
    for slot in valid:
        page, column = divmod(slot, 16)
        offsets.extend(range(page * 16384 + column * 512, page * 16384 + (column + 1) * 512))
        if encode:
            offsets.extend(range(page * 16384 + 8192 + column * 256,
                                 page * 16384 + 8192 + (column + 1) * 256))
            offsets.extend(range(page * 16384 + 12288 + column * 8,
                                 page * 16384 + 12288 + (column + 1) * 8))
        else:
            offsets.extend(range(page * 16384 + 8192 + column * 512,
                                 page * 16384 + 8192 + (column + 1) * 512))
    assert len(offsets) == len(set(offsets))
    return offsets


def consumption_gate(actual, original_bf16, limit=0.02):
    """Component V-quantization gate: <=2% relative RMS per complete query row.

    Empty/zero reference rows require exact zero output. This does not establish
    model precision and must never set the full-model precision_gate flag.
    """
    assert len(actual) == len(original_bf16) and actual
    finite = all(math.isfinite(v) for v in (*actual, *original_bf16))
    error2 = math.fsum((a-b)**2 for a, b in zip(actual, original_bf16))
    reference2 = math.fsum(v*v for v in original_bf16)
    relative_rmse = math.sqrt(error2/reference2) if reference2 else (0.0 if error2 == 0 else math.inf)
    return dict(finite=finite, relative_rmse=relative_rmse,
                max_abs=max(abs(a-b) for a, b in zip(actual, original_bf16)),
                relative_rmse_limit=limit, passed=finite and relative_rmse <= limit)


def numeric_timing_policy(numeric_passed, hard_checks, diagnostic=False):
    """Diagnostics may retain failed numeric gates; structural faults stay fatal."""
    required = {'finite', 'codec', 'slot_mapping', 'unmapped_bytes', 'graph_eager'}
    assert set(hard_checks) == required
    assert all(value is True for value in hard_checks.values()), ('Hard validation failure', hard_checks)
    assert numeric_passed or diagnostic, 'Numeric precision gate failed; timing disabled'
    return dict(numeric_passed=bool(numeric_passed), hard_checks_passed=True,
                component_checks_passed=bool(numeric_passed), diagnostic_only=bool(diagnostic),
                timing_rounds=1 if diagnostic else 3, precision_gate=False)


def attribute_coordinate(trace):
    """CPU explanatory model, never an acceptance oracle or a relaxed tolerance.

    Separate validated codec differences, BF16(P*scale), output BF16 rounding,
    and the remaining device/model difference at one saved output coordinate.
    The streaming model uses FP64 scores/recurrence, not GPU FP32 QK/exp/FMA.
    """
    scores = trace['scores']
    canonical_values = [[code] for code in trace['canonical_codes']]
    actual_values = [[code] for code in trace['validated_actual_codes']]
    canonical = closed(scores, canonical_values, trace['canonical_scales'], dimensions=1)[0]
    validated_codec = closed(scores, actual_values, trace['validated_actual_scales'], dimensions=1)[0]
    weighted_p = streaming(scores, actual_values, trace['validated_actual_scales'], trace['segments'],
                           round_weighted_p=True, dimensions=1)[0]
    rounded_output = bfloat16(weighted_p)
    actual = trace['actual_output']
    contributions = dict(validated_codec_minus_canonical=validated_codec-canonical,
                         bf16_weighted_p_minus_validated_codec=weighted_p-validated_codec,
                         final_bf16_output_rounding=rounded_output-weighted_p,
                         device_minus_cpu_model_residual=actual-rounded_output)
    assert math.isclose(math.fsum(contributions.values()), actual-canonical, rel_tol=1e-12, abs_tol=1e-12)
    return dict(coordinate=trace.get('coordinate'), actual=actual, canonical_fp64=canonical,
                validated_codec_fp64=validated_codec, bf16_weighted_p_fp64_recurrence=weighted_p,
                bf16_weighted_p_plus_bf16_output=rounded_output,
                actual_minus_canonical=actual-canonical, contributions=contributions,
                acceptance_tolerance_changed=False, gpu_fp32_qk_exp_accumulation_emulated=False)


def _bf16_bytes(values):
    return b"".join(struct.pack("<H", _bits(v) >> 16) for v in values)


def reference_append_bytes(before, slots, new_keys, new_values, encode):
    """Build a CPU byte oracle from frozen old bytes and new BF16 tokens."""
    pages = len(before) // 16384
    mapped_byte_offsets(slots, pages, encode)
    after = bytearray(before)
    for token, slot in enumerate(slots):
        if not 0 <= slot < pages * 16:
            continue
        page, column = divmod(slot, 16)
        for head in range(2):
            key_offset = page*16384 + column*512 + head*256
            after[key_offset:key_offset+256] = _bf16_bytes(new_keys[token][head])
            if encode:
                codes, scale = quantize_values(new_values[token][head])
                value_offset = page*16384 + 8192 + column*256 + head*128
                after[value_offset:value_offset+128] = struct.pack("128b", *codes)
                struct.pack_into("<f", after, page*16384+12288+column*8+head*4, scale)
            else:
                value_offset = page*16384 + 8192 + column*512 + head*256
                after[value_offset:value_offset+256] = _bf16_bytes(new_values[token][head])
    return after


def validate_writer_bytes(before, after, slots, new_keys, new_values, encode):
    """Small CPU fixture gate also detects omitted/stale/misdirected writers."""
    assert len(before) == len(after) and len(after) % 16384 == 0
    pages = len(after) // 16384
    allowed = set(mapped_byte_offsets(slots, pages, encode))
    assert all(index in allowed or old == new for index, (old, new) in enumerate(zip(before, after))), "Unmapped bytes changed"
    exact = reference_append_bytes(before, slots, new_keys, new_values, encode)
    for token, slot in enumerate(slots):
        if not 0 <= slot < pages * 16:
            continue
        page, column = divmod(slot, 16)
        for head in range(2):
            key_offset = page*16384 + column*512 + head*256
            assert after[key_offset:key_offset+256] == exact[key_offset:key_offset+256], "K writer missing/stale"
            if encode:
                value_offset = page*16384+8192+column*256+head*128
                codes = list(struct.unpack_from("128b", after, value_offset))
                scale = struct.unpack_from("<f", after, page*16384+12288+column*8+head*4)[0]
                validate_codec_vector(new_values[token][head], codes, scale)
            else:
                value_offset = page*16384+8192+column*512+head*256
                assert after[value_offset:value_offset+256] == exact[value_offset:value_offset+256], "V writer missing/stale"
    return dict(mapped_bytes=len(allowed), unmapped_bytes_unchanged=True, writer_validated=True)
