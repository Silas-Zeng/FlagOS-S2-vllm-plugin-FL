"""No torch/Triton imports or GPU access: source, layout and scalar algebra checks."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import struct

from biv150_v8_bf16p_reference import bfloat16, closed, quantize_values, streaming

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1"


def source_check():
    original = (ROOT / "vllm_fl/patches/biv150_gqa_decode.py").read_bytes()
    assert hashlib.sha256(original).hexdigest() == SOURCE_SHA
    baseline = next(n for n in ast.parse(original).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_gqa_decode_attention")
    candidate_tree = ast.parse((ROOT / "tools/kernels/biv150_v8_bf16p.py").read_text())
    candidate = copy.deepcopy(next(n for n in candidate_tree.body
                                  if isinstance(n, ast.FunctionDef) and n.name == "gqa_decode_v8_bf16p"))
    expected_pv = ast.parse("tl.dot((probability * value_scale[None, :]).to(tl.bfloat16), V.to(tl.bfloat16))",
                           mode="eval").body
    restored_pv = ast.parse("tl.dot(probability.to(V.dtype), V)", mode="eval").body

    class RestoreBaseline(ast.NodeTransformer):
        scales, pv = 0, 0

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "value_scale":
                self.scales += 1
                assert ast.unparse(node.value.args[0]) == ("value_scale_ptr + page * scale_stride_block + "
                                                          "column_slot * scale_stride_slot + kv_head_idx * scale_stride_head")
                if self.scales == 2:
                    assert [(k.arg, ast.unparse(k.value)) for k in node.value.keywords] == [("mask", "column_valid"), ("other", "0.0")]
                else:
                    assert not node.value.keywords
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.dump(node) == ast.dump(expected_pv):
                self.pv += 1
                return copy.deepcopy(restored_pv)
            return self.generic_visit(node)

    normalizer = RestoreBaseline()
    candidate = normalizer.visit(candidate)
    candidate.name = baseline.name
    added = {"value_scale_ptr", "scale_stride_block", "scale_stride_slot", "scale_stride_head"}
    candidate.args.args = [arg for arg in candidate.args.args if arg.arg not in added]
    assert (normalizer.scales, normalizer.pv) == (2, 2)
    assert ast.dump(candidate) == ast.dump(baseline), "PAGE16/QK/softmax/partial body changed"
    # Writer remains a narrow model-specific probe with no serving registration.
    names = {n.name for n in candidate_tree.body if isinstance(n, ast.FunctionDef)}
    assert names == {"gqa_decode_v8_bf16p", "_quantized_values", "encode_cache_v", "append_token_kv"}
    assert not any(isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names)
                   for n in candidate_tree.body)
    return dict(baseline_source_sha256=SOURCE_SHA, normalized_producer_ast_equal=True,
                pv_sites=normalizer.pv, scale_load_sites=normalizer.scales)


def layout_check():
    pages = 3
    raw = bytearray([0xA5]) * (pages * 16384)
    original = raw[:]
    touched = set()
    for page in range(pages):
        for slot in range(16):
            for head in range(2):
                codes, scale = quantize_values([bfloat16((d - 64) * (head + 1) / (slot + 1))
                                                for d in range(128)])
                offset = page * 16384 + 8192 + slot * 256 + head * 128
                raw[offset:offset+128] = struct.pack("128b", *codes)
                scale_offset = page * 16384 + 12288 + (slot * 2 + head) * 4
                struct.pack_into("<f", raw, scale_offset, scale)
                touched.update(range(offset, offset + 128))
                touched.update(range(scale_offset, scale_offset + 4))
                assert all(-127 <= code <= 127 and bfloat16(code) == code for code in codes)
                assert math.isfinite(scale) and scale > 0
    for index, (old, new) in enumerate(zip(original, raw)):
        assert index in touched or old == new, "K or padding overlapped V/scales"
    assert len(touched) == pages * (4096 + 128)
    assert all(bfloat16(code) == code for code in range(-127, 128))
    assert quantize_values([0.0] * 128) == ([0] * 128, 1.0)
    return dict(allocated_bytes_per_page=16384, key_bytes=8192, code_bytes=4096,
                scale_bytes=128, payload_bytes=12416, padding_bytes=3968,
                hypothetical_aligned_compact_bytes=12544, capacity_reclaimed=False,
                signed_codes_exact_in_bfloat16=True)


def algebra_check():
    rng = random.Random(20261027)
    rows, max_exact_error, max_rounded_error = 0, 0.0, 0.0
    for length in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 257, 4096, 16384):
        scores = [rng.uniform(-10, 10) for _ in range(length)]
        values = [[rng.randint(-127, 127) for _ in range(4)] for _ in scores]
        scales = [2**rng.randint(-8, -2) * rng.uniform(0.75, 1.25) for _ in scores]
        expected = closed(scores, values, scales, dimensions=4)
        for segments in (1, 2, 4, 16):
            ideal = streaming(scores, values, scales, segments, dimensions=4)
            rounded = streaming(scores, values, scales, segments, round_weighted_p=True, dimensions=4)
            max_exact_error = max(max_exact_error, max(abs(a-b) for a, b in zip(ideal, expected)))
            max_rounded_error = max(max_rounded_error, max(abs(a-b) for a, b in zip(rounded, expected)))
            assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(ideal, expected))
            # The BF16 weighted-P error is bounded by the sum of absolute terms;
            # this is an algebra sanity bound, not a model acceptance tolerance.
            if length:
                peak = max(scores)
                weights = [math.exp(score-peak) for score in scores]
                denominator = math.fsum(weights)
                for dimension in range(4):
                    bound = math.fsum(p*abs(v[dimension])*scale for p, v, scale in
                                      zip(weights, values, scales)) / denominator / 128
                    assert abs(rounded[dimension]-expected[dimension]) <= bound + 1e-12
            rows += 1
    return dict(length_segment_combinations=rows, max_fp64_recurrence_error=max_exact_error,
                max_weighted_bf16_p_error=max_rounded_error,
                gpu_numerics_tested=False, model_accuracy_tested=False)


if __name__ == "__main__":
    print(json.dumps(dict(source=source_check(), layout=layout_check(), algebra=algebra_check(),
                          precision_gate=False, official_score=False), indent=2))
