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
from biv150_v8_bf16p_validation import (attribute_coordinate, consumption_gate, mapped_byte_offsets,
                                       numeric_timing_policy,
                                       reference_append_bytes, validate_codec_vector,
                                       validate_writer_bytes)

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


def falsification_check():
    """The shared validation gates must reject plausible self-consistent faults."""
    rejected = []

    def reject(label, action):
        try:
            action()
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError('Fault escaped: '+label)

    # Includes a valid, independently derived half tie at |ratio|=1.5.
    tie_values = [127.0, 1.5, -1.5]+[0.0]*125
    codes, scale = quantize_values(tie_values)
    adjusted = codes[:]
    adjusted[1], adjusted[2] = 1, -1
    tie = validate_codec_vector(tie_values, adjusted, scale)
    assert tie['half_boundary_code_differences'] == 2
    wrong = codes[:]; wrong[0] = 126
    reject('one_code_error_away_from_half_tie', lambda: validate_codec_vector(tie_values, wrong, scale))
    reject('huge_scale_zero_codes', lambda: validate_codec_vector(tie_values, [0]*128, 1e20))
    reject('wrong_scale_with_self_consistent_codes',
           lambda: validate_codec_vector(tie_values, [round(v/2) for v in tie_values], 2.0))
    assert validate_codec_vector([0.0]*128, [0]*128, 1.0)['scale_ulps'] == 0
    tiny = [2**-133, -2**-133]+[0.0]*126
    tiny_codes, tiny_scale = quantize_values(tiny)
    assert tiny_scale == 2**-126
    validate_codec_vector(tiny, tiny_codes, tiny_scale)
    reject('zero_vector_wrong_scale', lambda: validate_codec_vector([0.0]*128, [0]*128, 2.0))
    reject('missing_min_normal_clamp', lambda: validate_codec_vector(tiny, tiny_codes, 2**-133))

    pages, slots = 3, (-1, 17, 46, 48)
    old_keys = [[[bfloat16((d+head+token)/32) for d in range(128)] for head in range(2)] for token in range(4)]
    old_values = [[[bfloat16((d-64)*(head+1)/64) for d in range(128)] for head in range(2)] for token in range(4)]
    new_keys = [[[bfloat16(-v-.25) for v in head] for head in token] for token in old_keys]
    new_values = [[[bfloat16(v*1.5+.5) for v in head] for head in token] for token in old_values]
    for encode in (False, True):
        before = bytearray([0x59])*(pages*16384)
        warm = reference_append_bytes(before, slots, old_keys, old_values, encode)
        correct = reference_append_bytes(warm, slots, new_keys, new_values, encode)
        valid = validate_writer_bytes(warm, correct, slots, new_keys, new_values, encode)
        assert valid['unmapped_bytes_unchanged']
        poisoned = bytearray(warm)
        for index in mapped_byte_offsets(slots, pages, encode):
            poisoned[index] = 0xA5
        reject('omitted_graph_writer_'+str(encode),
               lambda: validate_writer_bytes(warm, poisoned, slots, new_keys, new_values, encode))
        reject('stale_graph_token_'+str(encode),
               lambda: validate_writer_bytes(warm, warm, slots, new_keys, new_values, encode))
        key_only = bytearray(warm)
        value_only = bytearray(correct)
        for slot in (17, 46):
            page, column = divmod(slot, 16)
            begin = page*16384+column*512
            key_only[begin:begin+512] = correct[begin:begin+512]
            value_only[begin:begin+512] = warm[begin:begin+512]
        reject('graph_writer_only_K_'+str(encode),
               lambda: validate_writer_bytes(warm, key_only, slots, new_keys, new_values, encode))
        reject('graph_writer_only_V_'+str(encode),
               lambda: validate_writer_bytes(warm, value_only, slots, new_keys, new_values, encode))
        clobbered = bytearray(correct)
        clobbered[16383] ^= 1
        reject('unmapped_padding_write_'+str(encode),
               lambda: validate_writer_bytes(warm, clobbered, slots, new_keys, new_values, encode))
        # Invalid slots are present in the same map; no bytes may map to them.
        assert len(mapped_byte_offsets(slots, pages, encode)) == 2*(776 if encode else 1024)
        if encode:
            bad_codec = bytearray(correct)
            for slot in (17, 46):
                page, column = divmod(slot, 16)
                bad_codec[page*16384+8192+column*256:page*16384+8192+(column+1)*256] = bytes(256)
                for head in range(2):
                    struct.pack_into('<f', bad_codec, page*16384+12288+column*8+head*4, 1e20)
            reject('self_consistent_bad_codec_in_graph',
                   lambda: validate_writer_bytes(warm, bad_codec, slots, new_keys, new_values, True))
    assert consumption_gate([1.005, -.995], [1., -1.])['passed']
    assert consumption_gate([0., 0.], [0., 0.])['passed']
    assert not consumption_gate([0., 0.], [1., -1.])['passed']
    assert not consumption_gate([1.05, -1.05], [1., -1.])['passed']
    assert not consumption_gate([.001, 0.], [0., 0.])['passed']
    return dict(rejected_mutants=rejected, permitted_half_tie_differences=2,
                zero_and_min_normal_scale_checked=True, consumption_gate_limit=0.02,
                zero_code_consumption_rejected=True, gpu_executed=False)


def benchmark_contract_check():
    """Prove writer validation precedes eager and oracle calls use CPU inputs."""
    tree = ast.parse((ROOT/'tools/bench_biv150_v8_bf16p.py').read_text())
    refs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'reference']
    assert refs and all(isinstance(node.args[0], ast.Name) and node.args[0].id == 'expected_cpu' for node in refs)
    reference = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'reference')
    assert not ({'quant', 'encoded_v', 'new_key', 'new_value'} &
                {node.id for node in ast.walk(reference) if isinstance(node, ast.Name)})
    for_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'encode'
                    and any(isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == 'replay' for sub in ast.walk(node)))
    calls = lambda statement, name: any(isinstance(node, ast.Call) and
            ((isinstance(node.func, ast.Attribute) and node.func.attr == name) or
             (isinstance(node.func, ast.Name) and node.func.id == name)) for node in ast.walk(statement))
    replay = next(i for i, node in enumerate(for_loop.body) if calls(node, 'replay'))
    writer = next(i for i, node in enumerate(for_loop.body) if calls(node, 'check_writer'))
    eager = next(i for i, node in enumerate(for_loop.body) if calls(node, 'run'))
    assert replay < writer < eager
    assert any(calls(node, 'index_fill_') for node in for_loop.body[:replay])
    policy = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == 'numeric_timing_policy']
    assert len(policy) == 1 and ast.unparse(policy[0].args[-1]) == 'args.numeric_diagnostic'
    baseline = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'baseline_records' for target in node.targets))
    expected_baseline = ast.parse("[record for record in row['numeric'] if 'cpu' in record and not record['quantized']]", mode='eval').body
    assert ast.dump(baseline.value) == ast.dump(expected_baseline)
    hard = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'hard_checks' for target in node.targets))
    baseline_gate = next(keyword.value for keyword in hard.keywords if keyword.arg == 'baseline_numeric')
    expected_gate = ast.parse("len(baseline_records) == 2 and all(record['cpu']['passed'] for record in baseline_records)", mode='eval').body
    assert ast.dump(baseline_gate) == ast.dump(expected_gate)
    return dict(reference_calls_from_expected_cpu=True, mapped_cache_poison_before_replay=True,
                graph_writer_validated_before_eager=True, explicit_numeric_diagnostic_policy=True,
                baseline_cpu_pointwise_required_for_both_patterns=True)


def numeric_diagnostic_check():
    hard = dict.fromkeys(('finite', 'baseline_numeric', 'codec', 'slot_mapping', 'unmapped_bytes', 'graph_eager'), True)
    assert numeric_timing_policy(True, hard)['timing_rounds'] == 3
    decision = numeric_timing_policy(False, hard, diagnostic=True)
    assert decision['timing_rounds'] == 1 and not decision['numeric_passed']
    assert not decision['component_checks_passed'] and decision['precision_gate'] is False
    rejected = []
    for diagnostic in (False, True):
        for fault in hard:
            bad = dict(hard); bad[fault] = False
            try:
                numeric_timing_policy(False, bad, diagnostic=diagnostic)
            except AssertionError:
                rejected.append((diagnostic, fault))
            else:
                raise AssertionError(('Diagnostic bypassed hard fault', diagnostic, fault))
    try:
        numeric_timing_policy(False, hard)
    except AssertionError:
        rejected.append((False, 'numeric'))
    else:
        raise AssertionError('Default mode timed failed numerics')
    # A finite but independently inaccurate baseline cannot qualify a diagnostic
    # comparison, even if an inconsistent caller marks aggregate numerics true.
    bad_baseline = dict(hard, baseline_numeric=False)
    assert bad_baseline['finite'] is True
    for numeric_passed in (False, True):
        try:
            numeric_timing_policy(numeric_passed, bad_baseline, diagnostic=True)
        except AssertionError:
            pass
        else:
            raise AssertionError('Finite baseline CPU pointwise failure reached timing')
    trace = dict(coordinate=[0, 0, 0], scores=[i*.037 for i in range(17)],
                 canonical_codes=[2]+[i%7-3 for i in range(16)],
                 validated_actual_codes=[1]+[i%7-3 for i in range(16)],
                 canonical_scales=[1.]*17, validated_actual_scales=[1.]*17,
                 segments=2, actual_output=.25)
    attributed = attribute_coordinate(trace)
    assert attributed['contributions']['validated_codec_minus_canonical'] != 0
    assert attributed['contributions']['bf16_weighted_p_minus_validated_codec'] != 0
    assert not attributed['acceptance_tolerance_changed']
    return dict(numeric_failure_retained_in_diagnostic=True, diagnostic_timing_pairs=1,
                rejected_fault_modes=rejected, default_numeric_failure_rejected=True,
                finite_baseline_pointwise_failure_rejected=True,
                cpu_codec_and_weighted_p_effects_separated=True, gpu_executed=False)


if __name__ == "__main__":
    print(json.dumps(dict(source=source_check(), layout=layout_check(), algebra=algebra_check(),
                          validation_falsification=falsification_check(),
                          benchmark_contract=benchmark_contract_check(),
                          numeric_diagnostic=numeric_diagnostic_check(),
                          precision_gate=False, official_score=False), indent=2))
