"""Test exact softmax rescaling elision after a page does not change any maximum."""
import argparse
import json
from pathlib import Path

import torch

from bench_biv150_fastpaths import (load_module, inputs, full_call, graph_time,
                                   check_reference)

OLD = '''        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(V.dtype), V
        )
        running_max = new_max'''
NEW = '''        partial = tl.dot(probability.to(V.dtype), V)
        page_sum = tl.sum(probability, 1)
        if tl.sum((new_max > running_max).to(tl.int32), 0) > 0:
            rescale = tl.math.exp2(running_max - new_max)
            running_sum = running_sum * rescale + page_sum
            accumulator = accumulator * rescale[:, None] + partial
        else:
            running_sum = running_sum + page_sum
            accumulator = accumulator + partial
        running_max = new_max'''
LAZY_OLD = '''        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
''' + OLD
LAZY_NEW = '''        # Anchor shifts are integral in log2 space, hence exact powers of two.
        # Allow probabilities up to 256 before renormalizing. All previous sums
        # and accumulators retain their common scale between those rare shifts.
        candidate_max = tl.maximum(running_max, tl.ceil(tl.max(scores, 1)))
        new_max = tl.where(candidate_max > running_max + 8.0, candidate_max, running_max)
        probability = tl.math.exp2(scores - new_max[:, None])
''' + NEW


def long_reference(c):
    for i in [0, min(7, c["q"].shape[0]-1)]:
        length = int(c["lengths"][i])
        if not length:
            continue
        ids = c["table"][i, :(length+15)//16].long()
        k = c["k"][ids].reshape(-1, 2, 128)[:length].float().repeat_interleave(8, 1)
        v = c["v"][ids].reshape(-1, 2, 128)[:length].float().repeat_interleave(8, 1)
        q = c["q"][i].float()
        p = torch.softmax(torch.einsum("hd,khd->hk", q, k)*128**-0.5, -1)
        ref = torch.einsum("hk,khd->hd", p, v).to(torch.bfloat16)
        torch.testing.assert_close(c["out"][i], ref, atol=0.005, rtol=0.005)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source.read_text()
    assert source.count(OLD) == 2
    assert source.count(LAZY_OLD) == 2
    variants = {"baseline": source, "branch": source.replace(OLD, NEW, 1),
                "lazy8ceil": source.replace(LAZY_OLD, LAZY_NEW, 1)}
    variants["fp16_dot"] = source.replace("tl.dot(Q, K)", "tl.dot(Q.to(tl.float16), K.to(tl.float16))").replace(
        "probability.to(V.dtype), V", "probability.to(tl.float16), V.to(tl.float16)")
    variants["cache_cg"] = source.replace("K = tl.load(key_cache_ptr + k_offsets)",
        'K = tl.load(key_cache_ptr + k_offsets, cache_modifier=".cg")').replace(
        "V = tl.load(value_cache_ptr + v_offsets)",
        'V = tl.load(value_cache_ptr + v_offsets, cache_modifier=".cg")')
    mods = {}
    for name, text in variants.items():
        path = args.out / f"softmax_{name}.py"
        path.write_text(text)
        mods[name] = load_module(path, f"softmax_{name}")
    result = []
    for b, kv in [(64, 4096), (64, 5120), (32, 16384), (16, 17408), (8, 32768), (1, 100)]:
        seg = mods["baseline"]._pick_segments(b, 2, 16)
        for mixed in [False, True]:
            c = inputs(b, kv, mixed=mixed)
            full_call(mods["baseline"], c, seg)
            ref = c["out"].clone()
            base = graph_time(lambda: full_call(mods["baseline"], c, seg))[0]
            for name in ["branch", "lazy8ceil", "fp16_dot", "cache_cg"]:
                full_call(mods[name], c, seg)
                # Control flow may prevent the compiler fusing the old accumulator
                # into dot's FP32 accumulator; record bit equality rather than
                # assuming an identical reduction order.
                torch.testing.assert_close(c["out"], ref, atol=0.005, rtol=0.005)
                error = (c["out"].float()-ref.float()).abs().max().item()
                if mixed:
                    check_reference(c)
                else:
                    long_reference(c)
                other = graph_time(lambda: full_call(mods[name], c, seg))[0]
                row = dict(batch=b, kv=kv, segments=seg, mixed=mixed, name=name,
                           baseline_ms=base, candidate_ms=other, speedup=base/other,
                           bit_exact=error == 0, max_abs_diff=error)
                result.append(row)
                print(json.dumps(row), flush=True)
        del c, ref
        torch.cuda.empty_cache()
    (args.out / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
