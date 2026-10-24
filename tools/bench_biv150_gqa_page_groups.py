"""Fused BF16 decode over several logical KV pages per online-softmax tile.

No score materialization, quantization or service hook. Physical pages remain
arbitrary. The reduction uses the same logical segment boundaries as the main.
"""

import argparse
import hashlib
import json
import socket
import statistics
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from biv150_benchmark_utils import full_call, graph_time, inputs, load_module

AUDITED_SHA = "1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a"


def grouped_source(original, pages):
    start = original.index("@triton.jit\ndef _gqa_decode_attention(")
    end = original.index("@triton.jit\ndef _reduce_partials(", start)
    body = original[start:end]
    changes = {
        "def _gqa_decode_attention(": "def _gqa_decode_page_groups(",
        "    seq_idx = tl.program_id(0)": (
            f"    STEP: tl.constexpr = PAGE_SIZE * {pages}\n"
            f"    offs_page = tl.arange(0, {pages})\n"
            "    seq_idx = tl.program_id(0)"
        ),
        "SEGMENTS * PAGE_SIZE)": "SEGMENTS * STEP)",
        "steps_per_segment * PAGE_SIZE": "steps_per_segment * STEP",
        "tl.arange(0, PAGE_SIZE)": "tl.arange(0, STEP)",
        "((upper - lower) // PAGE_SIZE) * PAGE_SIZE": "((upper - lower) // STEP) * STEP",
        "position += PAGE_SIZE": "position += STEP",
        "        page = tl.load(block_table_row + position // PAGE_SIZE).to(tl.int64)": (
            "        pages = tl.load(block_table_row + position // PAGE_SIZE + offs_page).to(tl.int64)\n"
            f"        page = tl.reshape(tl.broadcast_to(pages[:, None], ({pages}, PAGE_SIZE)), (STEP,))"
        ),
        "        page = tl.load(block_table_row + full_upper // PAGE_SIZE).to(tl.int64)\n"
        "        column_valid = full_upper + offs_step < upper": (
            "        column_valid = full_upper + offs_step < upper\n"
            "        pages = tl.load(block_table_row + full_upper // PAGE_SIZE + offs_page,\n"
            "            mask=full_upper + offs_page * PAGE_SIZE < upper, other=0).to(tl.int64)\n"
            f"        page = tl.reshape(tl.broadcast_to(pages[:, None], ({pages}, PAGE_SIZE)), (STEP,))"
        ),
        "page * k_stride_block": "page[None, :] * k_stride_block",
        "page * v_stride_block": "page[:, None] * v_stride_block",
    }
    for before, after in changes.items():
        if before not in body:
            raise RuntimeError(f"Unexpected audited source: missing {before!r}")
        body = body.replace(before, after)
    return original[:start] + body + original[end:]


class Launch:
    def __init__(self, kernel, warps, stages):
        self.kernel, self.warps, self.stages = kernel, warps, stages
        self.last = None

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            kwargs.update(num_warps=self.warps, num_stages=self.stages)
            self.last = self.kernel[grid](*args, **kwargs)
            return self.last

        return invoke


class GroupReduce:
    def __init__(self, kernel, pages):
        self.kernel, self.pages = kernel, pages

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            kwargs["page_size"] *= self.pages
            return self.kernel[grid](*args, **kwargs)

        return invoke


def dense_reference(case, row):
    length = int(case["lengths"][row].cpu())
    if length == 0:
        return torch.zeros((16, 128), dtype=torch.bfloat16)
    ids = case["table"][row, : (length + 15) // 16].long()
    # Gather BF16 first; every FP64 conversion and arithmetic is on CPU.
    k = case["k"][ids].reshape(-1, 2, 128)[:length].cpu().double()
    v = case["v"][ids].reshape(-1, 2, 128)[:length].cpu().double()
    q = case["q"][row].reshape(2, 8, 128).cpu().double()
    p = torch.einsum("hgd,thd->hgt", q, k).mul(128**-0.5).softmax(-1)
    return torch.einsum("hgt,thd->hgd", p, v).reshape(16, 128).to(torch.bfloat16)


def graph_changed_inputs(module, baseline, case, segments):
    saved = {key: case[key].clone() for key in ("q", "lengths", "table")}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        full_call(module, case, segments)
    try:
        for lengths in ([0, 1, 15, 16, 17, 31, 64, 127], [127] * 8, [0] * 8):
            case["q"].normal_().mul_(0.75)
            case["lengths"].copy_(
                torch.tensor(lengths, device="cuda", dtype=case["lengths"].dtype)
            )
            case["table"].copy_(saved["table"].flip(1))
            for key in ("scratch", "maxima", "sums"):
                case[key].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            got = case["out"].clone()
            full_call(baseline, case, segments)
            torch.testing.assert_close(got, case["out"], atol=0.005, rtol=0.005)
            for row in (0, 1, 7):
                torch.testing.assert_close(
                    got[row].cpu(), dense_reference(case, row), atol=0.005, rtol=0.005
                )
    finally:
        for key, tensor in saved.items():
            case[key].copy_(tensor)
        full_call(module, case, segments)


def main(args):
    with socket.socket() as guard:
        if guard.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Service running; no overlapping GPU probe")
    original = args.source.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != AUDITED_SHA:
        raise RuntimeError("Unrecognized shipping GQA source")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    baseline = load_module(args.source, "page_groups_baseline")
    variants = []
    for pages in [2, 4, 8]:
        path = args.out.parent / f"gqa-page-groups-{pages}.py"
        path.write_text(grouped_source(original, pages))
        raw = load_module(path, f"page_groups_{pages}")
        for warps in [1, 2, 4]:
            for stages in [1, 2]:
                launch = Launch(raw._gqa_decode_page_groups, warps, stages)
                module = SimpleNamespace(
                    _gqa_decode_attention=launch,
                    _reduce_partials=GroupReduce(raw._reduce_partials, pages),
                    LOG2E=raw.LOG2E,
                )
                variants.append((pages, warps, stages, module, launch))
    result = dict(source_sha256=AUDITED_SHA, service_installed=False, records=[])
    cases = [
        (64, 4096, False),
        (32, 16384, False),
        (31, 16896, False),
        (40, 17023, True),
        (8, 127, True),
        (1, 100, False),
    ]
    for batch, kv, mixed in cases:
        case = inputs(batch, kv, mixed=mixed)
        # Use actual service-like packed K/V strides, not separate contiguous KV.
        raw = torch.empty(
            (case["k"].shape[0], 2, 16, 2, 128), device="cuda", dtype=torch.bfloat16
        )
        k, v = raw.unbind(1)
        k.copy_(case["k"])
        v.copy_(case["v"])
        case.update(k=k, v=v)
        segments = baseline._pick_segments(batch, 2, 16)
        full_call(baseline, case, segments)
        gold = case["out"].clone()
        rows = sorted({0, min(1, batch - 1), batch - 1})
        refs = {row: dense_reference(case, row) for row in rows}
        for row, ref in refs.items():
            torch.testing.assert_close(gold[row].cpu(), ref, atol=0.005, rtol=0.005)
        for pages, warps, stages, module, launch in variants:
            record = dict(
                batch=batch,
                kv=kv,
                mixed=mixed,
                pages=pages,
                warps=warps,
                stages=stages,
                segments=segments,
                k_stride=k.stride(),
            )
            try:
                full_call(module, case, segments)
                torch.testing.assert_close(case["out"], gold, atol=0.005, rtol=0.005)
                for row, ref in refs.items():
                    torch.testing.assert_close(
                        case["out"][row].cpu(), ref, atol=0.005, rtol=0.005
                    )
                if batch == 8 and (pages, warps, stages) == (2, 1, 1):
                    graph_changed_inputs(module, baseline, case, segments)
                    record["graph_changed_inputs_passed"] = True
                rounds = []
                for index in range(3):
                    calls = [
                        ("baseline_ms", partial(full_call, baseline, case, segments)),
                        ("candidate_ms", partial(full_call, module, case, segments)),
                    ]
                    if index % 2:
                        calls.reverse()
                    times = {
                        name: graph_time(fn, repeats=9, unroll=4)[0]
                        for name, fn in calls
                    }
                    times["speedup"] = times["baseline_ms"] / times["candidate_ms"]
                    rounds.append(times)
                record.update(
                    passed=True,
                    rounds=rounds,
                    median_speedup=statistics.median(t["speedup"] for t in rounds),
                    max_abs_diff=(case["out"].float() - gold.float())
                    .abs()
                    .max()
                    .item(),
                    n_regs=launch.last.n_regs,
                    n_spills=launch.last.n_spills,
                    shared=launch.last.metadata.shared,
                )
            except Exception as error:
                record.update(passed=False, error=str(error)[:1600])
            result["records"].append(record)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(record), flush=True)
        del case, raw, k, v, gold, refs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
