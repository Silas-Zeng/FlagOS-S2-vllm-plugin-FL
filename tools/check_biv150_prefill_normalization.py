"""Independent FP32 checks of an audited zero-initial-mass prefill clone."""

import argparse
import importlib
import importlib.util
import json
import socket
import statistics
import sys
from functools import partial
from pathlib import Path

import torch


def invoke(fn, dest, **kwargs):
    fn(out=dest, **kwargs)


def measure(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            fn()
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 3)
    return statistics.median(times)


def main(args):
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", 9031)) != 0, "Service running"
    module = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    spec = importlib.util.spec_from_file_location(
        "biv150_normalization_check", args.source
    )
    patch = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = patch
    spec.loader.exec_module(patch)
    baseline_function = module.unified_attention
    from vllm_fl.patches.biv150_decode_launch_runtime import install as install_launch
    from vllm_fl.patches.biv150_gqa_decode import install as install_gqa

    launch_state = install_launch()
    gqa_state = install_gqa()
    assert launch_state["installed"] and gqa_state["installed"]
    metadata = patch.install()
    assert metadata["installed"], metadata
    normalized = module.unified_attention
    metadata["existing_stack"] = {"decode_launch": launch_state, "gqa": gqa_state}
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata["reference_tf32_allowed"] = False
    torch.manual_seed(20261023)
    cases = [
        ([16], [128]),
        ([32], [32]),
        ([127], [257]),
        ([2048], [2048]),
        ([2048], [4096]),
        ([2048], [16384]),
        ([1, 16, 40, 0], [65, 127, 257, 0]),
        ([8, 33, 1], [16384, 4097, 100]),
    ]
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for qlens, klens in cases:
        counts = [(k + 15) // 16 for k in klens]
        total = sum(counts)
        maxpages = ((max(counts) + 3) // 4) * 4
        q = torch.randn(sum(qlens), 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(total, 16, 2, 128, device=q.device, dtype=q.dtype)
        v = torch.randn_like(k)
        table = torch.zeros((len(qlens), maxpages), device=q.device, dtype=torch.int32)
        shuffled = torch.randperm(total, device=q.device, dtype=torch.int32)
        offset = 0
        for i, count in enumerate(counts):
            table[i, :count] = shuffled[offset : offset + count]
            offset += count
        starts = [0]
        for size in qlens:
            starts.append(starts[-1] + size)
        cu = torch.tensor(starts, device=q.device, dtype=torch.int32)
        lens = torch.tensor(klens, device=q.device, dtype=torch.int32)
        out = torch.empty_like(q)
        old = torch.empty_like(q)
        dense = torch.empty_like(q)

        call = partial(
            invoke,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            max_seqlen_q=max(qlens),
            seqused_k=lens,
            max_seqlen_k=max(klens),
            softmax_scale=128**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0.0,
            q_descale=1.0,
            k_descale=1.0,
            v_descale=1.0,
        )

        call(normalized, out)
        call(baseline_function, old)
        for i, (qlen, kv, count) in enumerate(zip(qlens, klens, counts)):
            if not qlen:
                continue
            ids = table[i, :count].long()
            kd = k[ids].reshape(-1, 2, 128)[:kv].float().repeat_interleave(8, 1)
            vd = v[ids].reshape(-1, 2, 128)[:kv].float().repeat_interleave(8, 1)
            for start in range(0, qlen, 64):
                stop = min(start + 64, qlen)
                query = q[starts[i] + start : starts[i] + stop].float()
                scores = torch.einsum("qhd,khd->hqk", query, kd) * 128**-0.5
                valid = (
                    torch.arange(kv, device=q.device)[None, :]
                    <= kv - qlen + torch.arange(start, stop, device=q.device)[:, None]
                )
                probability = scores.masked_fill(~valid[None], -float("inf")).softmax(
                    -1
                )
                dense[starts[i] + start : starts[i] + stop] = torch.einsum(
                    "hqk,khd->qhd", probability, vd
                ).to(q.dtype)
        if qlens == [16]:
            sink = torch.zeros(16, device=q.device, dtype=torch.float32)

            sink_call = partial(call, sinks=sink)

            a = torch.empty_like(q)
            b = torch.empty_like(q)
            calls_before = patch._state["calls"]
            sink_call(normalized, a)
            sink_call(baseline_function, b)
            assert torch.equal(a, b) and patch._state["calls"] == calls_before, (
                "Sink fallback changed"
            )
            metadata["sink_fallback"] = "bit-exact original; normalization not selected"
        row = dict(
            qlens=qlens,
            klens=klens,
            max_abs_diff=(out.float() - dense.float()).abs().max().item(),
            old_max_abs_diff=(old.float() - dense.float()).abs().max().item(),
        )
        try:
            torch.testing.assert_close(out, dense, atol=0.005, rtol=0.005)
            row["fp32_reference"] = "passed"
        except Exception as error:
            row["error"] = str(error)[:500]
        row["old_ms"] = measure(partial(call, baseline_function, old))
        row["new_ms"] = measure(partial(call, normalized, out))
        row["speedup"] = row["old_ms"] / row["new_ms"]
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps(dict(metadata=metadata, records=rows), indent=2))
        del call, q, k, v, table, shuffled, cu, lens, out, old, dense
        torch.cuda.empty_cache()
    assert all("error" not in row for row in rows), (
        "Independent reference failure; do not install"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
