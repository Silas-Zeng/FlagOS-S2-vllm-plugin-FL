"""Production wrapper checks: real cache stride, graph inputs, fallback and FP32."""

import argparse
import hashlib
import json
import socket
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_biv150_int8_kv_decode import dense_reference
from biv150_benchmark_utils import full_call, graph_time, inputs, load_module


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline = load_module(args.baseline, "adaptive_wrapper_baseline")
    candidate = load_module(args.source, "adaptive_wrapper_candidate")
    fallbacks = []

    def fallback(**kwargs):
        fallbacks.append(True)
        return "fallback"

    wrapper, _ = candidate._make_wrapper(
        SimpleNamespace(unified_attention=fallback, reduce_segments=None)
    )
    result = dict(
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), records=[]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for batch in [1, 2, 3, 4, 8, 64]:
        for length in [100, 4096, 16384]:
            c = inputs(batch, length)
            packed = torch.stack([c["k"], c["v"]], 1)
            c["k"], c["v"] = packed.unbind(1)
            width = baseline._pick_segments(batch, 2, 16)
            full_call(baseline, c, width)
            reference = c["out"].clone()
            kwargs = dict(
                q=c["q"],
                k=c["k"],
                v=c["v"],
                out=c["out"],
                cu_seqlens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
                max_seqlen_q=1,
                seqused_k=c["lengths"],
                max_seqlen_k=length,
                softmax_scale=128**-0.5,
                causal=True,
                block_table=c["table"],
                seq_threshold_3D=64,
                num_par_softmax_segments=16,
                softmax_segm_output=c["scratch"],
                softmax_segm_max=c["maxima"],
                softmax_segm_expsum=c["sums"],
                alibi_slopes=None,
                sinks=None,
                qq_bias=None,
                mm_prefix_range=None,
                output_scale=None,
                k_scale_cache=None,
                v_scale_cache=None,
                softcap=0.0,
                use_alibi_sqrt=False,
                window_size=(-1, -1),
                chunk_lookback=-1,
                use_td=False,
                kv_quant_mode=0,
            )
            row = dict(batch=batch, length=length, k_stride=c["k"].stride(), checks=[])
            try:
                before = candidate.stats()["calls"]
                wrapper(**kwargs)
                assert candidate.stats()["calls"] == before + 1
                torch.testing.assert_close(c["out"], reference, atol=0.005, rtol=0.005)
                old_ms = graph_time(
                    partial(full_call, baseline, c, width), repeats=15, unroll=4
                )[0]
                new_ms = graph_time(partial(wrapper, **kwargs), repeats=15, unroll=4)[0]
                row.update(
                    baseline_ms=old_ms, candidate_ms=new_ms, speedup=old_ms / new_ms
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    wrapper(**kwargs)
                for n in sorted(
                    {0, 1, min(17, length), min(513, length), min(2049, length), length}
                ):
                    c["lengths"].fill_(n)
                    if batch > 1:
                        c["lengths"][0] = 0
                    # Update K/V in their real strided views before each replay.
                    c["k"].add_(0.001)
                    c["v"].mul_(0.999)
                    c["out"].fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    for i in sorted({0, batch - 1}):
                        gold = dense_reference(c, c["k"], c["v"], i).to(torch.bfloat16)
                        torch.testing.assert_close(
                            c["out"][i], gold, atol=0.005, rtol=0.005
                        )
                    assert torch.isfinite(c["out"]).all().item()
                    row["checks"].append(dict(length=n, passed=True))
                for key, value in [
                    ("sinks", torch.zeros(16, device="cuda")),
                    ("kv_quant_mode", 1),
                    ("max_seqlen_q", 2),
                ]:
                    changed = dict(kwargs)
                    changed[key] = value
                    assert wrapper(**changed) == "fallback"
                row["passed"] = True
            except Exception as error:
                row["error"] = str(error)[:1500]
            result["records"].append(row)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
            del c, packed, kwargs, reference, graph
            torch.cuda.empty_cache()
    assert all(r.get("passed") for r in result["records"]), "Wrapper validation failed"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
