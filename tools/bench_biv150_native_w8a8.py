"""Numerical, graph-capture and fullgraph tests for the native linear operator."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_biv150_fastpaths import graph_time


def main(args):
    spec = importlib.util.spec_from_file_location("native_w8a8", args.source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library)
    native.load_library()
    torch.manual_seed(20261023)
    shapes = [(2560, 2048), (2048, 2048), (12288, 2048)]
    if args.extended:
        shapes = [(2048, 6144), (130560, 2048)]
    for n, k in shapes:
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        wq, sw = native.quantize_weight(weight)
        marker = torch.zeros((), dtype=weight.dtype, device=weight.device).expand_as(weight) if args.compressed else weight
        for m in ([1, 16, 17, 32, 64] if n > 20000 else [1, 16, 17, 32, 64, 2048, 8192]):
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            op = lambda: torch.ops.biv150_native.linear(x, wq, sw, marker, None, args.min_rows)
            got = op()
            if not args.compressed and m < max(17, args.min_rows):
                ref = F.linear(x, weight)
                torch.testing.assert_close(got, ref, atol=0, rtol=0)
            else:
                sa = x.float().abs().amax(1).clamp_min(1.e-6) / 127.0
                scaled = x.float() / sa[:, None]
                aq = torch.trunc(scaled + torch.where(scaled >= 0, 0.5, -0.5)).clamp_(-127, 127).to(torch.int8)
                if m < 32:
                    aq = F.pad(aq, (0, 0, 0, 32-m))
                ref = (torch._int_mm(aq, wq)[:m].float() * (sa[:, None] * sw[None, :])).to(torch.bfloat16)
                torch.testing.assert_close(got, ref, atol=0.01, rtol=0.01)
            diff = (got.float() - ref.float()).abs().max().item()
            bf16_ms = graph_time(lambda: F.linear(x, weight), repeats=9, unroll=4)[0]
            native_ms = graph_time(op, repeats=9, unroll=4)[0]
            print(json.dumps(dict(m=m, n=n, k=k, min_rows=args.min_rows,
                                  bf16_ms=bf16_ms, native_ms=native_ms,
                                  speedup=bf16_ms / native_ms, qdq_max_diff=diff)), flush=True)
            del x, got, ref
        del weight, wq, sw
        torch.cuda.empty_cache()
    # The opaque operator must be traceable with both a tiny decode and large prefill M.
    weight = torch.randn(2560, 2048, dtype=torch.bfloat16, device="cuda") * 0.02
    wq, sw = native.quantize_weight(weight)
    marker = torch.zeros((), dtype=weight.dtype, device=weight.device).expand_as(weight) if args.compressed else weight

    def apply(x):
        return torch.ops.biv150_native.linear(x, wq, sw, marker, None, args.min_rows)

    compiled = torch.compile(apply, fullgraph=True, dynamic=True)
    for m in [32, 2048, 64, 8192]:
        x = torch.randn(m, 2048, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(compiled(x), apply(x), atol=0, rtol=0)
        print(json.dumps(dict(fullgraph="passed", m=m)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=1024)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    main(parser.parse_args())
