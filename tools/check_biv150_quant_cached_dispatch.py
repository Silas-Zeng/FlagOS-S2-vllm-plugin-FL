"""Check selected native dispatch, row boundaries, CPU quantization and Graphs."""

import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import torch


def main(args):
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight

    load_library()
    torch.manual_seed(20261024)
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    q, scales = quantize_weight(weight)
    marker = torch.zeros((), device="cuda", dtype=weight.dtype).expand_as(weight)
    result = dict(
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(), records=[]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for rows in (
        17,
        18,
        31,
        32,
        33,
        39,
        40,
        41,
        63,
        64,
        65,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
    ):
        for pitch in (2048, 2080):
            x = torch.randn(rows, pitch, device="cuda", dtype=weight.dtype)[:, :2048]

            selected = partial(
                torch.ops.biv150_native.gate_silu_auto, x, q, scales, marker, 17
            )
            old = partial(torch.ops.biv150_native.gate_silu, x, q, scales, marker, 17)

            for factor in (0.0, 1e-9, 1.0, 100.0):
                x.normal_().mul_(factor)
                torch.testing.assert_close(selected(), old(), atol=0, rtol=0)
                if rows <= 2048:
                    cached_q, cached_s = torch.ops.biv150_native.quantize_rows_probe(
                        x, 256
                    )
                    old_q, old_s = torch.ops.biv150_native.quantize_rows_probe(x, 0)
                    torch.testing.assert_close(cached_q, old_q, atol=0, rtol=0)
                    torch.testing.assert_close(cached_s, old_s, atol=0, rtol=0)
                    host = x.cpu().float()
                    host_s = host.abs().amax(1).clamp_min(1e-6) / 127
                    raw = host / host_s[:, None]
                    gold = (
                        raw.abs()
                        .add(0.5)
                        .floor()
                        .copysign(raw)
                        .clamp(-127, 127)
                        .to(torch.int8)
                    )
                    torch.testing.assert_close(
                        cached_q[:rows].cpu(), gold, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        cached_s[:rows].cpu(), host_s, atol=0, rtol=0
                    )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = selected()
            for factor in (0.0, 0.5, 4.0):
                x.normal_().mul_(factor)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(replayed, old(), atol=0, rtol=0)
            row = dict(
                m=rows,
                pitch=pitch,
                cached_selected=rows <= 2048,
                bit_exact=True,
                changed_input_graph=True,
            )
            result["records"].append(row)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
            del graph, replayed
    result["completed"] = True
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
