"""Diagnose unchanged GQA splits near the B31/B32 launch-policy boundary."""

import argparse
import hashlib
import json
import socket
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
from biv150_benchmark_utils import full_call, graph_time, inputs, load_module


def reference(case, row):
    length = int(case["lengths"][row].cpu())
    if not length:
        return torch.zeros((16, 128), dtype=torch.bfloat16)
    ids = case["table"][row, : (length + 15) // 16].long()
    k = case["k"][ids].reshape(-1, 2, 128)[:length].cpu().double()
    v = case["v"][ids].reshape(-1, 2, 128)[:length].cpu().double()
    q = case["q"][row].reshape(2, 8, 128).cpu().double()
    p = torch.einsum("hgd,thd->hgt", q, k).mul(128**-0.5).softmax(-1)
    return torch.einsum("hgt,thd->hgd", p, v).reshape(16, 128).bfloat16()


def main(args):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Existing service")
    if "No running processes found" not in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ):
        raise RuntimeError("GPU occupied")
    text = args.source.read_bytes()
    expected = "1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a"
    if hashlib.sha256(text).hexdigest() != expected:
        raise RuntimeError("Unrecognized shipping GQA source")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise RuntimeError("Never overwrite evidence")
    module = load_module(args.source, "gqa_boundary_shipping")
    result = dict(source_sha256=expected, service_installed=False, records=[])
    for batch in (30, 31, 32, 33, 40, 64):
        for length in (4096, 16896):
            case = inputs(batch, length, mixed=batch == 40)
            packed = torch.empty(
                (case["k"].shape[0], 2, 16, 2, 128),
                device="cuda",
                dtype=torch.bfloat16,
            )
            k, v = packed.unbind(1)
            k.copy_(case["k"])
            v.copy_(case["v"])
            case.update(k=k, v=v)
            original = module._pick_segments(batch, 2, 16)
            full_call(module, case, original)
            gold = case["out"].clone()
            rows = sorted({0, 1, batch - 1})
            refs = {row: reference(case, row) for row in rows}
            for row in rows:
                torch.testing.assert_close(
                    gold[row].cpu(), refs[row], atol=0.005, rtol=0.005
                )
            for segments in (2, 4, 8, 16):
                row = dict(
                    batch=batch,
                    kv=length,
                    ragged=batch == 40,
                    baseline_segments=original,
                    segments=segments,
                    kv_stride=list(k.stride()),
                )
                try:
                    for key in ("scratch", "maxima", "sums"):
                        case[key].fill_(float("nan"))
                    full_call(module, case, segments)
                    torch.testing.assert_close(
                        case["out"], gold, atol=0.005, rtol=0.005
                    )
                    for index in rows:
                        torch.testing.assert_close(
                            case["out"][index].cpu(),
                            refs[index],
                            atol=0.005,
                            rtol=0.005,
                        )
                    rounds = []
                    for iteration in range(3):
                        calls = [("baseline_ms", original), ("candidate_ms", segments)]
                        if iteration % 2:
                            calls.reverse()
                        timings = {
                            name: graph_time(
                                partial(full_call, module, case, width),
                                repeats=9,
                                unroll=4,
                            )[0]
                            for name, width in calls
                        }
                        timings["speedup"] = (
                            timings["baseline_ms"] / timings["candidate_ms"]
                        )
                        rounds.append(timings)
                    row.update(
                        passed=True,
                        rounds=rounds,
                        median_speedup=statistics.median(r["speedup"] for r in rounds),
                    )
                except Exception as error:
                    row["error"] = repr(error)
                result["records"].append(row)
                args.out.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
            del case, packed, k, v, gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
