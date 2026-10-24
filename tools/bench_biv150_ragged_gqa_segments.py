"""Measure GQA segment policy across uneven program-grid waves."""

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
    torch.manual_seed(20261024)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise RuntimeError("Never overwrite evidence")
    module = load_module(args.source, "gqa_boundary_shipping")
    result = dict(source_sha256=expected, service_installed=False, records=[])
    batches = args.batches
    assert batches and len(set(batches)) == len(batches) and all(1 <= b <= 64 for b in batches)
    expected_cases = len(batches) * 4 * 2 * 5
    for batch, ragged in [
        (b, r) for b in batches for r in (False, True)
    ]:
        for length in (4096, 5119, 16384, 17407):
            case = inputs(batch, length, mixed=ragged)
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
            base_query, base_lengths = case["q"].clone(), case["lengths"].clone()
            for segments in (1, 2, 4, 8, 16):
                row = dict(
                    batch=batch,
                    kv=length,
                    ragged=ragged,
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
                    original_q = case["q"].clone()
                    original_lengths = case["lengths"].clone()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        full_call(module, case, segments)
                    for factor in (0.5, 1.25):
                        case["q"].copy_(original_q * factor)
                        case["lengths"].copy_(original_lengths // 2)
                        case["lengths"][-1] = 0
                        for key in ("scratch", "maxima", "sums"):
                            case[key].fill_(float("nan"))
                        full_call(module, case, original)
                        changed_gold = case["out"].clone()
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(case["out"], changed_gold,
                                                   atol=0.005, rtol=0.005)
                        assert torch.equal(case["out"][-1], torch.zeros_like(case["out"][-1]))
                    case["q"].copy_(original_q)
                    case["lengths"].copy_(original_lengths)
                    del graph, original_q, original_lengths, changed_gold
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
                        changed_input_graph=True,
                        independent_cpu_reference=True,
                        rounds=rounds,
                        median_speedup=statistics.median(r["speedup"] for r in rounds),
                    )
                except Exception as error:
                    row["error"] = repr(error)
                finally:
                    case["q"].copy_(base_query)
                    case["lengths"].copy_(base_lengths)
                result["records"].append(row)
                args.out.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
            del case, packed, k, v, gold, base_query, base_lengths

    if len(result["records"]) != expected_cases or not all(r.get("passed") for r in result["records"]):
        raise RuntimeError("Incomplete numerical gates; no service claim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[15,16,29,30,31,32,60,61,62,63,64])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
