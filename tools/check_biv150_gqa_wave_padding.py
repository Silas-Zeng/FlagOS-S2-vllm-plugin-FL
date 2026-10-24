"""Gate physical-B40 split policy on zero-padded live rows before service tests."""

import argparse
import hashlib
import json
import math
import socket
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
from biv150_benchmark_utils import full_call, graph_time, inputs, load_module

BASE = "1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a"
NEW = "2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1"


def reference(c, row):
    length = int(c["lengths"][row].cpu())
    if not length:
        return torch.zeros((16, 128), dtype=torch.bfloat16)
    ids = c["table"][row, : (length + 15) // 16].long()
    k = c["k"][ids].reshape(-1, 2, 128)[:length].cpu().double()
    v = c["v"][ids].reshape(-1, 2, 128)[:length].cpu().double()
    q = c["q"][row].reshape(2, 8, 128).cpu().double()
    p = torch.einsum("hgd,thd->hgt", q, k).mul(128**-0.5).softmax(-1)
    return torch.einsum("hgt,thd->hgd", p, v).reshape(16, 128).bfloat16()


def set_live(c, original_table, live, length):
    c["table"].copy_(original_table)
    c["table"][live:] = -1
    c["lengths"].fill_(length)
    c["lengths"][live:] = 0
    for key in ("scratch", "maxima", "sums"):
        c[key].fill_(float("nan"))


def verify(c, gold, live):
    torch.testing.assert_close(c["out"], gold, atol=0.005, rtol=0.005)
    for row in sorted({0, max(0, live - 1), min(live, 39), 39}):
        torch.testing.assert_close(
            c["out"][row].cpu(), reference(c, row), atol=0.005, rtol=0.005
        )
    if live < 40:
        assert torch.count_nonzero(c["out"][live:]).item() == 0


def main(args):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Existing service")
    if "No running processes found" not in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ):
        raise RuntimeError("GPU occupied")
    source = args.source.resolve()
    old = source / "tools/reference/biv150_gqa_decode_before_wave.py"
    new = source / "vllm_fl/patches/biv150_gqa_decode.py"
    assert hashlib.sha256(old.read_bytes()).hexdigest() == BASE
    assert hashlib.sha256(new.read_bytes()).hexdigest() == NEW
    baseline = load_module(old, "wave_padding_before")
    candidate = load_module(new, "wave_padding_after")
    assert baseline._pick_segments(40, 2, 16) == 4
    assert candidate._pick_segments(40, 2, 16) == 16
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(baseline_sha256=BASE, candidate_sha256=NEW, records=[])
    result_file = args.out / "result.json"
    for length in (4096, 5119, 16384, 17407):
        c = inputs(40, length)
        packed = torch.empty(
            (c["k"].shape[0], 2, 16, 2, 128), device="cuda", dtype=torch.bfloat16
        )
        k, v = packed.unbind(1)
        k.copy_(c["k"])
        v.copy_(c["v"])
        c.update(k=k, v=v)
        original_table = c["table"].clone()
        for live in (0, 1, 8, 32, *range(33, 41)):
            row = dict(
                physical_rows=40, live_rows=live, kv=length, kv_stride=list(k.stride())
            )
            try:
                set_live(c, original_table, live, length)
                full_call(baseline, c, 4)
                gold = c["out"].clone()
                set_live(c, original_table, live, length)
                full_call(candidate, c, 16)
                verify(c, gold, live)
                timings = []
                for iteration in range(3):
                    calls = [
                        ("baseline_ms", baseline, 4),
                        ("candidate_ms", candidate, 16),
                    ]
                    if iteration % 2:
                        calls.reverse()
                    timing = {
                        name: graph_time(
                            partial(full_call, mod, c, width), repeats=9, unroll=4
                        )[0]
                        for name, mod, width in calls
                    }
                    timing["speedup"] = timing["baseline_ms"] / timing["candidate_ms"]
                    timings.append(timing)
                row.update(
                    rounds=timings,
                    median_speedup=statistics.median(t["speedup"] for t in timings),
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    full_call(candidate, c, 16)
                for changed_live in (40, 33, 0, live):
                    c["q"].normal_()
                    set_live(c, original_table, changed_live, length)
                    graph.replay()
                    torch.cuda.synchronize()
                    replayed = c["out"].clone()
                    full_call(baseline, c, 4)
                    gold.copy_(c["out"])
                    c["out"].copy_(replayed)
                    verify(c, gold, changed_live)
                row.update(
                    passed=True, changed_live_graph=True, padded_rows_exact_zero=True
                )
                del graph, replayed, gold
            except Exception as error:
                row.update(passed=False, error=repr(error))
            result["records"].append(row)
            result_file.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
        del c, packed, k, v, original_table
    records = result["records"]
    numeric = len(records) == 48 and all(x.get("passed") for x in records)
    perf = [
        x["median_speedup"] for x in records if x.get("passed") and x["live_rows"] >= 33
    ]
    minimum = min(perf) if perf else 0
    geomean = math.exp(statistics.mean(math.log(x) for x in perf)) if perf else 0
    result["performance_gate"] = dict(
        records=len(perf),
        minimum_speedup=minimum,
        geometric_mean_speedup=geomean,
        minimum_allowed=0.97,
        mean_required=1.03,
        passed=numeric and len(perf) == 32 and minimum >= 0.97 and geomean >= 1.03,
    )
    result_file.write_text(json.dumps(result, indent=2))
    if not numeric or not result["performance_gate"]["passed"]:
        raise RuntimeError(
            "Padded-row numeric/performance gate failed; do not start model"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
