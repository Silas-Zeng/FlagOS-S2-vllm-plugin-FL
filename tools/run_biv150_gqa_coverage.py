"""Replay coverage diagnostic; fixed requests, no credited throughput score."""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import run_biv150_gate_team_acceptance as acceptance
from biv150_gqa_coverage import atomic_json


def analyze(trace):
    verdict = dict(diagnostic_only=True, errors=trace["errors"], phases={})
    for phase in ("score-4096", "score-16384"):
        counts = trace["replays"].get(phase, {})
        b40, launches, unknown = 0, 0, []
        for key, count in counts.items():
            entry = trace["entries"].get(key)
            if entry is None:
                unknown.append(key)
                continue
            gqa = entry["gqa_grids"]
            if not gqa:
                unknown.append(key)
            hits = sum(x["calls"] for x in gqa.values() if x["grid"] == [40, 2, 16])
            if hits:
                b40 += count
                launches += count * hits
        eager = trace["eager_gqa"].get(phase, {}).get("[40, 2, 16]", 0)
        verdict["phases"][phase] = dict(
            full_graph_replays=sum(counts.values()),
            b40_full_graph_replays=b40,
            b40_graph_kernel_launches=launches,
            b40_eager_kernel_launches=eager,
            b40_selected=b40 + eager > 0,
            unmapped_entries=unknown,
        )
    return verdict


def main(args):
    source = args.source.resolve()
    assert not subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0
    assert "No running processes found" in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    )
    trace_root = args.out.with_name(args.out.name + "-coverage")
    trace_root.mkdir(parents=True, exist_ok=False)
    generation = 0
    atomic_json(trace_root / "phase.json", dict(name="startup", generation=generation))
    original_popen, original_benchmark = subprocess.Popen, acceptance.benchmark

    class Proxy:
        def __getattr__(self, name):
            return getattr(subprocess, name)

        def Popen(self, command, *positional, **kwargs):
            if command[:2] == ["vllm", "serve"]:
                env = dict(kwargs["env"])
                env["PYTHONPATH"] = os.pathsep.join(
                    str(p)
                    for p in (
                        source / "tools/gqa_coverage_bootstrap",
                        source / "tools",
                        source,
                    )
                )
                env["BIV150_GQA_COVERAGE_DIR"] = str(trace_root)
                kwargs["env"] = env
                atomic_json(
                    trace_root / "injected-environment.json",
                    dict(
                        PYTHONPATH=env["PYTHONPATH"],
                        BIV150_GQA_COVERAGE_DIR=str(trace_root),
                        diagnostic_only=True,
                        service_command=command,
                    ),
                )
            return original_popen(command, *positional, **kwargs)

    def phase(name):
        nonlocal generation
        generation += 1
        value = dict(name=name, generation=generation)
        atomic_json(trace_root / "phase.json", value)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            traces = list(trace_root.glob("trace-*.json"))
            if len(traces) > 1:
                raise RuntimeError("More than one traced engine")
            if len(traces) == 1:
                trace = json.loads(traces[0].read_text())
                ack = trace_root / f"ack-{trace['pid']}.json"
                if (
                    ack.is_file()
                    and json.loads(ack.read_text()).get("generation") == generation
                ):
                    if trace["errors"]:
                        raise RuntimeError(trace["errors"])
                    if not trace["entries"] or not any(
                        e["gqa_grids"] for e in trace["entries"].values()
                    ):
                        raise RuntimeError(
                            "No actual wrapper capture/GQA linkage; refuse benchmark"
                        )
                    return
            time.sleep(0.1)
        raise TimeoutError("Coverage phase not acknowledged; no requests issued")

    def benchmark(run_dir, name, *values):
        phase(name)
        result = original_benchmark(run_dir, name, *values)
        phase(name + "-complete")
        traces = list(trace_root.glob("trace-*.json"))
        trace = json.loads(traces[0].read_text())
        if not trace["replays"].get(name):
            raise RuntimeError(
                "No actual wrapper replay observed; refuse further diagnostic load"
            )
        return result

    acceptance.subprocess, acceptance.benchmark = Proxy(), benchmark
    acceptance.main(argparse.Namespace(source=source, out=args.out, mode="throughput"))
    traces = list(trace_root.glob("trace-*.json"))
    assert len(traces) == 1
    verdict = analyze(json.loads(traces[0].read_text()))
    log = (args.out / "server.log").read_text()
    verdict.update(
        compute_order_selected="BIV150_COMPUTE_ORDER_FIRST_SELECTED" in log,
        scope_errors="BIV150_COMPUTE_ORDER_SCOPE_REJECTED" in log,
        q_fragment_selected="BIV150_Q_FRAGMENT_SELECTED" in log,
        q_fragment_fallback="BIV150_Q_FRAGMENT_FALLBACK" in log,
    )
    atomic_json(trace_root / "verdict.json", verdict)
    print(json.dumps(verdict), flush=True)
    if (
        verdict["errors"]
        or verdict["scope_errors"]
        or verdict["q_fragment_fallback"]
        or not verdict["q_fragment_selected"]
    ):
        raise RuntimeError("Diagnostic scope invalid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
