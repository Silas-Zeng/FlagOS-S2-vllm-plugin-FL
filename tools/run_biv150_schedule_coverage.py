"""Fixed 4K/16K requests with CPU schedule and actual model binding audit."""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import run_biv150_gate_team_acceptance as acceptance
from biv150_schedule_coverage import atomic_json


def analyze(trace):
    result = dict(diagnostic_only=True, errors=trace["errors"], phases={},
                  actual_models=len(trace["models"]))
    for name in ("score-4096", "score-16384"):
        phase = trace["phases"].get(name)
        if not phase or phase["steps"] <= 0:
            raise RuntimeError(f"No actual scheduled queries for {name}")
        result["phases"][name] = dict(
            steps=phase["steps"], request_rows=phase["request_rows"],
            query_tokens=phase["query_tokens"], short_steps=phase["short_steps"],
            short_request_rows=phase["short_request_rows"],
            short_query_tokens=phase["short_query_tokens"],
            short_query_token_fraction=phase["short_query_tokens"] / phase["query_tokens"],
            short_request_row_fraction=phase["short_request_rows"] / phase["request_rows"],
            qlen_histogram=phase["qlen_histogram"],
            short_qlen_kv_histogram=phase["short_qlen_kv_histogram"],
            eager_gqa_grids=trace["eager_gqa"].get(name, {}),
            new_split_eager_kernel_launches=sum(
                trace["eager_gqa"].get(name, {}).get(json.dumps([batch,2,segments]),0)
                for batch,segments in ((29,4),(30,4),(60,2),(61,2),(62,2),(63,2))
            ),
        )
    # Frequency is not a device-time profile or an upper bound on speedup.
    result["cost_profile_available"] = False
    if not trace["models"]:
        raise RuntimeError("No actual model rotary/attention bindings")
    return result


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
                env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (
                    source / "tools/schedule_coverage_bootstrap", source / "tools", source,
                ))
                env["BIV150_SCHEDULE_COVERAGE_DIR"] = str(trace_root)
                kwargs["env"] = env
                atomic_json(trace_root / "injected-environment.json", dict(
                    diagnostic_only=True, service_command=command,
                    PYTHONPATH=env["PYTHONPATH"],
                    BIV150_SCHEDULE_COVERAGE_DIR=str(trace_root),
                ))
            return original_popen(command, *positional, **kwargs)

    def phase(name):
        nonlocal generation
        generation += 1
        atomic_json(trace_root / "phase.json", dict(name=name, generation=generation))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            paths = list(trace_root.glob("trace-*.json"))
            if len(paths) > 1:
                raise RuntimeError("More than one traced engine")
            if paths:
                trace = json.loads(paths[0].read_text())
                ack = trace_root / f"ack-{trace['pid']}.json"
                if ack.is_file() and json.loads(ack.read_text()).get("generation") == generation:
                    if trace["errors"] or not trace["models"]:
                        raise RuntimeError(f"Actual model audit invalid: {trace['errors']}")
                    return trace
            time.sleep(0.1)
        raise TimeoutError("Audit phase not acknowledged; no requests issued")

    def benchmark(run_dir, name, *values):
        phase(name)
        result = original_benchmark(run_dir, name, *values)
        trace = phase(name + "-complete")
        observed = trace["phases"].get(name)
        if not observed or not observed["steps"]:
            raise RuntimeError("Actual schedule alias was not called; stop diagnostic load")
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
    if (verdict["errors"] or verdict["scope_errors"] or verdict["q_fragment_fallback"]
            or not verdict["q_fragment_selected"]):
        raise RuntimeError("Diagnostic scope invalid")
    print(json.dumps(verdict), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
