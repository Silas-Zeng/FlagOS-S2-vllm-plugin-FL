"""CPU execution-shape diagnostics at a KV capacity boundary; never a score.

Requires an isolated historical source worktree with the opt-in Recorder hook.
Official serve command and scheduler defaults are preserved. Request workloads
are diagnostic-only: 32 requests at 16000/16384 and 33 at 15360/16384; 256 output tokens.
"""

import argparse
import hashlib
import json
import os
import random
import signal
import socket
import subprocess
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def snapshot(folder):
    records = {}
    for path in folder.glob("shape-*.json"):
        record = json.loads(path.read_text())
        if record["calls_observed"]:
            assert not record["trace_disabled"] and not record["trace_errors"], record
            records[path.name] = record
    assert len(records) == 1, f"Expected one traced engine: {list(records)}"
    return records


def difference(before, after):
    keys = (
        "scheduled_tokens",
        "requests",
        "max_tokens_per_request",
        "graph_mode",
        "padded_tokens",
        "padded_requests",
    )

    def counts(data):
        result = Counter()
        for record in data.values():
            for row in record["counts"]:
                result[tuple(row[k] for k in keys)] += row["calls"]
        return result

    delta = counts(after)
    delta.subtract(counts(before))
    assert all(n >= 0 for n in delta.values()), "Trace counters reset"
    return [dict(zip(keys, key), calls=n) for key, n in delta.items() if n]


def requests(folder, length, output, count, seed):
    rng = random.Random(seed)
    prompts = [[rng.randrange(100, 10000) for _ in range(length)] for _ in range(count)]

    def one(index):
        payload = dict(
            model="minicpm",
            prompt=prompts[index],
            max_tokens=output,
            ignore_eos=True,
            temperature=1.0,
            top_p=0.95,
            stream=False,
        )
        request = urllib.request.Request(
            "http://127.0.0.1:9031/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=1200) as response:
            result = json.load(response)
        usage = result["usage"]
        assert (
            usage["prompt_tokens"] == length and usage["completion_tokens"] == output
        ), usage
        return dict(index=index, seconds=time.monotonic() - start, usage=usage)

    with ThreadPoolExecutor(max_workers=count) as executor:
        rows = list(executor.map(one, range(count)))
    folder.write_text(
        json.dumps(
            dict(
                diagnostic_only=True,
                input_tokens=length,
                output_tokens=output,
                requests=count,
                seed=seed,
                results=rows,
            ),
            indent=2,
        )
    )


def main(args):
    source = args.source.resolve()
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0, "Service already active"
    assert not subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"]
    ).strip()
    worker = source / "vllm_fl/worker/model_runner.py"
    assert "BIV150_EXEC_SHAPE_TRACE_INSTALL" in worker.read_text(), (
        "Missing isolated trace hook"
    )
    args.out.mkdir(parents=True, exist_ok=False)
    trace = args.out / "shape-traces"
    trace.mkdir()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(
            ("VLLM_FL_BIV150_", "BIV150_NATIVE_", "BIV150_SHAPE_")
        ) or key in ("USE_C_EXTENSION", "FLAGTUNE"):
            env.pop(key)
    env.update(
        VLLM_PLUGINS="fl",
        PYTHONPATH=os.pathsep.join([str(source), str(source / "tools")]),
        VLLM_DISABLE_COMPILE_CACHE="1",
        TORCHINDUCTOR_CACHE_DIR=str(args.out / "inductor-cache"),
        PYTHONUNBUFFERED="1",
        BIV150_SHAPE_TRACE_DIR=str(trace),
        VLLM_FL_BIV150_NATIVE_W8A8="1" if args.native else "0",
    )
    if args.native:
        assert args.library and args.library.is_file()
        env.update(
            BIV150_NATIVE_W8A8_LIBRARY=str(args.library.resolve()),
            BIV150_NATIVE_W8A8_LAYERS="qkv_proj,o_proj,gate_up_proj",
            BIV150_NATIVE_W8A8_MIN_ROWS="17",
            BIV150_NATIVE_W8A8_COMPRESSED_ONLY="1",
            BIV150_NATIVE_W8A8_SKIP_EDGE_LAYERS="4",
            BIV150_NATIVE_W8A8_FUSE_SILU="1",
        )
    command = [
        "vllm",
        "serve",
        "/workspace/MiniCPM5-2B",
        "--port",
        "9031",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--served-model-name",
        "minicpm",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "131072",
    ]
    manifest = dict(
        diagnostic_only=True,
        not_a_performance_score=True,
        command=command,
        native=args.native,
        source=str(source),
        git_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(
            (source / "tools/biv150_execution_shape_trace.py").read_bytes()
        ).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest()
        if args.native
        else None,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log = (args.out / "server.log").open("w")
    server = subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )
    (args.out / "server.pid").write_text(str(server.pid))
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("Diagnostic service exited")
            try:
                get_json("http://127.0.0.1:9031/v1/models")
                break
            except Exception:
                time.sleep(3)
        else:
            raise TimeoutError("Service not ready")
        requests(args.out / "warm-requests.json", 4096, 32, 8, 202610230)
        summary = {}
        for count, length in [(32, 16000), (32, 16384), (33, 15360), (33, 16384)]:
            before = snapshot(trace)
            requests(
                args.out / f"requests-{count}x{length}.json",
                length,
                256,
                count,
                20261023 + length + count * 100000,
            )
            after = snapshot(trace)
            rows = difference(before, after)
            assert rows, "No actual execution shapes captured"
            summary[f"{count}x{length}"] = dict(before=before, after=after, delta=rows)
            (args.out / "diagnostic-shapes.json").write_text(
                json.dumps(summary, indent=2)
            )
            print(
                json.dumps(
                    dict(
                        diagnostic_completed=f"{count}x{length}",
                        trace_calls=sum(r["calls"] for r in rows),
                    )
                ),
                flush=True,
            )
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--library", type=Path)
    main(parser.parse_args())
