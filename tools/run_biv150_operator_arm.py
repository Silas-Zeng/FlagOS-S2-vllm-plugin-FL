"""Isolated service arm: equal warmup, one scored run per case, owned cleanup."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import hashlib


def get_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def benchmark(run_dir, name, input_len, output_len, count, seed):
    folder = run_dir / name
    folder.mkdir()
    command = ["vllm", "bench", "serve", "--backend", "vllm", "--model", "minicpm",
               "--tokenizer", "/workspace/MiniCPM5-2B", "--endpoint", "/v1/completions",
               "--host", "127.0.0.1", "--port", "9031", "--dataset-name", "random",
               "--ignore-eos", "--random-input-len", str(input_len),
               "--random-output-len", str(output_len), "--max-concurrency", "64",
               "--num-prompts", str(count), "--seed", str(seed), "--disable-tqdm",
               "--save-result", "--result-dir", str(folder), "--label", name]
    (folder / "command.json").write_text(json.dumps(command, indent=2))
    with (folder / "benchmark.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400)
    result_files = [p for p in folder.glob("*.json") if p.name != "command.json"]
    if len(result_files) != 1:
        raise RuntimeError(f"Expected one result JSON, got {result_files}")
    result = json.loads(result_files[0].read_text())
    if result.get("completed") != count:
        raise RuntimeError(f"Incomplete requests: {result.get('completed')} / {count}")
    keys = ["completed", "duration", "total_input_tokens", "total_output_tokens",
            "total_token_throughput", "output_throughput", "mean_ttft_ms", "mean_tpot_ms"]
    print(json.dumps(dict(name=name, **{k: result.get(k) for k in keys})), flush=True)
    return result


def accuracy(run_dir):
    folder = run_dir / "math500-level3"
    subprocess.run(["bash", "/workspace/experiments/minicpm-s2/scripts/run_math500_level3.sh",
                    str(folder)], check=True, timeout=5400)
    reports = list(folder.glob("*/reports/minicpm/math_500.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one accuracy report, got {reports}")
    report = json.loads(reports[0].read_text())
    metrics = [m for m in report["metrics"] if m["identity"]["name"] == "accuracy"]
    if len(metrics) != 1:
        raise RuntimeError("Accuracy metric missing or ambiguous")
    metric = metrics[0]
    predictions = reports[0].parents[2] / "predictions/minicpm/math_500_Level 3.jsonl"
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    errors = [r["index"] for r in rows if r["model_output"].get("error")
              or not r["model_output"].get("choices")]
    verdict = dict(score=metric["score"], metric_count=metric["num"],
                   predictions=len(rows), unique_questions=len({r["metadata"]["question_id"] for r in rows}),
                   failed_predictions=errors, report=str(reports[0]))
    (run_dir / "accuracy-verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(dict(accuracy=verdict)), flush=True)
    if metric["num"] != 105 or len(rows) != 105 or verdict["unique_questions"] != 105 or errors or metric["score"] < 0.95:
        raise RuntimeError(f"Accuracy gate failed: {verdict}")


def main(args):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Port 9031 already in use; never stop an unowned server")
    args.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("VLLM_FL_BIV150_") or key in ("USE_C_EXTENSION", "FLAGTUNE"):
            env.pop(key)
    env.update(VLLM_PLUGINS="fl", PYTHONPATH=str(args.source),
               VLLM_DISABLE_COMPILE_CACHE="1", PYTHONUNBUFFERED="1")
    env["VLLM_FL_BIV150_NATIVE_W8A8"] = "1" if args.native else "0"
    env["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library)
    env["BIV150_NATIVE_W8A8_MIN_ROWS"] = str(args.min_rows)
    env["BIV150_NATIVE_W8A8_LAYERS"] = args.layers
    env["BIV150_NATIVE_W8A8_COMPRESSED_ONLY"] = "1" if args.compressed_only else "0"
    env["BIV150_NATIVE_W8A8_SKIP_EDGE_LAYERS"] = str(args.skip_edge_layers)
    env["BIV150_NATIVE_W8A8_FUSE_SILU"] = "1" if args.fuse_silu else "0"
    if args.shipping_default:
        for key in list(env):
            if key.startswith("BIV150_NATIVE_") or key == "VLLM_FL_BIV150_NATIVE_W8A8":
                env.pop(key)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(args.out / "inductor-cache")
    command = ["vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
               "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
               "--served-model-name", "minicpm", "--gpu-memory-utilization", "0.85",
               "--max-model-len", "131072"]
    manifest = dict(command=command, source=str(args.source), native=args.native,
                    git_commit=subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
                    git_status=subprocess.check_output(["git", "-C", str(args.source), "status", "--short"], text=True),
                    source_hashes={str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [args.source / "vllm_fl/patches/biv150_native_w8a8.py",
                                             args.source / "vllm_fl/patches/biv150_gqa_decode.py",
                                             args.source / "vllm_fl/worker/model_runner.py"]},
                    library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                    mode=args.mode, cases=args.cases, shipping_default=args.shipping_default,
                    min_rows=None if args.shipping_default else args.min_rows,
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    env={k: env[k] for k in env if k.startswith(("VLLM_", "BIV150_", "TORCHINDUCTOR_"))})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log = (args.out / "server.log").open("w")
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, start_new_session=True)
    (args.out / "server.pid").write_text(str(server.pid))
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited {server.returncode}; inspect server.log")
            try:
                get_json("http://127.0.0.1:9031/v1/models")
                break
            except Exception:
                time.sleep(3)
        else:
            raise TimeoutError("Server not ready")
        print(json.dumps(dict(server_ready=True, pid=server.pid, native=args.native)), flush=True)
        if args.accuracy_first:
            accuracy(args.out)
        scores = {}
        for input_len, prompts in [(4096, 256), (16384, 128)]:
            if str(input_len) not in args.cases:
                continue
            prior = None
            stabilized = False
            for index in range(4):
                warm = benchmark(args.out, f"warm-{input_len}-{index}", input_len, 128, 64,
                                 920000 + index)
                now = warm.get("mean_tpot_ms")
                if index >= 1 and prior and now and abs(now / prior - 1) < 0.03:
                    stabilized = True
                    break
                prior = now
            if not stabilized:
                raise RuntimeError(f"Warmup did not stabilize for {input_len}; no scored run")
            score_count = prompts if args.mode == "full" else (64 if input_len == 4096 else 16)
            scores[str(input_len)] = benchmark(args.out, f"score-{input_len}", input_len,
                                                1024, score_count, 0)
            (args.out / "scores.json").write_text(json.dumps(scores, indent=2))
        print(json.dumps(dict(completed=True, mode=args.mode)), flush=True)
        if args.accuracy and not args.accuracy_first:
            accuracy(args.out)
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
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--min-rows", type=int, default=1024)
    parser.add_argument("--mode", choices=("probe", "full"), default="probe")
    parser.add_argument("--layers", default="gate_up_proj")
    parser.add_argument("--compressed-only", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--accuracy-first", action="store_true")
    parser.add_argument("--skip-edge-layers", type=int, default=0)
    parser.add_argument("--fuse-silu", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=("4096", "16384"), default=["4096", "16384"])
    parser.add_argument("--shipping-default", action="store_true")
    main(parser.parse_args())
