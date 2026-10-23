"""Owned service validation for GPU adaptive low-batch split-K decode."""

import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def get_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def benchmark(run_dir, name, input_len, output_len, count, seed):
    folder = run_dir / name
    folder.mkdir()
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        "minicpm",
        "--tokenizer",
        "/workspace/MiniCPM5-2B",
        "--endpoint",
        "/v1/completions",
        "--host",
        "127.0.0.1",
        "--port",
        "9031",
        "--dataset-name",
        "random",
        "--ignore-eos",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--max-concurrency",
        "64",
        "--num-prompts",
        str(count),
        "--seed",
        str(seed),
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        str(folder),
        "--label",
        name,
    ]
    (folder / "command.json").write_text(json.dumps(command, indent=2))
    with (folder / "benchmark.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400
        )
    result_files = [p for p in folder.glob("*.json") if p.name != "command.json"]
    if len(result_files) != 1:
        raise RuntimeError(f"Expected one result JSON, got {result_files}")
    result = json.loads(result_files[0].read_text())
    if result.get("completed") != count:
        raise RuntimeError(f"Incomplete requests: {result.get('completed')} / {count}")
    keys = [
        "completed",
        "duration",
        "total_input_tokens",
        "total_output_tokens",
        "total_token_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
    ]
    print(json.dumps(dict(name=name, **{k: result.get(k) for k in keys})), flush=True)
    return result


def throughput(run_dir):
    scores = {}
    warmups = {}
    for input_len, count in [(4096, 256), (16384, 128)]:
        history = []
        for index in range(4):
            result = benchmark(
                run_dir, f"warm-{input_len}-{index}", input_len, 128, 64, 920000 + index
            )
            now = result["mean_tpot_ms"]
            history.append(now)
            warmups[str(input_len)] = history
            (run_dir / "warmups.json").write_text(json.dumps(warmups, indent=2))
            if index >= 1 and history[-2] > 0 and abs(now / history[-2] - 1) < 0.03:
                break
        else:
            raise RuntimeError(f"Warmup did not stabilize for {input_len}; no score")
        scores[str(input_len)] = benchmark(
            run_dir, f"score-{input_len}", input_len, 1024, count, 0
        )
        (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    return scores


def accuracy(run_dir):
    folder = run_dir / "math500-level3"
    folder.mkdir(parents=True, exist_ok=False)
    command = [
        "evalscope",
        "eval",
        "--model",
        "minicpm",
        "--api-url",
        "http://127.0.0.1:9031/v1/chat/completions",
        "--api-key",
        "EMPTY",
        "--eval-type",
        "openai_api",
        "--datasets",
        "math_500",
        "--dataset-args",
        '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}',
        "--eval-batch-size",
        "8",
        "--timeout",
        "3600",
        "--generation-config",
        '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}',
        "--work-dir",
        str(folder),
        "--ignore-errors",
    ]
    (folder / "command.json").write_text(json.dumps(command, indent=2))
    with (folder / "math500-level3.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=5400
        )
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
    errors = [
        r["index"]
        for r in rows
        if r["model_output"].get("error") or not r["model_output"].get("choices")
    ]
    verdict = dict(
        score=metric["score"],
        metric_count=metric["num"],
        predictions=len(rows),
        unique_questions=len({r["metadata"]["question_id"] for r in rows}),
        failed_predictions=errors,
        report=str(reports[0]),
    )
    (run_dir / "accuracy-verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(dict(accuracy=verdict)), flush=True)
    if (
        metric["num"] != 105
        or len(rows) != 105
        or verdict["unique_questions"] != 105
        or errors
        or metric["score"] < 0.95
    ):
        raise RuntimeError(f"Accuracy gate failed: {verdict}")


def main(args):
    source = args.source.resolve()
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Port9031 occupied; never stop an unowned service")
    args.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_FL_BIV150_", "BIV150_NATIVE_")) or key in (
            "USE_C_EXTENSION",
            "FLAGTUNE",
        ):
            env.pop(key)
    env.update(
        VLLM_PLUGINS="fl",
        PYTHONPATH=str(source),
        VLLM_DISABLE_COMPILE_CACHE="1",
        TORCHINDUCTOR_CACHE_DIR=str(args.out / "inductor-cache"),
        PYTHONUNBUFFERED="1",
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
    paths = [
        "vllm_fl/patches/biv150_prefill_normalized.py",
        "vllm_fl/worker/model_runner.py",
        "vllm_fl/patches/biv150_gqa_decode.py",
    ]
    manifest = dict(
        command=command,
        source=str(source),
        git_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ),
        source_hashes={
            p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in paths
        },
        mode=args.mode,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version,
        packages={
            name: importlib.metadata.version(name)
            for name in ["torch", "vllm", "flag_gems", "evalscope"]
        },
        env={
            key: value
            for key, value in env.items()
            if key.startswith(("VLLM_", "BIV150_", "TORCHINDUCTOR_"))
        },
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log_path = args.out / "server.log"
    log = log_path.open("w")
    server = subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    (args.out / "server.pid").write_text(str(server.pid))
    (args.out / "driver.pid").write_text(str(os.getpid()))
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("Service exited; inspect server.log")
            try:
                get_json("http://127.0.0.1:9031/v1/models")
                break
            except Exception:
                time.sleep(3)
        else:
            raise TimeoutError("Service not ready")
        marker = "BIV150_PREFILL_NORMALIZATION_INSTALL "
        states = []
        for line in log_path.read_text().splitlines():
            if marker in line:
                states.append(ast.literal_eval(line.split(marker, 1)[1]))
        if not states or states[-1].get("installed") is not True:
            raise RuntimeError(f"Normalization not installed: {states}")
        gqa_marker = "BIV150_GQA_DECODE_INSTALL "
        gqa_states = [
            ast.literal_eval(line.split(gqa_marker, 1)[1])
            for line in log_path.read_text().splitlines()
            if gqa_marker in line
        ]
        if not gqa_states or gqa_states[-1].get("adaptive_split_capacity") != {
            1: 128,
            2: 64,
            3: 32,
            4: 32,
        }:
            raise RuntimeError(f"Adaptive GQA policy missing: {gqa_states}")
        manifest["gqa_installation"] = gqa_states[-1]
        manifest["installation"] = states[-1]
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(
            json.dumps(
                dict(server_ready=True, pid=server.pid, installation=states[-1])
            ),
            flush=True,
        )
        evaluation_offset = log_path.stat().st_size
        if args.mode == "throughput":
            scores = throughput(args.out)
            print(
                json.dumps(dict(completed=True, mode=args.mode, scores=scores)),
                flush=True,
            )
            return
        accuracy(args.out)
        with log_path.open("rb") as events:
            events.seek(evaluation_offset)
            evaluation_log = events.read().decode(errors="replace")
        selected = [
            line
            for line in evaluation_log.splitlines()
            if "BIV150_PREFILL_NORMALIZATION_SELECTED" in line
        ]
        if not selected:
            raise RuntimeError("No real-evaluation normalization selection evidence")
        result = dict(
            completed=True,
            precision_gate=True,
            actual_evaluation_selection=selected,
            accuracy=json.loads((args.out / "accuracy-verdict.json").read_text()),
        )
        if args.mode == "all":
            result["throughput_scores"] = throughput(args.out)
            result["throughput_completed"] = True
        (args.out / "acceptance.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(completed=True, accuracy=result["accuracy"])), flush=True)
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
    parser.add_argument(
        "--mode", choices=["accuracy", "throughput", "all"], default="all"
    )
    main(parser.parse_args())
