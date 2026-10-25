"""Owned acceptance for residual gate on the PR22+PR23 team attention stack."""

import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import json
import math
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


def performance_gates(scores):
    """Working research limits; official scoring interpretation remains separate."""
    limits = {
        "4096": (4809.36, 4770.3209, 256),
        "16384": (3364.03, 156938.85, 128),
    }
    verdicts = {}
    for length, (baseline, ttft_limit, count) in limits.items():
        score = scores.get(length, {})
        tps = score.get("total_token_throughput", float("nan"))
        ttft = score.get("mean_ttft_ms", float("nan"))
        requests_ok = (
            score.get("num_prompts") == count
            and score.get("max_concurrency") == 64
            and score.get("completed") == count
            and score.get("failed") == 0
            and score.get("total_input_tokens") == int(length) * count
            and score.get("total_output_tokens") == 1024 * count
        )
        throughput_ok = math.isfinite(tps) and tps >= baseline * 0.99
        ttft_ok = math.isfinite(ttft) and 0 < ttft <= ttft_limit
        verdicts[length] = dict(
            requests_ok=requests_ok,
            throughput_ok=throughput_ok,
            ttft_ok=ttft_ok,
            passed=requests_ok and throughput_ok and ttft_ok,
            baseline_total_tokens_per_s=baseline,
            throughput_floor=baseline * 0.99,
            working_mean_ttft_limit_ms=ttft_limit,
            goal_20_percent=math.isfinite(tps) and tps >= baseline * 1.20,
        )
    return verdicts


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
                run_dir, f"warm-{input_len}-{index}", input_len, 128,
                128 if input_len==4096 else 64, 920000 + index
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
        or metric["score"] < 0.9428
    ):
        raise RuntimeError(f"Accuracy gate failed: {verdict}")


def main(args):
    if args.live_carry_control and args.mode!='throughput':
        raise ValueError('Disabled-operator control is throughput-only; no accuracy credit')
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
        VLLM_FL_BIV150_LIVE_CARRY_GATE='0' if args.live_carry_control else '1',
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
        "vllm_fl/patches/biv150_native_w8a8.py",
        "vllm_fl/patches/biv150_live_carry_gate.py",
        "vllm_fl/patches/biv150_live_carry_codegen.py",
        "vllm_fl/compile_semantics.py",
        "vllm_fl/patches/biv150_compute_order.py",
        "vllm_fl/patches/biv150_q_fragment_service.py",
        "csrc/iluvatar/native_w8a8.cpp",
        "csrc/iluvatar/native_w8a8.cu",
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
        live_carry_control=args.live_carry_control,
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
    manifest["native_library_sha256"] = hashlib.sha256(
        (source / "build-native-w8a8/biv150_native_w8a8.so").read_bytes()
    ).hexdigest()
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
        native_marker = "BIV150_NATIVE_W8A8_INSTALL "
        native_states = [
            ast.literal_eval(line.split(native_marker, 1)[1])
            for line in log_path.read_text().splitlines()
            if native_marker in line
        ]
        if (
            not native_states
            or native_states[-1].get("installed") is not True
            or tuple(native_states[-1].get("layers", ())) != ("gate_up_proj",)
            or native_states[-1].get("gate_operator") != "gate_silu_auto"
            or native_states[-1].get("activation_residual_rows") != [1, 16]
            or native_states[-1].get("cached_quantization_rows") != [17, 2048]
            or native_states[-1].get("cached_quantization_threads") != 256
            or native_states[-1].get("compressed_only") is not True
            or native_states[-1].get("skip_edge_layers") != 4
        ):
            raise RuntimeError(f"Gate-only native operator missing: {native_states}")
        fragment_marker = "BIV150_Q_FRAGMENT_INSTALL "
        fragments = [
            ast.literal_eval(line.split(fragment_marker, 1)[1])
            for line in log_path.read_text().splitlines()
            if fragment_marker in line
        ]
        if not fragments or fragments[-1].get("installed") is not True:
            raise RuntimeError(f"Team Q fragment missing: {fragments}")
        live=native_states[-1].get('live_carry_m64',{})
        if args.live_carry_control:
            if live.get('installed') is not False:
                raise RuntimeError(f'Control unexpectedly installed candidate: {live}')
        elif (live.get('installed') is not True or live.get('exact_rows')!=[64]
              or live.get('original_transparent_carry') is not True
              or live.get('persistent_weight_copy') is not False
              or live.get('original_decoder_forward_unchanged') is not True
              or Path(live.get('hook_file','')).resolve()!=source/'vllm_fl/patches/biv150_live_carry_codegen.py'):
            raise RuntimeError(f'Live-carry candidate missing or wrong source: {live}')
        startup_hits=[line for line in log_path.read_text().splitlines() if 'BIV150_LIVE_CARRY_M64_SELECTED' in line]
        generated_pairs=[line for line in log_path.read_text().splitlines() if 'BIV150_LIVE_CARRY_CODEGEN_PAIR' in line]
        if not args.live_carry_control and not startup_hits:
            raise RuntimeError('No M64 selection during startup Graph capture; do not score an inactive candidate')
        manifest['startup_capture_selection']=startup_hits
        manifest['generated_pairs']=generated_pairs
        if not args.live_carry_control and not generated_pairs:
            raise RuntimeError('No audited generated norm/gate pair transformed')
        manifest["q_fragment_installation"] = fragments[-1]
        manifest["native_installation"] = native_states[-1]
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
            hits=[line for line in log_path.read_text().splitlines() if 'BIV150_LIVE_CARRY_M64_SELECTED' in line]
            if bool(hits)==args.live_carry_control:
                raise RuntimeError('Candidate selection does not match requested control mode')
            (args.out/'throughput-verdict.json').write_text(json.dumps(dict(
                completed=True,accuracy_evaluated=False,precision_gate=None,
                live_carry_control=args.live_carry_control,
                working_performance_gates=performance_gates(scores),live_carry_selection=hits),indent=2))
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
            precision_gate=json.loads((args.out/'accuracy-verdict.json').read_text())['score']>=.95,
            actual_evaluation_selection=selected,
            accuracy=json.loads((args.out / "accuracy-verdict.json").read_text()),
        )
        if args.mode == "all":
            result["throughput_scores"] = throughput(args.out)
            result["throughput_completed"] = True
            result["working_performance_gates"] = performance_gates(
                result["throughput_scores"]
            )
            result["working_acceptance_passed"] = result['precision_gate'] and all(
                row["passed"] for row in result["working_performance_gates"].values()
            )
            with log_path.open("rb") as events:
                events.seek(evaluation_offset)
                serving_log = events.read().decode(errors="replace")
            order_hits = [
                line
                for line in serving_log.splitlines()
                if "BIV150_COMPUTE_ORDER_FIRST_SELECTED" in line
            ]
            result["compute_order_serving_selection"] = order_hits
            result["compute_order_scope_errors"] = [
                line
                for line in serving_log.splitlines()
                if "BIV150_COMPUTE_ORDER_SCOPE_REJECTED" in line
            ]
            complete_log = log_path.read_text()
            result["q_fragment_selection"] = [
                line
                for line in complete_log.splitlines()
                if "BIV150_Q_FRAGMENT_SELECTED" in line
            ]
            result["q_fragment_fallback"] = [
                line
                for line in complete_log.splitlines()
                if "BIV150_Q_FRAGMENT_FALLBACK" in line
            ]
            if not result["q_fragment_selection"] or result["q_fragment_fallback"]:
                result["working_acceptance_passed"] = False
            if not order_hits or result["compute_order_scope_errors"]:
                result["working_acceptance_passed"] = False
        result['live_carry_selection']=[line for line in log_path.read_text().splitlines() if 'BIV150_LIVE_CARRY_M64_SELECTED' in line]
        if not result['live_carry_selection'] and args.mode=='all':
            result['working_acceptance_passed']=False
        (args.out / "acceptance.json").write_text(json.dumps(result, indent=2))
        if args.mode == "all" and not result["working_acceptance_passed"]:
            raise RuntimeError("Working throughput/TTFT/request gates failed; no merge")
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
    parser.add_argument('--live-carry-control',action='store_true')
    parser.add_argument(
        "--mode", choices=["accuracy", "throughput", "all"], default="all"
    )
    main(parser.parse_args())
