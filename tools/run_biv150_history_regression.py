"""Owned historical regression replay. Diagnostic scores never imply model qualification."""
import argparse
import ast
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback



def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]), state=fields[0])
    except FileNotFoundError:
        return None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_inventory(root):
    result = {}
    if root.exists():
        for path in root.rglob("*.json"):
            if path.name.startswith("__grp__"):
                continue
            result[str(path.relative_to(root))] = dict(
                path=str(path.relative_to(root)), sha256=sha(path),
                mtime_ns=path.stat().st_mtime_ns)
    return result


def throughput(out, *, full_workload_warmup=False, lengths=None):
    scores, warmups = {}, {}
    for length, count in ((4096, 256), (16384, 128)):
        if lengths and length not in lengths:continue
        warmups[str(length)] = []
        for index in range(3):
            value = benchmark(out, f"warm-{length}-{index}", length, 128, 64, 920000 + index)
            warmups[str(length)].append(value)
            (out / "warmups.json").write_text(json.dumps(warmups, indent=2) + "\n")
        if full_workload_warmup:
            # Short 64-request warmups do not visit the late mixed/tail shapes
            # of the official multi-wave workload. Keep this unscored run in
            # the evidence rather than treating cache-cold stalls as a gain.
            value = benchmark(out, f"full-warm-{length}", length, 1024, count, 0)
            warmups[str(length)].append(dict(value, full_workload=True))
            (out / "warmups.json").write_text(json.dumps(warmups, indent=2) + "\n")
        before = cache_inventory(out / "triton-cache")
        scores[str(length)] = benchmark(out, f"score-{length}", length, 1024, count, 0)
        after = cache_inventory(out / "triton-cache")
        additions = [after[key] for key in sorted(after.keys() - before.keys())]
        (out / f"score-{length}-new-kernels.json").write_text(json.dumps(additions, indent=2) + "\n")
        (out / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    return scores


def main(args):
    global benchmark
    source = args.source.resolve()
    # Use one fixed benchmark/fixture helper for every replay source.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_biv150_gate_team_acceptance import get_json, accuracy, benchmark, performance_gates
    from biv150_peer_mixed_diagnostic import run as mixed_diagnostic
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == args.commit
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", 9031)) != 0, "Port 9031 is occupied"
    idle = subprocess.check_output(["/usr/local/corex/bin/ixsmi"], text=True)
    assert "No running processes found" in idle, "GPU is busy"
    assert sha(args.gate_library) == args.library_sha256
    assert not args.diagnostic_after_semantic_failure or (args.run_throughput and not args.run_accuracy)
    if args.legacy_no_small_o:
        assert args.stable_small_o=='0'
        assert not (source/'vllm_fl/patches/biv150_stable_small_o.py').exists()

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "idle.txt").write_text(idle)
    snapshot = args.tuning_snapshot.resolve()
    records = json.loads((snapshot / "manifest.json").read_text())["databases"]
    target = args.out / "flaggems-cache/config_cache"
    target.mkdir(parents=True)
    for record in records:
        name = record["name"]
        assert Path(name).name == name and name.endswith(".db")
        assert sha(snapshot / name) == record["sha256"]
        shutil.copy2(snapshot / name, target / name)
        assert sha(target / name) == record["sha256"]
    state = dict(complete=False, cleanup_complete=False, acceptance_passed=False,
                 precision_passed=None, diagnostic_only=True, phase="starting",
                 commit=args.commit, native_mode=args.native_mode)

    def save(phase, **values):
        state.update(phase=phase, updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
        temporary = args.out / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.out / "state.json")

    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_FL_BIV150_", "BIV150_NATIVE_", "BIV150_PROFILE_")) or key in ("FLAGTUNE", "USE_C_EXTENSION"):
            env.pop(key)
    env.update(VLLM_PLUGINS="fl", PYTHONPATH=str(source), PYTHONUNBUFFERED="1",
               VLLM_DISABLE_COMPILE_CACHE="1", TORCHINDUCTOR_CACHE_DIR=str(args.out / "inductor-cache"),
               TRITON_CACHE_DIR=str(args.out / "triton-cache"),
               FLAGGEMS_CACHE_DIR=str(args.out / "flaggems-cache"),
               BIV150_NATIVE_W8A8_LIBRARY=str(args.gate_library.resolve()),
               VLLM_FL_BIV150_NATIVE_W8A8="1" if args.native_mode == "on" else "0",
               VLLM_FL_BIV150_STABLE_SMALL_O=args.stable_small_o)
    command = ["/usr/local/bin/vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
               "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
               "--served-model-name", "minicpm", "--gpu-memory-utilization", "0.85", "--max-model-len", "131072"]
    manifest = dict(command=command, source=str(source), commit=args.commit,
                    native_mode=args.native_mode, stable_small_o=args.stable_small_o,
                    legacy_no_small_o=args.legacy_no_small_o,selected_lengths=args.length,
                    diagnostic_after_semantic_failure=args.diagnostic_after_semantic_failure,
                    benchmark_helper_sha256=sha(Path(__file__).with_name("run_biv150_gate_team_acceptance.py")), run_accuracy=args.run_accuracy, run_throughput=args.run_throughput, full_workload_warmup=args.full_workload_warmup, driver_sha256=sha(__file__),
                    diagnostic_sha256=sha(Path(__file__).with_name("biv150_peer_mixed_diagnostic.py")),
                    gate_library_sha256=sha(args.gate_library), tuning_snapshot=records,
                    env={k: v for k, v in env.items() if k.startswith(("VLLM_", "BIV150_", "TRITON_", "TORCHINDUCTOR_", "FLAGGEMS_"))})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = args.out / "server.log"
    log = log_path.open("x")
    server = None
    owned = {}
    launcher = None

    def remember():
        if server is None or launcher is None:
            return
        anchors = [launcher, *owned.values()]
        if not any((current := identity(member["pid"])) and
                   current["start_ticks"] == member["start_ticks"] and
                   current["sid"] == launcher["sid"] for member in anchors):
            return
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                member = identity(int(path.name))
                if member and member["sid"] == server.pid:
                    owned.setdefault(member["pid"], member)

    try:
        server = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launcher = identity(server.pid)
        assert launcher and launcher["sid"] == server.pid
        owned[server.pid] = launcher
        save("waiting_service", owner=launcher, driver=identity(os.getpid()))
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            remember()
            if server.poll() is not None:
                raise RuntimeError("Owned service exited; inspect server.log")
            try:
                get_json("http://127.0.0.1:9031/v1/models")
                break
            except Exception:
                time.sleep(3)
        else:
            raise TimeoutError("Service startup timeout")
        logs = log_path.read_text()
        installations = {}
        for marker in ("BIV150_NATIVE_W8A8_INSTALL",
                       "BIV150_Q_FRAGMENT_INSTALL", "BIV150_PREFILL_NORMALIZATION_INSTALL",
                       "BIV150_GQA_DECODE_INSTALL", "BIV150_MIXED_DECODE_INSTALLED", "BIV150_STABLE_SMALL_O_INSTALL"):
            if args.legacy_no_small_o and marker=="BIV150_STABLE_SMALL_O_INSTALL":continue
            values = [ast.literal_eval(line.split(marker + " ", 1)[1]) for line in logs.splitlines() if marker + " " in line]
            assert values, f"Missing installation provenance: {marker}"
            installations[marker] = values[-1]
            expected = ((args.native_mode == "on") if marker == "BIV150_NATIVE_W8A8_INSTALL"
                        else (args.stable_small_o == "1") if marker == "BIV150_STABLE_SMALL_O_INSTALL" else True)
            assert values[-1].get("installed") is expected, values[-1]
        manifest["installations"] = installations
        manifest["kv_cache_lines"] = [line for line in logs.splitlines() if "GPU KV cache size" in line]
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        save("ready")
        save("mixed_diagnostic")
        summary = mixed_diagnostic(args.out, args.fixtures)
        selected = [ast.literal_eval(line.split("BIV150_STABLE_SMALL_O_SELECTED ",1)[1])
                    for line in log_path.read_text().splitlines() if "BIV150_STABLE_SMALL_O_SELECTED " in line]
        assert bool(selected) is (args.stable_small_o == "1")
        if args.stable_small_o == "1":assert all(any(value["rows"]==2 and value.get("output_columns")==n and value.get("input_columns")==k for value in selected) for n,k in ((2048,2048),(12288,2048),(2560,2048),(2048,6144))), "M2 QKV/O/down/edge-gate actual selection absent"
        manifest["selected_small_o"] = selected
        (args.out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        passed = summary["all_correct"] and summary["http_overlap"]
        save("mixed_completed", diagnostic=summary, semantic_passed=passed)
        if args.run_accuracy and passed:
            save("accuracy")
            try:
                accuracy(args.out)
            except RuntimeError:
                path=args.out/"accuracy-verdict.json"
                if not path.exists():raise
                verdict=json.loads(path.read_text())
                if not (verdict["metric_count"]==verdict["predictions"]==verdict["unique_questions"]==105
                        and not verdict["failed_predictions"] and verdict["score"]>=.94):raise
            verdict=json.loads((args.out/"accuracy-verdict.json").read_text())
            state.update(accuracy=verdict,precision_passed=verdict["score"]>=.95,official_accuracy_evaluated=True)
        if args.run_throughput:
            assert passed or args.diagnostic_after_semantic_failure, "Mixed semantic gate must pass before qualified scoring"
            save("throughput")
            scores=throughput(args.out, full_workload_warmup=args.full_workload_warmup,lengths=args.length)
            for length,count in ((4096,256),(16384,128)):
                if str(length) not in scores:continue
                value=scores[str(length)]
                expected=dict(completed=count,failed=0,num_prompts=count,max_concurrency=64,total_input_tokens=length*count,total_output_tokens=1024*count)
                assert all(value.get(key)==val for key,val in expected.items()), expected
            state.update(scores=scores,working_gates=performance_gates(scores) if set(scores)=={"4096","16384"} else None,matched_validated_control_required=True)
        save("workloads_completed", work_completed=True, diagnostic=summary,
             semantic_passed=passed, official_throughput_claim=False,
             diagnostic_scores_only=True,qualifies_historical_baseline=False,
             accuracy_skipped_due_mixed_failure=bool(args.run_accuracy and not passed))
    except BaseException as error:
        save("failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        remember()
        if server is not None:
            for member in list(owned.values()):
                current = identity(member["pid"])
                if current and current["start_ticks"] == member["start_ticks"] and current["sid"] == server.pid and current["state"] != "Z":
                    try:
                        os.kill(member["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                remember()
                if not any((identity(pid) or {}).get("state") not in (None, "Z") for pid in owned):
                    break
                time.sleep(1)
            for member in owned.values():
                current = identity(member["pid"])
                if current and current["start_ticks"] == member["start_ticks"] and current["sid"] == server.pid and current["state"] != "Z":
                    try:
                        os.kill(member["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            deadline = time.monotonic() + 10
            while True:
                living = [current for member in owned.values()
                          if (current := identity(member["pid"])) and
                          current["start_ticks"] == member["start_ticks"] and
                          current["state"] != "Z"]
                if not living or time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
            with socket.socket() as sock:
                port_closed = sock.connect_ex(("127.0.0.1", 9031)) != 0
            cleanup_ok = not living and port_closed
            state.update(owned_service_members=list(owned.values()), cleanup_complete=cleanup_ok,
                         cleanup_remaining=living, port_9031_closed=port_closed,
                         complete=bool(state.get("work_completed") and cleanup_ok))
            if state["complete"]:
                save("completed")
            elif state.get("work_completed"):
                save("failed", cleanup_error="Owned process or port still active; no further GPU task allowed")
            else:
                save(state["phase"])
        log.close()
    if not state.get("cleanup_complete"):
        raise RuntimeError("Owned service cleanup did not complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length",type=int,choices=(4096,16384),action="append")
    parser.add_argument("--legacy-no-small-o",action="store_true")
    parser.add_argument("--diagnostic-after-semantic-failure",action="store_true")
    parser.add_argument("--library-sha256",choices=("5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe","eacac719d61705d1adb94cee8666bebe55f3411fc035e1e536f14847b5fe79b9"),default="5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--gate-library", type=Path, required=True)
    parser.add_argument("--native-mode", choices=("on", "off"), required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--stable-small-o", choices=("0","1"), required=True)
    parser.add_argument("--run-accuracy", action="store_true")
    parser.add_argument("--run-throughput", action="store_true")
    parser.add_argument("--full-workload-warmup", action="store_true")
    parser.add_argument("--tuning-snapshot", type=Path, required=True)
    main(parser.parse_args())
