"""Owned Q32 candidate/control acceptance with the fixed competition workload."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

from run_biv150_gate_team_acceptance import accuracy, get_json, benchmark, performance_gates


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]), state=fields[0])
    except FileNotFoundError:
        return None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def throughput(out):
    scores, warmups = {}, {}
    for length, count in ((4096, 256), (16384, 128)):
        warmups[str(length)] = []
        for index in range(3):
            value = benchmark(out, f"warm-{length}-{index}", length, 128, 64, 920000 + index)
            warmups[str(length)].append(value)
            (out / "warmups.json").write_text(json.dumps(warmups, indent=2) + "\n")
        scores[str(length)] = benchmark(out, f"score-{length}", length, 1024, count, 0)
        (out / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    return scores


def main(args):
    source = args.source.resolve()
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == args.commit
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", 9031)) != 0, "Port 9031 is occupied"
    idle = subprocess.check_output(["/usr/local/corex/bin/ixsmi"], text=True)
    assert "No running processes found" in idle, "GPU is busy"
    assert sha(args.gate_library) == "5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe"
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "idle.txt").write_text(idle)
    state = dict(complete=False, cleanup_complete=False, acceptance_passed=False,
                 precision_passed=None, diagnostic_only=False, phase="starting",
                 commit=args.commit, candidate=args.candidate)

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
               BIV150_NATIVE_W8A8_LIBRARY=str(args.gate_library.resolve()),
               VLLM_FL_BIV150_PREFILL_Q32="1" if args.candidate else "0")
    command = ["/usr/local/bin/vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
               "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
               "--served-model-name", "minicpm", "--gpu-memory-utilization", "0.85", "--max-model-len", "131072"]
    manifest = dict(command=command, source=str(source), commit=args.commit,
                    candidate=args.candidate, mode=args.mode, driver_sha256=sha(__file__),
                    gate_library_sha256=sha(args.gate_library),
                    env={k: v for k, v in env.items() if k.startswith(("VLLM_", "BIV150_", "TRITON_", "TORCHINDUCTOR_"))})
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
        for marker in ("BIV150_NATIVE_W8A8_INSTALL", "BIV150_PREFILL_Q32_INSTALL",
                       "BIV150_Q_FRAGMENT_INSTALL", "BIV150_PREFILL_NORMALIZATION_INSTALL",
                       "BIV150_GQA_DECODE_INSTALL", "BIV150_MIXED_DECODE_INSTALLED"):
            values = [ast.literal_eval(line.split(marker + " ", 1)[1]) for line in logs.splitlines() if marker + " " in line]
            assert values, f"Missing installation provenance: {marker}"
            installations[marker] = values[-1]
            expected = args.candidate if marker == "BIV150_PREFILL_Q32_INSTALL" else True
            assert values[-1].get("installed") is expected, values[-1]
        manifest["installations"] = installations
        manifest["kv_cache_lines"] = [line for line in logs.splitlines() if "GPU KV cache size" in line]
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        save("ready")
        if args.mode == "all":
            save("accuracy")
            try:
                accuracy(args.out)
            except RuntimeError:
                path = args.out / "accuracy-verdict.json"
                if not path.exists():
                    raise
                verdict = json.loads(path.read_text())
                # User explicitly permits throughput for the near-miss 99/105 case.
                if not (verdict["metric_count"] == verdict["predictions"] == verdict["unique_questions"] == 105
                        and not verdict["failed_predictions"] and verdict["score"] >= 0.94):
                    raise
                save("accuracy_near_miss", accuracy=verdict, precision_passed=False, diagnostic_only=True)
            verdict = json.loads((args.out / "accuracy-verdict.json").read_text())
            state.update(accuracy=verdict, precision_passed=verdict["score"] >= 0.95)
        save("throughput")
        scores = throughput(args.out)
        for length, count in ((4096, 256), (16384, 128)):
            result = scores[str(length)]
            expected = dict(completed=count, failed=0, num_prompts=count, max_concurrency=64,
                            total_input_tokens=length * count, total_output_tokens=1024 * count)
            assert all(result.get(key) == value for key, value in expected.items()), expected
        final_logs = log_path.read_text()
        selected_lines = [line for line in final_logs.splitlines() if "BIV150_PREFILL_Q32_SELECTED " in line]
        assert bool(selected_lines) is args.candidate, "Q32 selection provenance does not match requested arm"
        assert "BIV150_PREFILL_Q32_FALLBACK" not in final_logs, "Q32 compilation fell back; no candidate acceptance"
        manifest["q32_selected_lines"] = selected_lines
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        save("workloads_completed", work_completed=True, scores=scores, working_gates=performance_gates(scores),
             matched_control_comparison_required=True)
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--gate-library", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--mode", choices=("all", "throughput"), default="all")
    main(parser.parse_args())
