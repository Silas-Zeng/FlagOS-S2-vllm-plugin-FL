"""Run a throughput-only control after one exact owned predecessor exits.

Reuse the same frozen runtime directory and private native library. Only the
owned BF16 operator flag changes; the official server/benchmark CLI is intact.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def main(args):
    runtime = args.runtime_source.resolve()
    driver = Path(__file__).resolve().with_name("run_biv150_gate_team_acceptance.py")
    state_path = args.out.with_suffix(".queue.json")
    state = dict(pid=os.getpid(), runtime=str(runtime), driver=str(driver),
                 after_pid=args.after_pid, after_start_ticks=args.after_start_ticks,
                 throughput_only=True, stage="waiting",
                 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("x") as f:
        json.dump(state, f, indent=2)
    deadline = time.monotonic() + 10800
    while True:
        current = identity(args.after_pid)
        if current is None or current[0] == "Z" or current[1] != args.after_start_ticks:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Owned predecessor still live; no control started")
        time.sleep(5)
    deadline = time.monotonic() + 120
    while True:
        with socket.socket() as port:
            port_free = port.connect_ex(("127.0.0.1", 9031)) != 0
        gpu_free = "No running processes found" in subprocess.check_output(
            ["/usr/local/corex/bin/ixsmi"], text=True)
        if port_free and gpu_free:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU/port busy: refuse to start or interrupt other work")
        time.sleep(5)
    actual = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    assert actual == args.runtime_commit
    assert not subprocess.check_output(["git", "-C", str(runtime), "status", "--porcelain"], text=True).strip()
    library = runtime / "build-native-w8a8/biv150_native_w8a8.so"
    reference = json.loads((args.candidate_run / "manifest.json").read_text())
    assert reference["git_commit"] == actual and Path(reference["source"]).resolve() == runtime
    assert reference["env"]["VLLM_FL_BIV150_BF16_SMALL_SPLITK"] == "1"
    assert hashlib.sha256(library.read_bytes()).hexdigest() == reference["native_library_sha256"]
    state.update(runtime_commit=actual, native_library=str(library),
                 native_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    command = [sys.executable, "-u", str(driver), "--source", str(runtime),
               "--out", str(args.out), "--mode", "throughput", "--bf16-small-control"]
    with args.out.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(command, cwd=runtime,
            env=dict(os.environ, PYTHONPATH=str(runtime), PYTHONUNBUFFERED="1"),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state.update(stage="control", child_pid=process.pid, command=command)
        state_path.write_text(json.dumps(state, indent=2))
        code = process.wait()
    if code == 0:
        controlled = json.loads((args.out / "manifest.json").read_text())
        assert controlled["command"] == reference["command"]
        assert controlled["native_library_sha256"] == reference["native_library_sha256"]
        assert all(controlled["source_hashes"][k] == v for k, v in reference["source_hashes"].items())
        allowed = {"VLLM_FL_BIV150_BF16_SMALL_SPLITK", "TORCHINDUCTOR_CACHE_DIR"}
        assert {k: v for k, v in controlled["env"].items() if k not in allowed} == {
            k: v for k, v in reference["env"].items() if k not in allowed}
        state["matched_runtime_library_cli_and_other_flags"] = True
    state.update(stage="finished", exit_code=code)
    state_path.write_text(json.dumps(state, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-source", type=Path, required=True)
    p.add_argument("--runtime-commit", required=True)
    p.add_argument("--candidate-run", type=Path, required=True)
    p.add_argument("--after-pid", type=int, required=True)
    p.add_argument("--after-start-ticks", required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
