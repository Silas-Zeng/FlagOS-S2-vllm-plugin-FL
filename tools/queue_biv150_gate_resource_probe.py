"""Wait for one owned queue identity, then refuse busy GPU/port before probing."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def main(args):
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Dirty source")
    assert hashlib.sha256(args.library.read_bytes()).hexdigest()==args.library_sha256
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.out.with_suffix(".queue.json")
    state = dict(
        queue_pid=os.getpid(),
        after_pid=args.after_pid,
        after_start_ticks=args.after_start_ticks,
        stage="waiting",
        source=str(source),
        commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    with state_path.open("x") as handle:
        handle.write(json.dumps(state, indent=2))
    deadline = time.monotonic() + 10800
    while True:
        current = identity(args.after_pid)
        if current is None or current[0] == "Z" or current[1] != args.after_start_ticks:
            break
        if time.monotonic() >= deadline:
            state.update(stage="wait_timeout")
            state_path.write_text(json.dumps(state, indent=2))
            raise TimeoutError("Owned predecessor is still live; no probe started")
        time.sleep(5)
    # The queue may exit just before its server's final GPU cleanup.
    deadline = time.monotonic() + 120
    while True:
        with socket.socket() as probe:
            port_free = probe.connect_ex(("127.0.0.1", 9031)) != 0
        gpu_free = "No running processes found" in subprocess.check_output(
            ["/usr/local/corex/bin/ixsmi"], text=True
        )
        if port_free and gpu_free:
            break
        if time.monotonic() >= deadline:
            state.update(stage="busy_refused")
            state_path.write_text(json.dumps(state, indent=2))
            raise RuntimeError("GPU/port still occupied; never interrupt another task")
        time.sleep(5)
    assert hashlib.sha256(args.library.read_bytes()).hexdigest()==args.library_sha256
    command = [sys.executable,"-u",str(source/"tools/bench_biv150_int8_gemm_silu.py"),
        "--library",str(args.library),"--out",str(args.out),"--resource-tiles","--weight-layer","20",
        "--rows","32","64","40","63","256","2048"]
    with args.out.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=dict(os.environ, PYTHONPATH=str(source)),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        state.update(stage="probe", pid=process.pid, command=command)
        state_path.write_text(json.dumps(state, indent=2))
        code = process.wait()
    state.update(stage="finished", exit_code=code)
    state_path.write_text(json.dumps(state, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-pid", type=int, required=True)
    parser.add_argument("--after-start-ticks", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--library-sha256",required=True)
    main(parser.parse_args())
