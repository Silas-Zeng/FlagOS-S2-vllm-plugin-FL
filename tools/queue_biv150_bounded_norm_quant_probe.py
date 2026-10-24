"""Wait for one owned queue identity, then refuse busy GPU/port before probing."""

import argparse
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


def require_completed(path, commit):
    if path is None and commit is None:
        return
    if path is None or commit is None:
        raise ValueError("Require both completed-state path and commit")
    predecessor = json.loads(path.read_text())
    if predecessor.get("completed") is not True or predecessor.get("commit") != commit:
        raise RuntimeError("Required upstream formal acceptance did not complete")


def main(args):
    if args.live_carry_chain:
        if not all((args.library, args.library_source, args.library_sha256)):
            raise ValueError("Joint probe requires pinned private library/source/SHA")
    if (args.require_completed_state is None) != (args.require_commit is None):
        raise ValueError("Require both completed-state path and commit")
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Dirty source")
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
    try:
        require_completed(args.require_completed_state, args.require_commit)
    except Exception as error:
        state.update(stage="upstream_not_validated", error=repr(error))
        state_path.write_text(json.dumps(state, indent=2))
        raise
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
    command = [
        sys.executable,
        "-u",
        str(source / "tools/run_biv150_bounded_norm_quant_probe.py"),
        "--source",
        str(source),
        "--out",
        str(args.out),
    ]
    env = dict(os.environ, PYTHONPATH=str(source))
    if args.live_carry_chain:
        subprocess.run([sys.executable, str(source / "tools/check_biv150_bounded_norm_cpu.py"),
                        "--out", str(args.out.with_suffix(".cpu.json"))], check=True)
        command = [sys.executable, "-u", str(source / "tools/bench_biv150_live_carry_chain.py"),
                   "--library", str(args.library), "--library-source", str(args.library_source),
                   "--library-sha256", args.library_sha256, "--out", str(args.out)]
        env["TORCHINDUCTOR_CACHE_DIR"] = str(args.out.with_name(args.out.name + "-inductor-cache"))
    with args.out.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=env,
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
    parser.add_argument("--require-completed-state", type=Path)
    parser.add_argument("--require-commit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live-carry-chain", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--library-source", type=Path)
    parser.add_argument("--library-sha256")
    main(parser.parse_args())
