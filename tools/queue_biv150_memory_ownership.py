"""Serialize allocation diagnostics after an exact owned predecessor identity."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from probe_biv150_memory_ownership import require_idle, sha


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def main(args):
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Dirty source")
    if sha(args.library) != args.library_sha256:
        raise RuntimeError("Unexpected native SO")
    state = dict(queue_pid=os.getpid(), after_pid=args.after_pid,
        after_start_ticks=args.after_start_ticks, source=str(source), stage="waiting",
        commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        library=str(args.library), library_sha256=args.library_sha256)
    state_path = args.out.with_suffix(".queue.json")
    with state_path.open("x") as handle:
        handle.write(json.dumps(state, indent=2))

    def save():
        state_path.write_text(json.dumps(state, indent=2))

    deadline = time.monotonic() + 10800
    while True:
        current = identity(args.after_pid)
        if current is None or current[0] == "Z" or current[1] != args.after_start_ticks:
            break
        if time.monotonic() >= deadline:
            state.update(stage="wait_timeout")
            save()
            raise TimeoutError("Predecessor still live; no diagnostics started")
        time.sleep(5)
    deadline = time.monotonic() + 120
    while True:
        try:
            require_idle()
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                state.update(stage="busy_refused")
                save()
                raise
            time.sleep(5)
    if sha(args.library) != args.library_sha256:
        raise RuntimeError("Native SO changed while waiting")
    command = [sys.executable, "-u", str(source / "tools/probe_biv150_memory_ownership.py"),
        "--library", str(args.library), "--weights", str(args.weights), "--out", str(args.out)]
    with args.out.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(command, cwd=source, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        state.update(stage="probe", pid=process.pid, command=command,
                     started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        code = process.wait()
    state.update(stage="finished", exit_code=code)
    save()
    raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-pid", type=int, required=True)
    parser.add_argument("--after-start-ticks", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
