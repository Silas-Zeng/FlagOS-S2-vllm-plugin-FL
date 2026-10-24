"""Serial, owned build/check/model acceptance for the residual service candidate."""

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def main(args):
    source = args.source.resolve()
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Candidate source must be clean")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Existing service: refuse to start")
    if "No running processes found" not in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ):
        raise RuntimeError("GPU occupied: refuse to start")
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONUNBUFFERED="1", MAX_JOBS="2")
    build = source / "build-native-w8a8"
    steps = [
        (
            "build",
            [
                sys.executable,
                str(source / "tools/build_biv150_native_w8a8.py"),
                "--build-dir",
                str(build),
            ],
        ),
        (
            "operator-check",
            [
                sys.executable,
                str(source / "tools/check_biv150_gate_auto.py"),
                "--library",
                str(build / "biv150_native_w8a8.so"),
                "--out",
                str(args.out / "operator-check.json"),
            ],
        ),
        (
            "cached-dispatch-check",
            [
                sys.executable,
                str(source / "tools/check_biv150_quant_cached_dispatch.py"),
                "--library",
                str(build / "biv150_native_w8a8.so"),
                "--out",
                str(args.out / "cached-dispatch-check.json"),
            ],
        ),
        (
            "acceptance",
            [
                sys.executable,
                str(source / "tools/run_biv150_gate_team_acceptance.py"),
                "--source",
                str(source),
                "--out",
                str(args.out / "full"),
                "--mode",
                "all",
            ],
        ),
    ]
    state = dict(queue_pid=os.getpid(), source=str(source), commit=commit)
    for stage, command in steps:
        with (args.out / f"{stage}.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            state.update(
                stage=stage,
                pid=process.pid,
                command=command,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            (args.out / "state.json").write_text(json.dumps(state, indent=2))
            print(json.dumps(state), flush=True)
            code = process.wait()
        state["exit_code"] = code
        (args.out / "state.json").write_text(json.dumps(state, indent=2))
        if code:
            raise SystemExit(code)
        state.pop("exit_code")
    state["completed"] = True
    (args.out / "state.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
