"""Private build and standalone norm/quantization/gate probe, no service launch."""

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def main(args):
    source = args.source.resolve()
    assert not subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 9031)) != 0, "Existing service"
    assert "No running processes found" in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ), "GPU busy"
    args.out.mkdir(parents=True, exist_ok=False)
    build = source / "build-native-w8a8"
    env = dict(os.environ, PYTHONPATH=str(source), MAX_JOBS="2", PYTHONUNBUFFERED="1")
    state = dict(
        driver_pid=os.getpid(),
        commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    steps = [
        ("bounded-abi-cpu-check", [sys.executable,
            str(source / "tools/check_biv150_bounded_norm_cpu.py"),
            "--out", str(args.out / "bounded-abi-cpu-check.json")]),
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
            "existing-operator-check",
            [
                sys.executable,
                str(source / "tools/check_biv150_gate_auto.py"),
                "--library",
                str(build / "biv150_native_w8a8.so"),
                "--out",
                str(args.out / "existing-check.json"),
            ],
        ),
        ("cached-dispatch-check", [sys.executable,
            str(source / "tools/check_biv150_quant_cached_dispatch.py"),
            "--library", str(build / "biv150_native_w8a8.so"),
            "--out", str(args.out / "cached-dispatch-check.json")]),
        (
            "probe",
            [
                sys.executable,
                str(source / "tools/bench_biv150_bounded_norm_quant.py"),
                "--library",
                str(build / "biv150_native_w8a8.so"),
                "--out",
                str(args.out / "probe"),
            ],
        ),
    ]
    for stage, command in steps:
        with (args.out / (stage + ".log")).open("x") as log:
            p = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            state.update(stage=stage, pid=p.pid, command=command)
            (args.out / "state.json").write_text(json.dumps(state, indent=2))
            code = p.wait()
        state["exit_code"] = code
        (args.out / "state.json").write_text(json.dumps(state, indent=2))
        if code:
            raise SystemExit(code)
    state["completed"] = True
    (args.out / "state.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
