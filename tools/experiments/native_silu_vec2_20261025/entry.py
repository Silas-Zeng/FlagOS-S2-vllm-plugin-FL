"""有界 BI-V150 组件入口；只有 root 启动后才运行设备子进程。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path("/workspace/experiments/minicpm-s2")
DB = Path("/root/.flaggems/config_cache/TunedConfig_iluvatar_triton_3_6.db")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/workspace/MiniCPM5-2B"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        from worker import worker
        return worker(args.run_dir, args.model)
    import fcntl
    import reused_helpers as owner
    for name, expected in read(HERE / "asset-pins.json").items():
        if sha(HERE / name) != expected:
            raise RuntimeError("入口资产变化：" + name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = ROOT / "framework" / (stamp + "-native-silu-vec2-component")
    run.mkdir(exist_ok=False)
    assets = run / "source"
    assets.mkdir()
    for name in read(HERE / "asset-pins.json"):
        shutil.copyfile(HERE / name, assets / name)
    shutil.copyfile(HERE / "asset-pins.json", assets / "asset-pins.json")
    state = dict(started_utc=stamp, owner_pid=os.getpid(),
        scope="独立M2048尾处理组件；只加载私有测试库，生产native不变", run=str(run),
        command=[sys.executable, *sys.argv], owned_processes=[], exit_code=1)
    save(run / "manifest.json", state)
    lock = child = monitor = None
    locked = copied_db = False
    status = 1
    try:
        lock = (ROOT / "framework-device.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        state["preflight"] = owner.preflight(owner.IXSMI)
        gpu = subprocess.check_output([owner.IXSMI], text=True, timeout=10)
        (run / "preflight-ixsmi.txt").write_text(gpu)
        if "No running processes found" not in gpu:
            raise RuntimeError("设备已有进程")
        shutil.copyfile(DB, run / "database-original.sqlite")
        copied_db = True
        state["database_original_sha256"] = sha(DB)
        shutil.copyfile(args.model / "config.json", assets / "model-config.json")
        env = dict(os.environ)
        # 固定两臂同一个 worker/DB；本组件只加载私有命名空间测试库，不加载框架或生产 native。
        for key in tuple(env):
            if key.startswith(("VLLM_FL_BIV150_", "BIV150_", "MINICPM_NATIVE_", "MINICPM_DECODE_GAP_")):
                del env[key]
        env.pop("PYTHONPATH", None)
        env.update(PYTHONPATH=str(assets), PYTHONDONTWRITEBYTECODE="1")
        state["environment"] = {k: v for k, v in env.items() if k in
            ("VLLM_PLUGINS", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
             "OMP_NUM_THREADS", "MKL_NUM_THREADS", "USE_FLAGGEMS", "USE_FLAGTUNE", "TRITON_CACHE_DIR",
             "CUDA_VISIBLE_DEVICES", "IX_VISIBLE_DEVICES", "COREX_VISIBLE_DEVICES")}
        command = [sys.executable, str(assets / "entry.py"), "--worker", "--run-dir", str(run), "--model", str(args.model)]
        state["worker_command"] = command
        with (run / "worker.log").open("x") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        state["owned_processes"].append(dict(role="worker", pid=child.pid))
        (run / "worker.pid").write_text(str(child.pid) + "\n")
        with (run / "monitor.log").open("x") as log:
            monitor = subprocess.Popen(["bash", str(assets / "monitor_gpu.sh"), str(child.pid), str(run / "gpu.csv")],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        state["owned_processes"].append(dict(role="monitor", pid=monitor.pid))
        save(run / "manifest.json", state)
        print(json.dumps(dict(run=str(run), worker_pid=child.pid, owner_pid=os.getpid())), flush=True)
        status = child.wait(timeout=420)
    except BaseException:
        state["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        for process in (child, monitor):
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                except BaseException:
                    state["cleanup_error"] = traceback.format_exc()
                    status = 1
        if copied_db:
            try:
                if child is not None and child.poll() is None:
                    raise RuntimeError("自有 worker 尚未退出，不覆盖它可能正在使用的 DB")
                state["database_after_sha256"] = sha(DB)
                if sha(DB) != state["database_original_sha256"]:
                    shutil.copyfile(DB, run / "database-after.sqlite")
                    shutil.copyfile(run / "database-original.sqlite", DB)
                state["database_restored"] = sha(DB) == state["database_original_sha256"]
            except BaseException:
                state["database_restore_error"], status = traceback.format_exc(), 1
        if locked:
            try:
                state["postflight"] = owner.preflight(owner.IXSMI)
                post = subprocess.check_output([owner.IXSMI], text=True, timeout=10)
                (run / "postflight-ixsmi.txt").write_text(post)
                if "No running processes found" not in post:
                    raise RuntimeError("worker 退出后仍有 GPU 进程")
            except BaseException:
                state["postflight_error"], status = traceback.format_exc(), 1
        state.update(exit_code=status, finished_utc=datetime.now(timezone.utc).isoformat())
        save(run / "manifest.json", state)
        (run / "exit-code").write_text(str(status) + "\n")
        if lock is not None:
            lock.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
