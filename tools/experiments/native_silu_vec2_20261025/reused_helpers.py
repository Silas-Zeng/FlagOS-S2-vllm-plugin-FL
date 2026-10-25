"""从既有 owner 提取的原函数；不改设备门和计时口径。"""
import socket, subprocess, math, time
from pathlib import Path
from datetime import datetime, timezone
IXSMI="/usr/local/corex-4.5.0/bin/ixsmi"

def utc():
    return datetime.now(timezone.utc).isoformat()

def preflight(ixsmi):
    # This function uses no torch/vLLM imports and initializes no CUDA context.
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family) as sock:
                sock.settimeout(1)
                if sock.connect_ex((address, 9031)) == 0:
                    raise RuntimeError("port 9031 occupied")
        except OSError as exc:
            if family == socket.AF_INET:
                raise RuntimeError("could not verify port 9031") from exc
    output = subprocess.check_output(
        [ixsmi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].isdigit():
        raise RuntimeError(f"expected one accelerator and numeric free MiB: {lines!r}")
    free_mib = int(lines[0])
    if free_mib < 30000:
        raise RuntimeError(f"accelerator occupied: free MiB={free_mib}, required>=30000")
    return {"checked_utc": utc(), "port_9031": "free", "free_mib": free_mib}

def cpu_snapshot():
    names = {"stat": "/sys/fs/cgroup/cpu/cpu.stat",
             "quota": "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
             "period": "/sys/fs/cgroup/cpu/cpu.cfs_period_us"}
    raw = {name: Path(path).read_text(encoding="utf-8").strip() for name, path in names.items()}
    counters = {line.split()[0]: int(line.split()[1]) for line in raw["stat"].splitlines()}
    if (not {"nr_periods", "nr_throttled", "throttled_time"} <= counters.keys()
            or int(raw["quota"]) != 600000 or int(raw["period"]) != 100000):
        raise RuntimeError("expected the actual unchanged v1 six-CPU quota")
    return {"paths": names, "raw": raw, "counters": counters}

def timed_batch(torch, callback, pair, loops):
    torch.cuda.synchronize()
    before_cpu = cpu_snapshot()
    start, end = pair
    begin = time.perf_counter_ns()
    start.record()
    thread_begin, process_begin = time.thread_time_ns(), time.process_time_ns()
    for _ in range(loops):
        callback()
    submit_thread_ns = time.thread_time_ns() - thread_begin
    submit_process_ns = time.process_time_ns() - process_begin
    end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter_ns() - begin) / 1e6
    after_cpu = cpu_snapshot()
    event = start.elapsed_time(end)
    if not math.isfinite(event) or event <= 0:
        raise RuntimeError("invalid timing Event elapsed time")
    return {"calls": loops, "wall_total_ms": wall, "wall_per_call_ms": wall / loops,
            "event_total_ms": event, "event_per_call_ms": event / loops,
            "submit_thread_cpu_ns": submit_thread_ns, "submit_process_cpu_ns": submit_process_ns,
            "cpu_stat_before": before_cpu, "cpu_stat_after": after_cpu,
            "cpu_stat_delta": {name: after_cpu["counters"][name] - before_cpu["counters"][name]
                               for name in before_cpu["counters"]}}
