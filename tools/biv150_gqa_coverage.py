"""CPU-only capture/replay observer; diagnostic runs are not throughput scores."""

import functools
import hashlib
import importlib
import json
import os
import threading
import time
from pathlib import Path

PINS = {
    "vllm_fl.compilation.graph": "ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2",
    "vllm_fl.worker.model_runner": "a478883e87addd87302fee5c8d331dad42f8279be9212283fb4539bf34335a16",
    "vllm_fl.patches.biv150_gqa_decode": "2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1",
}


def analyze(trace):
    verdict = dict(diagnostic_only=True, errors=trace["errors"], phases={})
    for phase in ("score-4096", "score-16384"):
        counts = trace["replays"].get(phase, {})
        b40, launches, unknown = 0, 0, []
        for key, count in counts.items():
            entry = trace["entries"].get(key)
            if entry is None:
                unknown.append(key)
                continue
            gqa = entry["gqa_grids"]
            if not gqa:
                unknown.append(key)
            hits = sum(x["calls"] for x in gqa.values() if x["grid"] == [40, 2, 16])
            if hits:
                b40 += count
                launches += count * hits
        eager = trace["eager_gqa"].get(phase, {}).get("[40, 2, 16]", 0)
        verdict["phases"][phase] = dict(
            full_graph_replays=sum(counts.values()),
            b40_full_graph_replays=b40,
            b40_graph_kernel_launches=launches,
            b40_eager_kernel_launches=eager,
            b40_selected=b40 + eager > 0,
            unmapped_entries=unknown,
        )
    return verdict


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


class Coverage:
    def __init__(self, root, sources):
        self.root, self.pid = Path(root), os.getpid()
        self.lock, self.active = threading.RLock(), threading.local()
        self.stop = threading.Event()
        self.data = dict(
            pid=self.pid,
            diagnostic_only=True,
            sources=sources,
            phase="startup",
            generation=0,
            phase_history=[],
            wrapper_instances={},
            entries={},
            replays={},
            eager_gqa={},
            errors=[],
        )

    def error(self, value):
        with self.lock:
            if len(self.data["errors"]) < 64:
                self.data["errors"].append(str(value))

    def flush(self):
        with self.lock:
            snapshot = json.loads(json.dumps(self.data))
        atomic_json(self.root / f"trace-{self.pid}.json", snapshot)

    def control(self):
        while not self.stop.is_set():
            try:
                phase = json.loads((self.root / "phase.json").read_text())
                with self.lock:
                    if phase["generation"] != self.data["generation"]:
                        self.data.update(
                            phase=phase["name"], generation=phase["generation"]
                        )
                        self.data["phase_history"].append(
                            dict(phase, acknowledged_ns=time.time_ns())
                        )
                self.flush()
                atomic_json(
                    self.root / f"ack-{self.pid}.json", dict(pid=self.pid, **phase)
                )
            except Exception as error:
                self.error(repr(error))
            self.stop.wait(0.1)

    def wire(self, wrapper_type, kernel, context, none_mode):
        original_init = wrapper_type.__init__
        original_call, original_run = wrapper_type.__call__, kernel.run

        @functools.wraps(original_init)
        def observed_init(wrapper, *args, **kwargs):
            original_init(wrapper, *args, **kwargs)
            with self.lock:
                self.data["wrapper_instances"][f"{id(wrapper):x}"] = dict(
                    module=type(wrapper).__module__,
                    name=type(wrapper).__qualname__,
                    runtime_mode=str(wrapper.runtime_mode),
                )

        @functools.wraps(original_call)
        def observed_call(wrapper, *args, **kwargs):
            ctx = context()
            key, capture = None, False
            if (
                ctx is not None
                and ctx.cudagraph_runtime_mode != none_mode
                and ctx.cudagraph_runtime_mode == wrapper.runtime_mode
            ):
                desc = ctx.batch_descriptor
                if desc is not None:
                    descriptor = {
                        name: getattr(desc, name, None)
                        for name in ("num_tokens", "num_reqs", "uniform", "has_lora")
                    }
                    key = f"{id(wrapper):x}:" + json.dumps(descriptor, sort_keys=True)
                    entry = wrapper.concrete_graph_entries.get(desc)
                    capture = entry is None or entry.graph is None
                    with self.lock:
                        self.data["entries"].setdefault(
                            key, dict(descriptor=descriptor, gqa_grids={})
                        )
            previous = getattr(self.active, "key", None)
            self.active.key = key if capture else None
            try:
                result = original_call(wrapper, *args, **kwargs)
            finally:
                self.active.key = previous
            if key is not None and not capture:
                with self.lock:
                    counts = self.data["replays"].setdefault(self.data["phase"], {})
                    counts[key] = counts.get(key, 0) + 1
            return result

        @functools.wraps(original_run)
        def observed_run(*args, **kwargs):
            compiled = original_run(*args, **kwargs)
            if kwargs.get("warmup", False):
                return compiled
            grid = kwargs.get("grid")
            if not isinstance(grid, (tuple, list)) or len(grid) != 3:
                self.error(f"Unknown GQA grid: {grid!r}")
                return compiled
            grid = [int(x) for x in grid]
            label = json.dumps(grid)
            with self.lock:
                key = getattr(self.active, "key", None)
                if key is not None:
                    grids = self.data["entries"][key]["gqa_grids"]
                    value = grids.setdefault(
                        label,
                        dict(
                            grid=grid,
                            calls=0,
                            compiled_hash=getattr(compiled, "hash", None),
                            n_regs=getattr(compiled, "n_regs", None),
                            n_spills=getattr(compiled, "n_spills", None),
                            shared=getattr(
                                getattr(compiled, "metadata", None), "shared", None
                            ),
                        ),
                    )
                    value["calls"] += 1
                else:
                    counts = self.data["eager_gqa"].setdefault(self.data["phase"], {})
                    counts[label] = counts.get(label, 0) + 1
            return compiled

        wrapper_type.__init__ = observed_init
        wrapper_type.__call__ = observed_call
        kernel.run = observed_run


def install(root):
    modules, sources = {}, {}
    for name, expected in PINS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Unpinned diagnostic source: {name}:{digest}")
        modules[name], sources[name] = module, dict(path=str(path), sha256=digest)
    graph = modules["vllm_fl.compilation.graph"]
    gqa = modules["vllm_fl.patches.biv150_gqa_decode"]
    trace = Coverage(root, sources)

    def context():
        return (
            graph.get_forward_context()
            if graph.is_forward_context_available()
            else None
        )

    runner = modules["vllm_fl.worker.model_runner"]
    if runner.GraphWrapper is not graph.GraphWrapper:
        raise RuntimeError("Runner does not bind the observed project GraphWrapper")
    if runner.is_breakable_cudagraph_enabled():
        raise RuntimeError("Breakable graph enabled; this observer targets project GraphWrapper")
    trace.wire(
        graph.GraphWrapper,
        gqa._gqa_decode_attention,
        context,
        graph.CUDAGraphMode.NONE,
    )
    trace.flush()
    threading.Thread(
        target=trace.control, daemon=True, name="gqa-coverage-control"
    ).start()
    print(
        "BIV150_GQA_COVERAGE_INSTALL", dict(pid=os.getpid(), root=str(root)), flush=True
    )
    return trace
