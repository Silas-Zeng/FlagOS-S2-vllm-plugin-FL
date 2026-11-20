"""Check observer call preservation and capture/replay attribution without GPU."""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import builtins

from biv150_gqa_coverage import Coverage, analyze, atomic_json


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = True
    has_lora: bool = False


def main(args):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        ctx = SimpleNamespace(
            cudagraph_runtime_mode="FULL", batch_descriptor=Descriptor(40)
        )
        output = object()

        class Kernel:
            def __init__(self):
                self.calls = 0

            def run(self, *values, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    hash="cpu-only",
                    n_regs=93,
                    n_spills=0,
                    metadata=SimpleNamespace(shared=8192),
                )

            def __getitem__(self, grid):
                return lambda *values, **kwargs: self.run(
                    *values, grid=grid, warmup=False, **kwargs
                )

        kernel = Kernel()

        class FakeDeviceGraph:
            def replay(self):
                pass

        class Graph:
            def __init__(self):
                self.concrete_graph_entries = {}
                self.fail = False
                self.runtime_mode = "FULL"
                self.cudagraph_options = SimpleNamespace(
                    debug_log_enable=False, gc_disable=False, weak_ref_output=False
                )
                self.graph_pool = None
                self.is_debugging_mode = False

            def runnable(self, *args, **kwargs):
                if self.fail:
                    raise ValueError("expected original error")
                desc = ctx.batch_descriptor
                kernel[(desc.num_tokens, 2, 16 if desc.num_tokens == 40 else 4)]()
                return output

        # Execute the project wrapper dispatch itself, including its actual
        # capture/replay branches. Device operations alone use CPU stand-ins.
        actual = Path(__file__).resolve().parents[1] / "vllm_fl/compilation/graph.py"
        raw = actual.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == (
            "ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2"
        )
        cls = next(n for n in ast.parse(raw).body
                   if isinstance(n, ast.ClassDef) and n.name == "GraphWrapper")
        method = next(n for n in cls.body
                      if isinstance(n, ast.FunctionDef) and n.name == "__call__")
        offloader = SimpleNamespace(sync_prev_onload=lambda: None,
                                    join_after_forward=lambda: None)
        def cpu_import(name, *args, **kwargs):
            if name == "vllm.model_executor.offloader.base":
                return SimpleNamespace(get_offloader=lambda: offloader)
            return builtins.__import__(name, *args, **kwargs)
        namespace = dict(
            __builtins__=dict(vars(builtins), __import__=cpu_import),
            is_forward_context_available=lambda: True,
            get_forward_context=lambda: ctx,
            CUDAGraphMode=SimpleNamespace(NONE="NONE"),
            GraphEntry=lambda **kw: SimpleNamespace(graph=None, **kw),
            Graph=SimpleNamespace(graph=FakeDeviceGraph),
            torch=SimpleNamespace(Tensor=type("Tensor", (), {})),
            ExitStack=nullcontext,
            set_graph_pool_id=lambda value: None,
            validate_cudagraph_capturing_enabled=lambda: None,
            current_platform=SimpleNamespace(
                graph_pool_handle=lambda: None, device_type="cuda",
                torch_device_fn=SimpleNamespace(graph=lambda *a, **kw: nullcontext())
            ),
            weak_ref_tensors=lambda value: value,
            compilation_counter=SimpleNamespace(num_cudagraph_captured=0),
        )
        exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
        Graph.__call__ = namespace["__call__"]

        trace = Coverage(root, {})
        trace.wire(Graph, kernel, lambda: ctx, "NONE")
        graph = Graph()
        assert graph() is output
        assert kernel.calls == 1 and len(trace.data["wrapper_instances"]) == 1
        trace.data["phase"] = "score-4096"
        assert graph() is graph() is output
        assert kernel.calls == 1
        ctx.batch_descriptor = Descriptor(32)
        assert graph() is output
        assert kernel.calls == 2
        trace.data["phase"] = "score-16384"
        assert graph() is graph() is output
        verdict = analyze(trace.data)
        assert verdict["phases"]["score-4096"]["b40_full_graph_replays"] == 2
        assert verdict["phases"]["score-4096"]["b40_graph_kernel_launches"] == 2
        assert not verdict["phases"]["score-16384"]["b40_selected"]
        assert not verdict["phases"]["score-16384"]["unmapped_entries"]
        ctx.cudagraph_runtime_mode = "NONE"
        ctx.batch_descriptor = Descriptor(40)
        assert graph() is output
        assert (
            analyze(trace.data)["phases"]["score-16384"]["b40_eager_kernel_launches"]
            == 1
        )
        before = json.dumps(trace.data, sort_keys=True)
        kernel.run(grid=(40, 2, 16), warmup=True)
        assert json.dumps(trace.data, sort_keys=True) == before
        ctx.cudagraph_runtime_mode = "PIECEWISE"
        assert graph() is output
        assert len(trace.data["entries"]) == 2
        graph.fail = True
        try:
            graph()
        except ValueError as error:
            assert str(error) == "expected original error"
        else:
            raise AssertionError("Original exception swallowed")
        assert getattr(trace.active, "key", None) is None
        trace.flush()
        assert json.loads((root / f"trace-{os.getpid()}.json").read_text())["entries"]
        atomic_json(root / "phase.json", dict(name="cpu-phase", generation=1))
        thread = threading.Thread(target=trace.control, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        ack = root / f"ack-{os.getpid()}.json"
        while time.monotonic() < deadline:
            if ack.exists() and json.loads(ack.read_text())["generation"] == 1:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Control phase not acknowledged")
        trace.stop.set()
        thread.join(timeout=1)
        assert not thread.is_alive() and trace.data["phase"] == "cpu-phase"

        fixture = root / "fixture"
        package = fixture / "vllm_fl/patches"
        package.mkdir(parents=True)
        (fixture / "vllm_fl/__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "biv150_gqa_decode.py").write_text(
            "calls=0\ndef install():\n global calls\n calls+=1\n return {'installed':True,'identity':'original'}\n"
        )
        (fixture / "biv150_gqa_coverage.py").write_text(
            "from pathlib import Path\ndef install(root):\n Path(root,'observed').write_text('yes')\n"
        )
        bootstrap = Path(__file__).resolve().parent / "gqa_coverage_bootstrap"
        command = "import vllm_fl.patches.biv150_gqa_decode as m; assert m.install()=={'installed':True,'identity':'original'}; assert m.calls==1"
        subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            cwd=fixture,
            env=dict(
                os.environ,
                PYTHONPATH=os.pathsep.join((str(bootstrap), str(fixture))),
                BIV150_GQA_COVERAGE_DIR=str(root),
            ),
        )
        assert (root / "observed").read_text() == "yes"
    result = dict(
        cpu_only=True,
        pinned_project_dispatch_executed=True,
        original_calls_preserved=True,
        capture_excluded_from_replay=True,
        replay_to_gqa_mapping=True,
        eager_separated=True,
        warmup_ignored=True,
        original_exception_preserved=True,
        import_hook_preserves_install=True,
        phase_acknowledged_before_requests=True,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    main(parser.parse_args())
