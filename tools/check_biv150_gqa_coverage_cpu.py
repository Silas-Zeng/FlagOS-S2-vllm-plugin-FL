"""Check observer call preservation and capture/replay attribution without GPU."""

import argparse
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

from biv150_gqa_coverage import Coverage, atomic_json
from run_biv150_gqa_coverage import analyze


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

        class Graph:
            def __init__(self):
                self.runtime_mode = "FULL"
                self.concrete_cudagraph_entries = {}
                self.calls = 0
                self.fail = False

            def __call__(self):
                self.calls += 1
                if self.fail:
                    raise ValueError("expected original error")
                desc = ctx.batch_descriptor
                if ctx.cudagraph_runtime_mode != self.runtime_mode:
                    kernel[(desc.num_tokens, 2, 16)]()
                elif desc not in self.concrete_cudagraph_entries:
                    self.concrete_cudagraph_entries[desc] = SimpleNamespace(
                        cudagraph=None
                    )
                    for _ in range(3):
                        kernel[
                            (desc.num_tokens, 2, 16 if desc.num_tokens == 40 else 4)
                        ]()
                    self.concrete_cudagraph_entries[desc].cudagraph = object()
                return output

        trace = Coverage(root, {})
        trace.wire(Graph, kernel, lambda: ctx)
        graph = Graph()
        assert graph() is output
        assert kernel.calls == 3 and graph.calls == 1
        trace.data["phase"] = "score-4096"
        assert graph() is graph() is output
        assert kernel.calls == 3 and graph.calls == 3
        ctx.batch_descriptor = Descriptor(32)
        assert graph() is output
        assert kernel.calls == 6
        trace.data["phase"] = "score-16384"
        assert graph() is graph() is output
        verdict = analyze(trace.data)
        assert verdict["phases"]["score-4096"]["b40_full_graph_replays"] == 2
        assert verdict["phases"]["score-4096"]["b40_graph_kernel_launches"] == 6
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
        (package / "biv150_native_w8a8.py").write_text(
            "calls=0\ndef install():\n global calls\n calls+=1\n return {'installed':True,'identity':'original'}\n"
        )
        (fixture / "biv150_gqa_coverage.py").write_text(
            "from pathlib import Path\ndef install(root):\n Path(root,'observed').write_text('yes')\n"
        )
        bootstrap = Path(__file__).resolve().parent / "gqa_coverage_bootstrap"
        command = "import vllm_fl.patches.biv150_native_w8a8 as m; assert m.install()=={'installed':True,'identity':'original'}; assert m.calls==1"
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
