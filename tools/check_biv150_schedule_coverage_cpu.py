"""Check actual runner call site, CPU histogram semantics and audit transparency."""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from biv150_schedule_coverage import Coverage, PINS, atomic_json


def main(args):
    source = Path(__file__).resolve().parents[1]
    for name in ("vllm_fl.worker.model_runner", "vllm_fl.compilation.graph",
                 "vllm_fl.patches.biv150_compute_order"):
        path = source / (name.replace(".", "/") + ".py")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[name]
    tree = ast.parse((source / "vllm_fl/worker/model_runner.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerFL")
    execute = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    # Run the actual consecutive CPU schedule/alias/_prepare_inputs statements.
    statements = next(n.body for n in execute.body if isinstance(n, ast.With)
                      and any(isinstance(child, ast.Assign)
                              and ast.unparse(child.targets[0]) == "num_reqs"
                              for child in n.body))
    begin = next(i for i, n in enumerate(statements)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "num_reqs")
    end = next(i for i, n in enumerate(statements) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call)
               and ast.unparse(n.value.func) == "self._prepare_inputs")
    snippet = ast.Module(body=statements[begin:end + 1], type_ignores=[])
    assert len(snippet.body) == 8
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        trace = Coverage(root, {})
        sentinel, calls, preparations = object(), [], []

        def original(runner, output, scheduled, maximum):
            calls.append((runner, output, scheduled, maximum))
            if maximum == 999:
                raise ValueError("original exception")
            return sentinel if maximum > 1 else None

        class Wrapper:
            def __init__(self, runnable, config):
                self.runnable, self.compilation_config = runnable, config
                self.runtime_mode = "FULL"

        module = SimpleNamespace(_biv150_compute_order_plan=original)
        trace.wire(module, Wrapper)
        trace.data["phase"] = "score-4096"
        runner = SimpleNamespace()
        def prepare(output, scheduled, *, compute_order):
            preparations.append((scheduled.copy(), compute_order))
            return sentinel, None
        runner._prepare_inputs = prepare
        fixture_lengths = [[1, 1], [1, 2, 3, 4, 5, 6, 7, 8, 2048], [9, 16, 32]]
        for lengths in fixture_lengths:
            runner.input_batch = SimpleNamespace(
                num_reqs=len(lengths), req_ids=list(map(str, range(len(lengths)))),
                num_computed_tokens_cpu=np.arange(len(lengths), dtype=np.int32) * 100,
            )
            output = SimpleNamespace(num_scheduled_tokens=dict(zip(
                runner.input_batch.req_ids, lengths)), total_num_scheduled_tokens=sum(lengths))
            namespace = dict(self=runner, scheduler_output=output, np=np,
                             _biv150_compute_order_plan=module._biv150_compute_order_plan)
            exec(compile(snippet, "<pinned-runner-schedule>", "exec"), namespace)
            assert namespace["logits_indices"] is sentinel
            assert calls[-1][2].tolist() == lengths == preparations[-1][0].tolist()
            assert preparations[-1][1] is (sentinel if max(lengths) > 1 else None)
        phase = trace.data["phases"]["score-4096"]
        assert phase["steps"] == 3 and phase["request_rows"] == 14
        assert phase["short_steps"] == 1 and phase["short_request_rows"] == 7
        assert phase["short_query_tokens"] == 35
        assert phase["query_tokens"] == sum(map(sum, fixture_lengths))
        assert phase["compute_order_steps"] == 2 and phase["mixed_steps"] == 1
        assert phase["qlen_histogram"]["1"] == 3
        assert phase["short_qlen_kv_histogram"] == {f"{q}:{100*(q-1)+q}": 1 for q in range(2, 9)}
        before = json.dumps(trace.data, sort_keys=True)
        try:
            module._biv150_compute_order_plan(runner, output, np.array([999], dtype=np.int32), 999)
        except ValueError as error:
            assert str(error) == "original exception"
        else:
            raise AssertionError("Original exception swallowed")
        assert before == json.dumps(trace.data, sort_keys=True)
        module._biv150_compute_order_plan(runner, output, [1], 1)
        assert len(trace.data["errors"]) == 1  # Bad audit input does not change original result.
        trace.data["errors"].clear()

        def fn(*args):
            return sentinel
        rotary = SimpleNamespace(_forward_method=fn, forward_native=fn, forward_oot=fn,
                                 head_size=128, rotary_dim=128, is_neox_style=True)
        impl = SimpleNamespace(do_kv_cache_update=fn, kv_cache_dtype="auto")
        entries = [(f"model.layers.{i}.self_attn.rotary_emb", rotary) for i in range(42)]
        entries += [(f"model.layers.{i}.self_attn.attn", SimpleNamespace(impl=impl)) for i in range(42)]
        runnable = SimpleNamespace(named_modules=lambda: iter(entries))
        config = SimpleNamespace(custom_ops=["none"], enabled_custom_ops=set(),
                                 disabled_custom_ops={"rotary_embedding"})
        wrapper = Wrapper(runnable, config)
        assert wrapper.runnable is runnable and len(trace.data["models"]) == 1
        assert len(next(iter(trace.data["models"].values()))["rotary"]) == 42
        assert rotary._forward_method is fn and impl.do_kv_cache_update is fn
        trace.flush()
        assert json.loads((root / f"trace-{os.getpid()}.json").read_text())["phases"]
        fixture = root / "fixture"
        package = fixture / "vllm_fl/patches"
        package.mkdir(parents=True)
        (fixture / "vllm_fl/__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "biv150_native_w8a8.py").write_text(
            "calls=0\ndef install():\n global calls\n calls+=1\n return {'installed':True,'identity':'original'}\n")
        (fixture / "biv150_schedule_coverage.py").write_text(
            "from pathlib import Path\ndef install(root):\n Path(root,'observed').write_text('yes')\n")
        command = "import vllm_fl.patches.biv150_native_w8a8 as m; assert m.install()=={'installed':True,'identity':'original'}; assert m.calls==1"
        env = {k: v for k, v in os.environ.items() if k != "BIV150_SCHEDULE_COVERAGE_DIR"}
        env["PYTHONPATH"] = os.pathsep.join((str(source / "tools/schedule_coverage_bootstrap"), str(fixture)))
        subprocess.run([sys.executable, "-c", command], cwd=fixture, env=env, check=True)
        assert not (root / "observed").exists()
        env["BIV150_SCHEDULE_COVERAGE_DIR"] = str(root)
        subprocess.run([sys.executable, "-c", command], cwd=fixture, env=env, check=True)
        assert (root / "observed").read_text() == "yes"
    result = dict(cpu_only=True, actual_pinned_runner_schedule_executed=True,
                  original_result_and_exception_preserved=True, cpu_qlen_1_through_8_and_large_checked=True,
                  exact_cpu_length_histogram_checked=True, no_forward_methods_replaced=True,
                  import_hook_opt_in_and_install_preserved=True,
                  real_model_bindings_pending=True, gpu_diagnostic_pending=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    main(parser.parse_args())
