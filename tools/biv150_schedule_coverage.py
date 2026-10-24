"""Opt-in CPU metadata audit. Diagnostic runs never count as performance scores."""

import functools
import hashlib
import importlib
import inspect
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

PINS = {
    "vllm_fl.worker.model_runner": "677def25d5391887145b0bbabb2222531887c6f7c48b6ea53c1859ad08d8ef61",
    "vllm_fl.compilation.graph": "ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2",
    "vllm_fl.patches.biv150_compute_order": "80b88140b052954ad2d82fb4f666fddabfb5754d224c959f9b5e4b58f913c629",
    "vllm_fl.patches.biv150_gqa_decode": "37d11d08a6a2e563f550909b0011d0ea62f9495049f7983674f52f755f305256",
    "vllm.v1.attention.backends.triton_attn": "1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605",
    "vllm.model_executor.custom_op": "4b4ba01623005d6a93698eaec007a934dd512bf9ec7241090927de819621de2e",
    "vllm.model_executor.layers.rotary_embedding.base": "c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5",
    "vllm.model_executor.models.llama": "1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384",
}


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def callable_description(function):
    """Describe bound/compiled functions without invoking them or reading tensors."""
    chain, seen = [], set()
    while function is not None and id(function) not in seen:
        seen.add(id(function))
        raw = getattr(function, "__func__", function)
        code = getattr(raw, "__code__", None)
        path = getattr(code, "co_filename", None)
        entry = dict(module=getattr(raw, "__module__", None),
                     qualname=getattr(raw, "__qualname__", None), source=path)
        if path and Path(path).is_file():
            entry["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        chain.append(entry)
        function = getattr(raw, "_torchdynamo_orig_callable", None) or getattr(
            raw, "__wrapped__", None
        )
    return chain


class Coverage:
    def __init__(self, root, sources):
        self.root, self.pid = Path(root), os.getpid()
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.data = dict(pid=self.pid, diagnostic_only=True, sources=sources,
                         phase="startup", generation=0, phase_history=[],
                         phases={}, models={}, backend={}, eager_gqa={}, errors=[])

    def error(self, error):
        with self.lock:
            if len(self.data["errors"]) < 64:
                self.data["errors"].append(str(error))

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
                        self.data.update(phase=phase["name"], generation=phase["generation"])
                        self.data["phase_history"].append(
                            dict(phase, acknowledged_ns=time.time_ns())
                        )
                self.flush()
                atomic_json(self.root / f"ack-{self.pid}.json", dict(pid=self.pid, **phase))
            except Exception as error:
                self.error(repr(error))
            self.stop.wait(0.1)

    def record(self, runner, output, scheduled, maximum, selected):
        import numpy as np

        number = runner.input_batch.num_reqs
        computed = runner.input_batch.num_computed_tokens_cpu
        if (type(scheduled) is not np.ndarray or scheduled.dtype != np.int32
                or scheduled.shape != (number,) or type(computed) is not np.ndarray
                or computed.dtype != np.int32 or computed.ndim != 1
                or len(computed) < number):
            raise ValueError("Expected actual CPU int32 scheduled/computed arrays")
        lengths = scheduled.tolist()
        prefixes = computed[:number].tolist()
        total = sum(lengths)
        if (not lengths or min(lengths) <= 0 or max(lengths) != maximum
                or total != output.total_num_scheduled_tokens):
            raise ValueError("Inconsistent CPU schedule; no histogram credited")
        histogram = Counter(map(str, lengths))
        short = [(q, prefix + q) for q, prefix in zip(lengths, prefixes) if 2 <= q <= 8]
        with self.lock:
            phase = self.data["phases"].setdefault(self.data["phase"], dict(
                steps=0, request_rows=0, query_tokens=0, compute_order_steps=0,
                mixed_steps=0, short_steps=0, short_request_rows=0, short_query_tokens=0,
                qlen_histogram={}, short_qlen_kv_histogram={}, step_shapes={},
                short_examples=[],
            ))
            phase["steps"] += 1
            phase["request_rows"] += number
            phase["query_tokens"] += total
            phase["compute_order_steps"] += int(selected)
            phase["mixed_steps"] += int(min(lengths) == 1 and maximum > 1)
            phase["short_steps"] += bool(short)
            phase["short_request_rows"] += len(short)
            phase["short_query_tokens"] += sum(q for q, _ in short)
            for key, count in histogram.items():
                phase["qlen_histogram"][key] = phase["qlen_histogram"].get(key, 0) + count
            for q, kv in short:
                # Exact CPU optimistic length, not a GPU metadata readback.
                key = f"{q}:{kv}"
                table = phase["short_qlen_kv_histogram"]
                table[key] = table.get(key, 0) + 1
            key = f"{number}:{maximum}:{total}:{len(short)}"
            phase["step_shapes"][key] = phase["step_shapes"].get(key, 0) + 1
            if short and len(phase["short_examples"]) < 16:
                phase["short_examples"].append(dict(
                    num_reqs=number, scheduled=lengths, computed_cpu=prefixes,
                    compute_order_selected=bool(selected),
                ))

    def model(self, wrapper):
        runnable = wrapper.runnable
        named_modules = getattr(runnable, "named_modules", None)
        if named_modules is None:
            raise ValueError("Actual GraphWrapper runnable has no named_modules")
        rotary, attention = [], []
        # get_rope may reuse the same nn.Module for all 42 layer call sites.
        # Count layer paths rather than torch's default deduplicated objects.
        for name, module in named_modules(remove_duplicate=False):
            if name.endswith(".self_attn.rotary_emb"):
                entry = dict(name=name, object_id=f"{id(module):x}", class_module=type(module).__module__,
                             class_name=type(module).__qualname__,
                             selected=callable_description(module._forward_method),
                             native=callable_description(module.forward_native),
                             oot=callable_description(module.forward_oot))
                for field in ("head_size", "rotary_dim", "max_position_embeddings",
                              "is_neox_style", "base", "_enforce_enable"):
                    value = getattr(module, field, None)
                    if value is None or type(value) in (int, float, str, bool):
                        entry[field] = value
                cache = getattr(module, "cos_sin_cache", None)
                if cache is not None:
                    entry["cache"] = dict(shape=list(cache.shape), dtype=str(cache.dtype),
                                          device=str(cache.device))
                rotary.append(entry)
            impl = getattr(module, "impl", None)
            if impl is not None and name.endswith(".attn"):
                attention.append(dict(name=name, class_module=type(impl).__module__,
                                      class_name=type(impl).__qualname__,
                                      update=callable_description(impl.do_kv_cache_update),
                                      kv_cache_dtype=getattr(impl, "kv_cache_dtype", None)))
        if len(rotary) != 42 or len(attention) != 42:
            raise ValueError(f"Expected 42 actual rotary/attention objects: {len(rotary)}/{len(attention)}")
        config = wrapper.compilation_config
        model_config = wrapper.vllm_config.model_config
        text_config = model_config.hf_text_config
        architectures = list(model_config.hf_config.architectures)
        if architectures != ["LlamaForCausalLM"] or (
            text_config.hidden_size, text_config.num_hidden_layers,
            text_config.num_attention_heads, text_config.num_key_value_heads
        ) != (2048, 42, 16, 2):
            raise ValueError("Expected actual MiniCPM5 weights with LlamaForCausalLM config")
        record = dict(wrapper_module=type(wrapper).__module__,
                      runnable_class_module=type(runnable).__module__,
                      runnable_class_name=type(runnable).__qualname__,
                      configured_architectures=architectures,
                      runtime_mode=str(wrapper.runtime_mode), rotary=rotary, attention=attention,
                      unique_rotary_objects=len({entry["object_id"] for entry in rotary}),
                      custom_ops=list(config.custom_ops),
                      enabled_custom_ops=sorted(config.enabled_custom_ops),
                      disabled_custom_ops=sorted(config.disabled_custom_ops))
        with self.lock:
            self.data["models"][f"{id(wrapper):x}"] = record

    def wire(self, runner_module, graph_type):
        original_plan = runner_module._biv150_compute_order_plan
        original_init = graph_type.__init__

        @functools.wraps(original_plan)
        def observed_plan(runner, output, scheduled, maximum):
            result = original_plan(runner, output, scheduled, maximum)
            try:
                self.record(runner, output, scheduled, maximum, result is not None)
            except Exception as error:
                self.error(repr(error))
            return result

        @functools.wraps(original_init)
        def observed_init(wrapper, *args, **kwargs):
            original_init(wrapper, *args, **kwargs)
            try:
                self.model(wrapper)
            except Exception as error:
                self.error(repr(error))

        runner_module._biv150_compute_order_plan = observed_plan
        graph_type.__init__ = observed_init


    def wire_kernel(self, kernel, context, none_mode):
        original_run = kernel.run

        @functools.wraps(original_run)
        def observed_run(*args, **kwargs):
            compiled = original_run(*args, **kwargs)
            ctx = context()
            if kwargs.get("warmup", False) or ctx is None or ctx.cudagraph_runtime_mode != none_mode:
                return compiled
            grid = kwargs.get("grid")
            if not isinstance(grid, (tuple, list)) or len(grid) != 3:
                self.error(f"Unknown eager GQA grid: {grid!r}")
                return compiled
            values = tuple(int(v) for v in grid)
            if values[2] != kwargs.get("SEGMENTS"):
                self.error("Eager GQA grid/SEGMENTS mismatch")
                return compiled
            with self.lock:
                counts = self.data["eager_gqa"].setdefault(self.data["phase"], {})
                key = json.dumps(values)
                counts[key] = counts.get(key, 0) + 1
            return compiled

        kernel.run = observed_run


def install(root):
    modules, sources = {}, {}
    for name, expected in PINS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Unpinned schedule audit source: {name}:{digest}")
        modules[name], sources[name] = module, dict(path=str(path), sha256=digest)
    runner = modules["vllm_fl.worker.model_runner"]
    graph = modules["vllm_fl.compilation.graph"]
    order = modules["vllm_fl.patches.biv150_compute_order"]
    if (runner.GraphWrapper is not graph.GraphWrapper
            or runner.is_breakable_cudagraph_enabled()
            or runner._biv150_compute_order_plan is not order.make_plan):
        raise RuntimeError("Unexpected runner aliases or graph implementation")
    # Prove the alias is called for every nonempty execution, before input preparation.
    execute = inspect.getsource(runner.ModelRunnerFL.execute_model)
    if execute.count("compute_order = _biv150_compute_order_plan(") != 1:
        raise RuntimeError("Actual runner schedule call site changed")
    trace = Coverage(root, sources)
    backend = modules["vllm.v1.attention.backends.triton_attn"]
    trace.data["backend"] = {
        name: callable_description(getattr(backend, name))
        for name in ("triton_reshape_and_cache_flash", "triton_reshape_and_cache_flash_per_token_head_quant")
    }
    trace.wire(runner, graph.GraphWrapper)
    def context():
        return graph.get_forward_context() if graph.is_forward_context_available() else None
    trace.wire_kernel(modules["vllm_fl.patches.biv150_gqa_decode"]._gqa_decode_attention,
                      context, graph.CUDAGraphMode.NONE)
    trace.flush()
    threading.Thread(target=trace.control, daemon=True, name="schedule-audit-control").start()
    print("BIV150_SCHEDULE_COVERAGE_INSTALL", dict(pid=os.getpid(), root=str(root)), flush=True)
    return trace
