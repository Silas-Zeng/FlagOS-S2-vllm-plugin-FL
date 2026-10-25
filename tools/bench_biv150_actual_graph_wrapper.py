"""Explicit-run integration smoke for the installed FL GraphWrapper.

Default execution is CPU-only and imports no SDK. The GPU owner runs --run in a
fresh personal-instance process with --tools-dir pointing at diagnostic helpers.
This exercises the actual GraphWrapper.__call__, ForwardContext and GraphEntry;
it creates no vLLM engine or model. Dummy model/KV inventory exists solely for
the unchanged HiddenStateOutputGuard. Timings are diagnostic, never scores.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(args):
    global tl  # Triton resolves language symbols through the kernel's module globals.
    tools = Path(args.tools_dir).resolve()
    destination = Path(args.output_dir).resolve()
    require((tools / "biv150_profile_service.py").is_file(), "Missing service helper")
    sys.path.insert(0, str(tools))
    os.environ["BIV150_PROFILE_DIR"] = str(destination)
    import torch
    import triton
    import triton.language as tl
    import biv150_profile_service as service

    require(Path(service.__file__).resolve() == tools / "biv150_profile_service.py",
            "Imported the wrong service helper")
    require(service._STATE is None, "Use a fresh process without preinstalled profile hooks")
    service.install()
    state = service._STATE
    require(state is not None and not state.active, "Diagnostic helper did not install")

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, ForwardContext, override_forward_context
    from vllm_fl.compilation.graph import GraphEntry, GraphOptions, GraphWrapper

    # Triton must load AFTER service.install() registers the actual load hook.
    @triton.jit
    def actual_wrapper_smoke_finish(source, output, N: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(source + index, index < N, other=0).to(tl.float32)
        tl.store(output + index, value * 2.0 + 1.0, index < N)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(
                torch.tensor(3.0, device="cuda", dtype=torch.bfloat16), requires_grad=False)
            self.register_buffer("guard_buffer", torch.zeros(1, device="cuda"))

    model = DummyModel()
    source = torch.full((64, 2048), 2.0, dtype=torch.bfloat16, device="cuda")
    intermediate = torch.empty_like(source)
    output = torch.empty_like(source)
    kv = torch.zeros((2, 1, 16, 2, 128), dtype=torch.bfloat16, device="cuda")
    static_context = {"smoke.attention": SimpleNamespace(kv_cache=kv)}
    compilation_config = SimpleNamespace(static_forward_context=static_context)
    worker = SimpleNamespace(model_runner=SimpleNamespace(
        model=model, kv_caches=[kv], compilation_config=compilation_config))

    def body(x):
        # One native kernel plus one Triton kernel, both using stable storage.
        torch.add(x, model.bias, out=intermediate)
        actual_wrapper_smoke_finish[(triton.cdiv(x.numel(), 256),)](
            intermediate, output, x.numel(), BLOCK=256, num_warps=4)
        return output

    # Precompile/warm up outside capture and before the diagnostic window.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream), torch.no_grad():
        for _ in range(3):
            body(source)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        body(source)
    graph.replay()
    torch.cuda.synchronize()
    require(torch.equal(output.cpu(), torch.full((64, 2048), 11.0, dtype=torch.bfloat16)),
            "Initial original graph output is wrong")
    require(bool(state.kernel_loads), "Triton load hook missed the smoke kernel")

    descriptor = BatchDescriptor(num_tokens=64, num_reqs=64, uniform=True)
    entry = GraphEntry(batch_descriptor=descriptor, graph=graph, output=output,
                       input_addresses=[source.data_ptr()])
    # Deliberately skip constructor-only engine/config/pool work. All fields used
    # by the unmodified real replay path are initialized, with a real GraphEntry.
    wrapper = object.__new__(GraphWrapper)
    wrapper.runnable = body
    wrapper.runtime_mode = CUDAGraphMode.FULL
    wrapper.vllm_config = SimpleNamespace(compilation_config=compilation_config)
    wrapper.compilation_config = compilation_config
    wrapper.first_run_finished = True
    wrapper.is_debugging_mode = True  # Exercise real input-address validation too.
    wrapper._runnable_str = "actual_wrapper_smoke_body"
    wrapper.graph_pool = None
    wrapper.cudagraph_options = GraphOptions(debug_log_enable=False, weak_ref_output=False)
    wrapper.concrete_graph_entries = {descriptor: entry}
    context = ForwardContext(no_compile_layers=static_context, attn_metadata={},
                             slot_mapping={}, cudagraph_runtime_mode=CUDAGraphMode.FULL,
                             batch_descriptor=descriptor)
    request_ids = [f"smoke-{index}" for index in range(args.live_rows)]
    scheduler = SimpleNamespace(
        num_scheduled_tokens={name: 1 for name in request_ids}, total_num_scheduled_tokens=args.live_rows,
        scheduled_cached_reqs=SimpleNamespace(req_ids=request_ids,
            num_computed_tokens=[4096] * args.live_rows, num_output_tokens=[1] * args.live_rows,
            resumed_req_ids=set()), scheduled_new_reqs=[], scheduled_spec_decode_tokens={},
        finished_req_ids=set(), preempted_req_ids=set())

    previous_worker = state.scopes.current_worker
    try:
        state.start()
        state.scopes._local.worker = worker
        with override_forward_context(context), torch.no_grad():
            for index in range(32):
                # Same ScopeHandle observer and callback used by WorkerFL's
                # wrapper, with dummy CPU scheduler data and current-worker TLS.
                metadata = state.scopes._observe(
                    "worker.execute_model", scheduler, "worker.execute_model.begin")
                require(metadata is not None, "Scheduler scope observation failed")
                try:
                    returned = wrapper(source)
                    require(returned is output, "Actual GraphWrapper returned a different output")
                finally:
                    state.scopes._observe_end("worker.execute_model.end", metadata)
                if index < 31:
                    require(not state.samples, "Sampling happened before replay 32")
        torch.cuda.synchronize()
        require(len(state.counts) == 1 and sum(state.counts.values()) == 32,
                f"Actual wrapper replay counts are wrong: {state.counts}")
        require(len(state.samples) == 1, f"Expected one sample, got {len(state.samples)}")
        require(len(state.steps) == 32, f"Expected 32 scheduler steps, got {len(state.steps)}")
        sample = state.samples[0]
        require(sample["graph_replay"] == 32 and sample["scheduler"]["step"] == 32,
                "Sample is not associated with the 32nd observed step")
        require(sample["descriptor"]["num_tokens"] == 64, "Wrong padded batch")
        audit = sample["output_alias_audit"]
        require(audit["validated"], "Output alias audit failed")
        counts = audit["protected_tensor_counts"]
        require(all(counts[key] >= 1 for key in
                    ("args", "parameters", "buffers", "kv_caches", "static_context_kv")),
                f"Incomplete real guard inventory: {counts}")
        control = sample["original_control"]
        require(all(control[key] is True for key in
                    ("original_poison_recovery", "numeric_exact", "finite")),
                "Original graph poison recovery failed")
        observations = {row["variant"]: row for row in sample["observations"]}
        require(set(observations) == {"boundary", "full"}, "Missing clone variant")
        for label, row in observations.items():
            require(all(row[key] is True for key in
                        ("output_poisoned", "numeric_exact", "finite", "all_api_calls_succeeded")),
                    f"{label} clone failed poison/numeric/API checks")
            require(math.isfinite(row["graph_total_ms"]) and row["graph_total_ms"] > 0,
                    f"{label} clone has invalid total timing")
        require(observations["boundary"]["kernels"] == [], "Boundary probe timed individual kernels")
        require(len(observations["full"]["kernels"]) == 2, "Expected exactly native+Triton kernel timings")
        full_metadata = state.probes[sample["key"]][1].metadata()
        nodes = [node for node in full_metadata["nodes"] if node["type"] == 0]
        require(len(nodes) == 2 and all(node["name"] for node in nodes),
                "Driver kernel names missing or unexpected graph topology")
        named_triton = [node for node in nodes if node["function_handle"] in state.kernel_loads]
        native = [node for node in nodes if node["function_handle"] not in state.kernel_loads]
        require(len(named_triton) == 1 and len(native) == 1,
                "Expected one driver-handle-matched Triton kernel and one native kernel")
        binding = state.kernel_loads[named_triton[0]["function_handle"]]
        require("actual_wrapper_smoke_finish" in binding["name"] and bool(binding["files"])
                and bool(binding["hash"]), "Triton source binding is incomplete")
        require(not state.errors and not state.binding_errors, "Diagnostic helper recorded errors")
        require(not state.scopes.snapshot()["errors"], "CPU scope helper recorded errors")
        expected = torch.full((64, 2048), 11.0, dtype=torch.bfloat16)
        require(torch.equal(output.cpu(), expected), "Final graph result is not the known exact output")
        state.stop()
        window = destination / f"worker-{os.getpid()}-window-{state.epoch}.json"
        saved = json.loads(window.read_text())
        require(sum(saved["graph_replay_counts"].values()) == 32 and len(saved["samples"]) == 1,
                "Exported window lost replay/sample evidence")
        require(len(saved["graphs"]) == 1 and saved["graphs"][0]["metadata"]["kernel_count"] == 2,
                "Exported graph metadata is incomplete")
        require(not state.probes and not state.active, "stop() did not close the owned probes")
        report = {
            "passed": True, "diagnostic_only": True, "scored_throughput": False,
            "window": str(window), "graph_replays": 32, "samples": 1,
            "wrapper_source": inspect.getfile(GraphWrapper),
            "forward_context_source": inspect.getfile(ForwardContext),
            "service_helper": str(Path(service.__file__).resolve()),
            "service_sha256": hashlib.sha256(Path(service.__file__).read_bytes()).hexdigest(),
            "original_poison_control": control, "guard_counts": counts,
            "validated_live_rows": audit["validated_live_rows"], "padded_shape": audit["padded_shape"],
            "kernel_names": [node["name"] for node in nodes],
            "triton_binding": binding,
        }
        print("BIV150_ACTUAL_WRAPPER_SMOKE " + json.dumps(report, sort_keys=True), flush=True)
    finally:
        state.scopes._local.worker = previous_worker
        if state.active:
            # Failure evidence remains visible; stop synchronizes and closes only
            # this helper's owned clones, never the caller-owned original graph.
            state.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-rows", type=int, choices=range(1,65), default=64)
    parser.add_argument("--run", action="store_true", help="Actually execute on the GPU; owner only")
    parser.add_argument("--tools-dir", default="/private/tmp/flagos-baseline-profiler-20261025/tools")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"prepared_only": True, "gpu_started": False,
                          "next": "GPU owner supplies --run --tools-dir SDK_HELPERS --output-dir NEW_DIR"}))
        return
    if not args.output_dir:
        parser.error("--run requires --output-dir")
    run(args)


if __name__ == "__main__":
    main()
