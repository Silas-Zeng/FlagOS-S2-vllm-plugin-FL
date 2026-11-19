"""Process-local diagnostic hooks. No production files or math are changed.

Loaded only by a private service sitecustomize. The normal graph remains the
default. A few predetermined replays compare it with boundary-only and fully
instrumented clones, including exact output checks and instrumentation cost.
Those marked diagnostic steps synchronize; ordinary steps never do so here.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import json
import os
from pathlib import Path
import threading
import time

_INSTALLED = False
_STATE = None


def bind_worker(worker):
    if _STATE is None:
        raise RuntimeError('Install diagnostic state before binding its worker')
    _STATE.worker = worker


def install():
    global _INSTALLED, _STATE
    destination = os.environ.get("BIV150_PROFILE_DIR")
    if not destination or _INSTALLED:
        return
    import torch
    from vllm_fl.compilation.graph import GraphWrapper
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.profiler.wrapper import CudaProfilerWrapper
    from biv150_graph_probe import CorexAPI, GraphProbe
    from biv150_profile_output_guard import HiddenStateOutputGuard
    import biv150_profile_scopes
    import biv150_profile_routes

    class State:
        def __init__(self):
            self.active = False
            self.epoch = 0
            self.local = threading.local()
            self.scopes = None
            self.api = None
            self.probes = {}
            self.kernel_loads = {}
            self.binding_errors = []
            self.worker = None

        def cpu_event(self, event, metadata):
            if not self.active:
                return
            if event == "worker.execute_model.begin":
                self.step += 1
                self.local.scheduler = dict(metadata, step=self.step)
                # The complete CPU metadata stays in RAM until the window ends.
                self.steps.append({**self.local.scheduler, "event": event,
                                   "monotonic_ns": time.monotonic_ns()})
            elif event == "worker.execute_model.end":
                self.local.scheduler = None

        def start(self):
            if self.active:
                raise RuntimeError("Diagnostic window already active")
            self.epoch += 1
            self.step = 0
            self.steps, self.samples, self.errors = [], [], []
            self.counts = collections.Counter()
            self.probes = {}
            self.routes_before = biv150_profile_routes.snapshot(self.worker)
            self.started_ns = time.monotonic_ns()
            self.active = True
            self.scopes.set_enabled(True)

        def stop(self):
            if not self.active:
                return
            self.active = False
            self.scopes.set_enabled(False)
            routes_after = biv150_profile_routes.snapshot(self.worker)
            # End-of-window synchronization, never an ordinary-step barrier.
            torch.cuda.synchronize()
            graphs = []
            for key, pair in self.probes.items():
                graphs.append(dict(key=key, metadata=pair[1].metadata()))
                for probe in pair:
                    probe.close(caller_synchronized=True)
            self.probes = {}
            result = dict(diagnostic_only=True, pid=os.getpid(), epoch=self.epoch,
                          started_monotonic_ns=self.started_ns,
                          finished_monotonic_ns=time.monotonic_ns(),
                          scheduled_steps=self.steps, samples=self.samples,
                          graph_replay_counts=dict(self.counts), graphs=graphs,
                          triton_kernel_bindings=list(self.kernel_loads.values()),
                          kernel_binding_errors=self.binding_errors,
                          scope_diagnostics=self.scopes.snapshot(), errors=self.errors,
                          production_routes_before=self.routes_before,
                          production_routes_after=routes_after,
                          selected_padded_rows=[32, 64],
                          selected_graph_replays=[32, 128, 512, 900],
                          scored_throughput=False,
                          limits=["Selected steps include duplicate execution, copies and synchronization.",
                                  "Exclude biv150_diag NVTX ranges when examining normal CPU gaps.",
                                  "Kernel event intervals include instrumentation and may overlap."])
            folder = Path(destination)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"worker-{os.getpid()}-window-{self.epoch}.json").open("x") as f:
                json.dump(result, f, indent=2)

        def sample(self, graph, old_replay, entry, key, count, descriptor, inputs):
            stream = torch.cuda.current_stream()
            with torch.cuda.nvtx.range("biv150_diag.sample"):
                if key not in self.probes:
                    with torch.cuda.nvtx.range("biv150_diag.prepare_clones"):
                        if self.api is None:
                            self.api = CorexAPI()
                        raw = graph.raw_cuda_graph()
                        order, _, types = self.api.snapshot(raw)
                        kernels = sum(types[node] == 0 for node in order)
                        if kernels == 0:
                            raise RuntimeError("No top-level kernels in service graph")
                        boundary = GraphProbe(raw, (), api=self.api)
                        try:
                            full = GraphProbe(raw, tuple(range(kernels)), api=self.api)
                        except BaseException:
                            boundary.close()
                            raise
                        self.probes[key] = boundary, full
                pair = self.probes[key]
                old_replay(graph)
                scheduler = getattr(self.local, "scheduler", None)
                if scheduler is None or scheduler["scheduled_token_sum"] != scheduler["total_num_scheduled_tokens"]:
                    raise RuntimeError("Missing or inconsistent live-row scheduler metadata")
                guard = HiddenStateOutputGuard(entry.output, worker=self.scopes.current_worker,
                                               graph_args=inputs[0], graph_kwargs=inputs[1],
                                               valid_rows=scheduler["total_num_scheduled_tokens"])
                control = guard.original_positive_control(
                    lambda: old_replay(graph), stream.synchronize)
                observations = []
                # Fixed order; this is diagnostic calibration, not a speed score.
                for label, probe in zip(("boundary", "full"), pair):
                    with torch.cuda.nvtx.range("biv150_diag." + label):
                        guard.poison_for_clone()
                        probe.replay(stream.cuda_stream)
                        stream.synchronize()
                        timing = probe.read_after_sync()
                        numeric = guard.assert_clone_after_sync()
                    observations.append(dict(variant=label, **numeric, **timing))
                self.samples.append(dict(key=key, graph_replay=count, descriptor=descriptor,
                                         scheduler=getattr(self.local, "scheduler", None),
                                         output_alias_audit=guard.audit,
                                         original_control=control,
                                         observations=observations))

    state = State()
    _STATE = state
    import triton
    def kernel_loaded(module, function, name, metadata_group, kernel_hash):
        # This real SDK callback is synchronous during binary loading. Retain
        # file names only; do not read/write files or touch the GPU here.
        try:
            state.kernel_loads[int(function)] = dict(
                module_handle=int(module), function_handle=int(function),
                name=str(name), hash=str(kernel_hash),
                files={str(k): os.fsdecode(os.fspath(v))
                       for k, v in metadata_group.items()})
        except Exception as exc:
            if len(state.binding_errors) < 16:
                state.binding_errors.append(repr(exc))
    triton.knobs.runtime.kernel_load_end_hook.add(kernel_loaded)
    state.load_hook = kernel_loaded
    state.scopes = biv150_profile_scopes.install(
        callback=state.cpu_event, engine=True, worker=True, enabled=False)

    # This installed build exposes keep_graph in BOTH the Python __new__ shim
    # and the pybind C++ __init__. Override both; changing only __new__ failed
    # the real smoke because the later C++ initializer receives default False.
    old_new = torch.cuda.CUDAGraph.__new__
    old_init = torch.cuda.CUDAGraph.__init__
    @functools.wraps(old_new)
    def graph_new(cls, *args, **kwargs):
        if args:
            args = (True, *args[1:])
            kwargs.pop("keep_graph", None)
        else:
            kwargs["keep_graph"] = True
        return old_new(cls, *args, **kwargs)
    torch.cuda.CUDAGraph.__new__ = staticmethod(graph_new)
    @functools.wraps(old_init)
    def graph_init(graph, *args, **kwargs):
        return old_init(graph, keep_graph=True)
    torch.cuda.CUDAGraph.__init__ = graph_init

    old_call = GraphWrapper.__call__
    @functools.wraps(old_call)
    def graph_call(wrapper, *args, **kwargs):
        previous = getattr(state.local, "entry", None)
        try:
            if state.active and is_forward_context_available():
                context = get_forward_context()
                descriptor = context.batch_descriptor
                if context.cudagraph_runtime_mode == wrapper.runtime_mode and descriptor is not None:
                    entry = wrapper.concrete_graph_entries.get(descriptor)
                    if entry is not None and entry.graph is not None:
                        state.local.entry = (entry, descriptor, (args, kwargs))
            return old_call(wrapper, *args, **kwargs)
        finally:
            state.local.entry = previous
    GraphWrapper.__call__ = graph_call

    old_replay = torch.cuda.CUDAGraph.replay
    @functools.wraps(old_replay)
    def replay(graph):
        context = getattr(state.local, "entry", None)
        if not state.active or context is None or context[0].graph is not graph:
            return old_replay(graph)
        entry, descriptor, inputs = context
        fields = dataclasses.asdict(descriptor)
        key = json.dumps(dict(graph_object=id(graph), descriptor=fields), sort_keys=True)
        state.counts[key] += 1
        count = state.counts[key]
        if descriptor.num_tokens not in (32, 64) or count not in (32, 128, 512, 900):
            return old_replay(graph)
        try:
            return state.sample(graph, old_replay, entry, key, count, fields, inputs)
        except BaseException as exc:
            state.errors.append(dict(key=key, replay=count, error=repr(exc)))
            raise  # Never silently accept a partial or numerically invalid probe.
    torch.cuda.CUDAGraph.replay = replay

    old_start, old_stop = CudaProfilerWrapper._start, CudaProfilerWrapper._stop
    @functools.wraps(old_start)
    def start(profiler):
        old_start(profiler)
        state.start()
    @functools.wraps(old_stop)
    def stop(profiler):
        try:
            state.stop()
        finally:
            old_stop(profiler)
    CudaProfilerWrapper._start, CudaProfilerWrapper._stop = start, stop
    _INSTALLED = True
    print("BIV150_DIAGNOSTIC_HOOK_INSTALLED " + json.dumps(
        dict(pid=os.getpid(), source=__file__, keep_graph_constructor="__new__+__init__",
             scopes=state.scopes.snapshot()["installation"])), flush=True)
