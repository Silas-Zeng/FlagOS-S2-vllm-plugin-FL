"""Process-local CPU/NVTX scopes for the installed BI-V150 vLLM diagnostic.

Import is inert: no Torch/vLLM import, GPU work, environment change or I/O.
Call install(engine=True, worker=False) in EngineCore's process, and
install(engine=False, worker=True) in the FL worker process after its normal
modules are importable. With a shared process use both True. Installation must
precede construction of EngineCore (its step_fn may retain a bound method).
Call handle.set_enabled(True/False) outside the desired diagnostic window.
Callbacks run synchronously on CPU; the caller must keep them bounded and avoid
I/O, GPU access or synchronization. Flush their data after the window. No helper
method changes server/benchmark arguments, scheduler data, returns or exceptions.
NVTX ranges show host execution; their widths are not GPU kernel durations.

Existing imported aliases of record_function_or_nullcontext remain valid: that
function reads vllm.v1.utils._PROFILER_FUNC on every call. This bypasses the SDK's
optional third-party nvtx import using torch.cuda.nvtx.range_push/range_pop.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import os
import threading


_ACTIVE_HANDLE = None


def scheduler_metadata(output):
    """Copy CPU fields only. Cached/new membership is NOT a decode classifier."""
    scheduled = dict(output.num_scheduled_tokens)
    cached = output.scheduled_cached_reqs
    new = output.scheduled_new_reqs
    req_ids = list(cached.req_ids)
    # These lists are CPU metadata in this exact SDK, not device tensors.
    computed = list(cached.num_computed_tokens)
    output_counts = list(cached.num_output_tokens)
    if len(req_ids) != len(computed) or len(req_ids) != len(output_counts):
        raise ValueError("CachedRequestData CPU metadata lengths disagree")
    return {
        "total_num_scheduled_tokens": int(output.total_num_scheduled_tokens),
        "num_scheduled_tokens": {str(k): int(v) for k, v in scheduled.items()},
        "scheduled_request_count": len(scheduled),
        "scheduled_token_sum": sum(int(v) for v in scheduled.values()),
        "new_request_count": len(new),
        "cached_request_count": len(req_ids),
        "new_requests": [
            {"req_id": str(req.req_id),
             "num_computed_tokens": int(req.num_computed_tokens),
             "prompt_token_count": (len(req.prompt_token_ids)
                                    if req.prompt_token_ids is not None else None)}
            for req in new
        ],
        "cached_requests": [
            {"req_id": str(req_id), "num_computed_tokens": int(n_computed),
             "num_output_tokens": int(n_output),
             "resumed": req_id in cached.resumed_req_ids}
            for req_id, n_computed, n_output in zip(req_ids, computed, output_counts)
        ],
        "speculative_token_counts": {
            str(k): len(v) for k, v in output.scheduled_spec_decode_tokens.items()
        },
        "finished_request_count": len(output.finished_req_ids),
        "preempted_request_count": len(output.preempted_req_ids or ()),
        "phase_inferred": False,
    }


class _Range:
    __slots__ = ("handle", "name", "pushed")

    def __init__(self, handle, name):
        self.handle, self.name, self.pushed = handle, name, False

    def __enter__(self):
        if self.handle.enabled:
            try:
                self.handle._push(self.name)
                self.pushed = True
            except Exception as exc:
                self.handle._error("nvtx_push", exc)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.pushed:
            try:
                self.handle._pop()
            except Exception as failure:
                self.handle._error("nvtx_pop", failure)
        return False


class ScopeHandle:
    def __init__(self, nvtx, *, enabled, callback, on_scheduler, select_scheduler):
        self.enabled = bool(enabled)
        self._local = threading.local()
        self.callback = callback
        self.on_scheduler = on_scheduler
        self.select_scheduler = select_scheduler
        self._push, self._pop = nvtx.range_push, nvtx.range_pop
        self._patches = []
        self._counts = {}
        self._errors = []
        self._error_total = 0
        self._callback_disabled = False
        self._pid = os.getpid()
        self._uninstalled = False
        self.installation = {"pid": self._pid, "diagnostic_only": True,
                             "sources": [], "patched": []}

    def _error(self, where, exc):
        # Keep a bounded record; never log or mask application exceptions.
        self._error_total += 1
        if len(self._errors) < 16:
            self._errors.append({"where": where, "type": type(exc).__name__,
                                 "message": str(exc)[:300]})

    def range(self, name):
        return _Range(self, name)

    @property
    def current_worker(self):
        return getattr(self._local, "worker", None)

    def set_enabled(self, enabled):
        if self._uninstalled:
            raise RuntimeError("Scopes have already been uninstalled")
        self.enabled = bool(enabled)

    def _observe(self, source, output, event):
        if not self.enabled or (self.on_scheduler is None and self.callback is None) or self._callback_disabled:
            return None
        sequence = self._counts.get(source, 0) + 1
        self._counts[source] = sequence
        try:
            # Selector sees CPU object by reference; it must never mutate it.
            if self.select_scheduler is not None and not self.select_scheduler(source, sequence, output):
                return None
            value = scheduler_metadata(output)
            value.update(source=source, sequence=sequence, pid=self._pid)
            if self.on_scheduler is not None:
                self.on_scheduler(value)
            if self.callback is not None:
                self.callback(event, value)
            return value
        except Exception as exc:
            self._error("scheduler_callback", exc)
            self._callback_disabled = True
            return None

    def _observe_end(self, event, metadata):
        # Pair a delivered begin even if the window was disabled mid-call.
        if metadata is not None and self.callback is not None:
            try:
                self.callback(event, metadata)
            except Exception as exc:
                self._error("scheduler_callback_end", exc)
                self._callback_disabled = True

    def _patch(self, owner, name, replacement, description):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        self._patches.append((owner, name, original, replacement))
        self.installation["patched"].append(description)

    def _method(self, cls, name, label, *, observe_result=False, observe_argument=False):
        original = getattr(cls, name)
        handle = self

        @functools.wraps(original)
        def wrapped(instance, *args, **kwargs):
            with handle.range(label):
                metadata = None
                previous_worker = handle.current_worker
                if observe_argument:
                    handle._local.worker = instance
                    # The SDK's first non-self argument is scheduler_output.
                    output = args[0] if args else kwargs.get("scheduler_output")
                    metadata = handle._observe(label, output, label + ".begin")
                try:
                    result = original(instance, *args, **kwargs)
                    if observe_result:
                        handle._observe(label, result, label + ".end")
                    return result
                finally:
                    if observe_argument:
                        try:
                            handle._observe_end(label + ".end", metadata)
                        finally:
                            handle._local.worker = previous_worker

        self._patch(cls, name, wrapped, f"{cls.__module__}.{cls.__name__}.{name}")

    def _wait_context(self, cls):
        original = cls.log_iteration_details
        handle = self

        @functools.wraps(original)
        @contextlib.contextmanager
        def wrapped(instance, *args, **kwargs):
            # This SDK brackets future.result() and possible sample_tokens()
            # with log_iteration_details in both EngineCore step variants.
            with handle.range("engine.wait_for_output_plus_sampling"):
                with original(instance, *args, **kwargs) as value:
                    yield value

        self._patch(cls, "log_iteration_details", wrapped,
                    f"{cls.__module__}.{cls.__name__}.log_iteration_details")

    def snapshot(self):
        """Read CPU diagnostics after the window; no I/O or synchronization."""
        return {"installation": dict(self.installation), "enabled": self.enabled,
                "observation_counts": dict(self._counts), "errors": list(self._errors),
                "error_total": self._error_total,
                "callback_disabled": self._callback_disabled}

    def uninstall(self):
        """Call outside active scopes. Preserve a later third-party replacement."""
        global _ACTIVE_HANDLE
        self.enabled = False
        conflicts = []
        for owner, name, original, replacement in reversed(self._patches):
            if getattr(owner, name) is replacement:
                setattr(owner, name, original)
            else:
                conflicts.append(name)
        self._patches.clear()
        self._uninstalled = True
        if _ACTIVE_HANDLE is self:
            _ACTIVE_HANDLE = None
        return {"restored": not conflicts, "replacement_conflicts": conflicts}


def install(callback=None, *, engine=True, worker=True, enabled=False, on_scheduler=None,
            select_scheduler=None, modules=None, nvtx=None):
    """Install in THIS process; does not automatically propagate to children.

    callback(event, metadata_dict) receives worker.execute_model.begin/end and
    scheduler.schedule.end. Begin is before the original method; end is paired
    in finally even if the method raises. It never suppresses that exception.
    on_scheduler(metadata_dict) is an optional one-argument observation hook.
    select_scheduler(source, sequence, scheduler_output)
    can bound collection; return False to avoid copying metadata. `modules` and
    `nvtx` allow CPU-only tests and explicit already-loaded modules.
    """
    global _ACTIVE_HANDLE
    if _ACTIVE_HANDLE is not None:
        raise RuntimeError("A ScopeHandle is already installed in this process")
    modules = modules or {}

    def load(name):
        return modules[name] if name in modules else importlib.import_module(name)

    if nvtx is None:
        nvtx = load("torch").cuda.nvtx
    handle = ScopeHandle(nvtx, enabled=enabled, callback=callback, on_scheduler=on_scheduler,
                         select_scheduler=select_scheduler)
    try:
        utils = load("vllm.v1.utils")
        if not hasattr(utils, "_PROFILER_FUNC"):
            raise RuntimeError("SDK lacks the verified _PROFILER_FUNC hook")
        # Imported aliases still execute this module's global lookup.
        handle._patch(utils, "_PROFILER_FUNC", handle.range, "vllm.v1.utils._PROFILER_FUNC")
        handle.installation["sources"].append(getattr(utils, "__file__", "<injected>"))
        if engine:
            core = load("vllm.v1.engine.core")
            scheduler = load("vllm.v1.core.sched.scheduler")
            for name in ("step", "step_with_batch_queue", "post_step", "_process_aborts_queue"):
                handle._method(core.EngineCore, name, "engine." + name)
            handle._wait_context(core.EngineCore)
            handle._method(scheduler.Scheduler, "schedule", "scheduler.schedule", observe_result=True)
            for name in ("update_from_output", "get_grammar_bitmask"):
                handle._method(scheduler.Scheduler, name, "scheduler." + name)
            handle.installation["sources"].extend(
                [getattr(core, "__file__", "<injected>"), getattr(scheduler, "__file__", "<injected>")])
        if worker:
            plugin = load("vllm_fl.worker.worker")
            handle._method(plugin.WorkerFL, "execute_model", "worker.execute_model", observe_argument=True)
            handle._method(plugin.WorkerFL, "sample_tokens", "worker.sample_tokens")
            handle.installation["sources"].append(getattr(plugin, "__file__", "<injected>"))
        _ACTIVE_HANDLE = handle
        return handle
    except BaseException:
        handle.uninstall()
        raise
