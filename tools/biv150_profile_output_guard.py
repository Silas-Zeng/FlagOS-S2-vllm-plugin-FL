"""Fail-closed output poisoning for MiniCPM final hidden-state diagnostics.

Import is inert. Construct only inside a selected diagnostic step after the
ordinary graph ran. The caller supplies that worker and the exact current
CUDAGraphWrapper args/kwargs. No tensor other than the validated output is ever
written here. CPU copies and synchronization belong only to diagnostic steps.

SDK fields verified: WorkerFL.model_runner; runner.model.parameters()/buffers();
runner.kv_caches list populated by vllm.v1.worker.utils.bind_kv_cache; the same
routine binds static_forward_context[layer].kv_cache. IntermediateTensors has a
dict[str, Tensor] field named tensors. Unknown argument containers are rejected.
"""
from __future__ import annotations


class OutputGuardError(RuntimeError):
    pass


def _overlap(left, right):
    # Full storage ranges, not just data_ptr: disjoint views of one storage are
    # deliberately rejected too. Range tests also catch SDK weak_ref_tensor
    # wrappers that share memory without sharing the Python Storage object.
    return left[0] == right[0] and left[1] < right[2] and right[1] < left[2]


class HiddenStateOutputGuard:
    def __init__(self, output, *, worker, graph_args, graph_kwargs, valid_rows=None):
        import torch
        from vllm.sequence import IntermediateTensors

        self.torch = torch
        self._expected = None
        self._original_passed = False
        self._clone_poisoned = False
        if not isinstance(output, torch.Tensor):
            raise OutputGuardError("Only one final hidden-state Tensor is supported")
        if (not output.is_cuda or output.dtype != torch.bfloat16 or
                tuple(output.shape) not in ((32, 2048), (64, 2048)) or
                not output.is_contiguous() or output.requires_grad):
            raise OutputGuardError("Expected contiguous CUDA BF16 hidden states [32|64,2048], without grad")
        if not isinstance(graph_args, tuple) or not isinstance(graph_kwargs, dict):
            raise OutputGuardError("Exact wrapper args tuple and kwargs dict are required")
        padded_shape = tuple(output.shape)
        if valid_rows is None:
            valid_rows = output.shape[0]
        if type(valid_rows) is not int or not 1 <= valid_rows <= output.shape[0]:
            raise OutputGuardError("Invalid live-row count for the captured output")
        # A padded graph can leave unused rows unspecified. Validate and poison
        # every live output row; do not interpret padding as model output.
        output = output[:valid_rows]
        self.output = output
        self._identity = (int(output.data_ptr()), tuple(output.shape), tuple(output.stride()))

        def storage(tensor):
            if not isinstance(tensor, torch.Tensor):
                raise OutputGuardError("Protected field is not a Tensor")
            if tensor.device.type == "meta":
                raise OutputGuardError("Cannot verify meta-tensor storage")
            memory = tensor.untyped_storage()
            begin, size = int(memory.data_ptr()), int(memory.nbytes())
            if tensor.numel() and (begin == 0 or size <= 0):
                raise OutputGuardError("Nonempty tensor has unverifiable storage")
            return str(tensor.device), begin, begin + size

        self._storage = storage
        self.output_storage = storage(output)
        if self.output_storage[2] <= self.output_storage[1]:
            raise OutputGuardError("Output storage is empty")
        self._protected = []
        counts = {"args": 0, "parameters": 0, "buffers": 0, "kv_caches": 0,
                  "static_context_kv": 0}

        def protect(tensor, path, category):
            interval = storage(tensor)
            if _overlap(self.output_storage, interval):
                raise OutputGuardError(f"Output aliases protected storage: {path}")
            # Retain protected tensors so their allocations cannot disappear
            # between this audit and the selected poison/replay checks.
            self._protected.append(tensor)
            counts[category] += 1
            return interval

        visited = set()
        def walk(value, path):
            if isinstance(value, torch.Tensor):
                protect(value, path, "args")
            elif value is None or type(value) in (bool, int, float, str, bytes):
                return
            elif isinstance(value, (torch.dtype, torch.device)):
                return
            elif isinstance(value, IntermediateTensors):
                if not isinstance(value.tensors, dict):
                    raise OutputGuardError("IntermediateTensors.tensors is not a dict")
                walk(value.tensors, path + ".tensors")
            elif isinstance(value, (tuple, list, dict)):
                if id(value) in visited:
                    return
                visited.add(id(value))
                if isinstance(value, dict):
                    for key, item in value.items():
                        if type(key) not in (str, int):
                            raise OutputGuardError(f"Unverifiable argument key at {path}")
                        walk(item, f"{path}[{key!r}]")
                else:
                    for index, item in enumerate(value):
                        walk(item, f"{path}[{index}]")
            else:
                raise OutputGuardError(f"Unknown argument container at {path}: {type(value)!r}")

        walk(graph_args, "args")
        walk(graph_kwargs, "kwargs")
        if counts["args"] == 0:
            raise OutputGuardError("No current graph input tensors were available to audit")
        runner = getattr(worker, "model_runner", None)
        model = getattr(runner, "model", None)
        if runner is None or model is None:
            raise OutputGuardError("Worker model_runner/model is unavailable")
        if not callable(getattr(model, "parameters", None)) or not callable(getattr(model, "buffers", None)):
            raise OutputGuardError("Model does not expose complete parameters()/buffers()")
        for index, tensor in enumerate(model.parameters()):
            protect(tensor, f"model.parameters[{index}]", "parameters")
        for index, tensor in enumerate(model.buffers()):
            protect(tensor, f"model.buffers[{index}]", "buffers")
        if counts["parameters"] == 0:
            raise OutputGuardError("Empty model parameter inventory cannot establish safety")

        caches = getattr(runner, "kv_caches", None)
        if not isinstance(caches, list) or not caches:
            raise OutputGuardError("runner.kv_caches must be a bound nonempty Tensor list")
        cache_intervals = set()
        for index, tensor in enumerate(caches):
            if not isinstance(tensor, torch.Tensor) or not tensor.numel():
                raise OutputGuardError(f"Unsupported or empty KV cache at index {index}")
            cache_intervals.add(protect(tensor, f"runner.kv_caches[{index}]", "kv_caches"))
        context = getattr(getattr(runner, "compilation_config", None), "static_forward_context", None)
        if not isinstance(context, dict) or not context:
            raise OutputGuardError("Static forward context unavailable; KV completeness unverified")
        for name, layer in context.items():
            if not hasattr(layer, "kv_cache"):
                continue
            tensor = layer.kv_cache
            if not isinstance(tensor, torch.Tensor) or not tensor.numel():
                raise OutputGuardError(f"Unbound or unsupported static KV cache: {name}")
            interval = protect(tensor, f"static_forward_context[{name!r}].kv_cache", "static_context_kv")
            if interval not in cache_intervals:
                raise OutputGuardError(f"Static KV cache missing from runner inventory: {name}")
        if counts["static_context_kv"] != counts["kv_caches"]:
            raise OutputGuardError("Static context and runner KV cache counts disagree")
        self.audit = {"validated": True, "shape": list(output.shape), "dtype": str(output.dtype),
                      "padded_shape": list(padded_shape), "validated_live_rows": valid_rows,
                      "device": str(output.device), "output_storage": list(self.output_storage),
                      "protected_tensor_counts": counts,
                      "scope": "current wrapper inputs, registered model state, bound KV caches"}

    def _check_identity(self):
        current = (int(self.output.data_ptr()), tuple(self.output.shape), tuple(self.output.stride()))
        if current != self._identity or self._storage(self.output) != self.output_storage:
            raise OutputGuardError("Output identity/storage changed after safety audit")

    def _snapshot(self):
        self._check_identity()
        value = self.output.detach().cpu().clone()
        if not bool(self.torch.isfinite(value).all()):
            raise OutputGuardError("Final hidden-state output is non-finite")
        return value

    def _poison(self):
        self._check_identity()
        with self.torch.no_grad():
            self.output.fill_(float("nan"))

    def original_positive_control(self, replay, synchronize):
        """Save unpoisoned original output, poison, rerun original, prove exact recovery.

        replay and synchronize are caller callbacks. Use the same current stream
        for poison, graph launch and synchronization; selected diagnostics only.
        """
        if self._expected is not None:
            raise OutputGuardError("Original positive control was already attempted")
        self._expected = self._snapshot()
        self._poison()
        replay()
        synchronize()
        actual = self._snapshot()
        if not self.torch.equal(actual, self._expected):
            raise OutputGuardError("Original graph failed exact recovery after output poisoning")
        self._original_passed = True
        return {"original_poison_recovery": True, "numeric_exact": True, "finite": True}

    def poison_for_clone(self):
        if not self._original_passed or self._clone_poisoned:
            raise OutputGuardError("Clone poison requires successful original control and no pending check")
        self._poison()
        self._clone_poisoned = True

    def assert_clone_after_sync(self):
        """Caller must synchronize clone replay first; CPU copy is diagnostic only."""
        if not self._original_passed or not self._clone_poisoned:
            raise OutputGuardError("Clone was not poisoned after a successful original control")
        actual = self._snapshot()
        if not self.torch.equal(actual, self._expected):
            raise OutputGuardError("Instrumented clone failed finite exact recovery of poisoned output")
        self._clone_poisoned = False
        return {"output_poisoned": True, "numeric_exact": True, "finite": True}

