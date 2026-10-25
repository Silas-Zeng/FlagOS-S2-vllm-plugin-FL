"""Bounded Corex CUDA-graph diagnostics; never mutates the caller's graph.

GraphProbe clones one caller-owned raw graph. Kernel indices are indices in the
original graph's topological kernel list, described by ``metadata()``. The
default empty subset measures only graph-wide device elapsed time. Selecting a
subset adds two event-record nodes per selected kernel, plus two total events.

No method synchronizes, prints, writes files, or imports Torch. The caller owns
GPU exclusion, the context, original graph/tensor lifetime, numeric validation,
and synchronization. Only one unread replay is permitted per probe. Keep each
probe alive until that replay has completed, then read and close it explicitly.

All timing is diagnostic: inserted graph nodes/dependencies affect scheduling.
Kernel intervals on independent branches can overlap; never sum them as wall
time. The total interval is recorded INSIDE the graph, excluding host gaps
between launch API calls. Cross-stream work outside the graph is not covered.

ABI sources: Corex 4.5.0 ix_runtime_api.h, driver_types.h, ix.h. Function-name
mapping uses driver ixdrvGraphKernelNodeGetParams -> ixdrvFuncGetName, never the
runtime ixFuncGetName API that expects a host symbol.
"""
from __future__ import annotations

import copy
import ctypes as C
import math

P = C.c_void_p
PP = C.POINTER(P)
Z = C.c_size_t
ZP = C.POINTER(Z)
I = C.c_int


class KernelNodeParams(C.Structure):
    _fields_ = [
        ("func", P),
        ("gridDimX", C.c_uint), ("gridDimY", C.c_uint), ("gridDimZ", C.c_uint),
        ("blockDimX", C.c_uint), ("blockDimY", C.c_uint), ("blockDimZ", C.c_uint),
        ("sharedMemBytes", C.c_uint), ("kernelParams", PP), ("extra", PP),
    ]


NODE_TYPES = {0: "kernel", 1: "memcpy", 2: "memset", 3: "host",
              4: "child_graph", 5: "empty", 6: "event_wait", 7: "event_record",
              10: "mem_alloc", 11: "mem_free", 13: "conditional"}
RUNTIME_SIGNATURES = {
    "ixGraphGetNodes": [P, PP, ZP],
    "ixGraphGetEdges": [P, PP, PP, ZP],
    "ixGraphNodeGetDependencies": [P, PP, ZP],
    "ixGraphNodeGetDependentNodes": [P, PP, ZP],
    "ixGraphNodeGetType": [P, C.POINTER(I)],
    "ixGraphClone": [PP, P],
    "ixGraphNodeFindInClone": [PP, P, P],
    "ixGraphAddEventRecordNode": [PP, P, PP, Z, P],
    "ixGraphAddDependencies": [P, PP, PP, Z],
    "ixGraphInstantiateWithFlags": [PP, P, C.c_ulonglong],
    "ixGraphLaunch": [P, P],
    "ixGraphExecDestroy": [P],
    "ixGraphDestroy": [P],
    "ixEventCreateWithFlags": [PP, C.c_uint],
    "ixEventQuery": [P],
    "ixEventElapsedTime": [C.POINTER(C.c_float), P, P],
    "ixEventDestroy": [P],
}


class ProbeError(RuntimeError):
    """Explicit API or graph-invariant failure; partial data is not a result."""


class APIError(ProbeError):
    def __init__(self, api, return_code, context=None):
        self.api = api
        self.return_code = int(return_code)
        self.context = context
        super().__init__(f"{api} returned {return_code}; context={context!r}")


def _topological(nodes, edges):
    incoming = {node: 0 for node in nodes}
    outgoing = {node: [] for node in nodes}
    position = {node: index for index, node in enumerate(nodes)}
    for a, b in edges:
        if a not in incoming or b not in incoming:
            raise ProbeError("Edge references a node absent from enumeration")
        incoming[b] += 1
        outgoing[a].append(b)
    ready = [node for node in nodes if incoming[node] == 0]
    result = []
    while ready:
        ready.sort(key=position.__getitem__)
        node = ready.pop(0)
        result.append(node)
        for successor in outgoing[node]:
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
    if len(result) != len(nodes):
        raise ProbeError("Graph is cyclic")
    return result


class CorexAPI:
    """ctypes bindings only; construction does not initialize a device context."""
    def __init__(self, runtime_path="/usr/local/corex/lib64/libcudart.so",
                 driver_path="/usr/local/corex/lib64/libcuda.so"):
        if C.sizeof(P) != 8 or C.sizeof(KernelNodeParams) != 56:
            raise ProbeError("Only the verified Corex 64-bit ABI is supported")
        expected = {"func": 0, "gridDimX": 8, "blockDimX": 20,
                    "sharedMemBytes": 32, "kernelParams": 40, "extra": 48}
        if any(getattr(KernelNodeParams, k).offset != v for k, v in expected.items()):
            raise ProbeError("KernelNodeParams layout differs from installed Corex ABI")
        self.runtime_path, self.driver_path = runtime_path, driver_path
        self.runtime = C.CDLL(runtime_path)
        self.driver = C.CDLL(driver_path)
        self.functions = {}
        for name, args in RUNTIME_SIGNATURES.items():
            self._bind(self.runtime, name, args)
        self._bind(self.driver, "ixdrvGraphKernelNodeGetParams", [P, C.POINTER(KernelNodeParams)])
        self._bind(self.driver, "ixdrvFuncGetName", [C.POINTER(C.c_char_p), P])

    def _bind(self, lib, name, args):
        fn = getattr(lib, name)  # AttributeError exposes a missing actual symbol.
        fn.argtypes, fn.restype = args, I
        self.functions[name] = fn

    def call(self, name, *args, context=None):
        rc = int(self.functions[name](*args))
        if rc:
            raise APIError(name, rc, context)

    def handles(self, name, owner):
        count = Z()
        self.call(name, P(owner), None, C.byref(count))
        if count.value > 1000000:
            raise ProbeError("Graph enumeration exceeded defensive bound")
        if not count.value:
            return []
        buf = (P * count.value)()
        capacity = count.value
        self.call(name, P(owner), buf, C.byref(count))
        if count.value > capacity:
            raise ProbeError("Graph changed during enumeration")
        values = [int(buf[i] or 0) for i in range(count.value)]
        if not all(values) or len(values) != len(set(values)):
            raise ProbeError("Null or duplicate graph node handle")
        return values

    def edges(self, graph):
        count = Z()
        self.call("ixGraphGetEdges", P(graph), None, None, C.byref(count))
        if count.value > 10000000:
            raise ProbeError("Graph edge enumeration exceeded defensive bound")
        if not count.value:
            return []
        capacity = count.value
        src, dst = (P * capacity)(), (P * capacity)()
        self.call("ixGraphGetEdges", P(graph), src, dst, C.byref(count))
        if count.value > capacity:
            raise ProbeError("Graph changed during edge enumeration")
        edges = [(int(src[i] or 0), int(dst[i] or 0)) for i in range(count.value)]
        if len(edges) != len(set(edges)):
            raise ProbeError("Duplicate graph edges")
        return edges

    def snapshot(self, graph):
        nodes = self.handles("ixGraphGetNodes", graph)
        if not nodes:
            raise ProbeError("Empty graph cannot produce useful timing")
        edges = self.edges(graph)
        order = _topological(nodes, edges)
        types = {}
        for node in order:
            kind = I(-1)
            self.call("ixGraphNodeGetType", P(node), C.byref(kind), context=node)
            types[node] = kind.value
        return order, edges, types

    def kernel_metadata(self, node):
        params = KernelNodeParams()
        self.call("ixdrvGraphKernelNodeGetParams", P(node), C.byref(params), context=node)
        if not params.func:
            raise ProbeError("Kernel params returned a null driver function")
        name = C.c_char_p()
        self.call("ixdrvFuncGetName", C.byref(name), P(params.func), context=node)
        if not name.value:
            raise ProbeError("Driver kernel name is empty")
        return {"function_handle": int(params.func), "name": name.value.decode(errors="replace"),
                "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
                "block": [params.blockDimX, params.blockDimY, params.blockDimZ],
                "shared_mem_bytes": params.sharedMemBytes,
                "kernel_params_pointer": C.cast(params.kernelParams, P).value,
                "extra_pointer": C.cast(params.extra, P).value}


class GraphProbe:
    """One cloned graph with optional top-level kernel timing and total timing.

    ``kernel_indices=()`` adds only graph-wide timestamps. Indices refer to the
    original graph's topological kernel list, not all-node indices. Child graphs
    remain intact and are included in total time but are not decomposed.

    Build outside timed serving windows. ``replay(stream_handle)`` enqueues once;
    after caller synchronization, ``read_after_sync()`` returns timings in ms.
    ``close(caller_synchronized=True)`` is useful in a caller's failure cleanup,
    but that assertion is truthful only after the caller actually synchronized.
    """
    def __init__(self, raw_graph, kernel_indices=(), *, api=None):
        self.api = api if api is not None else CorexAPI()
        self.raw_graph = int(raw_graph)
        if not self.raw_graph:
            raise ValueError("raw_graph must be a live non-null caller-owned graph")
        selected = tuple(kernel_indices)
        if any(type(v) is not int or v < 0 for v in selected) or len(set(selected)) != len(selected):
            raise ValueError("kernel_indices must contain unique nonnegative integers")
        self.clone = None
        self.executable = None
        self.events = []
        self.probes = []
        self.total_start = self.total_end = None
        self._pending = False
        self._launch_failed = False
        self._closed = False
        self._replay_count = 0
        self._metadata = {}
        try:
            self._prepare(selected)
        except BaseException as exc:
            try:
                self.close(caller_synchronized=True)
            except BaseException as cleanup_exc:
                exc.add_note(f"GraphProbe constructor cleanup failed: {cleanup_exc!r}")
            raise

    def _event(self):
        event = P()
        self.api.call("ixEventCreateWithFlags", C.byref(event), C.c_uint(0))
        if not event.value:
            raise ProbeError("Event creation returned null")
        self.events.append(event.value)
        return event.value

    def _event_node(self, dependencies, event):
        deps = (P * len(dependencies))(*dependencies) if dependencies else None
        node = P()
        self.api.call("ixGraphAddEventRecordNode", C.byref(node), P(self.clone), deps,
                      Z(len(dependencies)), P(event))
        if not node.value:
            raise ProbeError("Event node creation returned null")
        return node.value

    def _add_edges(self, edges):
        if edges:
            src = (P * len(edges))(*(a for a, _ in edges))
            dst = (P * len(edges))(*(b for _, b in edges))
            self.api.call("ixGraphAddDependencies", P(self.clone), src, dst, Z(len(edges)))

    def _prepare(self, selected):
        api = self.api
        order, original_edges, types = api.snapshot(self.raw_graph)
        kernels = [node for node in order if types[node] == 0]
        if selected and max(selected) >= len(kernels):
            raise ValueError(f"kernel index out of bounds; graph has {len(kernels)} kernels")
        selected_set = set(selected)
        kernel_index = {node: index for index, node in enumerate(kernels)}
        node_index = {node: index for index, node in enumerate(order)}
        node_metadata = []
        for node in order:
            entry = {"node_index": node_index[node], "original_handle": node,
                     "type": types[node], "type_name": NODE_TYPES.get(types[node], "unknown"),
                     "dependencies": [node_index[a] for a, b in original_edges if b == node],
                     "successors": [node_index[b] for a, b in original_edges if a == node]}
            if types[node] == 0:
                entry.update(api.kernel_metadata(node))
                entry["kernel_index"] = kernel_index[node]
                entry["selected"] = kernel_index[node] in selected_set
            node_metadata.append(entry)
        clone = P()
        api.call("ixGraphClone", C.byref(clone), P(self.raw_graph))
        if not clone.value:
            raise ProbeError("Graph clone returned null")
        self.clone = clone.value
        mapping = {}
        for entry in node_metadata:
            mapped = P()
            api.call("ixGraphNodeFindInClone", C.byref(mapped), P(entry["original_handle"]), P(self.clone))
            if not mapped.value:
                raise ProbeError("Node mapping into clone returned null")
            mapping[entry["original_handle"]] = mapped.value
            entry["clone_handle"] = mapped.value
        cloned_order, cloned_edges, cloned_types = api.snapshot(self.clone)
        expected_edges = {(mapping[a], mapping[b]) for a, b in original_edges}
        if set(cloned_order) != set(mapping.values()) or set(cloned_edges) != expected_edges:
            raise ProbeError("Cloned graph structure differs from original")
        if any(cloned_types[mapping[node]] != types[node] for node in order):
            raise ProbeError("Cloned node type differs from original")

        self.total_start, self.total_end = self._event(), self._event()
        start_node = self._event_node([], self.total_start)
        destinations = {b for _, b in cloned_edges}
        roots = [node for node in cloned_order if node not in destinations]
        self._add_edges([(start_node, node) for node in roots])
        for original in kernels:  # Original topological order, regardless of caller list order.
            index = kernel_index[original]
            if index not in selected_set:
                continue
            node = mapping[original]
            predecessors = api.handles("ixGraphNodeGetDependencies", node)
            successors = api.handles("ixGraphNodeGetDependentNodes", node)
            begin, end = self._event(), self._event()
            begin_node = self._event_node(predecessors, begin)
            self._add_edges([(begin_node, node)])
            end_node = self._event_node([node], end)
            self._add_edges([(end_node, successor) for successor in successors])
            self.probes.append({"kernel_index": index, "node": node, "start": begin, "end": end,
                                "start_node": begin_node, "end_node": end_node})
        current_nodes = api.handles("ixGraphGetNodes", self.clone)
        current_edges = api.edges(self.clone)
        sources = {a for a, _ in current_edges}
        leaves = [node for node in current_nodes if node not in sources]
        end_node = self._event_node(leaves, self.total_end)
        final_order, final_edges, final_types = api.snapshot(self.clone)
        if not expected_edges.issubset(set(final_edges)):
            raise ProbeError("Instrumentation lost an original edge")
        if len(final_order) != len(order) + 2 * len(selected) + 2:
            raise ProbeError("Unexpected instrumented node count")
        added_event_nodes = [start_node, end_node]
        for probe in self.probes:
            added_event_nodes.extend([probe["start_node"], probe["end_node"]])
        if any(final_types[node] != 7 for node in added_event_nodes):
            raise ProbeError("Added timing node is not an event-record node")
        final_destinations, final_sources = {b for _, b in final_edges}, {a for a, _ in final_edges}
        if {v for v in final_order if v not in final_destinations} != {start_node}:
            raise ProbeError("Graph total-start event is not the unique root")
        if {v for v in final_order if v not in final_sources} != {end_node}:
            raise ProbeError("Graph total-end event is not the unique leaf")
        executable = P()
        api.call("ixGraphInstantiateWithFlags", C.byref(executable), P(self.clone), C.c_ulonglong(0))
        if not executable.value:
            raise ProbeError("Instantiation returned null executable")
        self.executable = executable.value
        self._metadata = {
            "diagnostic_only": True, "original_graph_unchanged": True,
            "raw_graph": self.raw_graph, "clone_graph": self.clone,
            "original_edges_retained": True, "nodes": node_metadata,
            "original_edges": [[node_index[a], node_index[b]] for a, b in original_edges],
            "instrumented_edges_handles": final_edges,
            "original_node_count": len(order), "instrumented_node_count": len(final_order),
            "kernel_count": len(kernels), "selected_kernel_indices": sorted(selected),
            "undecomposed_child_graphs": sum(types[node] == 4 for node in order),
            "total_start_node": start_node, "total_end_node": end_node,
            "intervals_may_overlap": True,
        }

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def replay(self, stream_handle):
        """Enqueue only. No allocation, synchronization, output, or trace export."""
        if self._closed or self.executable is None:
            raise ProbeError("Probe is closed or uninitialized")
        if self._pending:
            raise ProbeError("Previous replay is unread; events would be overwritten")
        self._pending = True
        try:
            self.api.call("ixGraphLaunch", P(self.executable), P(int(stream_handle)))
        except BaseException:
            self._launch_failed = True
            raise
        self._replay_count += 1

    def _elapsed(self, begin, end):
        value = C.c_float(float("nan"))
        self.api.call("ixEventElapsedTime", C.byref(value), P(begin), P(end))
        result = float(value.value)
        if not math.isfinite(result) or result < 0:
            raise ProbeError(f"Invalid event elapsed time {result}")
        return result

    def read_after_sync(self):
        """Read the last replay after caller synchronization; never waits itself."""
        if self._closed or not self._pending or self._launch_failed:
            raise ProbeError("No successfully enqueued unread replay")
        # This is only an additional API/error check, NOT proof of completion:
        # a graph event may still report a previous replay's successful record
        # while the current launch is queued. Caller synchronization is required.
        self.api.call("ixEventQuery", P(self.total_end), context="caller must synchronize before read")
        total_ms = self._elapsed(self.total_start, self.total_end)
        if total_ms <= 0:
            raise ProbeError("Graph-wide elapsed time is nonpositive")
        records = []
        for probe in self.probes:
            start_ms = self._elapsed(self.total_start, probe["start"])
            end_ms = self._elapsed(self.total_start, probe["end"])
            elapsed_ms = self._elapsed(probe["start"], probe["end"])
            tolerance = max(0.02, total_ms * 1e-5)
            if end_ms + tolerance < start_ms or end_ms > total_ms + tolerance:
                raise ProbeError("Kernel event interval lies outside total graph interval")
            if abs((end_ms - start_ms) - elapsed_ms) > tolerance:
                raise ProbeError("Relative event offsets disagree with measured kernel interval")
            records.append({"kernel_index": probe["kernel_index"], "start_ms": start_ms,
                            "end_ms": end_ms, "elapsed_ms": elapsed_ms})
        self._pending = False
        return {"diagnostic_only": True, "replay_index": self._replay_count,
                "graph_total_ms": total_ms, "kernels": records,
                "zero_duration_kernel_intervals": sum(row["elapsed_ms"] == 0 for row in records),
                "all_api_calls_succeeded": True}

    def close(self, *, caller_synchronized=False):
        """Release only owned resources; never synchronize or destroy raw_graph.

        A pending launch always requires an explicit caller sync assertion.
        EventQuery on an in-graph event can report a stale completed record while
        a newer replay is queued, so it cannot make asynchronous cleanup safe.
        API cleanup errors raise and retain remaining handles.
        """
        if self._closed:
            return
        if self._pending and not caller_synchronized:
            raise ProbeError("Unread/failed replay: caller must synchronize before cleanup")
        if self.executable is not None:
            self.api.call("ixGraphExecDestroy", P(self.executable))
            self.executable = None
        if self.clone is not None:
            self.api.call("ixGraphDestroy", P(self.clone))
            self.clone = None
        while self.events:
            self.api.call("ixEventDestroy", P(self.events[-1]))
            self.events.pop()
        self._closed = True
        self._pending = False

