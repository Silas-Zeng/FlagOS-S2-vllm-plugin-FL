#!/usr/bin/env python3
"""Capability smoke for Corex event-record nodes inside a tiny CUDA graph.

This file has no effect when imported. --run is explicit GPU authorization for
the caller. It never opens or modifies a serving/model graph. Corex ABI was read
from /usr/local/corex/include/IX/ixrt/ix_runtime_api.h and driver_types.h.

The test uses a cloned, dedicated mm+add graph. It retains every original edge
and adds predecessor -> start-event -> kernel -> end-event -> successor paths.
Timing includes event/graph scheduling overhead and is diagnostic only.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


P = C.c_void_p
PP = C.POINTER(P)
Z = C.c_size_t
ZP = C.POINTER(Z)
I = C.c_int
SIGNATURES = {
    "ixGraphGetNodes": [P, PP, ZP],
    "ixGraphGetEdges": [P, PP, PP, ZP],
    "ixGraphNodeGetDependencies": [P, PP, ZP],
    "ixGraphNodeGetDependentNodes": [P, PP, ZP],
    "ixGraphNodeGetType": [P, C.POINTER(I)],
    "ixGraphAddEventRecordNode": [PP, P, PP, Z, P],
    "ixGraphAddDependencies": [P, PP, PP, Z],
    "ixGraphInstantiateWithFlags": [PP, P, C.c_ulonglong],
    "ixGraphLaunch": [P, P],
    "ixGraphExecDestroy": [P],
    "ixGraphClone": [PP, P],
    "ixGraphDestroy": [P],
    "ixEventCreateWithFlags": [PP, C.c_uint],
    "ixEventRecord": [P, P],
    "ixEventSynchronize": [P],
    "ixEventElapsedTime": [C.POINTER(C.c_float), P, P],
    "ixEventDestroy": [P],
    "ixStreamSynchronize": [P],
}
NODE_TYPES = {
    0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "child_graph",
    5: "empty", 6: "event_wait", 7: "event_record", 10: "mem_alloc",
    11: "mem_free", 13: "conditional",
}


def topological_order(nodes, edges):
    """No device use; also rejects cycles or malformed graph enumeration."""
    incoming = {n: 0 for n in nodes}
    successors = {n: [] for n in nodes}
    for a, b in edges:
        if a not in incoming or b not in incoming:
            raise AssertionError("Edge references a node absent from enumeration")
        incoming[b] += 1
        successors[a].append(b)
    ready = sorted(n for n in nodes if incoming[n] == 0)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for nxt in successors[n]:
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(order) != len(nodes):
        raise AssertionError("Enumerated graph is cyclic")
    return order


class Recorder:
    def __init__(self, out):
        # Exclusive reservation prevents clobbering a previous capability result.
        self.path = Path(out).resolve()
        self.handle = self.path.open("x")
        self.state = {
            "diagnostic_only": True,
            "production_graph_modified": False,
            "capability_proven": False,
            "started_unix_s": time.time(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "calls": [], "records": [], "cleanup_errors": [],
        }
        self.save()

    def save(self):
        self.handle.seek(0)
        json.dump(self.state, self.handle, indent=2, allow_nan=False)
        self.handle.write("\n")
        self.handle.truncate()
        self.handle.flush()

    def emit(self, label, **values):
        item = {"label": label, **values}
        self.state["records"].append(item)
        self.save()
        print(json.dumps(item, allow_nan=False), flush=True)


class Runtime:
    def __init__(self, path, rec):
        self.rec = rec
        self.lib = C.CDLL(str(Path(path).resolve()))
        self.funcs = {}
        for name, args in SIGNATURES.items():
            fn = getattr(self.lib, name)  # Missing symbols fail before device use.
            fn.argtypes = args
            fn.restype = I
            self.funcs[name] = fn
        self.error_string = getattr(self.lib, "ixGetErrorString", None)
        if self.error_string is not None:
            self.error_string.argtypes = [I]
            self.error_string.restype = C.c_char_p
        rec.emit("runtime_bound", library=str(Path(path).resolve()),
                 symbols=sorted(self.funcs), node_types=NODE_TYPES,
                 event_flags=0, kernel_node_type=0, event_record_node_type=7)

    def call(self, name, *args, context=None, required=True):
        rc = int(self.funcs[name](*args))
        message = None
        if rc and self.error_string is not None:
            raw = self.error_string(rc)
            message = raw.decode(errors="replace") if raw else None
        item = {"api": name, "return_code": rc, "context": context,
                "error_string": message}
        self.rec.state["calls"].append(item)
        self.rec.save()
        print(json.dumps(item), flush=True)
        if rc and required:
            raise RuntimeError(f"{name} returned {rc}: {message}; {context}")
        return rc

    def handles(self, name, owner):
        count = Z()
        self.call(name, P(owner), None, C.byref(count), context="count")
        if count.value > 10000:
            raise AssertionError("Unexpected graph size for a tiny smoke")
        if count.value == 0:
            return []
        buf = (P * count.value)()
        self.call(name, P(owner), buf, C.byref(count), context="values")
        return [int(buf[i]) for i in range(count.value)]

    def edges(self, graph):
        count = Z()
        self.call("ixGraphGetEdges", P(graph), None, None, C.byref(count),
                  context="count")
        if count.value > 100000:
            raise AssertionError("Unexpected edge count for a tiny smoke")
        if count.value == 0:
            return []
        src = (P * count.value)()
        dst = (P * count.value)()
        self.call("ixGraphGetEdges", P(graph), src, dst, C.byref(count),
                  context="values")
        return [(int(src[i]), int(dst[i])) for i in range(count.value)]

    def snapshot(self, graph):
        nodes = self.handles("ixGraphGetNodes", graph)
        edges = self.edges(graph)
        types = {}
        deps = {}
        for node in nodes:
            kind = I(-1)
            self.call("ixGraphNodeGetType", P(node), C.byref(kind),
                      context={"node": hex(node)})
            types[node] = kind.value
            deps[node] = self.handles("ixGraphNodeGetDependencies", node)
        from_deps = {(pred, node) for node, ds in deps.items() for pred in ds}
        if from_deps != set(edges):
            raise AssertionError("GetEdges and per-node dependencies disagree")
        order = topological_order(nodes, edges)
        return nodes, edges, types, order

    def create_event(self, owned):
        event = P()
        self.call("ixEventCreateWithFlags", C.byref(event), C.c_uint(0),
                  context="timing enabled, flags=0")
        if not event.value:
            raise AssertionError("Success returned a null event")
        owned.append(event.value)
        return event.value

    def event_node(self, graph, dependencies, event):
        buf = (P * len(dependencies))(*dependencies) if dependencies else None
        node = P()
        self.call("ixGraphAddEventRecordNode", C.byref(node), P(graph), buf,
                  Z(len(dependencies)), P(event),
                  context={"event": hex(event),
                           "dependencies": [hex(n) for n in dependencies]})
        if not node.value:
            raise AssertionError("Success returned a null event node")
        return node.value

    def add_edges(self, graph, edges):
        if not edges:
            return
        src = (P * len(edges))(*(a for a, _ in edges))
        dst = (P * len(edges))(*(b for _, b in edges))
        self.call("ixGraphAddDependencies", P(graph), src, dst, Z(len(edges)),
                  context={"edges": [[hex(a), hex(b)] for a, b in edges]})

    def instantiate(self, graph, owned_execs):
        executable = P()
        self.call("ixGraphInstantiateWithFlags", C.byref(executable), P(graph),
                  C.c_ulonglong(0), context={"graph": hex(graph)})
        if not executable.value:
            raise AssertionError("Success returned a null graph executable")
        owned_execs.append(executable.value)
        return executable.value

    def elapsed(self, start, end, context):
        value = C.c_float(float("nan"))
        self.call("ixEventElapsedTime", C.byref(value), P(start), P(end),
                  context=context)
        # Invalid values are reported without producing invalid JSON.
        return float(value.value) if math.isfinite(value.value) else None


def run(args):
    rec = Recorder(args.out)
    rt = None
    graph = None
    clone = None
    owned_execs = []
    owned_events = []
    stream_handle = None
    failure = None
    try:
        # The script itself is never run on a busy service device.
        status = subprocess.run(["/usr/local/corex/bin/ixsmi"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=15, check=False)
        rec.emit("idle_preflight", return_code=status.returncode, output=status.stdout)
        if status.returncode or "No running processes found" not in status.stdout:
            raise RuntimeError("Idle GPU preflight did not confirm no running processes")
        rt = Runtime(args.library, rec)
        import torch
        if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
            raise RuntimeError("Installed Torch lacks CUDAGraph.raw_cuda_graph")
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        rec.emit("torch", version=torch.__version__, cuda=torch.version.cuda,
                 device=torch.cuda.get_device_name(0), matrix_size=args.size)
        n = args.size
        # Keep all input/output tensors alive until both graph executables die.
        x = torch.linspace(-0.5, 0.5, n * n).reshape(n, n).to("cuda", torch.bfloat16)
        y = torch.linspace(0.4, -0.4, n * n).reshape(n, n).to("cuda", torch.bfloat16)
        bias = torch.full((n, n), 0.125, device="cuda", dtype=torch.bfloat16)
        product = torch.empty_like(x)
        output = torch.empty_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        stream_handle = stream.cuda_stream

        def operation():
            torch.mm(x, y, out=product)
            torch.add(product, bias, out=output)

        with torch.cuda.stream(stream):
            for _ in range(3):
                operation()
        stream.synchronize()
        expected = output.cpu().clone()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            operation()
        raw = int(graph.raw_cuda_graph())
        if not raw:
            raise AssertionError("Torch returned null raw graph")
        rec.emit("captured", raw_graph=hex(raw), keep_graph=True,
                 output_ptr=output.data_ptr(), product_ptr=product.data_ptr())
        nodes, edges, types, order = rt.snapshot(raw)
        rec.emit("original_graph", nodes=[{"handle": hex(k), "type": types[k],
                 "type_name": NODE_TYPES.get(types[k], "unknown")} for k in order],
                 edges=[[hex(a), hex(b)] for a, b in edges])
        # Raw execs are owned here; never call graph.replay/instantiate as well.
        original_exec = rt.instantiate(raw, owned_execs)
        # An unexecuted graph must fail: eager/capture preparation has already
        # populated these buffers with the expected values.
        with torch.cuda.stream(stream):
            product.fill_(float("nan"))
            output.fill_(float("nan"))
        rt.call("ixGraphLaunch", P(original_exec), P(stream_handle), context="original")
        rt.call("ixStreamSynchronize", P(stream_handle), context="original")
        original_exact = torch.equal(output.cpu(), expected)
        rec.emit("original_numeric", exact=original_exact)
        if not original_exact:
            raise AssertionError("Original captured graph differs from eager computation")

        cloned = P()
        rt.call("ixGraphClone", C.byref(cloned), P(raw))
        clone = cloned.value
        if not clone:
            raise AssertionError("Success returned null clone")
        cnodes, cedges, ctypes, corder = rt.snapshot(clone)
        if sorted(types.values()) != sorted(ctypes.values()) or len(edges) != len(cedges):
            raise AssertionError("Clone differs in node types or edge count")
        kernel_nodes = [node for node in corder if ctypes[node] == 0]
        if not kernel_nodes:
            raise RuntimeError("No top-level kernel nodes; nested graph instrumentation not attempted")
        if len(kernel_nodes) > 32:
            raise RuntimeError("Unexpected >32 kernel nodes in tiny graph; refusing instrumentation")
        probes = []
        for index, node in enumerate(kernel_nodes):
            # Fetch current dependencies: earlier inserted end nodes become
            # predecessors, so adjacent probe intervals remain correctly ordered.
            predecessors = rt.handles("ixGraphNodeGetDependencies", node)
            successors = rt.handles("ixGraphNodeGetDependentNodes", node)
            start = rt.create_event(owned_events)
            end = rt.create_event(owned_events)
            start_node = rt.event_node(clone, predecessors, start)
            rt.add_edges(clone, [(start_node, node)])
            end_node = rt.event_node(clone, [node], end)
            rt.add_edges(clone, [(end_node, nxt) for nxt in successors])
            probes.append({"index": index, "kernel": node, "start": start, "end": end,
                           "start_node": start_node, "end_node": end_node})
        inodes, iedges, itypes, iorder = rt.snapshot(clone)
        if not set(cedges).issubset(set(iedges)):
            raise AssertionError("Instrumentation removed an original dependency edge")
        if len(inodes) != len(cnodes) + 2 * len(kernel_nodes):
            raise AssertionError("Unexpected instrumented graph node count")
        for probe in probes:
            if itypes[probe["start_node"]] != 7 or itypes[probe["end_node"]] != 7:
                raise AssertionError("Added event nodes report the wrong type")
        rec.emit("instrumented_graph", original_edges_retained=True,
                 original_node_count=len(cnodes), instrumented_node_count=len(inodes),
                 kernel_count=len(kernel_nodes),
                 topological_order=[hex(v) for v in iorder],
                 edges=[[hex(a), hex(b)] for a, b in iedges],
                 probes=[{k: (hex(v) if k != "index" else v) for k, v in p.items()}
                         for p in probes])
        instrumented_exec = rt.instantiate(clone, owned_execs)
        outer_start = rt.create_event(owned_events)
        outer_end = rt.create_event(owned_events)
        rounds = []
        for variant in range(2):
            if variant:
                # One fixed changed-input case proves replay does not freeze data.
                with torch.cuda.stream(stream):
                    x.mul_(0.5)
                    bias.add_(0.0625)
                    operation()
                stream.synchronize()
                expected = output.cpu().clone()
            for repeat in range(args.repeats):
                ctx = {"variant": variant, "repeat": repeat}
                # Sentinel fills precede outer_start on the same stream and are
                # excluded from timing. Both mm and add must overwrite them.
                with torch.cuda.stream(stream):
                    product.fill_(float("nan"))
                    output.fill_(float("nan"))
                rt.call("ixEventRecord", P(outer_start), P(stream_handle), context=ctx)
                rt.call("ixGraphLaunch", P(instrumented_exec), P(stream_handle), context=ctx)
                rt.call("ixEventRecord", P(outer_end), P(stream_handle), context=ctx)
                rt.call("ixEventSynchronize", P(outer_end), context=ctx)
                total_ms = rt.elapsed(outer_start, outer_end, {**ctx, "kind": "outer"})
                per_kernel = [rt.elapsed(p["start"], p["end"],
                              {**ctx, "kernel_index": p["index"]}) for p in probes]
                actual = output.cpu()
                exact = torch.equal(actual, expected)
                finite = bool(torch.isfinite(actual).all())
                max_abs = float((actual.float() - expected.float()).abs().max())
                positive = total_ms is not None and total_ms > 0 and all(
                    v is not None and v > 0 for v in per_kernel)
                # A positive interval outside the graph is not proof that inner
                # events updated. Check pairwise cross-event ordering as well.
                relative_bounds = []
                for p in probes:
                    lo = rt.elapsed(outer_start, p["start"],
                                    {**ctx, "kind": "offset_start", "kernel_index": p["index"]})
                    hi = rt.elapsed(outer_start, p["end"],
                                    {**ctx, "kind": "offset_end", "kernel_index": p["index"]})
                    relative_bounds.append([lo, hi])
                bounded = total_ms is not None and all(
                    lo is not None and hi is not None and -0.02 <= lo <= hi + 0.02
                    and hi <= total_ms + 0.02 for lo, hi in relative_bounds)
                row = {**ctx, "outer_ms": total_ms, "kernel_intervals_ms": per_kernel,
                       "relative_bounds_ms": relative_bounds,
                       "product_and_output_poisoned_before_replay": True,
                       "numeric_exact": exact, "finite": finite, "max_abs": max_abs,
                       "positive_intervals": positive, "within_outer_bounds": bounded}
                rounds.append(row)
                rec.emit("replay", **row)
                if not (exact and finite and positive and bounded):
                    raise AssertionError("Numeric or graph event timing capability check failed")
        rec.state["capability_proven"] = bool(rounds)
        rec.state["verdict"] = "tiny_top_level_kernel_event_nodes_passed"
        rec.state["limits"] = [
            "Not a production trace or a throughput measurement.",
            "Event instrumentation adds graph nodes, dependencies and timing overhead.",
            "No production graph or nested child graph was mutated.",
            "Only this tiny graph, dtype, runtime and input variants were tested.",
        ]
    except BaseException as exc:
        failure = exc
        rec.state["verdict"] = "failed_or_unsupported"
        rec.state["exception"] = repr(exc)
        rec.state["traceback"] = traceback.format_exc()
        rec.state["capability_proven"] = False
    finally:
        # Destroy executable graphs before event handles, and graph objects last.
        # Cleanup rc is retained too. If device sync fails, do not continue with
        # an aggressive cleanup sequence against possibly active work.
        safe_cleanup = True
        if rt is not None and stream_handle is not None:
            rc = rt.call("ixStreamSynchronize", P(stream_handle), context="cleanup", required=False)
            safe_cleanup = rc == 0
            if not safe_cleanup:
                rec.state["cleanup_errors"].append({"api": "ixStreamSynchronize", "rc": rc})
        if rt is not None and safe_cleanup:
            for executable in reversed(owned_execs):
                rc = rt.call("ixGraphExecDestroy", P(executable), context="cleanup", required=False)
                if rc:
                    rec.state["cleanup_errors"].append({"api": "ixGraphExecDestroy", "rc": rc})
            if clone:
                rc = rt.call("ixGraphDestroy", P(clone), context="cleanup", required=False)
                if rc:
                    rec.state["cleanup_errors"].append({"api": "ixGraphDestroy", "rc": rc})
            if graph is not None:
                try:
                    graph.reset()  # Torch owns only the original raw graph.
                except BaseException as exc:
                    rec.state["cleanup_errors"].append({"api": "torch_graph.reset", "error": repr(exc)})
            for event in reversed(owned_events):
                rc = rt.call("ixEventDestroy", P(event), context="cleanup", required=False)
                if rc:
                    rec.state["cleanup_errors"].append({"api": "ixEventDestroy", "rc": rc})
        if rec.state["cleanup_errors"]:
            rec.state["capability_proven"] = False
            rec.state["verdict"] = "cleanup_failed"
        rec.state["completed"] = True
        rec.state["finished_unix_s"] = time.time()
        rec.save()
        print(json.dumps({"result": str(rec.path), "verdict": rec.state["verdict"],
                          "capability_proven": rec.state["capability_proven"]}), flush=True)
        rec.handle.close()
    return 0 if failure is None and rec.state["capability_proven"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the GPU capability smoke")
    parser.add_argument("--out", type=Path, help="new JSON result file; parent must exist")
    parser.add_argument("--library", default="/usr/local/corex/lib64/libcudart.so")
    parser.add_argument("--size", type=int, default=256, choices=(128, 256, 512))
    parser.add_argument("--repeats", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"gpu_executed": False, "usage": "--run --out /new/result.json",
                          "symbols": sorted(SIGNATURES), "node_types": NODE_TYPES}, indent=2))
        return 0
    if args.out is None:
        parser.error("--run requires --out pointing at a new result file")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
