"""Explicit-CPU-metadata component router; no service hook or GPU dependency.

The eligible boundary is an ALREADY selected Q32 launch, not the raw SDK's
BM128 call into the M2/Q32 policy chain. The caller retains that whole chain.
Only source preparation reads files/hashes/AST; selection performs no such work.
"""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
from pathlib import Path
from types import SimpleNamespace

from biv150_prefill_reverse import (
    HELPER_SHA, OLD, ALWAYS_REVERSE, eligible, source_factory,
)

REVERSE_SHA = "38198704c37479e4b5c26778d7dd1da2a894409d1c81ff2626e1d988facca104"
FRAGMENT_SHA = "38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82"
FACTORY_SHA = "7f06589453c2d59310394bc47406d91c6b7a0d28b4781a9073047bcb78daba0f"
PROVENANCE = "explicit-cpu-component-v1"
BOUNDARY = "parent-selected-Q32"
PACKING = "unencoded-bf16-page16-kvheads2-d128"
_CONSTANTS = ("IS_3D", "HEAD_SIZE", "HEAD_SIZE_PADDED", "num_query_heads",
    "num_queries_per_kv", "BLOCK_M", "BLOCK_Q", "BLOCK_SIZE", "TILE_SIZE",
    "NUM_SEGMENTS_PER_SEQ", "USE_CAUSAL", "SLIDING_WINDOW", "CHUNK_LOOKBACK",
    "CHUNK_SIZE", "num_warps", "num_stages", "USE_FP8", "Q_IS_FP8", "USE_TD",
    "USE_TD_QO", "USE_ALIBI_SLOPES", "USE_ALIBI_SQRT", "USE_QQ_BIAS",
    "USE_SOFTCAP", "USE_SINKS", "USE_PER_SEQ_CAUSAL", "USE_MM_PREFIX")


@dataclass(frozen=True, slots=True)
class PreparedSource:
    baseline: str
    reverse: str
    proof: dict
    descriptor_guard: object


def prepare_source(sdk_text, helper_text, source_root):
    """Recover fca's old source with the unchanged140 factory, without JIT."""
    root = Path(source_root)
    if hashlib.sha256(helper_text.encode()).hexdigest() != HELPER_SHA:
        raise ValueError("Pinned SDK helper source required")
    factory = root / "tools/biv150_prefill_reverse.py"
    fragment = root / "vllm_fl/patches/biv150_q_fragment_service.py"
    if hashlib.sha256(factory.read_bytes()).hexdigest() != FACTORY_SHA:
        raise ValueError("Frozen140/fca-compatible factory changed")
    text = fragment.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != FRAGMENT_SHA:
        raise ValueError("Actual parent descriptor guard changed")
    baseline, reverse, proof = source_factory(sdk_text, root, mode="always")
    if (proof["reversed_source_sha256"] != REVERSE_SHA
            or reverse.replace(ALWAYS_REVERSE, OLD, 1) != baseline
            or "route_q_start" in reverse or proof["metadata_loads"] != 0):
        raise ValueError("Not the old unconditional reverse source")
    functions = [n for n in ast.parse(text).body
                 if isinstance(n, ast.FunctionDef) and n.name == "_eligible"]
    if len(functions) != 1:
        raise ValueError("Unique actual parent descriptor guard required")
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 "<actual-parent-descriptor-guard>", "exec"), namespace)
    proof = dict(proof, helpers_sha256=HELPER_SHA,
                 component_only=True, jit_executed=False,
                 service_binding_verified=False, host_component_device_verified=False)
    return PreparedSource(baseline, reverse, proof, namespace["_eligible"])


@dataclass(frozen=True, slots=True)
class Call:
    """Original caller-owned launch record. Selection never edits its payload."""
    entry: object
    args: tuple
    grid: object
    kwargs: dict
    parent_chain: object
    stamp: object = field(default_factory=object, compare=False)

    def run(self):
        # Ordinary Python **kwargs creates the callee's keyword dict. The record
        # retains the exact caller dict and all Tensor/value references.
        return self.entry[self.grid](*self.args, **self.kwargs)


@dataclass(frozen=True, slots=True)
class ComponentScope:
    boundary: str
    packing: str
    runtime_mode: str
    component_only: bool
    hooks_empty: bool
    async_mode_none: bool
    snapshot_coherent_for_launch: bool
    parent_chain: object
    parent_entry: object
    reverse_entry: object


@dataclass(slots=True)
class _Lease:
    bound: bool = False


@dataclass(frozen=True, slots=True)
class CpuMetadata:
    stamp: object
    query: object
    qmax: object
    kv_upper: object
    cu_q: object
    upper_bound_proven: object
    provenance: object
    capturing: object
    _lease: _Lease = field(default_factory=_Lease, init=False, repr=False, compare=False)


@dataclass(slots=True)
class _Frame:
    metadata: CpuMetadata
    taken: bool = False
    executed: bool = False


@dataclass(frozen=True, slots=True)
class _ReverseCall(Call):
    original: Call = field(kw_only=True, repr=False)
    frame: _Frame = field(kw_only=True, repr=False)
    router: object = field(kw_only=True, repr=False)

    def run(self):
        if self.frame.executed:
            raise RuntimeError("Component selection already executed; do not retry")
        self.frame.executed = True
        if _CURRENT.get() is not self.frame or not self.router.enabled:
            # A stored decision outliving its CPU metadata scope is a miss,
            # before any candidate entry is accessed. Never cache host choices.
            return self.original.run()
        return Call.run(self)


_CURRENT = ContextVar("biv150_explicit_cpu_component", default=None)
_INVALID = object()


@contextmanager
def cpu_metadata(metadata):
    """One-shot, non-nestable component metadata; always restore outer state."""
    state = _INVALID
    if type(metadata) is CpuMetadata and not metadata._lease.bound:
        metadata._lease.bound = True
        if _CURRENT.get() is None:
            state = _Frame(metadata)
    token = _CURRENT.set(state)
    try:
        yield
    finally:
        _CURRENT.reset(token)


class Router:
    def __init__(self, prepared, scope=None):
        self.prepared, self.scope = prepared, scope
        self.enabled = (
            type(prepared) is PreparedSource and type(scope) is ComponentScope
            and type(scope.boundary) is str and type(scope.packing) is str
            and type(scope.runtime_mode) is str
            and scope.boundary == BOUNDARY and scope.packing == PACKING
            and scope.runtime_mode == "NONE"
            and all(getattr(scope, k) is True for k in
                    ("component_only", "hooks_empty", "async_mode_none",
                     "snapshot_coherent_for_launch"))
            and scope.parent_chain is not None and scope.parent_entry is not None
            and scope.reverse_entry is not None
            and scope.parent_entry is not scope.reverse_entry
            and callable(getattr(type(scope.parent_entry), "__getitem__", None))
            and callable(getattr(type(scope.reverse_entry), "__getitem__", None))
            and prepared.proof.get("reversed_source_sha256") == REVERSE_SHA
            and prepared.proof.get("route_mode") == "always"
        )
        self.fragment = (SimpleNamespace(_eligible=prepared.descriptor_guard)
                         if type(prepared) is PreparedSource else None)

    def invalidate(self):
        """Explicit component qualification revocation; no automatic re-enable."""
        self.enabled = False

    def select(self, original):
        """Return original identity on every miss; only replace entry on a hit."""
        frame = _CURRENT.get()
        if type(frame) is not _Frame or frame.taken:
            return original
        frame.taken = True
        if not self.enabled or type(original) is not Call:
            return original
        metadata, scope = frame.metadata, self.scope
        if (type(original.stamp) is not object or metadata.stamp is not original.stamp
                or type(metadata.provenance) is not str
                or metadata.provenance != PROVENANCE
                or metadata.capturing is not False or metadata.upper_bound_proven is not True
                or original.parent_chain is not scope.parent_chain
                or original.entry is not scope.parent_entry
                or type(original.args) is not tuple or original.args
                or type(original.kwargs) is not dict
                or type(original.grid) not in (tuple, list)):
            return original
        q, upper = metadata.qmax, metadata.kv_upper
        if (type(q) is not int or type(upper) is not int
                or not 1024 <= q <= 2**31 - 1
                or not q <= upper <= min(5 * q, 2**31 - 1)
                or type(metadata.cu_q) is not tuple or len(metadata.cu_q) != 2
                or any(type(x) is not int for x in metadata.cu_q)
                or metadata.cu_q != (0, q)):
            return original
        values = original.kwargs
        if (type(values.get("num_seqs")) is not int or values["num_seqs"] != 1
                or metadata.query is not values.get("query_ptr")
                or "ir_override" in values):
            return original
        # Malformed Tensor-valued constexprs must not trigger Python Tensor
        # truth/equality in the old descriptor guard. Preserve real mode0 enums.
        if any(type(values.get(k, False)) not in (bool, int) for k in _CONSTANTS):
            return original
        mode = values.get("KV_QUANT_MODE")
        if type(mode) is not int and not (isinstance(mode, IntEnum) and type(mode.value) is int):
            return original
        try:
            if (type(metadata.query.shape[0]) is not int or metadata.query.shape[0] != q
                    or not eligible(values, original.grid, self.fragment)):
                return original
        except Exception:
            # Descriptor-only qualification, before any launch. Invocation is
            # deliberately outside this catch; a device/caller failure propagates.
            return original
        return _ReverseCall(scope.reverse_entry, original.args, original.grid,
                            original.kwargs, original.parent_chain, original.stamp,
                            original=original, frame=frame, router=self)

    def run(self, original):
        return self.select(original).run()
