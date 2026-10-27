"""Private diagnostic binding at SDK explicit-maxima / parent-selected Q32.

Import/setup is stdlib-only. Real JIT construction is isolated in the explicitly
device-only factory. No vLLM forward-context/custom-op/module binding is changed.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

import biv150_prefill_host_component as host

TARGET_PINS = {
    "model_executor/layers/attention/attention.py": "3414faca4cc04c699dbb880ba1761230e1ed97035f2934604cad9a5490f4dba1",
    "forward_context.py": "1c6fa51e9a02a120fed5e7357afb1190516cbdfefd1c644dfcd860603f1527c5",
}
NATIVE_ENV = "BIV150_NATIVE_W8A8_LIBRARY"


def native_runtime_identity(torch, library, *, modules=None, maps_text=None):
    """Observe an identity file/optional extension; never import/load either SO."""
    library=Path(library).resolve();modules=sys.modules if modules is None else modules
    loaded=vars(torch.ops).get("loaded_libraries")
    known=isinstance(loaded,(set,frozenset,list,tuple))and all(isinstance(p,str)for p in loaded)
    ops_loaded=(library in {Path(p).resolve()for p in loaded})if known else None
    if maps_text is None:
        try:maps_text=Path("/proc/self/maps").read_text()
        except OSError:pass
    mapped=None
    if maps_text is not None:
        paths=[]
        for line in maps_text.splitlines():
            fields=line.split(None,5)
            if len(fields)==6 and fields[5].startswith("/"):
                paths.append(Path(fields[5].removesuffix(" (deleted)").replace("\\040"," ")).resolve())
        mapped=library in paths
    loader=modules.get("vllm_fl.patches.biv150_native_w8a8")
    optional=modules.get("vllm_fl._C");filename=vars(optional).get("__file__")if optional is not None else None
    optional_path=Path(filename).resolve()if isinstance(filename,str)else None
    optional_sha=(hashlib.sha256(optional_path.read_bytes()).hexdigest()
                  if optional_path is not None and optional_path.is_file()else None)
    return dict(identity_file=dict(path=str(library),sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        role="biv150_native Torch Library W8 identity; not Python vllm_fl._C"),
        native_w8_required_by_attention=False,load_requested_by_diagnostic=False,
        native_w8_loaded=(mapped if mapped is not None else True if ops_loaded else None),
        mapped_in_process=mapped,listed_in_torch_ops_loaded_libraries=ops_loaded,
        native_loader_module_present=loader is not None,
        native_loader_loaded_flag=vars(loader).get("_LOADED")if loader is not None else None,
        optional_python_C=dict(present=optional is not None,path=str(optional_path)if optional_path else None,
            sha256=optional_sha,used_as_W8_identity=False,required_by_diagnostic=False))


@contextmanager
def target_source_directory(path):
    path = Path(path)
    if path.is_dir():
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="biv150-actual-target-sources-") as temporary:
        root = Path(temporary)
        with tarfile.open(path) as archive:
            expected = set(TARGET_PINS) | {"manifest.json"}
            members = archive.getmembers()
            if len(members) != 3 or {m.name for m in members} != expected or any(not m.isfile() for m in members):
                raise ValueError("Unexpected actual-target source archive members")
            archive.extractall(root, filter="data")
        yield root


def verify_target_sources(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["version"] != "0.24.0+empty":
        raise ValueError("Actual target source manifest required")
    records = {r["relative"]: r for r in manifest["files"]}
    if set(records) != set(TARGET_PINS):
        raise ValueError("Target source set changed")
    result = {}
    for name, expected in TARGET_PINS.items():
        raw = (root / name).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != expected
                or records[name]["sha256"] != expected
                or records[name]["bytes"] != len(raw)):
            raise ValueError("Actual target source changed: " + name)
        result[name] = dict(sha256=expected, bytes=len(raw))
    return result


@dataclass(slots=True)
class _Lease:
    used: bool = False


@dataclass(frozen=True, slots=True)
class CpuContract:
    """Fixture/verified caller assertions, not a reader of device metadata."""
    query: object
    cu_tensor: object
    lengths_tensor: object
    cu_q: object
    upper_bound_proven: object
    packing: object
    coherent: object
    _lease: _Lease = field(default_factory=_Lease, init=False, repr=False)


class Boundary:
    def __init__(self, original, router, parent_chain):
        self.original, self.router, self.parent_chain = original, router, parent_chain
        self.calls = []
        self.omitted = 0
        self.last = None
        self.max_records = 512

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            frame = host._CURRENT.get()
            stamp = frame.metadata.stamp if type(frame) is host._Frame else object()
            original = host.Call(self.original, args, grid, kwargs, self.parent_chain, stamp)
            selected = self.router.select(original)
            route = "reverse" if selected.entry is self.router.scope.reverse_entry else "parent"
            # Execute outside any qualification catch: no retry on launch errors.
            result = selected.run()
            self.last = dict(route=route, original=original, selected=selected, result=result)
            if len(self.calls) < self.max_records:
                self.calls.append(dict(route=route, grid=list(grid), rows=kwargs["query_ptr"].shape[0],
                    hash=getattr(result, "hash", None), payload_identity=(selected.args is args
                    and selected.kwargs is kwargs and selected.grid is grid),
                    parent_call_identity=(selected is original)))
            else:
                self.omitted += 1
            return result
        return invoke


class LocalSdkBinding:
    """Serial component scope on the ORIGINAL SDK callable and selected launcher.

    Baseline calls the original with no adapter/tap. candidate() grants a private
    diagnostic extent; only an eligible SDK call temporarily installs the selected
    boundary. The outer M2/Q32 object is retained. Long calls have no new tap.
    """
    def __init__(self, function, prepared, q32_manager, parent_chain,
                 reverse_entry, capturing, intact):
        self.function, self.manager = function, q32_manager
        self.original = q32_manager.wrapper
        self.parent_chain, self.capturing, self.intact = parent_chain, capturing, intact
        self.router = host.Router(prepared, host.ComponentScope(host.BOUNDARY,
            host.PACKING, "NONE", True, True, True, True,
            parent_chain, self.original, reverse_entry))
        if not self.router.enabled:
            raise ValueError("Complete private component scope required")
        self.boundary = Boundary(self.original, self.router, parent_chain)
        self.active = False
        self.restored = True
        self.direct_parent_calls = 0
        self.scoped_calls = 0

    def call(self, *args, contract=None, **kwargs):
        if not self.active or not self.intact():
            raise RuntimeError("Private component parent binding changed/not active")
        valid = type(contract) is CpuContract and not contract._lease.used
        if type(contract) is CpuContract:
            contract._lease.used = True
        valid = (valid and not args and contract.query is kwargs.get("q")
                 and contract.cu_tensor is kwargs.get("cu_seqlens_q")
                 and contract.lengths_tensor is kwargs.get("seqused_k")
                 and contract.upper_bound_proven is True and contract.coherent is True
                 and type(contract.packing) is str and contract.packing == host.PACKING)
        q, upper = kwargs.get("max_seqlen_q"), kwargs.get("max_seqlen_k")
        ready = (valid and type(q) is int and type(upper) is int
            and 1024 <= q <= 2**31-1 and q <= upper <= min(5*q,2**31-1)
            and type(contract.cu_q) is tuple and len(contract.cu_q)==2
            and all(type(x)is int for x in contract.cu_q)and contract.cu_q==(0,q))
        if ready:
            try:
                ready=(type(contract.query.shape[0])is int and contract.query.shape[0]==q
                    and type(contract.lengths_tensor.shape[0])is int and contract.lengths_tensor.shape[0]==1)
            except Exception:
                ready=False
        # A long/unknown call has no new metadata Frame or selected-Q32 tap.
        # Nested calls must mask the outer metadata while keeping its ownership.
        if not ready or host._CURRENT.get() is not None or self.capturing() is not False:
            self.direct_parent_calls += 1
            if host._CURRENT.get() is None:
                result=self.function(*args,**kwargs)
            else:
                with host.cpu_metadata(None):result=self.function(*args,**kwargs)
        else:
            if self.manager.wrapper is not self.original:
                raise RuntimeError("Selected-Q32 wrapper ownership changed")
            metadata=host.CpuMetadata(object(),kwargs["q"],q,upper,contract.cu_q,
                True,host.PROVENANCE,False)
            self.manager.wrapper=self.boundary
            self.restored=False
            self.scoped_calls += 1
            try:
                # Local to this explicit SDK call. No outer forward-context hook.
                with host.cpu_metadata(metadata):result=self.function(*args,**kwargs)
            finally:
                if self.manager.wrapper is self.boundary:
                    self.manager.wrapper=self.original;self.restored=True
                else:
                    self.router.invalidate();self.restored=False
        if not self.intact():
            raise RuntimeError("Parent scope changed during component call; no retry")
        return result

    @contextmanager
    def candidate(self):
        if self.active or not self.intact() or self.manager.wrapper is not self.original:
            raise RuntimeError("Private component scope is busy or changed")
        self.active, self.restored = True, True
        try:
            yield self.call
        finally:
            self.active = False
            if self.manager.wrapper is self.original:
                self.restored = True
            else:
                self.router.invalidate()
                # Preserve unknown ownership and the original exception.
                self.restored = False

    def close(self):
        self.router.invalidate()
        if self.active or not self.restored or self.manager.wrapper is not self.original:
            raise RuntimeError("Private component scope did not restore")


def make_private_reverse_manager(attention, backend, parent_manager, q32module,
                                 prepared, out):
    """DEVICE-ONLY: old fca source + unchanged verified Manager; never run on CPU."""
    import inspect
    import linecache
    import importlib
    if inspect.getsource(parent_manager.raw.fn) != prepared.baseline:
        raise ValueError("Actual parent Q32 raw source changed")
    source = prepared.reverse
    name = "<biv150-host-unconditional-" + prepared.proof["reversed_source_sha256"] + ".py>"
    linecache.cache[name] = (len(source), None, source.splitlines(keepends=True), name)
    namespace = dict(parent_manager.raw.fn.__globals__)
    exec(compile(source, name, "exec"), namespace)
    raw = namespace["kernel_biv150_prefill_stride"]
    jit = importlib.import_module("triton.runtime.jit")
    clone = jit.JITFunction(raw.fn, version=raw.version,
        do_not_specialize=list(raw.do_not_specialize),
        do_not_specialize_on_alignment=list(raw.do_not_specialize_on_alignment),
        debug=raw.debug, noinline=raw.noinline, repr=raw._repr,
        launch_metadata=raw.launch_metadata)
    manager = q32module._VerifiedManager(attention, backend, {"kernel": raw},
        "kernel", raw, clone, block_m=256, num_warps=16, num_stages=1, tile_size=64)
    manager.install()
    (out / "baseline-normalized.py").write_text(prepared.baseline)
    (out / "old-fca-unconditional.py").write_text(source)
    return manager
