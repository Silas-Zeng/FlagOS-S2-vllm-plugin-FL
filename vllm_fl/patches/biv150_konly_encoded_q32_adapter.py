"""Opt-in fresh Q32 operand-hoist preparation; no import-time framework work.

Only prelaunch preparation is allowed to retain the correct private unhoisted
reader. The caller performs exactly one launch, outside this module's exception
handler, then checks the returned actual object. No BF16-format fallback exists.
"""
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import threading

ENV = 'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'
SCOPE_CONSTANTS = ('BLOCK_M', 'BLOCK_Q', 'TILE_SIZE', 'num_seqs', 'num_query_heads',
                   'num_queries_per_kv', 'KV_QUANT_MODE', 'IS_3D')


def need(ok, message):
    if not ok: raise ValueError('Private Q32 hoist: ' + message)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def typed(value):
    """Exact primitive metadata, retaining dict key and bool/int type distinctions."""
    if value is None or type(value) in (bool, str, int):
        return [type(value).__name__, value]
    if type(value) is float:
        need(math.isfinite(value), 'nonfinite host metadata')
        return ['float', value.hex()]
    if isinstance(value, int):
        return [type(value).__module__+'.'+type(value).__qualname__, int(value)]
    if type(value) in (list, tuple):
        return [type(value).__name__, [typed(v) for v in value]]
    if type(value) is dict:
        pairs = [[typed(k), typed(v)] for k, v in value.items()]
        return ['dict', sorted(pairs, key=lambda p: json.dumps(p[0], sort_keys=True))]
    raise ValueError('Private Q32 hoist: unrepresented metadata type '+type(value).__name__)


def identity(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')))


def kernel_snapshot(kernel, fn):
    need(kernel.src.fn is fn, 'actual ASTSource is not the selected private JIT object')
    source = fn.src
    need(isinstance(source, str) and source, 'live JIT source absent')
    artifacts = kernel.asm
    need(all(isinstance(artifacts.get(e), str) for e in ('ttir', 'ttgir', 'llir', 'asm')),
         'actual compiler text artifacts incomplete')
    need(isinstance(artifacts.get('cubin'), bytes) and artifacts['cubin'][:6] == b'\x7fELF\x02\x01',
         'actual target cubin absent')
    md = kernel.metadata
    need((md.num_warps, md.num_stages, md.warp_size) == (16, 1, 64), 'actual compiler geometry')
    return dict(hash=str(kernel.hash), source_digest=sha(source), function_name=md.name,
        ABI={k: typed(getattr(kernel.src, k)) for k in ('signature', 'constants', 'attrs')},
        artifacts={k: sha(v) for k, v in artifacts.items() if isinstance(v, (str, bytes))},
        launch_geometry=[md.num_warps, md.num_stages, md.warp_size])


def loaded(kernel):
    kernel._init_handles()
    md = kernel.metadata
    fields = dict(n_regs=getattr(kernel, 'n_regs', None), n_spills=getattr(kernel, 'n_spills', None),
        shared=getattr(md, 'shared', None), num_warps=md.num_warps,
        num_stages=md.num_stages, warp_size=md.warp_size)
    need(type(fields['n_regs']) is int and fields['n_regs'] > 0, 'actual loaded register count unknown')
    need(type(fields['n_spills']) is int and fields['n_spills'] == 0, 'unknown/spilled actual kernel')
    need(type(fields['shared']) is int and 0 <= fields['shared'] <= 65536, 'actual shared bound')
    return dict(fields, phase='handles_initialized_pre_reader', known=True)


class Adapter:
    PROFILE = 'q32'
    BINDING_SCHEMA = 'cjj-encoded-q32-live-binding-v1'
    LOG_PREFIX = 'BIV150_KONLY_Q32_HOIST'

    def __init__(self, proposal, native, *, directory, target, pins, implementation_digest, jit, device, fn):
        self.proposal, self.native = proposal, native
        self.directory = Path(directory)
        need(self.directory.is_absolute(), 'private absolute artifact directory required')
        need(target == dict(backend='corex', arch=71, warp_size=64), 'actual installed target')
        need(pins == proposal.PINS, 'actual installed SDK/helper/JIT/codec pins')
        self.target, self.pins = target, dict(pins)
        self.jit, self.device = jit, device
        self.fn, self.source = fn, fn.src
        self.implementation_digest = implementation_digest
        self.calls, self.bindings, self.by_hash = {}, {}, {}
        self.compiled_objects, self.geometry_records = {}, {}
        self.lock = threading.RLock()
        self.counts = dict(prepared=0, unhoisted_warmups=0, cold_audits=0,
            admitted_host_calls=0, fallback_host_calls=0, failures={}, capture_bypasses=0,
            geometry_metadata_validations=0, compiler_binding_reuse_attempts=0)

    @staticmethod
    def installed_pins(fn, source_module, installed):
        """JIT file bytes and the exact decorated codec slice, not a guessed dist."""
        pins = dict(sdk=installed['vllm.v1.attention.ops.triton_unified_attention'],
            helpers=installed['vllm.v1.attention.ops.triton_attention_helpers'])
        pins['jit'] = sha(Path(inspect.getfile(type(fn))).read_bytes())
        text = Path(source_module).read_text()
        n = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
                 and n.name == 'quantize_bf16_rne')
        need(len(n.decorator_list) == 1 and n.decorator_list[0].lineno == n.lineno-1,
             'exact decorated codec source span')
        pins['codec'] = sha(''.join(text.splitlines(True)[n.lineno-2:n.end_lineno]))
        return pins

    def jit_key(self, fn, launch):
        # Exact run-prefix from the SHA-pinned installed jit.py737-756. The
        # native binder determines pointer/scalar specialization, not our code.
        need(not fn.pre_run_hooks, 'foreign JIT pre-run hook')
        need(self.jit.driver.active.get_current_device() == self.device, 'current JIT device changed')
        cache, key_cache, target, backend, binder = fn.device_caches[self.device]
        need(dict(backend=target.backend, arch=target.arch, warp_size=target.warp_size) == self.target,
             'per-device binder target changed')
        options = dict(launch)
        options['debug'] = options.get('debug', fn.debug) or self.jit.knobs.runtime.debug
        options['instrumentation_mode'] = self.jit.knobs.compilation.instrumentation_mode
        self.jit._flagprism.apply_compile_options(options)
        _, specialization, compile_options = binder(**options)
        return cache, self.jit.compute_cache_key(key_cache, specialization, compile_options)

    def call_key(self, fn, launch, grid):
        cache, key = self.jit_key(fn, launch)
        geometry = {}
        for k, v in launch.items():
            if hasattr(v, 'untyped_storage'):
                # SME eligibility is derived on cache miss in jit.py761 and
                # depends on logical geometry/contiguity. Keep the full tensor
                # geometry in addition to its actual native specialization.
                geometry[k] = [str(v.dtype), str(v.device), list(v.shape), list(v.stride())]
        full = identity(dict(fn=id(fn), jit_key=key,
            pointer_geometry=geometry, grid=list(grid),
            admission_constants={k:typed(launch[k]) for k in SCOPE_CONSTANTS}))
        return full, cache.get(key), key

    def _meta(self, before, launch, grid):
        constants = {k: launch[k] for k in SCOPE_CONSTANTS}
        constants['KV_QUANT_MODE'] = int(constants['KV_QUANT_MODE'])
        return dict(schema=self.proposal.SCHEMA, enabled=True, profile=self.PROFILE,
            format=self.proposal.FORMAT, target=self.target,
            launch_options=dict(num_warps=launch['num_warps'], num_stages=launch['num_stages']),
            constants=constants, query_rows=int(launch['query_ptr'].shape[0]), grid=list(grid),
            kernel_hash=before['hash'], function_name=before['function_name'],
            source_digest=before['source_digest'], ttgir_sha256=before['artifacts']['ttgir'], pins=self.pins)

    def prepare(self, fn, launch, grid, *, capture=False):
        """Return override kwargs plus binding, or unchanged private launch kwargs."""
        # A foreign private reader/hook is not a legitimate unhoisted fallback.
        need(fn is self.fn and fn.src is self.source, 'installed private JIT source/identity changed')
        need(not fn.pre_run_hooks, 'foreign private JIT pre-run hook')
        if capture:
            self.counts['capture_bypasses'] += 1
            return {}, None
        with self.lock:
            try:
                key, cached, compiler_key = self.call_key(fn, launch, grid)
                if key in self.calls and self.calls[key][1] is cached:
                    binding = self.calls[key][0]
                else:
                    first = cached
                    # An actual immutable CompiledKernel and its complete JIT
                    # specialization/options identify a compiler proof. Host
                    # query length/grid do not create another native program.
                    object_key = (id(first), compiler_key)
                    existing = self.compiled_objects.get(object_key)
                    if existing is not None:
                        self._check_unhoisted(first, fn, existing)
                        before = existing['unhoisted']
                    else:
                        self.counts['unhoisted_warmups'] += 1
                        first = fn.warmup(grid=grid, **launch)
                        need(self.call_key(fn, launch, grid)[1] is first,
                             'actual warmup differs from the exact per-device JIT key')
                        object_key = (id(first), compiler_key)
                        before = kernel_snapshot(first, fn)
                    meta = self._meta(before, launch, grid)
                    scope = dict(actual_metadata=meta, tensor_geometry={k:[str(v.dtype),str(v.device),
                        list(v.shape),list(v.stride())] for k,v in launch.items() if hasattr(v,'untyped_storage')},
                        metadata_validated=False, native_binding_id=None, admitted_host_calls=0)
                    self.geometry_records[key] = scope
                    self.counts['geometry_metadata_validations'] += 1
                    # Never bypass scope checks for a reused compiler object:
                    # all constants, q>=1024, exact grid, target and pins remain.
                    self.proposal.validate_meta(first.asm['ttgir'], meta)
                    scope['metadata_validated'] = True
                    if existing is not None:
                        binding = existing
                        self.counts['compiler_binding_reuse_attempts'] += 1
                    else:
                        compiler_meta = {k:v for k,v in meta.items() if k not in ('query_rows','grid')}
                        spec = identity(dict(actual_compiled_object=id(first),before=before,
                                             compiler_meta=compiler_meta,jit_key=compiler_key))
                        binding = self._prepare(fn, first, before, meta, launch, grid, spec, compiler_key)
                        binding['_unhoisted_kernel'] = first
                        binding['_unhoisted_ast_source'] = first.src
                        binding['_unhoisted_ABI_objects'] = {k:getattr(first.src,k)
                            for k in ('signature','constants','attrs')}
                        self.bindings[spec] = binding
                        self.compiled_objects[object_key] = binding
                    scope['native_binding_id'] = binding['native_binding_id']
                    self.calls[key] = binding, first
                if binding is None:
                    self.counts['fallback_host_calls'] += 1
                    return {}, None
                # The normal JIT launch must hit the very object audited here.
                # If a cache was cleared/replaced this scope stays on private
                # unhoisted until its unhoisted compiled object changes.
                override = dict(launch, ir_override=binding['override_path'])
                cache, override_key = self.jit_key(fn, override)
                need(cache.get(override_key) is binding['_kernel'], 'audited override JIT cache changed')
                self.counts['admitted_host_calls'] += 1
                self.geometry_records[key]['admitted_host_calls'] += 1
                return dict(ir_override=binding['override_path']), binding
            except Exception as error:
                # No reader has launched. Preserve diagnostics and the same
                # correct private reader/args/grid; never use stock BF16 here.
                reason = type(error).__name__+': '+str(error)[:240]
                self.counts['failures'][reason] = self.counts['failures'].get(reason, 0)+1
                self.counts['fallback_host_calls'] += 1
                if 'key' in locals(): self.calls[key] = None, locals().get('first', cached)
                print(self.LOG_PREFIX+'_PRELAUNCH_RETAIN', json.dumps(dict(reason=reason)), flush=True)
                return {}, None

    @staticmethod
    def _check_unhoisted(kernel, fn, binding):
        # Same immutable-object contract as the admitted override's hot check;
        # no full artifact hashing is needed for each new host geometry.
        need(kernel is binding['_unhoisted_kernel'] and kernel.src is binding['_unhoisted_ast_source']
             and kernel.src.fn is fn and str(kernel.hash) == binding['unhoisted']['hash'],
             'unhoisted actual compiler object/source/hash changed')
        need(all(getattr(kernel.src,k) is v for k,v in binding['_unhoisted_ABI_objects'].items()),
             'unhoisted actual typed ABI metadata object changed')

    def _prepare(self, fn, first, before, meta, launch, grid, spec, compiler_key):
        self.counts['cold_audits'] += 1
        override, proof = self.proposal.propose(first.asm['ttgir'], meta)
        folder = self.directory / str(os.getpid()) / spec
        folder.mkdir(parents=True, exist_ok=False)
        path = folder/'override.ttgir'; path.write_text(override)
        self._preserve(folder/'unhoisted', first, before)
        second = fn.warmup(grid=grid, **launch, ir_override=str(path))
        after = kernel_snapshot(second, fn)
        self._preserve(folder/'hoisted', second, after)
        need(all(before[k] == after[k] for k in ('source_digest', 'function_name', 'ABI', 'launch_geometry')),
             'live source/complete typed ABI changed beyond override')
        need(first.asm['ttir'] == second.asm['ttir'], 'source TTIR changed beyond override')
        native = self.native.audit(override, *(second.asm[e] for e in ('ttgir', 'llir', 'asm')))
        resources = dict(unhoisted=loaded(first), hoisted=loaded(second))
        binding = dict(schema=self.BINDING_SCHEMA, implementation_sha256=self.implementation_digest,
            native_binding_id=spec, actual_JIT_specialization_options_key=compiler_key,
            unhoisted_compiled_object_identity=id(first),
            actual_metadata=meta, unhoisted=before, hoisted=after, override_path=str(path),
            override_sha256=sha(override), proposal=proof, native=native, resources=resources,
            artifact_directory=str(folder), numerical_verified=False, Graph_verified=False,
            precision_gate=False, performance_verified=False, service_verified=False)
        (folder/'binding.json').write_text(json.dumps(binding, indent=2)+'\n')
        # Keep only the actual compiled object in memory, never in JSON/stats.
        binding['_kernel'] = second
        binding['_ast_source'] = second.src
        binding['_fn_source'] = fn.src
        binding['_ABI_objects'] = {k:getattr(second.src,k) for k in ('signature','constants','attrs')}
        self.by_hash[after['hash']] = binding
        self.counts['prepared'] += 1
        print(self.LOG_PREFIX+'_PREPARED', json.dumps(dict(unhoisted_hash=before['hash'],
            hoisted_hash=after['hash'], override_sha256=binding['override_sha256'],
            artifact_directory=str(folder), actual_native_verified=True,
            loaded_resources=resources, numerical_verified=False, service_verified=False)), flush=True)
        return binding

    @staticmethod
    def _preserve(folder, kernel, snapshot):
        folder.mkdir()
        for ext, value in kernel.asm.items():
            if isinstance(value, (str, bytes)):
                (folder/('actual.'+ext)).write_bytes(value.encode() if isinstance(value, str) else value)
        (folder/'actual.jit_python_source.py').write_text(kernel.src.fn.src)
        (folder/'record.json').write_text(json.dumps(snapshot, indent=2)+'\n')

    def after_launch(self, handle, fn, binding):
        if binding is None: return
        # Deliberately outside prepare's catch. Any exception propagates; the
        # caller must never launch the original reader again after this point.
        # Full artifact SHA/native/ABI snapshots were checked once in _prepare.
        # The private CompiledKernel/ASTSource and its metadata dictionaries are
        # immutable after admission. Hot calls check their identities, not MB of
        # artifacts. Foreign in-place mutation is outside this private contract.
        need(handle is binding['_kernel'], 'launched compiled object identity changed')
        need(handle.src is binding['_ast_source'] and handle.src.fn is fn,
             'launched actual ASTSource/function identity changed')
        need(fn.src is binding['_fn_source'] and str(handle.hash) == binding['hoisted']['hash'],
             'launched source/hash changed')
        need(all(getattr(handle.src,k) is v for k,v in binding['_ABI_objects'].items()),
             'launched typed ABI metadata object changed')

    def stats(self):
        return dict(self.counts, enabled=True, bound_kernel_hashes=sorted(self.by_hash),
            bindings=[dict(hash=b['hoisted']['hash'], unhoisted_hash=b['unhoisted']['hash'],
                native_binding_id=b['native_binding_id'],
                artifact_directory=b['artifact_directory'], override_sha256=b['override_sha256'],
                resources=b['resources'], numerical_verified=False, service_verified=False)
                for b in self.bindings.values()],
            cache_key='SHA-pinned actual per-device binder/compute_cache_key plus tensor geometry/grid',
            native_proof_key='strong actual CompiledKernel identity plus full actual snapshot and JIT specialization/options',
            host_geometry_checks=list(self.geometry_records.values()),
            host_geometry_count=len(self.geometry_records), actual_compiler_proof_count=len(self.compiled_objects),
            expensive_audits_per_actual_compiled_object=True, host_counts_not_Graph_replays=True,
            numerical_verified=False, service_verified=False, performance_verified=False)
