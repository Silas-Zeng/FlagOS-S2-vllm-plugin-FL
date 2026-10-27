#!/usr/bin/env python3
"""CPU protocol fixtures plus immutable PR90 actual compiler-artifact replay.

Mock binders/compiled objects exercise admission only; no frontend/native/GPU
execution is simulated. The real immutable IR is audited independently below.
"""
import ast
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace as NS

from check_biv150_konly_encoded_cpu import load, Tensor, Storage, Kernel, rejected

ROOT = Path(__file__).resolve().parents[1]
PR90 = 'e4dd64e453fbe1bdeba4d8b2a1b551167e9446c4'
ARCHIVE = 'docs/evidence/konly-service-q32-hoist-20261027/fresh-native-v1/original-compile.tar.gz'
ARCHIVE_SHA = 'aae141e17f58233df0ca18f92c998d18fb3052a2e0a53de5673b79bb68e1340e'


def fixture(adapter):
    data = subprocess.check_output(['git', '-C', str(ROOT), 'show', PR90+':'+ARCHIVE])
    assert adapter.sha(data) == ARCHIVE_SHA
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        files = {e.name: tar.extractfile(e).read() for e in tar if e.isfile()}
    manifest = json.loads(files['download-manifest.json'])
    for r in manifest['files']:
        assert len(files[r['relative']]) == r['bytes'] and adapter.sha(files[r['relative']]) == r['sha256']
    return files


def thaw(value):
    return {ast.literal_eval(k.split(':', 1)[1]): v for k, v in value.items()}


class MockJIT:
    """Explicit protocol stand-in, not native_specialize_impl emulation."""
    knobs = NS(runtime=NS(debug=False), compilation=NS(instrumentation_mode=''))
    _flagprism = NS(apply_compile_options=lambda kwargs: None)
    driver = NS(active=NS(get_current_device=lambda: 0))
    @staticmethod
    def compute_cache_key(cache, spec, options):
        return json.dumps([spec, options], sort_keys=True)


class MockFunction:
    debug = False
    pre_run_hooks = []
    def __init__(self, files):
        self.src = files['out/unhoisted/actual.jit_python_source.py'].decode()
        self.calls = []
        self.specialization_revision = 'fixture-version-1'
        self.bind_calls = 0
        self.cache = {}
        target = NS(backend='corex', arch=71, warp_size=64)
        self.device_caches = {0: (self.cache, {}, target, object(), self.binder)}
        self.before = self.compiled(files, 'unhoisted')
        self.after = self.compiled(files, 'proposed-hoist')

    def compiled(self, files, role):
        prefix = 'out/'+role+'/'
        record = json.loads(files[prefix+'record.json'])
        artifacts = {e: files[prefix+'actual.'+e] for e in ('ttir', 'ttgir', 'llir', 'asm', 'cubin')}
        artifacts = {e: b if e == 'cubin' else b.decode() for e, b in artifacts.items()}
        md = thaw(record['metadata'])
        result = NS(hash=record['hash'], src=NS(fn=self,
            **{k: thaw(record[k]) for k in ('signature', 'constants', 'attrs')}),
            asm=artifacts, metadata=NS(**md), n_regs=120, n_spills=0,
            _init_handles=lambda: None)
        return result

    def binder(self, **kwargs):
        self.bind_calls += 1
        options = {k: kwargs[k] for k in ('num_warps', 'num_stages', 'debug', 'instrumentation_mode')}
        if 'ir_override' in kwargs: options['ir_override'] = kwargs['ir_override']
        return {}, [self.specialization_revision], options

    def warmup(self, *, grid, **kwargs):
        self.calls.append((grid, kwargs))
        options = dict(kwargs, debug=False, instrumentation_mode='')
        _, spec, opts = self.binder(**options)
        kernel = self.after if 'ir_override' in kwargs else self.before
        self.cache[MockJIT.compute_cache_key({}, spec, opts)] = kernel
        return kernel


def run():
    a, h, native = load('q32_adapter'), load('hoist_proposal'), load('hoist_native')
    files = fixture(a)
    original = files['out/unhoisted/actual.ttgir'].decode()
    meta = json.loads(files['out/proposal.json'])['proposal']['actual_metadata']
    moved, proof = h.propose(original, meta)
    assert proof['exact_byte_inverse'] and moved == files['out/fresh-service-override.ttgir'].decode()
    actual = native.audit(moved, *(files['out/proposed-hoist/actual.'+e].decode() for e in ('ttgir', 'llir', 'asm')))
    assert actual['actual_I8_operand_hoist_verified'] and not actual['numerical_verified']
    fn = MockFunction(files)
    query = Tensor(Storage(1, 0x1000000, 2048*16*128*2), 'bfloat16', (2048,16,128), (2048,128,1))
    launch = dict(meta['constants'], query_ptr=query, num_warps=16, num_stages=1)
    grid = tuple(meta['grid'])
    with tempfile.TemporaryDirectory(prefix='encoded-q32-adapter-cpu-') as temp, contextlib.redirect_stdout(io.StringIO()):
        def make(folder, selected_fn=fn):
            return a.Adapter(h, native, directory=Path(temp)/folder, target=meta['target'],
                pins=h.PINS, implementation_digest='CPU-fixture-only', jit=MockJIT, device=0, fn=selected_fn)
        adapter = make('good')
        override, binding = adapter.prepare(fn, launch, grid)
        assert binding is not None and Path(override['ir_override']).read_text() == moved
        assert len(fn.calls) == 2 and adapter.counts['cold_audits'] == 1
        assert binding['unhoisted']['ABI'] == binding['hoisted']['ABI']
        assert not binding['numerical_verified'] and not binding['service_verified']
        # New addresses with the same explicitly mocked native specialization
        # do not cause another warmup or structural/native audit.
        changed = dict(launch, query_ptr=Tensor(Storage(2, 0x2000000, 2048*16*128*2),
            'bfloat16', (2048,16,128), (2048,128,1)))
        assert adapter.prepare(fn, changed, grid)[1] is binding and len(fn.calls) == 2
        assert adapter.prepare(fn, launch, grid, capture=True) == ({}, None)
        assert len(fn.calls) == 2
        adapter.after_launch(fn.after, fn, binding)
        rejected(lambda:adapter.after_launch(copy.copy(fn.after),fn,binding))
        bad = copy.copy(fn.after); bad.hash = 'changed-after-launch'
        rejected(lambda: adapter.after_launch(bad, fn, binding))
        old_hash=fn.after.hash;fn.after.hash='changed'
        rejected(lambda:adapter.after_launch(fn.after,fn,binding));fn.after.hash=old_hash
        old_src=fn.after.src;fn.after.src=copy.copy(old_src)
        rejected(lambda:adapter.after_launch(fn.after,fn,binding));fn.after.src=old_src
        old_fn=fn.after.src.fn;fn.after.src.fn=object()
        rejected(lambda:adapter.after_launch(fn.after,fn,binding));fn.after.src.fn=old_fn
        old_abi=fn.after.src.signature;fn.after.src.signature=dict(old_abi)
        rejected(lambda:adapter.after_launch(fn.after,fn,binding));fn.after.src.signature=old_abi
        # Repeated admitted hot checks never call the full artifact snapshot.
        original_snapshot=a.kernel_snapshot
        a.kernel_snapshot=lambda *x:(_ for _ in ()).throw(AssertionError('hot artifact hashing'))
        for _ in range(8):adapter.after_launch(fn.after,fn,binding)
        a.kernel_snapshot=original_snapshot
        # An evicted override cannot cause an unaudited compile+launch.
        _, key = adapter.jit_key(fn, dict(launch, **override)); del fn.cache[key]
        assert adapter.prepare(fn, launch, grid) == ({}, None)
        # Complete ABI/type drift, native IR drift, unknown/spilled resources
        # all retain the private unhoisted call before any reader launch.
        failures = []
        for name, mutate in (
            ('ABI-type', lambda f: f.after.src.constants.update({(16,): True})),
            ('TTIR', lambda f: f.after.asm.update(ttir='changed')),
            ('native-ASM', lambda f: f.after.asm.update(asm='changed')),
            ('unknown-registers', lambda f: setattr(f.after, 'n_regs', None)),
            ('spills', lambda f: setattr(f.after, 'n_spills', 1)),
        ):
            f = MockFunction(files); mutate(f); probe = make(name,f)
            assert probe.prepare(f, launch, grid) == ({}, None)
            assert probe.counts['fallback_host_calls'] == 1 and not probe.by_hash
            failures.append(name)
        probe = make('wrong-grid')
        assert probe.prepare(fn, launch, (64,2)) == ({}, None)
        failures.append('incomplete-grid')
        rejected(lambda: make('identity').prepare(MockFunction(files), launch, grid))
        f = MockFunction(files); f.pre_run_hooks = [lambda:None]
        rejected(lambda: make('hook',f).prepare(f,launch,grid))
        assert a.typed({(1,): True}) != a.typed({(1,): 1})
        rejected(lambda: make('unused').__class__(h,native,directory=temp,target=meta['target'],
            pins=dict(h.PINS,jit='changed'),implementation_digest='CPU',jit=MockJIT,device=0,fn=fn))
        source = ROOT/'vllm_fl/patches/biv150_konly_encoded_q32.py'
        pins = a.Adapter.installed_pins(fn, source,
            {'vllm.v1.attention.ops.triton_unified_attention':h.PINS['sdk'],
             'vllm.v1.attention.ops.triton_attention_helpers':h.PINS['helpers']})
        assert pins['codec'] == h.PINS['codec'] and pins['jit'] != h.PINS['jit']
        # The real facade's only launch is outside prepare's catch. Inject a
        # postlaunch rejection and verify no second launch occurs.
        r, broker = load('runtime'), load('views')
        runtime = r.Runtime(NS(cuda=NS(is_current_stream_capturing=lambda:False)),broker,{'q32':Kernel()},NS())
        runtime.q32_hoist = NS(prepare=lambda *x,**kw: ({'ir_override':'CPU-only'}, {'mock':True}),
            after_launch=lambda *x: (_ for _ in ()).throw(ValueError('postlaunch')))
        # Execute the actual dispatch tail, with a reader that records exactly
        # one mock launch and a postlaunch validator that rejects.
        tree = ast.parse((ROOT/'vllm_fl/patches/biv150_konly_encoded_runtime.py').read_text())
        invoke = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='invoke')
        launches = [n for n in ast.walk(invoke) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript)
                    and isinstance(n.func.value,ast.Name) and n.func.value.id=='fn']
        assert len(launches)==1 and not any(isinstance(n,ast.Try) for n in ast.walk(invoke))
        start=next(i for i,n in enumerate(invoke.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='fn' for t in n.targets))
        tail=copy.deepcopy(invoke.body[start:-1])
        module=ast.Module(body=tail,type_ignores=[]);ast.fix_missing_locations(module)
        rejected(lambda:exec(compile(module,'<actual-facade-dispatch-tail>','exec'),
            dict(runtime=runtime,profile='q32',target_grid=grid,selected=launch)))
        assert len(runtime.kernels['q32'].calls)==1
        stats = adapter.stats()
        assert not stats['numerical_verified'] and not stats['service_verified']
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True, CPU_only=True, JIT_executed=False, GPU_executed=False,
        actual_PR90_compile_archive_sha256=ARCHIVE_SHA, actual_native_artifact_replay=True,
        protocol_mock_only=True, call_key_uses_actual_pinned_binder=True,
        address_changes_do_not_force_cold_audits_in_mock=True, exact_codec_slice_pin=True,
        prelaunch_rejections=failures, postlaunch_no_retry=True, generated_kernel_math_changed=False,
        adapter_service_numeric_verified=False, model_verified=False, performance_verified=False)


if __name__ == '__main__': print(json.dumps(run(), indent=2))
