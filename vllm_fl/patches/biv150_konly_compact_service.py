"""Default-off private encoded-cache service binding; no import-time device work."""
import functools
import atexit
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

ENV = 'VLLM_FL_BIV150_KONLY_COMPACT_SERVICE'
_STATE = None
PINS = {
    'vllm_fl.patches.biv150_native_gqa': '86f8ca199ce632ccb3bbc251bb5c632e57da3373d8b87bd5f53334ef818a26c4',
    'vllm.v1.attention.backends.triton_attn': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'vllm.v1.attention.ops.triton_unified_attention': '54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f',
    'vllm.v1.attention.ops.triton_attention_helpers': '59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1',
    'vllm_fl.patches.biv150_gqa_decode': 'f1877ad06e8be4cef5960ef2d66d59045b4ac1ac0d7f2fe448618bbdec0d7f40',
    'vllm_fl.patches.biv150_prefill_q32': '3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b',
    'vllm_fl.patches.biv150_prefill_multi2': '669a2a4d7b0e5a08f28e3492fd8fa45999f48e5521be94791f103e0fecd6b1eb',
    'vllm_fl.patches.biv150_prefill_normalized': '957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1',
    'vllm_fl.patches.biv150_q_fragment_service': 'c3c5bb34086681918cc328a6bbc2b349dd9019899b730fc99f9b2dc02f3ae1a9',
}


def need(value, message):
    if not value: raise RuntimeError('K-only encoded installation: ' + message)


def install(runner_class, *, environ=None):
    """Explicit private diagnostic only. Actual service-shape numeric gates pending.

    Off does not import GPU/runtime helpers, inspect a target or mutate a binding.
    On is cold-worker only and rejects identity/target/ABI mismatches before bind.
    """
    global _STATE
    env = os.environ if environ is None else environ
    if env.get(ENV, '0') != '1': return dict(installed=False, reason='disabled')
    need(env.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE', '0') != '1', 'two cache formats enabled')
    from .biv150_konly_compact_identity import verify_package
    verify_package()
    need(_STATE is None, 'install once in a fresh worker')
    need(env.get('VLLM_FL_BIV150_STOCK_QK8_DIAGNOSTIC', '0') != '1', 'conflicting stock diagnostic')
    modules = {name: importlib.import_module(name) for name in PINS}
    for name, expected in PINS.items():
        need(hashlib.sha256(Path(modules[name].__file__).read_bytes()).hexdigest() == expected, 'source identity ' + name)
    backend = modules['vllm.v1.attention.backends.triton_attn']
    attention = modules['vllm.v1.attention.ops.triton_unified_attention']
    gqa = modules['vllm_fl.patches.biv150_gqa_decode']
    fragment = modules['vllm_fl.patches.biv150_q_fragment_service']
    need(attention.current_platform.vendor_name == 'iluvatar' and not attention.is_batch_invariant, 'target platform')
    need(backend.unified_attention is attention.unified_attention, 'backend/SDK function alias')
    need(gqa._state['calls'] == gqa._state['fallbacks'] == 0, 'worker has already executed attention')
    launcher = gqa._launch_producer
    need(inspect.isfunction(launcher) and launcher.__globals__ is vars(gqa)
         and launcher.__code__.co_name == '_launch_producer' and not launcher.__closure__
         and not hasattr(launcher,'__wrapped__'), 'exact new main producer launcher required')
    native_parent = gqa._native_s2
    if env.get('VLLM_FL_BIV150_NATIVE_GQA','1') == '1':
        need(type(native_parent) is modules['vllm_fl.patches.biv150_native_gqa']._Runtime,
             'default-on main native GQA parent was not installed')
    base = fragment._MANAGER
    need(base is not None and base.active, 'main normalized/Q-fragment manager required')
    q32 = modules['vllm_fl.patches.biv150_prefill_q32']
    multi2 = modules['vllm_fl.patches.biv150_prefill_multi2']
    outer = base.holder.get(base.kernel_key)
    need(type(outer) is multi2._Policy and outer is multi2._POLICY
         and outer.manager is multi2._MANAGER and outer.manager.active
         and type(outer.base) is q32._Policy and outer.base.manager is q32._MANAGER
         and outer.base.base is base.wrapper, 'exact enabled main M2/Q32 parent chain required')

    need(hasattr(runner_class, 'initialize_kv_cache'), 'actual runner initialize seam')
    torch = importlib.import_module('torch'); triton = importlib.import_module('triton')
    target = triton.runtime.driver.active.get_current_target()
    need((target.backend, target.arch, target.warp_size) == ('corex', 71, 64), 'actual CoreX target')
    package = Path(__file__).resolve().parent
    provenance = json.loads((package / 'biv150_konly_compact_sources.json').read_text())
    gpu = {}; generated = provenance['generated']
    # Source files are independent of upstream paths and use their own JIT caches.
    for key in ('writer','decode','normalized2d128','q32','multi2','raw3d'):
        path = package / Path(generated[key]['path']).name
        need(hashlib.sha256(path.read_bytes()).hexdigest() == generated[key]['sha256'], 'private source ' + key)
        module = importlib.import_module('vllm_fl.patches.' + path.stem)
        name = 'encode_append' if key == 'writer' else '_gqa_decode_konly_i8' if key == 'decode' else 'kernel_cjj_konly_' + key
        fn = getattr(module, name)
        need(not fn.pre_run_hooks, 'foreign JIT hook ' + key)
        gpu[key] = fn
    need(len({id(fn.device_caches) for fn in gpu.values()}) == len(gpu), 'private JIT caches alias')
    need(all(fn.device_caches is not base.raw.device_caches and fn.device_caches is not gqa._gqa_decode_attention.device_caches
             for fn in gpu.values()), 'private JIT cache aliases main')
    view_path = package / Path(generated['views']['path']).name
    need(hashlib.sha256(view_path.read_bytes()).hexdigest() == generated['views']['sha256'], 'typed view broker source')
    broker = importlib.import_module('vllm_fl.patches.biv150_konly_compact_views')
    runtime_module = importlib.import_module('vllm_fl.patches.biv150_konly_compact_runtime')
    runtime = runtime_module.Runtime(torch, broker, gpu, gqa)
    from .biv150_konly_compact_host import Host
    host = Host(runner_class, torch, runtime, attention.current_platform)
    runtime.compact_host = host
    runtime.actual_target = [target.backend,target.arch,target.warp_size]
    runtime.verified_installed_sources = {**PINS, **host.verified_core_pins}
    if env.get('VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST', '0') == '1':
        adapter = importlib.import_module('vllm_fl.patches.biv150_konly_compact_q32_adapter')
        proposal = importlib.import_module('vllm_fl.patches.biv150_konly_compact_hoist_proposal')
        native = importlib.import_module('vllm_fl.patches.biv150_konly_encoded_hoist_native')
        cache = env.get('TRITON_CACHE_DIR')
        need(cache is not None and Path(cache).is_absolute(), 'hoist needs absolute private TRITON_CACHE_DIR')
        fn = gpu['q32']; source_path = package / Path(generated['q32']['path']).name
        runtime.q32_hoist = adapter.Adapter(proposal, native,
            directory=Path(cache).parent/'konly-compact-q32-hoist',
            target=dict(backend=target.backend, arch=target.arch, warp_size=target.warp_size),
            pins=adapter.Adapter.installed_pins(fn, source_path, PINS),
            implementation_digest=implementation_digest(), jit=inspect.getmodule(type(fn)),
            device=triton.runtime.driver.active.get_current_device(), fn=fn,
            source_module=importlib.import_module('vllm_fl.patches.biv150_konly_compact_q32'),
            source_sha256=generated['q32']['sha256'])
    runtime.q32_hoist_identity = runtime.q32_hoist
    runtime.stats_path = env.get('VLLM_FL_BIV150_KONLY_COMPACT_STATS_PATH')
    if runtime.stats_path is None and env.get('TRITON_CACHE_DIR'):
        runtime.stats_path = str(Path(env['TRITON_CACHE_DIR']).parent/'konly-compact-final-{pid}.json')
    if runtime.stats_path is not None:
        need(Path(runtime.stats_path).is_absolute(),'private stats output must be absolute')

    cls = backend.TritonAttentionImpl
    private_gqa = runtime.facade(gqa._gqa_decode_attention, 'gqa')
    producer = runtime.launch_producer(launcher,private_gqa)
    jobs = [
        (vars(gqa), '_launch_producer', producer),
        (vars(gqa), '_gqa_decode_attention', private_gqa),
        (vars(attention), 'kernel_unified_attention', runtime.facade(attention.kernel_unified_attention, 'sdk')),
        (base.holder, base.kernel_key, runtime.facade(base.holder[base.kernel_key], 'normalized')),
    ]
    attributes = [(cls, 'do_kv_cache_update', runtime.writer(cls.do_kv_cache_update)),
                  (runner_class, 'initialize_kv_cache', host.initialize(runtime.initialize(runner_class.initialize_kv_cache))),
                  (runner_class, 'shutdown', runtime.shutdown(runner_class.shutdown))]
    # Disable only this private backend's unqualified fused writer before FX
    # compilation/capture. The existing separate RoPE->custom-op writer survives.
    def no_fused(self): return False
    def reject_fused(*args, **kwargs): raise RuntimeError('K-only encoded fused RoPE/cache writer is not qualified')
    attributes += [(cls, 'fused_rope_kvcache_supported', no_fused), (cls, 'do_rope_and_kv_cache_update', reject_fused)]
    # No connector is admitted by Runtime.initialize. Fence known byte-transfer
    # seams too; no unverified encoded transfer falls through to stock semantics.
    custom = importlib.import_module('vllm._custom_ops')
    for name in ('swap_blocks', 'copy_blocks'):
        if hasattr(custom, name): jobs.append((vars(custom), name, runtime.forbid_transfer(getattr(custom, name))))
    platform = importlib.import_module('vllm_fl.platform')
    decl = vars(platform.PlatformFL).get('copy_kv_blocks')
    if isinstance(decl, classmethod):
        attributes.append((platform.PlatformFL, 'copy_kv_blocks', classmethod(runtime.forbid_transfer(decl.__func__))))
    elif isinstance(decl, staticmethod):
        attributes.append((platform.PlatformFL, 'copy_kv_blocks', staticmethod(runtime.forbid_transfer(decl.__func__))))
    elif decl is not None:
        attributes.append((platform.PlatformFL, 'copy_kv_blocks', runtime.forbid_transfer(decl)))
    jobs.append((vars(host.module), '_reshape_attention_kv_cache', host.reshape_hook))
    attributes.append((runner_class, 'get_kv_cache_spec', host.get_spec_hook))
    seen = set(); old = []; attrs = []
    for holder, key, _ in jobs:
        identity = (id(holder), key); need(identity not in seen, 'duplicate binding seam'); seen.add(identity)
    try:
        for holder, key, value in jobs:
            old.append((holder, key, holder[key])); holder[key] = value
        for owner, key, value in attributes:
            attrs.append((owner, key, vars(owner).get(key))); setattr(owner, key, value)
    except BaseException:
        for owner, key, value in reversed(attrs):
            if value is None: delattr(owner, key)
            else: setattr(owner, key, value)
        for holder, key, value in reversed(old): holder[key] = value
        raise
    runtime.facade_bindings = [(holder, key, value, previous) for (holder, key, value), (_, _, previous) in zip(jobs, old)
                               if isinstance(value, runtime_module.KernelFacade)]
    runtime.attribute_bindings = [(owner, key, value) for owner, key, value in attributes]
    runtime.producer_binding = (gqa,producer,launcher,native_parent,producer.__code__)
    runtime.facade_type = runtime_module.KernelFacade
    runtime.facade_code = runtime_module.KernelFacade.__getitem__.__code__
    runtime.profile_identities = dict(runtime.kernels)
    runtime.installed_digest = implementation_digest()
    _STATE = runtime
    atexit.register(runtime.finalize,'atexit_fallback')
    return dict(installed=True, default_off=True, scope='cold MiniCPM BF16 NHD single-device private encoded lifetime',
        current_profiles=['decode79','normalized2d128','q32-unhoisted','multi2-unhoisted','raw3d-q1'],
        encoded_fallback='format-aware all-sequence normalized2D, never stock BF16',
        multi2_and_raw3d_source_routes_enabled=True, arbitrary_service_geometry_verified=False, mixed85_pair_enabled=False, live_slot_freshness='inherited scheduler/custom-op/stream contract',
        implementation_sha256=runtime.installed_digest,final_stats_path=runtime.stats_path,
        stock_native_gqa_parent_preserved=True, encoded_GQA_uses_stock_native=False,
        producer_boundary='owned-encoded-private-before-stock-native',
        q32_hoist_enabled=runtime.q32_hoist is not None, q32_I8_fragment_hoist_verified=False, service_numeric_verified=False, precision_gate=False)


def stats():
    return dict(installed=False) if _STATE is None else dict(installed=True, **_STATE.stats())


def implementation_digest(package=None):
    directory = Path(__file__).resolve().parent if package is None else Path(package)
    paths = sorted(directory.glob('biv150_konly_compact_*.py')) + [directory/'biv150_konly_compact_sources.json'] + [directory/name for name in
        ('biv150_konly_encoded_q32_adapter.py','biv150_konly_encoded_hoist_native.py')]
    records = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def compute_order_receipt(attention, normalized_kernel):
    """Validate our actual dispatch separately; parent .fn is interface-only.

    This does not certify new cache math/numerics. Reordering remains the original
    metadata permutation; codecs reduce within one token/head, never across rows.
    """
    state = _STATE
    need(state is not None and implementation_digest() == state.installed_digest, 'unregistered/stale encoded runtime')
    need(state.facade_type.__getitem__.__code__ is state.facade_code, 'encoded dispatch code changed')
    need(state.q32_hoist is getattr(state,'q32_hoist_identity',None), 'encoded Q32 adapter binding changed')
    boundary = getattr(state,'producer_binding',None)
    need(type(boundary) is tuple and len(boundary)==5, 'new main producer boundary receipt absent')
    gqa,producer,parent,native_parent,code = boundary
    need(gqa._launch_producer is producer and producer.__code__ is code
         and producer.__wrapped__ is parent and gqa._native_s2 is native_parent,
         'format-aware producer or preserved native parent changed')
    need(set(state.kernels) == set(state.profile_identities) and all(state.kernels[k] is v for k,v in state.profile_identities.items()), 'private profile binding changed')
    host = state.compact_host
    need(host.initialized, 'compact host epoch not initialized')
    host.receipt(host.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec)
    parents = {}
    for holder,key,facade,original in state.facade_bindings:
        need(type(facade) is state.facade_type and holder.get(key) is facade and facade.runtime is state and facade.original is original,
             'encoded facade identity/binding changed')
        parents[id(facade)] = original
    for owner,key,value in state.attribute_bindings:
        need(vars(owner).get(key) is value, 'encoded writer/lifetime/transfer binding changed')
    need(id(attention.kernel_unified_attention) in parents and id(normalized_kernel) in parents,
         'unknown encoded raw/normalized facade')
    need(all(e.observation.storage_id in state.registry.owned_storages for e in state.registry.entries.values()),
         'cache ownership registry incomplete')
    return dict(parent_raw=parents[id(attention.kernel_unified_attention)], parent_normalized=parents[id(normalized_kernel)],
        encoded_dispatch_registered=True, implementation_sha256=state.installed_digest,
        producer_boundary_registered=True, stock_native_parent_preserved=True, encoded_GQA_uses_stock_native=False,
        parent_fn_is_interface_only=True, encoded_math_device_certified=False)


def compact_spec_is_owned(spec):
    need(_STATE is not None, 'compact runtime missing')
    _STATE.compact_host.receipt(spec)
    return True
