#!/usr/bin/env python3
"""CPU metadata/call-routing fixtures, never a GPU numeric or lifetime proof."""
import builtins
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    p = ROOT / 'vllm_fl/patches' / ('biv150_konly_encoded_' + name + '.py')
    spec = importlib.util.spec_from_file_location('cjj_encoded_cpu_' + name, p)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


class Storage:
    def __init__(self, identity, address, size): self._cdata, self.address, self.size = identity, address, size
    def data_ptr(self): return self.address
    def nbytes(self): return self.size


class Tensor:
    widths = {'bfloat16': 2, 'float32': 4, 'int8': 1, 'int64': 8, 'int32': 4}
    def __init__(self, storage, dtype, shape, strides, offset=0):
        self.storage, self.dtype, self.shape, self.strides, self.offset = storage, 'torch.' + dtype, tuple(shape), tuple(strides), offset
        self.device = 'cuda:0'; self.ndim = len(shape)
    def untyped_storage(self): return self.storage
    def data_ptr(self): return self.storage.address + self.offset * self.widths[self.dtype.removeprefix('torch.')]
    def stride(self, dim=None): return self.strides if dim is None else self.strides[dim]
    def storage_offset(self): return self.offset
    def set_(self, storage, offset, shape, strides):
        self.storage, self.offset, self.shape, self.strides = storage, offset, tuple(shape), tuple(strides)
        self.ndim = len(shape); return self
    def cpu(self): raise AssertionError('No device-data readback permitted')
    def item(self): raise AssertionError('No scalar data readback permitted')


class Kernel:
    def __init__(self): self.calls = []; self.pre_run_hooks = []
    def __getitem__(self, grid):
        def invoke(*args, **kwargs): self.calls.append((grid, args, kwargs)); return SimpleNamespace(hash='cpu-mock-only')
        return invoke


def rejected(call):
    try: call()
    except (RuntimeError, ValueError): return
    raise AssertionError('Expected contract rejection')


def run():
    service = load('service'); calls = []
    old_import = service.importlib.import_module
    def no_import(name): calls.append(name); raise AssertionError(name)
    service.importlib.import_module = no_import
    try: assert service.install(object, environ={}) == dict(installed=False, reason='disabled') and not calls
    finally: service.importlib.import_module = old_import
    runtime_module, broker = load('runtime'), load('views')
    capture = [False]
    torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: capture[0]))
    for kind in Tensor.widths: setattr(torch, kind, kind)
    def empty(_size, *, dtype, device):
        return Tensor(Storage(99999, 0, 0), dtype, (0,), (1,))
    torch.empty = empty
    kernels = {name: Kernel() for name in ('writer','decode','normalized2d128','q32','multi2','raw3d')}
    gqa = SimpleNamespace(_eligible=lambda *args: False)
    runtime = runtime_module.Runtime(torch, broker, kernels, gqa)
    cache = Tensor(Storage(1, 0x100000, 2 * 16384 + 512), 'bfloat16', (2,2,16,2,128), (8192,4096,256,128,1), 128)
    backing = Storage(2, 0x200000, 100000)
    key = Tensor(backing, 'bfloat16', (8,2,128), (2560,128,1), 2048)
    value = Tensor(backing, 'bfloat16', (8,2,128), (2560,128,1), 2304)
    slots = Tensor(Storage(3, 0x300000, 64), 'int64', (8,), (1,))
    impl = SimpleNamespace(num_heads=16, num_kv_heads=2, head_size=128, _kv_quant_mode=0,
        _is_per_token_head_quant=False, kv_sharing_target_layer_name=None)
    rejected(lambda: runtime.registry.claim(impl, cache, key, value))
    runtime.registry.armed = True
    writer = runtime.writer(lambda *a: (_ for _ in ()).throw(AssertionError('stock writer reached')))
    assert writer(impl, None, key, value, cache, slots) is None
    entry = next(iter(runtime.registry.entries.values()))
    assert entry.observation.pointer == 0x100100
    views = entry.roles['decode79']
    assert views['key_cache_ptr'].data_ptr() == views['key_scale_ptr'].data_ptr() == cache.data_ptr()
    assert views['value_cache_ptr'].data_ptr() == cache.data_ptr() + 8192
    assert views['key_cache_ptr'].stride(0) == 16384 and views['value_cache_ptr'].stride(0) == 8192
    assert kernels['writer'].calls[-1][2]['key_row_stride'] == kernels['writer'].calls[-1][2]['value_row_stride'] == 2560
    assert runtime.registry.claim(impl, cache, key, value) is entry
    kmeta = Tensor(cache.storage, 'bfloat16', (2,16,2,128), (8192,256,128,1), 128)
    vmeta = Tensor(cache.storage, 'bfloat16', (2,16,2,128), (8192,256,128,1), 128 + 4096)
    assert runtime.registry.lookup(kmeta, vmeta) is entry
    bad_alias = Tensor(cache.storage, 'bfloat16', (2,16,2,128), (8192,256,128,1), 129)
    rejected(lambda: runtime.registry.lookup(bad_alias, vmeta))
    replaced = Tensor(Storage(4, 0x400000, 32768), 'bfloat16', cache.shape, cache.strides)
    rejected(lambda: runtime.registry.claim(impl, replaced, key, value))
    rejected(lambda: runtime.registry.claim(SimpleNamespace(), cache, kmeta, value))
    capture[0] = True
    rejected(lambda: runtime.registry.claim(SimpleNamespace(), replaced, key, value)); capture[0] = False
    q = Tensor(Storage(5, 0x500000, 4096 * 2048 * 2), 'bfloat16', (4096,16,128), (2048,128,1))
    original = Kernel(); facade = runtime.facade(original, 'normalized')
    base = dict(query_ptr=q, key_cache_ptr=kmeta, value_cache_ptr=vmeta, num_seqs=1, IS_3D=False,
        HEAD_SIZE=128, HEAD_SIZE_PADDED=128, BLOCK_SIZE=16, num_query_heads=16,num_queries_per_kv=8,
        USE_CAUSAL=True,SLIDING_WINDOW=0,CHUNK_LOOKBACK=-1,CHUNK_SIZE=-1,KV_QUANT_MODE=0,
        **{k:False for k in ('USE_TD','USE_TD_QO','Q_IS_FP8','USE_FP8','USE_PER_SEQ_CAUSAL','USE_MM_PREFIX',
                             'USE_ALIBI_SLOPES','USE_ALIBI_SQRT','USE_QQ_BIAS','USE_SOFTCAP','USE_SINKS')})
    facade[(1,2)](**base)
    grid, _, kw = kernels['q32'].calls[-1]
    assert grid == (129,2) and kw['BLOCK_M'] == 256 and kw['BLOCK_Q'] == 32 and kw['num_warps'] == 16
    assert kw['key_code_scale_ptr'].data_ptr() == cache.data_ptr() and kw['value_cache_ptr'].data_ptr() == cache.data_ptr()+8192
    runtime.multi2_enabled=False;facade[(1,2)](**dict(base,num_seqs=2))
    assert kernels['normalized2d128'].calls[-1][0] == (258,2)
    assert runtime.counts['profile_fallback']['multi2_to_encoded_2d_pending_numeric'] == 1
    runtime.multi2_enabled=True;facade[(1,2)](**dict(base,num_seqs=2))
    assert kernels['multi2'].calls[-1][0] == (130,2)
    facade[(1,2)](**dict(base,num_seqs=64))
    assert kernels['normalized2d128'].calls[-1][0] == (320,2)
    runtime.raw3d_enabled=False;rejected(lambda:facade[(1,2)](**dict(base,IS_3D=True)))
    rejected(lambda:facade[(1,2)](**dict(base,USE_SINKS=True)))
    gqa_facade=runtime.facade(Kernel(),'gqa')
    gqa_args=dict(query_ptr=q,key_cache_ptr=kmeta,value_cache_ptr=vmeta,HEAD_SIZE=128,PAGE_SIZE=16,ROWS=8,queries_per_kv=8)
    gqa_facade[(8,2,16)](**gqa_args)
    assert kernels['decode'].calls[-1][2]['k_stride_block']==16384
    assert kernels['decode'].calls[-1][2]['v_stride_block']==8192
    rejected(lambda:gqa_facade[(1,2)](**dict(gqa_args,HEAD_SIZE=64)))
    # Explicit dispatch receipt checks actual facades; parent .fn is never a
    # numerical certificate. Wrong binding/type/digest/profile objects reject.
    raw=runtime.facade(Kernel(),'sdk');norm=runtime.facade(Kernel(),'normalized')
    holders=[{'raw':raw},{'normal':norm}]
    runtime.facade_bindings=[(holders[0],'raw',raw,raw.original),(holders[1],'normal',norm,norm.original)]
    runtime.attribute_bindings=[];runtime.facade_type=runtime_module.KernelFacade
    runtime.facade_code=runtime_module.KernelFacade.__getitem__.__code__
    runtime.profile_identities=dict(runtime.kernels);runtime.installed_digest=service.implementation_digest()
    gqa._native_s2=object()
    parent=lambda grid,**kwargs:None
    producer=runtime.launch_producer(parent,gqa_facade)
    gqa._launch_producer=producer
    runtime.producer_binding=(gqa,producer,parent,gqa._native_s2,producer.__code__)
    service._STATE=runtime; attention=SimpleNamespace(kernel_unified_attention=raw)
    receipt=service.compute_order_receipt(attention,norm)
    assert receipt['parent_fn_is_interface_only'] and not receipt['encoded_math_device_certified']
    holders[0]['raw']=norm;rejected(lambda:service.compute_order_receipt(attention,norm));holders[0]['raw']=raw
    previous=runtime.kernels['decode'];runtime.kernels['decode']=Kernel()
    rejected(lambda:service.compute_order_receipt(attention,norm));runtime.kernels['decode']=previous
    runtime.installed_digest='wrong';rejected(lambda:service.compute_order_receipt(attention,norm))
    runtime.installed_digest=service.implementation_digest()
    previous_native=gqa._native_s2;gqa._native_s2=object()
    rejected(lambda:service.compute_order_receipt(attention,norm));gqa._native_s2=previous_native
    gqa._launch_producer=parent
    rejected(lambda:service.compute_order_receipt(attention,norm));gqa._launch_producer=producer
    service._STATE=None
    rejected(lambda: runtime.forbid_transfer(lambda x: None)([cache]))
    assert runtime.forbid_transfer(lambda x: 3)([q]) == 3
    assert runtime.counts['quantization_in_forward'] == 0 and not runtime.stats()['dynamic_I8_hoist_verified']
    text = (ROOT/'vllm_fl/patches/biv150_konly_encoded_runtime.py').read_text()
    assert '.cpu(' not in text and '.item(' not in text
    import hashlib, subprocess
    def baseline(path): return subprocess.check_output(['git','-C',str(ROOT),'show','b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd:'+path])
    path='vllm_fl/worker/model_runner.py'; before=baseline(path); after=(ROOT/path).read_bytes()
    assert after.startswith(before); addition=after[len(before):]
    assert addition.count(b'\n')==addition.count(b'\r\n') and addition.count(b'import as')==0
    for on in (False,True):
        captured=[]; imported=[]
        def importer(name,globals=None,locals=None,fromlist=(),level=0):
            imported.append(name)
            if name=='os': return SimpleNamespace(environ={service.ENV:'1' if on else '0'})
            if name=='vllm_fl.patches.biv150_konly_encoded_service':
                return SimpleNamespace(install=lambda cls: captured.append(cls) or {'installed':True,'CPU_mock_only':True})
            raise AssertionError('Unexpected tail import '+name)
        runner=type('ActualTailRunnerCPUFixture',(),{})
        namespace=dict(current_platform=SimpleNamespace(vendor_name='iluvatar'),ModelRunnerFL=runner,
            __builtins__={**vars(builtins),'__import__':importer,'print':lambda *a,**kw:None})
        assert 'os' not in namespace
        exec(compile(addition.decode(),'<actual-new-worker-tail-cpu-only>','exec'),namespace)
        assert captured==([runner] if on else [])
        assert imported==(['os','vllm_fl.patches.biv150_konly_encoded_service'] if on else ['os'])
    from tempfile import TemporaryDirectory
    patch=(ROOT/'docs/evidence/konly-encoded-service-20261027/production-changes-v3.patch').read_bytes()
    names=['vllm_fl/worker/model_runner.py','vllm_fl/patches/biv150_compute_order.py','vllm_fl/compile_semantics.py']
    exact=subprocess.check_output(['git','-C',str(ROOT),'diff','--binary','--full-index','b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd','--',*names])
    assert exact==patch
    with TemporaryDirectory(prefix='cjj-encoded-inverse-') as temp:
        directory=Path(temp)
        for name in names:
            target=directory/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        source_patch=directory/'changes.patch';source_patch.write_bytes(patch)
        subprocess.run(['git','-C',str(directory),'apply','--reverse',str(source_patch)],check=True,capture_output=True)
        assert all((directory/name).read_bytes()==baseline(name) for name in names)
    spec=importlib.util.spec_from_file_location('encoded_semantics_cpu',ROOT/'vllm_fl/compile_semantics.py')
    sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
    model=SimpleNamespace(**sem._GEOMETRY);config=SimpleNamespace(model_config=SimpleNamespace(hf_config=model))
    common=dict(vendor_name='iluvatar',abi={'torch':'CPU-fixture','cxx11_abi':False})
    with TemporaryDirectory(prefix='cjj-native-gqa-fingerprint-cpu-') as temp:
        library=Path(temp)/'CPU-fixture-not-loadable-native-gqa.so'
        library.write_bytes(b'CPU-only native GQA fingerprint fixture')
        env={'VLLM_FL_BIV150_NATIVE_W8A8':'0','BIV150_NATIVE_GQA_LIBRARY':str(library)}
        off=sem.build_fingerprint(config,environ=env,**common)
        on=sem.build_fingerprint(config,environ=dict(env,**{service.ENV:'1'}),**common)
        assert off['sha256']!=on['sha256'] and 'konly_encoded' not in off
        assert on['konly_encoded']['implementation_sha256']==service.implementation_digest()
        assert on['konly_encoded']['producer_boundary']=='owned-encoded-before-stock-native'
        hoist=sem.build_fingerprint(config,environ=dict(env,**{service.ENV:'1',
            'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':'1'}),**common)
        assert not on['konly_encoded']['q32_hoist_enabled'] and hoist['konly_encoded']['q32_hoist_enabled']
        assert hoist['sha256'] != on['sha256']
        native_off=sem.build_fingerprint(config,environ=dict(env,VLLM_FL_BIV150_NATIVE_GQA='0'),**common)
        assert native_off['sha256']!=off['sha256']
        library.write_bytes(b'changed CPU fingerprint fixture, no library loading')
        assert sem.build_fingerprint(config,environ=env,**common)['sha256']!=off['sha256']
    # Export is shutdown-only, atomic and one-shot; never claims GPU replay counts.
    with TemporaryDirectory(prefix='cjj-encoded-stats-cpu-') as folder:
        runtime.stats_path=Path(folder)/'final-{pid}.json'
        result=runtime.shutdown(lambda *a: 'original-shutdown-result')(object())
        assert result=='original-shutdown-result'
        output=next(Path(folder).glob('final-*.json'));first=output.read_bytes()
        payload=json.loads(first)
        assert payload['reason']=='model_runner_shutdown' and payload['shape_records_complete']
        assert not payload['graph_kernel_replay_counts_available'] and not payload['service_numeric_verified']
        assert payload['host_shapes'] and payload['actual_kernel_records']
        assert all(not r['post_launch_resource_fields_known'] for r in payload['actual_kernel_records'])
        assert runtime.finalize('atexit_fallback')==payload and output.read_bytes()==first
        assert not list(Path(folder).glob('*.tmp-*'))
    # Type-preserving ABI metadata never reprs or reads a nonprimitive object.
    class NoRepr:
        def __repr__(self):raise AssertionError('metadata repr forbidden')
    assert runtime.typed(NoRepr())['unrepresented']
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True,cpu_metadata_routing_only=True,source_and_slot_readback=False,
        source_destination_distinct=True,nonzero_layer_offset_checked=True,Kbyte_Vbf16_units_separate=True,
        default_off_no_import=True,stock_writer_not_called=True,encoded_stock_fallbacks=0,
        reallocation_alias_capture_transfer_rejections=True,profile_grid_contracts=True,
        gpu_executed=False,service_numeric_verified=False,precision_gate=False)


if __name__ == '__main__':
    import contextlib,io
    with contextlib.redirect_stdout(io.StringIO()) as capture:
        result=run()
    result['synthetic_selected_host_events']=len(capture.getvalue().splitlines())
    print(json.dumps(result,indent=2))
