#!/usr/bin/env python3
"""Actual full modules/CodeTypes and ABI calls with explicit CPU-only handles.

No framework import, compiler, device, native/IR qualification or model claim.
"""
import argparse
import ast
from copy import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace as S
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'vllm_fl/patches'
for name, path in [('vllm_fl',ROOT/'vllm_fl'),('vllm_fl.patches',PACKAGE)]:
    m=types.ModuleType(name);m.__path__=[str(path)];m.__file__=str(path/'__init__.py');sys.modules[name]=m


def load(name):return importlib.import_module('vllm_fl.patches.biv150_konly_'+name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid compact binding admitted')


class Storage:
    def __init__(self, identity, base, size):self._cdata,self.base,self.size=identity,base,size
    def data_ptr(self):return self.base
    def nbytes(self):return self.size


class Tensor:
    widths={'bfloat16':2,'int8':1,'float32':4}
    def __init__(self,storage,dtype,shape,strides,offset=0):
        self.storage,self.dtype,self.shape,self.strides,self.offset=storage,'torch.'+dtype,shape,strides,offset
        self.device='cuda:0'
    def untyped_storage(self):return self.storage
    def storage_offset(self):return self.offset
    def data_ptr(self):return self.storage.base+self.offset*self.widths[self.dtype[6:]]
    def stride(self):return self.strides
    def set_(self,storage,offset,shape,strides):
        self.storage,self.offset,self.shape,self.strides=storage,offset,shape,strides;return self
    def __eq__(self,other):raise AssertionError('Opaque Tensor equality')
    def item(self):raise AssertionError('GPU scalar read')
    def cpu(self):raise AssertionError('GPU data read')


class Kernel:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def call(**kw):self.calls.append((grid,kw));return S(hash='CPU-protocol-mock')
        return call


def run():
    a=load('compact_q32_adapter');base=load('encoded_q32_adapter')
    proposal=load('compact_hoist_proposal');native=load('encoded_hoist_native')
    runtime_mod=load('compact_runtime');broker=load('compact_views');identity=load('compact_identity')
    manifest=identity.verify_package()
    old=subprocess.check_output(['git','-C',str(ROOT),'show','5ca6a54:vllm_fl/patches/biv150_konly_encoded_hoist_proposal.py'])
    now=(PACKAGE/'biv150_konly_compact_hoist_proposal.py').read_bytes()
    assert old.replace(b"SCHEMA='cjj-konly-service-q32-hoist-v1'",b"SCHEMA='cjj-konly-compact-q32-hoist-v1'").replace(
        b"FORMAT='cjj-konly-rn32-rne-v2'",b"FORMAT='cjj-konly-compact12544-rn32-rne-v1'")==now
    for role in ('writer','decode','normalized2d128','q32','multi2','raw3d'):
        path='vllm_fl/patches/biv150_konly_compact_'+role+'.py'
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','5ca6a54:'+path])
    # Preserve complete actual CodeType fields, not a partial AST reconstruction.
    for module in (a,base,proposal,native,runtime_mod):
        path=Path(module.__file__)
        code=compile(path.read_bytes(),str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)
        for key,value in list(vars(module).items()):
            if isinstance(value,types.FunctionType) and value.__module__==module.__name__:
                a.same_code(value.__code__,a.code_at(code,key))
            elif isinstance(value,type) and value.__module__==module.__name__:
                if value.__name__=='Entry':continue  # Generated dataclass methods.
                for name,fn in vars(value).items():
                    if isinstance(fn,staticmethod):fn=fn.__func__
                    if isinstance(fn,types.FunctionType):a.same_code(fn.__code__,a.code_at(code,key,name))
    for name in ('prepare','_prepare','after_launch','jit_key','call_key'):
        assert getattr(a.Adapter,name).__code__ is getattr(base.Adapter,name).__code__
    source=PACKAGE/'biv150_konly_compact_q32.py'
    source_module=types.ModuleType('CPU_private_compact_q32');source_module.__file__=str(source)
    full=compile(source.read_bytes(),str(source),'exec',dont_inherit=True,optimize=sys.flags.optimize)
    for name in ('_cjj_konly_scale','quantize_bf16_rne','kernel_cjj_konly_q32'):
        fn=types.FunctionType(a.code_at(full,name),vars(source_module))
        setattr(source_module,name,S(fn=fn))
    fn=source_module.kernel_cjj_konly_q32
    fn.src=ast.get_source_segment(source.read_text(),next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_cjj_konly_q32'))
    fn.pre_run_hooks=[];fn.device_caches={};fn.debug=False
    kernel=Kernel()
    class Function:
        def __getitem__(self,grid):return kernel[grid]
        def warmup(self,**kw):raise RuntimeError('explicit CPU compiler-unavailable stand-in')
    selected_fn=Function();selected_fn.__dict__.update(vars(fn));source_module.kernel_cjj_konly_q32=selected_fn
    fn=selected_fn
    capture=[False]
    torch=S(cuda=S(is_current_stream_capturing=lambda:capture[0]),int8='int8',float32='float32',bfloat16='bfloat16')
    torch.empty=lambda n,*,dtype,device:Tensor(Storage(999,0,0),dtype,(0,),(1,))
    kernels={k:Kernel() for k in ('writer','decode','normalized2d128','multi2','raw3d')};kernels['q32']=fn
    runtime=runtime_mod.Runtime(torch,broker,kernels,S());runtime.registry.armed=True
    class NoDecode:
        def try_produce(self,*args):raise AssertionError('Prefill reached NativeK8 decode adapter')
        def stats(self):return {'CPU_poison_route_only':True}
    runtime.native_decode=NoDecode()
    storage=Storage(1,0x100000,3*12544+512)
    owner=Tensor(storage,'bfloat16',(3,2,16,2,128),(6272,2176,256,128,1),128)
    source_storage=Storage(2,0x1000000,32768)
    key_source=Tensor(source_storage,'bfloat16',(3,2,128),(2560,128,1))
    value_source=Tensor(source_storage,'bfloat16',(3,2,128),(2560,128,1),256)
    entry=runtime.registry.claim(S(),owner,key_source,value_source)
    roles=entry.roles['prefill80']
    key=Tensor(storage,'bfloat16',(3,16,2,128),(6272,256,128,1),128)
    value=Tensor(storage,'bfloat16',(3,16,2,128),(6272,256,128,1),128+2176)
    query=Tensor(Storage(3,0x2000000,2048*4096),'bfloat16',(2048,16,128),(2048,128,1))
    launch=dict(query_ptr=query,key_cache_ptr=key,value_cache_ptr=value,HEAD_SIZE=128,
        HEAD_SIZE_PADDED=128,BLOCK_SIZE=16,num_query_heads=16,num_queries_per_kv=8,
        num_seqs=1,IS_3D=False,USE_CAUSAL=True,SLIDING_WINDOW=0,CHUNK_LOOKBACK=-1,CHUNK_SIZE=-1,KV_QUANT_MODE=0)
    launch.update({k:False for k in ('USE_TD','USE_TD_QO','Q_IS_FP8','USE_FP8','USE_PER_SEQ_CAUSAL',
        'USE_MM_PREFIX','USE_ALIBI_SLOPES','USE_ALIBI_SQRT','USE_QQ_BIAS','USE_SOFTCAP','USE_SINKS')})
    class Parent(Kernel):
        def __getitem__(self,grid):
            def bad(**kw):raise AssertionError('Owned compact reached stock BF16 parent')
            return bad
    facade=runtime.facade(Parent(),'sdk')
    facade[(99,99)](**launch)
    grid,selected=kernel.calls[-1]
    assert grid==(65,2) and selected['key_cache_ptr'] is roles['key_cache_ptr']
    assert selected['value_cache_ptr'] is roles['value_cache_ptr']
    assert selected['key_code_scale_ptr'].data_ptr()==owner.data_ptr()
    assert selected['value_cache_ptr'].data_ptr()==owner.data_ptr()+4352
    # All physical pages/slots/head/dim use exactly the same role equations.
    addresses=0
    for p in range(3):
        for slot in range(16):
            for h in range(2):
                for d in range(128):
                    actual=broker.element_addresses(entry.observation,'prefill80',p,slot,h,d)
                    expected=(owner.data_ptr()+p*12544+slot*256+h*128+d,
                              owner.data_ptr()+p*12544+4096+slot*8+h*4,
                              owner.data_ptr()+p*12544+4352+slot*512+h*256+d*2)
                    assert actual==expected;addresses+=1
    with tempfile.TemporaryDirectory(prefix='cjj-compact-q32-host-') as tmp:
        jit=S(driver=S(active=S(get_current_device=lambda:0)))
        adapter=a.Adapter(proposal,native,directory=tmp,target=dict(backend='corex',arch=71,warp_size=64),
            pins=proposal.PINS,implementation_digest='CPU-protocol-only',jit=jit,device=0,fn=fn,
            source_module=source_module,source_sha256=sha(source.read_bytes()))
        runtime.q32_hoist=adapter
        adapter.validate_call(runtime,entry,selected)
        # A new pointer never replaces format or epoch validation.
        count=0
        for field,delta in [('key_code_scale_ptr',1024),('value_cache_ptr',2176),('key_cache_ptr',1)]:
            tensor=roles[field];old_offset=tensor.offset;tensor.offset+=delta
            reject(lambda:adapter.validate_call(runtime,entry,selected));tensor.offset=old_offset;count+=1
        old_generation=entry.generation;entry.generation=0
        reject(lambda:adapter.validate_call(runtime,entry,selected));entry.generation=old_generation;count+=1
        entry.generation+=1;runtime.registry.generation+=1
        reject(lambda:adapter.validate_call(runtime,entry,selected));entry.generation=old_generation;runtime.registry.generation-=1;count+=1
        old_owner=owner.storage;owner.storage=Storage(40,0x300000,old_owner.size)
        reject(lambda:adapter.validate_call(runtime,entry,selected));owner.storage=old_owner;count+=1
        runtime.registry.armed=False
        reject(lambda:adapter.validate_call(runtime,entry,selected));runtime.registry.armed=True;count+=1
        old_helper=source_module._cjj_konly_scale;source_module._cjj_konly_scale=copy(old_helper)
        reject(lambda:adapter.validate_call(runtime,entry,selected));source_module._cjj_konly_scale=old_helper;count+=1
        original_code=old_helper.fn.__code__;old_helper.fn.__code__=original_code.replace(co_firstlineno=1)
        reject(lambda:a.guard_source(fn,source_module,sha(source.read_bytes())));old_helper.fn.__code__=original_code;count+=1
        wrong=types.FunctionType(original_code,{})
        old_helper.fn=wrong
        reject(lambda:a.guard_source(fn,source_module,sha(source.read_bytes())))
        old_helper.fn=adapter.source_binding[4];count+=1
        reject(lambda:a.guard_source(fn,source_module,'0'*64));count+=1
        # Actual Base.prepare retains the same private function before any reader
        # when the mocked binder/compiler is unavailable; no alternative reader.
        before=len(kernel.calls)
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):facade[(1,)](**launch)
        assert len(kernel.calls)==before+1 and kernel.calls[-1][1]['key_cache_ptr'] is roles['key_cache_ptr']
        assert 'ir_override' not in kernel.calls[-1][1]
        # Exact installed facade: postlaunch validator exception must propagate,
        # with one reader only. This is a protocol mock, not a native proof.
        adapter.prepare=lambda *x,**kw:({'ir_override':'CPU-mock-only'},{'mock':True})
        failure=RuntimeError('postlaunch sentinel')
        adapter.after_launch=lambda *x:(_ for _ in ()).throw(failure)
        before=len(kernel.calls)
        try:facade[(1,)](**launch)
        except RuntimeError as error:assert error is failure
        else:raise AssertionError('Postlaunch exception swallowed')
        assert len(kernel.calls)==before+1
        runtime.q32_hoist=None
        cases=[(1,1023,False,'normalized2d128',16),(2,1026,False,'multi2',32),
               (3,1024,False,'normalized2d128',16),(1,2048,True,'normalized2d128',16)]
        for n,q,cap,profile,bq in cases:
            capture[0]=cap
            qview=Tensor(query.storage,'bfloat16',(q,16,128),(2048,128,1))
            facade[(999,)](**dict(launch,num_seqs=n,query_ptr=qview))
            assert kernels[profile].calls[-1][0]==(q//bq+n,2)
        capture[0]=False
        tree=ast.parse((PACKAGE/'biv150_konly_compact_runtime.py').read_bytes())
        invoke=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='invoke')
        assert not any(isinstance(n,ast.Try) for n in ast.walk(invoke))
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,actual_CodeType_fields_verified=True,
        inherited_prepare_native_resource_and_specialization_protocol_unchanged=True,
        compact_proposal_exact_two_header_literal_inverse=True,all_six_GPU_kernels_byte_unchanged=True,
        source_kernel_scale_helper_and_codec_code_identity_bound=True,
        complete_compact_source_manifest_verified=True,nonzero_layer_offset=True,
        physical_addresses_checked=addresses,role_epoch_source_rejections=count,
        actual_facade_parameters_grid_and_parent_selection=True,
        prelaunch_retains_same_compact_call=True,postlaunch_exception_identity_and_one_launch=True,
        graph_capture_retains_private_normalized_route=True,
        prefill_does_not_consult_native_decode_adapter=True,
        actual_compact_TTGIR_or_native_verified=False,actual_JIT_binder_verified=False,
        compiler_or_GPU_executed=False,numerical_verified=False,service_verified=False,
        precision_gate=False,performance_verified=False)


def bind_CPU_mock_q32(kernels):
    """Reused host fixture only: full actual CodeTypes, mocked JIT/compiler pins."""
    module=sys.modules['vllm_fl.patches.biv150_konly_compact_q32']
    code=compile(Path(module.__file__).read_bytes(),module.__file__,'exec',dont_inherit=True,optimize=sys.flags.optimize)
    adapter=load('compact_q32_adapter');proposal=load('compact_hoist_proposal')
    for name in ('_cjj_konly_scale','quantize_bf16_rne','kernel_cjj_konly_q32'):
        function=types.FunctionType(adapter.code_at(code,name),vars(module))
        wrapped=kernels['q32'] if name=='kernel_cjj_konly_q32' else S()
        wrapped.fn=function
        setattr(module,name,wrapped)
    kernels['q32'].src='explicit CPU JIT-source stand-in, not compiled'
    sys.modules['triton'].runtime.driver.active.get_current_device=lambda:0
    return adapter,proposal,module


def installer_on():
    # Existing actual installer fixture injects CPU-only framework modules. The
    # native JIT pin realization is a deliberate stand-in, never a device proof.
    import check_biv150_konly_compact_install_cpu as f
    torch,kv,_,_=f.foundation()
    service,Runner,runner,kernels,backend,attention,parent=f.setup_install(torch,kv)
    adapter,proposal,module=bind_CPU_mock_q32(kernels)
    with tempfile.TemporaryDirectory(prefix='compact-hoist-install-') as temp:
        env={f.ENV:'1','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':'1','TRITON_CACHE_DIR':temp}
        with patch.object(adapter.Adapter,'installed_pins',return_value=proposal.PINS):
            receipt=service.install(Runner,environ=env)
        assert receipt['q32_hoist_enabled'] and not receipt['q32_I8_fragment_hoist_verified']
        state=service._STATE
        assert type(state.q32_hoist) is adapter.Adapter
        assert state.q32_hoist.fn is kernels['q32']
        assert state.q32_hoist.proposal.FORMAT==state.registry.broker.FORMAT
        assert state.q32_hoist.source_binding[0] is module
        import atexit
        atexit.unregister(state.finalize)
        sem=types.ModuleType('_compact_full_semantics_cpu');sem.__file__=str(ROOT/'vllm_fl/compile_semantics.py')
        exec(compile(Path(sem.__file__).read_bytes(),sem.__file__,'exec',dont_inherit=True),vars(sem))
        runner.vllm_config.model_config.hf_config=S(**sem._GEOMETRY)
        flags={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0',f.ENV:'1'}
        off=sem.build_fingerprint(runner.vllm_config,vendor_name='iluvatar',abi={'CPU_mock':True},environ=flags)
        on=sem.build_fingerprint(runner.vllm_config,vendor_name='iluvatar',abi={'CPU_mock':True},environ={**flags,'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':'1'})
        assert off['sha256']!=on['sha256']
        assert on['konly_compact']['q32_hoist_enabled'] and not off['konly_compact']['q32_hoist_enabled']
        assert on['konly_compact']['implementation_sha256']==service.implementation_digest()
    assert all(getattr(sys.modules[n],'_CPU_mock',False) for n in ('torch','triton','vllm'))
    return dict(actual_installer_on_route=True,source_CodeType_guard_executed=True,
                exact_compact_proposal_and_private_q32_object_bound=True,
                compile_semantics_flag_and_implementation_digest_bound=True,
                actual_JIT_pin_realization_mocked=True,GPU_or_compiler_executed=False,
                service_or_numeric_verified=False,precision_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--installer-on',action='store_true');args=p.parse_args()
    import contextlib,io
    with contextlib.redirect_stdout(io.StringIO()):result=installer_on() if args.installer_on else run()
    print(json.dumps(result,sort_keys=True,indent=2))
