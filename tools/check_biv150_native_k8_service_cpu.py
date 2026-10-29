#!/usr/bin/env python3
"""Actual native attachment/fence/C3 code with CPU metadata, plus actual8b86 ELF ABI proof."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace as NS
from check_biv150_konly_encoded_cpu import load,Storage,Kernel,rejected
from check_biv150_konly_native_boundary_cpu import tensor,DEVICE,run as old_boundary

ROOT=Path(__file__).resolve().parents[1]
PARENT='2e3abf3f9bcae78e37d8cf3a63d8fcfc6a609db9'
SO=Path('/private/tmp/cjj-native-k8-v-overlap-compile-v1-raw-20261028/out/build/cjj_native_k8_v_overlap_probe.so')

# Exact narrow inverse; all original wiring/fence/C3/ELF cases below remain.
STORAGE_OBSERVATION_PATCHES = (('def overlaps(a,b):\n    _,x,y=storage(a);_,u,v=storage(b);return x<v and u<y',
  'def overlaps(a,b,backing):\n    _,x,y=backing[id(a)];_,u,v=backing[id(b)];return x<v and u<y'),
 ('    _,start,end=storage(t);pointer=int(t.data_ptr());offset=int(t.storage_offset())',
  '    '
  'backing=storage(t);_,start,end=backing;pointer=int(t.data_ptr());offset=int(t.storage_offset())'),
 ("    need(pointer==start+offset*width and start<=pointer and pointer+extent*width<=end,'operand "
  "backing bounds')",
  "    need(pointer==start+offset*width and start<=pointer and pointer+extent*width<=end,'operand "
  "backing bounds')\n"
  '    return backing'),
 ('        for t in tensors:bounds(t)',
  '        # Only this call: tensors keeps every object alive; retain all bounds\n'
  '        # checks in order and reuse their observed whole-storage intervals.\n'
  '        backing={id(t):bounds(t) for t in tensors}'),
 ('overlaps(src,k)', 'overlaps(src,k,backing)'),
 ('overlaps(out,src)', 'overlaps(out,src,backing)'),
 ('overlaps(out,other)', 'overlaps(out,other,backing)'))

def restore_storage_observation(source):
    for old,new in reversed(STORAGE_OBSERVATION_PATCHES):
        assert source.count(new)==1, 'storage observation inverse drift: '+new
        source=source.replace(new,old,1)
    return source

def actual_elf(path,native):
    b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==native.LIBRARY_SHA
    assert b[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',b,18)[0]==62
    schema=native.SCHEMA.split('::',1)[1].encode();assert schema in b
    assert native.OPERATOR.split('::',1)[0].encode() in b
    names=['launch_cjj_native_k8_v_overlap_s2','launch_cjj_native_k8_v_overlap_s4']
    assert all(n.encode() in b for n in names)
    return dict(actual_candidate_library_parsed_CPU=True,bytes=len(b),sha256=native.LIBRARY_SHA,
                ELF_host_machine=62,registered_schema_bytes=schema.decode(),schema_offset=b.index(schema),launch_symbols=names,
                library_loaded_or_executed=False)

class Op:
    def __init__(self,schema):self._schema=schema;self.calls=[];self.default=self;self.throw=False
    def __call__(self,*args):
        self.calls.append(args)
        if self.throw:raise RuntimeError('injected native launch error')

class Namespace:
    def __init__(self,ops):self.ops=ops
    @property
    def produce(self):
        if not self.ops.loaded_libraries:raise AttributeError('unregistered')
        return self.ops.op

class Ops:
    def __init__(self,schema):self.loaded_libraries=set();self.op=Op(schema);self.cjj_native_k8_v_overlap_probe=Namespace(self)
    def load_library(self,path):self.loaded_libraries.add(path)

def fixture(native,batch,segments,pitch,path):
    capture=[False];ops=Ops(native.SCHEMA)
    torch=NS(__version__='2.10.0',__file__='<CPU metadata/dispatcher recorder; not Torch>',ops=ops,
        _C=NS(_GLIBCXX_USE_CXX11_ABI=False,_dispatch_has_kernel_for_dispatch_key=lambda *a:True),
        cuda=NS(is_current_stream_capturing=lambda:capture[0]))
    for k in ('bfloat16','int8','float32','int32','int64'):setattr(torch,k,'torch.'+k)
    torch.empty=lambda size,*,dtype,device:tensor(Storage(9999,0,0),dtype,(0,),(1,))
    mod,broker,service=load('runtime'),load('views'),load('service')
    kernels={k:Kernel() for k in ('decode','q32','multi2','raw3d','normalized2d128','writer')}
    runtime=mod.Runtime(torch,broker,kernels,NS());env={native.ENV:'1','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1',native.PATH_ENV:str(path),native.SHA_ENV:native.LIBRARY_SHA}
    with contextlib.redirect_stdout(io.StringIO()):assert native.attach(runtime,env)
    runtime.registry.armed=True
    cache=tensor(Storage(1,0x100000,4*16384+512),'bfloat16',(4,2,16,2,128),(8192,4096,256,128,1),128)
    source=Storage(2,0x200000,1024*1024)
    kv=tensor(source,'bfloat16',(batch,2,128),(2560,128,1),2048)
    vv=tensor(source,'bfloat16',(batch,2,128),(2560,128,1),2304)
    impl=NS(num_heads=16,num_kv_heads=2,head_size=128,_kv_quant_mode=0,_is_per_token_head_quant=False,kv_sharing_target_layer_name=None)
    entry=runtime.registry.claim(impl,cache,kv,vv)
    q=tensor(Storage(3,0x10000000,batch*pitch*2+512),'bfloat16',(batch,16,128),(pitch,128,1),64)
    lens=tensor(Storage(4,0x20000000,batch*4),'int32',(batch,),(1,));table=tensor(Storage(5,0x30000000,batch*8192*4),'int32',(batch,8192),(8192,1))
    partial=tensor(Storage(6,0x40000000,batch*32768*4),'float32',(batch,16,segments,128),(32768,2048,128,1))
    maximum=tensor(Storage(7,0x50000000,batch*256*4),'float32',(batch,16,segments),(256,16,1));summation=tensor(Storage(8,0x60000000,batch*256*4),'float32',(batch,16,segments),(256,16,1))
    km=tensor(cache.storage,'bfloat16',(4,16,2,128),(8192,256,128,1),128);vm=tensor(cache.storage,'bfloat16',(4,16,2,128),(8192,256,128,1),4224)
    kw=dict(query_ptr=q,key_cache_ptr=km,value_cache_ptr=vm,seq_lens_ptr=lens,block_table_ptr=table,
        segm_output_ptr=partial,segm_max_ptr=maximum,segm_expsum_ptr=summation,softmax_scale_log2e=native.SCALE,
        HEAD_SIZE=128,PAGE_SIZE=16,ROWS=8,queries_per_kv=8,SEGMENTS=segments,num_warps=1,num_stages=1,
        k_stride_block=8192,k_stride_slot=256,k_stride_head=128,v_stride_block=8192,v_stride_slot=256,v_stride_head=128,
        query_stride_token=pitch,query_stride_head=128,block_table_stride=8192,segment_stride_token=32768,
        segment_stride_head=2048,segment_stride_segment=128,max_stride_token=256,max_stride_head=16)
    stock=Kernel();facade=runtime.facade(stock,'gqa')
    def poisoned_stock(*a,**k):raise AssertionError('owned cache reached stock producer')
    producer=runtime.launch_producer(poisoned_stock,facade)
    return runtime,entry,kw,producer,kernels,ops,capture,service,mod

def check(path):
    native=load('native');assert native.library_config({}) is None
    original_library='22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd'
    for name in ('vllm_fl/patches/biv150_konly_encoded_native.py','vllm_fl/compile_semantics.py'):
        original=subprocess.check_output(['git','-C',str(ROOT),'show','41f0f4c45fb87a824bdefaa39f91847fff96543b:'+name])
        restored=(ROOT/name).read_text()
        if name.endswith('biv150_konly_encoded_native.py'):restored=restore_storage_observation(restored)
        restored=restored.replace(native.LIBRARY_SHA,original_library).replace(
            'cjj_native_k8_v_overlap_probe','cjj_native_k8_qk_probe').replace(
            'Default-off verified-8b86 V-overlap NativeK8 adapter','Default-off existing-22e NativeK8 adapter')
        assert restored.encode()==original,'service change beyond identity: '+name
    manifest=json.loads((ROOT/'vllm_fl/patches/biv150_konly_encoded_sources.json').read_text())
    assert manifest['native_decode']['library_sha256']==native.LIBRARY_SHA
    assert manifest['native_decode']['adapter']['sha256']==hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    original=json.loads(subprocess.check_output(['git','-C',str(ROOT),'show','41f0f4c45fb87a824bdefaa39f91847fff96543b:vllm_fl/patches/biv150_konly_encoded_sources.json']))
    manifest['native_decode']['library_sha256']=original['native_decode']['library_sha256']
    manifest['native_decode']['adapter']['sha256']=original['native_decode']['adapter']['sha256']
    assert manifest==original
    for name in ('biv150_native_k8_codec.cuh','biv150_native_k8_v_overlap_codec.cu',
                 'biv150_native_k8_v_overlap_qk.cpp','biv150_native_k8_v_overlap_qk.cu'):
        name='tools/kernels/'+name
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','6a85b304b920fb1efbfb4710e74aefdc58245713:'+name])
    rejected(lambda:native.library_config({native.ENV:'1','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1',
                                          native.PATH_ENV:str(path),native.SHA_ENV:original_library}))
    from unittest.mock import patch
    import os
    worker=ast.parse((ROOT/'vllm_fl/worker/model_runner.py').read_bytes())
    node=next(n for n in worker.body if isinstance(n,ast.If) and native.ENV in ast.unparse(n.test))
    code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-new-worker-preflight>','exec')
    for combo,allowed in [({},True),({native.ENV:'1'},False),({native.ENV:'1','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1'},True),({native.ENV:'1','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1','VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'1'},False)]:
        base={native.ENV:'0','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'0','VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'0'}
        with patch.dict(os.environ,{**base,**combo}):
            if allowed:exec(code,{})
            else:rejected(lambda:exec(code,{}))
    proof=actual_elf(path,native);cases=[]
    for batch,s in ((64,2),(32,4)):
        for pitch in (2048,2560):
            runtime,entry,kw,producer,kernels,ops,capture,service,mod=fixture(native,batch,s,pitch,path)
            grid=(batch,2,s)
            with contextlib.redirect_stdout(io.StringIO()):
                assert producer(grid,**kw) is None
                capture[0]=True;assert producer(grid,**kw) is None;capture[0]=False
            assert len(ops.op.calls)==2 and not kernels['decode'].calls
            assert ops.op.calls[-1][1] is entry.roles['decode79']['key_cache_ptr']
            assert ops.op.calls[-1][2].data_ptr()==ops.op.calls[-1][1].data_ptr()
            assert ops.op.calls[-1][3].data_ptr()==ops.op.calls[-1][1].data_ptr()+8192
            # All mismatched/owned pointers reject BEFORE stock or native.
            bad=tensor(entry.owner.storage,'bfloat16',kw['key_cache_ptr'].shape,kw['key_cache_ptr'].stride(),129)
            rejected(lambda:producer(grid,**dict(kw,key_cache_ptr=bad)))
            rejected(lambda:producer(grid,**dict(kw,segm_output_ptr=kw['query_ptr'])))
            rejected(lambda:producer(grid,**dict(kw,block_table_stride=1)))
            old=entry.generation;entry.generation=runtime.registry.generation+1
            rejected(lambda:producer(grid,**kw));entry.generation=old
            assert len(ops.op.calls)==2
            # Unverified options use the same private TL, never stock.
            producer(grid,**dict(kw,num_stages=2));assert len(kernels['decode'].calls)==1 and len(ops.op.calls)==2
            # Real C3 receipt/installed wrapper identity, not parent .fn claims.
            raw=runtime.facade(Kernel(),'sdk');norm=runtime.facade(Kernel(),'normalized');a={'raw':raw};b={'norm':norm}
            runtime.facade_bindings=[(a,'raw',raw,raw.original),(b,'norm',norm,norm.original)];runtime.attribute_bindings=[]
            runtime.facade_type=mod.KernelFacade;runtime.facade_code=mod.KernelFacade.__getitem__.__code__;runtime.profile_identities=dict(runtime.kernels)
            runtime.installed_digest=service.implementation_digest();runtime.q32_hoist_identity=None
            gqa=NS(_launch_producer=producer,_native_s2=object());runtime.producer_binding=(gqa,producer,producer.__wrapped__,gqa._native_s2,producer.__code__)
            service._STATE=runtime;attention=NS(kernel_unified_attention=raw)
            assert service.compute_order_receipt(attention,norm)['native_decode_registered']['library']['sha256']==native.LIBRARY_SHA
            keep=runtime.native_decode_identity;runtime.native_decode_identity=None
            rejected(lambda:service.compute_order_receipt(attention,norm));runtime.native_decode_identity=keep
            keepop=runtime.native_decode.op;runtime.native_decode.op=object()
            rejected(lambda:service.compute_order_receipt(attention,norm));runtime.native_decode.op=keepop
            ops.op.throw=True
            rejected(lambda:producer(grid,**kw));assert len(ops.op.calls)==3 and len(kernels['decode'].calls)==1
            service._STATE=None
            cases.append(dict(batch=batch,segments=s,query_stride=pitch,actual_native_attach_and_runtime_fence=True,
                actual_C3_receipt_and_identity_mutants=True,wrong_pointer_epoch_alias_scalars_refused=True,
                private_TL_fallback=True,capture_recording=True,postlaunch_error_no_retry=True))
    boundary=old_boundary();assert boundary['passed']
    for name in ('decode','writer','q32','normalized2d128','multi2','raw3d'):
        p='vllm_fl/patches/biv150_konly_encoded_'+name+'.py'
        assert (ROOT/p).read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':'+p])
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,actual_8b86_ELF_ABI=proof,CPU_dispatcher_is_recorder_only=True,actual_attach_wrapper_and_C3_executed=True,
        cases=cases,stock_native_parent_boundary_preserved=True,component_GPU_sources_unchanged=True,service_identity_only_inverse=False,
        service_inverse_restores_storage_observation_then_identity=True,
        new_service_smoke_model_or_performance_verified=False,GPU_or_SSH_executed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,default=SO);a=p.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):result=check(a.library)
    print(json.dumps(result,indent=2))
