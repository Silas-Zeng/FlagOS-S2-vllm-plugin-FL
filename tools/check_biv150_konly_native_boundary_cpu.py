#!/usr/bin/env python3
"""Execute actual main1cb host dispatch/eligibility with CPU metadata recorders only."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

from check_biv150_konly_encoded_cpu import load, Tensor, Storage, Kernel, rejected

ROOT=Path(__file__).resolve().parents[1]
MAIN='1cbac531ca9afaa0e9a68624d5d10cb1d0412055'


class Device:
    index=0
    def __str__(self):return 'cuda:0'
DEVICE=Device()


def tensor(storage,dtype,shape,strides,offset=0):
    result=Tensor(storage,dtype.removeprefix('torch.'),shape,strides,offset)
    result.device=DEVICE;result.is_cuda=True;result.requires_grad=False
    def contiguous():
        stride=1
        for n,s in zip(reversed(result.shape),reversed(result.stride())):
            if n!=1 and s!=stride:return False
            stride*=n
        return True
    result.is_contiguous=contiguous
    return result


def function(text,name,namespace):
    node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
    exec(compile(module,'<actual-main-host-code-cpu-metadata-only>','exec'),namespace)
    return namespace[name]


def run():
    runtime_module,broker,service=load('runtime'),load('views'),load('service')
    # These exact incoming main files remain unchanged by this experiment.
    names=('patches/biv150_gqa_decode.py','patches/biv150_native_gqa.py',
        'patches/biv150_prefill_multi2.py','patches/biv150_q_fragment_service.py')
    for name in names:
        path='vllm_fl/'+name;data=(ROOT/path).read_bytes()
        assert data==subprocess.check_output(['git','-C',str(ROOT),'show',MAIN+':'+path])
        assert hashlib.sha256(data).hexdigest()==service.PINS['vllm_fl.'+name.removesuffix('.py').replace('/','.')]
    for name in ('native_gqa_s2.cpp','native_gqa_s2.cu'):
        path='csrc/iluvatar/'+name
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',MAIN+':'+path])
    torch=NS(cuda=NS(is_current_stream_capturing=lambda:False))
    for dtype in Tensor.widths:setattr(torch,dtype,'torch.'+dtype)
    torch.empty=lambda size,*,dtype,device:tensor(Storage(99999,0,0),dtype,(0,),(1,))
    private,stock=Kernel(),Kernel()
    runtime=runtime_module.Runtime(torch,broker,{'decode':private},NS())
    cache=tensor(Storage(1,0x100000,4*16384),'bfloat16',(4,2,16,2,128),(8192,4096,256,128,1))
    source=Storage(2,0x200000,1024*1024)
    key=tensor(source,'bfloat16',(64,2,128),(2560,128,1),2048)
    value=tensor(source,'bfloat16',(64,2,128),(2560,128,1),2304)
    impl=NS(num_heads=16,num_kv_heads=2,head_size=128,_kv_quant_mode=0,
        _is_per_token_head_quant=False,kv_sharing_target_layer_name=None)
    runtime.registry.armed=True;runtime.registry.claim(impl,cache,key,value)
    cache_k=tensor(cache.storage,'bfloat16',(4,16,2,128),(8192,256,128,1),0)
    cache_v=tensor(cache.storage,'bfloat16',(4,16,2,128),(8192,256,128,1),4096)
    q=tensor(Storage(3,0x300000,64*2048*2),'bfloat16',(64,16,128),(2048,128,1))
    lengths=tensor(Storage(4,0x400000,64*4),'int32',(64,),(1,))
    table=tensor(Storage(5,0x500000,64*8192*4),'int32',(64,8192),(8192,1))
    partial=tensor(Storage(6,0x900000,64*16*32*128*4),'float32',(64,16,2,128),(65536,4096,128,1))
    maximum=tensor(Storage(7,0x3000000,64*16*32*4),'float32',(64,16,2),(512,32,1))
    summation=tensor(Storage(8,0x4000000,64*16*32*4),'float32',(64,16,2),(512,32,1))
    kw=dict(query_ptr=q,key_cache_ptr=cache_k,value_cache_ptr=cache_v,seq_lens_ptr=lengths,
        block_table_ptr=table,segm_output_ptr=partial,segm_max_ptr=maximum,segm_expsum_ptr=summation,
        softmax_scale_log2e=.125,queries_per_kv=8,ROWS=8,HEAD_SIZE=128,PAGE_SIZE=16,SEGMENTS=2,
        num_warps=1,num_stages=1,k_stride_block=8192,k_stride_slot=256,k_stride_head=128,
        v_stride_block=8192,v_stride_slot=256,v_stride_head=128,query_stride_token=2048,
        query_stride_head=128,block_table_stride=8192,segment_stride_token=65536,
        segment_stride_head=4096,segment_stride_segment=128,max_stride_token=512,max_stride_head=32)
    native_text=(ROOT/'vllm_fl/patches/biv150_native_gqa.py').read_text()
    eligible=function(native_text,'_eligible',{})
    actual_native=NS(torch=torch,device=0)
    grid=(64,2,2)
    assert eligible(actual_native,grid,kw) is True
    class Probe:
        calls=0;accept=True
        def try_produce(self,grid,kwargs):
            self.calls+=1
            assert not runtime.registry.owns((kwargs['key_cache_ptr'],kwargs['value_cache_ptr'])), 'encoded reached native'
            return self.accept and eligible(actual_native,grid,kwargs)
    probe=Probe()
    namespace={'_native_s2':probe,'_gqa_decode_attention':stock}
    original=function((ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text(),'_launch_producer',namespace)
    facade=runtime.facade(stock,'gqa')
    wrapped=runtime.launch_producer(original,facade)
    with contextlib.redirect_stdout(io.StringIO()):
        result=wrapped(grid,**kw)
    assert result.hash=='cpu-mock-only' and len(private.calls)==1 and probe.calls==0 and not stock.calls
    assert str(private.calls[0][2]['key_cache_ptr'].dtype)=='torch.int8'
    # Nominal dtype/geometry passes the real native predicate; ownership alone
    # must route first, independent of incidental eligibility differences.
    assert eligible(actual_native,grid,kw) is True
    alias=dict(kw,key_cache_ptr=tensor(cache.storage,'bfloat16',(4,16,2,128),(8192,256,128,1),1))
    rejected(lambda:wrapped(grid,**alias))
    unknown=Storage(20,0x6000000,4*16384)
    other_k=tensor(unknown,'bfloat16',(4,16,2,128),(8192,256,128,1))
    other_v=tensor(unknown,'bfloat16',(4,16,2,128),(8192,256,128,1),4096)
    rejected(lambda:wrapped(grid,**dict(kw,value_cache_ptr=other_v)))
    rejected(lambda:wrapped(grid,**dict(kw,value_cache_ptr=None)))
    stock_kw=dict(kw,key_cache_ptr=other_k,value_cache_ptr=other_v)
    assert eligible(actual_native,grid,stock_kw)
    rejected(lambda:wrapped(grid,**stock_kw))
    assert probe.calls==0 and len(private.calls)==1 and not stock.calls
    # Before an encoded epoch, true main launcher/native/Triton fallback remain
    # intact. Feature-off installation itself is checked by the core CPU tool.
    before=runtime_module.Runtime(torch,broker,{'decode':Kernel()},NS())
    before_facade=before.facade(stock,'gqa')
    namespace['_gqa_decode_attention']=before_facade
    stock_wrapper=before.launch_producer(original,before_facade)
    assert stock_wrapper(grid,**stock_kw) is None and probe.calls==1 and not stock.calls
    probe.accept=False
    assert stock_wrapper(grid,**stock_kw).hash=='cpu-mock-only'
    assert probe.calls==2 and len(stock.calls)==1
    # Parent identity and private producer boundary form a distinct receipt,
    # never a claim that encoded payload executed native BF16 instructions.
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,GPU_executed=False,JIT_executed=False,imported_GPU_framework=False,
        main=MAIN,actual_native_eligibility_accepts_nominal_BF16_encoded_metadata=True,
        owned_encoded_native_calls=0,owned_private_reader_calls=1,
        stock_pre_epoch_actual_parent_native_and_Triton_fallback_preserved=True,
        unknown_after_epoch_and_owned_alias_pair_missing_pointer_refused=True,
        GPU_sources_changed=False,encoded_GQA_uses_stock_native=False,
        latest_main_model_or_performance_verified=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
