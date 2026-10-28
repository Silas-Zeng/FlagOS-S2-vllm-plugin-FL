"""Default-off verified-8b86 V-overlap NativeK8 adapter, AFTER the private format fence.

No compilation, Tensor-data reads, stock fallback, allocator or kernel changes.
Service qualification is separate from the recorded component evidence.
"""
import hashlib
import json
import math
import os
from pathlib import Path

ENV='VLLM_FL_BIV150_KONLY_NATIVE_DECODE'
PATH_ENV='VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY'
SHA_ENV='VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256'
LIBRARY_SHA='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
FORMAT='cjj-konly-rn32-rne-v2'
OPERATOR='cjj_native_k8_v_overlap_probe::produce'
SCHEMA='cjj_native_k8_v_overlap_probe::produce(Tensor query, Tensor key_code, Tensor scale_full, Tensor value_offset, Tensor lengths, Tensor table, Tensor(a!) partial, Tensor(b!) maximum, Tensor(c!) summation, float scale_log2e, int segments) -> ()'
SCALE=128**-.5*1.4426950408889634


def need(value,message):
    if not value:raise RuntimeError('NativeK8 service: '+message)


def library_config(env=None):
    env=os.environ if env is None else env
    if env.get(ENV,'0')!='1':return None
    need(env.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE','0')=='1','requires owned16KiB encoded service')
    need(env.get('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','0')!='1','compact12544 is unsupported')
    path=Path(env.get(PATH_ENV,''));need(path.is_absolute() and path.is_file(),'explicit existing absolute library')
    need(env.get(SHA_ENV)==LIBRARY_SHA,'explicit fixed SHA pair')
    path=path.resolve();need(hashlib.sha256(path.read_bytes()).hexdigest()==LIBRARY_SHA,'library contents')
    return dict(path=str(path),sha256=LIBRARY_SHA)


def storage(t):
    s=t.untyped_storage();return (int(s._cdata),int(s.data_ptr()),int(s.data_ptr())+int(s.nbytes()))


def overlaps(a,b):
    _,x,y=storage(a);_,u,v=storage(b);return x<v and u<y


def bounds(t):
    widths={'torch.bfloat16':2,'torch.int8':1,'torch.float32':4,'torch.int32':4,'torch.int64':8}
    need(str(t.dtype) in widths,'known operand dtype')
    width=widths[str(t.dtype)];shape=tuple(t.shape);strides=tuple(t.stride())
    need(len(shape)==len(strides) and all(type(n) is int and n>=0 for n in shape) and all(type(v) is int and v>0 for v in strides),'operand geometry')
    _,start,end=storage(t);pointer=int(t.data_ptr());offset=int(t.storage_offset())
    extent=0 if 0 in shape else 1+sum((n-1)*s for n,s in zip(shape,strides))
    need(pointer==start+offset*width and start<=pointer and pointer+extent*width<=end,'operand backing bounds')


def geometry(t):
    return dict(dtype=str(t.dtype),shape=list(t.shape),strides=list(t.stride()),offset=t.storage_offset(),device=str(t.device))


class Adapter:
    def __init__(self,runtime,config):
        self.runtime=runtime;self.torch=runtime.torch;self.library=dict(config)
        torch=self.torch
        need(str(torch.__version__)=='2.10.0' and torch._C._GLIBCXX_USE_CXX11_ABI is False,'actual Torch ABI')
        namespace=torch.ops.cjj_native_k8_v_overlap_probe
        try:prior=namespace.produce.default
        except (AttributeError,RuntimeError):prior=None
        loaded={str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
        need(prior is None or config['path'] in loaded,'operator preloaded from unknown library')
        torch.ops.load_library(config['path'])
        need(hashlib.sha256(Path(config['path']).read_bytes()).hexdigest()==LIBRARY_SHA,'library changed during load')
        self.op=torch.ops.cjj_native_k8_v_overlap_probe.produce.default
        need(str(self.op._schema)==SCHEMA,'exact registered produce ABI')
        need(torch._C._dispatch_has_kernel_for_dispatch_key(OPERATOR,'CUDA'),'actual CUDA registration missing')
        need(config['path'] in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries},'actual loaded-library set')
        self.binding=(runtime,self.op,config['path'],config['sha256'])
        self.code_binding=(type(self).try_produce.__code__,type(self).validate.__code__,type(self).verify.__code__)
        self.helper_binding=tuple((globals()[n],globals()[n].__code__) for n in ('need','storage','overlaps','bounds','geometry'))
        self.constant_binding=(ENV,PATH_ENV,SHA_ENV,LIBRARY_SHA,FORMAT,OPERATOR,SCHEMA,SCALE)
        self.calls=0;self.fallbacks={};self.records={};self.overflow=0
        self.imports=dict(installed=True,enabled=True,format=FORMAT,physical_page_bytes=16384,torch_file=str(torch.__file__),module_file=str(Path(__file__).resolve()),
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),library=dict(config),
            operator=OPERATOR,schema=SCHEMA,loaded_library_confirmed=True,runtime_compile=False)
        print('BIV150_KONLY_NATIVE_DECODE_INSTALL',json.dumps(self.imports,sort_keys=True),flush=True)

    def verify(self,runtime):
        if any(n in vars(self) for n in ('verify','validate','try_produce')):raise RuntimeError('NativeK8 instance method override')
        if self.helper_binding!=tuple((globals()[n],globals()[n].__code__) for n in ('need','storage','overlaps','bounds','geometry')) or self.constant_binding!=(ENV,PATH_ENV,SHA_ENV,LIBRARY_SHA,FORMAT,OPERATOR,SCHEMA,SCALE):
            raise RuntimeError('NativeK8 helper/constants changed')
        need(self.runtime is runtime and self.binding==(runtime,self.op,self.library['path'],self.library['sha256']), 'adapter/library/op binding changed')
        need(self.code_binding==(type(self).try_produce.__code__,type(self).validate.__code__,type(self).verify.__code__),'installed native code changed')
        need(self.library['sha256']==LIBRARY_SHA and self.op is self.torch.ops.cjj_native_k8_v_overlap_probe.produce.default,'dispatcher identity changed')
        return dict(enabled=True,library=dict(self.library),operator=OPERATOR,stock_BF16_native_used_for_encoded=False)

    def validate(self,runtime,entry,kw):
        self.verify(runtime);r=runtime.registry
        need(r.armed and 0<entry.generation<=r.generation and any(e is entry for e in r.entries.values()),'unowned/stale epoch')
        need(r.observe(entry.owner)==entry.observation,'registered allocation changed')
        need(entry.observation.strides==(8192,4096,256,128,1),'only physical16384B page, never compact')
        recipe=r.broker.parameters(entry.observation,'decode79',format_id=FORMAT)
        role=entry.roles['decode79']
        for key,wanted in recipe.items():
            if type(wanted) is r.broker.View:
                need(r.observe(role[key])==wanted,'typed role changed '+key)
            else:need(role[key]==wanted,'role scalar changed '+key)
        for key in ('key_cache_ptr','key_scale_ptr','value_cache_ptr'):
            need(kw[key] is role[key] and r.observe(kw[key])==recipe[key],'selected role mismatch '+key)
        need(kw['query_code_ptr'] is role['key_cache_ptr'] and kw['query_scale_ptr'] is role['key_scale_ptr'],'foreign unused private-Q operands')
        tensors=[kw[k] for k in ('query_ptr','key_cache_ptr','key_scale_ptr','value_cache_ptr','seq_lens_ptr','block_table_ptr','segm_output_ptr','segm_max_ptr','segm_expsum_ptr')]
        q,k,scale,v,lens,table,partial,maximum,summation=tensors
        for t in tensors:bounds(t)
        need(all(str(t.device)=='cuda:0' for t in tensors),'single actual CUDA0 roles')
        need(str(q.dtype)=='torch.bfloat16' and tuple(q.shape[1:])==(16,128),'BF16 query heads/dim')
        need(str(lens.dtype)==str(table.dtype)=='torch.int32','I32 metadata')
        need(all(str(t.dtype)=='torch.float32' for t in (partial,maximum,summation)),'F32 scratch')
        for src in (q,lens,table):need(not overlaps(src,k),'source/live-cache backing alias')
        for i,out in enumerate((partial,maximum,summation)):
            need(all(not overlaps(out,src) for src in (q,k,scale,v,lens,table)),'scratch/source backing alias')
            need(all(not overlaps(out,other) for other in (partial,maximum,summation)[:i]),'scratch backing alias')
        need(tuple(maximum.stride())==tuple(summation.stride()),'scalar common strides')
        observed={'query_stride_token':q.stride(0),'query_stride_head':q.stride(1),'block_table_stride':table.stride(0),
            'k_stride_block':k.stride(0),'k_stride_slot':k.stride(1),'k_stride_head':k.stride(2),
            'v_stride_block':v.stride(0),'v_stride_slot':v.stride(1),'v_stride_head':v.stride(2),
            'segment_stride_token':partial.stride(0),'segment_stride_head':partial.stride(1),'segment_stride_segment':partial.stride(2),
            'max_stride_token':maximum.stride(0),'max_stride_head':maximum.stride(1)}
        need(all(kw.get(n)==value for n,value in observed.items()),'launch scalar/actual Tensor metadata mismatch')
        return tensors

    def try_produce(self,runtime,entry,kw,grid):
        tensors=self.validate(runtime,entry,kw)
        q,k,scale,v,lens,table,partial,maximum,summation=tensors
        n=q.shape[0];s=kw.get('SEGMENTS')
        reason=None
        if (n,s) not in ((64,2),(32,4)):reason='shape_or_segments'
        elif tuple(q.stride()) not in ((2048,128,1),(2560,128,1)):reason='query_stride'
        elif tuple(grid)!=(n,2,s):reason='grid'
        elif any(kw.get(name)!=value for name,value in {'HEAD_SIZE':128,'PAGE_SIZE':16,'ROWS':8,'queries_per_kv':8,'num_warps':1,'num_stages':1}.items()):reason='launch_options'
        elif tuple(lens.shape)!=(n,) or tuple(lens.stride())!=(1,):reason='lengths_layout'
        elif table.ndim!=2 or table.shape[0]!=n or table.stride(1)!=1 or table.stride(0)!=8192 or not 0<table.shape[1]<=8192:reason='table_layout'
        elif partial.ndim!=4 or partial.shape[0]<n or tuple(partial.shape[1:])!=(16,s,128) or tuple(partial.stride())!=(32768,2048,128,1):reason='partial_not_narrowed_from_width16'
        elif any(t.ndim!=3 or t.shape[0]<n or tuple(t.shape[1:])!=(16,s) or tuple(t.stride())!=(256,16,1) for t in (maximum,summation)):reason='scalar_not_narrowed_from_width16'
        elif type(kw.get('softmax_scale_log2e')) not in (float,int) or kw['softmax_scale_log2e']!=SCALE:reason='unverified_attention_scale'
        if reason:
            self.fallbacks[reason]=self.fallbacks.get(reason,0)+1;return False
        # Existing SO, current-stream dispatcher, exactly one call. No exception
        # is caught here: a postlaunch error must never cause a second producer.
        self.op(q,k,scale,v,lens,table,partial,maximum,summation,kw['softmax_scale_log2e'],s)
        self.calls+=1;runtime.counts['selected']['native_decode']=runtime.counts['selected'].get('native_decode',0)+1
        capture=bool(self.torch.cuda.is_current_stream_capturing())
        info=dict(shape=n,segments=s,capture=capture,entry_generation=entry.generation,
            query=geometry(q),table=geometry(table),partial=geometry(partial),maximum=geometry(maximum),summation=geometry(summation),
            cache_pages=k.shape[0],format=FORMAT,physical_page_bytes=16384,library=dict(self.library),operator=OPERATOR,
            component_qualification='large_fullstep' if (n,s,q.stride(0)) in ((64,2,2048),(32,4,2560)) else 'small_consumer_stride_only',
            host_or_capture_event_not_Graph_replay_count=True,native_launch_returned=True,service_numeric_verified=False)
        key=json.dumps(info,sort_keys=True)
        # Epoch/layer identities are separate; bounded records retain complete
        # actual geometry while printing each selected/captured variant once.
        if key in self.records:self.records[key]['host_calls']+=1
        elif len(self.records)<4096:
            self.records[key]=dict(**info,host_calls=1)
            print('BIV150_KONLY_NATIVE_DECODE_CAPTURED' if capture else 'BIV150_KONLY_NATIVE_DECODE_SELECTED',json.dumps(info,sort_keys=True),flush=True)
        else:self.overflow+=1
        return True

    def stats(self):
        return dict(enabled=True,library=self.library,actual_imports=self.imports,selected_host_calls=self.calls,
            private_Triton_fallbacks=dict(self.fallbacks),records=list(self.records.values()),overflow=self.overflow,
            Graph_replays_counted=False,service_numeric_model_or_performance_verified=False)


def attach(runtime,environ):
    """Exact install routine used by the real service and CPU wiring tests."""
    config=library_config(environ)
    if config is None:return False
    need(runtime.native_decode is None and not runtime.registry.armed,'install only before the encoded epoch')
    need(not runtime.torch.cuda.is_current_stream_capturing(),'no native install while capturing')
    adapter=Adapter(runtime,config)
    runtime.native_decode=adapter
    runtime.native_decode_identity=(adapter,type(adapter).verify.__code__,type(adapter).validate.__code__,type(adapter).try_produce.__code__)
    return True
