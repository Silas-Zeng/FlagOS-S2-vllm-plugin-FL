"""Default-OFF exact natural-prefix NativeK8 candidate; qualification is pending.

No query/length padding, allocator change, tensor readback, writer or stream fork.
The original width16 scratch strides are retained, not reshaped into packed heads.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading

ENV='VLLM_FL_BIV150_KONLY_MIXED_NATIVE'
PATH_ENV=ENV+'_LIBRARY'
SHA_ENV=ENV+'_LIBRARY_SHA256'
NAMESPACE='cjj_native_k8_mixed_probe'
OPERATOR=NAMESPACE+'::produce'
SCHEMA=OPERATOR+'(Tensor query, Tensor key_code, Tensor scale_full, Tensor value_offset, Tensor lengths, Tensor table, Tensor(a!) partial, Tensor(b!) maximum, Tensor(c!) summation, float scale_log2e, int segments) -> ()'
PAIRS={29:4,30:4,31:4,60:2,61:2,63:2}
# These are deliberately unknown. A source-reviewed qualification update must
# bind the actual new SO and both new component reports before service opt-in.
QUALIFICATION={'library_sha256':None,'actual_so_review_sha256':None,
               'consumer_review_sha256':None,'fullstep_review_sha256':None,
               'physical_page_bytes':16384,'rows_segments':[[29,4],[30,4],[31,4],[60,2],[61,2],[63,2]],
               'service_verified':False}
_STATE=None


def need(ok,why):
    if not ok:raise RuntimeError('NativeK8 mixed: '+why)


def library_config(env):
    if env.get(ENV,'0')!='1':return None
    need(env.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE','0')=='1' and
         env.get('VLLM_FL_BIV150_KONLY_NATIVE_DECODE','0')=='1' and
         env.get('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','0')!='1','requires original owned16KiB/native parent')
    pins=QUALIFICATION
    need(all(type(pins.get(k)) is str and len(pins[k])==64 and
             all(c in '0123456789abcdef' for c in pins[k]) for k in
             ('library_sha256','actual_so_review_sha256','consumer_review_sha256','fullstep_review_sha256')),
         'actual SO, consumer and complete-step qualification pending')
    need(pins['physical_page_bytes']==16384 and pins['rows_segments']==[[29,4],[30,4],[31,4],[60,2],[61,2],[63,2]],'exact qualification scope')
    path=Path(env.get(PATH_ENV,''));need(path.is_absolute() and path.is_file(),'explicit existing library')
    need(env.get(SHA_ENV)==pins['library_sha256'],'explicit exact new library pin')
    need(hashlib.sha256(path.read_bytes()).hexdigest()==pins['library_sha256'],'new library bytes')
    return {'path':str(path.resolve()),'sha256':pins['library_sha256']}


def signature(t):
    return (int(t.data_ptr()),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),int(t.storage_offset()))


def scratch_metadata(plan,segments,parent):
    """Independent dataclass and strided prefix views; original metadata intact."""
    md=plan.decode;n=plan.prefix_tokens
    need((n,segments) in ((29,4),(30,4),(31,4),(60,2),(61,2),(63,2)) and md.num_par_softmax_segments==16,'original mixed scratch width16')
    originals=(md.softmax_segm_output,md.softmax_segm_max,md.softmax_segm_expsum)
    for t in originals:parent.bounds(t)
    p,m,l=originals
    need(p.ndim==4 and p.shape[0]>=n and tuple(p.shape[1:])==(16,16,128) and
         tuple(p.stride())==(32768,2048,128,1),'actual width16 partial strides')
    need(all(t.ndim==3 and t.shape[0]>=n and tuple(t.shape[1:])==(16,16) and
             tuple(t.stride())==(256,16,1) for t in (m,l)),'actual width16 scalar strides')
    key=(n,segments,tuple(signature(t) for t in originals))
    cached=getattr(plan,'_native_mixed_scratch',None)
    if cached is not None:
        need(cached[0]==key and cached[1] is md,'changed metadata in reused plan')
        return cached[2]
    narrowed=tuple(t.narrow(2,0,segments) for t in originals)
    for old,new in zip(originals,narrowed):
        parent.bounds(new)
        need(new.data_ptr()==old.data_ptr() and new.storage_offset()==old.storage_offset() and
             tuple(new.stride())==tuple(old.stride()),'narrow must retain original strided addresses')
    result=replace(md,num_par_softmax_segments=segments,softmax_segm_output=narrowed[0],
                   softmax_segm_max=narrowed[1],softmax_segm_expsum=narrowed[2])
    need(all(getattr(result,k) is getattr(md,k) for k in
             ('query_start_loc','seq_lens','block_table','slot_mapping')),'metadata inputs copied or replaced')
    plan._native_mixed_scratch=(key,md,result)
    return result


class Adapter:
    def __init__(self,runtime,config,parent,mixed):
        self.runtime=runtime;self.torch=runtime.torch;self.parent=parent;self.mixed=mixed
        self.library=dict(config);self.active=None;self.calls=0;self.selected=set()
        need(runtime.native_decode is not None,'accepted original Native adapter required')
        runtime.native_decode.verify(runtime)
        torch=self.torch;namespace=getattr(torch.ops,NAMESPACE)
        try:prior=namespace.produce.default
        except (RuntimeError,AttributeError):prior=None
        loaded={str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
        need(prior is None or config['path'] in loaded,'foreign new namespace already loaded')
        torch.ops.load_library(config['path'])
        need(hashlib.sha256(Path(config['path']).read_bytes()).hexdigest()==config['sha256'],'library changed during load')
        self.op=getattr(torch.ops,NAMESPACE).produce.default
        need(str(self.op._schema)==SCHEMA and torch._C._dispatch_has_kernel_for_dispatch_key(OPERATOR,'CUDA'),'new exact operator/dispatch ABI')
        need(config['path'] in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries},'loaded library identity')
        self.binding=(runtime,self.op,config['path'],config['sha256'],runtime.native_decode,parent,mixed)
        self.codes=tuple(getattr(type(self),k).__code__ for k in ('verify','forward','try_produce'))
        self.helpers=(signature,signature.__code__,scratch_metadata,scratch_metadata.__code__,need,need.__code__)
        self.constants=(ENV,PATH_ENV,SHA_ENV,NAMESPACE,OPERATOR,SCHEMA,tuple(PAIRS.items()),json.dumps(QUALIFICATION,sort_keys=True))

    def verify(self,runtime):
        need(not any(k in vars(self) for k in ('verify','forward','try_produce')),'instance override')
        need(self.runtime is runtime and self.binding==(runtime,self.op,self.library['path'],self.library['sha256'],runtime.native_decode,self.parent,self.mixed),'installed identity')
        need(self.codes==tuple(getattr(type(self),k).__code__ for k in ('verify','forward','try_produce')),'installed code')
        need(self.helpers==(signature,signature.__code__,scratch_metadata,scratch_metadata.__code__,need,need.__code__),'helper changed')
        need(self.constants==(ENV,PATH_ENV,SHA_ENV,NAMESPACE,OPERATOR,SCHEMA,tuple(PAIRS.items()),json.dumps(QUALIFICATION,sort_keys=True)),'constant/qualification changed')
        need(self.op is getattr(self.torch.ops,NAMESPACE).produce.default and self.mixed._NATIVE_MIXED is self,'operator/mixed hook changed')
        need(runtime.mixed_native is self and _STATE is self,'runtime/module hook changed')
        runtime.native_decode.verify(runtime)
        return {'enabled':True,'library':dict(self.library),'host_calls_not_graph_replays':self.calls,
                'service_verified':False,'rows_segments':[[29,4],[30,4],[31,4],[60,2],[61,2],[63,2]]}

    def forward(self,original,impl,layer,q,k,v,cache,plan,out,output_scale,output_block_scale):
        self.verify(self.runtime);r=self.runtime.registry;n=plan.prefix_tokens
        def ordinary(md):
            return original(impl,layer,q[:n],k[:n],v[:n],cache,md,out[:n],
                            output_scale=output_scale,output_block_scale=output_block_scale)
        # The outer natural-prefix wrapper already enforces model, mode, dtype,
        # output and noncapture predicates. No new standalone forward entry.
        if n not in PAIRS or not r.armed or not r.owns(cache):return ordinary(plan.decode)
        need(not self.torch.cuda.is_current_stream_capturing(),'mixed path must be Graph NONE')
        need(self.active is None,'nested/reentrant mixed producer')
        identity=r.impl_owner.get(id(impl));entry=r.entries.get(identity)
        need(entry is not None and r.impl_references.get(id(impl)) is impl and
             r.observe(cache)==entry.observation and r.observe(entry.owner)==entry.observation,
             'actual writer-owned cache/generation')
        need(entry.observation.strides==(8192,4096,256,128,1) and 0<entry.generation<=r.generation,'16KiB epoch')
        need(q.ndim==out.ndim==3 and q.shape[0]>=plan.total_tokens and out.shape[0]>=plan.total_tokens and
             tuple(q.shape[1:])==tuple(out.shape[1:])==(16,128) and
             tuple(q.stride()) in ((2048,128,1),(2560,128,1)) and tuple(out.stride())==(2048,128,1),
             'full mixed Q/output geometry')
        readonly=(q,k,v,cache,plan.decode.query_start_loc,plan.decode.seq_lens,plan.decode.block_table,plan.decode.slot_mapping)
        for t in (*readonly,out):self.parent.bounds(t)
        need(all(not self.parent.overlaps(out,t) for t in readonly),'output/read-only backing alias')
        md=scratch_metadata(plan,PAIRS[n],self.parent)
        for t in (md.softmax_segm_output,md.softmax_segm_max,md.softmax_segm_expsum):
            need(not self.parent.overlaps(out,t),'output/scratch backing alias')
        need(tuple(md.seq_lens.shape)==(n,) and md.block_table.shape[0]==n,'real metadata rows, no padding')
        ticket={'entry':entry,'n':n,'s':PAIRS[n],'q':signature(q[:n]),'lens':signature(md.seq_lens),
                'table':signature(md.block_table),'scratch':tuple(signature(t) for t in
                    (md.softmax_segm_output,md.softmax_segm_max,md.softmax_segm_expsum)),
                'thread':threading.get_ident(),'stream':self.torch.cuda.current_stream(q.device),'calls':0}
        self.active=ticket
        try:
            result=ordinary(md)
            need(ticket['calls']==1,'selected mixed forward did not execute exactly one Native producer')
            return result
        finally:
            self.active=None # Never swallow/retry a producer or reducer failure.

    def try_produce(self,runtime,entry,kw,grid):
        if self.active is None:return False
        self.verify(runtime);t=self.active
        need(t['calls']==0 and t['entry'] is entry and t['thread']==threading.get_ident(),'scoped producer entry/thread')
        need(self.torch.cuda.current_stream(kw['query_ptr'].device)==t['stream'] and
             not self.torch.cuda.is_current_stream_capturing(),'scoped current stream')
        q,k,scale,v,lens,table,p,m,l=runtime.native_decode.validate(runtime,entry,kw)
        need(signature(q)==t['q'] and signature(lens)==t['lens'] and signature(table)==t['table'] and
             tuple(signature(x) for x in (p,m,l))==t['scratch'],'scoped exact input/scratch views')
        n,s=t['n'],t['s'];need(tuple(grid)==(n,2,s) and kw.get('SEGMENTS')==s,'producer/reducer segment agreement')
        need(tuple(q.shape)==(n,16,128) and tuple(q.stride()) in ((2048,128,1),(2560,128,1)),'real Q rows/stride')
        need(tuple(lens.stride())==(1,) and table.ndim==2 and table.stride(0)==8192 and table.stride(1)==1 and 0<table.shape[1]<=8192,'metadata layout')
        need(tuple(p.shape[1:])==(16,s,128) and tuple(p.stride())==(32768,2048,128,1) and
             all(tuple(x.shape[1:])==(16,s) and tuple(x.stride())==(256,16,1) for x in (m,l)),
             'actual width16 scratch prefix layout')
        need(all(kw.get(key)==value for key,value in {'HEAD_SIZE':128,'PAGE_SIZE':16,'ROWS':8,
             'queries_per_kv':8,'num_warps':1,'num_stages':1}.items()) and kw.get('softmax_scale_log2e')==self.parent.SCALE,'exact launch math/options')
        self.op(q,k,scale,v,lens,table,p,m,l,kw['softmax_scale_log2e'],s)
        t['calls']+=1;self.calls+=1
        if n not in self.selected:
            self.selected.add(n);print('BIV150_KONLY_MIXED_NATIVE_SELECTED',json.dumps({'rows':n,'segments':s,
                'library_sha256':self.library['sha256'],'host_selection_not_replay_count':True}),flush=True)
        return True


def attach(runtime,env,parent,mixed):
    global _STATE
    config=library_config(env)
    if config is None:return False
    need(_STATE is None and mixed._NATIVE_MIXED is None,'one cold install')
    adapter=Adapter(runtime,config,parent,mixed)
    _STATE=adapter;runtime.mixed_native=adapter;mixed._NATIVE_MIXED=adapter
    adapter.verify(runtime)
    return True
