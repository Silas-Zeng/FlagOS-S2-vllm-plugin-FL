#!/usr/bin/env python3
"""Actual host routing + original launch AST with opaque CPU tensors/dispatcher.

No tensor numbers, real CUDA imports, device execution or new SO qualification.
"""
import ast,contextlib,hashlib,io,json,tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import biv150_native_k8_mixed_factory as factory
from check_biv150_konly_encoded_cpu import Tensor,Storage,Kernel,load,rejected
from check_biv150_konly_native_boundary_cpu import tensor,DEVICE
from check_biv150_native_k8_service_cpu import fixture,SO,Op

ROOT=factory.ROOT

def new_view(t,shape,offset):
    return tensor(t.storage,t.dtype,shape,t.stride(),offset)
def sliced(t,s):
    assert isinstance(s,slice) and s.step in (None,1)
    start,end,_=s.indices(t.shape[0]);return new_view(t,(end-start,*t.shape[1:]),t.offset+start*t.stride(0))
def narrow(t,dim,start,length):
    assert 0<=start<=start+length<=t.shape[dim]
    shape=list(t.shape);shape[dim]=length;return new_view(t,shape,t.offset+start*t.stride(dim))
Tensor.__getitem__=sliced;Tensor.narrow=narrow

@dataclass
class Metadata:
    num_par_softmax_segments:int
    softmax_segm_output:object
    softmax_segm_max:object
    softmax_segm_expsum:object
    query_start_loc:object
    seq_lens:object
    block_table:object
    slot_mapping:object

@dataclass
class Plan:
    prefix_tokens:int
    total_tokens:int
    decode:object
    prefill:object

class NewNamespace:
    def __init__(self,ops,path,op):self.ops,self.path,self.op=ops,path,op
    @property
    def produce(self):
        if self.path not in self.ops.loaded_libraries:raise AttributeError('unregistered')
        return self.op

source=(ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text();tree=ast.parse(source)
pick_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick_segments')
ns={'TARGET_PROGRAMS':256};exec(compile(ast.Module(body=[pick_node],type_ignores=[]),'<actual-pick-segments>','exec'),ns)
pick=ns['_pick_segments']
launch_nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and
              ((isinstance(n.func,ast.Name) and n.func.id=='_launch_producer') or
               (isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Name) and n.func.value.id=='_reduce_partials'))]
assert len(launch_nodes)==2
launch_code=compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(n) for n in launch_nodes],type_ignores=[])),
                    '<actual-original-producer-reducer-launch-AST>','exec')


def make(n,pitch,directory):
    parent=load('native');mixed=load('mixed_native')
    runtime,entry,kw,producer,kernels,ops,capture,service,mod=fixture(parent,n,16,pitch,SO)
    runtime.torch.cuda.current_stream=lambda *a,**k:'CPU-stream-A'
    # Labeled opaque artifacts only. Production source retains all None pins.
    dummy=(directory/('cpu-only-not-a-shared-library-'+str(n)+'-'+str(pitch))).resolve();dummy.write_bytes(b'CPU dispatcher fixture; not ELF or device evidence\n')
    digest=hashlib.sha256(dummy.read_bytes()).hexdigest()
    for key in ('library_sha256','actual_so_review_sha256','consumer_review_sha256','fullstep_review_sha256'):
        mixed.QUALIFICATION[key]=digest
    op=Op(mixed.SCHEMA);setattr(ops,mixed.NAMESPACE,NewNamespace(ops,str(dummy),op))
    env={mixed.ENV:'1',mixed.PATH_ENV:str(dummy),mixed.SHA_ENV:digest,
         'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1','VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'1'}
    module=NS(_NATIVE_MIXED=None)
    assert mixed.attach(runtime,env,parent,module)
    adapter=runtime.mixed_native;impl=next(iter(runtime.registry.impl_references.values()))
    q=tensor(Storage(40,0x90000000,2048*pitch*2+512),'bfloat16',(2048,16,128),(pitch,128,1),64)
    out=tensor(Storage(41,0xa0000000,2048*2048*2+512),'bfloat16',(2048,16,128),(2048,128,1),64)
    k=tensor(Storage(42,0xb0000000,2048*2560*2+512),'bfloat16',(2048,2,128),(2560,128,1),2048)
    v=tensor(k.storage,'bfloat16',(2048,2,128),(2560,128,1),2304)
    starts=tensor(Storage(43,0xc0000000,(n+1)*4),'int32',(n+1,),(1,))
    slots=tensor(Storage(44,0xd0000000,n*8),'int64',(n,),(1,))
    md=Metadata(16,kw['segm_output_ptr'],kw['segm_max_ptr'],kw['segm_expsum_ptr'],starts,kw['seq_lens_ptr'],kw['block_table_ptr'],slots)
    suffix=NS(marker='unchanged prefill object');plan=Plan(n,2048,md,suffix)
    records=[];reducers=Kernel();fault={'producer':False,'reducer':False,'double':False}
    def backend(impl,layer,q,k,v,cache,md,out,**options):
        s=pick(n,2,md.num_par_softmax_segments)
        pieces=[getattr(md,key) for key in ('softmax_segm_output','softmax_segm_max','softmax_segm_expsum')]
        if s<md.num_par_softmax_segments:pieces=[x.narrow(2,0,s) for x in pieces]
        local=dict(num_seqs=n,segments=s,q=q,segm_output=pieces[0],segm_max=pieces[1],segm_expsum=pieces[2],
            out=out,seq_lens=md.seq_lens,block_table=md.block_table,_KV_HEADS=2,_QUERY_HEADS=16,
            _HEAD_SIZE=128,_PAGE_SIZE=16,LOG2E=1.4426950408889634,
            bound={'k':kw['key_cache_ptr'],'v':kw['value_cache_ptr'],'softmax_scale':128**-.5},
            _launch_producer=producer,_reduce_partials=reducers)
        records.append((md,out,local));op.throw=fault['producer']
        exec(launch_code,local)
        if fault['double']:exec(launch_code,local)
        if fault['reducer']:raise RuntimeError('injected reducer failure after Native call')
        return out
    def call():return adapter.forward(backend,impl,None,q,k,v,entry.owner,plan,out,None,None)
    return NS(**locals())


def run():
    transform=factory.verify();pending=load('mixed_native')
    assert pending.library_config({}) is None
    rejected(lambda:pending.library_config({pending.ENV:'1','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1','VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'1'}))
    results=[];negative=[]
    with tempfile.TemporaryDirectory(prefix='cjj-mixed-native-cpu-') as directory:
        directory=Path(directory)
        for n in (31,63):
            for pitch in (2048,2560):
                x=make(n,pitch,directory);original=dict(vars(x.md));suffix=dict(vars(x.suffix))
                with contextlib.redirect_stdout(io.StringIO()):x.call();x.call()
                s={31:4,63:2}[n];assert len(x.op.calls)==2 and x.adapter.active is None
                assert not x.kernels['decode'].calls and len(x.reducers.calls)==2
                assert vars(x.md)==original and vars(x.suffix)==suffix and x.plan.prefill is x.suffix
                assert x.records[0][0] is x.records[1][0] and x.records[0][0] is not x.md
                raw=x.op.calls[0];rgrid,_,rkw=x.reducers.calls[0]
                assert rgrid==(n,16) and rkw['num_segments']==s
                assert raw[6] is rkw['segm_output_ptr'] and raw[7] is rkw['segm_max_ptr'] and raw[8] is rkw['segm_expsum_ptr']
                assert x.mixed.signature(x.records[0][1])==x.mixed.signature(x.out[:n])
                p=raw[6];m=raw[7];addresses=0
                for row in range(n):
                    for head in range(16):
                        for segment in range(s):
                            pa=row*p.stride(0)+head*p.stride(1)+segment*p.stride(2)
                            ra=row*rkw['segment_stride_token']+head*rkw['segment_stride_head']+segment*rkw['segment_stride_segment']
                            ma=row*m.stride(0)+head*m.stride(1)+segment
                            mr=row*rkw['max_stride_token']+head*rkw['max_stride_head']+segment*rkw['max_stride_segment']
                            assert pa==ra and ma==mr
                            assert pa+127<row*32768+(head+1)*2048
                            addresses+=128
                assert x.out[:n].data_ptr()+n*2048*2==x.out[n:].data_ptr()
                results.append({'rows':n,'segments':s,'baseline_segments':pick(n,2,16),'pitch':pitch,'partial_scalar_addresses_verified':addresses,
                                'actual_original_launch_AST':True,'same_scratch_objects_for_Native_and_reducer':True,'metadata_reused_across_layers':True,
                                'suffix_unchanged':True,'only_real_rows_launched':True,'new_SO_loaded':False})
        mutations={
            'partial_wrong_width':lambda x:setattr(x.md.softmax_segm_output,'shape',(x.n,16,8,128)),
            'partial_wrong_stride':lambda x:setattr(x.md.softmax_segm_output,'strides',(36864,2304,128,1)),
            'scalar_wrong_stride':lambda x:setattr(x.md.softmax_segm_max,'strides',(257,16,1)),
            'table_wrong_stride':lambda x:setattr(x.md.block_table,'strides',(8191,1)),
            'bad_epoch':lambda x:setattr(x.entry,'generation',x.runtime.registry.generation+1),
            'query_pitch':lambda x:setattr(x.q,'strides',(2561,128,1)),
            'output_alias_query':lambda x:setattr(x.out,'storage',x.q.storage),
            'capture':lambda x:x.capture.__setitem__(0,True),
            'foreign_op':lambda x:setattr(x.adapter,'op',Op(x.mixed.SCHEMA)),
        }
        for name,mutation in mutations.items():
            x=make(31,2560,directory);mutation(x);rejected(x.call)
            assert len(x.op.calls)==0 and not x.kernels['decode'].calls and x.adapter.active is None
            negative.append(name)
        for name in ('producer','reducer','double'):
            x=make(63,2560,directory);x.fault[name]=True
            with contextlib.redirect_stdout(io.StringIO()):rejected(x.call)
            assert len(x.op.calls)==1 and not x.kernels['decode'].calls and x.adapter.active is None
            negative.append(name+'_no_retry')
        inherited=[]
        for n in (32,64):
            x=make(n,2560,directory)
            with contextlib.redirect_stdout(io.StringIO()):x.call()
            assert not x.op.calls and len(x.ops.op.calls)==1 and not x.kernels['decode'].calls
            assert x.records[0][0] is x.md and x.reducers.calls[0][2]['num_segments']==pick(n,2,16)
            inherited.append({'rows':n,'original_Native_selected':True,'segments':pick(n,2,16)})
        x=make(29,2560,directory);x.call()
        assert not x.op.calls and len(x.kernels['decode'].calls)==1 and x.records[0][0] is x.md
        assert x.reducers.calls[0][2]['num_segments']==8 and not hasattr(x.plan,'_native_mixed_scratch')
    assert all(v is None for k,v in pending.QUALIFICATION.items() if k.endswith('sha256'))
    from check_biv150_native_k8_fingerprint_cpu import run as fingerprint_check,HELPER,LIBRARY
    fingerprint=fingerprint_check(HELPER,LIBRARY)
    import importlib.util
    spec=importlib.util.spec_from_file_location('mixed_fingerprint_actual_helper',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    env={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0',
         'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1','VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'1',
         pending.ENV:'1','VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY':str(LIBRARY),
         'VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256':load('native').LIBRARY_SHA}
    rejected(lambda:helper.expected_fingerprint(ROOT,env,{'torch':'2.10.0','cxx11_abi':False}))
    return {'passed':True,'scope':'CPU opaque dispatch/storage fixtures; production qualification remains pending; fixture receipts are not GPU evidence',
            'source_transform':transform,'cases':results,'negative':negative,'inherited_Native_routes':inherited,
            'actual94db_fingerprint_regression':fingerprint,'mixed_pending_fingerprint_rejected':True,'unknown_shape_original_fallback':True,
            'default_OFF_and_pending_rejection':True,'new_library_or_GPU_executed':False,'numeric_verified':False,'service_verified':False}

if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
