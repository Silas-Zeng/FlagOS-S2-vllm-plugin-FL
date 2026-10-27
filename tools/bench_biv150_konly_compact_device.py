#!/usr/bin/env python3
"""Default CPU source contract. Root-only explicit compact writer/GQA numeric frontier; no timing."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace as NS
from biv150_konly_compact_device_contract import ROOT,BASE,MAIN,PARENT,FORMAT,CASES,PAGE,GUARD,sha,load,refs,packed,append,mapping,plan,bootstrap_receipt,owned_load


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def raw(t):return t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()
def bits(t):
    torch=__import__('torch');h=t.detach().cpu().contiguous()
    return h.view({1:torch.int8,2:torch.int16,4:torch.int32,8:torch.int64}[h.element_size()])
def equal(actual,expected,label):
    assert actual==expected,(label,'byte mismatch')
    return dict(bytes=len(actual),sha256=sha(actual),exact=True)

def cache(pages,torch,compact=True):
    size=PAGE if compact else 16384
    backing=torch.full((pages*size+2*GUARD,),90,dtype=torch.uint8,device='cuda')
    rawview=backing[GUARD:-GUARD];wide=rawview.view(torch.bfloat16)
    carrier=torch.as_strided(wide,(pages,2,16,2,128),(size//2,2176 if compact else 4096,256,128,1),wide.storage_offset())
    key,value=carrier.unbind(1)
    code=torch.as_strided(rawview.view(torch.int8),(pages,16,2,128),(size,256,128,1),rawview.storage_offset())
    return dict(backing=backing,raw=rawview,carrier=carrier,key=code if compact else key,
                value=value,scale=rawview.view(torch.float32),code_full=rawview.view(torch.int8),value_full=wide)

def guarded(shape,dtype,torch,fill=math.nan):
    n=math.prod(shape);b=torch.full((n+128,),42,dtype=dtype,device='cuda');v=b[64:-64].reshape(shape);v.fill_(fill)
    return b,v

def outputs(batch,segments,torch):
    back={};c={}
    for k,shape,dtype in [('scratch',(batch,16,16,128),torch.float32),('maxima',(batch,16,16),torch.float32),
                         ('sums',(batch,16,16),torch.float32),('out',(batch,16,128),torch.bfloat16)]:
        b,t=guarded(shape,dtype,torch);back[k]=(b,t);c[k]=t if k=='out' else t.narrow(2,0,segments)
    return c,back

def poison(back):
    for b,t in back.values():t.fill_(math.nan)

def check_outputs(c,back,lengths,segments,torch):
    arrays={}
    for k,(b,t) in back.items():
        h=b.cpu();assert bool((h[:64]==42).all() and (h[-64:]==42).all())
        h=t.cpu()
        if k=='out':
            assert bool(torch.isfinite(h).all())
            for row,n in enumerate(lengths):
                if n==0:assert bool((h[row]==0).all())
        else:
            for row,n in enumerate(lengths):
                steps=(n+segments*16-1)//(segments*16)
                for s in range(16):
                    active=s<segments and s*steps*16<n
                    assert bool((torch.isfinite(h[row,:,s]) if active else torch.isnan(h[row,:,s])).all())
        arrays[k]=bits(c[k])
    return arrays

def descriptor(t):
    s=t.untyped_storage()
    return dict(dtype=str(t.dtype),shape=list(t.shape),strides=list(t.stride()),storage_offset=t.storage_offset(),
                storage_identity=int(s._cdata),storage_base=s.data_ptr(),storage_bytes=s.nbytes(),pointer=t.data_ptr(),device=str(t.device))

def validate_roles(c,view_module):
    def observe(t):
        d=descriptor(t);return view_module.View(str(t.dtype).removeprefix('torch.'),tuple(t.shape),tuple(t.stride()),
            d['storage_identity'],d['storage_base'],d['storage_bytes'],d['storage_offset'],d['pointer'],d['device'])
    layer=observe(c['carrier']);actual={'key_cache_ptr':c['key'],'key_scale_ptr':c['scale'],'value_cache_ptr':c['value']}
    expected=view_module.parameters(layer,'decode79',format_id=FORMAT)
    for k,t in actual.items():assert observe(t)==expected[k],k
    for k,t in [('code_full',c['code_full']),('scale_full',c['scale']),('value_full',c['value_full'])]:
        assert observe(t)==view_module.parameters(layer,'writer83',format_id=FORMAT)[k],k
    assert c['raw'].storage_offset()==GUARD and c['value'].data_ptr()-c['raw'].data_ptr()==4352
    return {k:descriptor(t) for k,t in c.items() if k!='backing'}

def preserve(kernel,out,label,phase):
    folder=out/label;folder.mkdir(exist_ok=True);md=kernel.metadata
    record=dict(hash=str(kernel.hash),phase=phase,n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),
                metadata={k:getattr(md,k,None) for k in ('shared','num_warps','num_stages','warp_size','use_sme')},files={})
    def freeze(x):
        if x is None or type(x) in (bool,int,float,str):return x
        if isinstance(x,dict):return {type(k).__name__+':'+repr(k):freeze(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [freeze(v) for v in x]
        return dict(type=type(x).__name__,unrepresented=True)
    record['typed_ABI']={k:freeze(getattr(kernel.src,k)) for k in ('signature','constants','attrs')}
    record['source_digest']=sha(kernel.src.fn.src.encode());text={}
    (folder/'actual.jit_python_source.py').write_text(kernel.src.fn.src)
    for ext,value in kernel.asm.items():
        if not isinstance(value,(str,bytes)):continue
        data=value.encode() if isinstance(value,str) else value;(folder/('actual.'+ext)).write_bytes(data)
        record['files'][ext]=dict(bytes=len(data),sha256=sha(data))
        if isinstance(value,str):text[ext]=value
    assert {'source','ttir','ttgir','llir','asm','cubin'}<=set(record['files'])
    record['loaded_resources_known']=phase=='post_launch' and all(type(v) is int for v in (record['n_regs'],record['n_spills'],record['metadata']['shared']))
    save(folder/'record.json',record);return record,text

class Proxy:
    def __init__(self,kernel,warm=False,**extra):self.kernel,self.warm,self.extra=kernel,warm,extra
    def __getitem__(self,grid):
        def call(**kw):
            return self.kernel.warmup(grid=grid,**kw,**self.extra) if self.warm else self.kernel[grid](**kw,**self.extra)
        return call

def allocator_probe(torch):
    # Execute the actual frozen byte allocator method with real Torch storage;
    # the one-layer config is an explicit component scope, not a service claim.
    source=(ROOT/'vllm_fl/worker/model_runner.py').read_text();tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunnerFL')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_allocate_kv_cache_tensors')
    scope={'torch':torch};body=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(body),'<actual-compact-byte-allocator>','exec'),scope)
    config=NS(kv_cache_tensors=[NS(block_stride=0,size=2*PAGE,shared_by=['component'])],kv_cache_groups=[NS(layer_names=['component'])])
    values=scope['_allocate_kv_cache_tensors'](NS(device='cuda:0',runner_only_attn_layers=set()),config)
    actual=values['component'];assert actual.untyped_storage().nbytes()==2*PAGE and actual.storage_offset()==0
    # Actual fixed class/reshape; registration and worker hooks are not activated.
    identity=owned_load(ROOT/'vllm_fl/patches/biv150_konly_compact_identity.py','vllm_fl.patches.biv150_konly_compact_identity')
    identity.verify_package();identity.verify_core(('vllm.v1.kv_cache_interface',))
    specmod=owned_load(ROOT/'vllm_fl/patches/biv150_konly_compact_spec.py','vllm_fl.patches.biv150_konly_compact_spec')
    spec=specmod.CompactFullAttentionSpec(block_size=16,num_kv_heads=2,head_size=128,dtype=torch.bfloat16)
    import biv150_konly_compact_kv_bridge as bridge
    carrier=bridge.reshape_compact(actual,spec,(2,2,16,2,128),(0,1,2,3,4),2,None,torch=torch,spec_class=specmod.CompactFullAttentionSpec)
    assert spec.page_size_bytes==PAGE and carrier.stride()==(6272,2176,256,128,1)
    return dict(actual_allocator_method_source_sha256=sha(ast.get_source_segment(source,fn).encode()),raw=descriptor(actual),carrier=descriptor(carrier),
                exact_storage_bytes=2*PAGE,spec_class=type(spec).__module__+'.'+type(spec).__qualname__,registered_or_service_activated=False)

def run(args,proof):
    assert args.commit and args.out and args.out.is_absolute() and not args.out.exists()
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==args.commit
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    assert not any(os.environ.get(k,'0')=='1' for k in ('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'))
    jit=Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py')
    assert sha(jit.read_bytes())=='66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0,'service9031 busy'
    import re
    smi=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True);m=re.search(r'(\d+)MiB\s*/\s*\d+MiB',smi)
    assert m and int(m[1])<4096,'GPU memory busy; Root owns scheduling'
    args.out.mkdir();(args.out/'idle-before.txt').write_text(smi)
    result=dict(source_commit=args.commit,base91=BASE,main=MAIN,completed=False,passed=False,errors=[],compile_only=args.compile_only,
                proof=proof,records=[],capacity_verified=False,model_verified=False,performance_verified=False,precision_gate=False,
                service_activated=False,old_16KiB_hoist_used=False,Torch_fixture_initialization_on_device=True,
                Triton_attention_codec_writer_launches_skipped=bool(args.compile_only))
    def checkpoint():save(args.out/'result.json',result)
    checkpoint()
    try:
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import torch
        import triton
        import numpy as np
        torch.set_num_threads(4);torch.set_flush_denormal(False)
        target=triton.runtime.driver.active.get_current_target();assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        mathsrc=Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/language/math.py')
        assert sha(mathsrc.read_bytes())=='474139317072193ad472f182967e09fa2b46c9b7a4115d4dc3bfba1960f25964'
        reference=refs();base=load(ROOT/'vllm_fl/patches/biv150_gqa_decode.py','cjj_compact_BF16_control')
        helper=load(ROOT/'tools/biv150_benchmark_utils.py','cjj_compact_calls')
        # The real platform bootstrap may already load this source package and
        # initialize CUDA. Authenticate it without disabling/replacing plugins.
        result['bootstrap_before_roles']=bootstrap_receipt();result['platform_cuda_initialized']=bool(torch.cuda.is_initialized());checkpoint()
        owned_load(ROOT/'vllm_fl/patches/biv150_konly_encoded_writer.py','vllm_fl.patches.biv150_konly_encoded_writer')
        writer=owned_load(ROOT/'vllm_fl/patches/biv150_konly_compact_writer.py','vllm_fl.patches.biv150_konly_compact_writer')
        own=load(ROOT/'vllm_fl/patches/biv150_konly_compact_decode.py','cjj_compact_decode')
        views=load(ROOT/'vllm_fl/patches/biv150_konly_compact_views.py','cjj_compact_views')
        probe=load(ROOT/'tools/kernels/biv150_compact_codec_probe.py','cjj_compact_codec_probe')
        result['imports']=dict(torch=torch.__file__,torch_version=str(torch.__version__),triton=triton.__file__,numpy=np.__file__,numpy_version=np.__version__)
        result['stage']='actual_byte_allocator_and_spec';checkpoint()
        result['allocator']=allocator_probe(torch);result['bootstrap_after_allocator']=bootstrap_receipt();checkpoint()
        result['stage']='M1_M8_codec';checkpoint()
        vectors=reference['codec_vectors']();source=torch.tensor(vectors,dtype=torch.bfloat16);qc,qs,_=reference['quantize_cpu'](source,torch,np)
        codec=[]
        for rows in (1,8):
            sb,x=guarded(source.shape,torch.bfloat16,torch);x.copy_(source)
            cb,c=guarded(source.shape,torch.int8,torch,fill=-128);vb,s=guarded((len(source),),torch.float32,torch)
            k=probe.probe.warmup(x,c,s,len(source),rows,grid=((len(source)+rows-1)//rows,),num_warps=1,num_stages=1)
            rec,_=preserve(k,args.out,'codec-M'+str(rows),'pre_launch')
            if not args.compile_only:
                k=probe.probe[((len(source)+rows-1)//rows,)](x,c,s,len(source),rows,num_warps=1,num_stages=1)
                torch.cuda.synchronize();assert torch.equal(c.cpu(),qc) and torch.equal(bits(s),bits(qs))
                assert raw(x)==raw(source)
                for b in (sb,cb,vb):assert bool((b.cpu()[:64]==42).all() and (b.cpu()[-64:]==42).all())
                rec,_=preserve(k,args.out,'codec-M'+str(rows),'post_launch')
            codec.append(dict(rows=rows,vectors=len(vectors),record=rec,exact=not args.compile_only))
        result['codec']=codec;checkpoint()
        # Edge writer: finite valid rows + NaN invalid rows, exact i64 slot masks.
        result['stage']='writer_edges';checkpoint()
        edges=[0,15,16,47,-1,48,2**63-1,-2**63];edge=cache(3,torch);validate_roles(edge,views)
        nk=torch.tensor([[vectors[i],vectors[(i+1)%len(vectors)]] for i in range(8)],dtype=torch.bfloat16);nv=nk.clone();nk[4:]=math.nan;nv[4:]=math.nan
        ak,aall=guarded((8,4,128),torch.bfloat16,torch);bk,ball=guarded((8,4,128),torch.bfloat16,torch)
        a,b=aall[:,1:3],ball[:,1:3];a.copy_(nk);b.copy_(nv);source_guards=(raw(ak),raw(bk));slots=torch.tensor(edges,dtype=torch.int64,device='cuda');edge['raw'].fill_(0xa5)
        writer_kw=dict(key_source=a,value_source=b,slots=slots,code_full=edge['code_full'],scale_full=edge['scale'],value_full=edge['value_full'],
                       key_row_stride=a.stride(0),value_row_stride=b.stride(0),key_rows=8,value_rows=8,slot_count=10,cache_pages=3,num_warps=1,num_stages=1)
        k=writer.encode_append.warmup(grid=(10,2),**writer_kw);rec,_=preserve(k,args.out,'edge-writer','pre_launch')
        if not args.compile_only:
            before=raw(edge['raw']);k=writer.encode_append[(10,2)](**writer_kw);torch.cuda.synchronize()
            expected=append(before,edges,nk.tolist(),nv.tolist(),reference['quantize_v2'])
            result['edge_writer']=equal(raw(edge['raw']),expected,'edge writer all bytes')
            for t,h in ((a,nk),(b,nv)):assert raw(t)==raw(h)
            assert slots.cpu().tolist()==edges and (raw(ak),raw(bk))==source_guards
            assert bool((edge['backing'].cpu()[:GUARD]==90).all() and (edge['backing'].cpu()[-GUARD:]==90).all())
            rec,_=preserve(k,args.out,'edge-writer','post_launch')
        result['edge_writer_record']=rec;checkpoint()
        for name in args.case or CASES:
            result['stage']=name+'/compile';checkpoint()
            batch,length,segments=CASES[name];assert base._pick_segments(batch,2,16)==segments
            table,pages=mapping(batch,length);record=dict(case=name,batch=batch,length=length,segments=segments,grid=[batch,2,segments],pages=pages,
                physical_page_bytes=PAGE,input_seed=20261023+batch+length,input_generator='Torch CPU BF16',numeric=[],graph=[],side=False,resources={},original_bf16_proxy_passed=False)
            result['records'].append(record);checkpoint()
            encoded=cache(pages,torch);stock=cache(pages,torch,False);record['roles']=validate_roles(encoded,views)
            gen=torch.Generator(device='cpu').manual_seed(20261023+batch+length)
            if not args.compile_only:
                cold=torch.randn((pages,2,16,2,128),generator=gen,dtype=torch.bfloat16);coldk,coldv=cold.unbind(1)
                record['cold_BF16_source_sha256']=sha(raw(cold))
                code,scale,codec_stats=reference['quantize_cpu'](coldk,torch,np);record['CPU_codec']=codec_stats
                expected_cold=packed(code.numpy(),scale.numpy(),coldv.contiguous().view(torch.int16).numpy().view(np.uint16),np)
            q0=torch.randn((batch,16,128),generator=gen,dtype=torch.bfloat16);q=q0.cuda();table0=torch.full((batch,8192),-1,dtype=torch.int32);table0[:,:len(table[0])]=torch.tensor(table)
            tab=table0.cuda();lens=torch.full((batch,),length,dtype=torch.int32,device='cuda');slot=torch.empty(batch,dtype=torch.int64,device='cuda')
            newk=torch.empty((batch,2,128),dtype=torch.bfloat16,device='cuda');newv=torch.empty_like(newk)
            qcode=torch.full_like(q,-128,dtype=torch.int8);qscale=torch.full((batch,16),math.nan,dtype=torch.float32,device='cuda')
            cases={};backs={}
            for label,cache_role in [('baseline',stock),('compact',encoded)]:
                io,back=outputs(batch,segments,torch);cases[label]=dict(q=q,k=cache_role['key'],v=cache_role['value'],table=tab,lengths=lens,**io);backs[label]=back
            extra=dict(query_code_ptr=qcode,query_scale_ptr=qscale,key_scale_ptr=encoded['scale'])
            warm=NS(LOG2E=base.LOG2E,_gqa_decode_attention=Proxy(own._gqa_decode_konly_i8,True,**extra))
            wk=helper.main_call(warm,cases['compact'],segments);rec,strings=preserve(wk,args.out,name+'-compact','pre_launch')
            rec['actual_ir']=reference['audit'](strings,(ROOT/'vllm_fl/patches/biv150_konly_compact_decode.py').read_text());assert rec['actual_ir']['passed']
            record['resources']['compact']=rec;record['actual_reader_ir']=rec['actual_ir']
            for label in ('baseline','compact'):
                bk=helper.main_call(NS(LOG2E=base.LOG2E,_gqa_decode_attention=Proxy(base._gqa_decode_attention,True)),cases['baseline'],segments) if label=='baseline' else wk
                rk=helper.reduce_call(NS(_reduce_partials=Proxy(base._reduce_partials,True)),cases[label],segments,4)
                record['resources'][label+'-compile']=preserve(bk,args.out,name+'-'+label+'-compile','pre_launch')[0]
                record['resources'][label+'-reducer-compile']=preserve(rk,args.out,name+'-'+label+'-reducer-compile','pre_launch')[0]
            checkpoint()
            if args.compile_only:continue
            initialk=coldk.reshape(-1,2,128).cuda();initialv=coldv.reshape(-1,2,128).cuda()
            assert initialk.untyped_storage()._cdata!=encoded['raw'].untyped_storage()._cdata
            assert initialv.untyped_storage()._cdata!=encoded['raw'].untyped_storage()._cdata
            record['initial_source_destination_distinct']=True
            initialslots=torch.arange(pages*16,dtype=torch.int64,device='cuda')
            def encode():
                encoded['raw'].fill_(0xa5)
                return writer.encode_append[(pages*16,2)](initialk,initialv,initialslots,encoded['code_full'],encoded['scale'],encoded['value_full'],
                    initialk.stride(0),initialv.stride(0),pages*16,pages*16,pages*16,pages,num_warps=1,num_stages=1)
            initial_kernel=encode();torch.cuda.synchronize();record['initial_encoder']=equal(raw(encoded['raw']),expected_cold,'full initial encoding')
            record['resources']['initial_encoder']=preserve(initial_kernel,args.out,name+'-initial-encoder','post_launch')[0]
            controls=[];eagers=[];compiled={}
            for iteration in range(3):
                tq=q0.clone();tt=table0.clone();ll=[length]*batch
                if iteration:
                    tq=tq.roll(iteration,1);tq=-tq if iteration==1 else torch.zeros_like(tq);tt=tt.roll(iteration,0)
                    for i,n in enumerate((0,1,15,16,17,63,100,length-1)):ll[i]=n
                sk=[int(tt[t,(n-1)//16])*16+(n-1)%16 if n else (-1 if iteration==1 else 2**63-1) for t,n in enumerate(ll)]
                kk=torch.randn((batch,2,128),generator=gen,dtype=torch.bfloat16);vv=torch.randn((batch,2,128),generator=gen,dtype=torch.bfloat16)
                controls.append(dict(q=tq,table=tt,lengths=ll,slots=sk,new_k=kk,new_v=vv))
            def prepare(control):
                encode();q.copy_(control['q']);tab.copy_(control['table']);lens.copy_(torch.tensor(control['lengths'],dtype=torch.int32))
                slot.copy_(torch.tensor(control['slots'],dtype=torch.int64));newk.copy_(control['new_k']);newv.copy_(control['new_v'])
                post=cold.clone()
                for t,s in enumerate(control['slots']):
                    if 0<=s<pages*16:
                        p,d=divmod(s,16);post[p,0,d]=control['new_k'][t];post[p,1,d]=control['new_v'][t]
                stock['carrier'].copy_(post)
                control['_baseline_cache_sha256']=sha(raw(post))
            def step(label):
                if label=='compact':
                    compiled['writer']=writer.encode_append[(batch,2)](newk,newv,slot,encoded['code_full'],encoded['scale'],encoded['value_full'],
                        newk.stride(0),newv.stride(0),batch,batch,batch,pages,num_warps=1,num_stages=1)
                proxy=NS(LOG2E=base.LOG2E,_gqa_decode_attention=Proxy(own._gqa_decode_konly_i8,False,**extra)) if label=='compact' else base
                compiled[label]=helper.main_call(proxy,cases[label],segments);compiled[label+'-reducer']=helper.reduce_call(base,cases[label],segments,4)
            def verify(control):
                torch.cuda.synchronize()
                expected=append(expected_cold,control['slots'],control['new_k'].tolist(),control['new_v'].tolist(),reference['quantize_v2'])
                equal(raw(encoded['raw']),expected,'all cache bytes incl reserved/unmapped')
                assert sha(raw(stock['raw']))==control['_baseline_cache_sha256'],'baseline cache modified by reader'
                for c in (encoded,stock):
                    full=c['backing'].cpu();assert bool((full[:GUARD]==90).all() and (full[-GUARD:]==90).all())
                for d,h in ((q,control['q']),(tab,control['table']),(newk,control['new_k']),(newv,control['new_v'])):assert raw(d)==raw(h)
                assert lens.cpu().tolist()==control['lengths'] and slot.cpu().tolist()==control['slots']
                assert bool((qcode.cpu()==-128).all() and torch.isnan(qscale.cpu()).all())
                return {label:check_outputs(cases[label],backs[label],control['lengths'],segments,torch) for label in cases}
            for i,control in enumerate(controls):
                result['stage']=name+'/numeric-'+str(i);checkpoint()
                prepare(control)
                for label in cases:poison(backs[label]);step(label)
                snapshot=verify(control);eagers.append(snapshot)
                selected=list(range(batch))
                answers,_=reference['attention_references'](coldk,coldv,code,scale,control,selected,torch,np)
                checks=[dict(row=a['row'],length=a['length'],integer_bound=a['integer_bound'],**reference['reference_gates'](
                    cases['baseline']['out'][a['row']].cpu().float().flatten().tolist(),cases['compact']['out'][a['row']].cpu().float().flatten().tolist(),
                    a['original'].flatten().tolist(),a['quantized'].flatten().tolist())) for a in answers]
                witness=args.out/(name+'-numeric-'+str(i)+'.npz')
                np.savez_compressed(witness,**{label+'_'+k:v.numpy() for label,arrays in snapshot.items() for k,v in arrays.items()},
                    q_bits=bits(control['q']).numpy(),table=control['table'].numpy(),lengths=np.array(control['lengths']),slots=np.array(control['slots'],dtype=np.int64),
                    new_k_bits=bits(control['new_k']).numpy(),new_v_bits=bits(control['new_v']).numpy(),selected_rows=np.array(selected),
                    original_reference=np.stack([a['original'].numpy() for a in answers]),quantized_reference=np.stack([a['quantized'].numpy() for a in answers]))
                record['numeric'].append(dict(iteration=i,checks=checks,witness=dict(path=witness.name,bytes=witness.stat().st_size,sha256=sha(witness.read_bytes()))));checkpoint()
                assert all(v['baseline_hard_passed'] and v['quantized_operand']['finite'] and v['quantized_operand']['pointwise_passed'] for v in checks)
            for label,k in compiled.items():
                loaded,strings=preserve(k,args.out,name+'-'+label+'-loaded','post_launch');record['resources'][label]=loaded
                if label=='compact':
                    record['actual_reader_ir_post_launch']=reference['audit'](strings,(ROOT/'vllm_fl/patches/biv150_konly_compact_decode.py').read_text())
                    assert record['actual_reader_ir_post_launch']['passed']
            result['stage']=name+'/changed_graph';checkpoint()
            graphs={};prepare(controls[0])
            for label in cases:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):step(label)
                graphs[label]=graph
            for i in (1,2,0):
                prepare(controls[i])
                for label in cases:poison(backs[label]);graphs[label].replay()
                actual=verify(controls[i]);assert all(torch.equal(v,eagers[i][label][k]) for label,arrays in actual.items() for k,v in arrays.items())
                record['graph'].append(dict(iteration=i,changed_Q_K_V_slots_table_lengths=True,poisoned_scratch_and_cache=True,all_four_fields_bitwise=True));checkpoint()
            result['stage']=name+'/side_stream';checkpoint()
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                prepare(controls[1])
                for label in cases:poison(backs[label]);step(label)
                side.synchronize();actual=verify(controls[1]);assert all(torch.equal(v,eagers[1][label][k]) for label,arrays in actual.items() for k,v in arrays.items())
            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();record['side']=True
            assert raw(initialk)==raw(coldk.reshape(-1,2,128)) and raw(initialv)==raw(coldv.reshape(-1,2,128))
            record['original_bf16_proxy_passed']=all(v['candidate_numeric_passed'] for n in record['numeric'] for v in n['checks'])
            record['hard_checks']=dict(actual_ir=record['actual_reader_ir_post_launch']['passed'],codec=True,cache_bytes=True,
                full_output_and_scratch_guards=True,sources_unchanged=True,baseline_FP64=True,quantized_operand_FP64=True,
                graph_eager=True,side_eager=True)
            record['implementation_hard_gates']=all(record['hard_checks'].values());checkpoint()
        result['stage']='completed';result['completed']=True;result['compile_passed']=all(r['actual_reader_ir']['passed'] for r in result['records'])
        result['implementation_numeric_verified']=not args.compile_only and all(r.get('implementation_hard_gates') for r in result['records'])
        result['passed']=result['implementation_numeric_verified'] and all(r['original_bf16_proxy_passed'] for r in result['records'])
        result['status']='compile_only' if args.compile_only else 'implementation_numeric_only_original_quality_retained'
        checkpoint();return result
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)));checkpoint();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-gpu',action='store_true');parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--case',action='append',choices=CASES);parser.add_argument('--commit');parser.add_argument('--out',type=Path)
    args=parser.parse_args();assert len(set(args.case or CASES))==len(args.case or CASES),'duplicate case';proof=plan()
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules),'default path imported a GPU/numeric framework'
    if args.run_gpu:run(args,proof)
    else:
        assert not args.compile_only and args.commit is None and args.out is None
        print(json.dumps(proof,indent=2))
