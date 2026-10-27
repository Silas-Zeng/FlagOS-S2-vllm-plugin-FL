#!/usr/bin/env python3
"""Default stdlib plan; explicit Root-only same-b3 native/Triton BF16 pair."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace as NS
from biv150_native_tl_pair_contract import (ROOT,MAIN,LIBRARY_SHA,ARENA_PAGES,CASES,ORDER,
                                           plan,pins,table,append_slots,metrics)


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def digest(value):return hashlib.sha256(value).hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result

def bits(t):
    host=t.detach().cpu().contiguous()
    return host.view({1:__import__('torch').int8,2:__import__('torch').int16,4:__import__('torch').int32}[host.element_size()])

def raw(t):return bits(t).numpy().tobytes()


def preserve(kernel,folder):
    folder.mkdir();record=dict(hash=str(kernel.hash),n_regs=getattr(kernel,'n_regs',None),
        n_spills=getattr(kernel,'n_spills',None),metadata=str(kernel.metadata),files={})
    for ext,data in kernel.asm.items():
        if not isinstance(data,(str,bytes)):continue
        b=data.encode() if isinstance(data,str) else data
        (folder/('actual.'+ext)).write_bytes(b);record['files'][ext]=dict(bytes=len(b),sha256=digest(b))
    assert {'ttir','ttgir','llir','asm','cubin'}<=set(record['files'])
    def freeze(x):
        if x is None or type(x) in (bool,int,float,str):return x
        if isinstance(x,dict):return {type(k).__name__+':'+repr(k):freeze(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [freeze(v) for v in x]
        return dict(type=type(x).__name__,unrepresented=True)
    record['typed_ABI']={k:freeze(getattr(kernel.src,k)) for k in ('signature','constants','attrs')}
    record['source_digest']=digest(kernel.src.fn.src.encode())
    save(folder/'record.json',record);return record


def run(args):
    assert args.commit and args.out and args.library and not args.out.exists()
    assert args.out.is_absolute() and args.library.is_absolute()
    assert len(set(args.case or CASES))==len(args.case or CASES),'duplicate case'
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==args.commit
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(ROOT),'diff',MAIN,'--','vllm_fl','csrc'])
    pins();assert digest(args.library.read_bytes())==LIBRARY_SHA
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0,'service9031 busy'
    args.out.mkdir();result=dict(commit=args.commit,main=MAIN,completed=False,passed=False,errors=[],records=[],
        component_only=True,K8_in_pair=False,model_extrapolation=False,library=dict(path=str(args.library),sha256=LIBRARY_SHA))
    def checkpoint():save(args.out/'result.json',result)
    checkpoint()
    try:
        smi=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        (args.out/'idle-before.txt').write_text(smi)
        # Namespace process tables may be empty even under load. Memory is an
        # independent conservative refusal gate, never proof of exclusive GPU.
        import re
        usage=re.search(r'(\d+)MiB\s*/\s*\d+MiB',smi)
        assert usage and int(usage[1])<4096,'GPU memory busy; Root must own scheduling'
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import torch
        import numpy as np
        from vllm.triton_utils import triton
        target=triton.runtime.driver.active.get_current_target()
        assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        torch.set_num_threads(4)
        base=module(ROOT/'vllm_fl/patches/biv150_gqa_decode.py','cjj_pair_b3_gqa')
        helper=module(ROOT/'tools/biv150_benchmark_utils.py','cjj_pair_b3_calls')
        native=module(ROOT/'vllm_fl/patches/biv150_native_gqa.py','cjj_pair_b3_native')
        support=module(ROOT/'tools/kernels/biv150_pair_append_support.py','cjj_pair_append')
        torch.ops.load_library(str(args.library));native_runtime=native._Runtime(torch,args.library)
        result['imports']=dict(torch=torch.__file__,triton=triton.__file__,base=base.__file__,native=native.__file__)
        class CaptureKW:
            def __getitem__(self,grid):return lambda **kw:kw
        recorder=NS(_gqa_decode_attention=CaptureKW(),LOG2E=base.LOG2E)
        def event_ms(fn,repeats):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):fn()
            end.record();end.synchronize();return start.elapsed_time(end)/repeats
        for name in args.case or CASES:
            started=time.monotonic();batch,length,segments=CASES[name]
            assert base._pick_segments(batch,2,16)==segments
            record=dict(case=name,batch=batch,length=length,segments=segments,arena_pages=ARENA_PAGES,
                active_pages=batch*((length+15)//16),grid=[batch,2,segments],numeric=[],timing=None,
                input_seed=20261023+batch+length,input_generator='Torch CPU BF16, version recorded',
                torch_version=str(torch.__version__),native_loaded_resources_known=False)
            result['records'].append(record);checkpoint()
            generator=torch.Generator(device='cpu').manual_seed(20261023+batch+length)
            q0=torch.randn((batch,16,128),generator=generator,dtype=torch.bfloat16)
            cold=torch.randn((ARENA_PAGES,2,16,2,128),generator=generator,dtype=torch.bfloat16)
            mapping=table(batch,length);tab_cpu=torch.full((batch,8192),-1,dtype=torch.int32)
            tab_cpu[:,:len(mapping[0])]=torch.tensor(mapping,dtype=torch.int32)
            # One fixed arena. Source and resident destination are separated.
            backing=torch.full((ARENA_PAGES*16384+512,),90,device='cuda',dtype=torch.uint8)
            arena=backing[256:-256].view(torch.bfloat16).reshape(ARENA_PAGES,2,16,2,128)
            k,v=arena.unbind(1);initial=cold.to('cuda');torch.cuda.synchronize()
            record['initial_pack_copy_ms']=event_ms(lambda:arena.copy_(initial),1)
            del initial
            record['initial_source_sha256']=digest(raw(cold));record['table_sha256']=digest(raw(tab_cpu))
            q=q0.to('cuda');tab=tab_cpu.to('cuda');lens=torch.empty(batch,device='cuda',dtype=torch.int32)
            slots=torch.empty(batch,device='cuda',dtype=torch.int64)
            nk=torch.empty((batch,2,128),device='cuda',dtype=torch.bfloat16);nv=torch.empty_like(nk)
            dummy=torch.empty(1,device='cuda',dtype=torch.float32)
            cases={};backs={}
            for arm in ('native','triton'):
                c=dict(q=q,k=k,v=v,table=tab,lengths=lens);b={}
                for field,shape,dtype in [('scratch',(batch,16,16,128),torch.float32),('maxima',(batch,16,16),torch.float32),
                                         ('sums',(batch,16,16),torch.float32),('out',(batch,16,128),torch.bfloat16)]:
                    size=math.prod(shape);full=torch.full((size+128,),42.5,device='cuda',dtype=dtype)
                    wide=full[64:-64].reshape(shape);c[field]=wide if field=='out' else wide.narrow(2,0,segments)
                    b[field]=(full,wide)
                cases[arm]=c;backs[arm]=b
            def append():return support.append_kv[(batch,2)](nk,nv,slots,k,dummy,v,ARENA_PAGES*16,False,num_warps=1,num_stages=1)
            def geometry(t):return dict(dtype=str(t.dtype),shape=list(t.shape),stride=list(t.stride()),
                storage_offset=t.storage_offset(),pointer_alignment64=t.data_ptr()%64,device=str(t.device))
            record['tensor_descriptors']={arm:{k:geometry(v) for k,v in c.items()} for arm,c in cases.items()}
            record['append_source_descriptors']={name:geometry(t) for name,t in dict(new_k=nk,new_v=nv,slots=slots).items()}
            record['native_operator']='corex_native_gqa_s'+str(segments)+'_probe::produce'
            handles={}
            def produce(arm):
                c=cases[arm]
                if arm=='native':
                    kw=helper.main_call(recorder,c,segments)
                    assert native_runtime._eligible((batch,2,segments),kw),'actual native ABI/shape rejected'
                    assert native_runtime.try_produce((batch,2,segments),kw),'actual native not selected'
                else:handles['producer']=helper.main_call(base,c,segments)
            def consume(arm):handles['reducer']=helper.reduce_call(base,cases[arm],segments,4)
            def step(arm):append();produce(arm);consume(arm)
            def poison(arm):
                for _,wide in backs[arm].values():wide.fill_(math.nan)
            def capture_output(arm,lengths):
                payload={}
                for field,(full,wide) in backs[arm].items():
                    host=full.detach().cpu();middle=wide.detach().cpu()
                    assert torch.all(host[:64]==42.5) and torch.all(host[-64:]==42.5),'output guard'
                    if field=='out':
                        assert torch.isfinite(middle).all()
                        for row,n in enumerate(lengths):
                            if n==0:assert torch.count_nonzero(middle[row])==0,'empty output not zero'
                    else:
                        for row,n in enumerate(lengths):
                            pages_per_segment=(n+segments*16-1)//(segments*16)
                            for seg in range(16):
                                active=seg<segments and seg*pages_per_segment*16<n
                                values=middle[row,:,seg]
                                assert torch.isfinite(values).all() if active else torch.isnan(values).all()
                                if active and field=='sums':assert (values>0).all()
                    payload[field]=bits(cases[arm][field]).numpy().copy()
                guard=backing.detach().cpu();assert torch.all(guard[:256]==90) and torch.all(guard[-256:]==90)
                return payload
            def fingerprints():
                return {label:digest(raw(t)) for label,t in dict(q=q,table=tab,lengths=lens,slots=slots,new_k=nk,new_v=nv).items()}
            def reference(lengths,keys,values):
                rows=[]
                for row,n in enumerate(lengths):
                    if n==0:rows.append(torch.zeros((16,128),dtype=torch.float64));continue
                    ids=tab_cpu[row,:(n+15)//16].long()
                    kk=cold[ids,0].reshape(-1,2,128)[:n].clone();vv=cold[ids,1].reshape(-1,2,128)[:n].clone()
                    kk[-1]=keys[row];vv[-1]=values[row]
                    qq=q_cpu[row].double().reshape(2,8,128)
                    scores=torch.einsum('hgd,thd->hgt',qq,kk.double())*128**-.5
                    rows.append(torch.einsum('hgt,thd->hgd',scores.softmax(-1),vv.double()).reshape(16,128))
                return torch.stack(rows)
            control0=None
            for control in (0,1):
                lengths=[length]*batch
                if control:
                    for i,n in enumerate((0,1,15,16,17,31,32,33,63,64,65,length-1)):lengths[i]=n
                q_cpu=q0.clone() if not control else -q0.roll(1,1)
                new_k=torch.randn((batch,2,128),generator=generator,dtype=torch.bfloat16)
                new_v=torch.randn((batch,2,128),generator=generator,dtype=torch.bfloat16)
                slot_cpu=torch.tensor(append_slots(mapping,lengths),dtype=torch.int64)
                q.copy_(q_cpu);lens.copy_(torch.tensor(lengths,dtype=torch.int32));slots.copy_(slot_cpu);nk.copy_(new_k);nv.copy_(new_v)
                arena.copy_(cold);expected=cold.clone()
                for row,slot in enumerate(slot_cpu.tolist()):
                    if slot>=0:
                        page,column=divmod(slot,16);expected[page,0,column]=new_k[row];expected[page,1,column]=new_v[row]
                expected_sha=digest(raw(expected));del expected
                handles['append']=append();torch.cuda.synchronize()
                assert digest(raw(arena))==expected_sha,'complete BF16 append mismatch'
                readonly=fingerprints();ref=reference(lengths,new_k,new_v);outputs={}
                for arm in ('native','triton'):
                    poison(arm);step(arm);torch.cuda.synchronize();first=capture_output(arm,lengths)
                    assert fingerprints()==readonly and digest(raw(arena))==expected_sha,'source/cache mutation'
                    poison(arm);produce(arm);consume(arm);torch.cuda.synchronize();second=capture_output(arm,lengths)
                    assert all(np.array_equal(first[f],second[f]) for f in first),'same-arm repeat bits differ'
                    assert fingerprints()==readonly and digest(raw(arena))==expected_sha,'repeat mutated source/cache'
                    outputs[arm]=first;got=cases[arm]['out'].detach().cpu().float()
                    check=metrics(got.reshape(-1).tolist(),ref.reshape(-1).tolist())
                    assert check['passed'],(name,control,arm,'FP64 gate',check)
                    record['numeric'].append(dict(control=control,arm=arm,full_output_elements=got.numel(),FP64=check,
                        repeat_full_output_partial_bitwise=True,source_readonly=True,cache_append_sha256=expected_sha))
                cross={f:bool(np.array_equal(outputs['native'][f],outputs['triton'][f])) for f in outputs['native']}
                np.savez_compressed(args.out/(name+'-control'+str(control)+'.npz'),
                    **{arm+'_'+f:a for arm,d in outputs.items() for f,a in d.items()},reference=ref.numpy(),q_bits=bits(q_cpu).numpy(),
                    new_k_bits=bits(new_k).numpy(),new_v_bits=bits(new_v).numpy(),table=tab_cpu.numpy(),lengths=np.array(lengths),slots=slot_cpu.numpy())
                record.setdefault('cross_arm_bitwise',[]).append(dict(control=control,fields=cross,required=False,
                    note='Each arm independently passes hard FP64; different arithmetic orders may differ in bits'))
                if control==0:control0=(q_cpu,new_k,new_v,slot_cpu,lengths,expected_sha,readonly)
                checkpoint()
            for role,kernel in handles.items():
                kernel._init_handles();record.setdefault('actual_compiled',{})[role]=preserve(kernel,args.out/(name+'-'+role))
            # Restore exactly the dense canonical control, outside all timing.
            qc,kc,vc,sc,ll,expected_sha,dense_readonly=control0
            q.copy_(qc);nk.copy_(kc);nv.copy_(vc);slots.copy_(sc);lens.copy_(torch.tensor(ll,dtype=torch.int32))
            arena.copy_(cold);append();torch.cuda.synchronize()
            assert digest(raw(arena))==expected_sha and fingerprints()==dense_readonly
            graphs={}
            for arm in ('native','triton'):
                for _ in range(8):step(arm)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.unroll):step(arm)
                graphs[arm]=graph
                poison(arm);graph.replay();torch.cuda.synchronize();graph_bits=capture_output(arm,ll)
                poison(arm);step(arm);torch.cuda.synchronize();eager_bits=capture_output(arm,ll)
                assert all(np.array_equal(graph_bits[f],eager_bits[f]) for f in graph_bits),'Graph/eager mismatch'
            warm={arm:event_ms(graphs[arm].replay,args.warm_replays)/args.unroll for arm in graphs}
            measurements=[]
            for arm in ORDER:
                wall=time.monotonic()
                ms=event_ms(graphs[arm].replay,args.timing_replays)/args.unroll
                wall_ms=(time.monotonic()-wall)*1000/(args.timing_replays*args.unroll)
                measurements.append(dict(arm=arm,complete_append_producer_reducer_ms=ms,
                    Graph_host_launch_and_sync_wall_ms_per_step=wall_ms,replays=args.timing_replays,unroll=args.unroll))
            assert fingerprints()==dense_readonly and digest(raw(arena))==expected_sha
            means={arm:sum(v['complete_append_producer_reducer_ms'] for v in measurements if v['arm']==arm)/2 for arm in graphs}
            record['timing']=dict(order=list(ORDER),ABBA=measurements,means=means,
                native_over_triton=means['triton']/means['native'],warm_ms_per_step=warm,warm_replays=args.warm_replays,
                no_model_extrapolation=True,complete_step=True)
            assert digest(raw(cold))==record['initial_source_sha256'],'CPU cold source changed'
            record['elapsed_s']=time.monotonic()-started;record['passed']=True
            record['Graph_eager_full_bitwise']=True;record['timing_source_readonly']=True;checkpoint()
            del graphs,cases,backs,backing,arena,k,v,cold
            torch.cuda.empty_cache()
        result.update(completed=True,passed=True)
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)))
        raise
    finally:checkpoint()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-device',action='store_true')
    p.add_argument('--commit');p.add_argument('--out',type=Path);p.add_argument('--library',type=Path)
    p.add_argument('--case',action='append',choices=CASES);p.add_argument('--unroll',type=int,default=8)
    p.add_argument('--warm-replays',type=int,default=128);p.add_argument('--timing-replays',type=int,default=64)
    args=p.parse_args()
    assert 1<=args.unroll<=16 and 32<=args.warm_replays<=256 and 16<=args.timing_replays<=128
    if args.run_device:run(args)
    else:print(json.dumps(plan(),indent=2))
if __name__=='__main__':main()
