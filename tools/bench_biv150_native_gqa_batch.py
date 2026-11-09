"""CPU plan by default. Root-only short N48/56 native/TL+FP64, no timing."""
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
from biv150_native_gqa_batch_factory import ROOT,MAIN,INTEGRATION,render,sha,CFLAGS,CUDAFLAGS,NAMESPACE
from biv150_native_gqa_batch_runtime import ENV,PRIVATE,StockFixture,create_runtime
from biv150_native_gqa_batch_admission import Admission


def plan():
    return dict(main=MAIN,integration_main=INTEGRATION,rows=[48,56],segments=4,max_context=193,
        patterns=['dense contiguous Q','ragged including zero/tail, padded Q stride2560'],
        arms=['new private native S4','byte-exact b3 TL GQA plus b3 reducer'],
        FP64_pointwise='abs_error <= .005+.005*abs(reference)',FP64_relative_RMSE_max=.005,
        default_CPU=True,service_activation=False,GPU_executed=False,timing=None,writer=None,fullcost=None,
        Graph_verified=False,side_stream_verified=False,N40_coverage=False,
        actual_4k_16k_numeric_verified=False,performance_verified=False)


def save(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod



def run(a):
    assert a.out and a.out.is_absolute() and not a.out.exists() and a.library and a.build_proof
    assert a.commit==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    assert not subprocess.check_output(['git','-C',str(ROOT),'diff',MAIN,'--','vllm_fl','csrc'])
    assert all(os.environ.get(k,'0')!='1' for k in PRIVATE),'private cache format enabled'
    proof=json.loads(a.build_proof.read_text());assert sha(a.library.read_bytes())==proof['library_sha256']
    source_proof=a.build_proof.parent/'source-proof.json';assert sha(source_proof.read_bytes())==proof['source_proof_sha256']
    manifest=json.loads(source_proof.read_text());generated=manifest['generated']
    assert manifest['cflags']==CFLAGS and manifest['cuda_flags']==CUDAFLAGS and manifest['namespace']==NAMESPACE
    for name,text in render()[0].items():assert sha(text.encode())==generated[name] and (source_proof.parent/name).read_bytes()==text.encode()
    a.out.mkdir();result=dict(**plan(),completed=False,passed=False,errors=[],records=[],commit=a.commit,
        library_sha256=proof['library_sha256'],build_proof_sha256=sha(a.build_proof.read_bytes()),
        diagnostic_admission=dict(tiny_initialization_outside_numerics=True))
    checkpoint=lambda:save(a.out/'summary.json',result);checkpoint()
    try:
        with socket.socket() as s:port_code=s.connect_ex(('127.0.0.1',9031))
        result['diagnostic_admission']['port9031_probe_code']=port_code
        assert port_code!=0,'service9031 busy'
        result['diagnostic_admission']['port9031_closed']=True
        guard=Admission(a.root_owned_exclusive)
        guard.report.update(result['diagnostic_admission'])
        result['diagnostic_admission']=guard.report
        guard.check('pre-CUDA')
        os.environ['TRITON_CACHE_DIR']=str(a.out/'triton-cache')
        import torch
        import numpy as np
        torch.set_num_threads(4);tiny=torch.zeros(1,device='cuda');torch.cuda.synchronize()
        result['GPU_executed']=True;result['diagnostic_admission']['tiny_cuda_initializations']=1
        guard.check('post-tiny-initialization')
        def admission():guard.check()
        admission();checkpoint()
        from vllm.triton_utils import triton
        target=triton.runtime.driver.active.get_current_target();assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        props=torch.cuda.get_device_properties(0);assert 'BI-V150' in props.name and props.multi_processor_count==16
        assert not hasattr(torch.ops.cjj_native_gqa_batch_s4_20261028,'produce'),'namespace already registered'
        torch.ops.load_library(str(a.library))
        base=module(ROOT/'vllm_fl/patches/biv150_gqa_decode.py','cjj_batch_b3_gqa')
        helper=module(ROOT/'tools/biv150_benchmark_utils.py','cjj_batch_b3_helper')
        result['imports']=dict(torch=torch.__file__,triton=triton.__file__,base=base.__file__,helper=helper.__file__)
        result['torch_version']=str(torch.__version__);result['actual_target']=[target.backend,target.arch,target.warp_size]
        def bits(t):
            h=t.detach().cpu().contiguous();return h.view({2:torch.int16,4:torch.int32}[h.element_size()]).numpy().copy()
        def digest(t):return sha(bits(t).tobytes())
        handles={}
        for n in a.rows or (48,56):
            admission();assert base._pick_segments(n,2,16)==4
            g=torch.Generator(device='cpu').manual_seed(20261028+n);pages=n*13
            cpu=torch.randn((pages,2,16,2,128),generator=g,dtype=torch.bfloat16)
            fixture=StockFixture(torch,cpu);runtime=create_runtime(torch,a.library,fixture,environ={ENV:'1'})
            q0=torch.randn((n,16,128),generator=g,dtype=torch.bfloat16)
            tab_cpu=torch.full((n,8192),-1,dtype=torch.int32)
            # Permuted unique physical pages, including nonzero token offsets.
            perm=torch.randperm(pages,generator=g).reshape(n,13);tab_cpu[:,:13]=perm.int()
            table=tab_cpu.to('cuda');key,value=fixture.key,fixture.value;cache_hash=digest(fixture.cache)
            for control in (0,1):
                admission();lengths=[193]*n
                if control:
                    for i,x in enumerate((0,1,15,16,17,31,32,33,63,64,65,127,128,129,192,193)):lengths[i]=x
                qc=q0 if not control else -q0.roll(1,1)
                qback=torch.zeros((n,16 if not control else 20,128),device='cuda',dtype=torch.bfloat16)
                q=qback[:,:16];q.copy_(qc);lens=torch.tensor(lengths,dtype=torch.int32,device='cuda')
                ref=[]
                for r,length in enumerate(lengths):
                    if not length:ref.append(torch.zeros((16,128),dtype=torch.float64));continue
                    ids=tab_cpu[r,:(length+15)//16].long()
                    k=cpu[ids,0].reshape(-1,2,128)[:length].double();v=cpu[ids,1].reshape(-1,2,128)[:length].double()
                    scores=torch.einsum('hgd,thd->hgt',qc[r].double().reshape(2,8,128),k)*128**-.5
                    ref.append(torch.einsum('hgt,thd->hgd',scores.softmax(-1),v).reshape(16,128))
                ref=torch.stack(ref);record=dict(rows=n,control=control,grid=[n,2,4],numeric=[],timing=None)
                result['records'].append(record);arrays=dict(reference=ref.numpy(),q_bits=bits(qc),cache_bits=bits(cpu),table=tab_cpu.numpy(),lengths=np.array(lengths))
                readonly=dict(q=digest(q),lengths=sha(lens.cpu().numpy().tobytes()),table=sha(table.cpu().numpy().tobytes()))
                for arm in ('triton','native'):
                    handles={};admission();c=dict(q=q,k=key,v=value,table=table,lengths=lens);guards={}
                    for field,shape,dtype in [('scratch',(n,16,16,128),torch.float32),('maxima',(n,16,16),torch.float32),('sums',(n,16,16),torch.float32),('out',(n,16,128),torch.bfloat16)]:
                        full=torch.full((math.prod(shape)+128,),42.5,device='cuda',dtype=dtype)
                        wide=full[64:-64].reshape(shape);wide.fill_(math.nan);guards[field]=(full,wide)
                        c[field]=wide if field=='out' else wide.narrow(2,0,4)
                    class Capture:
                        def __getitem__(self,grid):return lambda **kw:kw
                    kw=helper.main_call(type('Recorder',(),dict(_gqa_decode_attention=Capture(),LOG2E=base.LOG2E))(),c,4)
                    if arm=='native':assert runtime.try_produce((n,2,4),kw),'new native scope rejected'
                    else:handles['producer']=helper.main_call(base,c,4)
                    handles['reducer']=helper.reduce_call(base,c,4,4);torch.cuda.synchronize();admission()
                    actual=c['out'].detach().cpu().double();arrays[arm+'_out']=actual.numpy()
                    checks=[]
                    for field,(full,wide) in guards.items():
                        h=full.cpu();mid=wide.cpu();checks.append(bool(torch.all(h[:64]==42.5)&torch.all(h[-64:]==42.5)))
                        arrays[arm+'_'+field+'_bits']=bits(wide)
                        if field!='out':
                            for r,length in enumerate(lengths):
                                step=(length+63)//64
                                for seg in range(16):
                                    active=seg<4 and seg*step*16<length
                                    checks.append(bool(torch.isfinite(mid[r,:,seg]).all()) if active else bool(torch.isnan(mid[r,:,seg]).all()))
                                    if active and field=='sums':checks.append(bool((mid[r,:,seg]>0).all()))
                        else:
                            checks.append(bool(torch.isfinite(mid).all()))
                            for row,length in enumerate(lengths):
                                if length==0:checks.append(int(torch.count_nonzero(mid[row]))==0)
                    checks.append(bool(torch.all(fixture.backing[:256]==90)&torch.all(fixture.backing[-256:]==90)))
                    checks.append(digest(fixture.cache)==cache_hash and digest(q)==readonly['q'])
                    checks.append(sha(lens.cpu().numpy().tobytes())==readonly['lengths'] and sha(table.cpu().numpy().tobytes())==readonly['table'])
                    delta=actual-ref;viol=int((delta.abs()>(.005+.005*ref.abs())).sum());rms=float(delta.square().mean().sqrt());rr=float(ref.square().mean().sqrt())
                    relative=rms/rr if rr else (0. if rms==0 else math.inf)
                    item=dict(arm=arm,finite=bool(torch.isfinite(actual).all()),pointwise_violations=viol,relative_rmse=relative,
                              max_abs=float(delta.abs().max()),hard_guards=all(checks),passed=all(checks) and viol==0 and relative<=.005)
                    record['numeric'].append(item)
                    record.setdefault('actual_compiled',{})[arm]={}
                    for role,h in handles.items():
                        folder=a.out/(str(n)+'-'+str(control)+'-'+arm+'-'+role);folder.mkdir();files={}
                        for ext,data in h.asm.items():
                            if isinstance(data,(str,bytes)):
                                b=data.encode() if isinstance(data,str) else data;(folder/('actual.'+ext)).write_bytes(b);files[ext]=sha(b)
                        loaded=dict(hash=str(h.hash),n_regs=getattr(h,'n_regs',None),n_spills=getattr(h,'n_spills',None),metadata=str(h.metadata),files=files)
                        save(folder/'loaded.json',loaded);record['actual_compiled'][arm][role]=loaded
                    np.savez_compressed(a.out/(str(n)+'-'+str(control)+'.npz'),**arrays);checkpoint()
                    assert item['passed'],item
                record['passed']=True;checkpoint()
        result.update(completed=True,passed=True,GPU_executed=True);checkpoint()
    except BaseException as e:
        result['errors'].append(dict(type=type(e).__name__,message=str(e)));checkpoint();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-device',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--commit');p.add_argument('--out',type=Path)
    p.add_argument('--library',type=Path);p.add_argument('--build-proof',type=Path);p.add_argument('--rows',type=int,choices=(48,56),action='append');a=p.parse_args()
    if not a.run_device:print(json.dumps(plan(),indent=2))
    else:
        assert a.root_owned_exclusive,'Root exclusive scheduling acknowledgement required'
        assert len(set(a.rows or (48,56)))==len(a.rows or (48,56)),'duplicate rows'
        run(a)
