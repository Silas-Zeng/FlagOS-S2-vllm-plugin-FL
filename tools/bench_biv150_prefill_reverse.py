#!/usr/bin/env python3
"""Component-only reverse query-CTA experiment on the actual main Q32 policy.

No production or SDK files are edited. GPU execution requires --run-gpu and
an idle coordinated BI-V150; graph timings replay warmed proved private kernels.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import traceback
from types import SimpleNamespace

from biv150_prefill_reverse import MAIN, CASES, eligible, private_manager
from biv150_resource_records import resource_evidence, compare_resource_records


def atomic(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


class Tap:
    def __init__(self, manager, fragment):
        self.manager,self.fragment=manager,fragment
        self.grid=self.kwargs=self.compiled=None

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            if args or not eligible(kwargs,grid,self.fragment):
                raise RuntimeError('Experiment requires exact parent-selected single-sequence 2D Q32 grid')
            self.grid,self.kwargs=tuple(grid),kwargs.copy()
            self.compiled=self.manager.wrapper[grid](**kwargs)
            if self.compiled.hash not in self.manager.optimized:
                raise RuntimeError('Actual Q-hoist proof was not selected')
            return self.compiled
        return invoke


def descriptors(kwargs):
    result={}
    for key,value in kwargs.items():
        if hasattr(value,'data_ptr'):
            result[key]=dict(address=value.data_ptr(),shape=list(value.shape),
                stride=list(value.stride()),dtype=str(value.dtype),storage_offset=value.storage_offset())
        else:
            result[key]=value if isinstance(value,(str,int,float,bool,type(None))) else repr(value)
    return result


def resources(compiled,fragment,out,label,manager=None):
    record=fragment._resources(compiled)
    loaded_record=manager.optimized.get(compiled.hash) if manager is not None else None
    record['post_launch_resource_evidence']=resource_evidence(record,loaded_record,already_launched=True)
    record['files']={}
    for kind,value in compiled.asm.items():
        if not isinstance(value,(str,bytes)):continue
        data=value.encode() if isinstance(value,str) else value
        path=out/(label+'-'+compiled.hash+'.'+kind)
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
        record['files'][kind]=dict(path=path.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return record


def main(args):
    if not args.run_gpu:
        print(json.dumps(dict(cpu_only=True,gpu_executed=False,cases=CASES,
            component_only=True,official_score=False,precision_gate=False,
            exact_parent_policy=True,geometry=[256,32,64,16,1],dynamic_metadata_prepared=True,resource_mode=args.resource_mode)))
        return
    source=args.source.resolve()
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==args.commit
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    subprocess.run(['git','-C',str(source),'diff','--exit-code',MAIN,'--','vllm_fl','csrc'],check=True)
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0,'Existing service; nothing stopped'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True),'GPU busy'
    assert os.environ.get('VLLM_FL_BIV150_PREFILL_Q32','1')=='1','Main default Q32 policy disabled'
    args.out.mkdir(parents=False,exist_ok=False)
    sys.path.insert(0,str(source))
    import torch
    import vllm_fl
    import biv150_prefill_probe_fixtures as fixture
    assert Path(vllm_fl.__file__).resolve()==source/'vllm_fl/__init__.py','Wrong imported plugin checkout'
    fixture.torch=torch;torch.set_num_threads(4)
    props=torch.cuda.get_device_properties(0)
    assert 'BI-V150' in props.name and props.multi_processor_count==16
    function,base,states=fixture.install_baseline()
    fragment=importlib.import_module('vllm_fl.patches.biv150_q_fragment_service')
    q32=importlib.import_module('vllm_fl.patches.biv150_prefill_q32')
    installed=q32.install();assert installed['installed'],installed
    policy=base.holder[base.kernel_key]
    parent=q32._MANAGER
    assert isinstance(policy,q32._Policy) and policy.manager is parent
    original_wrapper=parent.wrapper
    # Tap calls manager.wrapper, so baseline's stable launch object must not
    # follow the temporary parent.wrapper route selector recursively.
    baseline_view=type('BaselineView',(),{})()
    baseline_view.wrapper=original_wrapper;baseline_view.optimized=parent.optimized
    attention=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    backend=importlib.import_module('vllm.v1.attention.backends.triton_attn')
    reverse,proof=private_manager(attention,backend,parent,q32,source,args.out)
    taps={'baseline':Tap(baseline_view,fragment),'candidate':Tap(reverse,fragment)}
    managers={'baseline':parent,'candidate':reverse}
    result=dict(completed=False,component_only=True,official_score=False,precision_gate=False,
        production_unchanged=True,service_installed=False,source_commit=args.commit,
        production_baseline=MAIN,source_transform=proof,baseline_installations=states,
        q32_installation=installed,actual_parent_policy=True,
        dynamic_metadata_requested=args.dynamic_metadata,resource_mode=args.resource_mode,
        refined_route_numerical_matrix_complete=False,complete_resource_equivalence_claimed=False,cases=[],errors=[])
    def save():atomic(args.out/'result.json',result)
    save()
    try:
        if args.dynamic_metadata:
            from biv150_prefill_reverse_dynamic import run_dynamic_metadata
            run_dynamic_metadata(SimpleNamespace(torch=torch,fixture=fixture,parent=parent,
                base=base,policy=policy,function=function,taps=taps,managers=managers,
                reverse=reverse,fragment=fragment,result=result,save=save,out=args.out,
                descriptors=descriptors,resources=resources))
        for name in args.case or CASES:
            qlens,klens=CASES[name]
            row=dict(name=name,qlens=qlens,klens=klens,checks=[],rounds=[],completed=False)
            result['cases'].append(row);save()
            host=fixture.cpu_inputs(qlens,klens,20261027,0)
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                device={key:value.cuda() for key,value in host.items()}
                key_cache,value_cache=device['packed'].unbind(1)
                output_storage=torch.full((sum(qlens)+2,16,128),43.0,device='cuda',dtype=torch.bfloat16)
                output=output_storage[1:-1]
            stream.synchronize()
            def call(label):
                parent.wrapper=taps[label]
                assert base.holder[base.kernel_key] is policy
                function(q=device['q'],k=key_cache,v=value_cache,out=output,
                    cu_seqlens_q=device['cu_q'],max_seqlen_q=max(qlens),
                    seqused_k=device['lengths'],max_seqlen_k=max(klens),
                    softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=device['table'],
                    softcap=0,q_descale=None,k_descale=device['k_scale'],v_descale=device['v_scale'])
                assert not policy.disabled,'Parent policy disabled after unproved cold candidate'
                assert base.holder[base.kernel_key] is policy
            graphs={}
            for pattern in (0,1):
                host=fixture.cpu_inputs(qlens,klens,20261027,pattern)
                with torch.cuda.stream(stream):
                    for key,value in host.items():device[key].copy_(value)
                    output.fill_(float('nan'));call('baseline')
                stream.synchronize();expected=output.cpu().clone()
                with torch.cuda.stream(stream):
                    output.fill_(float('nan'));call('candidate')
                stream.synchronize();actual=output.cpu().clone()
                assert taps['baseline'].grid==taps['candidate'].grid==(sum(qlens)//32+1,2)
                assert descriptors(taps['baseline'].kwargs)==descriptors(taps['candidate'].kwargs)
                assert fragment._abi(taps['baseline'].compiled.src)==fragment._abi(taps['candidate'].compiled.src)
                indices,reference=fixture.reference(host,qlens,klens,pattern==0,False)
                check=dict(pattern=pattern,direct=fixture.compare(actual,expected,0,0),
                    baseline_cpu=fixture.compare(expected[indices],reference,.005,.005),
                    candidate_cpu=fixture.compare(actual[indices],reference,.005,.005),
                    reference_guard=fixture.reference_guard(actual[indices],expected[indices],reference),
                    graph_checks=[])
                row['checks'].append(check);save()
                assert check['direct']['passed'] and check['reference_guard']['passed'],'Numerical identity/reference failed'
                for label,manager in managers.items():
                    tap=taps[label]
                    if label not in graphs:
                        with torch.cuda.stream(stream):
                            for _ in range(3):manager.candidate[tap.grid](**tap.kwargs)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=stream):
                                for _ in range(4):manager.candidate[tap.grid](**tap.kwargs)
                        stream.synchronize();graphs[label]=graph
                    with torch.cuda.stream(stream):
                        output.fill_(float('nan'));graphs[label].replay()
                    stream.synchronize()
                    comparison=fixture.compare(output.cpu(),expected if label=='baseline' else actual,0,0)
                    check['graph_checks'].append(dict(route=label,changed_input=pattern==1,**comparison))
                    assert comparison['passed'],'Changed-input graph replay differs'
                side=torch.cuda.Stream()
                side.wait_stream(stream)
                with torch.cuda.stream(side):
                    output.fill_(float('nan'));call('candidate')
                side.synchronize()
                check['side_stream']=fixture.compare(output.cpu(),actual,0,0)
                assert check['side_stream']['passed'],'Side-stream output differs'
                assert bool(torch.all(output_storage[0]==43).item()) and bool(torch.all(output_storage[-1]==43).item()),'Output canary changed'
                for key,value in host.items():
                    assert torch.equal(device[key].cpu().contiguous().view(-1).view(torch.uint8),
                                       value.contiguous().view(-1).view(torch.uint8)),'Input modified: '+key
                check['output_canaries_unchanged']=check['inputs_unchanged']=True
                save()
            row['grid']=list(taps['baseline'].grid)
            row['launch_descriptors']=descriptors(taps['baseline'].kwargs)
            for label in managers:
                row[label+'_resources']=resources(taps[label].compiled,fragment,args.out,label,managers[label])
            old,new=row['baseline_resources'],row['candidate_resources']
            for key in ('num_warps','num_stages','shared','use_sme','warp_size'):
                assert old['metadata'][key]==new['metadata'][key], 'Launch/shared resource changed: '+key
            assert old['metadata']['num_warps']==16 and old['metadata']['num_stages']==1
            assert old['metadata']['use_sme']==114691
            assert old['metadata']['shared']==65536 and old['metadata']['warp_size']==64
            comparison=compare_resource_records(old,new)
            row['resource_comparison']=comparison
            row['same_registers_and_spills']=comparison['same_registers_and_spills']
            row['resource_changed']=comparison['resource_changed']
            row['resource_mode']=args.resource_mode
            save()
            # Unknown does not pass either mode. Record mode permits known
            # register/spill changes and attributes timing to the full candidate.
            allowed=comparison['strict_allowed'] if args.resource_mode=='strict' else comparison['record_allowed']
            assert allowed, 'Observed resource gate failed or counts unknown: '+repr(comparison)
            for iteration in range(args.iterations):
                order=['baseline','candidate'] if iteration%2==0 else ['candidate','baseline']
                timing=dict(iteration=iteration,order=order)
                for label in order:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        graphs[label].replay();graphs[label].replay();begin.record()
                        for _ in range(16):graphs[label].replay()
                        end.record()
                    end.synchronize();timing[label+'_ms']=begin.elapsed_time(end)/64
                timing['speedup']=timing['baseline_ms']/timing['candidate_ms']
                row['rounds'].append(timing)
            row['median_speedup']=statistics.median(record['speedup'] for record in row['rounds'])
            row['completed']=True;save()
            print(json.dumps(dict(case=name,median_speedup=row['median_speedup'],rounds=row['rounds'])),flush=True)
            del graphs,device,key_cache,value_cache,output,output_storage,actual,expected
            torch.cuda.empty_cache()
        result['completed']=True
        result['full_matrix']=set(args.case or CASES)==set(CASES)
        result['refined_route_numerical_matrix_complete']=bool(result['full_matrix'] and
            result.get('dynamic_metadata',{}).get('completed',False))
    except BaseException as error:
        result['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        parent.wrapper=original_wrapper
        result['parent_policy_restored']=base.holder[base.kernel_key] is policy and parent.wrapper is original_wrapper
        result['terminal_manager_proofs']={}
        for label,manager in managers.items():
            result['terminal_manager_proofs'][label]=manager.snapshot()
            shutil.copytree(manager.artifact_dir,args.out/'compiler-ir'/label,dirs_exist_ok=True)
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--commit')
    p.add_argument('--out',type=Path)
    p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--dynamic-metadata',action='store_true',help='Require fixed-pointer Graph route flips and actual route-ID probe before timing')
    p.add_argument('--case',action='append',choices=CASES)
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--resource-mode',choices=('strict','record'),default='strict',help='Strict requires known matching post-launch counts; record permits known count changes and marks full-candidate costs')
    args=p.parse_args()
    if args.run_gpu and (not args.commit or not args.out):p.error('--commit and new --out required with --run-gpu')
    if args.iterations<1:p.error('--iterations must be positive')
    main(args)
