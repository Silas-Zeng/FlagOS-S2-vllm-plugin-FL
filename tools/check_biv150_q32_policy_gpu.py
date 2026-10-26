"""Validate the actual opt-in Q32 serving policy against its BM128 fallback."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import traceback

CASES = {
    'fresh_minimum1024': ([1024], [1024]),
    'minimum16k': ([1024], [16384]),
    'suffix4k': ([1985], [4096]),
    'suffix16k': ([2017], [16384]),
    'small_fallback': ([256], [4096]),
    'multi_fallback': ([1,17,33], [65,130,257]),
    'below_threshold': ([1023], [4096]),
    'at_threshold': ([1024], [4096]),
    'partial_q_tile': ([1025], [4096]),
    'observed_tail1739': ([1739], [4096]),
    'observed_tail1798': ([1798], [4096]),
    'fresh_chunk2048': ([2048], [2048]),
}


def main(args):
    cases = {name: CASES[name] for name in args.case} if args.case else CASES
    if not args.run_gpu:
        print(json.dumps(cases))
        return
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',9031)) != 0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    import biv150_prefill_probe_fixtures as fixture
    fixture.torch=torch
    torch.set_num_threads(4)
    function,base,states=fixture.install_baseline()
    original=base.holder[base.kernel_key]
    os.environ.pop('VLLM_FL_BIV150_PREFILL_Q32', None)
    module=importlib.import_module('vllm_fl.patches.biv150_prefill_q32')
    installed=module.install()
    assert installed['installed'] and installed['num_warps']==16,installed
    policy=base.holder[base.kernel_key]
    manager=module._MANAGER
    result=dict(completed=False,official_score=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),installed=installed,
                feature_env_absent='VLLM_FL_BIV150_PREFILL_Q32' not in os.environ,cases=[],errors=[])
    def save():
        temp=args.out/'result.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.out/'result.json')
    save()
    try:
        for name,(qlens,klens) in cases.items():
            row=dict(name=name,checks=[]);result['cases'].append(row)
            host=fixture.cpu_inputs(qlens,klens,20261025,0)
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                device={k:v.cuda() for k,v in host.items()}
                key,value=device['packed'].unbind(1);out=torch.empty_like(device['q'])
            stream.synchronize()
            def call(route):
                base.holder[base.kernel_key]=original if route=='baseline' else policy
                with torch.cuda.stream(stream):
                    out.fill_(float('nan'))
                    function(q=device['q'],k=key,v=value,out=out,cu_seqlens_q=device['cu_q'],
                             max_seqlen_q=max(qlens),seqused_k=device['lengths'],max_seqlen_k=max(klens),
                             softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=device['table'],
                             softcap=0,q_descale=None,k_descale=device['k_scale'],v_descale=device['v_scale'])
                stream.synchronize()
                return out.cpu().clone()
            for pattern in (0,1):
                host=fixture.cpu_inputs(qlens,klens,20261025,pattern)
                with torch.cuda.stream(stream):
                    for k,v in host.items():device[k].copy_(v)
                stream.synchronize()
                expected=call('baseline');before=manager.selected_calls;actual=call('candidate')
                should_select=len(qlens)==1 and sum(qlens)>=1024
                assert (manager.selected_calls>before)==should_select
                indices,reference=fixture.reference(host,qlens,klens,pattern==0,False)
                check=dict(pattern=pattern,selected=should_select,
                           direct=fixture.compare(actual,expected,.005,.005),
                           candidate_cpu=fixture.compare(actual[indices],reference,.005,.005),
                           baseline_cpu=fixture.compare(expected[indices],reference,.005,.005))
                row['checks'].append(check);save()
                if not all(check[k]['passed'] for k in ['direct','candidate_cpu','baseline_cpu']):
                    torch.save(dict(actual=actual,baseline=expected,reference=reference,reference_rows=indices),args.out/(name+'-failure.pt'))
                    raise RuntimeError(name+' numerical gate failed')
                if not should_select:assert torch.equal(actual,expected)
                for k,v in host.items():
                    assert torch.equal(device[k].cpu().contiguous().view(-1).view(torch.uint8),v.contiguous().view(-1).view(torch.uint8)),k
            row['completed']=True;save()
            print(json.dumps({'case':name,'passed':True}),flush=True)
        assert not policy.disabled
        result['candidate_proof']=manager.snapshot()
        selected=[r for r in result['candidate_proof']['compile_records'] if r['mode']=='selected']
        assert selected and all(r['candidate_resources']['metadata']['num_warps']==16
                                and r['candidate_resources']['metadata']['num_stages']==1
                                and r['mechanism']['mechanism_pass'] for r in selected)
        result['completed']=True
    except Exception as error:
        result['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        base.holder[base.kernel_key]=policy
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--case', action='append', choices=sorted(CASES),
                   help='Select cases; default is the entire numerical/scope matrix')
    args=p.parse_args()
    if args.run_gpu and args.out is None:p.error('--out required')
    main(args)
