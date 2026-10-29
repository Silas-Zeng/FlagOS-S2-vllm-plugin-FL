#!/usr/bin/env python3
"""Narrow source/policy checks; optionally verify the two complete actual archives."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
sys.dont_write_bytecode=True
import biv150_konly_q32_w8_numeric as adapter

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-raw',type=Path);p.add_argument('--candidate-raw',type=Path);p.add_argument('--static-review',type=Path);a=p.parse_args()
    f=adapter.framework();s=f.source;c=f.contract;checks=[]
    def check(ok,label):adapter.need(ok,label);checks.append(label)
    def reject(fn,label):
        try:fn()
        except (ValueError,KeyError,AssertionError):checks.append(label);return
        raise ValueError('mutation accepted: '+label)
    plan=c.plan();worker=s.run_source();ast.parse(worker)
    check(c.WARPS=={'w16':16,'w8':8} and c.STAGES=={'w16':1,'w8':1},'accepted w16/S1 versus fixed w8/S1 only')
    check(not any(x in worker for x in ('elapsed_time','perf_counter','Event(','performance_measured=True')),'no timing or timing warmup')
    check("launch_options[role]=dict(num_warps=f.WARPS[role],num_stages=1" in worker,'actual per-role launch geometry')
    check("graph_b={mode:stage('Graph-B-first'" in worker and worker.index("graph_b={mode:stage('Graph-B-first'")<worker.index("stage('eager-B-after-Graph'"),'unchanged changed Graph before fresh eager')
    check("c.need(readonly==immutable and byte_cpu(cache_after)==encoded" in worker and "persist('pattern-'" in worker,'complete output/cache/CPU inputs and readonly retained')
    check('cross_layout_full_output_bitwise=' in worker and 'cross_layout_bitwise_required=False' in worker and 'hoisted_unhoisted_bitwise=' not in worker,'cross-layout equality recorded outside hard gates')
    check(c.LENGTHS=={0:6144,1:6143} and c.LENGTHS[1]%16==15 and c.LENGTHS[1]%64==63,'unchanged real causal tail and physical descriptors')
    with tempfile.TemporaryDirectory(prefix='cjj-w8-numeric-policy-') as tmp:
        folder=Path(tmp)/'references';c.export_references(folder);ref,policy,expanded=c.load_reference(folder)
        hard={k:True for k in ('finite','codec','output_canaries','read_only_inputs','actual_IR','quantized_reference')}
        check(policy(hard,False,True)['original_BF16_quality_passed'] is False,'diagnostic retains BF16 proxy failure')
        reject(lambda:policy(hard,False,False),'strict default rejects BF16 proxy failure')
        for key in hard:reject(lambda k=key:policy(dict(hard,**{k:False}),False,True),'diagnostic cannot bypass '+key)
        check(policy(hard,True,False)['requested_component_gates_passed'] is True,'different layout bits cannot replace FP64 gates')
        sources,_=s.dependencies().contract.reference_sources(folder)
        old=ast.parse(sources['biv150_konly_q32_numeric_contract.py']);new=ast.parse(expanded)
        funcs=lambda t:{n.name:ast.dump(n) for n in t.body if isinstance(n,ast.FunctionDef) and n.name!='references'}
        check(funcs(old)==funcs(new),'original codec and numerical metric AST unchanged')
        check(c.snapshot_module('w16') is not None and c.snapshot_module('w8') is not None,'separate recorders load stdlib only')
    actual=None
    if any((a.control_raw,a.candidate_raw,a.static_review)):
        adapter.need(all((a.control_raw,a.candidate_raw,a.static_review)),'all actual archives/review required')
        pre=c.preflight(a.control_raw,a.candidate_raw,a.static_review);check(True,'all actual3219+36 SHA and accepted S1 audit replayed')
        for role,raw in (('w16',a.control_raw),('w8',a.candidate_raw)):
            blob=(raw/'out/triton-cache'/pre['cache'][role]['directory']/'kernel_cjj_konly_q32.cubin').read_bytes()
            resource=c.RESOURCES[role];metadata=types.SimpleNamespace(**pre['metadata'][role])
            stub=types.SimpleNamespace(kernel=blob,_init_handles=lambda:None,n_regs=resource['n_regs'],n_spills=0,metadata=metadata)
            observed=c.check_loaded(role,stub,pre['records'][role],pre['records'][role])
            check(observed['threads']==c.WARPS[role]*64,'CPU metadata-stub loader geometry only '+role)
        stub.metadata.num_warps=16
        reject(lambda:c.check_loaded('w8',stub,pre['records']['w8'],pre['records']['w8']),'reject wrong candidate loader geometry (CPU stub)')
        for role,key,value in (('w8','ABI',{}),('w16','hash',c.HASHES['w8'])):
            records=copy.deepcopy(pre['records']);records[role][key]=value
            reject(lambda:c.check_pair(records,pre['metadata']),'reject actual '+role+'/'+key)
        for key,value in (('use_sme',0),('num_warps',True),('shared',None)):
            md=copy.deepcopy(pre['metadata']);md['w8'][key]=value
            reject(lambda:c.check_pair(pre['records'],md),'reject actual metadata '+key)
        with tempfile.TemporaryDirectory(prefix='cjj-w8-admission-') as tmp:
            path=Path(tmp)/'pending.json';args=types.SimpleNamespace(commit='f'*40,preflight_sha256='e'*64,admission=path)
            for key,value in (('Root_numeric_approved',False),('Root_numeric_approved',1),('num_seqs',2),('timing_allowed',True)):
                receipt=dict(c.admission_template(args.commit,args.preflight_sha256),Root_numeric_approved=True);receipt[key]=value
                path.write_text(json.dumps(receipt));args.admission_sha256=adapter.sha(path.read_bytes())
                reject(lambda:c.admit(args,pre),'reject Root receipt '+key+'/'+str(value))
        actual=dict(provenance=pre['provenance'],hashes=c.HASHES,cubins=c.CUBINS,complete_typed_ABI_equal=True,
            metadata_difference=['hash','ir_override','num_warps'],static_review_sha256=c.REVIEW_SHA,
            prepared_proof_sha256=adapter.sha(json.dumps(pre,sort_keys=True,indent=2,allow_nan=False)+'\n'))
    check(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'CPU/no framework import')
    print(json.dumps(dict(schema='cjj-w16-w8-numeric-cpu-v1',checks=checks,count=len(checks),actual=actual,
        driver_sha256=adapter.sha(worker),plan_sha256=adapter.sha(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+'\n'),
        GPU_executed=False,numeric_verified=False,Graph_verified=False,precision_gate=False,performance_measured=False,service_verified=False),sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
