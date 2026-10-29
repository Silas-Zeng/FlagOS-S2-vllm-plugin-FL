#!/usr/bin/env python3
"""Narrow binding/negative tests; neither imports GPU frameworks nor claims numerics."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import types
import biv150_konly_q64_compile_source as c
import biv150_konly_q64_screen_contract as f


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--references',required=True,type=Path);p.add_argument('--execution-sources',required=True,type=Path);p.add_argument('--failed-raw',required=True,type=Path);a=p.parse_args()
    pre=f.preflight(a.raw,a.review,a.references);old,expanded=f.reference_sources(a.references)
    source_proof=f.execution_sources(a.execution_sources)
    failed=json.loads((a.failed_raw/'out/q32/record.json').read_text())
    c.need(c.sha((a.failed_raw/'download-manifest.json').read_bytes())=='c05a64453e02915ce67b4062c3778a64b8c9fc7c94450c3414cbaed9037eff45','original failed raw identity')
    for item in json.loads((a.failed_raw/'download-manifest.json').read_text())['files']:
        data=(a.failed_raw/item['relative']).read_bytes();c.need(len(data)==item['bytes'] and c.sha(data)==item['sha256'],'failed raw member')
    c.need(expanded.replace(f.ROWS_AFTER,f.ROWS_BEFORE)==old['biv150_konly_q32_numeric_contract.py'],'only reference row-list expansion')
    source=(c.ROOT/'tools/probe_biv150_konly_q64_screen.py').read_text();tree=ast.parse(source)
    def uniform(t):
        return next(n for n in ast.walk(t) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=='pattern == 0' and any(isinstance(k,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='visible' for x in k.targets) for k in n.body))
    c.need(ast.dump(uniform(tree))==ast.dump(uniform(ast.parse(old['bench_biv150_konly_service_q32_bitwise.py']))),'uniform full-row original math changed')
    reference,policy,_=f.load_reference(a.references)
    metrics=reference.metrics([1.0],[1.0]);c.need(metrics['pointwise_passed'] and metrics['relative_rmse']==0,'original metric callable')
    c.need(not reference.metrics([1.02],[1.0])['pointwise_passed'],'original .005+.005 threshold preserved')
    count=0
    def reject(call):
        nonlocal count
        try:call()
        except (ValueError,KeyError,TypeError):count+=1;return
        raise AssertionError('negative binding accepted')
    hard={k:True for k in ('finite','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise','quantized_reference')}
    c.need(policy(hard,False,True)['precision_gate'] is False,'BF16 proxy diagnostic must not grant precision')
    reject(lambda:policy(hard,False,False))
    for name in hard:
        bad=dict(hard);bad[name]=False;reject(lambda:policy(bad,False,True))
    for role in f.ROLES:
        expected=pre['records'][role];resource=f.RESOURCES[role]
        def kernel(**override):
            fields={k:resource[k] for k in ('n_regs','n_spills','shared')};fields.update(override)
            return types.SimpleNamespace(kernel=(a.raw/'out'/role/'hoisted/actual.cubin').read_bytes(),n_regs=fields['n_regs'],n_spills=fields['n_spills'],metadata=types.SimpleNamespace(shared=fields['shared'],num_warps=16,num_stages=1,warp_size=64),_init_handles=lambda:None)
        f.check_loaded(role,kernel(),copy.deepcopy(expected),expected)
        if role=='q32':
            observed=f.check_loaded(role,kernel(),failed,expected)
            c.need(observed['diagnostic_ASM']['bytes_identical'] is False,'actual first-stop ASM difference lost')
        for key in ('source','ttir','ttgir','llir','cubin'):
            bad=copy.deepcopy(expected);bad['artifacts'][key]='0'*64
            reject(lambda:f.check_loaded(role,kernel(),bad,expected))
        for value in (None,'unknown'):
            bad=copy.deepcopy(expected);bad['artifacts']['asm']=value
            reject(lambda:f.check_loaded(role,kernel(),bad,expected))
        wrong=kernel();wrong.kernel=wrong.kernel+b'changed'
        reject(lambda:f.check_loaded(role,wrong,expected,expected))
        for change in (dict(n_regs=None),dict(n_regs=resource['n_regs']+1),dict(n_spills=1),dict(n_spills=False),dict(shared=65536),dict(shared=None)):
            reject(lambda:f.check_loaded(role,kernel(**change),expected,expected))
        for key in ('hash','source_digest','function_name','ABI','launch_geometry','artifacts'):
            bad=copy.deepcopy(expected);bad[key]=None;reject(lambda:f.check_loaded(role,kernel(),bad,expected))
    c.need(source.index("graph_b={mode:stage('Graph-B-first'")<source.index("stage('eager-B-after-Graph'"),'Graph B must precede fresh eager B')
    c.need('normalized2d128' not in source and 'capture_bypass' not in source,'no fallback Graph role')
    for name in ('biv150_konly_q64_screen_contract.py','probe_biv150_konly_q64_screen.py','check_biv150_konly_q64_screen_cpu.py'):
        compile((c.ROOT/'tools'/name).read_bytes(),name,'exec')
    c.need(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'CPU checker imported framework')
    result=dict(completed=True,errors=[],compile_manifest_sha256=f.MANIFEST,actual_review_sha256=f.REVIEW,
        reference_commit=f.REFERENCE_COMMIT,reference_original_pins=f.REFERENCES,expanded_reference_sha256=c.sha(expanded),
        reference_exact_inverse=True,uniform_fullrow_math_AST_identical=True,negative_cases_rejected=count,execution_path_source_proof=source_proof,actual_failed_ASM_only_snapshot_replayed=True,
        resource_binding='exact Q32/Q64 counts; runtime0 and ELF scalar spill96/485 separate',
        sources={name:c.sha((c.ROOT/'tools'/name).read_bytes()) for name in ('biv150_konly_q64_screen_contract.py','probe_biv150_konly_q64_screen.py','check_biv150_konly_q64_screen_cpu.py')},
        GPU_executed=False,numeric_verified=False,Graph_verified=False,side_stream_verified=False,performance_verified=False,service_verified=False)
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=='__main__':main()
