#!/usr/bin/env python3
"""N32 scope/inverse/admission CPU tests. No array regeneration or GPU packages."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from biv150_native117_n32_scope import programs,CASES,REVIEW
from check_biv150_native117_n64_pair_cpu import reject
import biv150_native117_pair_contract as original
import probe_biv150_native117_n64_fullstep as original_fullstep

def check(a):
    c,consumer,full=programs();root=a.parent_source
    assert original.CASES==((64,2,2048),(64,2,2560)) and original.REVIEW_SHA!=REVIEW
    assert original.plan()['consumer_cases']==[[64,2,2048],[64,2,2560]]
    assert c.plan()['consumer_cases']==[[32,4,2048],[32,4,2560]]
    assert c.plan()['fullstep_case']=={'16k':[32,16384,4,2560]}
    assert c.admission.__code__ is original.admission.__code__
    assert c.admission.__globals__ is not original.admission.__globals__
    assert c.consumer_source(root)==original.consumer_source(root)
    assert c.receipt_source(root)==original.receipt_source(root)
    source,proof=full.adapted_source(root)
    assert (source,proof)==original_fullstep.adapted_source(root);ast.parse(source)
    assert '(n,16,16,128)' in source and '(16384,256,128,1)' in source and 'f.UNROLL' in source
    for record in c.plan()['scope_adaptation']['scope_function_edits'].values():assert record['inverse_byte_exact']
    calls=[]
    f=SimpleNamespace(CASES={'4k':(64,4096,2,2048),'16k':(32,16384,4,2560)},
        golden=lambda *args:calls.append(args) or args,patched_chunk=lambda *args:calls.append(args) or args)
    assert original_fullstep.fixture(f).CASES=={'4k':(64,4096,2,2048)}
    selected=full.fixture(f);assert selected.CASES=={'16k':(32,16384,4,2560)}
    for role in c.ROLES:
        assert selected.golden(1,2,3,4,role)==(1,2,3,4,'k8')
        assert selected.patched_chunk(role,1,2,3)==('k8',1,2,3)
    samples={role:dict(records=[dict(case=list(case),snapshots={stage:{
        field:dict(bytes=4,sha256='0'*64,path='synthetic-CPU-only-not-a-receipt')
        for field in ('partial','maximum','summation','output')} for stage in c.STAGES}) for case in CASES]) for role in c.ROLES}
    c.compare_pair(samples);faults=[]
    for field in ('partial','maximum','summation','output'):
        bad=copy.deepcopy(samples);bad['lookahead']['records'][0]['snapshots']['A-after'][field]['sha256']='1'*64
        reject(lambda:c.compare_pair(bad));faults.append('N32_'+field+'_bit_difference')
    bad=copy.deepcopy(samples)
    for arm in bad.values():
        for row in arm['records']:row['case'][0]=64;row['case'][1]=2
    reject(lambda:c.compare_pair(bad));faults.append('N64_cases_cannot_license_N32')
    bad=copy.deepcopy(samples)
    for arm in bad.values():del arm['records'][0]['snapshots']['graphB-after']
    reject(lambda:c.compare_pair(bad));faults.append('missing_both_Graph_stages')
    tools=Path(__file__).resolve().parent;cli=0
    for name,flags in [('consumer',[['--gpu'],['--gpu','--root-owned-exclusive']]),
                       ('fullstep',[['--gpu'],['--case','4k'],['--case','16k']])]:
        script=tools/('probe_biv150_native117_n32_'+name+'.py')
        result=subprocess.run([sys.executable,'-B','-S',str(script)],capture_output=True,text=True,check=True)
        assert json.loads(result.stdout)['consumer_cases']==[[32,4,2048],[32,4,2560]]
        for argv in flags:
            result=subprocess.run([sys.executable,'-B','-S',str(script),*argv],capture_output=True,text=True)
            assert result.returncode==2;cli+=1
    if a.actual_raw:
        compile=a.actual_raw/'out/compile'
        args=SimpleNamespace(root_owned_exclusive=True,parent_source=root,
            compile_summary=compile/'compile-summary.json',assembly_review=a.actual_review,
            assembly_review_sha256=REVIEW,main8b_library=a.actual_main8b,
            library=compile/'build/cjj_native_k8_klookahead_probe.so',codec_summary=a.actual_codec,
            codec_proof_sha256=c.CODEC_SHA,out=Path('/private/tmp/cjj117-n32-cpu-no-device-created-20261029'))
        c.admission(args)
        old=copy.copy(args);old.assembly_review=a.actual_n64_review;old.assembly_review_sha256=original.REVIEW_SHA
        reject(lambda:c.admission(old));faults.append('actual_N64_static_receipt_rejected')
        # The real N64 summaries are read solely to prove the new shape gate
        # refuses them; no original N64 execution or NumPy reference rerun.
        pair=json.loads(a.actual_n64_summary.read_text())
        records={role:json.loads((a.actual_n64_summary.parent/x['summary_relative']).read_text()) for role,x in pair['arms'].items()}
        reject(lambda:c.compare_pair(records));faults.append('actual_passed_N64_records_rejected')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(completed=True,scope=c.plan()['scope_adaptation'],consumer_cases=CASES,
        fullstep_case=[32,16384,4,2560],fullstep_body_exact_to_83=True,
        original_N64_modules_and_entry_scopes_unchanged=True,negative_checks=faults,
        CLI_rejections=cli,actual_CPU_admission_checked=bool(a.actual_raw),
        reference_arrays_executed=False,GPU_executed=False,numeric_verified=False,
        performance_verified=False,precision_gate=False,service_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True)
    for key in ('actual-raw','actual-review','actual-main8b','actual-codec','actual-n64-review','actual-n64-summary'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();values=[v for k,v in vars(a).items() if k!='parent_source']
    assert all(values) or not any(values)
    print(json.dumps(check(a),indent=2,sort_keys=True))
