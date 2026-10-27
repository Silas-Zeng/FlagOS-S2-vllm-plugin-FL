#!/usr/bin/env python3
"""Admission/policy/receipt checks; optional small full-D128 CPU oracle check."""
import argparse
import contextlib
from copy import deepcopy
import importlib.abc
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import biv150_compact_q32_probe_contract as c
import probe_biv150_compact_q32_hoist as p


def rejects(fn):
    try:fn()
    except (ValueError,KeyError,OSError):return
    raise AssertionError('Bad admission accepted')


def check(oracle):
    needset={'finite','quantized_FP64_pointwise','pair_bitwise','full_output_guards','whole_cache',
             'readonly','fresh_actual_IR_native_resources','actual_path_bound'}
    good=dict.fromkeys(needset,True)
    for key in good:
        rejects(lambda:c.policy({**good,key:False},True,False))
    rejects(lambda:c.policy(good,False,False))
    assert c.policy(good,False,True)['precision_gate'] is False
    identity=p.source_identity()
    assert identity['format']==c.FORMAT
    assert c.plan()['Graph_hoisted_qualification'] is False
    for phase in range(3):
        for h in (0,1,8,15):
            bits=[c.counter_bits(1,h,d,phase,0) for d in range(128)]
            assert bits[:16]!=bits[16:32] and bits[:32]!=bits[32:64]
    with tempfile.TemporaryDirectory(prefix='compact-q32-probe-cpu-') as folder:
        root=Path(folder);artifact=root/'adapter';artifact.mkdir()
        snapshots={}
        for role in ('unhoisted','hoisted'):
            directory=artifact/role;directory.mkdir();hashes={}
            for ext in ('ttir','ttgir','llir','asm','cubin'):
                raw=('explicit CPU receipt fixture '+role+' '+ext).encode()
                (directory/('actual.'+ext)).write_bytes(raw);hashes[ext]=c.sha(raw)
            snapshots[role]=dict(artifacts=hashes)
        override=artifact/'override.ttgir';override.write_text('CPU-only, not native evidence')
        b=dict(snapshots,native=dict(actual_I8_operand_hoist_verified=True),artifact_directory=str(artifact),
               override_path=str(override),override_sha256=c.sha(override.read_bytes()))
        record=dict(completed=True,compile_passed=True,errors=[],mode='compile-only',commit='CPU-fixture',
            source_identity=identity,attention_launches=0,writer_launches=0,encoder_launches=0,initialized_tensor_elements=0,binding=b)
        def verify(value):
            p.save(root/'summary.json',value);return p.compile_receipt(root,identity,'CPU-fixture')
        assert verify(record)['fresh_numeric_reaudit_required']
        faults=[]
        for key,value in [('completed',False),('compile_passed',False),('errors',['failure']),
                          ('commit','wrong'),('source_identity',{}),('attention_launches',1),('initialized_tensor_elements',1)]:
            bad=deepcopy(record);bad[key]=value;faults.append(bad)
        bad=deepcopy(record);bad['binding']['native']['actual_I8_operand_hoist_verified']=False;faults.append(bad)
        bad=deepcopy(record);bad['binding']['unhoisted']['artifacts']={};faults.append(bad)
        for bad in faults:rejects(lambda:verify(bad))
        verify(record);(artifact/'hoisted/actual.llir').write_text('changed')
        rejects(lambda:p.compile_receipt(root,identity,'CPU-fixture'))
    for options in [[],['--gpu'],['--compile-only','--numeric-diagnostic']]:
        cmd=[sys.executable,'-B','-W','error',str(Path(p.__file__)),*options]
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
        if options:assert result.returncode!=0
        else:assert json.loads(result.stdout)['GPU_executed'] is False
    result=dict(passed=True,CPU_only=True,source_manifest_verified=True,default_plan_no_framework=True,
        hard_gate_rejections=len(good),compile_receipt_faults=len(faults)+1,missing_gpu_options_rejected=True,
        full_D128_counter_nonperiodic=True,Graph_path_label_normalized_only=True,
        partial_max_sum='not_exposed_by_2D_Q32',no_device_or_compiler_run=True,
        actual_IR_resource_numeric_Graph_service_pending=True)
    if oracle:
        import numpy as np
        original=(c.Q,c.LENGTHS)
        try:
            c.Q=2;c.LENGTHS=(5,6,7)
            data=c.canonical(0,np)
            for key,kind,heads in [('q_bits',0,16),('k_bits',1,2),('v_bits',2,2)]:
                for r in range(data[key].shape[0]):
                    for h in range(heads):
                        assert data[key][r,h].tolist()==[c.counter_bits(r,h,d,0,kind) for d in range(128)]
            _,kc,ks,_=c.packed(data,np);refs,_=c.references(data,kc,ks,np)
            qc,qs,_=c.quantize(data['q_bits'],np);values=c.decode(data['v_bits'],np)
            for row in range(2):
                visible=5-2+row+1
                for head in range(16):
                    kh=head//8
                    score=[sum(int(qc[row,head,d])*int(kc[t,kh,d]) for d in range(128))*float(qs[row,head])*float(ks[t,kh])*128**-.5 for t in range(visible)]
                    weights=[math.exp(x-max(score)) for x in score];weights=[x/sum(weights) for x in weights]
                    golden=[sum(weights[t]*float(values[t,kh,d]) for t in range(visible)) for d in range(128)]
                    assert np.max(np.abs(refs['quantized_FP64'][row,head]-golden))<1e-12
            result['small_full_D128_INT64_FP64_scalar_crosscheck']=True
        finally:c.Q,c.LENGTHS=original
    assert not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules)
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--oracle',action='store_true');args=a.parse_args()
    print(json.dumps(check(args.oracle),sort_keys=True,indent=2))
