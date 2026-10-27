#!/usr/bin/env python3
"""Codec fixture/whole-byte guards and default CLI; no Torch/JIT/native compiler."""
import json
from pathlib import Path
import subprocess
import sys
import biv150_native_k8_codec_fixture as f


def check():
    data=f.fixtures();plan=f.plan();faults=[]
    for key in ('A','B','C'):
        ref=f.oracle(data[key]);arrays=dict(input_full=f.guarded_input(data[key]),code_full=f.guarded_codes(ref),scale_full=f.guarded_scales(ref))
        checks,_=f.full_check(data[key],ref,arrays);assert all(checks.values())
        for name,which,index in [('readonly','input_full',256),('code','code_full',128),('scale','scale_full',4),
                                 ('code_prefix','code_full',0),('code_suffix','code_full',len(arrays['code_full'])-1),
                                 ('scale_prefix','scale_full',0),('scale_suffix','scale_full',len(arrays['scale_full'])-1)]:
            bad=dict(arrays);raw=bytearray(bad[which]);raw[index]^=1;bad[which]=bytes(raw)
            checks,_=f.full_check(data[key],ref,bad);assert not all(checks.values());faults.append(key+'-'+name)
    # Eager/capture A data cannot masquerade as a fresh changed-input B replay.
    a=f.oracle(data['A']);b=f.oracle(data['B'])
    stale=dict(input_full=f.guarded_input(data['B']),code_full=f.guarded_codes(a),scale_full=f.guarded_scales(a))
    checks,_=f.full_check(data['B'],b,stale);assert checks['input_full'] and not checks['code_full'] and not checks['scale_full'];faults.append('stale_A_graph_outputs')
    root=Path(__file__).resolve().parent
    proc=subprocess.run([sys.executable,'-B','-W','error',str(root/'probe_biv150_native_k8_codec.py')],capture_output=True,text=True,check=True)
    assert json.loads(proc.stdout)==plan
    for args in (['--library','unused.so'],['--gpu']):
        bad=subprocess.run([sys.executable,'-B',str(root/'probe_biv150_native_k8_codec.py'),*args],capture_output=True,text=True)
        assert bad.returncode!=0
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(plan=plan,guard_faults_rejected=faults,CLI_default_matches=True,
      no_framework_import=True,device_executed=False,precision_gate=False)

if __name__=='__main__':print(json.dumps(check(),sort_keys=True,indent=2))
