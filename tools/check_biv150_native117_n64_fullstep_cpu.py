#!/usr/bin/env python3
"""Exact host inverse and encoded-role faults only; no reference arrays/device."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import probe_biv150_native117_n64_fullstep as p
from check_biv150_native117_n64_pair_cpu import reject

def check(root):
    source,adapter=p.adapted_source(root);tree=ast.parse(source)
    assert len(tree.body)==1 and tree.body[0].name=='run'
    assert adapter['inverse_byte_exact'] and adapter['arm_rename']['inverse_byte_exact']
    assert "torch.ops.cjj_native_k8_qk_probe" not in source and 'corex_native_gqa_s' not in source
    assert 'viewed.unbind(1)' not in source and "if arm=='main8b':" not in source
    assert '(16384,256,128,1)' in source and '(8192,256,128,1)' in source
    assert '(f.GUARD+8192)//2' in source and '(6272' not in source and '12544' not in source
    assert "(n,16,16,128)" in source and 'f.REPLAYS' in source and 'f.UNROLL' in source and 'f.WARM' in source
    assert "for writer_arm in ('main8b','lookahead'):" in source
    assert "for role in ('main8b_append','lookahead_append','reducer'):" in source
    assert "rec['cross_arm_full_OML_final_bitwise']=cross_arm(rec)" in source
    assert source.index("rec['cross_arm_full_OML_final_bitwise']")<source.index("rec['warm']")<source.index("rec['timing']=[]")
    assert 'changed_graph_before_eager' in source and 'changed_side_before_eager' in source
    calls=[]
    original=SimpleNamespace(CASES={'4k':(64,4096,2,2048),'16k':(32,16384,4,2560)},
        patched_chunk=lambda *args:calls.append(args) or args,golden=lambda *args:calls.append(args) or args)
    f=p.fixture(original);assert f.CASES=={'4k':(64,4096,2,2048)}
    assert f.ORDER==('main8b','lookahead','lookahead','main8b')
    for arm in p.c.ROLES:
        assert f.patched_chunk(arm,1,2,3)==('k8',1,2,3)
        assert f.golden(1,2,3,4,arm)==(1,2,3,4,'k8')
    reject(lambda:f.patched_chunk('bf16',1,2,3));reject(lambda:f.golden(1,2,3,4,'compact'))
    keys=((0,'initial_eager'),(0,'new_eager_or_repeat'),(1,'changed_graph_before_eager'),
          (1,'new_eager_or_repeat'),(2,'changed_side_before_eager'),(2,'new_eager_or_repeat'))
    record=dict(numeric=[dict(arm=arm,phase=phase,stage=stage,full_arrays={name:dict(sha256='0'*64,bytes=4,path='CPU-synthetic-only')
        for name in ('partial','maximum','summation','output')}) for arm in p.c.ROLES for phase,stage in keys])
    p.cross_arm(record);faults=['unknown_reference_arm','non-K8_reference_arm']
    for name in ('partial','maximum','summation','output'):
        bad=copy.deepcopy(record);bad['numeric'][-1]['full_arrays'][name]['sha256']='1'*64
        reject(lambda:p.cross_arm(bad));faults.append('fullstep_'+name+'_bit_difference')
    bad=copy.deepcopy(record);bad['numeric']=[x for x in bad['numeric'] if x['phase']!=1]
    reject(lambda:p.cross_arm(bad));faults.append('missing_both_Graph_phases')
    script=Path(p.__file__).resolve()
    default=subprocess.run([sys.executable,'-B','-S',str(script)],capture_output=True,text=True,check=True)
    assert json.loads(default.stdout)['performance_verified'] is False
    for argv in (['--gpu'],['--gpu','--root-owned-exclusive'],['--case','16k'],['--case','4k']):
        got=subprocess.run([sys.executable,'-B','-S',str(script),*argv],capture_output=True,text=True)
        assert got.returncode==2
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(completed=True,source_adapter=adapter,checks='exact inverse and 19 scoped host edits + literal arm rename',
        numerical_expressions_and_thresholds_rewritten=False,reference_arrays_executed=False,
        encoded_fixture_forwarding_both_arms_to_original_K8=True,negative_checks=faults,
        original_ABBA_and_loaded_resource_guards_retained=True,original_16KiB_layout_retained=True,
        true_width16_scratch_retained=True,case=[64,4096,2,2048],default_stdlib_only=True,
        GPU_executed=False,numeric_verified=False,performance_verified=False,precision_gate=False)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--parent-source',type=Path,required=True)
    print(json.dumps(check(a.parse_args().parent_source),indent=2,sort_keys=True))
