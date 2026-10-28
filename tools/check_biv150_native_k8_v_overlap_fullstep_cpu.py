#!/usr/bin/env python3
"""Narrow CPU identity/provenance negatives; no array replay or GPU qualification."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import probe_biv150_native_k8_v_overlap_fullstep as probe
from check_biv150_native_k8_v_overlap_consumer_cpu import must_reject


def check(parent):
    adapters={}
    for kind in ('fullstep','receipt'):
        source,adapters[kind]=probe.adapted_source(parent,kind)
        assert len(ast.parse(source).body)==1
        if kind=='fullstep':
            assert source.index('pre=require_consumer')<source.index('import torch')
            assert source.index("'actual83 typed ABI'")<source.index("rec['warm']")
        else:
            assert 'expected_sha==' not in source and 'f.sha(raw)==expected_sha' in source
    parent_source=(parent/'tools/biv150_native_k8_fullstep_fixture.py').read_text()
    constants={}
    for node in ast.parse(parent_source).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:constants[node.targets[0].id]=ast.literal_eval(node.value)
            except (ValueError,TypeError):pass
    protocol={k:constants[k] for k in ('BF16_SHA','PAGES','TABLE_COLS','BUILDER_SEGMENTS','CASES','ORDER','UNROLL','WARM','REPLAYS')}
    assert protocol['PAGES']==34276 and protocol['BUILDER_SEGMENTS']==16
    assert protocol['CASES']=={'4k':(64,4096,2,2048),'16k':(32,16384,4,2560)}
    assert (protocol['ORDER'],protocol['UNROLL'],protocol['WARM'],protocol['REPLAYS'])==(('bf16','k8','k8','bf16'),8,128,64)
    sys.path.insert(0,str(parent/'tools'))
    original=probe.load_module(parent,'probe_biv150_native_k8_fullstep','cjj_v_overlap_fullstep_cpu_import_check')
    original_plan=original.f.plan()
    assert original_plan['BF16_library_sha256']==protocol['BF16_SHA']
    assert original_plan['pages']==protocol['PAGES'] and original_plan['cases']==protocol['CASES']
    binding={'library_sha256':'1'*64};adapter=probe.consumer.adapted_run(parent)[1]
    sample=dict(library_sha256='1'*64,variant=dict(compile_admission=binding,
                source_proof=probe.consumer.factory.verify(),host_adapter=adapter,qualification_inherited_from_old_reader=False))
    probe.variant_provenance(sample,binding,adapter);faults=[]
    for field in ('compile_admission','source_proof','host_adapter','qualification_inherited_from_old_reader'):
        bad=copy.deepcopy(sample);del bad['variant'][field]
        must_reject(lambda bad=bad:probe.variant_provenance(bad,binding,adapter));faults.append('missing_'+field)
        bad=copy.deepcopy(sample);bad['variant'][field]=True
        must_reject(lambda bad=bad:probe.variant_provenance(bad,binding,adapter));faults.append('wrong_'+field)
    bad=copy.deepcopy(sample);bad['library_sha256']='2'*64
    must_reject(lambda:probe.variant_provenance(bad,binding,adapter));faults.append('foreign_library')
    script=Path(probe.__file__).resolve()
    default=subprocess.run([sys.executable,'-B',str(script)],capture_output=True,text=True,check=True)
    assert json.loads(default.stdout)['GPU_executed'] is False
    for argv in (['--gpu'],['--gpu','--root-owned-exclusive'],['--variant-consumer-sha256','1'*64],['--case','4k']):
        bad=subprocess.run([sys.executable,'-B',str(script),*argv],capture_output=True,text=True)
        assert bad.returncode==2 and 'error:' in bad.stderr
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(parent_commit=probe.consumer.PARENT,adapters=adapters,unchanged_protocol=protocol,
                original_fixture_pins_and_CPU_imports_verified=True,
                provenance_rejections=faults,incomplete_CLI_rejections=4,
                consumer_receipt_or_fullstep_executed=False,new_numeric_framework=False,
                full_arena_built=False,GPU_executed=False,precision_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True)
    a=p.parse_args();print(json.dumps(check(a.parent_source),sort_keys=True,indent=2))
