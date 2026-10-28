#!/usr/bin/env python3
"""Narrow frontend identity/argument tests; no reference or device rerun."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import probe_biv150_native_k8_v_overlap_consumer as probe


def must_reject(call):
    try:call()
    except (AssertionError,KeyError,TypeError,ValueError):return
    raise AssertionError('Unknown/incomplete review accepted')


def check(parent):
    source,adapter=probe.adapted_run(parent)
    assert len(ast.parse(source).body)==1 and ast.parse(source).body[0].name=='run'
    template=probe.review_template(dict(compile_summary_sha256='0'*64,library_sha256='1'*64,
                                        namespace=probe.factory.NAMESPACE,source_proof=probe.factory.verify()))
    must_reject(lambda:probe.review_fields(template))
    # Synthetic fields exercise validation only; never written as a device receipt.
    sample=copy.deepcopy(template)
    for flag in probe.REVIEW_FLAGS:sample[flag]=True
    sample['notes']='Synthetic CPU validation input, not an actual device review.'
    sample['resources']={str(s):dict(vgpr=100,sgpr=68,shared_bytes=8192,spill_bytes=0) for s in (2,4)}
    probe.review_fields(sample);faults=[]
    for flag in probe.REVIEW_FLAGS:
        bad=copy.deepcopy(sample);bad[flag]=False
        must_reject(lambda bad=bad:probe.review_fields(bad));faults.append(flag)
        bad=copy.deepcopy(sample);del bad[flag]
        must_reject(lambda bad=bad:probe.review_fields(bad));faults.append('missing_'+flag)
    bad=copy.deepcopy(sample)
    del bad['actual_V_wait_after_QK_and_P_softmax_packing']
    bad['actual_V_wait_after_QK_softmax']=True
    must_reject(lambda:probe.review_fields(bad));faults.append('ambiguous_legacy_wait_field')
    for seg in ('2','4'):
        for field in ('vgpr','sgpr','shared_bytes','spill_bytes'):
            bad=copy.deepcopy(sample);bad['resources'][seg][field]=None
            must_reject(lambda bad=bad:probe.review_fields(bad));faults.append(seg+'_unknown_'+field)
    for key,value in (('spill_bytes',4),('shared_bytes',4096),('vgpr',0),('sgpr',True)):
        bad=copy.deepcopy(sample);bad['resources']['2'][key]=value
        must_reject(lambda bad=bad:probe.review_fields(bad));faults.append('invalid_'+key)
    bad=copy.deepcopy(sample);del bad['resources']['4']
    must_reject(lambda:probe.review_fields(bad));faults.append('missing_S4')
    script=Path(probe.__file__).resolve()
    default=subprocess.run([sys.executable,'-B',str(script)],capture_output=True,text=True,check=True)
    plan=json.loads(default.stdout)
    assert plan['GPU_executed'] is False and plan['precision_gate'] is False
    for argv in (['--gpu'],['--gpu','--root-owned-exclusive'],['--review-template'],['--parent-source',str(parent)]):
        got=subprocess.run([sys.executable,'-B',str(script),*argv],capture_output=True,text=True)
        assert got.returncode==2 and 'error:' in got.stderr
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(parent_commit=probe.PARENT,actual_parent_modules_SHA_verified=True,
                adapter=adapter,fixtures_rewritten=False,reference_arrays_executed=False,
                unknown_review_template_rejected=True,review_field_rejections=faults,
                incomplete_CLI_rejections=4,compile_or_GPU_executed=False,
                resource_values_only_synthetic_CPU_fixture=True,precision_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True)
    a=p.parse_args();print(json.dumps(check(a.parent_source),sort_keys=True,indent=2))
