#!/usr/bin/env python3
"""Audit existing pre/post-launch records and future unknown handling on CPU."""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

from biv150_resource_records import resource_evidence,compare_resource_records
from summarize_biv150_prefill_loaded_resources import summarize


def test_evidence_rules():
    meta=dict(num_warps=16,num_stages=1,shared=65536,use_sme=114691,warp_size=64)
    unknown=dict(hash='fixture',n_regs=None,n_spills=None,metadata=meta,asm_sha256={'asm':'fixture-sha'})
    a=copy.deepcopy(unknown);a['post_launch_resource_evidence']=resource_evidence(a,already_launched=True)
    result=compare_resource_records(a,a)
    assert not result['observed_counts_known'] and result['same_registers_and_spills'] is None
    assert not result['strict_allowed'] and not result['record_allowed'] and result['resource_changed'] is None
    loaded=dict(unknown,n_regs=113,n_spills=0)
    record=dict(candidate_hash='fixture',candidate_resources=unknown,loaded_resources=loaded)
    recovered=resource_evidence(unknown,record,already_launched=False)
    assert recovered['n_regs']==113 and recovered['n_spills']==0
    assert recovered['basis']==['manager_loaded_resources'] and recovered['matching_manager_binary_identity']
    assert resource_evidence(loaded,already_launched=False)['counts_status']=='unknown'
    left=copy.deepcopy(loaded);left['post_launch_resource_evidence']=resource_evidence(left,already_launched=True)
    right=dict(loaded,n_regs=114);right['post_launch_resource_evidence']=resource_evidence(right,already_launched=True)
    changed=compare_resource_records(left,right)
    assert changed['resource_changed'] and not changed['strict_allowed'] and changed['record_allowed']
    bad=copy.deepcopy(record);bad['loaded_resources']['hash']='other'
    try:resource_evidence(unknown,bad)
    except ValueError:pass
    else:raise AssertionError('Mismatched loaded binary accepted')
    bad=copy.deepcopy(record);bad['loaded_resources']['n_regs']=114
    try:resource_evidence(loaded,bad,already_launched=True)
    except ValueError:pass
    else:raise AssertionError('Contradictory post-launch counts accepted')
    return dict(unknown_never_equals_unknown=True,prelaunch_counts_not_postlaunch_proof=True,
        manager_loaded_counts_require_binary_identity=True,conflicting_postlaunch_evidence_rejected=True,
        strict_rejects_known_count_change=True,record_allows_known_count_change_with_explicit_attribution=True,
        driver_api_called=False,gpu_executed=False)


def audit(path):
    raw=path.read_bytes();result=json.loads(raw)
    assert result['source_commit']=='fcaec8af5e5bb674c16920134c5101a22e853e92'
    loaded={};phases=[]
    def visit(value,location):
        if isinstance(value,dict):
            if value.get('mode')=='selected' and 'loaded_resources' in value:
                before=value['candidate_resources'];after=value['loaded_resources']
                evidence=resource_evidence(before,value,already_launched=False)
                assert evidence['counts_status']=='observed_post_launch'
                phases.append(dict(location=location,kernel_hash=value['candidate_hash'],
                    prelaunch_candidate_resources={key:before.get(key) for key in ('n_regs','n_spills')},
                    postlaunch_loaded_resources={key:after.get(key) for key in ('n_regs','n_spills')},
                    metadata=after['metadata'],asm_sha256=after['asm_sha256'],
                    binary_identity_verified=evidence['matching_manager_binary_identity']))
                loaded[value['candidate_hash']]=value
                return
            for name,child in value.items():visit(child,location+'/'+name)
        elif isinstance(value,list):
            for name,child in enumerate(value):visit(child,location+'/'+str(name))
    visit(result,'result.json')
    assert len(loaded)==2
    cases=[]
    for row in result['cases']:
        snapshots={}
        for label in ('baseline','candidate'):
            snapshot=copy.deepcopy(row[label+'_resources'])
            snapshot['post_launch_resource_evidence']=resource_evidence(snapshot,loaded[snapshot['hash']],already_launched=True)
            snapshots[label]=snapshot
        comparison=compare_resource_records(snapshots['baseline'],snapshots['candidate'])
        assert comparison['strict_allowed']
        cases.append(dict(name=row['name'],baseline=snapshots['baseline']['post_launch_resource_evidence'],
            candidate=snapshots['candidate']['post_launch_resource_evidence'],comparison=comparison,
            raw_same_registers_and_spills=row['same_registers_and_spills']))
    # Reuse the unmodified main CPU summarizer on a tiny deterministic wrapper
    # of the existing JSON. No kernel source or remote directory is changed.
    buffer=io.BytesIO()
    with gzip.GzipFile(filename='',fileobj=buffer,mode='wb',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            info=tarfile.TarInfo('result.json');info.size=len(raw);info.mtime=0
            archive.addfile(info,io.BytesIO(raw))
    with tempfile.TemporaryDirectory(prefix='prefill-resource-cpu-') as temp:
        wrapped=Path(temp)/'existing-first-result-only.tar.gz';wrapped.write_bytes(buffer.getvalue())
        crosscheck=summarize(wrapped)
    assert set(crosscheck['kernels'])==set(loaded)
    for key,value in crosscheck['kernels'].items():
        assert value['resources']['loaded_n_regs']==113 and value['resources']['loaded_n_spills']==0
    return dict(cpu_only=True,gpu_executed=False,source_result=str(path),
        source_result_sha256=hashlib.sha256(raw).hexdigest(),source_commit=result['source_commit'],
        phases=phases,cases=cases,
        unmodified_main_summarizer_crosscheck=crosscheck,
        future_evidence_rule_tests=test_evidence_rules(),
        conclusion='Pre-launch null counts prove nothing. Matching post-launch records and final row snapshots establish observed SDK counts 113/0 for these two exact kernels; known launch/shared metadata matches. No complete resource equivalence, occupancy, all local-memory behavior or CSE is claimed.',
        direct_driver_function_attribute_query=False,
        unqueried_driver_attributes=['LOCAL_SIZE_BYTES','SHARED_SIZE_BYTES'],
        caveat='No pinned COREX function-attribute binding/ABI or live handle was available in this CPU audit. No handle initialization or speculative cuFuncGetAttribute call was added.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();record=audit(args.result)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(passed=True,kernels=len(record['phases']),cases=len(record['cases']),
                         observed_registers=113,observed_spills=0,gpu_executed=False)))
