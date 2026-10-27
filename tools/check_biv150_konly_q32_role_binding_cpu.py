#!/usr/bin/env python3
"""Replay9f's original refusal and the narrowly corrected per-role binding."""
import argparse
import copy
import io
import json
from pathlib import Path
import tarfile

import bench_biv150_konly_q32_numeric as runner

DIRECTORY=runner.parent.ROOT/'docs/evidence/konly-prefill-reader-20261027/q32-eager-binding-refusal-v1'
ARCHIVE_SHA='2a57a09d2dd267a937d66709c64c491c568441d27bc1be22c570b07c176e9948'
MANIFEST_SHA='eeb8b805dd84eb7d0572f14e7383da0889651dcc44d32ba64d75ae81fe3fd0eb'


def original():
    packed=(DIRECTORY/'original-refusal.tar.gz').read_bytes()
    if runner.actual.old.sha(packed)!=ARCHIVE_SHA: raise ValueError('Original9f refusal archive changed')
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        members=archive.getmembers()
        if len(members)!=21 or len({m.name for m in members})!=21 or any(
                not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts for m in members):
            raise ValueError('Unsafe/unknown refusal membership')
        files={m.name:archive.extractfile(m).read() for m in members}
    if runner.actual.old.sha(files['download-manifest.json'])!=MANIFEST_SHA: raise ValueError('Original manifest changed')
    manifest=json.loads(files['download-manifest.json'])
    if (manifest['raw_files'],manifest['raw_bytes'],manifest['original_owner'])!=(20,1050358,
            dict(pid=627487,start_ticks=344209834,same_start=True,state='Z')):
        raise ValueError('Original refusal owner/size changed')
    if set(files)!={row['relative'] for row in manifest['files']}|{'download-manifest.json'}: raise ValueError('Refusal membership changed')
    for row in manifest['files']:
        value=files[row['relative']]
        if (len(value),runner.actual.old.sha(value))!=(row['bytes'],row['sha256']): raise ValueError('Original raw byte changed')
    return files


def rejects(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Unknown machine/ABI/IR accepted')


def run(extract=None):
    files=original();summary=json.loads(files['out/numeric-summary.json'])
    assert not summary['completed'] and summary['reader_launches']==0 and summary['encoder_launches']==0
    assert summary['errors']==[dict(type='ValueError',message='baseline: actual ttgir differs from the reviewed compile')]
    assert summary['admission']['initialization_attempted'] and summary['admission']['native_host_pid']==1759493
    d6,d6summary,_=runner.actual.old.original();bb,_,_=runner.actual.original()
    record=json.loads(files['out/ir/baseline/record.json'])
    expected=next(r for r in d6summary['records'] if r['label']=='baseline')
    observed={ext:files['out/ir/baseline/actual.'+ext] for ext in ('source','ttir','ttgir','llir','asm','cubin')}
    reviewed={ext:d6['out/ir/baseline/actual.'+ext] for ext in observed}
    assert set(runner.differences(record,expected))=={'asm'}
    proof=runner.verify_exact_machine(record,expected,'baseline',observed,reviewed)
    d6root='/workspace/experiments/minicpm-s2/cjj-konly-prefill-q32-compile-v1-20261027/sources'
    bbroot='/workspace/experiments/minicpm-s2/cjj-konly-q32-hoist-compile-v1-20261027/sources'
    path_counts={}
    for ext in ('ttir','ttgir','llir'):
        wrong=bb['out/ir/baseline/actual.'+ext].decode()
        assert wrong.replace(bbroot,d6root).encode()==observed[ext]
        path_counts[ext]=wrong.count(bbroot)
    # Unknown source, ABI/options, IR or cubin cannot use the ASM-sidecar contract.
    for ext in ('source','ttir','ttgir','llir','cubin'):
        mutant=copy.deepcopy(record);values=dict(observed);values[ext]+=b'changed'
        mutant['files'][ext]['sha256']=runner.actual.old.sha(values[ext])
        rejects(lambda mutant=mutant,values=values:runner.verify_exact_machine(mutant,expected,'baseline',values,reviewed))
    for key in ('hash','typed_ABI','actual_options'):
        mutant=copy.deepcopy(record);mutant[key]='unknown'
        rejects(lambda mutant=mutant:runner.verify_exact_machine(mutant,expected,'baseline',observed,reviewed))
    mutant=copy.deepcopy(record);values=dict(observed)
    values['asm']=values['asm'].replace(b'\tsl_barrier ',b'\tsl_unknown ',1)
    mutant['files']['asm']['sha256']=runner.actual.old.sha(values['asm'])
    rejects(lambda:runner.verify_exact_machine(mutant,expected,'baseline',values,reviewed))
    # All candidate role checks still run actual I8 grammar/SSA/loop audit.
    for role,origin,label in [('unhoisted',d6,'candidate'),('hoisted',bb,'candidate')]:
        row=json.loads(origin['out/ir/'+label+'/record.json'])
        value={ext:origin['out/ir/'+label+'/actual.'+ext] for ext in observed}
        runner.verify_exact_machine(row,row,role,value,value)
    roots=runner.preflight()[0]['role_source_roots']
    assert roots==dict(baseline=d6root,unhoisted=d6root,hoisted=bbroot)
    if extract:
        target=Path(extract);target.mkdir(parents=True,exist_ok=False)
        for name,value in files.items():
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    return dict(archive_sha256=ARCHIVE_SHA,original_raw_files=20,original_raw_bytes=1050358,
                original_refusal_preserved=True,original_precision_failure=False,reader_launches=0,encoder_launches=0,
                tiny_CUDA_initialization_was_executed=True,all_original_SHA_verified=True,
                wrong_bb_reference_path_occurrences=path_counts,correct_d6_exact_cubin=True,
                source_typed_ABI_options_unchanged=True,role_source_roots=roots,baseline_binding=proof,
                changed_query_rows_and_canary_do_not_change_observed_ABI=True,CPU_rejected_unknowns=9,
                numeric_precision_gate_unchanged=True,actual_new_numeric_run=False,precision_gate=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract');args=parser.parse_args()
    print(json.dumps(run(args.extract),indent=2,sort_keys=True))
