#!/usr/bin/env python3
"""Replay62 actual files and v3 same-IR/actual-ASM admission. No device imports."""
import argparse
import copy
import io
import json
from pathlib import Path
import re
import tarfile

import bench_biv150_konly_q32_numeric as runner

DIRECTORY=runner.parent.ROOT/'docs/evidence/konly-prefill-reader-20261027/q32-eager-backend-v2'
ARCHIVE_SHA='6f43a192438c7acc4bd98701d42fc4243b7e24f310b20633a976bfac0db4ae9c'
MANIFEST_SHA='1be1dbd11531a97482c50adf79218debd640483a12c0e87d0453db1e769e2983'


def original():
    packed=(DIRECTORY/'original-refusal.tar.gz').read_bytes()
    if runner.actual.old.sha(packed)!=ARCHIVE_SHA: raise ValueError('Original v2 archive changed')
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        members=archive.getmembers()
        if len(members)!=63 or len({m.name for m in members})!=63 or any(
                not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts for m in members):
            raise ValueError('Unsafe/unknown v2 membership')
        files={m.name:archive.extractfile(m).read() for m in members}
    if runner.actual.old.sha(files['download-manifest.json'])!=MANIFEST_SHA: raise ValueError('Download manifest changed')
    manifest=json.loads(files['download-manifest.json'])
    if (manifest['raw_files'],manifest['raw_bytes'],manifest['original_owner'])!=(62,5043823,
            dict(pid=628141,start_ticks=344328946,same_start=True,state='Z')):
        raise ValueError('Original v2 owner/count changed')
    if set(files)!={r['relative'] for r in manifest['files']}|{'download-manifest.json'}: raise ValueError('V2 membership changed')
    for row in manifest['files']:
        value=files[row['relative']]
        if (len(value),runner.actual.old.sha(value))!=(row['bytes'],row['sha256']): raise ValueError('Original raw bytes changed')
    return files


def reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Changed required IR/ABI/actual mechanism accepted')


def run(extract=None):
    files=original();summary=json.loads(files['out/numeric-summary.json'])
    assert not summary['completed'] and summary['reader_launches']==summary['encoder_launches']==0
    assert len(summary['actual_records'])==3 and len(summary['errors'])==1
    assert summary['errors'][0]['type']=='ValueError' and 'IR/cubin differs' in summary['errors'][0]['message']
    old,oldsummary,_=runner.actual.old.original();bb,bbsummary,_=runner.actual.original()
    proofs={};witnesses={}
    for role in ('baseline','unhoisted','hoisted'):
        origin,original_summary,label=(bb,bbsummary,'candidate') if role=='hoisted' else (old,oldsummary,'baseline' if role=='baseline' else 'candidate')
        expected=next(r for r in original_summary['records'] if r['label']==label)
        record=json.loads(files['out/ir/'+role+'/record.json'])
        observed={ext:files['out/ir/'+role+'/actual.'+ext] for ext in ('source','ttir','ttgir','llir','asm','cubin')}
        reviewed={ext:origin['out/ir/'+label+'/actual.'+ext] for ext in observed}
        assert set(runner.differences(record,expected))=={'asm','cubin'}
        proof=runner.verify_actual_compilation(record,expected,role,observed,reviewed)
        assert not proof['actual_cubin_exact'] and not proof['cubin_byte_equality_required']
        proofs[role]=proof;witnesses[role]=(record,expected,observed,reviewed)
    record,expected,observed,reviewed=witnesses['hoisted']
    for ext in ('source','ttir','ttgir','llir'):
        mutant=copy.deepcopy(record);values=dict(observed);values[ext]+=b'changed'
        mutant['files'][ext]['sha256']=runner.actual.old.sha(values[ext])
        reject(lambda mutant=mutant,values=values:runner.verify_actual_compilation(mutant,expected,'hoisted',values,reviewed))
    for key in ('typed_ABI','actual_options'):
        mutant=copy.deepcopy(record);mutant[key]='unknown'
        reject(lambda mutant=mutant:runner.verify_actual_compilation(mutant,expected,'hoisted',observed,reviewed))
    # Actual export SHA must bind the real cubin even though cross-run equality
    # is not required. A missing/unknown blob remains an error.
    values=dict(observed,cubin=b'not-an-ELF')
    reject(lambda:runner.verify_actual_compilation(record,expected,'hoisted',values,reviewed))
    mutant=copy.deepcopy(record);mutant['files']['cubin']['sha256']=runner.actual.old.sha(values['cubin'])
    reject(lambda:runner.verify_actual_compilation(mutant,expected,'hoisted',values,reviewed))
    # Preserve opcode sequence yet clobber invariant Q in the actual KV loop.
    rows=observed['asm'].decode().splitlines(keepends=True)
    span=proofs['hoisted']['actual_strict_audit']['ASM']
    index=next(i for i in range(span['KV_loop_body'],span['KV_exit']-1) if re.match(r'\s*ml_mov_b32 v[0-9]+,',rows[i]))
    rows[index]=re.sub(r'\bv[0-9]+(?=,)', 'v32',rows[index],count=1)
    values=dict(observed,asm=''.join(rows).encode())
    mutant=copy.deepcopy(record);mutant['files']['asm']['sha256']=runner.actual.old.sha(values['asm'])
    reject(lambda:runner.verify_actual_compilation(mutant,expected,'hoisted',values,reviewed))
    if extract:
        target=Path(extract);target.mkdir(parents=True,exist_ok=False)
        for name,value in files.items():
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    return dict(archive_sha256=ARCHIVE_SHA,original_raw_files=62,original_raw_bytes=5043823,
                all_original_SHA_verified=True,original_refusal_preserved=True,precision_failure=False,
                original_reader_launches=0,original_encoder_launches=0,proofs=proofs,
                CPU_rejected_unknowns=9,numeric_gates_unchanged=True,actual_new_numeric_run=False,precision_gate=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract');args=parser.parse_args()
    print(json.dumps(run(args.extract),indent=2,sort_keys=True))
