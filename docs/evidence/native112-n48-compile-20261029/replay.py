#!/usr/bin/env python3
"""Verify this small evidence package; --run replays only the CPU SO audit."""
import argparse, hashlib, io, json, subprocess, sys, tarfile
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
def sha(b):return hashlib.sha256(b).hexdigest()
def unpack(raw,dest=None):
    files={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tf:
        for member in tf.getmembers():
            path=PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert member.name not in files
            files[member.name]=tf.extractfile(member).read()
    if dest is not None:
        assert not dest.exists();dest.mkdir(parents=True)
        for name,data in files.items():
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return files
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    index=json.loads((HERE/'files.json').read_text())
    for row in index['files']:
        b=(HERE/row['path']).read_bytes();assert (len(b),sha(b))==(row['bytes'],row['sha256'])
    archive=HERE/'compile-original-records.tar.gz';data=archive.read_bytes()
    ai=json.loads((HERE/'archive-index.json').read_text())
    assert (len(data),sha(data))==(ai['archive_bytes'],ai['archive_sha256'])
    files=unpack(data);assert set(files)=={r['path'] for r in ai['members']}
    for row in ai['members']:assert (len(files[row['path']]),sha(files[row['path']]))==(row['bytes'],row['sha256'])
    manifest=json.loads(files['download-manifest.json'])
    assert files['download-manifest.json']==(HERE/'download-manifest.json').read_bytes()
    assert len(files)==34 and (manifest['raw_files'],manifest['raw_bytes'])==(33,1412002)
    assert set(files)=={'download-manifest.json',*(r['relative'] for r in manifest['files'])}
    for row in manifest['files']:assert (len(files[row['relative']]),sha(files[row['relative']]))==(row['bytes'],row['sha256'])
    flags=('actual_machine_code_bound_to_SO','actual_compact_addresses_and_SME_descriptor_verified',
      'actual_wait_tail_QK_PV_math_verified','actual_codec_preserved','actual_K_then_V_issue_order_preserved',
      'actual_V_wait_after_QK_and_P_softmax_packing')
    template=json.loads((HERE/'review-template-unapproved.json').read_text())
    pending=json.loads((HERE/'review-pending-root.json').read_text())
    approved=json.loads((HERE/'root-approved-consumer-review.json').read_text())
    for key,path in {'compile_summary_sha256':'out/compile-summary.json','inspection_sha256':'out/actual-so-inspection.json',
      'actual_SO_IR_sha256':'out/actual-so-IR.stdout.txt','actual_SO_SASS_sha256':'out/actual-so-sass.stdout.txt',
      'library_sha256':'out/build/cjj_native_k8_compact_n48_s4_probe.so'}.items():
        assert template[key]==pending[key]==approved[key]==sha(files[path])
    assert template['source_proof']==pending['source_proof']==approved['source_proof']
    assert template['scope']==pending['scope']==approved['scope']==dict(rows=48,segments=4,
      namespace='cjj_native_k8_compact_n48_s4_probe',consumer_only=True,service_or_model_verified=False)
    assert all(template[k] is False and pending[k] is False and approved[k] is True for k in flags)
    assert pending['manual_Root_approval'] is False and approved['manual_Root_approval'] is True
    assert {k for k in set(pending)|set(approved) if pending.get(k)!=approved.get(k)}=={*flags,'notes','manual_Root_approval'}
    assert pending['independent_audit_report_sha256']==approved['independent_audit_report_sha256']==sha((HERE/'actual-so-review.json').read_bytes())
    deps={}
    for row in json.loads((HERE/'existing-dependencies.json').read_text())['dependencies']:
        b=subprocess.check_output(['git','-C',str(REPO),'show',row['commit']+':'+row['path']])
        assert (len(b),sha(b))==(row['bytes'],row['sha256']);deps[row['role']]=b
    assert not {'torch','triton','numpy'}&set(sys.modules)
    if a.run:
        assert a.out and a.out.is_absolute() and not a.out.exists()
        a.out.mkdir();unpack(data,a.out/'raw');unpack(deps.pop('reference_archive'),a.out/'reference-raw')
        paths={}
        for name,b in deps.items():
            path=a.out/name;path.write_bytes(b);paths[name]=path
        command=[sys.executable,'-B','-W','error',str(HERE/'review_actual_so.py'),
          '--raw',str(a.out/'raw'),'--reference-raw',str(a.out/'reference-raw'),
          '--helper',str(paths['helper']),'--elf-helper',str(paths['elf_helper']),
          '--reference-report',str(paths['reference_report']),'--source-root',str(REPO),
          '--report',str(a.out/'review.json'),'--template',str(HERE/'review-template-unapproved.json'),
          '--review-out',str(a.out/'review-pending-root.json')]
        subprocess.run(command,check=True)
        assert (a.out/'review.json').read_bytes()==(HERE/'actual-so-review.json').read_bytes()
        assert (a.out/'review-pending-root.json').read_bytes()==(HERE/'review-pending-root.json').read_bytes()
    else:assert a.out is None, '--out requires --run'
    assert sha((HERE/'root-approved-consumer-review.json').read_bytes())=='14c8f26fa0d7afb7d00d480aad9ec5cf84f490d11cca8da8cb5f6a959a114a50'
    print(json.dumps({'package_verified':True,'raw_files':33,'archive_members':34,'existing_dependencies_verified':4,
      'CPU_SO_audit_replayed':a.run,'original_Root_consumer_approval_preserved':True,
      'this_verifier_grants_approval':False,'GPU_or_numeric_test_run':False},sort_keys=True))
if __name__=='__main__':main()
