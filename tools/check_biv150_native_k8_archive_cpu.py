#!/usr/bin/env python3
"""Actual compile evidence replay plus bounded corruption/zero-launch faults."""
import json
from pathlib import Path
import tempfile
import audit_biv150_native_k8_compile_cpu as a


def check():
    path=a.ROOT/'docs/evidence/native-k8-qk-feasibility-20261028/compile-v1/original-compile.tar.gz'
    files=a.from_archive(path);result=a.audit(files);faults=[]
    def rejects(name,bad):
        try:a.audit(bad)
        except (AssertionError,ValueError,KeyError):faults.append(name)
        else:raise AssertionError('Fault accepted '+name)
    for name,key in [('library_byte','out/build/cjj_native_k8_qk_probe.so'),('IR_byte','out/biv150_native_k8_codec.ll')]:
        bad=dict(files);raw=bytearray(bad[key]);raw[len(raw)//2]^=1;bad[key]=bytes(raw);rejects(name,bad)
    bad=dict(files);del bad['out/build/biv150_native_k8_qk.cuda.o'];rejects('missing_object',bad)
    bad=dict(files,unexpected=b'x');rejects('unmanifested_file',bad)
    bad=dict(files);key='out/compile-summary.json';before=bad[key];bad[key]=before.replace(b'"attention_launches": 0',b'"attention_launches": 1');assert before!=bad[key]
    manifest=json.loads(bad['download-manifest.json'])
    for row in manifest['files']:
        if row['relative']==key:row['sha256']=a.sha(bad[key])
    bad['download-manifest.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();rejects('resealed_nonzero_launch',bad)
    with tempfile.TemporaryDirectory(prefix='cjj-native-k8-archive-cpu-') as scratch:
        target=Path(scratch)/'copy.tar.gz';a.archive(files,target);assert target.read_bytes()==path.read_bytes()
    return dict(audit=result,deterministic_archive_replay=True,faults_rejected=faults)

if __name__=='__main__':print(json.dumps(check(),sort_keys=True,indent=2))
