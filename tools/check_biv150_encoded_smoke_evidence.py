#!/usr/bin/env python3
"""Archive/replay root's existing three-request smoke; CPU only, never rerun it."""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE='359c3e11c9cf7551fbdda649bfb5a331b4822157'
FOLDER=ROOT/'docs/evidence/konly-encoded-service-20261027/smoke-records-v1'
NOTES=Path('/private/tmp/flagos-notes-service-followup-20261025')
CONTROL='tools/run_encoded_service_smoke_20261027.py'


def digest(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(b,d):assert digest(b)=={k:d[k] for k in ('bytes','sha256')};return b


def git_blob(path,seen=()):
    assert path not in seen
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+path])
    row=subprocess.check_output(['git','-C',str(ROOT),'ls-tree',SOURCE,'--',path]).decode().split()
    if row[0]=='120000':return git_blob(posixpath.normpath(posixpath.join(posixpath.dirname(path),raw.decode())),(*seen,path))
    return raw


def build(raw,report):
    manifest=(raw/'download-manifest.json').read_bytes();m=json.loads(manifest)
    assert (m['raw_files'],m['raw_bytes'])==(9,91327)
    files={'raw/download-manifest.json':manifest}
    for item in m['files']:
        assert '__pycache__' not in item['relative'] and not item['relative'].endswith(('.db','.safetensors','.pt','.pth'))
        files['raw/'+item['relative']]=checked((raw/item['relative']).read_bytes(),item)
    files['root/recovered-evidence.json']=report.read_bytes()
    for commit in ('0abb772','4d150f6','f695098'):
        files['root/controller-'+commit+'.py']=subprocess.check_output(['git','-C',str(NOTES),'show',commit+':'+CONTROL])
    files['root/stdlib-helper-a48.py']=subprocess.check_output(['git','-C',str(ROOT),'show','a48cf1c82f8665b7e70e3e70fea5e0845eeee504:tools/run_biv150_stock_decode_105.py'])
    FOLDER.mkdir(parents=True,exist_ok=False)
    stream=io.BytesIO()
    with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as z:
        with tarfile.open(fileobj=z,mode='w') as tar:
            for name,data in sorted(files.items()):
                info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(data))
    data=stream.getvalue();assert len(data)<=2*1024*1024
    name='encoded-smoke-records-v1.tar.gz';(FOLDER/name).write_bytes(data)
    index=dict(schema='cjj-encoded-smoke-records-v1',source_commit=SOURCE,original_raw_files=9,original_raw_bytes=91327,
        original_owner=m['original_owner'],archive=dict(file=name,**digest(data)),
        members=[dict(path=n,**digest(b)) for n,b in sorted(files.items())],
        original_failure_preserved=True,root_recovery_separate=True,GPU_rerun=False,
        excluded_generated_cache_prefixes=m['excluded_cache_prefixes'],records_only_not_full_out=True,
        remote_after_hashes_scope='Root recovered report attestation+proof SHA only; raw remote proof is not present',
        model_precision_verified=False,throughput_measured=False)
    (FOLDER/'index.json').write_text(json.dumps(index,indent=2)+'\n')


def events(log,tag):
    result=[]
    for line in log.splitlines():
        match=re.search(r'\b'+re.escape(tag)+r' (\{.*\})$',line)
        if match:
            try:value=json.loads(match[1])
            except json.JSONDecodeError:value=ast.literal_eval(match[1])
            assert type(value) is dict;result.append(json.loads(json.dumps(value)))
    return result


def run():
    index=json.loads((FOLDER/'index.json').read_text());data=checked((FOLDER/index['archive']['file']).read_bytes(),index['archive']);files={}
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
        for item in tar:
            name=PurePosixPath(item.name);assert item.isfile() and not name.is_absolute() and '..' not in name.parts and item.name not in files
            files[item.name]=tar.extractfile(item).read()
    assert set(files)=={v['path'] for v in index['members']}
    for d in index['members']:checked(files[d['path']],d)
    manifest=json.loads(files['raw/download-manifest.json'])
    assert manifest['raw_files']==len(manifest['files'])==9 and manifest['raw_bytes']==sum(v['bytes'] for v in manifest['files'])==91327
    for item in manifest['files']:checked(files['raw/'+item['relative']],item)
    actual=json.loads(files['raw/out/manifest.json']);old=json.loads(files['raw/out/state.json']);root=json.loads(files['root/recovered-evidence.json'])
    assert actual['source_commit']==root['source_commit']==index['source_commit']==SOURCE
    assert old['completed'] is False and old['smoke_passed'] is False and old['cleanup_complete']
    assert old['errors']==[dict(type='RuntimeError',message='Need both encoded writer and an encoded attention reader')]
    assert hashlib.sha256(files['root/controller-0abb772.py']).hexdigest()==actual['controller_sha256']
    assert hashlib.sha256(files['root/stdlib-helper-a48.py']).hexdigest()==actual['helper_sha256']
    assert root['recovered_smoke_passed'] and root['original_controller_failure_retained'] and root['owned_cleanup_complete']
    assert not root['GPU_rerun'] and not root['full_model_precision_verified'] and not root['throughput_measured']
    assert json.loads(files['raw/out/owned-cleanup.json'])['complete']
    for name,expected in actual['source_hashes'].items():assert hashlib.sha256(git_blob(name)).hexdigest()==expected,name
    assert actual['native_sha256']=='5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
    assert [(d['bytes'],d['sha256']) for d in actual['tuning_snapshot']]==[
        (1605632,'5764f125233211d8ed8369f0bcdc6a08a58a00c9a3470bfaa36be0af7665ca55'),
        (15712256,'2c5aa448065453188f1ff1794d8e929bacb63dcad84a600e086ae05165821314')]
    cmd=actual['command']
    for option,value in [('--port','9031'),('--gpu-memory-utilization','0.85'),('--max-model-len','131072')]:assert cmd[cmd.index(option)+1]==value
    assert json.loads(cmd[cmd.index('--compilation-config')+1])==dict(cudagraph_mode='FULL_DECODE_ONLY')
    log=files['raw/out/server.log'].decode();expected=actual['expected_compile_semantics'];fp=events(log,'BIV150_COMPILE_SEMANTICS')
    assert fp and all(v==expected for v in fp) and expected!=actual['off_compile_semantics']
    installs=events(log,'BIV150_KONLY_ENCODED_INSTALL');assert len(installs)==1 and installs[0]['installed']
    impl=expected['konly_encoded']['implementation_sha256'];assert impl==installs[0]['implementation_sha256']
    parents=root['runtime']['preserved_parent_installations']
    assert len(parents)==9 and all(events(log,k)==v and all(r['installed'] for r in v) for k,v in parents.items())
    assert parents['BIV150_SMALL_OUT_LOWERING_INSTALL'][0]['large_linear']
    selected=events(log,'BIV150_KONLY_ENCODED_SELECTED')
    assert selected==root['runtime']['selected'] and {v['profile'] for v in selected}=={'writer83','decode','q32','normalized2d128'}
    assert all(v['implementation_sha256']==impl and re.fullmatch('[0-9a-f]{64}',v['hash']) and not v['I8_fragment_hoist_verified'] for v in selected)
    requests=json.loads(files['raw/out/smoke-requests.json']);assert len(requests)==3
    usage=[]
    for request,prompt in zip(requests,(18,18,5137)):
        response=request['response'];u=response['usage']
        assert u['prompt_tokens']==prompt and u['completion_tokens']==16 and u['total_tokens']==prompt+16
        assert response['choices'][0]['finish_reason']=='length' and request['request']['max_tokens']==16
        usage.append(u)
    return dict(passed=True,source_commit=SOURCE,raw_files=9,raw_bytes=91327,source_hashes_checked=len(actual['source_hashes']),
        original_controller_failed_state_preserved=True,independent_writer83_label_recovery=True,
        immutable_root_recovered_report_checked=True,selected=selected,usage=usage,
        parent_installers_preserved=9,source_hashes_and_initial_native_tuning_descriptors_verified=True,
        remote_after_recheck='Root attestation only; raw remote-proof payload absent',
        GPU_rerun=False,host_capture_events_only=True,graph_replays_counted=False,
        model_precision_verified=False,throughput_measured=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive-raw',type=Path);p.add_argument('--root-report',type=Path)
    a=p.parse_args()
    if a.archive_raw:assert a.root_report;build(a.archive_raw,a.root_report)
    print(json.dumps(run(),indent=2))
