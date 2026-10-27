#!/usr/bin/env python3
"""Reassemble all482 immutable actual reader files+5 manifests; no Torch load."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'docs/evidence/konly-prefill-reader-20261027/device-originals-v1'


def sha(data): return hashlib.sha256(data).hexdigest()


def original(dataset):
    index=json.loads((DIRECTORY/'index.json').read_text());row=next(x for x in index['datasets'] if x['dataset']==dataset)
    directory=DIRECTORY/dataset;raw=(directory/'provenance.json').read_bytes();assert sha(raw)==row['provenance_sha256'];p=json.loads(raw)
    assert p['omitted_files']==[] and p['all_original_bytes_preserved'] and (p['raw_files'],p['raw_bytes'])==(row['raw_files'],row['raw_bytes'])
    parts={}
    for i,record in enumerate(p['archives']):
        raw=(directory/record['file']).read_bytes();assert len(raw)==record['bytes']<2*1024**2 and sha(raw)==record['sha256']
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            members=archive.getmembers();assert len(members)==record['members']
            for m in members:
                assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts and (i,m.name) not in parts
                parts[i,m.name]=archive.extractfile(m).read()
    files={};used=set()
    for record in p['files']:
        name=record['relative'];assert name not in files and not name.startswith('/') and '..' not in Path(name).parts
        blocks=[]
        for part in record['parts']:
            key=part['archive'],part['member'];assert key not in used;used.add(key);data=parts[key]
            assert len(data)==part['bytes'] and sha(data)==part['sha256'];blocks.append(data)
        data=b''.join(blocks);assert len(data)==record['bytes'] and sha(data)==record['sha256'];files[name]=data
    assert used==set(parts) and len(files)==p['raw_files']+1 and sha(files['download-manifest.json'])==p['manifest_sha256']
    m=json.loads(files['download-manifest.json']);assert m['original_owner']==p['original_owner'] and m['read_only'] and not m['model_rerun']
    assert set(files)=={r['relative'] for r in m['files']}|{'download-manifest.json'}
    for r in m['files']: assert (len(files[r['relative']]),sha(files[r['relative']]))==(r['bytes'],r['sha256'])
    return files,p


def write(files,target):
    target.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)


def run(extract=None,arrays=False):
    index=json.loads((DIRECTORY/'index.json').read_text());assert (index['raw_files'],index['raw_bytes'],index['manifests'],index['archives'])==(482,294086418,5,166)
    if extract: extract.mkdir(parents=True,exist_ok=False)
    result=[]
    for row in index['datasets']:
        dataset=row['dataset'];files,p=original(dataset)
        record=dict(dataset=dataset,raw_files=p['raw_files'],raw_bytes=p['raw_bytes'],all_original_SHA_verified=True,
                    owner=p['original_owner'],archives=len(p['archives']),maximum_archive_bytes=max(r['bytes'] for r in p['archives']))
        if extract:
            target=extract/dataset;write(files,target)
            if arrays: record['CPU_replay']=replay(dataset,target)
        elif arrays:
            with tempfile.TemporaryDirectory(prefix='cjj80-archive-replay-') as tmp:
                target=Path(tmp)/dataset;write(files,target);record['CPU_replay']=replay(dataset,target)
        result.append(record)
    return dict(raw_files=482,raw_bytes=294086418,manifests=5,all_originals_preserved=True,datasets=result,
                no_GPU_or_JIT=True,precision_gate=False,performance_verified=False,service_verified=False,capacity_reclaimed=False)


def replay(dataset,target):
    review=ROOT/'docs/evidence/konly-prefill-reader-20261027/device-review-cpu-v1'
    if dataset.startswith('attention'):
        import check_biv150_konly_attention_device_evidence as audit
        actual=audit.run(target)
    else:
        import check_biv150_konly_allreader_device_evidence as audit
        actual=audit.run(target,dataset)
    expected=json.loads((review/(dataset+'-cpu-replay.json')).read_text());assert actual==expected
    return dict(full_arrays_codec_metrics_IR_resource_replay_equal=True,original_numeric_gate_unchanged=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract',type=Path);parser.add_argument('--arrays',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.extract,args.arrays),indent=2,sort_keys=True))
