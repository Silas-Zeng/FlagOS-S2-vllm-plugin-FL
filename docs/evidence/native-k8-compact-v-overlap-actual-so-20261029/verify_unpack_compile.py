#!/usr/bin/env python3
"""Verify the complete PR110 compile capsule; optionally unpack regular files only."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ARCHIVE_SHA='98863c1a9f8066658820285290f174440d59c86172e44b6800bc8e61dca78ed3'
def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,help='Optional NEW directory; existing paths are never overwritten.')
    a=p.parse_args();index_bytes=(HERE/'compile-package-index.json').read_bytes();index=json.loads(index_bytes)
    assert index['version']==1 and index['archive']['sha256']==ARCHIVE_SHA
    blob=(HERE/index['archive']['relative']).read_bytes()
    assert len(blob)==index['archive']['bytes']==354008 and sha(blob)==ARCHIVE_SHA
    expected={r['relative']:r for r in index['files']};assert len(expected)==len(index['files'])==34
    files={}
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
        for member in tar:
            path=PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert str(path)==member.name and member.name in expected and member.name not in files
            data=tar.extractfile(member).read();row=expected[member.name]
            assert (member.size,len(data),sha(data))==(row['bytes'],row['bytes'],row['sha256'])
            files[member.name]=data
    assert set(files)==set(expected) and sum(map(len,files.values()))==index['expanded_bytes']==1419289
    manifest=json.loads(files['download-manifest.json'])
    assert len(manifest['files'])==manifest['raw_files']==index['raw_files']==33
    assert manifest['raw_bytes']==index['raw_bytes']==1408237
    assert {r['relative'] for r in manifest['files']}==set(files)-{'download-manifest.json'}
    assert sum(r['bytes'] for r in manifest['files'])==1408237
    for row in manifest['files']:
        assert (len(files[row['relative']]),sha(files[row['relative']]))==(row['bytes'],row['sha256'])
    assert sha(files[index['library_relative']])==index['library_sha256']
    control=index['control_dependency'];control_blob=(REPO/control['archive_repository_relative']).read_bytes()
    assert (len(control_blob),sha(control_blob))==(control['bytes'],control['sha256'])
    if a.destination is not None:
        a.destination.mkdir(parents=True,exist_ok=False)
        for name,data in files.items():
            path=a.destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        for name,data in files.items():assert (a.destination/name).read_bytes()==data
    print(json.dumps(dict(CPU_only=True,archive_sha256=ARCHIVE_SHA,index_sha256=sha(index_bytes),
        members_verified=34,raw_files_verified=33,raw_bytes_verified=1408237,expanded_bytes_verified=1419289,
        library_sha256=index['library_sha256'],IR_sha256=sha(files['out/actual-so-IR.stdout.txt']),
        SASS_sha256=sha(files['out/actual-so-sass.stdout.txt']),control_archive_sha256=control['sha256'],
        destination=str(a.destination) if a.destination is not None else None,
        full_SO_audit_repeated=False,GPU_or_library_loading_performed=False,
        verifier_sha256=sha(Path(__file__).read_bytes())),sort_keys=True,indent=2))

if __name__=='__main__':main()
