#!/usr/bin/env python3
"""Verify the shared small evidence package; never substitutes for array replay."""
import hashlib,io,json,tarfile
from pathlib import Path
D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    index=json.loads((D/'evidence-files.json').read_text())
    for row in index:
        p=D/row['relative'];b=p.read_bytes()
        assert p.parent==D and len(b)==row['bytes'] and sha(b)==row['sha256']
    meta=json.loads((D/'metadata-package.json').read_text());packed=(D/meta['package']).read_bytes()
    assert len(packed)==meta['bytes'] and sha(packed)==meta['sha256']
    expected={r['relative']:r for r in meta['files']};manifest=json.loads((D/'original-download-manifest.json').read_text())
    original={r['relative']:r for r in manifest['files']}
    with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as t:
        members=t.getmembers();assert len(members)==len(expected) and {x.name for x in members}==set(expected)
        for member in members:
            assert member.isfile() and '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
            row=expected[member.name];b=t.extractfile(member).read()
            assert len(b)==row['bytes'] and sha(b)==row['sha256']
            if member.name=='download-manifest.json':assert b==(D/'original-download-manifest.json').read_bytes()
            else:assert row['sha256']==original[member.name]['sha256'] and row['bytes']==original[member.name]['bytes']
    report=json.loads((D/'independent-review.json').read_text())
    assert sha((D/'independent-review.json').read_bytes())=='c07937c3d6c3ffc78a5c7b213271e4f864a9463775fe932f9f2aadbff08ef4d0'
    assert report['script_sha256']==sha((D/'replay_consumer.py').read_bytes())
    assert report['download_manifest_sha256']==sha((D/'original-download-manifest.json').read_bytes())
    assert report['completed'] and report['errors']==[] and report['admissions_replayed']
    assert report['raw_files_rehashed']==manifest['raw_files']==170
    assert report['raw_bytes_rehashed']==manifest['raw_bytes']==59569068
    assert not any(report['scope'][k] for k in ('writer','encoder','fullstep','capacity42layers','model','performance'))
    print(json.dumps(dict(shared_files_verified=len(index),metadata_members_verified=len(expected),metadata_package_sha256=meta['sha256'],array_replay_repeated=False,large_arrays_external=True),indent=2,sort_keys=True))
if __name__=='__main__':main()
