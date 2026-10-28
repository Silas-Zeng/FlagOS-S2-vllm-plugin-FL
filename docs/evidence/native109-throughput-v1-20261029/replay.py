"""Verify the complete PR109 TP package; optional bounded CPU replay, never GPU."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def unpack_checked(archive,dest):
    """Only regular contained files/directories; no links or unsafe paths."""
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        assert len({m.name for m in members})==len(members)
        for member in members:
            relative=PurePosixPath(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts
            path=dest/relative
            if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
            assert member.isfile()
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(stream.extractfile(member).read())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--compact-so',type=Path,default=Path('/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out/build/cjj_native_k8_compact_probe.so'))
    p.add_argument('--gqa-so',type=Path,default=Path('/private/tmp/cjj-main105-native-gqa-build-v1-raw-20261028/out/corex_native_gqa_s2_probe.so'))
    a=p.parse_args()
    for name,record in read(HERE/'shared-files.json').items():
        path=HERE/name;assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],name
    for name,record in read(HERE/'existing-dependencies.json')['files'].items():
        path=REPO/name;assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],name
    index=read(HERE/'archive-members.json');archive=HERE/'originals.tar.gz'
    assert archive.stat().st_size==index['archive_bytes'] and sha(archive)==index['archive_sha256']
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();assert len(members)==37
        expected={x['path']:x for x in index['members']}
        assert len(expected)==37 and set(expected)=={x.name for x in members}
        for member in members:
            assert member.isfile()
            raw=stream.extractfile(member).read();record=expected[member.name]
            assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
    report=read(HERE/'review.json')
    assert report['review_script_sha256']==sha(HERE/'review.py') and report['audit_passed']
    if not a.run:
        print(json.dumps(dict(package_verified=True,raw_members=37,collected_files=36,collected_bytes=3077775,full_review_rerun=False,GPU=False)))
        return
    if a.out is None or a.out.exists():p.error('--run requires a new --out path')
    assert sha(a.compact_so)==report['dependencies']['compact_SO_sha256']
    assert sha(a.gqa_so)==report['dependencies']['GQA_SO_sha256']
    with tempfile.TemporaryDirectory(prefix='cjj-native109-TP-replay-') as temp:
        root=Path(temp);raw=root/'raw';ref=root/'reference';math=root/'math'
        unpack_checked(archive,raw)
        unpack_checked(REPO/'docs/evidence/native107-throughput-v1-20261028/originals.tar.gz',ref)
        unpack_checked(REPO/'docs/evidence/native109-math105-v1-20261029/cjj-native109-math105-v1-raw-20261029.tar.gz',math)
        command=[sys.executable,'-B','-W','error',str(HERE/'review.py'),'--repo',str(REPO),
                 '--raw',str(raw),'--reference-raw',str(ref/'cjj-native107-main105-throughput-v1-raw-20261028'),
                 '--math-raw',str(math/'cjj-native109-math105-v1-raw-20261029'),
                 '--launch',str(HERE/'launch-spec.json'),'--dispatcher',str(HERE/'archived-sdk-cudagraph-dispatcher-20261028.py'),
                 '--compact-so',str(a.compact_so.resolve()),'--gqa-so',str(a.gqa_so.resolve()),'--out',str(a.out.resolve())]
        subprocess.run(command,check=True,timeout=60)

if __name__=='__main__':main()
