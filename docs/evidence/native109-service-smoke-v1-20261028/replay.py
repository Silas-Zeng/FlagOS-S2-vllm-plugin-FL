#!/usr/bin/env python3
"""Verify shared bytes by default; --run explicitly replays the CPU-only audit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    pins = json.loads((here/'shared-files.json').read_text())
    for row in pins['files']:
        data = (here/row['path']).read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
    record = json.loads((here/'archive-members.json').read_text())
    archive = here/record['archive']
    data = archive.read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        expected = {r['path']:r for r in record['members']}
        assert len(tar.getmembers()) == len(expected)
        for member in tar.getmembers():
            assert member.isfile() and member.name in expected
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            raw = tar.extractfile(member).read()
            r = expected[member.name]
            assert len(raw) == r['bytes'] and hashlib.sha256(raw).hexdigest() == r['sha256']
        if a.run:
            if a.out is None or a.out.exists():
                raise ValueError('--run requires a new --out path')
            with tempfile.TemporaryDirectory(prefix='cjj-native109-smoke-cpu-') as temp:
                root = Path(temp)
                tar.extractall(root, filter='data')
                repo = subprocess.run(['git','-C',str(here),'rev-parse','--show-toplevel'],check=True,text=True,capture_output=True).stdout.strip()
                subprocess.run([sys.executable,'-B','-W','error',str(here/'review_smoke.py'),
                    '--raw',str(root/'smoke'),'--baseline-raw',str(root/'baseline107'),
                    '--helper',str(root/'helper94db.py'),'--repo',repo,'--out',str(a.out.resolve())],check=True)
    print(json.dumps(dict(shared_files_verified=len(pins['files']),archive_members_verified=len(expected),
                         cpu_audit_run=a.run,device_execution=False,archive_sha256=record['sha256']),sort_keys=True))


if __name__ == '__main__':
    main()
