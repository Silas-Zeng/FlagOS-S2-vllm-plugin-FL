#!/usr/bin/env python3
"""Default: verify the shared archive. --run explicitly replays the CPU audit."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    files = json.loads((here/'shared-files.json').read_text())['files']
    for entry in files:
        raw = (here/entry['path']).read_bytes()
        assert len(raw) == entry['bytes'] and digest(raw) == entry['sha256'], entry['path']
    parts = json.loads((here/'shared-archive-parts.json').read_text())
    chunks = []
    for entry in parts['parts']:
        raw = (here/entry['name']).read_bytes()
        assert len(raw) == entry['bytes'] and digest(raw) == entry['sha256']
        chunks.append(raw)
    data = b''.join(chunks)
    assert len(data) == parts['archive_bytes'] and digest(data) == parts['archive_sha256']
    index = json.loads((here/'archive-members.json').read_text())
    expected = {entry['path']:entry for entry in index['members']}
    assert len(expected) == 42
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        assert len(tar.getmembers()) == len(expected)
        for member in tar.getmembers():
            assert member.isfile() and member.name in expected
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            raw = tar.extractfile(member).read()
            entry = expected[member.name]
            assert len(raw) == entry['bytes'] and digest(raw) == entry['sha256'], member.name
        if a.run:
            if a.out is None or a.out.exists():
                raise ValueError('--run requires a new --out path')
            repo = subprocess.run(['git','-C',str(here),'rev-parse','--show-toplevel'],check=True,text=True,capture_output=True).stdout.strip()
            with tempfile.TemporaryDirectory(prefix='cjj-native109-math105-cpu-') as temp:
                tar.extractall(temp, filter='data')
                spec = importlib.util.spec_from_file_location('cjj_native109_saved_cpu_auditor',here/'review_math105.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # File-path bindings only. Keep all pinned source bytes, original
                # audit functions, thresholds and actual-data checks unchanged.
                deps = here/'dependencies'
                module.SMOKE = deps/'check_cjj_native109_service_smoke_actual_20261028.py'
                module.MATH = deps/'cjj-native-k8-service-math105-root-review-20261028.py'
                module.HELPER = deps/'cjj-actual-owned-service-helper-94db-20261028.py'
                args = argparse.Namespace(raw=Path(temp)/'cjj-native109-math105-v1-raw-20261029',repo=Path(repo))
                report = module.audit(args)
                a.out.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
                print(json.dumps(dict(report=str(a.out),report_sha256=digest(a.out.read_bytes()),audit_passed=report['audit_passed']),sort_keys=True))
    print(json.dumps(dict(shared_files_verified=len(files),archive_members_verified=len(expected),
                         all_41_originals_preserved=True,archive_sha256=parts['archive_sha256'],
                         full_CPU_audit_rerun=a.run,device_execution=False),sort_keys=True))


if __name__ == '__main__':
    main()
