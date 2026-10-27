#!/usr/bin/env python3
"""Verify the seven archived target sources as opaque data, never import/extract them."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def run():
    evidence = Path(__file__).resolve().parents[1] / 'docs/evidence/konly-i8-qk-20261027'
    review = json.loads((evidence / 'service-io-source-review-v1.json').read_text())
    descriptor = review['archive']
    data = (evidence / descriptor['file']).read_bytes()
    assert len(data) == descriptor['bytes'] <= 2 * 1024 * 1024
    assert hashlib.sha256(data).hexdigest() == descriptor['sha256']
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == 8 and len({m.name for m in members}) == 8
        assert all(m.isfile() and m.size < 600000 for m in members)
        files = {m.name: tar.extractfile(m).read() for m in members}
    assert hashlib.sha256(files['manifest.json']).hexdigest() == review['source_manifest_sha256']
    manifest = json.loads(files['manifest.json'])
    assert manifest['files'] == review['files'] and manifest['missing'] == review['missing']
    assert set(files) == {'manifest.json'} | {f['relative'] for f in manifest['files']}
    for item in manifest['files']:
        assert len(files[item['relative']]) == item['bytes']
        assert hashlib.sha256(files[item['relative']]).hexdigest() == item['sha256']
    assert len(manifest['files']) == review['source_files_verified'] == 7
    assert sum(f['bytes'] for f in manifest['files']) == review['source_bytes_verified'] == 569277
    assert not review['format_coverage_complete'] and not review['active_backend_and_real_tensor_strides_observed']
    assert not {'torch', 'triton', 'numpy'}.intersection(sys.modules)
    return dict(cpu_only=True, passed=True, sources=7, bytes=569277, service_io_coverage_complete=False,
                active_backend_and_real_strides_observed=False, framework_imported=False, gpu_executed=False)


if __name__ == '__main__':
    print(json.dumps(run()))
