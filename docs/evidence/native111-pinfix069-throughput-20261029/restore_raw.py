#!/usr/bin/env python3
"""Verify exact byte parts; optionally restore to a new directory (CPU only)."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--extract-to', type=Path)
a = p.parse_args()
info = json.loads((ROOT / 'archive-parts.json').read_text())
chunks = []
for row in info['parts']:
    assert Path(row['path']).name == row['path']
    raw = (ROOT / row['path']).read_bytes()
    assert len(raw) == row['bytes'] <= 2 * 1024 * 1024
    assert hashlib.sha256(raw).hexdigest() == row['sha256']
    chunks.append(raw)
archive = b''.join(chunks)
assert len(archive) == info['original_archive_bytes']
assert hashlib.sha256(archive).hexdigest() == info['original_archive_sha256']
assert info['original_archive_sha256'] == '2f31c1e3875752e552ca7444d0ab1c545742e7ad22611f7459b785267d5fbdd6'
if a.extract_to:
    assert not a.extract_to.exists()
    expected_manifest = (ROOT / 'original-download-manifest.json').read_bytes()
    assert hashlib.sha256(expected_manifest).hexdigest() == info['download_manifest_sha256']
    entries = json.loads(expected_manifest)['files']
    expected = {row['relative']: row for row in entries}
    assert len(expected) == len(entries) == info['raw_files']
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len({m.name for m in members}) == info['raw_files'] + 1
        assert {m.name for m in members} == set(expected) | {'download-manifest.json'}
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        a.extract_to.mkdir(parents=True)
        for member in members:
            raw = tar.extractfile(member).read()
            if member.name == 'download-manifest.json':
                assert raw == expected_manifest
            else:
                row = expected[member.name]
                assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
            path = a.extract_to / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as out:
                out.write(raw)
print(json.dumps(dict(exact_original_archive_verified=True, bytes=len(archive),
                      sha256=info['original_archive_sha256'], extracted_to=str(a.extract_to) if a.extract_to else None,
                      original_service_completed=True)))
