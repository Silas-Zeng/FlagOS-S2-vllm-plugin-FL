#!/usr/bin/env python3
"""Package immutable downloaded mixed-reader evidence; CPU/Git/files only."""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '4a3fc7773b37e745082d0d98f8d53713130384ea'
LIMIT = 2 * 1024 * 1024
CHUNK = 1024 * 1024
OWNERS = {'mixed4k': (636444, 345080447, 24505683), 'mixed16k': (637563, 345185303, 24254297)}


def digest(data): return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def source_files():
    def git(*args): return subprocess.check_output(['git', '-C', str(ROOT), *args])
    queue = git('ls-tree', '-r', '--name-only', SOURCE, '--', 'vllm_fl', 'csrc').decode().splitlines()
    queue += ['tools/bench_biv150_konly_mixed_numeric.py', 'tools/kernels/biv150_konly_decode_support.py']
    available = set(git('ls-tree', '-r', '--name-only', SOURCE, '--', 'tools').decode().splitlines())
    result = {}
    while queue:
        name = queue.pop()
        if name in result: continue
        data = git('show', SOURCE + ':' + name); result[name] = data
        if name.startswith('tools/') and name.endswith('.py'):
            for node in ast.walk(ast.parse(data)):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                           [v.name for v in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    candidate = 'tools/' + module.replace('.', '/') + '.py'
                    if candidate in available and candidate not in result: queue.append(candidate)
    return result


def tar_bytes(group):
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0, compresslevel=6) as z:
        with tarfile.open(fileobj=z, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name, data in sorted(group.items()):
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644; info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def run(raw, out):
    original = (raw / 'download-manifest.json').read_bytes(); manifest = json.loads(original)
    result = json.loads((raw / 'out/result.json').read_bytes()); case = result['case']
    pid, ticks, size = OWNERS[case]
    assert manifest['raw_files'] == len(manifest['files']) == 86
    assert manifest['raw_bytes'] == sum(v['bytes'] for v in manifest['files']) == size
    assert manifest['original_owner'] == dict(pid=pid, start_ticks=ticks, same_start=True, state='Z')
    assert result['source_commit'] == SOURCE and result['completed'] and not result['errors']
    assert result['implementation_hard_gates_passed'] and not result['passed'] and not result['original_bf16_proxy_passed']
    payload = {'raw/download-manifest.json': original}; expected = {'download-manifest.json'}; omitted = []
    for item in manifest['files']:
        name = item['relative']; path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and name not in expected
        expected.add(name); disk = raw / name; assert disk.is_file() and not disk.is_symlink()
        data = disk.read_bytes(); assert digest(data) == {k: item[k] for k in ('bytes', 'sha256')}
        if '__pycache__' in path.parts:
            omitted.append(dict(relative=name, **digest(data), reason='Opaque generated pycache omitted; original raw directory retained'))
        else: payload['raw/' + name] = data
    assert {str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()} == expected
    assert len(omitted) == 2
    sources = source_files(); payload.update({'source/' + k: v for k, v in sources.items()})
    # Large NPZ originals are split byte-for-byte, never rewritten or recompressed
    # as new witnesses. Every chunk, reconstructed file and original SHA is checked.
    physical = {}; logical = []
    for name, data in sorted(payload.items()):
        chunks = []
        for offset in range(0, len(data), CHUNK):
            member = ('chunks/' + name + '.chunk%04d' % len(chunks)) if len(data) > CHUNK else name
            piece = data[offset:offset + CHUNK]; assert member not in physical; physical[member] = piece
            chunks.append(dict(member=member, **digest(piece)))
        if not data: physical[name] = b''; chunks.append(dict(member=name, **digest(b'')))
        logical.append(dict(path=name, **digest(data), chunks=chunks))
    out.mkdir(parents=True, exist_ok=False); parts = []; pending = {}; estimate = 0
    def emit(group):
        data = tar_bytes(group)
        if len(data) > LIMIT:
            names = sorted(group); assert len(names) > 1; mid = len(names) // 2
            emit({n: group[n] for n in names[:mid]}); emit({n: group[n] for n in names[mid:]}); return
        filename = case + '-device-v1-part%02d.tar.gz' % (len(parts) + 1)
        (out / filename).write_bytes(data); parts.append(dict(file=filename, members=len(group), **digest(data)))
    for name, data in sorted(physical.items()):
        cost = len(gzip.compress(data, mtime=0)) + 1024
        if pending and estimate + cost > 1800 * 1024:
            emit(pending); pending = {}; estimate = 0
        pending[name] = data; estimate += cost
    if pending: emit(pending)
    index = dict(schema='cjj-mixed-reader-device-evidence-v1', case=case, source_commit=SOURCE,
        original_download_manifest=digest(original), original_owner=manifest['original_owner'],
        original_raw_files=86, original_raw_bytes=size, archived_raw_files=86 - len(omitted), omitted_files=omitted,
        raw_directory_retained=True, source_origin='git4a complete production tree plus recursive numeric tool dependencies',
        source_files=[dict(path=k, **digest(v)) for k, v in sorted(sources.items())],
        archives=parts, physical_members=len(physical), logical_files=logical,
        gpu_executed_for_archiving=False, numerical_thresholds_changed=False, source_runtime_changed=False,
        precision_gate=False, raw_top_level_passed=False, service_verified=False, performance_measured=False,
        cold_cache_archived=False, scratch_graph_replay_arrays_archived=False,
        scope='Actual private mixed decoder implementation only; original BF16 proxy fails. Saved full outputs/FP64 witnesses, no complete reference regeneration or independent partial/Graph array replay.')
    (out / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    return dict(case=case, original_raw_files=86, raw_shas_verified=86, omitted_pycache=len(omitted),
        source_files=len(sources), parts=len(parts), archive_bytes=sum(p['bytes'] for p in parts),
        max_part=max(p['bytes'] for p in parts), gpu_executed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--raw', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(run(a.raw.resolve(), a.out.resolve()), indent=2))
