#!/usr/bin/env python3
"""Package already downloaded micro evidence and Git-pinned source; CPU/files only."""
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
SOURCE = '867b725c08765fd8775d77de59a466da20bfe155'
LIMIT = 2 * 1024 * 1024


def digest(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def source_files():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args])
    names = set(git('ls-tree', '-r', '--name-only', SOURCE, '--', 'vllm_fl', 'csrc').decode().splitlines())
    queue = ['tools/bench_biv150_stock_service_micro.py', 'tools/biv150_benchmark_utils.py',
             'tools/kernels/biv150_stock_service_codec_probe.py']
    available = set(git('ls-tree', '-r', '--name-only', SOURCE, '--', 'tools').decode().splitlines())
    sources = {}
    while names or queue:
        name = names.pop() if names else queue.pop()
        if name in sources:
            continue
        data = git('show', SOURCE + ':' + name)
        sources[name] = data
        if name.startswith('tools/') and name.endswith('.py'):
            for node in ast.walk(ast.parse(data)):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                           [v.name for v in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    candidate = 'tools/' + module.replace('.', '/') + '.py'
                    if candidate in available and candidate not in sources:
                        queue.append(candidate)
    return sources


def tar_bytes(members):
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0, compresslevel=6) as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644; info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def run(raw, out):
    original = (raw / 'download-manifest.json').read_bytes()
    manifest = json.loads(original)
    assert manifest['raw_files'] == len(manifest['files']) == 276
    assert manifest['raw_bytes'] == sum(v['bytes'] for v in manifest['files']) == 27171679
    assert manifest['original_owner'] == dict(pid=627804, start_ticks=344295833, same_start=True, state='Z')
    payload = {'raw/download-manifest.json': original}
    expected = {'download-manifest.json'}
    for item in manifest['files']:
        name = item['relative']; path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        disk = raw / name
        assert disk.is_file() and not disk.is_symlink()
        data = disk.read_bytes()
        assert digest(data) == {k: item[k] for k in ('bytes', 'sha256')}, name
        assert name not in expected; expected.add(name)
        payload['raw/' + name] = data
    assert {str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()} == expected
    sources = source_files()
    payload.update({'source/' + name: data for name, data in sources.items()})
    result = json.loads(payload['raw/out/result.json'])
    assert result['source_commit'] == SOURCE and result['completed'] and not result['errors']
    assert result['implementation_hard_gates_passed'] and not result['passed']
    assert not result['original_bf16_proxy_passed'] and not result['timing_attempted'] and not result['model105_passed']
    out.mkdir(parents=True, exist_ok=False)
    parts, pending, estimated = [], {}, 0
    def flush(group):
        data = tar_bytes(group)
        if len(data) > LIMIT:
            names = sorted(group); assert len(names) > 1
            midpoint = len(names) // 2
            flush({n: group[n] for n in names[:midpoint]}); flush({n: group[n] for n in names[midpoint:]})
            return
        name = 'micro-device-v1-part%02d.tar.gz' % (len(parts) + 1)
        (out / name).write_bytes(data)
        parts.append(dict(file=name, members=len(group), **digest(data)))
    for name, data in sorted(payload.items()):
        cost = len(gzip.compress(data, mtime=0)) + 1024
        if pending and estimated + cost > 1800 * 1024:
            flush(pending); pending, estimated = {}, 0
        pending[name] = data; estimated += cost
    if pending:
        flush(pending)
    index = dict(schema='cjj-stock-service-micro-evidence-v1', source_commit=SOURCE,
        original_download_manifest=digest(original), original_owner=manifest['original_owner'],
        original_raw_files=manifest['raw_files'], original_raw_bytes=manifest['raw_bytes'],
        archived_raw_files=manifest['raw_files'], omitted_files=[],
        source_origin='git show867b725:<path>; complete production tree plus recursively resolved micro tools',
        source_files=[dict(path=name, **digest(data)) for name, data in sorted(sources.items())],
        archives=parts, payload_files=len(payload),
        raw_files_executed=False, gpu_executed_for_archiving=False, model105_passed=False,
        numerical_thresholds_changed=False, raw_top_level_passed=False,
        actual_result_scope='Packaged micro implementation hard gates only; original BF16 proxy failed; no service/model/timing',
        input_bundle_archived=False,
        note='All276 downloaded raw files including compiler cache preserved. No pycache was present. Cold cache/scratch tensors were not collected; saved FP64 reference/output witnesses are not complete reference regeneration.')
    (out / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    return dict(raw_files=276, source_files=len(sources), parts=len(parts),
                total_archive_bytes=sum(p['bytes'] for p in parts), maximum_part_bytes=max(p['bytes'] for p in parts),
                all_raw_shas_verified=True, gpu_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw.resolve(), args.out.resolve()), indent=2))
