#!/usr/bin/env python3
"""SHA replay of recorded primitive results only. No extraction/import/JIT/device work."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

from biv150_konly_i8_qk_ir import audit_artifacts

ROOT = Path(__file__).resolve().parents[1]


def run():
    evidence = ROOT/'docs/evidence/konly-i8-qk-20261027'
    provenance = json.loads((evidence/'primitive-device-provenance.json').read_text())
    kernel = (ROOT/'tools/kernels/biv150_konly_i8_qk.py').read_bytes()
    assert hashlib.sha256(kernel).hexdigest() == provenance['primitive_kernel_sha256']
    checks = []
    for entry in provenance['archives']:
        archive = (evidence/entry['file']).read_bytes()
        assert len(archive) == entry['bytes'] <= 2*1024*1024
        assert hashlib.sha256(archive).hexdigest() == entry['sha256']
        with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
            members = tar.getmembers()
            assert len({m.name for m in members}) == len(members) == entry['raw_files']+1
            assert all(m.isfile() and m.size<=2*1024*1024 for m in members)
            files = {m.name:tar.extractfile(m).read() for m in members}
        mb = files['download-manifest.json']
        assert hashlib.sha256(mb).hexdigest() == entry['manifest_sha256']
        manifest = json.loads(mb)
        assert len(manifest['files']) == entry['raw_files']
        assert set(files) == {'download-manifest.json'}|{f['relative'] for f in manifest['files']}
        for f in manifest['files']:
            assert len(files[f['relative']])==f['bytes']
            assert hashlib.sha256(files[f['relative']]).hexdigest()==f['sha256']
        assert sum(f['bytes'] for f in manifest['files']) == entry['raw_bytes']
        result = json.loads(files['out/result.json'])
        assert result['source_commit']==provenance['primitive_source_commit']
        assert result['kernel_sha256']==provenance['primitive_kernel_sha256']
        assert files['out/primitive-source.py.txt']==kernel
        assert not result['errors'] and result['scope']=='private_primitive_only'
        assert result['compiler']['resources']['hash']==provenance['actual_kernel_hash']
        prefix='out/compiler-ir/'+provenance['actual_kernel_hash']+'/actual.'
        artifacts={k:files[prefix+k].decode() for k in ('ttgir','llir','asm')}
        actual=audit_artifacts(artifacts)
        assert actual['passed'] and actual['llvm_cast_elements_per_lane']==4
        resource=result['compiler']['resources']
        numeric=entry['label']=='numeric-raw-v1'
        if numeric:
            assert result['passed'] and result['numeric_passed'] and len(result['checks'])==18
            assert resource['phase']=='post_launch' and resource['loaded_resources_known']
            assert (resource['n_regs'],resource['n_spills'],resource['metadata']['shared'])==(40,0,3072)
            for c in result['checks']:
                assert all(c[k] for k in ('integer_exact','score_fp32_bits_exact','canaries',
                           'packed_bytes_and_original_v_unchanged','dynamic_page_head_tail_metadata'))
        else:
            assert result['compile_only'] and not result['passed'] and not result['gpu_executed']
            assert resource['phase']=='pre_launch' and not resource['loaded_resources_known']
            assert resource['n_regs'] is None and resource['n_spills'] is None
        checks.append(dict(label=entry['label'],raw_files=entry['raw_files'],raw_bytes=entry['raw_bytes'],
                           actual_ir_passed=True,recorded_numeric_passed=numeric,
                           resources=resource,records=len(result['checks'])))
    root=provenance['root_replay'];data=(evidence/root['file']).read_bytes()
    assert len(data)==root['bytes'] and hashlib.sha256(data).hexdigest()==root['sha256']
    assert json.loads(data)['actual_ir']['passed']
    assert not {'torch','triton'}.intersection(sys.modules)
    return dict(cpu_replay_passed=True,executed_gpu=False,source_commit=provenance['primitive_source_commit'],
                kernel_sha256=provenance['primitive_kernel_sha256'],archives=checks,
                historical_primitive_numeric_passed=True,decode_verified=False,
                performance_verified=False,service_verified=False,capacity_reclaimed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);args=parser.parse_args()
    result=run()
    if args.out: args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
