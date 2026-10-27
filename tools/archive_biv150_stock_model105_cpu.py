#!/usr/bin/env python3
"""Archive already downloaded public Math105 records, no cache/model/device work."""
import argparse
import json
from pathlib import Path
import subprocess

from archive_biv150_stock_service_micro_cpu import digest, tar_bytes, LIMIT

ROOT=Path(__file__).resolve().parents[1]
SOURCE='a48cf1c82f8665b7e70e3e70fea5e0845eeee504'
CRITICAL=['tools/run_biv150_stock_decode_105.py','tools/run_biv150_normalization_accuracy.py',
          'vllm_fl/worker/model_runner.py','vllm_fl/patches/biv150_compute_order.py','vllm_fl/compile_semantics.py',
          'vllm_fl/patches/biv150_stock_qk8_kernel.py','vllm_fl/patches/biv150_stock_qk8_codec.py','vllm_fl/patches/biv150_stock_qk8_diagnostic.py']


def run(raw,out):
    original=(raw/'download-manifest.json').read_bytes();manifest=json.loads(original)
    assert manifest['raw_files']==len(manifest['files'])==18
    assert manifest['raw_bytes']==sum(v['bytes'] for v in manifest['files'])==4325172
    assert manifest['original_owner']==dict(pid=629199,start_ticks=344628210,same_start=True,state='Z')
    assert set(manifest['excluded_cache_prefixes'])=={'triton-cache','flaggems-cache','inductor-cache'}
    payload={'raw/download-manifest.json':original}
    for item in manifest['files']:
        path=raw/item['relative'];assert path.is_file() and not path.is_symlink()
        data=path.read_bytes();assert digest(data)=={k:item[k] for k in ('bytes','sha256')}
        assert not item['relative'].endswith(('.db','.safetensors','.pt','.pth'))
        payload['raw/'+item['relative']]=data
    sources=[]
    for name in CRITICAL:
        data=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+name])
        payload['source/'+name]=data;sources.append(dict(path=name,**digest(data)))
    state=json.loads(payload['raw/out/state.json']);assert state['completed'] and state['passed'] and state['cleanup_complete'] and not state['errors']
    out.mkdir(parents=True,exist_ok=False);parts=[]
    def emit(group):
        data=tar_bytes(group)
        if len(data)>LIMIT:
            names=sorted(group);assert len(names)>1;mid=len(names)//2
            emit({n:group[n] for n in names[:mid]});emit({n:group[n] for n in names[mid:]});return
        name='model105-records-v3-part%d.tar.gz'%(len(parts)+1);(out/name).write_bytes(data)
        parts.append(dict(file=name,members=len(group),**digest(data)))
    emit(payload)
    index=dict(schema='cjj-stock-puredecode-model105-records-v3',source_commit=SOURCE,
        production_base='4432fae71deca0d6c43482332b6937084daed10e',team_base='56f1838192abc940b3fb9accef490022ac125000',
        original_download_manifest=digest(original),original_raw_files=18,original_raw_bytes=4325172,
        original_owner=manifest['original_owner'],excluded_generated_cache_prefixes=manifest['excluded_cache_prefixes'],
        records_only_not_full_out=True,original_full_out_retained_remote=True,omitted_record_files=[],
        critical_git_source_files=sources,archives=parts,payload_files=len(payload),
        public_math_benchmark_only=True,model_weights_archived=False,private_sqlite_archived=False,
        gpu_executed_for_archiving=False,precision_gate=False,encoded_cache_verified=False,
        full_prefill_precision_verified=False,latest_dac_verified=False,throughput_measured=False)
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    return dict(parts=len(parts),bytes=sum(p['bytes'] for p in parts),max_part=max(p['bytes'] for p in parts),raw_files=18,gpu_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.raw.resolve(),a.out.resolve()),indent=2))
