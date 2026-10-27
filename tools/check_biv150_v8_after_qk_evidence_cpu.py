"""CPU-only byte archive, actual IR and recorded result replay; no PT execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile

import bench_biv150_v8_late_read as frozen

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'docs/evidence/v8-late-read-20261027/after-qk4k-v1'
SOURCE='ac6f7c2348a4e05b468cbed03635f36ba8c43917'
PINS={'tools/bench_biv150_v8_late_read.py':'d73ebcf28b3958c4a533431cc347d127e27b6593f6e65fc4ed0823c29aeae6d0',
      'tools/biv150_v8_late_read_checks.py':'4201644ca178af6650a3b07be0714d3f016c3cf3c38a09efb44356bd999a2626',
      'tools/kernels/biv150_v8_late_read.py':'b562d2cdbabf0dba9b7e495fd9b6a48b7dc70f7ca64220fe441efcbdd3cb66c8'}
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def unpack():
    manifest=json.loads((EVIDENCE/'archive-manifest.json').read_text())
    assert manifest['format']=='byte-chunks-v1' and manifest['source_commit']==SOURCE
    chunks={}
    for item in manifest['archives']:
        raw=(EVIDENCE/item['path']).read_bytes()
        assert len(raw)==item['bytes']<=2*1024*1024 and sha(raw)==item['sha256']
        with tarfile.open(EVIDENCE/item['path']) as archive:
            members=archive.getmembers()
            assert [x.name for x in members]==item['members']
            for member in members:
                assert member.isfile() and member.name not in chunks
                chunks[member.name]=archive.extractfile(member).read()
    files={};used=[]
    for item in manifest['files']:
        path=PurePosixPath(item['path']);assert not path.is_absolute() and '..' not in path.parts
        pieces=[]
        for chunk in item['chunks']:
            raw=chunks[chunk['member']]
            assert len(raw)==chunk['bytes'] and sha(raw)==chunk['sha256']
            used.append(chunk['member']);pieces.append(raw)
        raw=b''.join(pieces)
        assert item['path'] not in files and len(raw)==item['bytes'] and sha(raw)==item['sha256']
        files[item['path']]=raw
    assert len(used)==len(set(used)) and set(used)==set(chunks)
    return files,manifest


def replay(files,manifest):
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    for path,wanted in PINS.items():assert sha((ROOT/path).read_bytes())==wanted
    frozen.source_inverse()
    owners={};raw_summary={}
    read=lambda path:json.loads(files[path])
    for name,pid,start,count,size in [('compile4k',622278,341714398,21,1422848),('numeric4k',622409,342361705,17,6385974)]:
        download=read(name+'/download-manifest.json')
        assert download['original_owner']==dict(pid=pid,start_ticks=start,same_start=True,state='Z')
        assert download['raw_files']==count and download['raw_bytes']==size
        assert len(download['files'])==count and sum(x['bytes'] for x in download['files'])==size
        for item in download['files']:
            raw=files[name+'/'+item['relative']]
            assert len(raw)==item['bytes'] and sha(raw)==item['sha256']
        assert {p for p in files if p.startswith(name+'/')}=={name+'/'+x['relative'] for x in download['files']}|{name+'/download-manifest.json'}
        owner=read(next(name+'/'+x['relative'] for x in download['files'] if x['relative'].endswith('-owner.json')))
        assert owner['pid']==pid and owner['start_ticks']==start and owner['source_commit']==SOURCE
        owners[name]=owner;raw_summary[name]=dict(files=count,bytes=size,owner_snapshot=download['original_owner'])
    assert owners['numeric4k']['original_compile_owner']==owners['compile4k']
    proof=read('compile4k/out/compile-proof.json');result=read('numeric4k/out/result.json')
    contract=read('numeric4k/out/mechanism-contract.json');root=read('root-replay.json')
    proof_sha=sha(files['compile4k/out/compile-proof.json'])
    assert proof['source_commit']==result['source_commit']==contract['source_commit']==root['source_commit']==SOURCE
    assert proof['completed'] and proof['compile_only'] and proof['kernel_launches']==0 and proof['errors']==[]
    assert proof['mechanism_passed'] is False and proof['diagnostic_mechanism_passed'] is True
    assert contract['compile_proof_sha256']==root['compile_proof_sha256']==owners['numeric4k']['compile_proof_sha256']==proof_sha
    assert contract['strict_mechanism_passed'] is False and contract['diagnostic_mechanism_passed'] is True
    assert contract['selected_mechanism_contract']=='after_qk_diagnostic_v1' and contract['numeric_gate_unchanged'] is True
    assert len(proof['cases'])==1 and proof['cases'][0]['case']=='steady4k'
    row=copy.deepcopy(proof['cases'][0])
    with tempfile.TemporaryDirectory(prefix='v8-after-qk-ir-replay-') as folder:
        for label in ('bf16','frozen_v8','late_v8'):
            for kind,item in row[label]['files'].items():
                raw=files['compile4k/out/steady4k-'+label+'/actual.'+kind]
                assert sha(raw)==item['sha256'] and len(raw)==item['bytes']
                path=Path(folder)/(label+'.'+kind);path.write_bytes(raw);item['path']=str(path)
        strict=frozen.inspect_case(row);partial=frozen.inspect_after_qk_case(row)
    assert strict==row['lowering'] and partial==row['after_qk']
    assert strict['mechanism_passed'] is False and partial['diagnostic_mechanism_passed'] is True
    assert root['cases']==[dict(case='steady4k',strict_passed=False,after_qk_passed=True,late_regs=137,old_regs=143)]
    assert [row[x]['n_regs'] for x in ('bf16','frozen_v8','late_v8')]==[93,143,137]
    assert all(row[x]['n_spills']==0 for x in ('bf16','frozen_v8','late_v8'))
    assert result['completed'] and result['errors']==[] and result['numeric_passed'] and result['hard_checks_passed']
    assert result['model_accuracy_tested'] is False and result['service_installed'] is False
    assert len(result['cases'])==1;numeric=result['cases'][0]
    assert numeric['name']=='steady4k' and numeric['completed'] and numeric['numeric_passed'] and numeric['hard_checks_passed']
    assert numeric['hard_checks']==dict.fromkeys(('finite','baseline_numeric','codec','slot_mapping','unmapped_bytes','graph_eager'),True)
    records=[x for x in numeric['numeric'] if 'cpu' in x]
    assert {(x['pattern'],x['quantized']) for x in records}=={(p,q) for p in (0,1) for q in (False,True)} and len(records)==4
    for item in records:
        assert item['cpu']['finite'] and item['cpu']['passed'] and item['cpu']['tol_005_violations']==0 and item['changed_graph_equal']
        assert item['selected_rows']==[0,1,63]
        for label in ('graph_writer','eager_writer'):
            assert item[label]['target_writer_checked'] and item[label]['unmapped_bytes_unchanged']
        if item['quantized']:assert len(item['independent_bf16_consumption_gate'])==3
    full_rows=[x for x in numeric['numeric'] if 'all_rows_bf16_consumption_gate' in x]
    assert len(full_rows)==2 and {x['pattern'] for x in full_rows}=={0,1}
    assert all(len(x['all_rows_bf16_consumption_gate'])==64 for x in full_rows)
    for item in numeric['numeric']:
        for key in ('all_rows_bf16_consumption_gate','independent_bf16_consumption_gate'):
            assert all(x['passed'] for x in item.get(key,[]))
    assert numeric['initial_codec']['passed'] and numeric['last_codec']['passed']
    for name,label in [('baseline_resources','bf16'),('v8_bf16p_resources','late_v8')]:
        actual=numeric[name];expected=row[label]
        assert actual['hash']==expected['hash'] and actual['n_regs']==expected['n_regs'] and actual['n_spills']==expected['n_spills']
        for key in ('shared','num_warps','num_stages','use_sme'):assert actual[key]==expected['metadata'][key]
        for kind,wanted in actual['asm_sha256'].items():
            raw=files['numeric4k/out/compiler-ir/'+actual['hash']+'/actual.'+kind]
            assert sha(raw)==wanted==expected['files'][kind]['sha256']
    assert numeric['timing_rounds']==1 and len(numeric['rounds'])==1
    timing=numeric['rounds'][0];assert timing['order']==[False,True] and timing['index']==0
    ratio=timing['milliseconds']['baseline']/timing['milliseconds']['v8_bf16p']
    assert ratio==timing['speedup']==numeric['speedup_median']==0.6228189806905853
    assert numeric['attention_only_ms'] is None
    for obj in (proof,result,contract,root,numeric):assert obj['precision_gate'] is False
    assert all(obj['official_score'] is False for obj in (proof,result,contract,root))
    assert result['capacity_reclaimed'] is False and numeric['original_cache_bytes']==numeric['candidate_allocated_bytes']==268435456
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(source_commit=SOURCE,compile_proof_sha256=proof_sha,archives=len(manifest['archives']),
        archived_files=len(files),all_original_bytes_verified=True,raw=raw_summary,
        strict_mechanism_passed=False,after_qk_diagnostic_mechanism_passed=True,
        baseline_regs=93,frozen_v8_regs=143,late_v8_regs=137,register_delta=-6,
        recorded_numeric_passed=True,recorded_hard_checks_passed=True,
        fp64_selected_rows=[0,1,63],relative_rmse_rows=64,codec_coverage=numeric['codec_coverage'],
        full_cost_rounds=1,order='baseline-first',milliseconds=timing['milliseconds'],main_bf16_over_late=ratio,
        pointwise_records=[dict(pattern=x['pattern'],quantized=x['quantized'],cpu=x['cpu']) for x in records],
        pt_bytes_preserved_not_executed=True,numeric_oracle_recomputed=False,
        precision_gate=False,official_score=False,service_tested=False,model105_tested=False,
        capacity_reclaimed=False,jit_executed=False,gpu_executed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract',type=Path)
    args=parser.parse_args();files,manifest=unpack();summary=replay(files,manifest)
    if args.extract:
        args.extract.mkdir(parents=True,exist_ok=False)
        for name,raw in files.items():
            path=args.extract/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    print(json.dumps(summary,indent=2))
