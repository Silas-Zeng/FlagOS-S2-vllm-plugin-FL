"""CPU-only lossless archive and recorded 4K result replay. PT stays opaque."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

import biv150_v8_bias_byte_stage as contract
from biv150_native_pid_admission import QUERY

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/evidence/v8-bias-decode-20261027/numeric-steady4k-v2'
VALIDATOR='11ed97dc0800f195bb98688a43ae670cc1ab569b'
VALIDATOR_ROOT='/workspace/experiments/minicpm-s2/code/cjj-v8-bias-byte-stage-v2-20261027'
TAG='cjj-v8-bias-byte-stage-v2-20261027'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def unpack():
    manifest=json.loads((DATA/'archive-manifest.json').read_text())
    assert manifest['format']=='deduplicated-byte-chunks-v1' and manifest['validator_commit']==VALIDATOR
    chunks={}
    for item in manifest['archives']:
        path=DATA/item['path'];raw=path.read_bytes()
        assert len(raw)==item['bytes']<2*1024*1024 and sha(raw)==item['sha256']
        with tarfile.open(path) as archive:
            members=archive.getmembers();assert [x.name for x in members]==item['members']
            for member in members:
                assert member.isfile() and member.name not in chunks
                chunks[member.name]=archive.extractfile(member).read()
    files={};used=set()
    for item in manifest['files']:
        path=PurePosixPath(item['path']);assert not path.is_absolute() and '..' not in path.parts
        pieces=[]
        for part in item['chunks']:
            raw=chunks[part['member']]
            assert len(raw)==part['bytes'] and sha(raw)==part['sha256']
            pieces.append(raw);used.add(part['member'])
        raw=b''.join(pieces)
        assert item['path'] not in files and len(raw)==item['bytes'] and sha(raw)==item['sha256']
        files[item['path']]=raw
    assert used==set(chunks)
    return files,manifest


def replay(files,manifest):
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    read=lambda name:json.loads(files[name])
    main,aux=read('raw/download-manifest.json'),read('aux/download-manifest.json')
    assert main['original_owner']==dict(pid=623978,start_ticks=342792903,same_start=True,state='Z')
    assert main['raw_files']==len(main['files'])==33 and main['raw_bytes']==sum(x['bytes'] for x in main['files'])==18111262
    assert aux['original_owner']==623978 and len(aux['files'])==2
    for label,download in [('raw',main),('aux',aux)]:
        wanted={label+'/download-manifest.json'}
        for entry in download['files']:
            path=label+'/'+entry.get('relative',entry.get('name'));wanted.add(path);raw=files[path]
            assert len(raw)==entry['bytes'] and sha(raw)==entry['sha256']
        assert {p for p in files if p.startswith(label+'/')}==wanted
    assert len(files)==37
    owner=read('raw/exports/'+TAG+'-owner.json')
    assert owner['pid']==623978 and owner['start_ticks']==342792903 and owner['source_commit']==VALIDATOR
    binding=read('raw/out/byte-stage-runtime-binding.json')
    sidecar_raw=files['aux/'+TAG+'-sidecar.json'];sidecar=json.loads(sidecar_raw)
    admission=read('aux/'+TAG+'.admission.json')
    assert sha(sidecar_raw)==binding['sidecar_sha256'] and sidecar==binding['sidecar']
    assert admission==binding['admission_receipt'] and binding['admission_receipt_path'].endswith('/'+TAG+'.admission.json')
    identities=sidecar['binding']
    assert identities==admission['root_identity']
    assert identities['validator_commit']==VALIDATOR and identities['validator_root']==VALIDATOR_ROOT
    assert identities['original_source_commit']==contract.SOURCE and identities['original_source_root']==str(contract.OLD_ROOT)
    assert identities['original_output_root']==str(contract.OLD_OUT) and VALIDATOR_ROOT!=str(contract.OLD_ROOT)
    assert identities['original_source_pins']==contract.source_pins(ROOT)
    assert set(identities['validator_files'])=={'biv150_v8_bias_byte_stage.py','bench_biv150_v8_bias_byte_stage.py','biv150_native_pid_admission.py'}
    for name,wanted in identities['validator_files'].items():assert sha((ROOT/'tools'/name).read_bytes())==wanted
    # Replay all60 old compiled files and exact v1-false/v2-static-true contract.
    proof,static=contract.archive_audit()
    assert sidecar['report']==static and sidecar['original_proof_sha256']==contract.PROOF_SHA
    assert proof['completed'] is False and proof['mechanism_passed'] is False and proof['errors']
    assert binding['original_v1_completed'] is False and binding['original_v1_mechanism_passed'] is False
    assert admission['state']=='completed' and admission['errors']==[] and admission['query']==QUERY
    assert admission['container_identity']==dict(pid=623978,start_ticks=342792903)
    assert admission['native_host_pid']==1410043 and admission['native_host_pid']!=admission['container_identity']['pid']
    assert [q['pids'] for q in admission['queries']]==[[],[1410043],[1410043],[1410043]]
    assert [q['state'] for q in admission['queries']]==['new','initializing','active','active']
    assert admission['phases']==[dict(case='steady4k',variant=v,native_pids=[1410043]) for v in ('old','A')]
    assert admission['tiny_initialization_attempted'] and admission['cuda_initialization_executed'] and admission['cuda_work_may_have_started']
    assert admission['initialization_outside_numeric_and_timing'] and admission['consumer_run_completed'] and admission['consumer_run_error'] is None
    summary=read('raw/out/run-summary.json')
    assert summary['completed'] and summary['errors']==[] and summary['source_commit']==contract.SOURCE
    assert summary['compile_proof_sha256']==contract.PROOF_SHA
    assert [(x['case'],x['variant']) for x in summary['phases']]==[('steady4k','old'),('steady4k','A')]
    assert summary['adaptation']['old_v8_numeric_phase_untimed'] and summary['adaptation']['A_equivalence_required_before_timing']
    assert summary['adaptation']['hardware_guard_ast_inverse'] and summary['adaptation']['native_pid_phase_admission']
    equality=read('raw/out/A-vs-old-bitwise.json')
    assert len(equality)==2 and [x['pattern'] for x in equality]==[0,1]
    assert all(x['full_cpu_inputs_bitwise'] and x['scratch_max_sum_output_bitwise'] and x['precision_gate'] is False for x in equality)
    compiled=proof['cases'][0]['kernels'];cases={};runtime_files=0;pointwise=[]
    for variant,role in [('old','gqa_old'),('A','gqa_A')]:
        prefix='raw/out/steady4k-'+variant+'/'
        result=read(prefix+'result.json')
        assert result['completed'] and result['errors']==[] and result['numeric_passed'] and result['hard_checks_passed']
        assert result['source_commit']==contract.SOURCE and result['operator_sha256']==proof['sources'][variant]['sha256']
        assert result['service_installed'] is False and result['model_accuracy_tested'] is False
        assert len(result['cases'])==1;case=result['cases'][0];cases[variant]=case
        assert case['name']=='steady4k' and case['batch']==64 and case['length']==4096 and case['segments']==2
        assert case['completed'] and case['numeric_passed'] and case['hard_checks_passed']
        assert case['hard_checks']==dict.fromkeys(('finite','baseline_numeric','codec','slot_mapping','unmapped_bytes','graph_eager'),True)
        records=[x for x in case['numeric'] if 'cpu' in x]
        assert len(records)==4 and {(x['pattern'],x['quantized']) for x in records}=={(p,q) for p in (0,1) for q in (False,True)}
        for item in records:
            assert item['cpu']['finite'] and item['cpu']['passed'] and item['cpu']['tol_005_violations']==0
            assert item['changed_graph_equal'] and item['selected_rows']==[0,1,63]
            for label in ('graph_writer','eager_writer'):assert item[label]['target_writer_checked'] and item[label]['unmapped_bytes_unchanged']
            if item['quantized']:assert len(item['independent_bf16_consumption_gate'])==3
            pointwise.append(dict(variant=variant,pattern=item['pattern'],quantized=item['quantized'],cpu=item['cpu']))
        full_rows=[x for x in case['numeric'] if 'all_rows_bf16_consumption_gate' in x]
        assert len(full_rows)==2 and {x['pattern'] for x in full_rows}=={0,1}
        assert all(len(x['all_rows_bf16_consumption_gate'])==64 for x in full_rows)
        for item in case['numeric']:
            for label in ('all_rows_bf16_consumption_gate','independent_bf16_consumption_gate'):
                assert all(x['passed'] for x in item.get(label,[]))
        assert case['initial_codec']['passed'] and case['last_codec']['passed']
        for key,label in [('baseline_resources','gqa_main'),('v8_bf16p_resources',role)]:
            actual,expected=case[key],compiled[label]
            assert actual['hash']==expected['hash'] and actual['n_regs']==expected['n_regs'] and actual['n_spills']==expected['n_spills']
            for field in ('shared','num_warps','num_stages','use_sme'):assert actual[field]==expected['metadata'][field]
            for kind,wanted in actual['asm_sha256'].items():
                raw=files[prefix+'compiler-ir/'+actual['hash']+'/actual.'+kind]
                assert sha(raw)==wanted==expected['files'][kind]['sha256'];runtime_files+=1
        assert case['original_cache_bytes']==case['candidate_allocated_bytes']==268435456
        for obj in (result,case):assert obj['precision_gate'] is False and obj['capacity_reclaimed'] is False
        assert result['official_score'] is False and case.get('attention_only_ms') is None
    assert runtime_files==16
    assert cases['old']['timing_performed'] is False and cases['old']['rounds']==[] and cases['old'].get('speedup_median') is None
    assert cases['A']['timing_performed'] and len(cases['A']['rounds'])==1
    timing=cases['A']['rounds'][0];assert timing['index']==0 and timing['order']==['main_bf16','frozen_v8','bias_A']
    times=timing['milliseconds'];main_ratio=times['main_bf16']/times['bias_A'];old_ratio=times['frozen_v8']/times['bias_A']
    assert main_ratio==timing['speedup']==cases['A']['speedup_median']==0.6565636648753795
    assert old_ratio==timing['old_v8_over_A']==0.9086976941304727
    # PT is never deserialized. Corresponding complete files are identical bytes.
    for name in ('steady4k-cpu-inputs.pt','steady4k-pattern0-numeric-evidence.pt','steady4k-pattern1-numeric-evidence.pt'):
        assert files['raw/out/steady4k-old/'+name]==files['raw/out/steady4k-A/'+name]
    for obj in (summary,binding,sidecar,admission):assert obj['precision_gate'] is False and obj['official_score'] is False
    assert summary['service_tested'] is False and summary['capacity_reclaimed'] is False
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(tested_validator_commit=VALIDATOR,original_operator_source_commit=contract.SOURCE,
        original_proof_sha256=contract.PROOF_SHA,raw_files=33,raw_bytes=18111262,auxiliary_originals=2,
        preserved_files_including_manifests=len(files),archive_parts=len(manifest['archives']),all_original_bytes_verified=True,
        old_compile_files_replayed=60,v1_completed=False,v1_mechanism_passed=False,
        recorded_admission_state='completed',container_pid=623978,start_ticks=342792903,native_host_pid=1410043,
        native_transition=[[],[1410043],[1410043],[1410043]],recorded_numeric_and_hard_passed=True,
        recorded_two_pattern_four_field_bitwise=True,corresponding_opaque_pt_bytes_equal=True,runtime_ir_files_matched=16,
        old_numeric_phase_untimed=True,full_cost_rounds=1,order=timing['order'],milliseconds=times,
        main_bf16_over_A=main_ratio,old_v8_over_A=old_ratio,recorded_pointwise_checks=pointwise,
        pt_deserialized=False,numeric_oracle_recomputed=False,service_tested=False,model105_tested=False,
        steady16k_tested=False,ttft_tested=False,capacity_reclaimed=False,precision_gate=False,official_score=False,
        local_gpu_executed=False,local_jit_executed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract',type=Path)
    args=parser.parse_args();files,manifest=unpack();report=replay(files,manifest)
    if args.extract:
        args.extract.mkdir(parents=True,exist_ok=False)
        for name,raw in files.items():
            path=args.extract/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    print(json.dumps(report,indent=2))
