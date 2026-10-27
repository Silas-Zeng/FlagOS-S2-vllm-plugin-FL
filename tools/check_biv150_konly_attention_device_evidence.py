"""CPU-only actual DAC1 BF16 selection/IR/array/single-round arithmetic replay."""
import argparse
import ast
import json
import re
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bench_biv150_konly_q32_numeric as frozen
import biv150_konly_accepted_main as main
from biv150_cpu_tensor_archive import read_tensor_archive
from check_biv150_konly_q32_eager_evidence import independent_metrics
from check_biv150_konly_allreader_device_evidence import codec,sha

def run(root):
    import numpy as np
    root=Path(root);manifest=json.loads((root/'download-manifest.json').read_text());names=set()
    for row in manifest['files']:
        name=row['relative'];assert name not in names and not name.startswith('/') and '..' not in Path(name).parts;names.add(name)
        raw=(root/name).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
    assert len(names)==manifest['raw_files']==95 and sum(r['bytes'] for r in manifest['files'])==manifest['raw_bytes']
    assert manifest['read_only'] and not manifest['model_rerun'] and manifest['original_owner']['same_start'] and manifest['original_owner']['state']=='Z'
    out=root/'out';summary=json.loads((out/'attention-perf-summary.json').read_text());material=json.loads((out/'main-materialization.json').read_text());attempt=json.loads((out/'main-materializer-attempt.json').read_text())
    assert summary['completed'] and summary['errors']==[] and summary['per_length_implementation_passed'] and summary['timing_attempted']
    assert summary['accepted_main_commit']==main.MAIN and not summary['original_numeric_passed'] and not summary['precision_gate']
    assert summary['writer_ms'] is None and summary['fullcost_ms'] is None and not summary['performance_verified'] and not summary['service_verified'] and not summary['model105_verified']
    transition=summary['admission']['original_transition'];assert transition['state']=='completed' and transition['errors']==[] and transition['root_identity']['frontend_commit']=='e2563331e520204e6385d904ae69cd93c6f3bc31'
    assert all(p['native_host_pid']==transition['native_host_pid'] and p['token_sha256']==transition['token_sha256'] for p in summary['admission']['actual_phase_checks'])
    assert material['actual_parent_selected'] and material['actual_BF16_Q_hoist_verified'] and material['canonical_B256_never_fed_to_SDK_raw_B128']
    assert material['accepted_main_root']['commit']==main.MAIN and material['accepted_main_root']['attention_source_pins']==main.PINS
    assert all(v['installed'] is True for v in material['installations'].values())
    selected=material['manager_optimized_entry'];record=summary['actual_main_record'];assert selected['mode']=='selected' and selected['candidate_hash']==material['actual_control_hash']==record['hash']
    snapshot=attempt['manager_snapshot'];assert snapshot['selected_calls']==2 and snapshot['fallback_calls']==0 and snapshot['selected_compiles']==1 and snapshot['fallback_compiles']==0 and attempt['actual_call_count']==2 and attempt['parent_chain_intact']
    assert any(r['mode']=='selected' and r['candidate_hash']==record['hash'] for r in snapshot['compile_records'])
    override=(out/'actual-main-q-hoist-override.ttgir').read_bytes();assert sha(override)==selected['override_sha256']
    source=(frozen.parent.ROOT/'vllm_fl/patches/biv150_q_fragment_service.py').read_text();assert sha(source.encode())==main.PINS['biv150_q_fragment_service.py']
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='lowered_proof')
    ns={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned BF16 proof only>','exec'),ns)
    raw=(out/'main-normalized-raw.py.txt').read_text();assert raw==frozen.parent.factory.canonical(frozen.parent.factory.inputs(),'q32')==(out/'expected-normalized-raw.py.txt').read_text()
    qlines=set()
    for text in (raw,frozen.parent.factory.inputs()['sdk.py']):
        for i,line in enumerate(text.splitlines(),1):
            if 'query_ptr + query_offset' in line: qlines.update((i-1,i,i+1))
    proof=ns['lowered_proof']((out/'ir/accepted_main/actual.ttgir').read_text(),(out/'ir/accepted_main/actual.llir').read_text(),qlines,256)
    assert proof==material['actual_lowered_proof']==selected['mechanism'] and proof['mechanism_pass']
    old,previous,_=frozen.actual.old.original();hoisted,hs,_=frozen.actual.original();binding={}
    for role,label in [('raw_bf16','baseline'),('unhoisted','candidate'),('hoisted','candidate')]:
        origin,prior=(hoisted,hs) if role=='hoisted' else (old,previous)
        r=json.loads((out/'ir'/role/'record.json').read_text());expected=next(x for x in prior['records'] if x['label']==label)
        actual={e:(out/'ir'/role/('actual.'+e)).read_bytes() for e in ('source','ttir','ttgir','llir','asm','cubin')}
        before={e:origin['out/ir/'+label+'/actual.'+e] for e in actual};audit=frozen.verify_actual_compilation(r,expected,'baseline' if role=='raw_bf16' else role,actual,before)
        assert audit==json.loads((out/'ir'/role/'actual-compilation-binding.json').read_text());binding[role]=dict(hash=r['hash'],actual_cubin_sha256=audit['actual_cubin_sha256'])
    patterns=[]
    for row in summary['records']:
        p=summary['actual_geometry'];path=out/row['artifact']['path'];blob=path.read_bytes();assert len(blob)==row['artifact']['bytes'] and sha(blob)==row['artifact']['sha256'];data=read_tensor_archive(blob)
        output={k:v.array(np) for k,v in data['outputs'].items()};assert set(output)=={'raw_bf16','unhoisted','hoisted','main'}
        assert output['main'].tobytes()==output['raw_bf16'].tobytes() and output['unhoisted'].tobytes()==output['hoisted'].tobytes()
        assert all(a.shape==(2048,16,128) and ((a&0x7f80)!=0x7f80).all() for a in output.values())
        codes,scales=codec(data['original_CPU']['packed'].array(np)[:,0].copy(),np)
        assert codes.tobytes()==data['expected_codes'].array(np).tobytes() and scales.tobytes()==data['expected_scales'].array(np).tobytes()
        encoded=np.full((p['pages'],16384),0xa5,np.uint8);encoded[:,:4096]=codes.view(np.uint8).reshape(p['pages'],4096);encoded[:,4096:4224]=scales.view(np.uint8).reshape(p['pages'],128)
        encoded[:,8192:]=data['original_CPU']['packed'].array(np)[:,1].copy().view(np.uint8).reshape(p['pages'],8192)
        assert encoded.tobytes()==data['expected_raw'].array(np).tobytes()==data['actual_raw'].array(np).tobytes()
        metrics={};rows=data['reference']['rows']
        for name,role,ref in [('baseline','main','original'),('quantized_operand','hoisted','quantized'),('original_BF16','hoisted','original')]:
            actual=(output[role][rows].astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(-1).tolist();expected=data['reference'][ref].values(np).reshape(-1).tolist()
            metrics[name]=independent_metrics(actual,expected);assert metrics[name]==row['numeric_checks'][name]
        assert all(v is True for v in row['hard_checks'].values())
        patterns.append(dict(pattern=row['pattern'],full_elements=4194304,main_raw_BF16_bitwise=True,I8_hoisted_unhoisted_bitwise=True,full_CPU_codec_recomputed=True,full_encoded_bytes_equal=True,metrics=metrics))
    timings=summary['attention_only_diagnostic'];assert [x['role'] for x in timings]==['main','hoisted','hoisted','main'] and [x['order'] for x in timings]==list(range(4)) and all(x['replays']==64 and x['event_ms_per_replay']>0 for x in timings)
    bf16=(timings[0]['event_ms_per_replay']+timings[3]['event_ms_per_replay'])/2;i8=(timings[1]['event_ms_per_replay']+timings[2]['event_ms_per_replay'])/2
    assert bf16/i8==summary['main_over_I8_attention_only_ratio'];cache=summary['fixture_cache'];assert cache['fixture_replayed_without_flush'] and not cache['HBM_or_service_extrapolation_allowed'] and cache['logical_payload_is_not_measured_traffic']
    return dict(raw_files=95,raw_bytes=manifest['raw_bytes'],owner=manifest['original_owner'],all_SHA_verified=True,actual_main_hash=record['hash'],actual_main_lowered_proof=proof,
        actual_control_selected_replayed=True,actual_private_bindings=binding,geometry=summary['actual_geometry'],arrays=patterns,actual_postlaunch_resources=summary['actual_postlaunch_resources'],
        timing_scope=summary['timing_scope'],timings=timings,main_mean_ms=bf16,I8_mean_ms=i8,single_warm_ABBA_ratio=bf16/i8,fixture_cache=cache,
        precision_gate=False,original_numeric_passed=False,writer_ms=None,fullcost_ms=None,performance_verified=False,service_verified=False,
        limits='One fixed-fixture ABBA64 round, no stability/service/HBM claim; Graph/canary/read-only checks are recorded device checks, separate Graph-mode arrays not saved in this perf artifact; metrics use saved independent FP64 references')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',required=True,type=Path);args=parser.parse_args()
    print(json.dumps(run(args.raw),indent=2,sort_keys=True))
