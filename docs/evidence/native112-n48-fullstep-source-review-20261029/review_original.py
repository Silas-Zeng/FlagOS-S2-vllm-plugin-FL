#!/usr/bin/env python3
"""Finite CPU-only source/budget review; no consumer array replay or device work."""
import hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path('/private/tmp/flagos-cjj-compact-native-n48-fullstep-20261029')
FROZEN='f5695ab58394b4a46da76beaa539afb625e7b46f'
OUT=Path('/private/tmp/cjj-agent-native112-fullstep-source-review-20261029.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert not OUT.exists();sys.dont_write_bytecode=True
manifest=json.loads((ROOT/'docs/evidence/compact-native-n48-fullstep-cpu-20261029/manifest.json').read_text())
for row in manifest['source_files']:
    raw=(ROOT/row['path']).read_bytes();assert sha(raw)==row['sha256']
    assert raw==subprocess.check_output(['git','show',FROZEN+':'+row['path']],cwd=ROOT)
for row in manifest['files']:
    raw=(ROOT/row['path']).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
sys.path.insert(0,str(ROOT/'tools'))
import probe_biv150_native_k8_compact_n48_fullstep as p
text,proof=p.source.adapted_source();old=p.compact.adapted_source()[0]
unchanged={}
for name in ('timed','capture','reducer','poison','fingerprints','verify_inputs','step'):
    before=p.source.nested(old,name);after=p.source.nested(text,name)
    assert before==after;unchanged[name]=sha(after.encode())
assert sha(text.encode())=='3678f41e6df0e6e11fa52efa2c4174648928534c30da89806a4f4870b90a1d6f'
assert 'a.bf16_library' not in text and 'torch.ops.cjj_native_k8_qk_probe' not in text
assert "gold[arm,p]=f.golden(cold,packed,controls[p],s,arm) if arm=='tl' else gold['tl',p]" in text
assert "return parent.golden(cold,packed,logical(control_),segments,'k8')" in (ROOT/'tools/biv150_native_k8_compact_n48_fullstep_fixture.py').read_text()
reader=p.source.production_bytes();assert sha(reader)==p.source.READER_SHA
assert b'Q, query_scale = quantize_bf16_rne(Q)' in reader
assert reader.count(b'score_integer = tl.dot(Q, K, tl.zeros([ROWS, PAGE_SIZE], dtype=tl.int32), out_dtype=tl.int32)')==2
assert reader.count(b'page * 3136 + 1024 + column_slot * 2 + kv_head_idx')==2
assert 'result[\'actual_TL_reader_and_quant_helper\']=verify_loaded_reader(reader)' in text
assert "arenas[arm]['raw'].view(torch.bfloat16),nk.stride(0),nv.stride(0),nk.shape[0],nv.shape[0],slots.shape[0],f.PAGES" in text
assert "storage_offset=(f.GUARD+4352)//2" in text
assert text.index("graphs[arm].replay()")<text.index("second=capture(arm,p,'new_eager_or_repeat'")
assert p.f.ORDER==('tl','native','native','tl') and (p.f.WARM,p.f.REPLAYS,p.f.UNROLL)==(128,64,8)
rows=[]
for case in p.f.CASES:
    live=p.f.LIVE[case];base=p.f.CASES[case][1];width=math.ceil((base+3)/16)
    for phase in range(3):
        h=p.f.host_plan(case,phase);seen=set()
        for row,length in enumerate(h['lengths']):
            if row>=live:
                assert length==0 and h['slots'][row]==-1 and set(h['table'][row])=={-1};continue
            assert 0<length<=base+phase+1
            pages=h['table'][row][:width]
            assert len(pages)==len(set(pages))==width and not seen.intersection(pages);seen.update(pages)
            assert h['slots'][row]==pages[(length-1)//16]*16+(length-1)%16
            assert 0<=h['slots'][row]<44769*16
        assert len(seen)==live*width and max(seen)<44769
        rows.append(dict(case=case,phase=phase,rows=48,live=live,padding=48-live,base=base,
            reserved=live*width,spare=44769-live*width,active_pages=sum(math.ceil(x/16) for x in h['lengths'])))
assert not {'numpy','torch','triton'}&set(sys.modules)
report=dict(passed=True,blocking_findings=[],frozen_commit=FROZEN,source_files=manifest['source_files'],expanded_source_sha256=sha(text.encode()),
    review_script_sha256=sha(Path(__file__).read_bytes()),seven_unchanged_functions=unchanged,independent_phase_budgets=rows,
    source_findings=dict(same_compact12544_TL48_vs_actual2ad=True,dc5_writer_and_views_materialized_exact=True,actual_stride2560_Q_K_V_separate_backing=True,
        writer_V_full_base_consumer_V_plus4352=True,both_golden_original_I8_QK_FP64=True,reference_reader_fc25_full_file_unchanged=True,
        normal_quant_helper_import_SHA_object_identity_and_actual_paths_recorded=True,zero_padding_and_negative_slots=True,
        ABBA_warm128_replays64_unroll8_preserved=True,append_producer_reducer_all_inside_timed_graph=True,
        initial_encode_and_pack_costs_separate=True,whole44769_page_cache_checks_before_after=True,original_numeric_thresholds_Graphfirst_side_guards_preserved=True,
        actual_TL_and_stride2560_writer_JIT_identity_ABI_resources_recorded_before_timing=True,consumer_c079_actual_report_gate_bound=True),
    qualifications=dict(code_ready_for_Root_serial_device_validation=True,device_executed=False,consumer_array_replay_repeated=False,fullstep_numeric_passed=False,
        fullstep_timing_passed=False,model_passed=False,service_passed=False,capacity42layers_passed=False),
    limits=['Code review and CPU geometry only; actual TL/writer compilation, large-context arrays and timing remain device work.',
        'Separate stride2560 Q/K/V buffers do not establish the shared model QKV allocation path.',
        'A/B/C use independently defined prefixes, not a claim of three consecutive generated tokens.',
        'Same-shape TL/native component denominator is not latest-main model throughput.',
        'Source helper filename recording was added during review; final frozen code includes the change.'])
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(report=str(OUT),sha256=sha(OUT.read_bytes()),passed=True,frozen=FROZEN)))
