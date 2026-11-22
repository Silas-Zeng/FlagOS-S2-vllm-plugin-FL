"""Conditional original-clock Native127 consumer timing; no new numeric oracle."""
import ast
import json
import math
import re
from pathlib import Path
import sys
import time
import biv150_native127_v8_numeric as c
import biv150_native127_v8_fixture as fixture

PARENT='cd2f4ab846b79e4ab32a545d8c276bd802ed66d5'
# Root's actual complete saved-array replay; this is not final timing/slot permission.
NUMERIC=dict(summary_sha256='9ce384c64e2fe03aded001837bfe5851b833d417e711d339c4288928fce103b2',
 manifest_sha256='cfc7de650e89ec664e6341643ace0d87f0f987f4e7a2ea84cbc0c2f1fec88f66',
 Root_review_sha256='260414601892fdf8997e9978bcd3761bb77c5e9c67295961ce76663b07177a5e',
 raw_files=190,raw_bytes=175022287,review_bytes=113107)
TIMING_ADMISSION_SHA='df52edece78b5f047b6b22d46b5f81c4b4fe1fde48a233fa52ae6ade95ee340f'
UNROLL=8
WARM=128
REPLAYS=64
ORDER=('control','packed_v8','packed_v8','control')
FIELDS=('partial','maximum','summation','output')


def ready():
    for key in ('summary_sha256','manifest_sha256','Root_review_sha256'):
        assert type(NUMERIC[key]) is str and re.fullmatch('[0-9a-f]{64}',NUMERIC[key]),'actual independently reviewed numeric originals still pending'
    for key in ('raw_files','raw_bytes','review_bytes'):assert type(NUMERIC[key]) is int and NUMERIC[key]>0,'actual numeric scope pending'


def evidence(a):
    ready() # Missing/unknown actual result pins refuse before framework imports.
    assert a.root_timing_approved is True,'final reviewed-source and serial timing decision required'
    assert a.numeric_review_sha256==NUMERIC['Root_review_sha256'] and c.sha(a.numeric_review)==NUMERIC['Root_review_sha256']
    assert a.numeric_review.stat().st_size==NUMERIC['review_bytes'] and c.sha(a.numeric_summary)==NUMERIC['summary_sha256']
    actual=json.loads(a.numeric_summary.read_text());review=json.loads(a.numeric_review.read_text())
    assert actual['completed'] is True and actual['errors']==[] and actual['plan']==json.loads(json.dumps(c.plan()))
    assert actual['numeric_verified'] is actual['full_OML_final_bitwise'] is True
    assert actual['Graph_verified'] is actual['side_stream_verified'] is True
    assert all(actual[k] is False for k in ('performance_verified','precision_gate','model_verified','service_verified','capacity_recovered'))
    assert review['completed'] is True and review['errors']==[] and review['source_commit']==PARENT
    assert review['summary_sha256']==NUMERIC['summary_sha256'] and review['download_manifest_sha256']==NUMERIC['manifest_sha256']
    assert (review['raw_files'],review['raw_bytes'])==(NUMERIC['raw_files'],NUMERIC['raw_bytes'])
    for key in ('component_numeric_verified','full_OML_final_bitwise','Graph_verified','side_stream_verified','all_input_and_CPU_oracle_bytes_recomputed'):assert review[key] is True
    for key in ('performance_verified','precision_gate','model_verified','service_verified','capacity_recovered','writer_verified'):assert review[key] is False
    assert review['binding']==actual['binding'] and review['binding']['libraries']=={role:value[0] for role,value in c.ROLES.items()}
    assert review['counts']==dict(BF16_proxy=16,CPU_oracle_arrays=48,FP64_hard=88,cross_arm_pairs=80,golden_arrays=96,ideal_partial=24,ideal_partial_failed=24,readonly=160,snapshots=40)
    assert review['owner']==dict(pid=328892,start_ticks=367406093,state='Z')
    assert actual['original_BF16_proxy_passed'] is review['original_BF16_proxy_passed'] is True
    assert a.timing_admission_sha256==TIMING_ADMISSION_SHA and c.sha(a.timing_admission)==TIMING_ADMISSION_SHA
    receipt=json.loads(a.timing_admission.read_text())
    assert receipt==dict(schema='root_numeric_to_component_timing_scope_v1',source_commit=PARENT,
      development_main=c.DEVELOPMENT_BASE,actual_numeric_review_sha256=NUMERIC['Root_review_sha256'],
      actual_numeric_summary_sha256=NUMERIC['summary_sha256'],Root_read_and_replayed_numeric_auditor=True,
      approved_component_timing_only=True,requires_reviewed_timing_source_and_explicit_serial_launch=True,
      scope=dict(N=64,S=2,Q_pitches=[2048,2560],consumer_only=True,order=list(ORDER),unroll=8,warm_graph_replays=128),
      model_precision_verified=False,performance_verified=False,service_verified=False)
    return dict(parent_source=PARENT,numeric_summary_sha256=NUMERIC['summary_sha256'],
      numeric_manifest_sha256=NUMERIC['manifest_sha256'],Root_numeric_review_sha256=NUMERIC['Root_review_sha256'],
      raw_files=NUMERIC['raw_files'],raw_bytes=NUMERIC['raw_bytes'],Root_timing_scope_sha256=TIMING_ADMISSION_SHA,
      original_BF16_proxy_passed=True,original_BF16_proxy_max_RRMSE=review['original_BF16_proxy_max_RRMSE'],
      ideal_partial_diagnostic_failed=24,counts=review['counts'],side_inherited_not_rerun=True)


def clock_source(parent):
    text=c.previous.function(parent,'probe_biv150_native_k8_fullstep','run')
    tree=ast.parse(text);node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='timed')
    return ast.get_source_segment(text,node)


def plan():
    return dict(component='conditional producer+original reducer short component; no writer',
      numeric_source=PARENT,numeric_evidence=NUMERIC,Root_timing_scope_receipt_sha256=TIMING_ADMISSION_SHA,static_review_sha256=c.STATIC_REVIEW_SHA,
      roles=c.plan()['roles'],cases=fixture.CASES,page_bytes=16384,fixture_pages=389,fixture_cache_bytes=6373632,
      timed_phase='B mixed0/short/one6143/one6144; not64long requests',scratch='original consumer compactS2',
      capture_steps=UNROLL,warm_replays_per_arm=WARM,timed_replays_per_block=REPLAYS,
      complete_steps_per_sample=UNROLL*REPLAYS,order=ORDER,rounds_per_pitch=1,event_blocks_total=8,
      clocks=['original3eb CUDA event','original3eb synchronized host monotonic wall'],
      same_numeric_matrix_rerun=False,new_capture_validation='captureA8steps; changedB+poison Graph before freshB; full OML/FP64/readonly',
      side_stream_newly_executed=False,postblock='complete same-arm OML/cache/inputs/guards equal freshB',
      Root_timing_admission=False,device_admission=False,numeric_verified=False,performance_verified=False,
      precision_gate=False,model_verified=False,service_verified=False,capacity_recovered=False)


def adapted_run(parent):
    original,parent_adapter=c.adapted_run(parent)
    begin=original.index("            poison();eager('A',0)")
    end=original.index('            assert reducer_handle.src.fn is base._reduce_partials',begin)
    changes=[
      ("cases={p:f.fixture(n,s,pitch,p) for p in (0,1,2)};gold={p:f.golden(cases[p]) for p in cases}",
       "cases={p:f.fixture(n,s,pitch,p) for p in (0,1)};gold={1:f.golden(cases[1])}",1),
      ("            record['CPU_oracles']=f.cpu_oracles(cases,blob)",
       "            record['full_CPU_oracles_authority']='prior Root-reviewed cd2f numeric originals; current B inputs/gold/snapshots retained'",1),
      (original[begin:end],
       "            state=prepare_case(torch,args,record,result,producer,reducer,change,poison,check,snapshot,views,shapes,f)\n            PREPARED[args.role]=state\n",1),
      ("            record['completed']=True;write_json(folder/'record.json',record);write_json(out/'consumer-summary.json',result)",
       "            record['completed']=True;write_json(folder/'record.json',record);write_json(out/'consumer-summary.json',result)\n            def checkpoint_state():\n                write_json(folder/'record.json',record)\n                result['artifact_manifest']=[dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=f.sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='consumer-summary.json']\n                write_json(out/'consumer-summary.json',result)\n            state['checkpoint']=checkpoint_state\n",1),
      ('consumer_numeric_verified=True,Graph_verified=True,side_stream_verified=True,requested_consumer_gates_passed=True',
       'consumer_numeric_verified=False,Graph_verified=False,side_stream_verified=False,timing_capture_B_verified=True,requested_timing_capture_gates_passed=True',1)]
    value,proof=c.previous.edit(original,changes)
    proof['parent_cd2f_adapter']=parent_adapter
    return value,proof


def prepare_case(torch,args,record,result,producer,reducer,change,poison,check,snapshot,views,shapes,f):
    def step():
        producer();result['producer_calls']+=1
        reducer();result['reducer_calls']+=1
    poison()
    for _ in range(8):step() # Initial reducer compilation outside all timing.
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(UNROLL):step()
    result['capture_calls']+=1
    change(1);before=snapshot('timingGraphB-before')
    import struct
    for field,shape in shapes.items():
        width=2 if field=='output' else 4;count=1
        for size in shape:count*=size
        expected=struct.pack('<H' if width==2 else '<I',f.BF16_POISON if width==2 else f.F32_POISON)*count
        assert before[field][f.GUARD_ELEMENTS*width:-f.GUARD_ELEMENTS*width]==expected
    graph.replay();result['graph_replays']+=1;torch.cuda.synchronize()
    first=check('timingGraphB-after',1)
    poison();step();torch.cuda.synchronize();fresh=check('timingFreshB-after',1)
    assert first==fresh,'new8-step Graph B must match fresh1-step eager B, full inputs/OML/guards'
    record['timing_capture']=dict(unroll=UNROLL,Graph_B_before_fresh=True,full_parity=True,side_newly_executed=False)
    return dict(graph=graph,reference=fresh,snapshot=snapshot,record=record,result=result)


def run_pair(a,torch,states,timed,case):
    assert set(states)==set(c.ROLES)
    for field in FIELDS:assert states['control']['reference'][field]==states['packed_v8']['reference'][field],('cross-arm full bits',case,field)
    record=dict(case=case,warm={},samples=[],clock_source='original3eb timed()',unroll=UNROLL)
    for role,state in states.items():
        clocks=timed(state['graph'].replay,WARM);state['result']['graph_replays']+=WARM
        assert all(type(clocks[k]) is float and math.isfinite(clocks[k]) and clocks[k]>0 for k in ('gpu_ms','host_wall_ms'))
        record['warm'][role]=dict(clocks,graph_replays=WARM,steps=WARM*UNROLL)
    for role in ORDER:
        state=states[role];clocks=timed(state['graph'].replay,REPLAYS);state['result']['graph_replays']+=REPLAYS
        assert all(type(clocks[k]) is float and math.isfinite(clocks[k]) and clocks[k]>0 for k in ('gpu_ms','host_wall_ms'))
        item=dict(role=role,gpu_ms_per_step=clocks['gpu_ms']/UNROLL,
          host_wall_ms_per_step=clocks['host_wall_ms']/UNROLL,original_timer_per_replay=clocks,
          graph_replays=REPLAYS,complete_steps=REPLAYS*UNROLL)
        after=state['snapshot']('post-timing-block'+str(len(record['samples'])))
        assert after==state['reference'],'complete post-block OML/cache/readonly/guards differ from freshB'
        item['full_snapshot_sha256']={key:c.factory.sha(raw) for key,raw in after.items()}
        record['samples'].append(item);state['checkpoint']()
    for clock in ('gpu_ms_per_step','host_wall_ms_per_step'):
        means={role:sum(x[clock] for x in record['samples'] if x['role']==role)/2 for role in states}
        record[clock+'_means']=means;record[clock+'_control_over_packedV8']=means['control']/means['packed_v8']
    return record
