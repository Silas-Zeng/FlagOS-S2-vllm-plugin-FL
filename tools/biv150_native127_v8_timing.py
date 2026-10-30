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
# Actual result identities/schema will be pinned only after Root's full review.
NUMERIC=dict(summary_sha256=None,manifest_sha256=None,Root_review_sha256=None,
             raw_files=None,raw_bytes=None)
UNROLL=8
WARM=128
REPLAYS=64
ORDER=('control','packed_v8','packed_v8','control')
FIELDS=('partial','maximum','summation','output')


def ready():
    for key in ('summary_sha256','manifest_sha256','Root_review_sha256'):
        assert type(NUMERIC[key]) is str and re.fullmatch('[0-9a-f]{64}',NUMERIC[key]),'actual independently reviewed numeric originals still pending'
    for key in ('raw_files','raw_bytes'):assert type(NUMERIC[key]) is int and NUMERIC[key]>0,'actual numeric scope pending'


def evidence(a):
    ready() # Always before reading evidence/importing a framework.
    assert a.root_timing_approved is True,'separate Root timing/serial decision required'
    assert a.numeric_review_sha256==NUMERIC['Root_review_sha256'] and c.sha(a.numeric_review)==NUMERIC['Root_review_sha256']
    assert c.sha(a.numeric_summary)==NUMERIC['summary_sha256']
    actual=json.loads(a.numeric_summary.read_text())
    assert actual['completed'] is True and actual['errors']==[] and actual['plan']==c.plan()
    assert actual['numeric_verified'] is actual['full_OML_final_bitwise'] is True
    assert actual['Graph_verified'] is actual['side_stream_verified'] is True
    assert all(actual[k] is False for k in ('performance_verified','precision_gate','model_verified','service_verified','capacity_recovered'))
    # The exact independently replayed Root report is the authority for complete
    # arrays/FP64/codec, not those summary booleans alone. Its real schema is not
    # guessed here while all pins remainNone; the pinning commit must review it.
    return dict(parent_source=PARENT,numeric_summary_sha256=NUMERIC['summary_sha256'],
      numeric_manifest_sha256=NUMERIC['manifest_sha256'],Root_numeric_review_sha256=NUMERIC['Root_review_sha256'],
      raw_files=NUMERIC['raw_files'],raw_bytes=NUMERIC['raw_bytes'],
      original_BF16_proxy_passed=actual['original_BF16_proxy_passed'],side_inherited_not_rerun=True)


def clock_source(parent):
    text=c.previous.function(parent,'probe_biv150_native_k8_fullstep','run')
    tree=ast.parse(text);node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='timed')
    return ast.get_source_segment(text,node)


def plan():
    return dict(component='conditional producer+original reducer short component; no writer',
      numeric_source=PARENT,numeric_evidence=NUMERIC,static_review_sha256=c.STATIC_REVIEW_SHA,
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
    return dict(graph=graph,reference=fresh,check=check,record=record,result=result)


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
        after=state['check']('post-timing-block'+str(len(record['samples'])),1)
        assert after==state['reference'],'complete post-block OML/cache/readonly/guards differ from freshB'
        item['full_snapshot_sha256']={key:c.factory.sha(raw) for key,raw in after.items()}
        record['samples'].append(item);state['checkpoint']()
    for clock in ('gpu_ms_per_step','host_wall_ms_per_step'):
        means={role:sum(x[clock] for x in record['samples'] if x['role']==role)/2 for role in states}
        record[clock+'_means']=means;record[clock+'_control_over_packedV8']=means['control']/means['packed_v8']
    return record
