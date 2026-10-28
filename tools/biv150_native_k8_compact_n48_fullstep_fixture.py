"""Bounded N48 compact pair data; existing counter, codec and FP64 math."""
import random
import subprocess

import biv150_native_k8_fullstep_fixture as parent
import biv150_native_k8_compact_n48_factory as native

ROOT=parent.ROOT
PAGES=44769;PAGE=12544;TABLE_COLS=8192;GUARD=parent.GUARD
SOURCE_PITCH=2560;CASES={'valid41':(48,17408,4,2560),'valid42':(48,16896,4,2560)}
LIVE={'valid41':41,'valid42':42};ORDER=('tl','native','native','tl')
WARM=parent.WARM;REPLAYS=parent.REPLAYS;UNROLL=parent.UNROLL
LIBRARY_SHA='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'
sha=parent.sha;metrics=parent.metrics

def build_initial():return parent.build_initial(pages=PAGES)


def budget(live,base):
    if type(live) is not int or type(base) is not int or live<=0 or base<0:raise ValueError('integer live/base required')
    reserved=(base+3+15)//16
    if live*reserved>PAGES:raise ValueError('44769-page budget exceeded')
    if (live,base) not in ((41,17408),(42,16896)):
        raise ValueError('only the two page-budgeted N48 cases')
    return dict(rows=48,live_rows=live,padding_rows=48-live,base_length=base,
      phase_maximum_lengths=[base+1,base+2,base+3],pages_per_live_row=reserved,
      reserved_pages=live*reserved,spare_pages=PAGES-live*reserved,physical_pages=PAGES,
      independent_changed_snapshots=True,continuous_generation_claim=False)


def host_plan(case,phase):
    if case not in CASES or type(phase) is not int or phase not in (0,1,2):raise ValueError('case/phase')
    n,base,s,pitch=CASES[case];live=LIVE[case];b=budget(live,base)
    ids=list(range(PAGES));random.Random(20261028).shuffle(ids)
    width=b['pages_per_live_row'];mapping=[[-1]*TABLE_COLS for _ in range(n)]
    lengths=[base+phase+1]*live+[0]*(n-live)
    # All live rows stay nonempty. Padding rows exercise the legal early return.
    if phase:
        for row,value in enumerate((1,15,16,17,31,32,33,63,64,65,base-1)):
            lengths[(row+phase-1)%live]=value
    for row in range(live):
        own=ids[row*width:(row+1)*width]
        own=own[phase:]+own[:phase]
        mapping[row][:width]=own
    slots=parent.append_slots(mapping,lengths)
    all_ids=[p for row in mapping[:live] for p in row[:width]]
    assert len(all_ids)==len(set(all_ids))==b['reserved_pages']<=PAGES
    assert all(p>=0 for p in all_ids) and max(all_ids)<PAGES
    assert all(row==[-1]*TABLE_COLS for row in mapping[live:])
    assert slots[live:]==[-1]*(n-live)
    used=sum((value+15)//16 for value in lengths)
    assert used<=b['reserved_pages']
    return dict(**b,phase=phase,query_pitch=pitch,source_pitch=SOURCE_PITCH,
      lengths=lengths,table=mapping,slots=slots,actual_active_pages=used,
      append_rows=live,masked_append_rows=n-live)


def control(n,base,pitch,phase):
    import numpy as np
    case=next(k for k,v in CASES.items() if v==(n,base,4,pitch));h=host_plan(case,phase)
    q=np.full((n,pitch),0x4255,dtype='<u2')
    nk=np.full((n,SOURCE_PITCH),0x4255,dtype='<u2');nv=nk.copy()
    for row in range(n):
        for head in range(16):q[row,head*128:(head+1)*128]=parent.vector(0,phase,row,head)
        for head in range(2):
            nk[row,head*128:(head+1)*128]=parent.vector(1,phase+4,0x100000+row,head)
            nv[row,head*128:(head+1)*128]=parent.vector(2,phase+7,0x200000+row,head)
    return dict(query=q,table=np.array(h['table'],dtype='<i4'),lengths=np.array(h['lengths'],dtype='<i4'),
      slots=np.array(h['slots'],dtype='<i8'),new_k=nk,new_v=nv)


def logical(control_):
    result=dict(control_)
    for name in ('new_k','new_v'):result[name]=control_[name][:,:256].reshape(48,2,128)
    return result


def patched_chunk(arm,start,initial,control_):
    assert arm in ('tl','native')
    return parent.patched_chunk('k8',start,initial,logical(control_))


def golden(cold,packed,control_,segments,arm):
    assert arm in ('tl','native') and segments==4
    return parent.golden(cold,packed,logical(control_),segments,'k8')


def pins():
    # This pair has no BF16 native arm. Do not pretend that current main's
    # overlapping BF16 CUDA is the old b3/aace source required by parent.pins().
    files=('tools/biv150_native_k8_fullstep_fixture.py','tools/biv150_native_k8_consumer_fixture.py',
      'tools/biv150_native_k8_codec_fixture.py','tools/biv150_konly_decode_quant_reference.py',
      'tools/biv150_native_k8_pagewise_reference.py','tools/biv150_konly_strided_writer_contract.py',
      'vllm_fl/patches/biv150_gqa_decode.py')
    result={}
    for name in files:
        expected=subprocess.check_output(['git','-C',str(ROOT),'show','3ebdbebf61e0e90735cd8ad83eb1605627b6ef46:'+name])
        assert (ROOT/name).read_bytes()==expected,name;result[name]=sha(expected)
    native.verify();return result


def plan():
    return dict(component='N48 same-shape compact TL vs native2ad full step; not service',
      page_bytes=PAGE,pages=PAGES,arena_bytes_per_arm=PAGES*PAGE,table_columns=TABLE_COLS,
      cases={k:budget(LIVE[k],v[1]) for k,v in CASES.items()},query_pitch=2560,new_KV_pitch=SOURCE_PITCH,
      readonly_Q_K_V_backings_separate=True,shared_model_QKV_allocation_not_covered=True,
      arms=['production compact TL48 w1/s1','actual2ad N48/S4'],ABBA_order=list(ORDER),
      complete_step=['same exact dc5 stride2560 append','producer','same original stride-aware S4 reducer'],
      initial_encode_and_source_pack_costs_separate=True,CPU_counter_and_byte_oracle_separate=True,
      numerical_thresholds=dict(pointwise_atol=.005,pointwise_rtol=.005,RRMSE=.005),thresholds_changed=False,
      reference='unchanged full128D I64 quantized-QK / FP64 ideal-final and page16 BF16-P partial',
      ideal_partial_diagnostic_separate=True,scratch_builder_width=16,
      phases=['A base+1','changed Graph B at most base+2 before new eager B','side C at most base+3'],
      independent_prefix_snapshots=True,continuous_generation_claim=False,
      warm_replays=WARM,timing_replays=REPLAYS,unroll=UNROLL,
      timing_clocks=['GPU event per append+producer+reducer','Graph launch+sync host wall per step'],
      library_sha256=LIBRARY_SHA,consumer_summary_sha256=None,actual_future_consumer_and_all_artifacts_required=True,
      source_pins=pins(),twoN32_is_not_performance_control=True,
      default_CPU=True,GPU_executed=False,numerical_verified=False,timing_executed=False,
      service_verified=False,model_verified=False,capacity_claim=False,precision_gate=False)
