"""Stdlib descriptor/fixture plans only; not a live service admission hook."""
import math
import types

import biv150_native_k8_consumer_fixture as old_fixture
import biv150_native_k8_compact_n48_factory as source

ROWS=48;SEGMENTS=4;SMALL_PAGES=29;SERVICE_PAGES=44769;PAGE_BYTES=12544
SMALL_CASES=((48,4,2048),(48,4,2560))


def need(ok,message):
    if not ok:raise ValueError(message)


def descriptor(pitch=2048,pages=SMALL_PAGES):
    return dict(rows=48,segments=4,pitch=pitch,q_shape=[48,16,128],q_strides=[pitch,128,1],
      dtype=dict(query='bf16',key='i8',scale='f32',value='bf16',lengths='i32',table='i32',partial='f32',maximum='f32',summation='f32'),
      devices=['cuda:0']*9,key_shape=[pages,16,2,128],key_strides=[12544,256,128,1],
      value_shape=[pages,16,2,128],value_strides=[6272,256,128,1],scale_numel=pages*3136,
      cache_layer_offset=128,cache_backing_bytes=128+pages*PAGE_BYTES+128,
      scale_offset_from_layer=0,value_offset_from_layer=4352,
      lengths_shape=[48],lengths_stride=[1],table_shape=[48,16],table_strides=[16,1],
      partial_shape=[64,16,4,128],partial_strides=[32768,2048,128,1],
      maximum_shape=[64,16,4],maximum_strides=[256,16,1],
      summation_shape=[64,16,4],summation_strides=[256,16,1],
      cache_role_storage_ids=[1,1,1],source_storage_ids=[2,3,4],scratch_storage_ids=[5,6,7],
      grid=[48,2,4],threads=64,scale_log2e=128**-.5*1.4426950408889634,
      controlled_CPU_lengths_table=True,live_service=False)


def validate(d):
    """CPU integer model, not execution of the real C++ TORCH_CHECKs."""
    need(type(d.get('rows')) is int and type(d.get('segments')) is int and (d['rows'],d['segments'])==(48,4),'only N48/S4')
    need(type(d.get('pitch')) is int and d['pitch'] in (2048,2560),'Q pitch')
    need(d['q_shape']==[48,16,128] and d['q_strides']==[d['pitch'],128,1],'Q shape/strides')
    need(d['dtype']==descriptor()['dtype'],'typed roles')
    need(d['devices']==['cuda:0']*9,'one device')
    k=d['key_shape'];need(len(k)==4 and type(k[0]) is int and 0<k[0]<=2147483647 and k[1:]==[16,2,128],'K physical page geometry')
    pages=k[0]
    need(d['key_strides']==[12544,256,128,1] and d['value_shape']==k and d['value_strides']==[6272,256,128,1],'compact K/V strides')
    need(type(d['scale_numel']) is int and d['scale_numel']>=pages*3136,'full scale extent')
    need(d['scale_offset_from_layer']==0 and d['value_offset_from_layer']==4352,'full scale base / already-offset V')
    need(type(d['cache_layer_offset']) is int and d['cache_layer_offset']>=0 and d['cache_layer_offset']%4==0,'layer alignment')
    need(d['cache_backing_bytes']>=d['cache_layer_offset']+pages*12544,'full cache backing extent')
    need(d['lengths_shape']==[48] and d['lengths_stride']==[1],'lengths layout')
    t=d['table_shape'];need(len(t)==2 and t[0]==48 and type(t[1]) is int and t[1]>0 and d['table_strides'][1]==1 and d['table_strides'][0]>=t[1],'table layout')
    p=d['partial_shape'];ps=d['partial_strides']
    need(len(p)==4 and p[0]>=48 and p[1:]==[16,4,128] and ps[3]==1 and ps[2]==128 and ps[1]>=4*128 and ps[0]>=16*ps[1],'partial shape/strides')
    for name in ('maximum','summation'):
        shape=d[name+'_shape'];stride=d[name+'_strides']
        need(len(shape)==3 and shape[0]>=48 and shape[1:]==[16,4] and stride[2]==1 and stride[1]>=4 and stride[0]>=16*stride[1],'scalar shape/strides')
    need(d['maximum_strides']==d['summation_strides'],'scalar common strides')
    need(d['cache_role_storage_ids']==[1,1,1] and 1 not in d['source_storage_ids'],'source/cache storage aliases')
    need(len(set(d['scratch_storage_ids']))==3 and not set(d['scratch_storage_ids'])&set(d['source_storage_ids']+[1]),'scratch aliases')
    need(d['grid']==[48,2,4] and d['threads']==64,'launch geometry')
    need(type(d['scale_log2e']) in (float,int) and math.isfinite(d['scale_log2e']) and d['scale_log2e']>0,'finite scale')
    need(d['controlled_CPU_lengths_table'] is True and d['live_service'] is False,'controlled component only; no service activation')
    return dict(namespace=source.NAMESPACE,rows=48,segments=4,CUDA_or_framework_executed=False)


def small_fixture(pitch,phase):
    """Same full-D128 counter/codec; change only the controlled row whitelist."""
    assert pitch in (2048,2560) and phase in (0,1,2)
    globals48=dict(old_fixture.fixture.__globals__,CASES=SMALL_CASES)
    make=types.FunctionType(old_fixture.fixture.__code__,globals48,'fixture')
    case=make(48,4,pitch,phase)
    old=case['cache'];packed=bytearray(old[:128]+b'\xa5'*(29*12544)+old[-128:])
    for page in range(29):
        a=128+page*16384;b=128+page*12544
        packed[b:b+4224]=old[a:a+4224]
        packed[b+4352:b+12544]=old[a+8192:a+16384]
    # Reference-only backing permits two actual N32 calls. Preserve global row
    # counter identities, not a freshly generated local-row0..15 replacement.
    g=old_fixture.GUARD_ELEMENTS
    q64=[old_fixture.BF16_GUARD]*g+case['query_bits'][g:-g]+[0]*(16*pitch)+[old_fixture.BF16_GUARD]*g
    lens=case['lengths'][g:-g]+[0]*16
    table=case['table'][g:-g]+[-1]*(16*old_fixture.COLS)
    assert len(q64)==2*g+64*pitch and len(lens)==64 and len(table)==64*old_fixture.COLS
    return dict(canonical_case=case,compact_cache=bytes(packed),query_backing64_bits=q64,
      lengths64=lens,table64=table,control_calls=[dict(source_rows=list(range(32)),native_rows=32,segments=4),
      dict(source_rows=list(range(32,64)),native_rows=32,segments=4,valid_rows=16,padded_rows=16)],
      reference_compare_rows=list(range(48)),candidate_only_rows48_63_remain_poison=True,
      control_reducer_rows48_63_expected_zero=True,reference_only_not_timed_performance_control=True)


def fullstep_plan(valid,initial_length,pages=SERVICE_PAGES):
    need(type(valid) is int and 0<valid<=48 and type(initial_length) is int and initial_length>=0,'fullstep scope')
    # Reserve for initial token count + three successive append boundary steps.
    needed=valid*((initial_length+3+15)//16)
    need(needed<=pages,'independent KV page budget exceeded')
    return dict(rows=48,segments=4,valid_rows=valid,padded_rows=48-valid,
      initial_lengths=[initial_length]*valid+[0]*(48-valid),maximum_after_three_appends=initial_length+3,
      physical_pages=pages,pages_reserved_for_three_appends=needed,spare_pages=pages-needed,
      table_columns=8192,nonoverlapping_per_sequence_physical_mapping_required=True,
      row_padding_table_minus_one=True,scratch_backing_rows=64,scratch_backing_segments=16,
      append_source_row_pitch=256,service_source_pitch2560_not_timed_by_this_plan=True,
      timed_region=['append','N48 producer','same S4 reducer'],initial_encode_cost_separate=True,
      references=['full128D I64 quantized QK','FP64 pagewise BF16-P partial','ideal quantized-score FP64 final'],
      same_pointwise_atol=.005,same_pointwise_rtol=.005,same_RRMSE_limit=.005,
      whole_cache_bytes_and_Graph_changed_B_before_eager_B_and_side_required=True,
      numerical_verified=False,timing_verified=False)


def plan():
    return dict(candidate=source.verify(),default='stdlib CPU source/descriptor plan',actual_library_sha256=None,
      small=dict(pages=29,cache_layer_offset=128,maximum_context=193,Q_pitches=[2048,2560],
        phases=['A','changed B Graph before fresh eager B','side C'],scratch_backing_rows=64,scratch_backing_segments=16,
        complete_OML_final_INT64_FP64_required=True,ideal_partial_diagnostic_separate=True,
        original_BF16_proxy_separate=True,reference_N32_calls=2,reference_padding_rows=16,
        reference_codec_cache_layout='compact12544 only; old22e needs explicit16KiB remap and is not an implicit compact control'),
      fullstep=[fullstep_plan(41,17408),fullstep_plan(42,16896)],
      original32_64_library_qualification_unchanged=True,new_numeric_GPU_frontend='pending candidate compile and actual SO/IR audit; do not execute old frontend with wrong rows/SO',
      no_Torch_import=True,no_service_activation=True,no_performance_or_quality_claim=True)


if __name__=='__main__':
    import json
    print(json.dumps(plan(),indent=2,sort_keys=True))
