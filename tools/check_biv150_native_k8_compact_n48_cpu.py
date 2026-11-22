#!/usr/bin/env python3
"""Meaningful source/shape/row-copy checks; no framework/device/NumPy import."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import biv150_native_k8_compact_n48_factory as source
import biv150_native_k8_compact_n48_contract as contract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);a=p.parse_args()
    proof=source.verify();checks=[]
    for pitch in (2048,2560):
        contract.validate(contract.descriptor(pitch));checks.append('admit N48/S4 pitch'+str(pitch))
    refusals=[]
    def refuse(label,update):
        d=contract.descriptor();update(d)
        try:contract.validate(d)
        except (ValueError,KeyError,TypeError,IndexError):refusals.append(label);return
        raise AssertionError('admission unexpectedly allowed '+label)
    for n in (0,31,32,40,41,42,47,49,56,64):refuse('rows'+str(n),lambda d,n=n:d.update(rows=n,q_shape=[n,16,128]))
    for s in (0,2,8,16):refuse('S'+str(s),lambda d,s=s:d.update(segments=s))
    for label,update in (
      ('bool row',lambda d:d.update(rows=True)),('Q pitch2304',lambda d:d.update(pitch=2304)),
      ('Q head stride',lambda d:d.update(q_strides=[2048,129,1])),
      ('BF16 K reinterpretation',lambda d:d['dtype'].update(key='bf16')),
      ('scale offset twice',lambda d:d.update(scale_offset_from_layer=4096)),
      ('V offset twice',lambda d:d.update(value_offset_from_layer=8704)),
      ('16KiB K stride',lambda d:d.update(key_strides=[16384,256,128,1])),
      ('truncated cache',lambda d:d.update(cache_backing_bytes=100)),
      ('scale too short',lambda d:d.update(scale_numel=1)),
      ('misaligned layer',lambda d:d.update(cache_layer_offset=129)),
      ('noncontiguous lengths',lambda d:d.update(lengths_stride=[2])),
      ('wrong table rows',lambda d:d.update(table_shape=[32,16])),
      ('scratch S16 shape',lambda d:d.update(partial_shape=[64,16,16,128])),
      ('scratch token overlap',lambda d:d.update(partial_strides=[128,2048,128,1])),
      ('scalar stride mismatch',lambda d:d.update(summation_strides=[128,8,1])),
      ('source cache alias',lambda d:d.update(source_storage_ids=[1,3,4])),
      ('scratch alias',lambda d:d.update(scratch_storage_ids=[5,5,7])),
      ('scratch source alias',lambda d:d.update(scratch_storage_ids=[2,6,7])),
      ('wrong grid',lambda d:d.update(grid=[48,2,16])),
      ('NaN scale',lambda d:d.update(scale_log2e=float('nan'))),
      ('other device',lambda d:d.update(devices=['cuda:1']*9)),
      ('live service',lambda d:d.update(live_service=True)),
      ('unknown content provenance',lambda d:d.update(controlled_CPU_lengths_table=False))):refuse(label,update)
    fixtures=[]
    for pitch in (2048,2560):
        for phase in (0,1,2):
            f=contract.small_fixture(pitch,phase);case=f['canonical_case'];g=64;q=f['query_backing64_bits']
            # Two actual N32 controls consume the identical global query rows;
            # concatenating a separately generated second16 would be wrong.
            stitched=[]
            for call in f['control_calls']:
                for row in call['source_rows']:
                    if row<48:stitched.extend(q[g+row*pitch:g+row*pitch+2048])
                    else:assert all(v==0 for v in q[g+row*pitch:g+(row+1)*pitch]) and f['lengths64'][row]==0
            wanted=[v for row in range(48) for v in case['query_bits'][g+row*pitch:g+row*pitch+2048]]
            assert stitched==wanted and len(stitched)==48*16*128
            assert f['lengths64'][48:]==[0]*16 and all(x==-1 for x in f['table64'][48*16:])
            assert f['compact_cache'][:128]==case['cache'][:128] and f['compact_cache'][-128:]==case['cache'][-128:]
            for page in range(29):
                old=128+page*16384;new=128+page*12544
                assert f['compact_cache'][new:new+4224]==case['cache'][old:old+4224]
                assert f['compact_cache'][new+4352:new+12544]==case['cache'][old+8192:old+16384]
                assert f['compact_cache'][new+4224:new+4352]==b'\xa5'*128
            fixtures.append(dict(pitch=pitch,phase=phase,all48_rows_full128D_reference_mapping=True,
              query64_sha256=source.sha(struct.pack('<'+str(len(q))+'H',*q)),compact_cache_sha256=source.sha(f['compact_cache']),
              lengths64=f['lengths64'],Q_head_scales_distinct=all(len(set(row))==16 for row in case['qscales'])))
    full=[contract.fullstep_plan(41,17408),contract.fullstep_plan(42,16896)]
    for n in (42,48):
        try:contract.fullstep_plan(n,17408)
        except ValueError:refusals.append(str(n)+' independent17K rows do not fit44769 pages')
        else:raise AssertionError('physical overcommit accepted')
    # Compile entry is byte-inverse to PR110 except its two identity bindings.
    original=source.blob('tools/compile_biv150_native_k8_compact_v_overlap.py').decode()
    actual=(source.ROOT/'tools/compile_biv150_native_k8_compact_n48.py').read_text()
    restored=actual.replace('import biv150_native_k8_compact_n48_factory as factory','import biv150_native_k8_compact_v_overlap_factory as factory').replace("name='cjj_native_k8_compact_n48_s4_probe'","name='cjj_native_k8_compact_v_overlap_probe'")
    assert restored==original
    assert not {'torch','triton','numpy'}&set(sys.modules)
    result=dict(schema=1,passed=True,source=proof,positive_descriptors=checks,rejected_descriptors=refusals,
      six_small_fixtures=fixtures,fullstep_budgets=full,compile_frontend_identity_only_inverse=True,
      descriptor_model_not_actual_CPP_execution=True,full_FP64_or_device_numerical_test_not_run=True,
      actual_library_sha256=None,numeric_verified=False,Graph_verified=False,side_verified=False,
      performance_verified=False,precision_gate=False,service_installed=False,framework_imported=False)
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.out:
        assert not a.out.exists();a.out.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
