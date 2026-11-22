#!/usr/bin/env python3
"""Bounded codec/address/adapter faults. No device, NumPy, or cloned run framework."""
import argparse
import ast
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import biv150_native127_v8_numeric as c
import biv150_native127_v8_fixture as f


def reject(fn):
    try:fn()
    except (AssertionError,KeyError,TypeError,ValueError):return
    raise AssertionError('Bad input accepted')


def check(a):
    parent=c.previous.parent(a.parent_source)
    original,adapter=c.adapted_run(parent);ast.parse(original);assert adapter['inverse_byte_exact']
    before,after=f.pagewise_source(parent);assert after.replace('.reshape(rows,COLS)','.reshape(rows,16)')==before
    source=Path(__file__).with_name('probe_biv150_native127_v8_numeric.py')
    r=subprocess.run([sys.executable,'-B','-S',str(source)],capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)==json.loads(json.dumps(c.plan()))
    for args in (['--gpu'],['--gpu','--root-owned-exclusive'],['--static-review-sha256',c.STATIC_REVIEW_SHA]):
        assert subprocess.run([sys.executable,'-B','-S',str(source),*args],capture_output=True,text=True).returncode==2
    module=c.load(parent,'biv150_native_k8_consumer_fixture','cjj127_cpu_original')
    # Exact I8 integers and powers of2 are all representable in BF16.
    for code in range(-128,128):
        assert f.dequant_bf16(code,0.25)==module.bf16_bits(code*0.25)
    assert f.dequant_bf16(1,1.00390625)==0x3f80
    assert f.dequant_bf16(1,1.01171875)==0x3f82
    for code,scale in ((True,1.),(128,1.),(0,float('nan')),(0,float('inf')),(0,0.),(0,-1.)):
        reject(lambda:f.dequant_bf16(code,scale))
    assert module.quantize_v2([0.]*128)==((0,)*128,1.)
    assert module.quantize_v2([2.**-133]+[0.]*127)[1]==2.**-126
    bits=[[[module.bf16_bits(((slot*7+head*13+d)%127-63)/64.) for d in range(128)] for head in range(2)] for slot in range(16)]
    case=dict(cache=b'\x5a'*128+b'\xa5'*16384+b'\x5a'*128,value_bits=[bits])
    control,private,decoded=f.prepare_v(case,module.quantize_v2,module.bf16_value,pages=1)
    assert case['cache']==b'\x5a'*128+b'\xa5'*16384+b'\x5a'*128
    assert control[:128]==private[:128]==control[-128:]==private[-128:]==b'\x5a'*128
    assert control[128:8320]==private[128:8320]==case['cache'][128:8320]
    for slot in range(16):
        for head in range(2):
            expected,scale=module.quantize_v2([module.bf16_value(x) for x in bits[slot][head]])
            codes=struct.unpack_from('<128b',private,128+8192+slot*256+head*128)
            assert codes==expected and struct.unpack_from('<f',private,128+12288+slot*8+head*4)[0]==scale
            actual=struct.unpack_from('<128H',control,128+8192+slot*512+head*256)
            assert list(actual)==decoded[0][slot][head]==[f.dequant_bf16(int(x),scale) for x in expected]
    for phase in (0,1,2):
        lens=f.lengths(phase);table=f.mapping(lens,phase)
        assert len(lens)==64 and lens.count(6143)==lens.count(6144)==1 and 0 in lens
        tail=(384*7+phase*5)%f.PAGES
        for row,L in enumerate(lens):
            count=(L+15)//16;ids=table[row*f.COLS:(row+1)*f.COLS]
            assert all(type(x) is int and 0<=x<f.PAGES for x in ids[:count]) and ids[count:]==[-1]*(f.COLS-count)
            assert (tail in ids[:count])==(L==6143)
            if L==6143:assert ids[383]==tail
    for page in (0,1,2**31-1):
        base=page*16384
        assert base+8192+15*256+128+127<base+12288
        assert base+12288+15*8+4+4<=base+12416<base+16384<2**63
    arm=dict(records=[dict(case=list(case),snapshots={stage:{field:dict(bytes=4,sha256='0'*64) for field in ('partial','maximum','summation','output')} for stage in c.STAGES}) for case in f.CASES])
    arms={r:copy.deepcopy(arm) for r in c.ROLES};c.compare(arms)
    faults=[]
    for field in ('partial','maximum','summation','output'):
        bad=copy.deepcopy(arms);bad['packed_v8']['records'][0]['snapshots']['graphB-after'][field]['sha256']='1'*64
        reject(lambda:c.compare(bad));faults.append('cross_arm_'+field)
    bad=copy.deepcopy(arms);del bad['control']['records'][0]['snapshots']['graphB-after'];reject(lambda:c.compare(bad));faults.append('missing_Graph')
    actual_fields=(a.actual_raw,a.static_review,a.control_library,a.codec_summary)
    assert all(actual_fields) or not any(actual_fields)
    if all(actual_fields):
        args=SimpleNamespace(root_owned_exclusive=True,parent_source=a.parent_source,
          static_review=a.static_review,static_review_sha256=c.STATIC_REVIEW_SHA,
          compile_summary=a.actual_raw/'out/compile-summary.json',library=a.actual_raw/'out/build/cjj_native_k8_packed_v8_probe.so',
          control_library=a.control_library,codec_summary=a.codec_summary,codec_proof_sha256=c.previous.CODEC_SHA,
          out=Path('/private/tmp/cjj127_CPU_admission_only_no_device_or_output_20261030'))
        c.admission(args)
        for field,value in (('root_owned_exclusive',False),('static_review_sha256',None),('library',args.control_library),('control_library',args.library),('codec_proof_sha256',None)):
            bad=copy.copy(args);setattr(bad,field,value);reject(lambda:c.admission(bad));faults.append('admission_'+field)
    old=c.STATIC_REVIEW_SHA;c.STATIC_REVIEW_SHA=None
    reject(lambda:c.admission(SimpleNamespace(root_owned_exclusive=True)))
    c.STATIC_REVIEW_SHA=old;faults.append('missing_static_Root_pin')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(completed=True,consumer_adapter=adapter,pagewise_extent_only_inverse=True,
      V_I8_patterns=256,one_page_codec_and_explicit_RN32_BF16_bits_checked=True,
      unique_tail_invalid_source_not_shared_with_full_rows=True,phases=3,query_pitches=[2048,2560],
      negative_checks=faults,Root_static_review_sha256=c.STATIC_REVIEW_SHA,
      actual_compile_SO_static_and_codec_files_replayed=all(actual_fields),Root_exclusive_flag_was_CPU_stub_only=all(actual_fields),
      full_fixture_or_FP64_executed=False,framework_imported=False,GPU_executed=False,
      numeric_verified=False,performance_verified=False,precision_gate=False,service_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True)
    for name in ('actual-raw','static-review','control-library','codec-summary'):p.add_argument('--'+name,type=Path)
    print(json.dumps(check(p.parse_args()),indent=2,sort_keys=True))
