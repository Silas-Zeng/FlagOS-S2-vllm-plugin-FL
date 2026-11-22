#!/usr/bin/env python3
"""Stdlib checks for exact private source transform and requested codec semantics."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import biv150_native_k8_source_factory as f
from biv150_konly_decode_quant_reference import quantize_v2,rn32_div
from compile_biv150_native_k8_qk import cpu_plan


def reject(call):
    try:call()
    except (ValueError,AssertionError):return
    raise AssertionError('Fault accepted')


def codec_source(source):
    lines='\n'.join(line.split('//')[0] for line in source.splitlines())
    assert lines.count('__fdiv_rn(')==2 and lines.count('__float2int_rn(')==1
    assert 'fmaxf(__fdiv_rn(peak,127.0f),1.1754943508222875e-38f)' in lines
    assert 'peak==0.0f?1.0f:' in lines
    assert '__float2int_rn(__fdiv_rn(value,scale))' in lines
    assert 'max(-127,min(127,code))' in lines
    assert not any(token in lines for token in ('/','roundf(','nearbyint(','rcp','__float2int_rz'))


def check():
    proof=f.verify();source=f.DEST.read_text();original=f.original();faults=[]
    for name,changed in [
      ('PV_FMA',source.replace('running_l0=fmaf(running_l0,rho0,page_l0)','running_l0=running_l0*rho0+page_l0')),
      ('BF16_P_rounding',source.replace('__float2bfloat16_rn(p0)','__float2bfloat16_rn(p0*0.99f)')),
      ('partial_output',source.replace('acc[n][2];acc_out[a1','acc[n][0];acc_out[a1')),
      ('signed_score',source.replace('signed_i32_word(scores[0])','scores[0]')),
      ('double_scale_offset',source.replace('page)*4096+1024+col','page)*4096+2048+col')),
      ('Q_scale_input_pair',source.replace('(g/2+2)*16+col','g*16+col')),
    ]:
        assert changed!=source;reject(lambda changed=changed:f.inverse(changed));faults.append(name)
    header=(f.ROOT/'tools/kernels/biv150_native_k8_codec.cuh').read_text();codec_source(header)
    for name,bad in [('approx_div',header.replace('__fdiv_rn(value,scale)','(value/scale)')),
                     ('half_away',header.replace('__float2int_rn','__float2int_rz'))]:
        reject(lambda bad=bad:codec_source(bad));faults.append(name)
    # Independent requested C++ float2int RN simulation vs existing rational
    # codec. This cannot establish target FTZ/intrinsic lowering.
    vectors=[]
    ties=[127.,-.5,.5,1.5,2.5,-1.5,-2.5]+[0.]*121
    vectors.append(ties)
    for value in (0.,-0.,2.**-133,2.**-130,2.**-126,1.,127.,255.,(2.-2.**-7)*2.**127):
        vectors.append([value if i%2 else -value for i in range(128)])
    for values in vectors:
        peak=max(abs(v) for v in values)
        scale=1. if peak==0 else max(rn32_div(peak,127.),2.**-126)
        requested=tuple(max(-127,min(127,int(round(rn32_div(v,scale))))) for v in values)
        assert (requested,scale)==quantize_v2(values)
    assert quantize_v2(ties)[0][:7]==(127,0,0,2,2,-2,-2)
    # Original non-QK suffix is literal, including FMA / P rounding / PV / tail
    # V clear. Exact inverse catches any edit to otherwise unpatched source.
    suffix=' float page_m0=s0,page_m1=s1;'
    end=' running_m0=new_m0;running_m1=new_m1;'
    assert original[original.index(suffix):original.index(end)]==source[source.index(suffix):source.index(end)]
    abi=(f.ROOT/'tools/kernels/biv150_native_k8_qk.cpp').read_text()
    for required in ('k.scalar_type()==at::kChar','scale_full.scalar_type()==at::kFloat',
                     'v.scalar_type()==at::kBFloat16','kb+8192','scale_full.data_ptr())==kb',
                     'k.strides()==at::IntArrayRef({16384,256,128,1})','pages*16384<=backing_end(k)',
                     '!backing_overlap(src,k)','C10_CUDA_KERNEL_LAUNCH_CHECK()'):
        assert required in abi,required
    for path in (f.ROOT/'tools').glob('*native_k8*.py'):ast.parse(path.read_text(),filename=str(path))
    plan=cpu_plan();assert all(plan[k]==0 for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
    assert plan['codec_actual_IR_verified'] is False and plan['precision_gate'] is False
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(source_transform=proof,default_plan=plan,requested_codec_boundary_vectors=len(vectors),
      target_RN32_FTZ_verified=False,rejected=faults,default_imports_framework=False,
      compile_or_device_executed=False,precision_gate=False)

if __name__=='__main__':print(json.dumps(check(),sort_keys=True,indent=2))
