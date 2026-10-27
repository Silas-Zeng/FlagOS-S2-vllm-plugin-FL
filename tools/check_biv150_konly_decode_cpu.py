#!/usr/bin/env python3
"""Private decoder source/math/address proofs only; default stdlib, never JIT/GPU."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tarfile

from biv150_konly_decode_factory import build,restore,validate_binding
from biv150_konly_decode_quant_reference import FORMAT,quantize_v2,rn32_div
from biv150_konly_decode_sdk import check_frontend
from biv150_konly_decode_ir import audit
from biv150_konly_decode_reference import (append_reference,attention_reference,cost_plan,encode_reference,
                                          mapped_bytes,numeric_policy)
from biv150_konly_i8_qk_reference import bf16,bf16_bits,decode_key,f32,indices,quantize
from check_biv150_konly_i8_qk_cpu import fixture

ROOT=Path(__file__).resolve().parents[1]


def reject(fn):
    try: fn()
    except (ValueError,AssertionError): return
    raise AssertionError('Expected rejection')


def run():
    frontend=check_frontend(ROOT)
    source=(ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text()
    modes={mode:build(source,mode) for mode in ('inline','preq')}
    failures=0
    for mode,(generated,proof) in modes.items():
        assert restore(generated,mode)
        for changed in [generated.replace('rescale = tl.math.exp2','rescale = tl.math.exp',1),
                        generated.replace('probability.to(V.dtype), V','probability.to(tl.int8), V',1),
                        generated.replace('out_dtype=tl.int32','out_dtype=tl.float32',1),
                        generated.replace('page * 4096 + 1024','page * 8192 + 1024',1)]:
            reject(lambda:restore(changed,mode));failures+=1
    support=(ROOT/'tools/kernels/biv150_konly_decode_support.py').read_text()
    compile(support,'<support-python-syntax-only>','exec')
    tree=ast.parse(support)
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quantize_bf16_rne')
    divs=[n for n in ast.walk(helper) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.div_rn']
    assert len(divs)==2 and not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.fma' for n in ast.walk(helper))
    # Codec v1 is preserved; v2 deliberately rounds the FP32 quotient before integer RNE.
    boundary=[240.,120.,-120.]+[0.]*125
    v1,v2=quantize(boundary),quantize_v2(boundary)
    assert v1[0][:3]==(127,63,-63) and v2[0][:3]==(127,64,-64) and v1[1]==v2[1]
    assert rn32_div(2.**-150,1.)==0 and rn32_div(3*2.**-150,1.)==2.**-148
    assert math.copysign(1,rn32_div(-0.,1.))<0
    rng=random.Random(20261027);vectors=[]
    for exponent in (-133,-126,-125,-100,-10,0,10,100,120):
        for peak in (128,144,160,176,192,208,224,240,254,255):
            p=bf16(peak*2.**exponent)
            vectors.append([p,bf16(p/2),bf16(-p/2),-0.]+[bf16(rng.uniform(-p,p)) for _ in range(124)])
    vectors.extend([[0.]*128,[bf16((-1)**i*2.**-133) for i in range(128)]])
    for row in vectors:
        codes,scale=quantize_v2(row)
        assert all(abs(c)<=127 for c in codes) and math.isfinite(scale) and scale>0
        # Independent integer-RN32 routine agrees with host-double sanity targets on these inputs.
        assert scale==(1. if max(map(abs,row))==0 else max(f32(max(map(abs,row))/127),2.**-126))
        assert codes==tuple(max(-127,min(127,round(f32(x/scale)))) for x in row)
    for bad in ([math.nan]*128,[math.inf]*128,[0.1]*128,[0.]*127):
        reject(lambda bad=bad:quantize_v2(bad));failures+=1
    binding=dict(mode='inline',k_strides=(16384,256,128,1),v_strides=(8192,256,128,1),
                 scale_strides=(4096,2,1),q_strides=(2048,128,1),source_storage='original',
                 destination_storage='private',format_id=FORMAT)
    assert validate_binding(**binding)
    for changed in [dict(k_strides=(8192,256,128,1)),dict(v_strides=(16384,256,128,1)),
                    dict(scale_strides=(8192,2,1)),dict(destination_storage='original'),
                    dict(format_id='bf16'),dict(q_strides=(2048,128,2)),dict(mode='decode')]:
        reject(lambda changed=changed:validate_binding(**(binding|changed)));failures+=1
    keys,values,queries=fixture(2);original=repr((keys,values));packed=encode_reference(keys,values)
    assert repr((keys,values))==original
    addressed=0
    for page in range(3):
        assert packed[page*16384+4224:page*16384+8192]==bytes([0xa5])*3968
        for slot in range(16):
            for head in range(2):
                assert decode_key(packed,page,slot,head)==quantize_v2(keys[page][slot][head])
                for dim in range(128):
                    ko,so,vo=indices(page,slot,head,dim)
                    assert struct.unpack_from('<H',packed,vo)[0]==bf16_bits(values[page][slot][head][dim])
                    addressed+=1
    for slots in ([0,15,16,-1],[47,2,31,48],[-2**63,2**63-1,-1,48]):
        newk=[keys[i%3][i%16] for i in range(4)];newv=[values[i%3][i%16] for i in range(4)]
        actual=append_reference(packed,slots,newk,newv);allowed=mapped_bytes(slots,3)
        assert all(a==b for i,(a,b) in enumerate(zip(packed,actual)) if i not in allowed)
        if not allowed: assert actual==packed
    reject(lambda:mapped_bytes([0,0],3));failures+=1
    for n in (0,1,15,16,17,31,33):
        kt=[row for page in keys for row in page][:n];vt=[row for page in values for row in page][:n]
        original=attention_reference(queries+queries,kt,vt,False)
        quantized=attention_reference(queries+queries,kt,vt,True)
        assert all(math.isfinite(v) for rows in (original,quantized) for row in rows for v in row)
        if n==0: assert original==quantized==[[0.]*128 for _ in range(16)]
    ko,so,vo=indices(2**30+3,15,1,127)
    assert min(ko,so,vo)>2**43 and max(ko,so,vo)<2**63
    reject(lambda:numeric_policy(False,True,True,True));failures+=1
    reject(lambda:numeric_policy(True,False,True,True));failures+=1
    reject(lambda:numeric_policy(True,True,False,False));failures+=1
    assert numeric_policy(True,True,False,True)['diagnostic_only']
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','96118f1','--',
                    'tools/kernels/biv150_konly_i8_qk.py','tools/bench_biv150_konly_i8_qk.py',
                    'tools/biv150_konly_i8_qk_reference.py','tools/biv150_konly_i8_qk_ir.py',
                    'tools/check_biv150_konly_i8_qk_cpu.py','vllm_fl','csrc'],check=True)
    assert not audit({},modes['inline'][0])['passed']
    with tarfile.open(ROOT/'docs/evidence/konly-i8-qk-20261027/numeric-raw-v1.tar.gz','r:gz') as tar:
        prefix='out/compiler-ir/bc85d9fceb12218562e5c2abb53944eb97875dac3dae5daa7c67669d9b784307/actual.'
        primitive_ir={kind:tar.extractfile(prefix+kind).read().decode() for kind in ('ttgir','llir','asm')}
    assert not audit(primitive_ir,modes['inline'][0])['passed']
    defaults={}
    for mode in modes:
        text=subprocess.check_output([sys.executable,str(ROOT/'tools/bench_biv150_konly_decode.py'),'--mode',mode],text=True)
        defaults[mode]=json.loads(text)
        assert defaults[mode]['cpu_only'] and not defaults[mode]['torch_imported'] and not defaults[mode]['triton_imported']
        assert not defaults[mode]['numeric_and_timing_runner_prepared']
    assert not {'torch','triton'}.intersection(sys.modules)
    return dict(cpu_only=True,passed=True,codec=FORMAT,source_inverse={m:p for m,(_,p) in modes.items()},
                codec_vectors=len(vectors),codec_elements=len(vectors)*128,writer_byte_addresses=addressed,
                rejects=failures,v1_v2_boundary=dict(peak=240,value=120,v1=63,v2=64,scale=v2[1]),
                cost_plans={m:cost_plan(m) for m in modes},primitive_961_unchanged=True,
                default_cli_cpu_only=True,compile_entry_opt_in=True,numeric_and_timing_runner_prepared=False,
                support_sha256=hashlib.sha256(support.encode()).hexdigest(),gpu_executed=False,
                actual_decoder_ir_verified=False,primitive_ir_rejected_as_full_decoder=True,target_div_rn_frontend=frontend,
                target_div_rn_native_support_verified=False,
                device_quantizer_verified=False,precision_gate=False,service_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);args=parser.parse_args()
    result=run()
    if args.out: args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','cpu_only','codec_vectors','writer_byte_addresses','rejects','precision_gate')}))
