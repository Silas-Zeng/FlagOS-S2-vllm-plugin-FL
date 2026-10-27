#!/usr/bin/env python3
"""CPU source/oracle falsifiers only. There is deliberately no compile/GPU option."""
import argparse
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys

from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_i8_qk_reference import bf16, f32
import biv150_konly_prefill_factory as factory
import biv150_konly_prefill_reference as reference


def reject(call):
    try:
        call()
    except (ValueError, AssertionError):
        return
    raise AssertionError('CPU mutant incorrectly accepted')


def descriptors(total, pages, segments=None):
    def d(dtype, shape, stride, storage, offset=0, size=None):
        width = dict(bfloat16=2, int8=1, float32=4)[dtype]
        return dict(dtype=dtype, shape=shape, stride=stride, storage=storage,
                    storage_offset=offset, storage_bytes=size or width*math.prod(shape))
    out = dict(q=d('bfloat16',(total,16,128),(2048,128,1),'Q'),
               out=d('bfloat16',(total,16,128),(2048,128,1),'output'),
               k=d('int8',(pages,16,2,128),(16384,256,128,1),'packed',size=pages*16384),
               kscale=d('float32',(pages*4096,),(1,),'packed',size=pages*16384),
               v=d('bfloat16',(pages,16,2,128),(8192,256,128,1),'packed',4096,pages*16384))
    if segments:
        out.update(scratch=d('float32',(total,16,segments,128),(16*segments*128,segments*128,128,1),'scratch'),
                   maxima=d('float32',(total,16,segments),(16*segments,segments,1),'maxima'),
                   sums=d('float32',(total,16,segments),(16*segments,segments,1),'sums'))
    return out


def run(emit=None):
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    source = factory.inputs()
    proofs, generated, failures = {}, {}, 0
    for profile in factory.PROFILES:
        code, proof = factory.build(profile, source)
        assert factory.restore(code, profile, source) == factory.canonical(source, profile)
        assert ast.dump(ast.parse(factory.restore(code,profile,source))) == ast.dump(ast.parse(factory.canonical(source,profile)))
        tree = ast.parse(code)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == factory.name(profile))
        loop = next(n for n in fn.body if isinstance(n, ast.For))
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        quant = [n for n in calls if ast.unparse(n.func) == 'quantize_bf16_rne']
        assert len(quant) == 1 and quant[0].lineno < loop.lineno
        assert ast.unparse(quant[0]) == 'quantize_bf16_rne(Q)'
        assert not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == 'QCode' for n in ast.walk(loop))
        assert 'V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)' in code
        assert 'acc += tl.dot(P.to(V.dtype), V)' in code
        assert 'M, L, P, alpha = softmax_step(S, M, L)' in code
        integer_dots = [n for n in calls if ast.unparse(n.func) == 'tl.dot' and ast.unparse(n.args[0]) == 'QCode']
        assert len(integer_dots) == 1
        assert ast.unparse(integer_dots[0].args[2]) == 'tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.int32)'
        assert any(k.arg == 'out_dtype' and ast.unparse(k.value) == 'tl.int32' for k in integer_dots[0].keywords)
        for old, new in [
            ('V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)', 'V = V_load.to(tl.int8)'),
            ('acc = acc * alpha[:, None]', 'acc = acc'),
            ('query_code_scale[:, None]', 'query_code_scale[None, :]'),
            ('mask=mask, other=1.0', 'mask=mask, other=0.0'),
            ('out_dtype=tl.int32', 'out_dtype=tl.float32'),
            ('query_mask_0[:, None] & seq_mask', 'seq_mask'),
        ]:
            assert old in code
            reject(lambda old=old,new=new: factory.restore(code.replace(old,new,1),profile,source))
            failures += 1
        generated[profile], proofs[profile] = code, proof
    bad = dict(source); bad['helpers.py'] += '\n'
    reject(lambda: factory.build('q32',bad)); failures += 1
    reject(lambda: factory.build('raw2d_decode',source)); failures += 1

    mask_cases = [
        ('normalized2d128',[0,1,15,16,17,31,32,33],[5,17,31,32,33,47,48,49],1),
        ('normalized2d128',[1,3,1,17],[17,19,1,33],1),
        ('q32',[1025],[1041],1),
        ('multi2',[513,513],[529,545],1),
        ('multi2',[1025,1],[1041,17],1),
        ('raw3d',[0,1,1],[0,17,65],1),
        ('raw3d',[0,1,1],[0,17,65],4),
        ('raw3d',[0,1,1],[0,17,65],16),
    ]
    masks = [dict(profile=p, query_lengths=q, key_lengths=k, segments=s,
                  **reference.coverage(p,q,k,s)) for p,q,k,s in mask_cases]
    assert reference.grid('multi2',[513,513])[0] == 34
    assert reference.grid('multi2',[1025,1])[0] == 34
    assert reference.grid('multi2',[513,513])[0] != (1026//32+1)

    binding_cases = [('normalized2d128',[1,3,1],[17,19,1],1),('q32',[1025],[1041],1),
                     ('multi2',[513,513],[529,545],1),('raw3d',[1,1],[17,65],4)]
    for p,q,k,s in binding_cases:
        kwargs = dict(format_id=factory.FORMAT,query_lengths=q,key_lengths=k,pages=80,
                      tensors=descriptors(sum(q),80,s if p=='raw3d' else None),segments=s)
        reference.validate_binding(p,**kwargs)
        for role, field, wrong in [('k','stride',(8192,256,128,1)),('v','stride',(16384,256,128,1)),
                                   ('k','dtype','bfloat16'),('kscale','storage_offset',1024),
                                   ('kscale','shape',(80,16,2)),('v','storage_offset',0),
                                   ('kscale','storage','other'),('out','storage','packed'),
                                   ('out','shape',(True,16,128)),('k','storage_bytes',100)]:
            mutant = deepcopy(kwargs); mutant['tensors'][role][field] = wrong
            reject(lambda mutant=mutant:reference.validate_binding(p,**mutant)); failures += 1
        for field, wrong in [('format_id','bf16'),('kv_quant_mode',2),('use_td',True),('table_pitch',64)]:
            mutant = deepcopy(kwargs); mutant[field] = wrong
            reject(lambda mutant=mutant:reference.validate_binding(p,**mutant)); failures += 1
    raw = dict(format_id=factory.FORMAT,query_lengths=[1,1],key_lengths=[17,65],pages=80,
               tensors=descriptors(2,80,4),segments=4)
    raw['tensors']['scratch']['stride']=(32768,2048,128,1)
    reject(lambda:reference.validate_binding('raw3d',**raw)); failures += 1
    reject(lambda:reference.grid('raw3d',[1],3)); failures += 1

    addresses = 0
    for page in (0,1,2**16,2**31-1):
        codes, scales, values = set(), set(), set()
        for slot in range(16):
            for head in range(2):
                for dim in range(128):
                    ko, so, vo = reference.address(page,slot,head,dim)
                    codes.add(ko);scales.update(range(so,so+4));values.update((vo,vo+1));addresses += 1
        assert len(codes)==4096 and len(scales)==128 and len(values)==8192
        assert not (codes&scales or codes&values or scales&values)
        assert min(scales)==page*16384+4096 and max(scales)==page*16384+4223
    # Noncontiguous pages, partial last page and invalid masked metadata poison.
    tile = reference.tile_addresses([3,1,2**31-1,2**31-1],0,17,1)
    assert len(tile)==64 and all(x is not None for x in tile[:17]) and tile[17:]==[None]*47
    reject(lambda:reference.tile_addresses([0],0,1,0)); failures += 1
    reject(lambda:reference.address(0,16,0,0)); failures += 1
    reject(lambda:reference.address(True,0,0,0)); failures += 1

    vectors = [[0.]*128,[-0.]*128,[240.,120.,-120.]+[0.]*125,
               [2.**-133,-2.**-133]+[0.]*126,[bf16((i-64)/16) for i in range(128)]]
    for row in vectors:
        codes, scale = quantize_v2(row)
        assert all(type(c) is int and -127<=c<=127 for c in codes)
        assert math.isfinite(scale) and scale>=2**-126
    assert quantize_v2(vectors[2])[0][1:3] == (64,-64)
    assert quantize_v2(vectors[0])[1] == 1.0
    assert quantize_v2(vectors[3])[1] == 2**-126
    for bad in ([float('nan')]*128,[float('inf')]*128,[0.1]*128,[0.]*127):
        reject(lambda bad=bad:quantize_v2(bad)); failures += 1
    rng = random.Random(20261027)
    q = [bf16(rng.uniform(-.5,.5)) for _ in range(128)]
    keys = [[bf16(rng.uniform(-.5,.5)) for _ in range(128)] for _ in range(17)]
    values = [[bf16(rng.uniform(-.5,.5)) for _ in range(128)] for _ in range(17)]
    numeric = []
    for context,pos in ((0,0),(0,1),(12,0),(12,4)):
        base = reference.attention(q,keys,values,pos,context,False)
        quant = reference.attention(q,keys,values,pos,context,True)
        assert all(math.isfinite(v) for row in (base,quant) for v in row)
        numeric.append(dict(context=context,query_position=pos,max_abs_quantization_delta=max(abs(a-b) for a,b in zip(base,quant))))
    if emit:
        emit.mkdir(parents=True,exist_ok=False)
        for profile,code in generated.items():
            (emit/(profile+'.py')).write_text(code)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(cpu_only=True,passed=True,main=factory.MAIN,dependency_PR=79,dependency_commit=factory.DEPENDENCY,
                source_proofs=proofs,mask_cases=masks,address_vectors=addresses,codec_vectors=len(vectors),rejects=failures,
                closed_FP64_examples=numeric,FP32_score_order='RN32(RN32(RN32(I32*sQ)*sK)*natural_softmax_scale)',
                original_BF16_and_quantized_operand_FP64_distinct=True,BF16_P_PV_and_device_FP32_not_emulated=True,
                actual_I8_operand_hoist_verified=False,actual_resources_known=False,actual_tensor_binding_verified=False,
                compile_numeric_and_timing_runner_prepared=False,kernel_launches=0,JIT_executed=False,
                torch_imported=False,triton_imported=False,precision_gate=False,performance_verified=False,
                service_installed=False,capacity_reclaimed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--emit-dir',type=Path)
    args = parser.parse_args()
    result = run(args.emit_dir)
    text = json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(text)
    print(text,end='')


if __name__ == '__main__':
    main()
