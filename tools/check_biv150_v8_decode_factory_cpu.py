"""Bounded CPU proof of A/bias. Never import Torch/Triton or execute generated code."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys

from biv150_v8_decode_factory import (BIAS_EXPRESSION, FIXTURE, INSERT, MAIN,
    MAIN_GQA_SHA, ROOT, V8_COMMIT, code_address, fp32_bias_byte, make_vdecode,
    sha, validate_emitted)


def f32_bits(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


def bf16_bits(value):
    bits = f32_bits(value)
    return ((bits+0x7FFF+((bits>>16)&1))>>16)&0xFFFF


def check():
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    original = FIXTURE.read_bytes()
    generated,proof = make_vdecode()
    # Baseline comparison must remain against current-main BF16, not only the
    # retired slow V8 reference. Its GQA source is still the pinned original.
    current = (ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_bytes()
    assert sha(current)==MAIN_GQA_SHA
    digest = hashlib.sha256()
    edges = []
    for byte in range(256):
        signed = struct.unpack('<b',bytes([byte]))[0]  # independent two's-complement oracle
        value = fp32_bias_byte(byte)
        assert f32_bits(value)==f32_bits(float(signed)), (byte,value,signed)
        assert f32_bits(float(signed))&0xFFFF == 0, 'Every signed I8 integer is exact BF16'
        assert bf16_bits(value)==bf16_bits(float(signed))
        digest.update(struct.pack('<BIH',byte,f32_bits(value),bf16_bits(value)))
        if signed in (-128,-127,0,127):
            edges.append(dict(byte=byte,signed=signed,fp32_hex=f'{f32_bits(value):08x}',bf16_hex=f'{bf16_bits(value):04x}'))
    assert f32_bits(fp32_bias_byte(0))==0, 'Masked other=0 must yield positive zero'
    assert generated.replace(INSERT,b'')==original
    assert generated.count(b'V.to(tl.bfloat16)')==original.count(b'V.to(tl.bfloat16)')==2
    rejected = []
    def reject(label, action):
        try:action()
        except (AssertionError,ValueError,NotImplementedError,TypeError):rejected.append(label)
        else:raise AssertionError('Mutant was accepted: '+label)
    for label,before,after in [
        ('wrong_xor','^ 128','^ 127'),
        ('wrong_bias_bits','0x4B000000','0x4A800000'),
        ('wrong_subtract','8388736.0','8388608.0'),
        ('missing_integer_bitcast','tl.uint8, bitcast=True','tl.uint8'),
        ('missing_float_bitcast','tl.float32, bitcast=True','tl.float32'),
        ('tail_mask','mask=column_valid[:, None], other=0.0','mask=column_valid[:, None], other=1.0'),
        ('pv_math','probability * value_scale[None, :]','probability + value_scale[None, :]'),
        ('page_width','page * v_stride_block','page.to(tl.int32) * v_stride_block'),
        ('new_cache_write','        running_max = new_max','        tl.store(value_cache_ptr, V)\n        running_max = new_max')]:
        assert before.encode() in generated
        reject(label,lambda b=before,a=after:validate_emitted(generated.replace(b.encode(),a.encode())))
    reject('unfrozen_template',lambda:make_vdecode(original+b'\n'))
    reject('lookup_not_implemented',lambda:make_vdecode(variant='lookup'))
    # A never adds a table ABI. These rejections are not B's shape/dtype/alias
    # proof: B is deliberately unimplemented, including valid tables.
    for label,table in [('short_table',[0]*255),('wrong_dtype_table',bytes(256)),
                        ('possible_alias_table',object()),('full_table_disallowed_in_A',[0]*256)]:
        reject(label,lambda t=table:make_vdecode(table=t))
    for byte in (-1,256,False,1.0,None):
        reject('invalid_byte_'+repr(byte),lambda b=byte:fp32_bias_byte(b))
    addressed = set()
    for page in (0,2**17,2**31-1):
        for slot in range(16):
            for head in range(2):
                for dim in range(128):
                    address = code_address(page,slot,head,dim)
                    assert 8192 <= address-page*16384 < 12288
                    assert address not in addressed
                    addressed.add(address)
    for values in [(-1,0,0,0),(2**31,0,0,0),(False,0,0,0),(0,-1,0,0),
                   (0,16,0,0),(0,0,2,0),(0,0,0,128)]:
        reject('invalid_address_'+repr(values),lambda v=values:code_address(*v))
    assert max(addressed)>2**32 and max(addressed)<2**63
    # No accidental runtime/import/runner additions in the CPU factory module.
    module = ast.parse((ROOT/'tools/biv150_v8_decode_factory.py').read_text())
    imports = [alias.name for node in ast.walk(module) if isinstance(node,ast.Import) for alias in node.names]
    assert not any(name.startswith(('torch','triton')) for name in imports)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(main=MAIN,v8_reference_commit=V8_COMMIT,source=proof,
        all_256_bytes_fp32_and_bf16_exact=True,bitpattern_sha256=digest.hexdigest(),edges=edges,
        codec_range_stays_minus127_to127=True,primitive_minus128_does_not_relax_codec=True,
        masked_zero_bits='00000000',address_vectors=len(addressed),
        addresses_above_32bits=True,maximum_address=max(addressed),
        allocation_bytes_per_page=16384,code_payload_unchanged=True,scales_unchanged=True,
        rejected_mutants=rejected,table_factory_implemented=False,
        main_bf16_reference_still_required=True,full_consumer_connected=False,
        torch_or_triton_imported=False,jit_executed=False,gpu_executed=False,
        actual_ir_available=False,target_support_validated=False,
        precision_gate=False,official_score=False,physical_capacity_reclaimed=False)


if __name__ == '__main__':
    print(json.dumps(check(),indent=2))
