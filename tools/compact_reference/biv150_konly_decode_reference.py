"""Stdlib v2 packed-writer and independent whole-attention CPU oracles."""
import math
import struct

from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_i8_qk_reference import PAGE_BYTES, bf16_bits, indices, validate_vector


def valid_slots(slots,pages):
    if type(pages) is not int or not 0<pages<2**31: raise ValueError('bounded physical allocation required')
    if any(type(s) is not int or not -2**63<=s<2**63 for s in slots): raise ValueError('I64 slot metadata required')
    valid=[s for s in slots if 0<=s<pages*16]
    if len(valid)!=len(set(valid)): raise ValueError('Overlapping live writer destinations')
    return valid


def mapped_bytes(slots,pages):
    result=set()
    for s in valid_slots(slots,pages):
        page,slot=divmod(s,16)
        for head in range(2):
            ko,so,vo=indices(page,slot,head)
            result.update(range(ko,ko+128));result.update(range(so,so+4));result.update(range(vo,vo+256))
    return result


def append_reference(before,slots,keys,values):
    if len(before)%PAGE_BYTES or len(slots)!=len(keys) or len(keys)!=len(values): raise ValueError('Writer allocation/shape mismatch')
    pages=len(before)//PAGE_BYTES;valid_slots(slots,pages)
    result=bytearray(before)
    for token,s in enumerate(slots):
        if not 0<=s<pages*16: continue
        page,slot=divmod(s,16)
        if len(keys[token])!=2 or len(values[token])!=2: raise ValueError('Two KV heads required')
        for head in range(2):
            codes,scale=quantize_v2(keys[token][head]);vv=validate_vector(values[token][head])
            ko,so,vo=indices(page,slot,head)
            struct.pack_into('<128b',result,ko,*codes);struct.pack_into('<f',result,so,scale)
            struct.pack_into('<128H',result,vo,*[bf16_bits(v) for v in vv])
    return bytes(result)


def encode_reference(keys,values):
    if not keys or len(keys)!=len(values): raise ValueError('Source page count mismatch')
    if any(len(page)!=16 for page in (*keys,*values)): raise ValueError('Full PAGE16 source required')
    ks=[row for page in keys for row in page];vs=[row for page in values for row in page]
    # Allocate destination first; source rows and existing BF16 caches are untouched.
    return append_reference(bytes([0xa5])*(len(keys)*PAGE_BYTES),list(range(len(ks))),ks,vs)


def attention_reference(query,key_tokens,value_tokens,quantized=False):
    """Closed FP64 attention from original CPU BF16, not device cache/stream recurrence."""
    if len(query)!=16 or len(key_tokens)!=len(value_tokens): raise ValueError('Geometry mismatch')
    if not key_tokens: return [[0.]*128 for _ in range(16)]
    result=[]
    for head,q in enumerate(query):
        kv=head//8;validate_vector(q)
        qc,qs=quantize_v2(q) if quantized else (None,None)
        scores=[]
        for keys in key_tokens:
            k=validate_vector(keys[kv])
            if quantized:
                kc,ks=quantize_v2(k)
                raw=sum(a*b for a,b in zip(qc,kc))
                assert abs(raw)<2**24
                # This path deliberately does not reuse device FP32 scale operations.
                score=math.fsum((float(a)*qs)*(float(b)*ks) for a,b in zip(qc,kc))
                absolute_products=math.fsum(abs((float(a)*qs)*(float(b)*ks)) for a,b in zip(qc,kc))
                assert abs(score-raw*qs*ks)<=2**-48*absolute_products+2**-1074
            else: score=math.fsum(a*b for a,b in zip(q,k))
            scores.append(score*128**-.5)
        anchor=max(scores);weights=[math.exp(x-anchor) for x in scores];den=math.fsum(weights)
        result.append([math.fsum(w*v[kv][d] for w,v in zip(weights,value_tokens))/den for d in range(128)])
    return result


def numeric_policy(baseline_passed,hard_checks,candidate_passed,diagnostic=False):
    if not baseline_passed or not hard_checks: raise ValueError('Baseline/memory/codec/Graph hard failure cannot time')
    if not candidate_passed and not diagnostic: raise ValueError('Candidate numeric gate failed; timing disabled')
    return dict(precision_gate=False,diagnostic_only=bool(diagnostic),numeric_passed=bool(candidate_passed),
                timing_rounds=1 if diagnostic else 3)


def cost_plan(mode):
    if mode not in ('inline','preq'): raise ValueError('Unknown quantization position')
    return dict(mode=mode,initial_encoder='separately_recorded_once',
                baseline_step=['BF16_KV_append','BF16_GQA_producer','original_reducer4'],
                candidate_step=(['Qprep'] if mode=='preq' else [])+['Kcode_Kscale_and_original_BF16_V_append',
                                'I8_QK_GQA_producer_with_inline_Q8' if mode=='inline' else 'I8_QK_GQA_producer',
                                'original_reducer4'],
                producer_warps=1,reducer_warps=4,split_policy='actual_base._pick_segments',
                first_cases=[dict(batch=64,kv=4096),dict(batch=32,kv=16384)],
                initial_encode_source_destination_separate=True,physical_page_bytes=16384,
                default_runs_device=False,capacity_reclaimed=False,service_installed=False)
