"""CPU-only numeric policy and byte oracle for the complete private step runner."""
import math
import struct

from biv150_konly_decode_reference import valid_slots
from biv150_konly_i8_qk_reference import bf16_bits


def metrics(actual,reference):
    if not actual or len(actual)!=len(reference): raise ValueError('Nonempty equal output geometry required')
    finite=all(math.isfinite(v) for v in (*actual,*reference))
    if not finite: return dict(finite=False,pointwise_passed=False,relative_rmse=math.inf,max_abs=math.inf,rms=math.inf)
    errors=[a-b for a,b in zip(actual,reference)]
    rms=math.hypot(*errors)/math.sqrt(len(errors))
    reference_rms=math.hypot(*reference)/math.sqrt(len(reference))
    maximum=max(map(abs,errors))
    assert rms!=0 or maximum==0,'Contradictory CPU RMS/max metric'
    relative=rms/reference_rms if reference_rms else (0. if maximum==0 else math.inf)
    violations=sum(abs(a-b)>.005+.005*abs(b) for a,b in zip(actual,reference))
    return dict(finite=True,pointwise_passed=violations==0,pointwise_violations=violations,
                relative_rmse=relative,max_abs=maximum,rms=rms)


def reference_gates(baseline,candidate,original_reference,quantized_reference):
    base=metrics(baseline,original_reference)
    implementation=metrics(candidate,quantized_reference)
    original=metrics(candidate,original_reference)
    limit=max(.005,1.2*base['relative_rmse'])
    return dict(baseline=base,quantized_operand=implementation,original_bf16=original,
                baseline_hard_passed=base['finite'] and base['pointwise_passed'],
                candidate_numeric_passed=base['finite'] and base['pointwise_passed'] and implementation['finite'] and implementation['pointwise_passed'] and
                    original['finite'] and original['relative_rmse']<=limit,
                original_relative_rmse_limit=limit,thresholds_relaxed=False)


def timing_policy(hard_checks,numeric_passed,diagnostic):
    required={'finite','baseline_reference','codec','cache_bytes','sources_unchanged','scratch_canaries',
              'graph_eager','side_eager','actual_ir'}
    if set(hard_checks)!=required or not all(v is True for v in hard_checks.values()):
        raise ValueError('Hard failure forbids timing, including diagnostic timing')
    return dict(allowed=bool(numeric_passed or diagnostic),numeric_passed=bool(numeric_passed),
                diagnostic_only=bool(diagnostic),timing_rounds=1 if diagnostic else 3,precision_gate=False)


def append_bf16_reference(before,slots,keys,values):
    if len(before)%16384 or len(slots)!=len(keys) or len(keys)!=len(values): raise ValueError('BF16 writer geometry mismatch')
    pages=len(before)//16384;valid_slots(slots,pages);result=bytearray(before)
    for token,s in enumerate(slots):
        if not 0<=s<pages*16: continue
        page,column=divmod(s,16)
        for head in range(2):
            for plane,vectors in ((0,keys),(8192,values)):
                offset=page*16384+plane+column*512+head*256
                struct.pack_into('<128H',result,offset,*[bf16_bits(v) for v in vectors[token][head]])
    return bytes(result)


def codec_vectors():
    # Finite BF16 zeros, half boundaries, exponent extremes and mixed signs.
    from biv150_konly_i8_qk_reference import bf16
    import random
    rows=[[0.]*128,[-0.]*128,[2.**-133]*128,[-2.**-133]*128,
          [struct.unpack('<f',struct.pack('<I',0x7f7f0000))[0]]*128]
    for peak in range(128,256):
        rows.append([float(peak),peak/2,-peak/2,-0.]+[bf16((-1)**d*peak*((d%31)/32)) for d in range(124)])
    rng=random.Random(20261027)
    for exponent in (-133,-126,-100,-10,0,10,100,120):
        rows.append([bf16(rng.uniform(-1,1)*2.**exponent) for _ in range(128)])
    return rows
