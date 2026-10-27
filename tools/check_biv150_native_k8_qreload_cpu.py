#!/usr/bin/env python3
"""Source contract + finite BF16 logical oracle, never a GPU arithmetic simulator."""
import json
import math
from pathlib import Path
import random
import struct
import sys

import biv150_native_k8_qreload_factory as f
import biv150_native_k8_codec_fixture as vectors
from biv150_konly_decode_quant_reference import quantize_v2,rn32_div
from biv150_native_k8_qk_cpu import fragment_a


def address(seq,h,lane,k,which,pitch,layer=256):
    g,col=divmod(lane,16)
    return layer+2*(seq*pitch+(h*8+2*g+which)*128+16*k+col)


def main():
    proof=f.verify();generated=f.build();name='biv150_native_k8_qreload_qk.cu';source=generated[name]
    rejects=[]
    mutations={
      'second_Q_offset':('q_head_stride+16*i+col','q_head_stride+16*i+col+1'),
      'peak_order':('peak0=fmaxf(peak0,fabsf(value0));peak1=fmaxf(peak1,fabsf(value1));','peak1=fmaxf(peak1,fabsf(value1));peak0=fmaxf(peak0,fabsf(value0));'),
      'shuffle_order':('mask=8;mask;mask>>=1','mask=4;mask;mask>>=1'),
      'codec_rounding':('cjj_k8_qreload_code_rne(value0,input_scale0)','static_cast<int>(value0/input_scale0)'),
      'PV_C_chain':('acc[n][2]=acc[n][2]*rho0;','acc[n][2]=0.0f;'),
      'segments':('Segments==2 || Segments==4','Segments==1 || Segments==4'),
      'wait':('__ivcorex_sl_waitcnt(8ULL)','__ivcorex_sl_waitcnt(0ULL)'),
      'issue_order':('issue_v(dst,descriptor(value,page,head));','issue_v(dst,descriptor(value,page,head));__syncthreads();')}
    for label,(old,new) in mutations.items():
        assert old in source
        bad=dict(generated);bad[name]=source.replace(old,new,1)
        try:f.inverse(bad)
        except ValueError:rejects.append(label)
        else:raise AssertionError('Accepted mutant '+label)
    geometries=[];coordinates=0
    for rows,segments in ((64,2),(32,4)):
        for pitch in (2048,2560):
            for seq in (0,rows-1):
                for h in (0,1):
                    first=[address(seq,h,lane,k,w,pitch) for lane in range(64) for k in range(8) for w in (0,1)]
                    second=[address(seq,h,lane,2*k+component,w,pitch) for lane in range(64) for k in range(4) for component in range(2) for w in (0,1)]
                    assert first==second and len(set(first))==1024
                    expected={256+2*(seq*pitch+head*128+d) for head in range(h*8,h*8+8) for d in range(128)}
                    assert set(first)==expected;coordinates+=len(first)
            geometries.append(dict(rows=rows,segments=segments,Q_pitch=pitch,extra_requested_Q_bytes=rows*2*segments*2048))
    streams=vectors.fixtures();fixtures=[row for stream in ("A","B","C") for row in streams[stream]];codec_vectors=0;packed_words=0
    for start in range(len(fixtures)):
        rows=[[vectors.bf16_value(x) for x in fixtures[(start+h)%len(fixtures)]] for h in range(8)]
        assert all(math.isfinite(v) for row in rows for v in row)
        cached=[[[rows[2*(lane//16)+w][16*k+lane%16] for k in range(8)] for w in (0,1)] for lane in range(64)]
        peaks=[[0.0,0.0] for _ in range(64)]
        for k in range(8):
            for lane in range(64):
                for w in (0,1):peaks[lane][w]=max(peaks[lane][w],abs(cached[lane][w][k]))
        for mask in (8,4,2,1):peaks=[[max(peaks[lane][w],peaks[lane^mask][w]) for w in (0,1)] for lane in range(64)]
        codes=[quantize_v2(row)[0] for row in rows];scales=[quantize_v2(row)[1] for row in rows]
        for lane in range(64):
            g,col=divmod(lane,16)
            for w in (0,1):
                scale=1.0 if peaks[lane][w]==0 else max(rn32_div(peaks[lane][w],127.0),2.**-126)
                assert struct.pack('<f',scale)==struct.pack('<f',scales[2*g+w])
                for k in range(4):
                    for component in (0,1):
                        i=2*k+component;reloaded=rows[2*g+w][16*i+col]
                        assert struct.pack('<f',reloaded)==struct.pack('<f',cached[lane][w][i])
                        code=max(-127,min(127,round(rn32_div(reloaded,scale))))
                        assert code==codes[2*g+w][16*i+col]
            for k in range(4):
                for component in (0,1):
                    # Independent quartet packing from the same exact row bytes.
                    d=32*k+16*component+col;groups=((2*g)%4,(2*g+1)%4)
                    values=[codes[2*s+w][d] for s in groups for w in (0,1)]
                    manual=sum((v&255)<<(8*j) for j,v in enumerate(values))
                    assert manual==fragment_a(codes,lane,k,component);packed_words+=1
        codec_vectors+=8
    empty=[]
    for s in (2,4):
        for length in (0,1,15,16,17,31,32,33,4095,4096,16383,16384):
            steps=(length+s*16-1)//(s*16)
            entered=[segment for segment in range(s) if segment*steps*16<length]
            empty.append(dict(segments=s,length=length,entered=entered,extra_requested_Q_bytes_per_head=len(entered)*2048))
    assert not {'torch','triton','numpy','vllm'}.intersection(sys.modules)
    report=dict(passed=True,CPU_only=True,source=proof,geometries=geometries,address_coordinates=coordinates,
      original_finite_BF16_fixture_vectors=len(fixtures),lane_reduction_and_codec_rows=codec_vectors,
      packed_words_checked=packed_words,source_mutants_rejected=rejects,empty_and_tail_cases=empty,
      Q_readonly_precondition=True,finite_input_precondition=True,
      CPU_does_not_simulate_GPU_FTZ_rounding_or_register_allocation=True,
      actual_reload_or_resource_reduction_verified=False,GPU_numeric_verified=False,performance_verified=False,
      fullstep_future_reference='9695 pagewise BF16 P and 3eb full-final ideal gate; thresholds unchanged')
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=='__main__':main()
