"""Deterministic full-D128 counter fixture; NumPy CPU INT64/FP64 golden only."""
import hashlib
import math
import struct
from pathlib import Path
from biv150_konly_decode_quant_reference import quantize_v2
from biv150_native_k8_codec_fixture import bf16_bits,bf16_value,LIBRARY_SHA

ROOT=Path(__file__).resolve().parents[1]
REDUCER_FILE=ROOT/'vllm_fl/patches/biv150_gqa_decode.py'
REDUCER_SHA='f1877ad06e8be4cef5960ef2d66d59045b4ac1ac0d7f2fe448618bbdec0d7f40'
PAGES=29;COLS=16;LAYER_OFFSET=128;CACHE_GUARD=128;GUARD_ELEMENTS=64
F32_GUARD=0x428e0000;F32_POISON=0x7fc12345;BF16_GUARD=0x4255;BF16_POISON=0x7fc1
BOUNDARIES=(0,1,15,16,17,31,32,33,63,64,65,127,128,129,191,192,193)
CASES=((64,2,2048),(64,2,2560),(32,4,2048),(32,4,2560))

def sha(raw):return hashlib.sha256(raw).hexdigest()
def counter(kind,phase,a,b,d):
    x=(0x6a09e667f3bcc909+kind*0x9e3779b97f4a7c15+phase*0xd1b54a32d192ed03+a*0x94d049bb133111eb+b*0xbf58476d1ce4e5b9+d*0x369dea0f31a53f85)&((1<<64)-1)
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&((1<<64)-1)
    x=((x^(x>>27))*0x94d049bb133111eb)&((1<<64)-1)
    return x^(x>>31)

def vector(kind,phase,index,head):
    if kind==0:
        exponent=head%4-2;values=[(counter(kind,phase,index,head,d)%61-30)*2.**(exponent-6) for d in range(128)]
        values[(head*7+index*3+phase*11)%128]=(65+head+phase*17)*2.**(exponent-6)
    elif kind==1:
        exponent=(index+head+phase)%4-2
        values=[(counter(kind,phase,index,head,d)%63-31)*2.**(exponent-6) for d in range(128)]
        peak=64+(index*3+head*7+phase*5)%32
        values[(index*3+head*11+phase*13)%128]=(-1 if (index+head+phase)%2 else 1)*peak*2.**(exponent-6)
    else:values=[(counter(kind,phase,index,head,d)%127-63)/64. for d in range(128)]
    bits=[bf16_bits(v) for v in values]
    assert all(bf16_value(b)==v for b,v in zip(bits,values))
    assert values[:32]!=values[32:64] and values[:64]!=values[64:]
    return bits

def fixture(rows,segments,pitch,phase):
    assert (rows,segments,pitch) in CASES and phase in (0,1,2)
    query=[BF16_GUARD]*(2*GUARD_ELEMENTS+rows*pitch)
    qcodes=[];qscales=[]
    for row in range(rows):
        rc=[];rs=[]
        for head in range(16):
            bits=vector(0,phase,row,head);start=GUARD_ELEMENTS+row*pitch+head*128
            query[start:start+128]=bits;codes,scale=quantize_v2([bf16_value(b) for b in bits]);rc.append(codes);rs.append(scale)
        assert len(set(rs))==16
        qcodes.append(rc);qscales.append(rs)
    cache=bytearray(b'\x5a'*(LAYER_OFFSET+PAGES*16384+CACHE_GUARD));cache[LAYER_OFFSET:-CACHE_GUARD]=b'\xa5'*(PAGES*16384)
    keys=[];values=[];key_codes=[];key_scales=[]
    for page in range(PAGES):
        pk=[];pv=[];pc=[];ps=[]
        for slot in range(16):
            sk=[];sv=[];sc=[];ss=[]
            for head in range(2):
                index=page*16+slot;kb=vector(1,phase,index,head);vb=vector(2,phase,index,head)
                codes,scale=quantize_v2([bf16_value(b) for b in kb]);base=LAYER_OFFSET+page*16384
                cache[base+slot*256+head*128:base+slot*256+(head+1)*128]=bytes(v&255 for v in codes)
                struct.pack_into('<f',cache,base+4096+slot*8+head*4,scale)
                struct.pack_into('<128H',cache,base+8192+slot*512+head*256,*vb)
                sk.append(kb);sv.append(vb);sc.append(codes);ss.append(scale)
            pk.append(sk);pv.append(sv);pc.append(sc);ps.append(ss)
        keys.append(pk);values.append(pv);key_codes.append(pc);key_scales.append(ps)
    lengths=[BOUNDARIES[(row+phase*5)%len(BOUNDARIES)] for row in range(rows)]
    table=[-1]*(2*GUARD_ELEMENTS+rows*COLS)
    for row,length in enumerate(lengths):
        for col in range((length+15)//16):table[GUARD_ELEMENTS+row*COLS+col]=(row*3+col*7+phase*5)%PAGES
    return dict(rows=rows,segments=segments,pitch=pitch,phase=phase,query_bits=query,cache=bytes(cache),
      lengths=[-1234567]*GUARD_ELEMENTS+lengths+[-1234567]*GUARD_ELEMENTS,
      table=table,qcodes=qcodes,qscales=qscales,key_bits=keys,value_bits=values,kcodes=key_codes,kscales=key_scales)

def input_bytes(case):
    return dict(query=struct.pack('<'+str(len(case['query_bits']))+'H',*case['query_bits']),cache=case['cache'],
      lengths=struct.pack('<'+str(len(case['lengths']))+'i',*case['lengths']),
      table=struct.pack('<'+str(len(case['table']))+'i',*case['table']))

def golden(case):
    import numpy as np
    n,s=case['rows'],case['segments'];lens=case['lengths'][GUARD_ELEMENTS:-GUARD_ELEMENTS]
    q=np.array(case['qcodes'],dtype=np.int64);qs=np.array(case['qscales'],dtype=np.float64)
    k=np.array(case['kcodes'],dtype=np.int64);ks=np.array(case['kscales'],dtype=np.float64)
    v=np.array([bf16_value(b) for page in case['value_bits'] for slot in page for head in slot for b in head],dtype=np.float64).reshape(PAGES,16,2,128)
    kb=np.array([bf16_value(b) for page in case['key_bits'] for slot in page for head in slot for b in head],dtype=np.float64).reshape(PAGES,16,2,128)
    raw_q=np.array([bf16_value(case['query_bits'][GUARD_ELEMENTS+row*case['pitch']+head*128+d]) for row in range(n) for head in range(16) for d in range(128)],dtype=np.float64).reshape(n,16,128)
    out=np.zeros((n,16,128),dtype=np.float64);original=np.zeros_like(out)
    partial=np.full((n,16,s,128),np.nan,dtype=np.float64);maximum=np.full((n,16,s),np.nan,dtype=np.float64);summation=np.full_like(maximum,np.nan)
    active=np.zeros((n,s),dtype=bool);integer_checks=0
    for row,length in enumerate(lens):
        if not length:continue
        table=case['table'][GUARD_ELEMENTS+row*COLS:GUARD_ELEMENTS+(row+1)*COLS]
        ids=[table[t//16] for t in range(length)];slots=np.arange(length)%16
        kr=k[ids,slots];sr=ks[ids,slots];vr=v[ids,slots];br=kb[ids,slots]
        span=((length+s*16-1)//(s*16))*16
        for head in range(16):
            kh=head//8
            # Every one of the128 dimensions contributes to actual INT64 matmul.
            dots=kr[:,kh,:]@q[row,head,:]
            assert np.abs(dots).max()<=128*127**2
            integer_checks+=length*128
            score=dots.astype(np.float64)*qs[row,head]*sr[:,kh]*(128.**-.5)
            old_score=br[:,kh,:]@raw_q[row,head,:]*(128.**-.5)
            probability=np.exp(score-score.max());out[row,head]=probability@vr[:,kh,:]/probability.sum()
            old_p=np.exp(old_score-old_score.max());original[row,head]=old_p@vr[:,kh,:]/old_p.sum()
            for seg in range(s):
                lower=seg*span;upper=min(lower+span,length)
                if lower>=length:continue
                active[row,seg]=True;m=float(score[lower:upper].max());p=np.exp(score[lower:upper]-m)
                maximum[row,head,seg]=m;summation[row,head,seg]=p.sum();partial[row,head,seg]=p@vr[lower:upper,kh,:]
    return dict(output=out,original_BF16_output=original,partial=partial,maximum=maximum,summation=summation,
      active=active,full_D128_integer_products=integer_checks)

def metrics(actual,expected):
    import numpy as np
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    assert actual.shape==expected.shape
    diff=actual-expected;finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    pointwise=bool(finite and (np.abs(diff)<=.005+.005*np.abs(expected)).all())
    rrmse=float(np.sqrt(np.sum(diff*diff)/max(float(np.sum(expected*expected)),1e-30))) if finite else None
    return dict(finite=finite,pointwise_005=pointwise,relative_RMSE=rrmse,relative_RMSE_005=bool(finite and rrmse<=.005),
      maximum_absolute_error=(float(np.max(np.abs(diff))) if diff.size else 0.) if finite else None,passed=pointwise and finite and rrmse<=.005)

def plan():
    assert sha(REDUCER_FILE.read_bytes())==REDUCER_SHA
    return dict(component='native K8 consumer plus original b3 reducer only',library_sha256=LIBRARY_SHA,
      reducer_file='vllm_fl/patches/biv150_gqa_decode.py',reducer_module_sha256=REDUCER_SHA,reducer='_reduce_partials',
      cases=[list(c) for c in CASES],cache_pages=PAGES,maximum_context=193,table_columns=COLS,
      lengths=list(BOUNDARIES),layer_byte_offset=LAYER_OFFSET,page_bytes=16384,V_offset_from_layer=8192,
      scale_full_base=True,full128_nonperiodic_counter=True,all_query_head_scales_distinct=True,
      golden='CPU NumPy full128-D INT64 dot then dequantized FP64 softmax/PV, complete arrays',
      hard_gate='finite + every absdiff<=.005+.005*abs(reference) + relative RMSE<=.005',
      original_BF16_proxy_separate=True,codec_actual_pass_required_before_Torch=True,
      phases=['eager A','capture A','changed Q/cache/table/length B + poison then graph before new eager B','side C'],
      native_compile=False,encoder_launches=0,timing=None,default_CPU=True,consumer_numeric_verified=False,
      Graph_verified=False,side_stream_verified=False,model_verified=False,precision_gate=False,service_installed=False)
