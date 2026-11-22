"""Bounded CPU counter construction/byte oracle for the fixed full-step pair."""
import hashlib
import json
import math
import struct
from pathlib import Path
from biv150_konly_decode_quant_reference import quantize_v2
from biv150_native_k8_consumer_fixture import ROOT,vector,bf16_value,metrics,REDUCER_SHA
from biv150_native_k8_codec_fixture import LIBRARY_SHA
from biv150_native_tl_pair_contract import table as fixed_table,append_slots,pins as main_pins

BF16_SHA='aace693bdb302d2685ec1b5e11e859c8a4a77d7f45c0f4dacddc6b5e49b015b8'
PAGES=34276;TABLE_COLS=8192;GUARD=256;BUILDER_SEGMENTS=16
CASES={'4k':(64,4096,2,2048),'16k':(32,16384,4,2560)}
ORDER=('bf16','k8','k8','bf16');UNROLL=8;WARM=128;REPLAYS=64
WRITER_SHA='16afc46585f950f4b9c06c831b0e0dbf5515ee941a476a9ed300c492ebf947f9'
CONTRACT_SHA='cdc1384ff4443bd577759592fa9aa3ea0ff79032507c7d6e63f4abc8db2569b9'

def sha(raw):return hashlib.sha256(raw).hexdigest()

def pins():
    from biv150_native_k8_source_factory import verify
    assert verify()['inverse_byte_exact']
    result=main_pins()
    for name,expected in [('tools/kernels/biv150_konly_strided_writer.py',WRITER_SHA),('tools/biv150_konly_strided_writer_contract.py',CONTRACT_SHA)]:
        assert sha((ROOT/name).read_bytes())==expected;result[name]=expected
    native_pins={'tools/kernels/biv150_native_k8_codec.cu': '7b744ba7112083add31ccb49c35055d4b573d718df4343ccf9d10ab998abaad2', 'tools/kernels/biv150_native_k8_codec.cuh': '37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79', 'tools/kernels/biv150_native_k8_qk.cpp': 'bc6d9d3c289d49435f7598e6b5c2325cfc3f867ab915ab424ffb289bffb20b29', 'tools/kernels/biv150_native_k8_qk.cu': 'bf84c82ff7e73684c3923ca8944c6564f37c21fb636134e106d0542f5ba47cd9'}
    for name,expected in native_pins.items():assert sha((ROOT/name).read_bytes())==expected;result[name]=expected
    return result

def plan():
    return dict(component='fixed arena complete native BF16 vs native K8 step; not service',
      main='4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b',K8_library_sha256=LIBRARY_SHA,BF16_library_sha256=BF16_SHA,
      source_pins=pins(),cases=CASES,pages=PAGES,page_bytes=16384,arena_bytes=PAGES*16384,table_columns=TABLE_COLS,
      physical_permutation_seed=20261028,nonoverlap=True,query_nonperiodic_D128=True,
      new_KV_pitch=256,service_source_pitch2560_covered=False,
      K8_append='exact PR83 with actual separate pitches/row extents/slot_count',BF16_append='exact94 ENCODE=False',
      initial_K8_encode='exact94 encode_initial: V already offset',append_K8_V='FULL page0 base, offset added internally',
      fullstep=['append','native producer','same original b3 stride-aware reducer'],
      initial_CPU_counter_and_encode_oracle_and_GPU_pack_costs_separate=True,
      whole_cache_streaming_byte_equality=True,full_cache_saved_as_chunk_hashes_with_reconstructible_counter=True,
      OML_and_final_output_hard_gate='all finite; absdiff<=.005+.005*abs(ref); RRMSE<=.005',
      partial_reference='FP64 accumulator over source-defined FP32 score/rho and page16 BF16 P operands; ideal partial diagnostic separate',
      final_reference='unchanged ideal FP64, original BF16 or quantized-score by arm',
      scratch_builder_width=16,scratch_head_stride=16*128,maxima_head_stride=16,
      references_per_arm=dict(bf16='original BF16 FP64',k8='full-D128 I64 QK then dequant FP64'),
      inactive_poison_and_full_guards=True,Graph_changed_before_eager=True,side_stream=True,
      ABBA_order=ORDER,unroll=UNROLL,warm_replays=WARM,timing_replays=REPLAYS,
      clocks=['GPU event per append+producer+reducer step','Graph launch+sync host wall per step'],
      excludes=['QKV projection','RoPE','framework dispatch/metadata','service/model'],
      mandatory_actual_consumer_pass_before_Torch=True,actual_consumer_sha256='c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128',default_CPU=True,native_compile=False,
      numerical_verified=False,timing_executed=False,model_verified=False,precision_gate=False)

def chunk_sources(start,count):
    import numpy as np
    index=np.arange(start*16,(start+count)*16,dtype=np.uint64)[:,None,None]
    head=np.arange(2,dtype=np.uint64)[None,:,None];dim=np.arange(128,dtype=np.uint64)[None,None,:]
    def mixed(kind):
        with np.errstate(over='ignore'):
            x=np.uint64((0x6a09e667f3bcc909+kind*0x9e3779b97f4a7c15)&((1<<64)-1))+index*np.uint64(0x94d049bb133111eb)+head*np.uint64(0xbf58476d1ce4e5b9)+dim*np.uint64(0x369dea0f31a53f85)
            x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
            return x^(x>>np.uint64(31))
    basis=(mixed(1)%np.uint64(63)).astype(np.int32)-31
    ii=index[:,:,0].astype(np.int64);hh=head[:,:,0].astype(np.int64)
    exponent=(ii+hh)%4-2;peak=64+(ii*3+hh*7)%32;peakdim=(ii*3+hh*11)%128
    values=basis.astype(np.float32)*np.exp2(exponent-6).astype(np.float32)[:,:,None]
    r=np.arange(count*16)[:,None];h=np.arange(2)[None,:]
    values[r,h,peakdim]=np.where((ii+hh)%2,-1,1)*peak*np.exp2(exponent-6)
    v=((mixed(2)%np.uint64(127)).astype(np.int32)-63).astype(np.float32)/64.
    kbits=(values.view(np.uint32)>>16).astype('<u2');vbits=(v.view(np.uint32)>>16).astype('<u2')
    cold=np.empty((count,2,16,2,128),dtype='<u2');cold[:,0]=kbits.reshape(count,16,2,128);cold[:,1]=vbits.reshape(count,16,2,128)
    return cold,basis,peak,exponent,peakdim

def codebook():
    import numpy as np
    codes=np.empty((32,4,63),dtype=np.int8);scales=np.empty((32,4),dtype='<f4')
    for peak in range(64,96):
        for exp in range(-2,2):
            values=[peak*2.**(exp-6)]+[v*2.**(exp-6) for v in range(-31,32)]+[0.]*64
            q,s=quantize_v2(values);codes[peak-64,exp+2]=q[1:64];scales[peak-64,exp+2]=s
    return codes,scales

def encode_chunk(start,cold,basis,peak,exponent,peakdim,book):
    import numpy as np
    count=len(cold);codes_table,scale_table=book
    codes=codes_table[(peak-64)[:,:,None],(exponent+2)[:,:,None],basis+31].copy()
    rows=np.arange(count*16)[:,None];head=np.arange(2)[None,:];indices=np.arange(start*16,(start+count)*16)[:,None]
    codes[rows,head,peakdim]=np.where((indices+head)%2,-127,127)
    scales=scale_table[peak-64,exponent+2]
    packed=np.full((count,16384),0xa5,dtype=np.uint8);packed[:,:4096]=codes.reshape(count,4096).view(np.uint8)
    packed[:,4096:4224]=scales.reshape(count,32).view(np.uint8).reshape(count,128)
    packed[:,8192:]=cold[:,1].reshape(count,4096).view(np.uint8).reshape(count,8192)
    return packed

def build_initial(pages=PAGES,chunk=128):
    import numpy as np
    import time
    cold=np.empty((pages,2,16,2,128),dtype='<u2');packed=np.empty((pages,16384),dtype=np.uint8);book=codebook()
    source_s=oracle_s=0.
    for start in range(0,pages,chunk):
        count=min(chunk,pages-start);t=time.monotonic();part,*aux=chunk_sources(start,count);cold[start:start+count]=part;source_s+=time.monotonic()-t
        t=time.monotonic();packed[start:start+count]=encode_chunk(start,part,*aux,book);oracle_s+=time.monotonic()-t
    return cold,packed,dict(CPU_BF16_counter_seconds=source_s,CPU_exact_encoded_oracle_seconds=oracle_s,all_initial_CPU_costs_separate=True)

def control(n,length,pitch,phase):
    import numpy as np
    mapping=fixed_table(n,length)
    if phase: mapping=[row[phase:]+row[:phase] for row in mapping]
    lens=[length]*n
    if phase:
        for row,value in enumerate((0,1,15,16,17,31,32,33,63,64,65,length-1)):lens[(row+phase-1)%n]=value
    table=np.full((n,TABLE_COLS),-1,dtype='<i4');table[:,:len(mapping[0])]=mapping
    q=np.full((n,pitch),0x4255,dtype='<u2')
    for row in range(n):
        for head in range(16):q[row,head*128:(head+1)*128]=vector(0,phase,row,head)
    nk=np.array([[vector(1,phase+4,0x100000+row,h) for h in range(2)] for row in range(n)],dtype='<u2')
    nv=np.array([[vector(2,phase+7,0x200000+row,h) for h in range(2)] for row in range(n)],dtype='<u2')
    return dict(query=q,table=table,lengths=np.array(lens,dtype='<i4'),slots=np.array(append_slots(mapping,lens),dtype='<i8'),new_k=nk,new_v=nv)

def patched_chunk(arm,start,initial,control):
    import numpy as np
    raw=initial.view(np.uint8).reshape(len(initial),16384).copy() if arm=='bf16' else initial.copy()
    # Caller passes just the chunk; physical start distinguishes slot coordinates.
    for row,physical in enumerate(control['slots']):
        if physical<0:continue
        page,slot=divmod(int(physical),16)
        if not start<=page<start+len(raw):continue
        p=page-start
        for head in range(2):
            kb=control['new_k'][row,head];vb=control['new_v'][row,head]
            if arm=='bf16':raw[p,slot*512+head*256:slot*512+(head+1)*256]=kb.view(np.uint8)
            else:
                codes,scale=quantize_v2([bf16_value(int(b)) for b in kb]);raw[p,slot*256+head*128:slot*256+(head+1)*128]=np.array(codes,dtype=np.int8).view(np.uint8)
                raw[p,4096+slot*8+head*4:4100+slot*8+head*4]=np.frombuffer(struct.pack('<f',scale),dtype=np.uint8)
            raw[p,8192+slot*512+head*256:8192+slot*512+(head+1)*256]=vb.view(np.uint8)
    return raw

def golden(cold,packed,control,segments,arm):
    import numpy as np
    n=len(control['lengths']);out=np.zeros((n,16,128));part=np.full((n,16,segments,128),np.nan);ideal_part=np.full_like(part,np.nan);maximum=np.full((n,16,segments),np.nan);summation=np.full_like(maximum,np.nan);active=np.zeros((n,segments),bool)
    def wide(bits):return (np.asarray(bits,dtype='<u2').astype(np.uint32)<<16).view('<f4').astype(np.float64)
    q=wide(control['query'][:,:2048].reshape(n,16,128))
    for row,L in enumerate(control['lengths']):
        length=int(L)
        if not length:continue
        ids=control['table'][row,np.arange(length)//16];slots=np.arange(length)%16
        kb=cold[ids,0,slots].copy();vb=cold[ids,1,slots].copy();kb[-1]=control['new_k'][row];vb[-1]=control['new_v'][row]
        vv=wide(vb);kk=wide(kb);span=((length+segments*16-1)//(segments*16))*16
        for kh in range(2):
            if arm=='bf16':
                dot64=q[row,kh*8:kh*8+8]@kk[:,kh].T
                assert np.array_equal(dot64,dot64.astype(np.float32).astype(np.float64)),'counter-family BF16 dot must be exact F32'
                scores=dot64*(128.**-.5)
                score_log2=(dot64.astype(np.float32)*np.float32(128**-.5*1.4426950408889634)).astype(np.float64)
            else:
                key=np.stack([packed[int(p),int(s)*256+kh*128:int(s)*256+(kh+1)*128].view(np.int8) for p,s in zip(ids,slots)])
                scales=np.array([struct.unpack_from('<f',packed[int(p)],4096+int(s)*8+kh*4)[0] for p,s in zip(ids,slots)])
                key[-1],scales[-1]=quantize_v2([float(x) for x in kk[-1,kh]])
                qc=[];qs=[]
                for head in range(kh*8,kh*8+8):codes,scale=quantize_v2(q[row,head].tolist());qc.append(codes);qs.append(scale)
                integer=np.array(qc,dtype=np.int64)@key.astype(np.int64).T
                scores=integer.astype(np.float64)*np.array(qs)[:,None]*scales[None,:]*(128.**-.5)
                typed=integer.astype(np.float32)*np.array(qs,dtype=np.float32)[:,None]
                typed=typed*scales.astype(np.float32)[None,:]
                score_log2=(typed*np.float32(128**-.5*1.4426950408889634)).astype(np.float64)
            prob=np.exp(scores-scores.max(axis=1,keepdims=True));out[row,kh*8:kh*8+8]=(prob@vv[:,kh])/prob.sum(axis=1,keepdims=True)
            for seg in range(segments):
                lower=seg*span;upper=min(lower+span,length)
                if lower>=length:continue
                active[row,seg]=True;m=scores[:,lower:upper].max(axis=1);p=np.exp(scores[:,lower:upper]-m[:,None])
                maximum[row,kh*8:kh*8+8,seg]=m;summation[row,kh*8:kh*8+8,seg]=p.sum(axis=1);ideal_part[row,kh*8:kh*8+8,seg]=p@vv[lower:upper,kh]
                from biv150_native_k8_pagewise_reference import round_probability_bf16
                previous=np.full(8,-np.inf);acc=np.zeros((8,128))
                for page in range(lower,upper,16):
                    end=min(page+16,upper);new_m=np.maximum(previous,score_log2[:,page:end].max(axis=1));rho=np.exp2((previous-new_m).astype(np.float32).astype(np.float64)).astype(np.float32).astype(np.float64)
                    probability=np.exp2((score_log2[:,page:end]-new_m[:,None]).astype(np.float32).astype(np.float64));acc=acc*rho[:,None]+round_probability_bf16(probability)@vv[page:end,kh];previous=new_m
                part[row,kh*8:kh*8+8,seg]=acc
    return dict(output=out,partial=part,ideal_partial_diagnostic=ideal_part,maximum=maximum,summation=summation,active=active)
