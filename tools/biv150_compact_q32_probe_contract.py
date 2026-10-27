"""Bounded compact Q32 test contract; stdlib plan, no default array/device import."""
import hashlib
import math
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[1]
BASE='deab3ab474925c08dba48c4c8cdb00bf108b3b82'
Q=1024
LENGTHS=(1057,1056,1058)
PAGES=139
GUARD=256
OUT_GUARD_ROWS=32
ROLES=('unhoisted','hoisted')
FORMAT='cjj-konly-compact12544-rn32-rne-v1'
PINS=dict(sdk='54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f',
 helpers='59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1',
 jit='66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775')


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(value):return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()


def plan():
    return dict(default='stdlib CPU plan; Root-only explicit --compile-only or --gpu',
        format=FORMAT,query_rows=Q,lengths=list(LENGTHS),prefixes=[x-Q for x in LENGTHS],
        sequences=1,pages=PAGES,page_bytes=12544,layer_offset_bytes=GUARD,grid=[Q//32+1,2],
        geometry=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,num_warps=16,num_stages=1),
        input='full-D128 nonperiodic counter BF16; shared canonical A/B/C for both arms',
        FP64_reference='all1024 rows: independent INT64 quantized QK -> FP64 scales/softmax/PV; original BF16 separate',
        pointwise_atol=.005,pointwise_rtol=.005,original_BF16_RRMSE_proxy=.005,
        partial='not_exposed_by_2D_Q32',maximum='not_exposed_by_2D_Q32',summation='not_exposed_by_2D_Q32',
        Graph_actual_path='both arms bypass to unhoisted normalized2D128, compared to same-path fresh eagerB',
        Graph_hoisted_qualification=False,side_actual_path='Q32 unhoisted vs fresh hoisted',
        changed_Graph_replay_before_fresh_eager_B=True,full_output_cache_guards_readonly=True,
        writer_launches=0,encoder_launches=0,performance_measured=False,precision_gate=False,
        service_verified=False,mixed_or_42layer_capacity_verified=False,GPU_executed=False)


def policy(hard,quality,diagnostic):
    required={'finite','quantized_FP64_pointwise','pair_bitwise','full_output_guards','whole_cache',
              'readonly','fresh_actual_IR_native_resources','actual_path_bound'}
    need(set(hard)==required and all(type(v) is bool and v for v in hard.values()),'hard component gate failed')
    need(type(quality) is bool and type(diagnostic) is bool,'Boolean proxy/diagnostic flags')
    need(quality or diagnostic,'original BF16 proxy failed; explicit --numeric-diagnostic required')
    return dict(hard_gates_passed=True,original_BF16_proxy_passed=quality,numeric_diagnostic=diagnostic,
                precision_gate=False,model_verified=False,service_verified=False,thresholds_changed=False)


def counter_bits(row,head,dim,phase,kind):
    # Integer-only canonical generator, no RNG/16-D repetition. Head exponents
    # differ independently, so I8 packing/head-scale mistakes cannot cancel.
    x=((row+1)*0x9e3779b1+(head+3)*0x85ebca77+(dim+7)*0xc2b2ae3d+
       (phase+11)*0x27d4eb2f+(kind+19)*0x165667b1)&0xffffffff
    x^=x>>16;x=(x*0x7feb352d)&0xffffffff;x^=x>>15
    exponent=(125+(head%3)) if kind==0 else (126+head)
    return ((x>>31)<<15)|(exponent<<7)|(x&127)


def canonical(phase,np):
    length=LENGTHS[phase]
    def bits(rows,heads,kind):
        r=np.arange(rows,dtype=np.uint64)[:,None,None];h=np.arange(heads,dtype=np.uint64)[None,:,None]
        d=np.arange(128,dtype=np.uint64)[None,None,:]
        x=((r+1)*0x9e3779b1+(h+3)*0x85ebca77+(d+7)*0xc2b2ae3d+
           (phase+11)*0x27d4eb2f+(kind+19)*0x165667b1)&0xffffffff
        x=x^(x>>16);x=(x*0x7feb352d)&0xffffffff;x=x^(x>>15)
        exponent=125+h%3 if kind==0 else 126+h
        return (((x>>31)<<15)|(exponent<<7)|(x&127)).astype(np.uint16)
    table=np.full((1,8192),PAGES-1,dtype=np.int32)
    ids=[((i*37+phase*19+7)%(PAGES-1)) for i in range((length+15)//16)]
    need(len(ids)==len(set(ids)),'unique scattered physical pages')
    table[0,:len(ids)]=ids
    return dict(q_bits=bits(Q,16,0),k_bits=bits(length,2,1),v_bits=bits(length,2,2),
                table=table,lengths=np.array([length],dtype=np.int32),cu_q=np.array([0,Q],dtype=np.int32))


def decode(bits,np):
    return (bits.astype(np.uint32)<<16).view(np.float32).astype(np.float64)


def quantize(bits,np):
    # Same independent exact RN32 boundary repair as the old encoded probe.
    sys.path.insert(0,str(ROOT/'tools/compact_reference'))
    from biv150_konly_decode_quant_reference import rn32_div
    shape=bits.shape;wide=decode(bits,np).reshape(-1,128);peak=np.max(np.abs(wide),axis=1)
    scales=np.array([1. if v==0 else max(rn32_div(float(v),127.),2.**-126) for v in peak],dtype=np.float32)
    ratio=(wide/scales.astype(np.float64)[:,None]).astype(np.float32)
    codes=np.clip(np.rint(ratio),-127,127).astype(np.int8)
    mag=np.abs(ratio).astype(np.float64)
    near=np.abs(mag-np.floor(mag)-.5)<=2*np.spacing(np.abs(ratio)).astype(np.float64)
    repaired=0
    for row,dim in zip(*np.nonzero(near)):
        codes[row,dim]=max(-127,min(127,round(rn32_div(float(wide[row,dim]),float(scales[row])))))
        repaired+=1
    return codes.reshape(shape),scales.reshape(shape[:-1]),repaired


def packed(cpu,np):
    codes,scales,repaired=quantize(cpu['k_bits'],np)
    result=np.full((PAGES,12544),0xa5,dtype=np.uint8)
    for row in range(len(codes)):
        p=int(cpu['table'][0,row//16]);slot=row%16
        result[p,slot*256:(slot+1)*256]=codes[row].view(np.uint8).reshape(256)
        result[p,4096+slot*8:4096+(slot+1)*8]=scales[row].view(np.uint8).reshape(8)
        result[p,4352+slot*512:4352+(slot+1)*512]=cpu['v_bits'][row].view(np.uint8).reshape(512)
    return result,codes,scales,repaired


def references(cpu,kcodes,kscales,np):
    q=decode(cpu['q_bits'],np);k=decode(cpu['k_bits'],np);v=decode(cpu['v_bits'],np)
    qc,qs,repaired=quantize(cpu['q_bits'],np)
    original=np.empty((Q,16,128),dtype=np.float64);quant=np.empty_like(original)
    length=len(k);visible=length-Q+np.arange(Q)+1;max_integer=0
    for head in range(16):
        kh=head//8
        for start in range(0,Q,32):
            end=min(Q,start+32)
            integer=qc[start:end,head].astype(np.int64)@kcodes[:,kh].astype(np.int64).T
            max_integer=max(max_integer,int(np.max(np.abs(integer))))
            need(max_integer<=128*127**2,'independent INT64 QK bound')
            quant_score=integer.astype(np.float64)*qs[start:end,head,None].astype(np.float64)*kscales[None,:,kh].astype(np.float64)*128**-.5
            original_score=(q[start:end,head]@k[:,kh].T)*128**-.5
            mask=np.arange(length)[None,:]<visible[start:end,None]
            for score,out in ((quant_score,quant),(original_score,original)):
                score=np.where(mask,score,-np.inf);m=np.max(score,axis=1,keepdims=True)
                p=np.exp(score-m);p=p/np.sum(p,axis=1,keepdims=True)
                out[start:end,head]=p@v[:,kh]
    return dict(original_FP64=original,quantized_FP64=quant),dict(full_rows=Q,full_dimensions=128,
        independent_INT64_QK=True,max_abs_INT64_QK=max_integer,q_boundary_repairs=repaired,
        probability_BF16_or_online_rounding_modeled=False)


def metric(actual,reference,np):
    finite=bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:return dict(finite=False,pointwise_passed=False,relative_rmse=None)
    error=actual-reference;rms=float(np.sqrt(np.mean(error*error)));den=float(np.sqrt(np.mean(reference*reference)))
    maximum=float(np.max(np.abs(error)));rrms=rms/den if den else 0. if maximum==0 else None
    violations=int(np.count_nonzero(np.abs(error)>.005+.005*np.abs(reference)))
    return dict(finite=True,pointwise_passed=violations==0,pointwise_violations=violations,
                relative_rmse=rrms,max_abs=maximum,rms=rms)
