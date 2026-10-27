"""Independent FP64 online-page reference with original b3 BF16 P operand.

Full128-D INT64 QK then dequantized FP64 scores. Each16-token page first
updates max/rho, casts its own P through RN32 to BF16, rescales the prior
FP64 accumulator and adds BF16(P)*original BF16(V). L stays unrounded.
This is not final-max once-only rounding, and does not imitate target ASM.
"""
import numpy as np
from biv150_konly_decode_quant_reference import quantize_v2


def wide(bits):return (bits.astype(np.uint32)<<16).view('<f4').astype(np.float64)

def round_probability_bf16(values):
    bits=values.astype(np.float32).view(np.uint32)
    rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint32)<<16
    return rounded.view('<f4').astype(np.float64)

def from_inputs(inputs,rows,segments,pitch):
    # Decode actual byte roles, independently of the generator's logical lists.
    pages=(len(inputs['cache'])-256)//16384
    qbits=np.ndarray((rows,16,128),dtype='<u2',buffer=inputs['query'],offset=128,strides=(pitch*2,256,2))
    k=np.ndarray((pages,16,2,128),dtype=np.int8,buffer=inputs['cache'],offset=128,strides=(16384,256,128,1))
    scale=np.ndarray((pages,16,2),dtype='<f4',buffer=inputs['cache'],offset=128+4096,strides=(16384,8,4))
    v=np.ndarray((pages,16,2,128),dtype='<u2',buffer=inputs['cache'],offset=128+8192,strides=(16384,512,256,2))
    table=np.frombuffer(inputs['table'],dtype='<i4')[64:-64].reshape(rows,16)
    lengths=np.frombuffer(inputs['lengths'],dtype='<i4')[64:-64]
    shape=(rows,16,segments,128)
    partial=np.full(shape,np.nan);without_round=np.full(shape,np.nan);final_once=np.full(shape,np.nan)
    for row,L in enumerate(lengths):
        length=int(L)
        if not length:continue
        ids=table[row,np.arange(length)//16];slots=np.arange(length)%16
        assert bool(((ids>=0)&(ids<pages)).all())
        keys=k[ids,slots];scales=scale[ids,slots].astype(np.float64);values=wide(v[ids,slots])
        span=((length+segments*16-1)//(segments*16))*16
        for head in range(16):
            qc,qs=quantize_v2(wide(qbits[row,head]).tolist())
            dots=keys[:,head//8].astype(np.int64)@np.array(qc,dtype=np.int64)
            scores=dots.astype(np.float64)*qs*scales[:,head//8]*(128.**-.5)*1.4426950408889634
            for seg in range(segments):
                lower=seg*span;upper=min(lower+span,length)
                if lower>=length:continue
                previous_max=-np.inf;acc=np.zeros(128);pure=np.zeros(128)
                for page_start in range(lower,upper,16):
                    page_end=min(page_start+16,upper)
                    new_max=max(previous_max,float(scores[page_start:page_end].max()))
                    rho=np.exp2(previous_max-new_max)
                    probability=np.exp2(scores[page_start:page_end]-new_max)
                    acc=acc*rho+round_probability_bf16(probability)@values[page_start:page_end,head//8]
                    pure=pure*rho+probability@values[page_start:page_end,head//8]
                    previous_max=new_max
                partial[row,head,seg]=acc;without_round[row,head,seg]=pure
                last_max=float(scores[lower:upper].max())
                final_once[row,head,seg]=round_probability_bf16(np.exp2(scores[lower:upper]-last_max))@values[lower:upper,head//8]
    return dict(partial=partial,without_P_round_control=without_round,final_max_once_round_control=final_once)
