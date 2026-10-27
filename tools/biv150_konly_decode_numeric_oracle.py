"""Independent CPU oracles. NumPy/Torch are passed in only by explicit device runner."""
import math

from biv150_konly_decode_quant_reference import rn32_div


def bf16_from_bits(bits):
    exponent=(bits>>7)&255
    if exponent==255: raise ValueError('finite BF16 source required')
    value=math.ldexp((bits&127)+(128 if exponent else 0),max(exponent,1)-134)
    return -value if bits&0x8000 else value


def quantize_bits_cpu(bits,np):
    """Independent CPU RN32 codec over uint16 BF16 storage, with no device data."""
    assert bits.dtype==np.uint16 and bits.ndim==2 and bits.shape[1]==128
    codes=np.empty(bits.shape,dtype=np.int8);scales=np.empty(bits.shape[0],dtype=np.float32)
    repaired=0
    for start in range(0,len(bits),4096):
        block=bits[start:start+4096];exp=(block>>7)&255
        assert not np.any(exp==255),'nonfinite original CPU BF16 input'
        mantissa=(block&127).astype(np.float64)+(exp!=0)*128
        wide=np.ldexp(mantissa,np.maximum(exp.astype(np.int32),1)-134)
        wide=np.where((block&0x8000)!=0,-wide,wide)
        peak=np.max(np.abs(wide),axis=1)
        ss=np.array([1. if p==0 else max(rn32_div(float(p),127.),2.**-126) for p in peak],dtype=np.float32)
        ratio=(wide/ss.astype(np.float64)[:,None]).astype(np.float32)
        encoded=np.clip(np.rint(ratio),-127,127).astype(np.int8)
        magnitude=np.abs(ratio).astype(np.float64)
        near=np.abs(magnitude-np.floor(magnitude)-.5)<=2*np.spacing(np.abs(ratio)).astype(np.float64)
        rr,dd=np.nonzero(near)
        for r,d in zip(rr,dd):
            encoded[r,d]=max(-127,min(127,round(rn32_div(float(wide[r,d]),float(ss[r])))))
        repaired+=len(rr)
        codes[start:start+len(block)]=encoded;scales[start:start+len(block)]=ss
    return codes,scales,dict(rational_boundary_cells=repaired)


def quantize_cpu(source,torch,np):
    """Torch CPU wrapper; BF16 is reconstructed from bits to avoid FP32 FTZ/DAZ.

    Binary64 division may change exact RN32 by at most one FP32 ULP. Only the
    neighborhood of n+1/2 can change integer RNE; a two-ULP neighborhood is
    re-evaluated with integer-rational RN32. Codes do not rely on double rounding.
    """
    assert source.device.type=='cpu' and source.dtype==torch.bfloat16 and source.shape[-1]==128
    bits=source.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1,128)
    codes,scales,record=quantize_bits_cpu(bits,np)
    return (torch.from_numpy(codes.copy()).reshape(source.shape),
            torch.from_numpy(scales.copy()).reshape(source.shape[:-1]),record)


def pack_arrays(codes,scales,value_bits,np):
    pages=codes.shape[0]
    assert codes.shape==(pages,16,2,128) and scales.shape==(pages,16,2)
    assert value_bits.shape==codes.shape and value_bits.dtype==np.uint16
    assert codes.dtype==np.int8 and scales.dtype==np.float32
    result=np.full((pages,16384),0xa5,dtype=np.uint8)
    result[:,:4096]=codes.view(np.uint8).reshape(pages,4096)
    result[:,4096:4224]=scales.view(np.uint8).reshape(pages,128)
    result[:,8192:]=value_bits.view(np.uint8).reshape(pages,8192)
    return result.tobytes()


def encoded_cache_bytes(keys,values,torch,np):
    codes,scales,record=quantize_cpu(keys,torch,np)
    value_bits=values.contiguous().view(torch.int16).numpy().view(np.uint16)
    packed=pack_arrays(codes.contiguous().numpy(),scales.contiguous().numpy(),value_bits,np)
    return packed,codes,scales,record


def attention_references(cold_keys,cold_values,cold_codes,cold_scales,controls,rows,torch,np):
    """Original BF16 and independently reconstructed-operand closed FP64 attention."""
    assert all(t.device.type=='cpu' for t in (cold_keys,cold_values,cold_codes,cold_scales,controls['q']))
    qcode,qscale,_=quantize_cpu(controls['q'],torch,np)
    newcode,newscale,_=quantize_cpu(controls['new_k'],torch,np)
    answers=[]
    for row in rows:
        length=controls['lengths'][row]
        if length==0:
            zeros=torch.zeros((16,128),dtype=torch.float64)
            answers.append(dict(row=row,length=0,original=zeros,quantized=zeros.clone(),integer_bound=0));continue
        ids=controls['table'][row,:(length+15)//16].long()
        key=cold_keys[ids].reshape(-1,2,128)[:length].clone()
        value=cold_values[ids].reshape(-1,2,128)[:length].clone()
        kc=cold_codes[ids].reshape(-1,2,128)[:length].clone()
        ks=cold_scales[ids].reshape(-1,2)[:length].clone()
        positions={int(page):index for index,page in enumerate(ids.tolist())}
        for token,slot in enumerate(controls['slots']):
            if slot<0: continue
            page,column=divmod(slot,16)
            if page not in positions: continue
            position=positions[page]*16+column
            if position>=length: continue
            key[position]=controls['new_k'][token];value[position]=controls['new_v'][token]
            kc[position]=newcode[token];ks[position]=newscale[token]
        query=controls['q'][row].double().reshape(2,8,128)
        original_scores=torch.einsum('hgd,thd->hgt',query,key.double())*128**-.5
        original=torch.einsum('hgt,thd->hgd',original_scores.softmax(-1),value.double()).reshape(16,128)
        qc=qcode[row].long().reshape(2,8,128)
        qs=qscale[row].double().reshape(2,8)
        # Independent CPU INT64 arithmetic, separate from the FP64 Torch contraction.
        integer_array=np.einsum('hgd,thd->hgt',qc.numpy(),kc.long().numpy(),dtype=np.int64,optimize=True)
        integer=torch.from_numpy(integer_array)
        assert integer.dtype==torch.int64 and integer.device.type=='cpu'
        bound=int(integer.abs().max())
        assert bound<=128*127**2<2**24,'independent INT64 dot exceeds exact I32/F32 domain'
        qoperand=qc.double()*qs[:,:,None]
        koperand=kc.double()*ks.double()[:,:,None]
        reconstructed=torch.einsum('hgd,thd->hgt',qoperand,koperand)
        integer_scaled=integer.double()*qs[:,:,None]*ks.double().T[:,None,:]
        # FP64 dot can cancel nearly to zero; bound by absolute-product accumulation.
        absolute=torch.einsum('hgd,thd->hgt',qoperand.abs(),koperand.abs())
        assert bool(((reconstructed-integer_scaled).abs()<=absolute*2**-44+2**-1074).all())
        quantized=torch.einsum('hgt,thd->hgd',(reconstructed*128**-.5).softmax(-1),value.double()).reshape(16,128)
        answers.append(dict(row=row,length=length,original=original,quantized=quantized,integer_bound=bound,
                            int64_vs_reconstructed_fp64_max_abs=float((reconstructed-integer_scaled).abs().max())))
    return answers,(qcode,qscale)
