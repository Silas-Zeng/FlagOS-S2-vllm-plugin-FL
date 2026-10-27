"""CPU-only policy/oracles for the first eager2048-row Q32 numerical feedback."""
import math

from biv150_konly_decode_quant_reference import rn32_div


def plan():
    # Context is the prefix BEFORE2048 fresh queries, so total KV is6144.
    return dict(profile='q32',qlens=[2048],klens=[6144],context=4096,total=2048,
                sequences=1,pages=773,segments=1,grid=(65,2),
                config=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,IS_3D=False),
                requested_options=dict(num_warps=16,num_stages=1),
                table_pitch=8192,formal_batch_tokens_changed=False,
                patterns=[0,1],seed=20261025,timing_enabled=False,
                Graph_verified=False,side_stream_verified=False,context16k_tested=False)


def metrics(actual,reference):
    if not actual or len(actual)!=len(reference): raise ValueError('Equal nonempty numerical geometry required')
    finite=all(math.isfinite(x) for x in (*actual,*reference))
    if not finite: return dict(finite=False,pointwise_passed=False,relative_rmse=None)
    error=[a-b for a,b in zip(actual,reference)];rms=math.hypot(*error)/math.sqrt(len(error))
    rrms=math.hypot(*reference)/math.sqrt(len(reference));maximum=max(map(abs,error))
    violations=sum(abs(d)>.005+.005*abs(r) for d,r in zip(error,reference))
    return dict(finite=True,pointwise_passed=violations==0,pointwise_violations=violations,
                relative_rmse=rms/rrms if rrms else (0. if maximum==0 else math.inf),
                rms=rms,max_abs=maximum)


def gates(baseline,candidate,original,quantized):
    # Same explicit pointwise/relative thresholds as the private79 consumer.
    base=metrics(baseline,original);implementation=metrics(candidate,quantized);bf16=metrics(candidate,original)
    limit=max(.005,1.2*base['relative_rmse']) if base['finite'] else .005
    baseline_ok=base['finite'] and base['pointwise_passed']
    return dict(baseline=base,quantized_operand=implementation,original_BF16=bf16,
                baseline_hard_passed=baseline_ok,
                candidate_numeric_passed=baseline_ok and implementation['finite'] and implementation['pointwise_passed']
                    and bf16['finite'] and bf16['relative_rmse']<=limit,
                original_relative_rmse_limit=limit,thresholds_relaxed=False)


def policy(hard,numeric,diagnostic):
    names={'finite','baseline_reference','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise'}
    if set(hard)!=names or any(v is not True for v in hard.values()):
        raise ValueError('Eager hard failure cannot be bypassed')
    if numeric is not True and not diagnostic: raise ValueError('Original precision gate failed; explicit diagnostic required')
    return dict(eager_completed=True,eager_numeric_passed=numeric,numeric_diagnostic=diagnostic,
                precision_gate=False,Graph_verified=False,side_stream_verified=False,
                full_validation_passed=False,timing_attempted=False)


def quantize_cpu(source,torch,np):
    """Independent CPU BF16-bits→RN32/RNE; never uses a GPU scale as oracle.

    Same rational-boundary repair as79: double division is rounded to FP32;
    neighborhoods of half integers are independently recomputed with exact
    rational IEEE division. FP64 reconstruction avoids CPU BF16 FTZ.
    """
    if source.device.type!='cpu' or source.dtype!=torch.bfloat16 or source.shape[-1]!=128:
        raise ValueError('Finite original CPU BF16 vectors required')
    bits=source.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1,128)
    codes=np.empty(bits.shape,dtype=np.int8);scales=np.empty(len(bits),dtype=np.float32);repaired=0
    for start in range(0,len(bits),4096):
        block=bits[start:start+4096];exponent=(block>>7)&255
        if np.any(exponent==255): raise ValueError('Nonfinite original BF16 input')
        wide=np.ldexp((block&127).astype(np.float64)+(exponent!=0)*128,np.maximum(exponent.astype(np.int32),1)-134)
        wide=np.where((block&0x8000)!=0,-wide,wide);peak=np.max(np.abs(wide),axis=1)
        scale=np.array([1. if p==0 else max(rn32_div(float(p),127.),2.**-126) for p in peak],dtype=np.float32)
        ratio=(wide/scale.astype(np.float64)[:,None]).astype(np.float32)
        encoded=np.clip(np.rint(ratio),-127,127).astype(np.int8)
        magnitude=np.abs(ratio).astype(np.float64)
        near=np.abs(magnitude-np.floor(magnitude)-.5)<=2*np.spacing(np.abs(ratio)).astype(np.float64)
        for row,dim in zip(*np.nonzero(near)):
            encoded[row,dim]=max(-127,min(127,round(rn32_div(float(wide[row,dim]),float(scale[row])))))
            repaired+=1
        codes[start:start+len(block)]=encoded;scales[start:start+len(block)]=scale
    return (torch.from_numpy(codes.copy()).reshape(source.shape),
            torch.from_numpy(scales.copy()).reshape(source.shape[:-1]),dict(rational_boundary_cells=repaired))


def encoded_bytes(keys,values,torch,np):
    code,scale,proof=quantize_cpu(keys,torch,np);pages=keys.shape[0]
    result=np.full((pages,16384),0xa5,dtype=np.uint8)
    result[:,:4096]=code.contiguous().numpy().view(np.uint8).reshape(pages,4096)
    result[:,4096:4224]=scale.contiguous().numpy().view(np.uint8).reshape(pages,128)
    result[:,8192:]=values.contiguous().view(torch.int16).numpy().view(np.uint8).reshape(pages,8192)
    return result.tobytes(),code,scale,proof


def references(cpu,code,scale,torch,np):
    """Closed FP64 attention; distinct original and quantized-operand oracles.

    Neither reference models BF16 probability rounding or online softmax steps.
    This distinction is reported; the original consumer tolerances are retained.
    """
    rows=[0,1,15,16,31,32,1024,2047];length=6144
    ids=cpu['table'][0,:384].long()
    keys=cpu['packed'][ids,0].reshape(-1,2,128)[:length].double()
    values=cpu['packed'][ids,1].reshape(-1,2,128)[:length].double()
    kc=code[ids].reshape(-1,2,128)[:length].long();ks=scale[ids].reshape(-1,2)[:length].double()
    selected=cpu['q'][rows];qc,qs,_=quantize_cpu(selected,torch,np)
    q=selected.double().reshape(len(rows),2,8,128).permute(1,2,0,3)
    original_score=torch.einsum('hgrd,thd->hgrt',q,keys)*128**-.5
    integer=np.einsum('hgrd,thd->hgrt',qc.long().reshape(len(rows),2,8,128).permute(1,2,0,3).numpy(),kc.numpy(),dtype=np.int64,optimize=True)
    if int(np.max(np.abs(integer)))>128*127**2: raise ValueError('Independent INT64 QK bound failed')
    quant_score=torch.from_numpy(integer).double()*qs.double().reshape(len(rows),2,8).permute(1,2,0)[:,:,:,None]*ks.T[:,None,None,:]*128**-.5
    mask=torch.arange(length)[None,:]<4096+torch.tensor(rows)[:,None]+1
    def attention(score):
        probability=score.masked_fill(~mask[None,None],-math.inf).softmax(-1)
        return torch.einsum('hgrt,thd->rhgd',probability,values).reshape(len(rows),16,128)
    return dict(rows=rows,original=attention(original_score),quantized=attention(quant_score),
                quantized_reference='INT64 QK, FP64 dequantized scales/global softmax/BF16 V',
                probability_rounding_modeled=False,online_softmax_rounding_modeled=False)
