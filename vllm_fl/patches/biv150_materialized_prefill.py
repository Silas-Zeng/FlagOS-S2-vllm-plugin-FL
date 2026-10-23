"""Experimental single-sequence BF16/FP32 prefill; not installed in serving."""
import torch
import triton
import triton.language as tl


@triton.jit
def _causal_softmax(S,P,ROWS:tl.constexpr,KV:tl.constexpr,CONTEXT:tl.constexpr,
                    QUERY_START:tl.constexpr,SCALE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    index=tl.arange(0,BLOCK)
    # ROWS = query-chunk tokens times eight GQA heads; two KV groups.
    query=(row%ROWS)//8+QUERY_START
    score=tl.load(S+row*KV+index,index<KV,other=-float('inf'))*SCALE
    score=tl.where(index<=CONTEXT+query,score,-float('inf'))
    probability=tl.exp2((score-tl.max(score,0))*1.4426950408889634)
    probability=probability/tl.sum(probability,0)
    tl.store(P+row*KV+index,probability,index<KV)


def attention(q,k,v,table,kv_length,out,chunk=128):
    assert q.dtype==torch.bfloat16 and q.shape[1:]==(16,128)
    assert k.shape[1:]==v.shape[1:]==(16,2,128)
    qlen=q.shape[0];context=kv_length-qlen
    assert 0<=context and kv_length<=32768 and table.shape[0]==1
    ids=table[0,:(kv_length+15)//16].long()
    keys=k.index_select(0,ids).reshape(-1,2,128)[:kv_length].permute(1,0,2).contiguous()
    values=v.index_select(0,ids).reshape(-1,2,128)[:kv_length].permute(1,0,2).contiguous()
    query=q.reshape(qlen,2,8,128).permute(1,0,2,3).reshape(2,qlen*8,128).contiguous()
    for start in range(0,qlen,chunk):
        stop=min(start+chunk,qlen);rows=(stop-start)*8
        scores=torch.bmm(query[:,start*8:stop*8],keys.transpose(1,2),out_dtype=torch.float32)
        probability=torch.empty(scores.shape,dtype=q.dtype,device=q.device)
        _causal_softmax[(2*rows,)](scores,probability,rows,kv_length,context,start,
                                  128**-.5,triton.next_power_of_2(kv_length),num_warps=8)
        product=torch.empty((2,rows,128),dtype=q.dtype,device=q.device)
        # Explicit out overload makes the internal GEMM provider auditable.
        torch.ops.aten.bmm.out(probability,values,out=product)
        out[start:stop].copy_(product.reshape(2,stop-start,8,128).permute(1,0,2,3).reshape(stop-start,16,128))
    return out
