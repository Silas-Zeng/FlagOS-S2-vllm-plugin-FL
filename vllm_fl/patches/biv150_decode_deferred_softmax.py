"""Experimental BF16 decode: deferred softmax between QK and PV passes.

No serving registration. Caller owns scratch storage and serialized launches.
"""
import torch
import triton
import triton.language as tl

LOG2E=1.4426950408889634


@triton.jit
def _qk_scores(q,k,lengths,table,scores,local_max,
    q_token:tl.constexpr,q_head:tl.constexpr,
    k_block:tl.int64,k_slot:tl.int64,k_head:tl.int64,
    table_row:tl.int64,CAP:tl.constexpr,LM_SEGMENTS:tl.constexpr,
    SCALE:tl.constexpr,SEGMENTS:tl.constexpr,TILE:tl.constexpr):
    token=tl.program_id(0);kv=tl.program_id(1);seg=tl.program_id(2)
    n=tl.load(lengths+token)
    span=tl.cdiv(n,SEGMENTS*TILE)*TILE
    lower=seg*span;upper=tl.minimum(lower+span,n)
    if lower>=n:return
    r=tl.arange(0,8);d=tl.arange(0,128);j=tl.arange(0,TILE)
    pg=tl.arange(0,TILE//16)
    head=kv*8+r
    Q=tl.load(q+token*q_token+head[:,None]*q_head+d[None,:])
    maximum=tl.full((8,),float('-inf'),tl.float32)
    for position in range(lower,upper,TILE):
        pages=tl.load(table+token*table_row+position//16+pg,
                      mask=position+pg*16<upper,other=0).to(tl.int64)
        page=tl.reshape(tl.broadcast_to(pages[:,None],(TILE//16,16)),(TILE,))
        valid=position+j<upper
        K=tl.load(k+page[None,:]*k_block+(j%16)[None,:]*k_slot+kv*k_head+d[:,None],
                  mask=valid[None,:],other=0.)
        S=tl.dot(Q,K)*SCALE
        S=tl.where(valid[None,:],S,float('-inf'))
        maximum=tl.maximum(maximum,tl.max(S,1))
        tl.store(scores+(token*16+head[:,None])*CAP+position+j[None,:],S,mask=valid[None,:])
    tl.store(local_max+(token*16+head)*LM_SEGMENTS+seg,maximum)


@triton.jit
def _softmax_pv(v,lengths,table,scores,local_max,partial,partial_max,partial_sum,
    v_block:tl.int64,v_slot:tl.int64,v_head:tl.constexpr,
    table_row:tl.int64,CAP:tl.constexpr,LM_SEGMENTS:tl.constexpr,
    partial_token:tl.constexpr,partial_head:tl.constexpr,partial_segment:tl.constexpr,
    max_token:tl.constexpr,max_head:tl.constexpr,
    SEGMENTS:tl.constexpr,TILE:tl.constexpr):
    token=tl.program_id(0);kv=tl.program_id(1);seg=tl.program_id(2)
    n=tl.load(lengths+token)
    span=tl.cdiv(n,SEGMENTS*TILE)*TILE
    lower=seg*span;upper=tl.minimum(lower+span,n)
    if lower>=n:return
    active=tl.cdiv(n,span)
    r=tl.arange(0,8);d=tl.arange(0,128);j=tl.arange(0,TILE)
    pg=tl.arange(0,TILE//16);sg=tl.arange(0,SEGMENTS)
    head=kv*8+r
    maxima=tl.load(local_max+(token*16+head[:,None])*LM_SEGMENTS+sg[None,:],
                   mask=sg[None,:]<active,other=float('-inf'))
    maximum=tl.max(maxima,1)
    accumulator=tl.zeros((8,128),tl.float32)
    mass=tl.zeros((8,),tl.float32)
    for position in range(lower,upper,TILE):
        valid=position+j<upper
        S=tl.load(scores+(token*16+head[:,None])*CAP+position+j[None,:],
                  mask=valid[None,:],other=float('-inf'))
        P=tl.exp2(S-maximum[:,None])
        mass+=tl.sum(P,1)
        pages=tl.load(table+token*table_row+position//16+pg,
                      mask=position+pg*16<upper,other=0).to(tl.int64)
        page=tl.reshape(tl.broadcast_to(pages[:,None],(TILE//16,16)),(TILE,))
        V=tl.load(v+page[:,None]*v_block+(j%16)[:,None]*v_slot+kv*v_head+d[None,:],
                  mask=valid[:,None],other=0.)
        accumulator+=tl.dot(P.to(V.dtype),V)
    tl.store(partial+token*partial_token+head[:,None]*partial_head+seg*partial_segment+d[None,:],accumulator)
    offset=token*max_token+head*max_head+seg
    tl.store(partial_max+offset,maximum*0.6931471805599453)
    tl.store(partial_sum+offset,mass)


class DeferredLaunch:
    def __init__(self,scores,local_max,tile=64,warps=2,stages=1):
        self.scores,self.local_max=scores,local_max
        self.tile,self.warps,self.stages=tile,warps,stages
        self.last_qk=self.last_pv=None
    def __getitem__(self,grid):
        def invoke(**kw):
            assert kw['PAGE_SIZE']==16 and kw['HEAD_SIZE']==128 and kw['queries_per_kv']==8
            opts=dict(num_warps=self.warps,num_stages=self.stages)
            self.last_qk=_qk_scores[grid](
                kw['query_ptr'],kw['key_cache_ptr'],kw['seq_lens_ptr'],kw['block_table_ptr'],self.scores,self.local_max,
                kw['query_stride_token'],kw['query_stride_head'],kw['k_stride_block'],kw['k_stride_slot'],kw['k_stride_head'],
                kw['block_table_stride'],self.scores.shape[-1],self.local_max.shape[-1],kw['softmax_scale_log2e'],
                kw['SEGMENTS'],self.tile,**opts)
            self.last_pv=_softmax_pv[grid](
                kw['value_cache_ptr'],kw['seq_lens_ptr'],kw['block_table_ptr'],self.scores,self.local_max,
                kw['segm_output_ptr'],kw['segm_max_ptr'],kw['segm_expsum_ptr'],
                kw['v_stride_block'],kw['v_stride_slot'],kw['v_stride_head'],kw['block_table_stride'],
                self.scores.shape[-1],self.local_max.shape[-1],
                kw['segment_stride_token'],kw['segment_stride_head'],kw['segment_stride_segment'],
                kw['max_stride_token'],kw['max_stride_head'],kw['SEGMENTS'],self.tile,**opts)
            return self.last_pv
        return invoke


class DeferredReduce:
    def __init__(self,kernel,tile):self.kernel,self.tile=kernel,tile
    def __getitem__(self,grid):
        def invoke(**kw):
            kw['page_size']=self.tile
            return self.kernel[grid](**kw)
        return invoke
