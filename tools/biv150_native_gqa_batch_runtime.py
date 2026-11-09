"""Default-off fixture-only native runtime. No Torch import or service hook."""
import hashlib
import math
import os
from pathlib import Path

ENV='VLLM_FL_BIV150_NATIVE_GQA_BATCH_DIAGNOSTIC'
FORMAT='stock-bf16-nhd-fixture-v1'
PRIVATE=('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE','VLLM_FL_BIV150_KONLY_COMPACT_SERVICE',
         'VLLM_FL_BIV150_STOCK_QK8_DIAGNOSTIC','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST')


def need(ok,message):
    if not ok:raise RuntimeError('batch native diagnostic: '+message)


def no_private(env):need(not any(env.get(k,'0')=='1' for k in PRIVATE),'private format flag')
def desc(t):return (str(t.dtype),tuple(t.shape),tuple(t.stride()),str(t.device),t.storage_offset(),
                    t.untyped_storage()._cdata,t.data_ptr(),t.untyped_storage().nbytes())


class StockFixture:
    """Own only a fresh allocation copied from literal CPU BF16 values.

    No API admits external CUDA caches, nominal BF16 carriers or a format string.
    This is fixture provenance, not a production allocator receipt.
    """
    def __init__(self,torch,cpu,*,device='cuda:0',environ=None):
        no_private(os.environ if environ is None else environ)
        need(cpu.dtype==torch.bfloat16 and not cpu.is_cuda and not cpu.requires_grad,'CPU BF16 input')
        need(len(cpu.shape)==5 and tuple(cpu.shape[1:])==(2,16,2,128) and cpu.shape[0]>0 and cpu.is_contiguous(),'CPU stock geometry')
        need(bool(torch.isfinite(cpu).all()),'finite CPU cache')
        self.backing=torch.full((cpu.shape[0]*16384+512,),90,dtype=torch.uint8,device=device)
        self.cache=self.backing[256:-256].view(torch.bfloat16).reshape(tuple(cpu.shape))
        self.cache.copy_(cpu);self.key,self.value=self.cache.unbind(1)
        self._snapshot=(desc(self.cache),desc(self.key),desc(self.value));self.format=FORMAT

    def owns(self,key,value):
        return (self.format==FORMAT and key is self.key and value is self.value
                and self._snapshot==(desc(self.cache),desc(key),desc(value)))


class Runtime:
    def __init__(self,torch,library,admission,*,environ=None):
        env=os.environ if environ is None else environ;no_private(env)
        need(env.get(ENV,'0')=='1','explicit opt-in required')
        need(type(admission) is StockFixture,'stock fixture receipt missing')
        need(Path(library).is_absolute() and Path(library).is_file(),'built library required')
        self.torch,self.admission=torch,admission;self.device=torch.cuda.current_device()
        self.library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest();self.calls=0

    def _eligible(self,grid,kw):
        try:return self._check(grid,kw)
        except (AttributeError,IndexError,KeyError,TypeError,ValueError,RuntimeError):return False

    def _check(self,grid,kw):
        no_private(os.environ)
        if type(self.admission) is not StockFixture:return False
        if tuple(grid) not in ((48,2,4),(56,2,4)) or any(type(x) is not int for x in grid):return False
        rows,_,segments=grid
        fixed=dict(queries_per_kv=8,ROWS=8,HEAD_SIZE=128,PAGE_SIZE=16,SEGMENTS=4,num_warps=1,num_stages=1,
                   k_stride_block=8192,k_stride_slot=256,k_stride_head=128,
                   v_stride_block=8192,v_stride_slot=256,v_stride_head=128)
        if any(type(kw.get(k)) is not int or kw[k]!=v for k,v in fixed.items()):return False
        scale=kw.get('softmax_scale_log2e')
        if type(scale) not in (int,float) or not math.isfinite(scale):return False
        q,k,v=(kw[p] for p in ('query_ptr','key_cache_ptr','value_cache_ptr'))
        if not self.admission.owns(k,v):return False
        if q.dtype!=self.torch.bfloat16 or tuple(q.shape)!=(rows,16,128) or tuple(q.stride()) not in ((2048,128,1),(2560,128,1)):return False
        lengths,table=kw['seq_lens_ptr'],kw['block_table_ptr'];o,m,l=(kw[p] for p in ('segm_output_ptr','segm_max_ptr','segm_expsum_ptr'))
        tensors=(q,k,v,lengths,table,o,m,l)
        if any(not t.is_cuda or t.device!=q.device or t.requires_grad for t in tensors) or q.device.index!=self.device:return False
        if lengths.dtype!=self.torch.int32 or tuple(lengths.shape)!=(rows,) or not lengths.is_contiguous():return False
        if table.dtype!=self.torch.int32 or len(table.shape)!=2 or table.shape[0]!=rows or table.stride(1)!=1:return False
        if o.dtype!=self.torch.float32 or len(o.shape)!=4 or o.shape[0]<rows or tuple(o.shape[1:])!=(16,4,128):return False
        if tuple(o.stride()) not in ((8192,512,128,1),(32768,2048,128,1)):return False
        for t in (m,l):
            if t.dtype!=self.torch.float32 or len(t.shape)!=3 or t.shape[0]<rows or tuple(t.shape[1:])!=(16,4):return False
            if tuple(t.stride()) not in ((64,4,1),(256,16,1)):return False
        if m.stride()!=l.stride() or o.stride(1)!=m.stride(1)*128:return False
        if k.data_ptr()%64 or v.data_ptr()%64:return False
        # Only K and V may share storage; all output/input aliasing rejects.
        ids=[t.untyped_storage()._cdata for t in (q,k,lengths,table,o,m,l)]
        if len(ids)!=len(set(ids)):return False
        for t in tensors:
            end=t.storage_offset()+sum((n-1)*s for n,s in zip(t.shape,t.stride()))+1
            if any(n<1 or s<1 for n,s in zip(t.shape,t.stride())) or end*t.element_size()>t.untyped_storage().nbytes():return False
        for ptr,dim,name in [('query_ptr',0,'query_stride_token'),('query_ptr',1,'query_stride_head'),
             ('block_table_ptr',0,'block_table_stride'),('segm_output_ptr',0,'segment_stride_token'),
             ('segm_output_ptr',1,'segment_stride_head'),('segm_output_ptr',2,'segment_stride_segment'),
             ('segm_max_ptr',0,'max_stride_token'),('segm_max_ptr',1,'max_stride_head'),
             ('segm_expsum_ptr',0,'max_stride_token'),('segm_expsum_ptr',1,'max_stride_head')]:
            if type(kw.get(name)) is not int or kw[name]!=kw[ptr].stride(dim):return False
        return True

    def try_produce(self,grid,kw):
        if not self._eligible(grid,kw):return False
        self.torch.ops.cjj_native_gqa_batch_s4_20261028.produce(
          *(kw[p] for p in ('query_ptr','key_cache_ptr','value_cache_ptr','seq_lens_ptr','block_table_ptr',
                           'segm_output_ptr','segm_max_ptr','segm_expsum_ptr')),kw['softmax_scale_log2e'])
        self.calls+=1;return True


def create_runtime(torch=None,library=None,admission=None,*,environ=None):
    env=os.environ if environ is None else environ
    if env.get(ENV,'0')!='1':return None
    return Runtime(torch,library,admission,environ=env)
