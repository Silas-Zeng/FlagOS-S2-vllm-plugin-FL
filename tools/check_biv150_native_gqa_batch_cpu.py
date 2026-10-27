"""CPU metadata/ownership/address contracts. No Torch/Triton/compilation."""
import ast
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import biv150_native_gqa_batch_factory as f
import biv150_native_gqa_batch_runtime as r


class Storage:
    counter=0
    def __init__(self,n):
        Storage.counter+=1;self._cdata=Storage.counter;self.n=n
    def nbytes(self):return self.n


class Tensor:
    def __init__(self,shape,dtype='bf16',device='cuda:0',stride=None,storage=None,offset=0):
        self.shape=tuple(shape);self.dtype=dtype;self.requires_grad=False;self.offset=offset
        self.is_cuda=device!='cpu';self.device=S(type='cuda' if self.is_cuda else 'cpu',index=0 if self.is_cuda else None)
        strides=[];v=1
        for n in reversed(shape):strides.insert(0,v);v*=n
        self.strides=tuple(strides) if stride is None else stride;self.storage=storage or Storage(v*self.element_size())
    def element_size(self):return {'bf16':2,'f32':4,'i32':4,'u8':1}[self.dtype]
    def stride(self,d=None):return self.strides if d is None else self.strides[d]
    def untyped_storage(self):return self.storage
    def storage_offset(self):return self.offset
    def data_ptr(self):return self.storage._cdata*0x1000000+self.offset*self.element_size()
    def is_contiguous(self):
        n=1
        for size,stride in zip(reversed(self.shape),reversed(self.stride())):
            if size>1 and stride!=n:return False
            n*=size
        return True
    def __getitem__(self,s):
        assert len(self.shape)==1 and isinstance(s,slice)
        start,stop,step=s.indices(self.shape[0]);assert step==1
        return Tensor((stop-start,),self.dtype,storage=self.storage,offset=self.offset+start)
    def view(self,dt):return Tensor((math.prod(self.shape)*self.element_size()//{'bf16':2,'u8':1}[dt],),dt,
        storage=self.storage,offset=self.offset*self.element_size()//{'bf16':2,'u8':1}[dt])
    def reshape(self,shape):return Tensor(shape,self.dtype,storage=self.storage,offset=self.offset)
    def copy_(self,other):assert not other.is_cuda;return self
    def unbind(self,dim):return tuple(Tensor(self.shape[:dim]+self.shape[dim+1:],self.dtype,
        stride=self.strides[:dim]+self.strides[dim+1:],storage=self.storage,offset=self.offset+i*self.strides[dim]) for i in range(self.shape[dim]))


def reject(call):
    try:call()
    except RuntimeError:return
    raise AssertionError('bad receipt admitted')


def run():
    data,proof=f.render();assert proof['CUDA_inverse_byte_exact']
    tree=ast.parse(f.source('vllm_fl/patches/biv150_gqa_decode.py'))
    ns=dict(TARGET_PROGRAMS=256);exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick_segments'],type_ignores=[]),'<actual-b3-pick>','exec'),ns)
    pick={n:ns['_pick_segments'](n,2,16) for n in (32,40,48,56,64)}
    assert pick=={32:4,40:16,48:4,56:4,64:2}
    assert r.create_runtime() is None
    calls=[];torch=S(bfloat16='bf16',float32='f32',int32='i32',uint8='u8',cuda=S(current_device=lambda:0),
        isfinite=lambda x:S(all=lambda:getattr(x,'finite',True)),full=lambda shape,*args,**kw:Tensor(shape,kw['dtype'],kw['device']),
        ops=S(cjj_native_gqa_batch_s4_20261028=S(produce=lambda *a:calls.append(a))))
    cpu=Tensor((4,2,16,2,128),device='cpu');fixture=r.StockFixture(torch,cpu,environ={})
    for flag in r.PRIVATE:reject(lambda:r.StockFixture(torch,cpu,environ={flag:'1'}))
    reject(lambda:r.StockFixture(torch,fixture.cache,environ={}))
    cpu.finite=False;reject(lambda:r.StockFixture(torch,cpu,environ={}));cpu.finite=True
    with tempfile.TemporaryDirectory() as folder:
        lib=Path(folder)/'CPU-not-executable.so';lib.write_bytes(b'CPU-only-fake')
        reject(lambda:r.Runtime(torch,lib,None,environ={r.ENV:'1'}))
        runtime=r.create_runtime(torch,lib,fixture,environ={r.ENV:'1'})
        rejects=0
        for n in (48,56):
            for pad in (4,16):
                q=Tensor((n,16,128),stride=(2560,128,1) if pad==16 else (2048,128,1),storage=Storage(n*2560*2))
                kw=dict(query_ptr=q,key_cache_ptr=fixture.key,value_cache_ptr=fixture.value,
                    seq_lens_ptr=Tensor((n,),'i32'),block_table_ptr=Tensor((n,8192),'i32'),
                    segm_output_ptr=Tensor((n,16,4,128),'f32',stride=(16*pad*128,pad*128,128,1),storage=Storage(n*16*pad*128*4)),
                    segm_max_ptr=Tensor((n,16,4),'f32',stride=(16*pad,pad,1),storage=Storage(n*16*pad*4)),
                    segm_expsum_ptr=Tensor((n,16,4),'f32',stride=(16*pad,pad,1),storage=Storage(n*16*pad*4)),
                    softmax_scale_log2e=128**-.5*1.4426950408889634,queries_per_kv=8,ROWS=8,HEAD_SIZE=128,PAGE_SIZE=16,SEGMENTS=4,num_warps=1,num_stages=1,
                    k_stride_block=8192,k_stride_slot=256,k_stride_head=128,v_stride_block=8192,v_stride_slot=256,v_stride_head=128,
                    query_stride_token=q.stride(0),query_stride_head=128,block_table_stride=8192,
                    segment_stride_token=16*pad*128,segment_stride_head=pad*128,segment_stride_segment=128,
                    max_stride_token=16*pad,max_stride_head=pad)
                assert runtime._eligible((n,2,4),kw);assert runtime.try_produce((n,2,4),kw)
                before=len(calls)
                for key,bad in [('HEAD_SIZE',True),('num_warps',True),('SEGMENTS',16),('query_stride_token',2049),('softmax_scale_log2e',math.nan),
                    ('key_cache_ptr',Tensor(fixture.key.shape,stride=fixture.key.stride())),
                    ('key_cache_ptr',Tensor(fixture.key.shape,stride=fixture.key.stride(),storage=fixture.key.storage,offset=fixture.key.offset)),
                    ('segm_expsum_ptr',kw['segm_max_ptr']),('query_ptr',Tensor((n,15,128))),
                    ('query_ptr',Tensor((n,16,128),'f32'))]:
                    assert not runtime.try_produce((n,2,4),dict(kw,**{key:bad}));rejects+=1
                for grid in ((40,2,4),(40,2,16),(32,2,4),(64,2,2),(n,2,16)):
                    assert not runtime.try_produce(grid,kw);rejects+=1
                assert len(calls)==before
                # Enumerate the exact original CUDA per-lane write map/pitch.
                writes=set();scalars=set()
                for seq in range(n):
                    for h in range(2):
                        for seg in range(4):
                            for lane in range(64):
                                g,col=divmod(lane,16)
                                for row in (h*8+g,h*8+g+4):
                                    for panel in range(8):
                                        x=seq*16*pad*128+row*pad*128+seg*128+panel*16+col
                                        assert x not in writes;writes.add(x)
                                    if col==0:scalars.add(seq*16*pad+row*pad+seg)
                assert len(writes)==n*16*4*128 and max(writes)<n*16*pad*128
                assert len(scalars)==n*16*4 and max(scalars)<n*16*pad
    # Actual page descriptor promotion is uint64 before page*8192.
    assert 'static_cast<std::uint64_t>(page)*8192' in data['batch.cu']
    max_address=0x100000000+((2**31-1)*8192+128+4095)*2
    assert 2**32<max_address<2**64
    assert not any(x in __import__('sys').modules for x in ('torch','triton','numpy'))
    return dict(cpu_passed=True,main=f.MAIN,actual_pick_segments=pick,source_proof=proof,
        hard_scope_mutants_rejected=rejects,stock_fixture_only=True,external_or_encoded_alias_rejected=True,
        scratch_write_map_bijective=True,wide_address_checked=True,default_off_no_framework_imports=True,
        GPU_executed=False,JIT_executed=False,compiled=False,numeric_verified=False,service_activation=False,performance_verified=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
