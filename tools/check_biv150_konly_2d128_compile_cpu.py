#!/usr/bin/env python3
"""Stdlib descriptors/ABI/warmup-only checks; no framework import or device work."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import bench_biv150_konly_2d128_compile as driver
from biv150_konly_2d128_compile_contract import WarmupOnly, typed_cache_proof

ROOT=Path(__file__).resolve().parents[1]
WIDTH={'torch.bfloat16':2,'torch.int8':1,'torch.uint8':1,'torch.float32':4,'torch.int32':4}


def contiguous(shape):
    out=[];step=1
    for n in reversed(shape): out.insert(0,step);step*=n
    return tuple(out)


class Storage:
    next_id=100
    def __init__(self,size): self.size=size;self.pointer=Storage.next_id;Storage.next_id+=1
    def data_ptr(self): return self.pointer
    def nbytes(self): return self.size


class Tensor:
    def __init__(self,shape,dtype,storage=None,stride=None,offset=0):
        self.shape=tuple(shape);self.dtype=dtype;self.storage=storage or Storage(math.prod(shape)*WIDTH[dtype])
        self._stride=stride or contiguous(shape);self.offset=offset;self.device='cuda:0'
    def stride(self,dim=None): return self._stride if dim is None else self._stride[dim]
    def storage_offset(self): return self.offset
    def untyped_storage(self): return self.storage
    def is_contiguous(self): return all(n==1 or a==b for n,a,b in zip(self.shape,self._stride,contiguous(self.shape)))
    def unbind(self,dim):
        shape=self.shape[:dim]+self.shape[dim+1:];stride=self._stride[:dim]+self._stride[dim+1:]
        return tuple(Tensor(shape,self.dtype,self.storage,stride,self.offset+i*self._stride[dim]) for i in range(self.shape[dim]))
    def view(self,dtype):
        if len(self.shape)!=1 or not self.is_contiguous() or self.offset: raise ValueError('Fixture view scope')
        return Tensor((math.prod(self.shape)*WIDTH[self.dtype]//WIDTH[dtype],),dtype,self.storage)


class EmptyOnlyTorch:
    bfloat16='torch.bfloat16';int8='torch.int8';uint8='torch.uint8';float32='torch.float32';int32='torch.int32'
    def __init__(self): self.calls=[]
    def empty(self,shape,*,dtype,device):
        if device!='cuda': raise ValueError('Component device')
        self.calls.append(dict(shape=tuple(shape),dtype=dtype,bytes=math.prod(shape)*WIDTH[dtype]))
        return Tensor(shape,dtype)
    def as_strided(self,tensor,shape,stride,storage_offset):
        return Tensor(shape,tensor.dtype,tensor.storage,tuple(stride),storage_offset)


def run():
    manifest=json.loads((ROOT/'docs/evidence/konly-prefill-2d128-20261027/borrowed80.json').read_text())
    for name,row in manifest.items():
        b=(ROOT/name).read_bytes()
        if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']: raise ValueError('Frozen80 copy changed '+name)
    p=driver.contract.plan();old=driver.contract.old_numeric.plan()
    for name in ('qlens','klens','context','total','sequences','pages','segments'):
        assert p[name]==old[name],'Same Q32 small geometry must be retained'
    assert p['grid']==(129,2) and p['config']==dict(BLOCK_M=128,BLOCK_Q=16,TILE_SIZE=64,IS_3D=False)
    assert p['requested_options']==dict(num_warps=8,num_stages=1)
    fake=EmptyOnlyTorch();tensors,owners=driver.contract.allocate(fake,p)
    desc=driver.contract.descriptors(tensors);descriptor_proof=driver.contract.check_descriptors(p,desc)
    sources,baseline,candidate,source_proof=driver.source_bundle()
    ABI={};required_types={'key_cache_ptr':'*i8','key_code_scale_ptr':'*fp32','value_cache_ptr':'*bf16'}
    for label,source,is_candidate in [('baseline',baseline,False),('candidate',candidate,True)]:
        name=driver.legacy.factory.name(p['profile']) if is_candidate else 'kernel_biv150_prefill_stride'
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        names=[a.arg for a in node.args.args];required=set(names[:len(names)-len(node.args.defaults)])
        values=driver.contract.kwargs(p,tensors,is_candidate)
        assert set(values)<=set(names) and required<=set(values),'Raw kwargs ABI mismatch'
        ABI[label]=dict(raw_parameter_names=names,kwargs=sorted(values),source_sha256=hashlib.sha256(source.encode()).hexdigest())
        assert values['stride_k_cache_0']==(16384 if is_candidate else 8192)
        assert values['stride_v_cache_0']==8192 and values['KV_QUANT_MODE']==0
        if is_candidate: assert values['key_code_scale_ptr'] is tensors['kscale']
    rejects=[]
    def reject(name,fn):
        try: fn()
        except (ValueError,AssertionError,RuntimeError): rejects.append(name)
        else: raise AssertionError('Accepted '+name)
    for role,field,bad in [('k','stride',(8192,256,128,1)),('v','storage_offset',0),
                           ('kscale','storage_offset',1024),('kscale','shape',(773,16,2)),
                           ('table','shape',(1,384)),('k_descale','shape',(1,)),
                           ('baseline_v','storage_offset',0),('out','storage','allocation2')]:
        d=deepcopy(desc);d[role][field]=bad
        reject(role+'-'+field,lambda d=d:driver.contract.check_descriptors(p,d))
    for role,kind in required_types.items():
        assert role in ABI['candidate']['raw_parameter_names']
    class Kernel:
        def __init__(self): self.calls=0
        def warmup(self,**kwargs): self.calls+=1;return 'synthetic_CPU_compile_marker'
    raw=Kernel();fence=WarmupOnly(raw)
    assert fence.warmup(grid=p['grid'])=='synthetic_CPU_compile_marker' and raw.calls==1
    reject('launch-subscript',lambda:fence[(129,2)])
    reject('direct-call',lambda:fence())
    assert fence.warmups==1 and fence.forbidden_launch_attempts==2
    names=ABI['candidate']['raw_parameter_names'];signature={k:v for k,v in required_types.items()}
    typed_cache_proof(signature,names,True)
    typed_cache_proof({(names.index(k),):v for k,v in signature.items()},names,True)
    for role,bad in [('key_cache_ptr','*bf16'),('key_code_scale_ptr','*i8'),('value_cache_ptr','*i8')]:
        d=dict(signature);d[role]=bad;reject('typed-'+role,lambda d=d:typed_cache_proof(d,names,True))
    reject('typed-missing-scale',lambda:typed_cache_proof({k:v for k,v in signature.items() if k!='key_code_scale_ptr'},names,True))
    # The adapter exposes no launch path; actual compilation is still not executed.
    adapter_source=(ROOT/'tools/bench_biv150_konly_2d128_compile.py').read_text()
    tree=ast.parse(adapter_source)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) for n in ast.walk(tree))
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(cpu_only=True,passed=True,main=driver.contract.MAIN,dependencies=[79,80,82,86],
                borrowed80=manifest,plan=p,source_proof=source_proof,raw_ABI_plans=ABI,
                expected_candidate_cache_types=required_types,descriptors=desc,descriptor_proof=descriptor_proof,
                empty_allocations=fake.calls,total_empty_bytes=sum(r['bytes'] for r in fake.calls),
                rejected=rejects,kernel_launches=0,JIT_executed=False,actual_codegen_artifacts=False,
                actual_compiled_pointer_types_verified=False,actual_I8_operand_hoist_verified=False,
                mechanism_status='unknown_not_audited_for_BM128',loaded_resources_verified=False,
                numeric_verified=False,precision_gate=False,service_installed=False,performance_measured=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);args=p.parse_args()
    text=json.dumps(run(),sort_keys=True,indent=2)+'\n'
    if args.out:
        if args.out.exists(): raise ValueError('Fresh output required')
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    else: print(text,end='')
