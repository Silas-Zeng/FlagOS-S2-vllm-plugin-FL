#!/usr/bin/env python3
"""CPU-only descriptor/warmup-boundary fixtures; no Torch, SDK or JIT import."""
import argparse
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import biv150_konly_prefill_compile_contract as contract
import bench_biv150_konly_prefill_compile as driver

WIDTH={'torch.bfloat16':2,'torch.int8':1,'torch.uint8':1,'torch.float32':4,'torch.int32':4}


def strides(shape):
    out=[];step=1
    for n in reversed(shape):
        out.insert(0,step);step*=n
    return tuple(out)


class Storage:
    next_id=100
    def __init__(self,nbytes):
        self.size=nbytes;self.pointer=Storage.next_id;Storage.next_id+=1
    def data_ptr(self): return self.pointer
    def nbytes(self): return self.size


class Tensor:
    def __init__(self,shape,dtype,storage=None,stride=None,offset=0):
        self.shape=tuple(shape);self.dtype=dtype;self._stride=stride or strides(shape);self.offset=offset
        self.storage=storage or Storage(math.prod(shape)*WIDTH[dtype]);self.device='cuda:0'
    def stride(self,dim=None): return self._stride if dim is None else self._stride[dim]
    def storage_offset(self): return self.offset
    def untyped_storage(self): return self.storage
    def is_contiguous(self):
        return all(n==1 or a==b for n,a,b in zip(self.shape,self._stride,strides(self.shape)))
    def unbind(self,dim):
        shape=self.shape[:dim]+self.shape[dim+1:];stride=self._stride[:dim]+self._stride[dim+1:]
        return tuple(Tensor(shape,self.dtype,self.storage,stride,self.offset+i*self._stride[dim]) for i in range(self.shape[dim]))
    def view(self,dtype):
        assert len(self.shape)==1 and self.is_contiguous() and self.offset==0
        return Tensor((math.prod(self.shape)*WIDTH[self.dtype]//WIDTH[dtype],),dtype,self.storage)


class EmptyOnlyTorch:
    bfloat16='torch.bfloat16';int8='torch.int8';uint8='torch.uint8';float32='torch.float32';int32='torch.int32'
    def __init__(self): self.calls=[]
    def empty(self,shape,*,dtype,device):
        assert device=='cuda';self.calls.append(('empty',tuple(shape),dtype));return Tensor(shape,dtype)
    def as_strided(self,tensor,shape,stride,storage_offset):
        return Tensor(shape,tensor.dtype,tensor.storage,tuple(stride),storage_offset)


def reject(call):
    try: call()
    except (ValueError,AssertionError): return
    raise AssertionError('Invalid compile-only fixture accepted')


def run():
    records=[];rejects=0
    for profile in contract.PROFILES_SUPPORTED:
        p=contract.plan(profile);fake=EmptyOnlyTorch();t,owners=contract.allocate(fake,p)
        d=contract.descriptors(t);contract.check_descriptors(p,d)
        old,new=contract.kwargs(p,t,False),contract.kwargs(p,t,True)
        bundle=driver.source_bundle(profile)
        baseline_name='kernel_unified_attention' if profile=='raw3d' else 'kernel_biv150_prefill_stride'
        for source,name,values in ((bundle[1],baseline_name,old),(bundle[2],driver.factory.name(profile),new)):
            fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
            names=[a.arg for a in fn.args.args]
            required=set(names[:len(names)-len(fn.args.defaults)])
            assert set(values)<=set(names) and required<=set(values),'Warmup keyword ABI mismatch'
        assert old['stride_k_cache_0']==8192 and new['stride_k_cache_0']==16384
        assert new['stride_v_cache_0']==8192 and new['key_code_scale_ptr'] is t['kscale']
        assert old['q_scale'] is None and new['k_scale'].shape==new['v_scale'].shape==()
        assert d['table']['shape']==(p['sequences'],8192) and d['table']['stride']==(8192,1)
        assert d['kscale']['shape']==(p['pages']*4096,) and d['kscale']['storage_offset']==0
        assert not d['k']['is_contiguous'] and not d['baseline_k']['is_contiguous']
        if profile=='raw3d':
            assert p['requested_options']=={} and d['scratch']['stride']==(32768,2048,128,1)
            assert contract.reducer_kwargs(p,t)['NUM_SEGMENTS_PER_SEQ']==16
        else:
            assert p['requested_options']==dict(num_warps=16,num_stages=1)
            assert p['config']['BLOCK_M']==256 and p['config']['BLOCK_Q']==32
            reject(lambda:contract.reducer_kwargs(p,t));rejects+=1
        for role,field,bad in [('k','shape',(1,)),('k','stride',(8192,256,128,1)),('kscale','storage_offset',1024),
                               ('kscale','shape',(p['pages'],16,2)),('table','shape',(p['sequences'],256)),
                               ('k_descale','shape',(1,)),('baseline_v','storage_offset',0)]:
            mutant=deepcopy(d);mutant[role][field]=bad
            reject(lambda mutant=mutant:contract.check_descriptors(p,mutant));rejects+=1
        data=json.loads(subprocess.check_output([sys.executable,str(Path(driver.__file__)),'--profile',profile],text=True))
        assert data['cpu_only'] and not data['torch_imported'] and not data['triton_imported'] and not data['JIT_executed']
        assert not data['actual_I8_operand_hoist_verified'] and not data['runtime_binding_qualified']
        records.append(dict(profile=profile,grid=p['grid'],empty_allocations=len(fake.calls),descriptors=d,
                            no_tensor_values_initialized=True,source_hash=data['proof']['generated_sha256']))
    reject(lambda:contract.plan('normalized2d128'));rejects+=1
    # None/bool resources cannot accidentally become zero-spill evidence.
    metadata=SimpleNamespace(shared=4096,num_warps=16,num_stages=1,warp_size=64,use_sme=78)
    for reg,spill in ((None,None),(143,None),(143,False),(True,0),(143,-1)):
        obs=driver.observed_resources(SimpleNamespace(metadata=metadata,n_regs=reg,n_spills=spill))
        assert not obs['loaded_resource_fields_known'] and not obs['zero_spill_observed']
        rejects+=1
    assert driver.observed_resources(SimpleNamespace(metadata=metadata,n_regs=143,n_spills=0))['zero_spill_observed']
    metadata.use_sme=14
    assert driver.observed_resources(SimpleNamespace(metadata=metadata,n_regs=93,n_spills=0))['zero_spill_observed']
    # Synthetic archive fixture is explicitly CPU-only, never device evidence.
    with tempfile.TemporaryDirectory(prefix='cjj-konly-compile-cpu-fixture-') as tmp:
        kernel=SimpleNamespace(hash='synthetic_CPU_fixture',asm={k:b'CPU fixture' for k in ('ttgir','llir','asm','cubin')},
            src=SimpleNamespace(fn=SimpleNamespace(src='def synthetic_CPU_fixture(): pass\n'),signature={},constants={},attrs={}),
            metadata=metadata,n_regs=None,n_spills=None)
        out=driver.preserve(kernel,Path(tmp),'CPU_fixture',{},(1,),{}, {},{})
        assert out['complete_source_and_ir'] and not out['ready_for_numeric_or_timing']
        assert not out['actual_I8_hoist']['auditor_supported'] and not out['actual_I8_hoist']['verified']
        del kernel.asm['llir']
        assert not driver.preserve(kernel,Path(tmp),'CPU_missing_LLVM',{},(1,),{}, {},{})['complete_source_and_ir']
        rejects+=1
    reject(lambda:driver.compile_only(SimpleNamespace(commit=None,out=None),contract.plan(),driver.source_bundle('q32')));rejects+=1
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(cpu_only=True,passed=True,reader_source_commit=driver.READER_FREEZE,profiles=records,rejects=rejects,
                framework_imported=False,JIT_executed=False,kernel_launches=0,device_compile_executed=False,
                actual_I8_operand_hoist_verified=False,runtime_binding_qualified=False,actual_resources_known=False,
                accuracy_verified=False,precision_gate=False,performance_verified=False,service_installed=False,capacity_reclaimed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path)
    args=parser.parse_args();text=json.dumps(run(),indent=2,sort_keys=True)+'\n'
    if args.out:
        with args.out.open('x') as stream: stream.write(text)
    print(text,end='')


if __name__=='__main__': main()
