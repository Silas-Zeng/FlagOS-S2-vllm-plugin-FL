#!/usr/bin/env python3
"""Stdlib Graph/side protocol faults; no framework imports or device execution."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.dont_write_bytecode=True
import bench_biv150_konly_2d128_graph_side as graph
import biv150_konly_2d128_graph_contract as contract


def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('invalid Graph/side boundary accepted')


def run():
    proof=graph.pins();table=list(range(384))+[772]*(8192-384)
    changed,length=contract.changed_metadata(table,6144)
    assert changed[:384]==[383]+list(range(383)) and changed[384:]==table[384:] and length==6143
    for value,bound in ((table,6143),(table[:384],6144),([0]*8192,6144)):
        reject(lambda:contract.changed_metadata(value,bound))
    hard={n:True for n in ('finite','baseline_reference','CPU_codec_cache_bytes','output_canaries','read_only_inputs','actual_IR','quantized_implementation')}
    legal=dict(commit=contract.FROZEN,completed=True,errors=[],reader_launches=4,encoder_launches=0,
        actual_binding_verified=True,loaded_resources_verified=True,precision_gate=False,performance_measured=False,
        eager_numeric_passed=False,records=[dict(pattern=p,hard_checks=dict(hard),numeric_checks=dict(quantized_implementation_passed=True)) for p in (0,1)])
    contract.eager_admission(legal);rejected=[]
    for name in ('commit','completed','errors','reader_launches','actual_binding_verified','loaded_resources_verified','precision_gate','missing_guard'):
        bad=deepcopy(legal)
        if name=='commit':bad[name]='ef4034b23d194b8163062826447ee2b4aa950c36'
        elif name=='completed':bad[name]=False
        elif name=='errors':bad[name]=['original failure']
        elif name=='reader_launches':bad[name]=1
        elif name in ('actual_binding_verified','loaded_resources_verified'):bad[name]=False
        elif name=='precision_gate':bad[name]=True
        else:del bad['records'][0]['hard_checks']['output_canaries']
        reject(lambda:contract.eager_admission(bad));rejected.append(name)
    cpu=dict(q='q0',table='t0',lengths='l0');device=dict(query_ptr='q0',block_tables_ptr='t0',seq_lens_ptr='l0')
    assert contract.control_fingerprints(device,cpu)==cpu
    for name in device:
        reject(lambda:contract.control_fingerprints(dict(device,**{name:'stale'}),cpu));rejected.append('device_'+name)
    scope=graph.Scope(Path('/unused'))
    a=SimpleNamespace(state='active',token='token',owner=dict(pid=7,start_ticks=11),roots=('fixed',),native=701,
        identity=lambda:dict(pid=7,start_ticks=11),binding=lambda:('fixed',),closed=lambda:True,query=lambda:[701])
    scope.admission=a
    scope.guard('baseline','eager0');scope.guard('baseline','capture0')
    reject(lambda:scope.guard('baseline','eager1'));rejected.append('new_eager_before_graph')
    scope.guard('baseline','graph1');scope.guard('baseline','eager1');scope.guard('baseline','side1')
    reject(lambda:scope.guard('baseline','side1'));rejected.append('duplicate_mode')
    scope=graph.Scope(Path('/unused'));scope.admission=a;a.query=lambda:[701,702]
    reject(lambda:scope.guard('candidate','eager0'));rejected.append('foreign_native_PID')
    class Tensor:
        def __init__(self,dtype,strides,offset=0,base=100):self.dtype,self.strides,self.offset,self.base=dtype,strides,offset,base
        def stride(self):return self.strides
        def storage_offset(self):return self.offset
        def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.base,nbytes=lambda:2050*16*128*2)
    fake=SimpleNamespace(int8='i8',bfloat16='bf16',float32='f32')
    args=dict(KV_QUANT_MODE=0,BLOCK_M=128,BLOCK_Q=16,TILE_SIZE=64,num_seqs=1,
        key_cache_ptr=Tensor('i8',(16384,256,128,1)),value_cache_ptr=Tensor('bf16',(8192,256,128,1),4096),
        key_code_scale_ptr=Tensor('f32',(1,)),output_ptr=Tensor('bf16',(2048,128,1),2048,base=200))
    missing=object();previous=getattr(graph.fixture,'torch',missing);graph.fixture.torch=fake
    try:
        scope.pointer_contract('candidate',args)
        for name in ('scale_double_offset','V_full_base','I8_stride_in_BF16_units','wrong_scale_storage','wrong_BM'):
            bad=dict(args)
            if name=='scale_double_offset':bad['key_code_scale_ptr']=Tensor('f32',(1,),1024)
            elif name=='V_full_base':bad['value_cache_ptr']=Tensor('bf16',(8192,256,128,1),0)
            elif name=='I8_stride_in_BF16_units':bad['key_cache_ptr']=Tensor('i8',(8192,256,128,1))
            elif name=='wrong_scale_storage':bad['key_code_scale_ptr']=Tensor('f32',(1,),base=300)
            else:bad['BLOCK_M']=256
            reject(lambda:scope.pointer_contract('candidate',bad));rejected.append(name)
    finally:
        if previous is missing:del graph.fixture.torch
        else:graph.fixture.torch=previous
    class StorageInput:
        def __init__(self,base,nbytes,width):self.base,self.nbytes,self.width=base,nbytes,width
        def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.base,nbytes=lambda:self.nbytes)
        def element_size(self):return self.width
        def view(self,*args):raise AssertionError('must flatten full typed storage before byte view, including 0D')
    def as_strided(value,shape,stride,offset):
        assert shape==(value.nbytes//value.width,) and stride==(1,) and offset==0
        return SimpleNamespace(view=lambda dtype:SimpleNamespace(raw=b'x'*value.nbytes))
    previous=getattr(graph.fixture,'torch',missing)
    graph.fixture.torch=SimpleNamespace(Tensor=StorageInput,uint8='u8',as_strided=as_strided)
    try:
        scope.bytes=lambda value:value.raw
        got=scope.inputs_fingerprint(dict(k_scale=StorageInput(1,4,4),key_cache_ptr=StorageInput(2,16384,1),
                    value_cache_ptr=StorageInput(2,16384,2)))
        assert set(got)=={'k_scale','key_cache_ptr'} and got['k_scale']==graph.frozen.actual.sha(b'xxxx')
    finally:
        if previous is missing:del graph.fixture.torch
        else:graph.fixture.torch=previous
    # Hook restoration must happen on an exception, without performing a load.
    originals=(graph.frozen.parent.module,graph.fixture.cpu_inputs,graph.frozen.oracle.references,graph.frozen.Admission)
    try:
        with scope.adapt():raise RuntimeError('forced restore')
    except RuntimeError:pass
    assert originals==(graph.frozen.parent.module,graph.fixture.cpu_inputs,graph.frozen.oracle.references,graph.frozen.Admission)
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,source_contract=proof,runtime_KV_lengths=[6144,6143],query_rows=2048,
        changed_table_is_unique_rotation=True,Graph_before_new_eager_enforced=True,eager_hard_precondition_faults=rejected,
        private_hooks_restored=True,original_numeric_resource_readonly_canary_gates_unchanged=True,
        complete_CPU_guarded_output_used=True,actual_Graph_verified=False,actual_side_verified=False,
        GPU_executed=False,JIT_executed=False,precision_gate=False,performance_measured=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);a=p.parse_args();result=run()
    if a.out:a.out.write_bytes(graph.frozen.actual.encode(result))
    print(json.dumps(dict(CPU_contract_passed=True,rejections=len(result['eager_hard_precondition_faults'])+3,GPU_executed=False)))


if __name__=='__main__':main()
