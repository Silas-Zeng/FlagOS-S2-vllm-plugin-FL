#!/usr/bin/env python3
"""CPU page budgets, actual ABI-call plan and preserved fullstep gates."""
import argparse
import ast
import json
from pathlib import Path
import sys
import types

import probe_biv150_native_k8_compact_n48_fullstep as frontend


class Tensor:
    def __init__(self,shape,strides,name):self.shape=shape;self.strides=strides;self.name=name
    def stride(self,index):return self.strides[index]
    def view(self,dtype):return self


class Kernel:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def call(*args,**kwargs):self.calls.append(dict(grid=grid,args=args,kwargs=kwargs));return object()
        return call


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--cpu-data',action='store_true');a=p.parse_args()
    f=frontend.f;plan=frontend.plan();phases=[]
    for case in f.CASES:
        for phase in range(3):
            h=f.host_plan(case,phase);positive=[x for x in h['slots'] if x>=0]
            assert len(set(positive))==h['live_rows'] and max(positive)<f.PAGES*16
            for row,L in enumerate(h['lengths']):
                assert all(0<=v<f.PAGES for v in h['table'][row][:(L+15)//16])
                assert h['slots'][row]==(-1 if L==0 else h['table'][row][(L-1)//16]*16+(L-1)%16)
            record={k:v for k,v in h.items() if k!='table'}
            if a.cpu_data:
                import numpy as np
                c=f.control(*f.CASES[case][:2],2560,phase)
                for name in ('new_k','new_v'):
                    assert c[name].shape==(48,2560) and bool((c[name][:,256:]==0x4255).all())
                    logical=f.logical(c)[name]
                    assert logical.shape==(48,2,128) and logical.strides==(5120,256,2)
                    assert np.array_equal(logical[32,1],c[name][32,128:256])
                record['CPU_strided_source_bits_and_readonly_padding_verified']=True
            phases.append(record)
    try:f.budget(42,17408)
    except ValueError:pass
    else:raise AssertionError('non-fitting42x17408 admitted')
    text,proof=frontend.source.adapted_source();parent=frontend.compact.adapted_source()[0]
    unchanged={}
    for name in ('timed','capture','reducer','poison','fingerprints','verify_inputs','step'):
        old=frontend.source.nested(parent,name);new=frontend.source.nested(text,name)
        assert old==new,name;unchanged[name]=f.sha(old.encode())
    # Execute only the actual transformed producer/append Python definitions
    # with opaque tensor handles. This never imports or invokes Torch/Triton.
    q=Tensor((48,16,128),(2560,128,1),'q');lens=Tensor((48,),(1,),'lens');table=Tensor((48,8192),(8192,1),'table')
    z={name:Tensor(shape,strides,name) for name,shape,strides in (
      ('key',(44769,16,2,128),(12544,256,128,1)),('value',(44769,16,2,128),(6272,256,128,1)),
      ('scale',(44769*3136,),(1,)),('partial',(48,16,4,128),(32768,2048,128,1)),
      ('maximum',(48,16,4),(256,16,1)),('summation',(48,16,4),(256,16,1)))}
    tl=Kernel();writer=Kernel();native=[]
    def native_call(*args):native.append(args)
    ns=dict(handles={},reader=types.SimpleNamespace(_gqa_decode_konly_i8=tl),n=48,s=4,q=q,lens=lens,table=table,cases={'tl':z,'native':z},
      tl_arguments=frontend.tl_arguments,torch=types.SimpleNamespace(bfloat16=object(),ops=types.SimpleNamespace(cjj_native_k8_compact_n48_s4_probe=types.SimpleNamespace(produce=native_call))),
      writer=types.SimpleNamespace(encode_append=writer),nk=Tensor((48,2,128),(2560,128,1),'nk'),nv=Tensor((48,2,128),(2560,128,1),'nv'),
      slots=Tensor((48,),(1,),'slots'),arenas={arm:{'raw':Tensor((f.PAGES*f.PAGE,),(1,),'raw')} for arm in ('tl','native')},f=f)
    for name in ('producer','append'):exec(compile(frontend.source.nested(text,name),'<actual-N48-fullstep-'+name+'>','exec'),ns)
    for arm in ('tl','native'):ns['append'](arm);ns['producer'](arm)
    assert len(tl.calls)==1 and tl.calls[0]['grid']==(48,2,4) and len(native)==1
    assert native[0][0] is q and native[0][1] is z['key'] and native[0][2] is z['scale'] and native[0][3] is z['value'] and native[0][-1]==4
    call=tl.calls[0]['kwargs'];assert call['query_stride_token']==2560 and call['k_stride_block']==12544 and call['v_stride_block']==6272
    assert call['query_code_ptr'] is z['key'] and call['key_scale_ptr'] is call['query_scale_ptr'] is z['scale']
    assert call['num_warps']==call['num_stages']==1 and call['segment_stride_head']==2048
    for item in writer.calls:
        assert item['grid']==(48,2) and item['args'][6:]==(2560,2560,48,48,48,44769)
        assert item['kwargs']==dict(num_warps=1,num_stages=1)
    assert not {'torch','triton'}&set(sys.modules)
    if not a.cpu_data:assert 'numpy' not in sys.modules
    report=dict(passed=True,plan=plan,phase_page_proofs=phases,invalid42x17408_rejected=True,
      actual_transformed_ABI_call_plan_verified=True,unchanged_common_functions=unchanged,
      CPU_data_requested=a.cpu_data,no_device_execution=True,consumer_qualification_inherited=False,performance_verified=False)
    rendered=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if a.out:assert not a.out.exists();a.out.write_text(rendered)
    print(rendered,end='')


if __name__=='__main__':main()
