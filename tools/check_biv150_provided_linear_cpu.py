"""Scope and exact pinned launch ABI checks, without any device import."""
import contextlib
import json
import sys
from types import SimpleNamespace

import biv150_provided_linear as module


def main():
    dtype=object();device=SimpleNamespace(type='cuda');calls=[]
    class Tensor:
        def __init__(self,shape):self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=device;self.requires_grad=False
        def stride(self,axis):return 1 if axis==1 else self.shape[1]
    class Kernel:
        def __getitem__(self,grid):
            def run(*args,**kwargs):calls.append((grid({'BLOCK_M':32,'BLOCK_N':64}),args,kwargs))
            return run
    sys.modules['torch']=SimpleNamespace(bfloat16=dtype,cuda=SimpleNamespace(device=lambda d:contextlib.nullcontext()))
    sys.modules['triton']=SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b)
    module._KERNEL=Kernel();module._PROVENANCE={'cpu_fixture':True}
    count=0
    for n,k in module.GEOMETRIES:
        for rows in (9,16,24,31,32,33,40,48,64,128,2048):
            x,w,out=Tensor((rows,k)),Tensor((n,k)),Tensor((rows,n));assert module.linear_into(x,w,out) is out
            grid,args,kwargs=calls[-1];assert len(args)==14 and args[:4]==(x,w,w,out)
            assert args[4:]==(rows,n,k,k,1,k,1,n,1,0)
            assert kwargs=={'BIAS':False,'GROUP_M':8,'SPLIT_K':1}
            assert grid==(((rows+31)//32)*((n+63)//64),1);count+=1
    for rows in (0,1,2,4,8):
        before=len(calls)
        try:module.linear_into(Tensor((rows,2048)),Tensor((2048,2048)),Tensor((rows,2048)))
        except ValueError:pass
        else:raise AssertionError('Vulnerable tiny rows reached stock SDK')
        assert len(calls)==before
    x=Tensor((32,2048));x.requires_grad=True
    try:module.linear_into(x,Tensor((2048,2048)),Tensor((32,2048)))
    except ValueError:pass
    else:raise AssertionError('Autograd silently bypassed')
    print(json.dumps(dict(passed=True,gpu_executed=False,valid_descriptor_launch_cases=count,
                         tiny_rows_rejected=True,autograd_rejected=True,provided_output_identity=True,
                         limitation='CPU ABI mocks only; actual SDK selection, numerical/Graph correctness and performance pending.')))


if __name__=='__main__':main()
