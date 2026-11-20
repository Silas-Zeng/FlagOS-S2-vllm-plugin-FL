"""CPU routing, SDK promotion and compile identity contracts; no GPU claim."""
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    old=ast.parse((ROOT/'tools/biv150_provided_linear.py').read_text());new=ast.parse((ROOT/'vllm_fl/patches/biv150_provided_linear.py').read_text())
    def functions(tree):return {x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.FunctionDef)}
    assert functions(old)==functions(new),'Promoted SDK ABI/math changed'
    m=load('tested_out',ROOT/'vllm_fl/patches/biv150_small_out_lowering.py')
    assert not m._LARGE_LINEAR and m._PROVIDED_LINEAR is None and 'flag_gems' not in sys.modules
    torch=ModuleType('torch');torch.bfloat16=object();device=SimpleNamespace(type='cuda');calls=[]
    class Tensor:
        def __init__(self,shape,*,pitch=None,storage=None,grad=False,s1=1):
            self.shape=shape;self.ndim=len(shape);self.device=device;self.dtype=torch.bfloat16;self.requires_grad=grad;self.pitch=shape[1] if pitch is None else pitch;self.storage=id(self) if storage is None else storage;self.s1=s1
        def stride(self,d):return self.s1 if d==1 else self.pitch
        def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.storage)
        def t(self):return self
        def copy_(self,v):calls.append('copy');return self
    def mm(x,w,*,out):calls.append('mm');return out
    def provided(x,w,out):calls.append('provided');return out
    torch.mm=mm;torch.cuda=SimpleNamespace(device=lambda d:contextlib.nullcontext(),is_current_stream_capturing=lambda:False)
    torch.nn=SimpleNamespace(functional=SimpleNamespace(linear=lambda x,w:calls.append('functional') or x));sys.modules['torch']=torch
    small=ModuleType('vllm_fl.patches.biv150_stable_small_o');small._small_o_into=lambda x,w,out:calls.append('tiny') or out;sys.modules[small.__name__]=small
    m._PROVIDED_LINEAR=provided;count=0
    for active in [False,True]:
        m._LARGE_LINEAR=active
        for n,k in m._GEOMETRIES:
            for rows in [0,1,2,4,8,9,16,32,64,96,127,128,129,255,256,257,1024,1739,1798,1985,2017,2047,2048,2049,4096]:
                x,w,out=Tensor((rows,k)),Tensor((n,k)),Tensor((rows,n));calls.clear();assert m.small_linear_out(x,w,out=out) is out
                expected='tiny' if 0<rows<=8 else 'provided' if active and 128<=rows<=2048 else 'mm'
                assert calls==[expected],(active,n,k,rows,calls);count+=1
    for n,k in m._GEOMETRIES:
        for kind in ['alias_x','alias_weight','grad_x','grad_weight','grad_out','weight_padding','out_columns']:
            x,w,out=Tensor((128,k)),Tensor((n,k)),Tensor((128,n))
            if kind=='alias_x':out.storage=x.storage
            if kind=='alias_weight':out.storage=w.storage
            if kind=='grad_x':x.requires_grad=True
            if kind=='grad_weight':w.requires_grad=True
            if kind=='grad_out':out.requires_grad=True
            if kind=='weight_padding':w.pitch=k+16
            if kind=='out_columns':out.s1=2
            calls.clear();m.small_linear_out(x,w,out=out);assert calls==['mm'],(kind,calls);count+=1
    m._INSTALLED=True;os.environ['VLLM_FL_BIV150_LARGE_LINEAR']='0'
    try:m.install()
    except RuntimeError:pass
    else:raise AssertionError('Live feature change accepted')
    semantics=load('tested_semantics',ROOT/'vllm_fl/compile_semantics.py');config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**semantics._GEOMETRY)))
    def key(extra):return semantics.build_fingerprint(config,'iluvatar',package_root=ROOT/'vllm_fl',environ=extra,abi={})
    large='VLLM_FL_BIV150_LARGE_LINEAR';out='VLLM_FL_BIV150_SMALL_OUT_LOWERING'
    assert key({})==key({large:'1'})==key({out:'1',large:'1'})
    assert key({large:'0'})==key({out:'1',large:'0'})!=key({})
    assert key({out:'0'})==key({out:'0',large:'0'})==key({out:'0',large:'1'})
    assert key({out:'1',large:'1'})==key({out:'1',large:'1','VLLM_FL_BIV150_SMALL_OUT_RECORD_SELECTION':'1'})
    assert key({'VLLM_FL_BIV150_STABLE_SMALL_O':'0'})==key({'VLLM_FL_BIV150_STABLE_SMALL_O':'0',out:'1',large:'1'})
    print(json.dumps(dict(passed=True,gpu_executed=False,routing_cases=count,promoted_sdk_functions_exact=True,inert_import=True,active_cache_identity=True,tiny_protection_retained=True,live_change_rejected=True)))
if __name__=='__main__':main()
