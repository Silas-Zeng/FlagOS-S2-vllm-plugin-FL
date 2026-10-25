"""CPU-only compiler contracts for optional Inductor-owned BF16 outputs."""
import ast
import contextlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
BASE='34942f7e28da5c954ab05878dec75807da9d7a93'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    operator=load('out_lowering',ROOT/'vllm_fl/patches/biv150_small_out_lowering.py')
    torch=ModuleType('torch');torch.bfloat16=object();calls=[];cuda=SimpleNamespace(type='cuda')
    class Tensor:
        def __init__(self,shape,dtype=None,stride1=1,device=cuda):self.shape=tuple(shape);self.dtype=torch.bfloat16 if dtype is None else dtype;self.device=device;self.s1=stride1
        @property
        def ndim(self):return len(self.shape)
        def stride(self,d):return self.s1 if d==1 else self.shape[-1]
        def t(self):return Tensor(self.shape[::-1],self.dtype)
        def copy_(self,value):calls.append('copy');return self
    def gemm(x,w,*,out):calls.append('original_mm_out');return out
    def functional(x,w):calls.append('original_functional');return Tensor((*x.shape[:-1],w.shape[0]),x.dtype)
    torch.mm=gemm;torch.nn=SimpleNamespace(functional=SimpleNamespace(linear=functional));torch.cuda=SimpleNamespace(device=lambda x:contextlib.nullcontext());sys.modules['torch']=torch
    small=ModuleType('vllm_fl.patches.biv150_stable_small_o')
    def into(x,w,out):calls.append('unchanged_small_kernel');return out
    small._small_o_into=into;sys.modules[small.__name__]=small
    tested=0
    for n,k in operator._GEOMETRIES:
        for rows in [0,1,2,4,8,16,32,64,2048]:
            x,w,out=Tensor((rows,k)),Tensor((n,k)),Tensor((rows,n));calls.clear();assert operator.small_linear_out(x,w,out=out) is out
            assert calls==(['unchanged_small_kernel'] if 0<rows<=8 else ['original_mm_out']);tested+=1
        x,w,out=Tensor((2,k),stride1=2),Tensor((n,k)),Tensor((2,n));calls.clear();operator.small_linear_out(x,w,out=out);assert calls==['original_functional','copy'];tested+=1
    sys.modules['sympy']=SimpleNamespace(Integer=int)
    target=object();torch.ops=SimpleNamespace(biv150_stable_small_o=SimpleNamespace(linear=SimpleNamespace(default=target)))
    registry={};layouts=[]
    def fallback(op,add_to_fallback_set=True):
        assert op is target and not add_to_fallback_set
        def original(x,w):return 'previous_fallback'
        return original
    def register(op,**kwargs):
        assert op is target and kwargs=={'type_promotion_kind':None}
        def deco(fn):registry[op]=fn;return fn
        return deco
    lowering=ModuleType('torch._inductor.lowering');lowering.lowerings=registry;lowering.fallback_handler=fallback;lowering.register_lowering=register;sys.modules[lowering.__name__]=lowering
    ir=ModuleType('torch._inductor.ir');ir.FixedLayout=lambda *args:layouts.append(args) or args
    parent=ModuleType('torch._inductor');parent.ir=ir;sys.modules[parent.__name__]=parent
    class Choice:
        def __init__(self,kernel,**kw):assert kernel is operator.small_linear_out and kw['has_out_variant'] and kw['op_overload'] is target
        def bind(self,inputs,layout):return SimpleNamespace(output_node=lambda:('out_ir',inputs,layout))
    select=ModuleType('torch._inductor.select_algorithm');select.ExternKernelChoice=Choice;sys.modules[select.__name__]=select
    assert operator.install()['installed'];assert operator.install()['reason']=='already installed'
    class Node:
        def __init__(self,shape,dtype=None,device=cuda):self.shape=shape;self.dtype=torch.bfloat16 if dtype is None else dtype;self.device=device;self.realized=0
        def get_size(self):return self.shape
        def get_dtype(self):return self.dtype
        def get_device(self):return self.device
        def realize(self):self.realized+=1
    symbolic_m=object()
    for n,k in operator._GEOMETRIES:
        x,w=Node([symbolic_m,k]),Node([n,k]);result=registry[target](x,w);assert result[0]=='out_ir' and x.realized==w.realized==1;assert layouts[-1][2]==[symbolic_m,n] and layouts[-1][3]==[n,1]
    for x,w in [(Node([2,12]),Node([13,12])),(Node([2,2048],dtype=object()),Node([2048,2048])),(Node([2,2048]),Node([2048,6144])),(Node([2,2,2048]),Node([2048,2048]))]:assert registry[target](x,w)=='previous_fallback'
    source=ROOT/'vllm_fl/patches/biv150_stable_small_o.py';old=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':vllm_fl/patches/biv150_stable_small_o.py'],text=True)
    def funcs(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(source.read_text()),funcs(old)
    for key in ['_linear','_small_o_kernel','_small_o_into','_register']:assert a[key]==b[key],key
    semantics=load('pure_semantics',ROOT/'vllm_fl/compile_semantics.py');config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**semantics._GEOMETRY)))
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'no-load.so';library.write_bytes(b'fingerprint fixture only')
        def fingerprint(extra):return semantics.build_fingerprint(config,'iluvatar',package_root=ROOT/'vllm_fl',environ={'BIV150_NATIVE_W8A8_LIBRARY':str(library),**extra},abi={})
        key='VLLM_FL_BIV150_SMALL_OUT_LOWERING';assert fingerprint({})==fingerprint({key:'0'});assert fingerprint({})!=fingerprint({key:'1'});assert fingerprint({'VLLM_FL_BIV150_STABLE_SMALL_O':'0'})==fingerprint({'VLLM_FL_BIV150_STABLE_SMALL_O':'0',key:'1'})
    print(json.dumps({'passed':True,'gpu_executed':False,'output_callback_cases':tested,'symbolic_M_layout_cases':4,'unsupported_compiler_fallbacks':4,'output_buffer_identity_preserved':True,'kernel_math_ast_unchanged':True,'default_disabled_equivalent':True,'active_cache_identity_separated':True,'remaining':'Actual Torch compile, streams/changed Graphs, generated out calls, model105 and warmed4K/16K'},indent=2))

if __name__=='__main__':main()
