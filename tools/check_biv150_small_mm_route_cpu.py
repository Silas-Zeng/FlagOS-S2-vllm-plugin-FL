"""CPU-only route, stride, source-guard and cache-identity contract checks."""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'vllm_fl/patches/biv150_stable_small_o.py'


class Tensor:
    def __init__(self,shape,stride,dtype='bf16',device='cuda:0',cuda=True):
        self.shape,self._stride,self.dtype,self.device,self.is_cuda=shape,stride,dtype,device,cuda
    def stride(self,dim=None):return self._stride if dim is None else self._stride[dim]


def main():
    tree=ast.parse(PATH.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_install_mm_route')
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_small_o_kernel')
    prior=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show','d3a17a403362fc6662cf36d58e68b8018bd4733a:vllm_fl/patches/biv150_stable_small_o.py'],text=True))
    prior_kernel=next(n for n in prior.body if isinstance(n,ast.FunctionDef) and n.name=='_small_o_kernel')
    assert ast.dump(kernel)==ast.dump(prior_kernel),'GPU math changed; previous component evidence cannot be reused'
    pin=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MM_SHA' for t in n.targets))
    assert pin=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    fallback=[];launched=[]
    module=ModuleType('audited_mm_stub')
    def original(*args):
        fallback.append(args)
        return fallback
    module._launch_mm=original
    exec('def mm(*args): return _launch_mm(*args)\ndef mm_out(*args): return _launch_mm(*args)',module.__dict__)
    gems=SimpleNamespace(mm=module.mm,mm_out=module.mm_out)
    class Kernel:
        fail=False
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                if self.fail:raise RuntimeError('launch sentinel')
                launched.append((grid,args,kwargs))
            return call
    fake_kernel=Kernel()
    namespace={'torch':SimpleNamespace(bfloat16='bf16',cuda=SimpleNamespace(device=lambda _:nullcontext(),is_current_stream_capturing=lambda:True)),
               '_MM_SHA':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               '_OBSERVED_ROWS':set(),'_small_o_kernel':fake_kernel}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),namespace)
    with patch.dict(sys.modules,{'flag_gems':gems}),patch('inspect.getsourcefile',return_value=__file__),patch('importlib.import_module',return_value=module):
        expected=namespace['_MM_SHA'];namespace['_MM_SHA']='0'*64
        assert namespace['_install_mm_route']()['installed'] is False
        assert module._launch_mm is original
        namespace['_MM_SHA']=expected
        alien=ModuleType('alien');exec('def mm(*args): pass',alien.__dict__)
        gems.mm=alien.mm
        assert namespace['_install_mm_route']()['installed'] is False
        gems.mm=module.mm
        assert namespace['_install_mm_route']()['installed'] is True
    positive=0
    for m in [1,2,4,8]:
        for n,k in [(2048,2048),(2560,2048),(12288,2048),(2048,6144)]:
            a=Tensor((m,k),(k+32,1));b=Tensor((k,n),(1,k+64));c=Tensor((m,n),(n+64,1))
            # Calls traverse both already-bound functions' real global dict.
            for fn in [gems.mm,gems.mm_out]:
                assert fn(a,b,c,m,n,k) is c
                grid,args,kw=launched[-1]
                assert grid==(n//64,) and args==(a,b,c,m)
                assert kw==dict(K=k,XS0=k+32,XS1=1,WS0=k+64,WS1=1,YS0=n+64,YS1=1,num_warps=4,num_stages=1)
                positive+=1
    def rejected(a,b,c,m=2,n=2048,k=2048):
        count=len(launched);before=len(fallback)
        assert module._launch_mm(a,b,c,m,n,k) is fallback
        assert len(launched)==count and len(fallback)==before+1
        assert fallback[-1]==(a,b,c,m,n,k)
    a=Tensor((2,2048),(2048,1));b=Tensor((2048,2048),(1,2048));c=Tensor((2,2048),(2048,1))
    for m in [0,9,16,32,64,2048]:rejected(a,b,c,m=m)
    rejected(a,b,c,n=130560)
    rejected(Tensor(a.shape,a.stride(),dtype='int8'),b,c)
    rejected(a,Tensor(b.shape,b.stride(),device='cuda:1'),c)
    rejected(a,Tensor(b.shape,(2048,1)),c)
    rejected(a,b,Tensor(c.shape,c.stride(),cuda=False))
    before=len(fallback);fake_kernel.fail=True
    try:module._launch_mm(a,b,c,2,2048,2048)
    except RuntimeError as error:assert str(error)=='launch sentinel'
    else:raise AssertionError('launch error swallowed')
    assert len(fallback)==before
    spec=importlib.util.spec_from_file_location('isolated_semantics',ROOT/'vllm_fl/compile_semantics.py');sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
    config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**sem._GEOMETRY)))
    def fingerprint(enabled,binding=None):
        env={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_STABLE_SMALL_O':enabled}
        if binding is not None:env['VLLM_FL_BIV150_STABLE_SMALL_O_BINDING']=binding
        return sem.build_fingerprint(config,'iluvatar',package_root=ROOT/'vllm_fl',environ=env,abi={'torch':'cpu-contract','cxx11_abi':False})['sha256']
    assert fingerprint('1')==fingerprint('1','linear')
    assert fingerprint('1','mm')!=fingerprint('1','linear')
    assert fingerprint('0','mm')==fingerprint('0','linear')
    result={'passed':True,'gpu_executed':False,'math_ast_unchanged':True,'positive_routes':positive,
            'fallback_routes':len(fallback),'source_rejection':True,'symbol_rejection':True,
            'launch_errors_propagated':True,'binding_cache_identity':True}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
