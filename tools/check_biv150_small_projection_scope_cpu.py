"""CPU dispatch/cache/cleanup contracts for projection-scope ablation; no GPU import."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
BASE='34942f7e28da5c954ab05878dec75807da9d7a93'
GEOMETRIES={'qkv':(2560,2048),'o':(2048,2048),'down':(2048,6144),'gate':(12288,2048)}
PREFIX={'qkv':'self_attn.qkv_proj','o':'self_attn.o_proj','down':'mlp.down_proj','gate':'mlp.gate_up_proj'}
ENV='VLLM_FL_BIV150_STABLE_SMALL_O_PROJECTIONS'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def tree_functions(text):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}

def main():
    semantics=load('isolated_semantics',ROOT/'vllm_fl/compile_semantics.py')
    assert semantics.small_projection_scope({})==('qkv','o','down','gate')
    for bad in ['',',','gate,gate','unknown','all,gate','none,qkv']:
        try:semantics.small_projection_scope({ENV:bad})
        except ValueError:pass
        else:raise AssertionError(('accepted invalid scope',bad))
    old=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':vllm_fl/patches/biv150_stable_small_o.py'],text=True)
    funcs=tree_functions((ROOT/'vllm_fl/patches/biv150_stable_small_o.py').read_text());previous=tree_functions(old)
    for name in ['_linear','_small_o_kernel','_small_o_into','_register']:assert funcs[name]==previous[name],name
    original=torch=ModuleType('torch');torch.bfloat16=object();calls=[]
    class Tensor:
        def __init__(self,shape,dtype=None):self.shape=shape;self.dtype=torch.bfloat16 if dtype is None else dtype
        def reshape(self,*shape):
            if shape[0]==-1:shape=(math.prod(self.shape)//shape[1],shape[1])
            return Tensor(shape,self.dtype)
    def custom(x,w):calls.append(('custom',x.shape,w.shape));return Tensor((x.shape[0],w.shape[0]))
    torch.ops=SimpleNamespace(biv150_stable_small_o=SimpleNamespace(linear=custom));sys.modules['torch']=torch
    triton=ModuleType('triton');triton.jit=lambda **kw:lambda f:f;tl=ModuleType('triton.language');tl.constexpr=object();sys.modules['triton']=triton;sys.modules['triton.language']=tl
    package=ModuleType('vllm');package.envs=SimpleNamespace(VLLM_BATCH_INVARIANT=False);sys.modules['vllm']=package
    distributed=ModuleType('vllm.distributed');distributed.get_tensor_model_parallel_world_size=lambda:1;sys.modules[distributed.__name__]=distributed
    utils=ModuleType('vllm.model_executor.layers.utils');utils.default_unquantized_gemm=object();utils.dispatch_unquantized_gemm=lambda:utils.default_unquantized_gemm;sys.modules[utils.__name__]=utils
    linear=ModuleType('vllm.model_executor.layers.linear');sys.modules[linear.__name__]=linear
    native=ModuleType('vllm_fl.patches.biv150_native_w8a8');native._eligible_model=lambda:True;sys.modules[native.__name__]=native
    sys.modules['vllm_fl.compile_semantics']=semantics
    positives=fallbacks=0
    os.environ['VLLM_FL_BIV150_STABLE_SMALL_O']='1'
    for scope in ['all','gate','o,gate','qkv,o,down','none']:
        selected=set(semantics.small_projection_scope({ENV:scope}));os.environ[ENV]=scope
        class Method:
            def apply(self,layer,x,bias=None):calls.append(('original',layer.prefix));return Tensor((*x.shape[:-1],layer.weight.shape[0]))
        linear.UnquantizedLinearMethod=Method;method=Method();original_apply=Method.apply
        module=load('isolated_operator',ROOT/'vllm_fl/patches/biv150_stable_small_o.py');module._register=lambda:None
        result=module.install();assert result['installed']==bool(selected);assert set(result['projections'])==selected
        if not selected:assert Method.apply is original_apply
        for name,(n,k) in GEOMETRIES.items():
            layer=SimpleNamespace(prefix='model.layers.3.'+PREFIX[name],weight=Tensor((n,k)))
            for rows in [1,2,4,8,16,64,2048]:
                calls.clear();out=method.apply(layer,Tensor((rows,k)));assert out.shape==(rows,n)
                expected='custom' if name in selected else 'original';assert len(calls)==1 and calls[0][0]==expected,(scope,name,rows,calls)
                positives+=1
            for failure in ['bias','native','dtype','prefix','shape']:
                bad=SimpleNamespace(prefix=layer.prefix,weight=layer.weight);bias=None
                if failure=='bias':bias=object()
                if failure=='native':bad._biv150_native_wq=object()
                if failure=='dtype':bad.weight=Tensor((n,k),dtype=object())
                if failure=='prefix':bad.prefix='foreign.qkv_proj'
                if failure=='shape':bad.weight=Tensor((n+1,k))
                calls.clear();method.apply(bad,Tensor((2,k)),bias);assert len(calls)==1 and calls[0][0]=='original';fallbacks+=1
        if selected:
            installed=Method.apply;os.environ[ENV]='none'
            try:module.install()
            except RuntimeError:pass
            else:raise AssertionError('scope changed after install')
            assert Method.apply is installed
    config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**semantics._GEOMETRY)))
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'never-load.so';library.write_bytes(b'fingerprint only')
        env={'BIV150_NATIVE_W8A8_LIBRARY':str(library)}
        def fp(extra):return semantics.build_fingerprint(config,'iluvatar',package_root=ROOT/'vllm_fl',environ={**env,**extra},abi={})['sha256']
        assert fp({})==fp({ENV:'all'})==fp({ENV:'gate,down,o,qkv'})
        identities=[fp({ENV:v}) for v in ['all','none','gate','o,gate','qkv,o,down']];assert len(set(identities))==5
        assert fp({ENV:'gate,o'})==fp({ENV:'o,gate'})
        assert fp({'VLLM_FL_BIV150_STABLE_SMALL_O':'0',ENV:'bad'})==fp({'VLLM_FL_BIV150_STABLE_SMALL_O':'0'})
    controller=ast.parse((ROOT/'tools/run_biv150_small_scope_acceptance.py').read_text());base_controller=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':tools/run_biv150_small_bf16_acceptance.py'],text=True))
    def final(tree):return [ast.dump(n,include_attributes=False) for node in ast.walk(tree) if isinstance(node,ast.Try) and node.finalbody for n in node.finalbody]
    assert final(controller)==final(base_controller),'owned process cleanup changed'
    report={'passed':True,'gpu_executed':False,'dispatch_cases':positives,'fallback_cases':fallbacks,'scope_sets':5,'default_all_preserved':True,'kernel_and_cuda_math_ast_unchanged':True,'invalid_scopes_rejected':6,'live_scope_mutation_rejected':True,'canonical_compile_identity_separated':True,'owned_cleanup_ast_unchanged':True}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
