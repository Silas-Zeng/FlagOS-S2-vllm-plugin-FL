"""CPU source/dispatch/carry-identity/cache checks; no Torch import or GPU claim."""
import ast
import hashlib
import importlib.util
import inspect
import json
import linecache
from pathlib import Path
import subprocess
import tempfile
import textwrap
from types import SimpleNamespace as NS

r=Path(__file__).resolve().parents[1]
p=r/'vllm_fl/patches/biv150_live_carry_gate.py'
module=ast.parse(p.read_text())
provenance=json.loads((r/'docs/evidence/live-carry-service-20261025/kernel-ast-provenance.json').read_text())
def canonical_ast(node):
    if isinstance(node,ast.AST):
        return [type(node).__name__,[[key,canonical_ast(value)] for key,value in ast.iter_fields(node)]]
    if isinstance(node,list):return [canonical_ast(value) for value in node]
    return node

assert provenance['hash_format']=='json-all-ast-fields-v1'
for name,expected in provenance['functions'].items():
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert hashlib.sha256(json.dumps(canonical_ast(function),ensure_ascii=True,separators=(',',':')).encode()).hexdigest()==expected
assert not subprocess.check_output(['git','-C',str(r),'diff','0bf9d2e','--','csrc'],text=True)
allowed={'vllm_fl/patches/biv150_live_carry_gate.py','vllm_fl/patches/biv150_native_w8a8.py','vllm_fl/compile_semantics.py'}
assert set(subprocess.check_output(['git','-C',str(r),'diff','0bf9d2e','--name-only','--','vllm_fl'],text=True).splitlines())<=allowed
before=ast.parse(subprocess.check_output(['git','-C',str(r),'show','0bf9d2e:vllm_fl/patches/biv150_native_w8a8.py'],text=True))
after=ast.parse((r/'vllm_fl/patches/biv150_native_w8a8.py').read_text())
for name in ('load_library','quantize_weight','_eligible_model'):
    a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

class Tensor:
    def __init__(self,shape,dtype='bf16',device='cuda:0',contiguous=True,strides=None):
        self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=device
        self.contiguous=contiguous;self.is_cuda=device.startswith('cuda');self.strides=strides
    def is_contiguous(self):return self.contiguous
    def stride(self):return self.strides
class Act:pass
namespace={'torch':NS(bfloat16='bf16',int8='i8',float32='f32'),'_ACTIVATION_TYPE':Act}
function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_supported')
exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual-policy>','exec'),namespace)
checks=0
for m in (0,1,8,16,17,31,32,40,63,64,65,128,256,2048):
    x=Tensor((m,2048));res=Tensor((m,2048))
    linear=NS(bias=None,weight=Tensor((12288,2048),strides=(0,0)),
        _biv150_native_wq=Tensor((2048,12288),dtype='i8'),_biv150_native_sw=Tensor((12288,),dtype='f32'))
    norm=NS(weight=Tensor((2048,)),variance_epsilon=1e-6,variance_size_override=None,has_weight=True)
    layer=NS(post_attention_layernorm=norm,mlp=NS(gate_up_proj=linear,act_fn=Act()))
    assert namespace['_supported'](layer,x,res)==(m==64);checks+=1
    for obj,key,value in ((x,'dtype','f32'),(res,'device','cuda:1'),(norm,'variance_epsilon',1e-5),
        (norm,'has_weight',False),(norm,'variance_size_override',1024),
        (linear,'bias',object()),(linear.weight,'strides',(2048,1)),
        (linear._biv150_native_wq,'contiguous',False),(norm.weight,'dtype','f32')):
        old=getattr(obj,key);setattr(obj,key,value)
        assert not namespace['_supported'](layer,x,res);checks+=1
        setattr(obj,key,old)

rewrite=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_rewrite_forward_ast')
ns={'ast':ast,'textwrap':textwrap}
exec(compile(ast.Module(body=[rewrite],type_ignores=[]),'<actual-rewrite>','exec'),ns)
source='''def forward(self, positions, hidden_states, residual):
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
'''
original=ast.parse(source).body[0];new=ns['_rewrite_forward_ast'](source).body[0]
assert ast.dump(original.args,include_attributes=False)==ast.dump(new.args,include_attributes=False)
assert [ast.dump(n,include_attributes=False) for n in original.body[:2]]==[ast.dump(n,include_attributes=False) for n in new.body[:2]]
assert ast.dump(original.body[-1],include_attributes=False)==ast.dump(new.body[-1],include_attributes=False)
assert len(new.body)==len(original.body)-1
generated=str(p)+'.cpu-check.generated_forward.py'
generated_source=ast.unparse(ast.Module(body=[new],type_ignores=[]))+'\n'
linecache.cache[generated]=(len(generated_source),None,generated_source.splitlines(keepends=True),generated)
compiled={};exec(compile(generated_source,generated,'exec'),compiled)
assert inspect.getsource(compiled['forward'])==generated_source
try:ns['_rewrite_forward_ast'](source.replace('self.mlp(hidden_states)','self.other(hidden_states)'));raise AssertionError('changed structure accepted')
except AssertionError as error:assert 'changed structure accepted' not in str(error)

helper=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_post_norm_mlp')
x,res,normed,carry,gate,out,w,wq,sw=[object() for _ in range(9)]
class MLP:
    gate_up_proj=NS(_biv150_native_wq=wq,_biv150_native_sw=sw)
    def __call__(self,value):assert value is normed;return out
    def down_proj(self,value):assert value is gate;return out,None
class Norm:
    weight=w
    def __call__(self,a,b):assert a is x and b is res;return normed,carry
layer=NS(post_attention_layernorm=Norm(),mlp=MLP())
for selected in (False,True):
    def fused(a,b,c,d,e):assert (a,b,c,d,e)==(x,res,w,wq,sw);return gate
    ns={'_supported':lambda *args:selected,'gate_m64':fused}
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'<actual-helper>','exec'),ns)
    actual=ns['_post_norm_mlp'](layer,x,res);assert actual[0] is out and actual[1] is carry

spec=importlib.util.spec_from_file_location('live_carry_semantics',r/'vllm_fl/compile_semantics.py')
sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
model=NS(model_config=NS(hf_config=NS(**sem._GEOMETRY)))
with tempfile.TemporaryDirectory() as td:
    lib=Path(td)/'native.so';lib.write_bytes(b'fingerprint-only')
    env={'BIV150_NATIVE_W8A8_LIBRARY':str(lib)};abi={'torch':'test','cxx11_abi':False}
    default=sem.build_fingerprint(model,'iluvatar',environ=env,abi=abi)
    enabled=sem.build_fingerprint(model,'iluvatar',environ=dict(env,VLLM_FL_BIV150_LIVE_CARRY_GATE='1'),abi=abi)
    disabled=sem.build_fingerprint(model,'iluvatar',environ=dict(env,VLLM_FL_BIV150_LIVE_CARRY_GATE='0'),abi=abi)
    assert default==enabled and enabled!=disabled
print(json.dumps(dict(cpu_only=True,policy_cases=checks,exact_tested_kernel_ast=True,
    shipping_cpp_cuda_and_quantizer_unchanged=True,minimal_decoder_ast_rewrite=True,
    original_carry_identity_preserved=True,original_fallback_contract=True,
    default_enabled_disabled_cache_identity_checked=True,gpu_model_acceptance_pending=True)))
