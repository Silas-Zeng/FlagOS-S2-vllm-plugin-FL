"""Check promotion of tested math, bounded real hook policy and untouched native math."""
import ast,json,subprocess
from pathlib import Path
from types import SimpleNamespace
r=Path(__file__).resolve().parents[1]
pkg=ast.parse((r/'vllm_fl/patches/biv150_bf16_small_splitk.py').read_text())
old=ast.parse(subprocess.check_output(['git','-C',str(r),'show','755404f:tools/biv150_bf16_small_splitk.py'],text=True))
for name in ('partial_mm','reduce_partials','allocate','run_into','matmul'):
 a=next(x for x in pkg.body if isinstance(x,ast.FunctionDef) and x.name==name)
 b=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
assert not subprocess.check_output(['git','-C',str(r),'diff','0bf9d2e','--','csrc'],text=True)
allowed={'vllm_fl/patches/biv150_bf16_small_splitk.py','vllm_fl/patches/biv150_native_w8a8.py','vllm_fl/compile_semantics.py'}
assert set(subprocess.check_output(['git','-C',str(r),'diff','0bf9d2e','--name-only','--','vllm_fl'],text=True).splitlines())<=allowed
base=ast.parse(subprocess.check_output(['git','-C',str(r),'show','0bf9d2e:vllm_fl/patches/biv150_native_w8a8.py'],text=True))
cur=ast.parse((r/'vllm_fl/patches/biv150_native_w8a8.py').read_text())
for name in ('quantize_weight','load_library','_eligible_model'):
 a=next(x for x in base.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in cur.body if isinstance(x,ast.FunctionDef) and x.name==name)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
f=next(x for x in pkg.body if isinstance(x,ast.FunctionDef) and x.name=='_supported')
ctx={'_EXACT_ROWS':(1,8,16),'_GEOMETRIES':{(2048,2560),(2048,2048),(6144,2048)},'torch':SimpleNamespace(bfloat16='bf16',is_inference_mode_enabled=lambda:True)}
exec(compile(ast.Module(body=[f],type_ignores=[]),'<actual-hook-policy>','exec'),ctx)
class Tensor:
 def __init__(self,shape,stride,dtype='bf16',device='cuda:0',contiguous=True):self.shape=shape;self._stride=stride;self.dtype=dtype;self.device=device;self.is_cuda=device.startswith('cuda');self.contiguous=contiguous
 def stride(self):return self._stride
 def is_contiguous(self):return self.contiguous
checks=0
for m in (0,1,2,4,8,16,17,31,32,40,64,2048):
 for k,n in ((2048,2560),(2048,2048),(6144,2048),(2048,12288),(2048,130560)):
  a=Tensor((m,k),(k,1));b=Tensor((k,n),(1,k));c=Tensor((m,n),(n,1))
  assert ctx['_supported'](a,b,c,m,n,k)==(m in (1,8,16) and (k,n) in ctx['_GEOMETRIES'])
  for obj,key,value in ((a,'dtype','f32'),(b,'device','cuda:1'),(c,'contiguous',False)):
   prior=getattr(obj,key);setattr(obj,key,value);assert not ctx['_supported'](a,b,c,m,n,k);setattr(obj,key,prior)
  checks+=4
print(json.dumps(dict(cpu_only=True,policy_cases=checks,same_math_as1134_probe=True,
    native_quantizer_and_cuda_unchanged=True,caller_output_no_weight_copy_contract=True,device_hook_pending=True)))

import importlib.util,tempfile
spec=importlib.util.spec_from_file_location('bf16_semantics_check',r/'vllm_fl/compile_semantics.py');sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
model=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**sem._GEOMETRY)))
with tempfile.TemporaryDirectory() as td:
 lib=Path(td)/'native.so';lib.write_bytes(b'fingerprint-only immutable test artifact')
 env={'BIV150_NATIVE_W8A8_LIBRARY':str(lib)};abi={'torch':'test','cxx11_abi':False}
 default=sem.build_fingerprint(model,'iluvatar',environ=env,abi=abi)
 enabled=sem.build_fingerprint(model,'iluvatar',environ=dict(env,VLLM_FL_BIV150_BF16_SMALL_SPLITK='1'),abi=abi)
 disabled=sem.build_fingerprint(model,'iluvatar',environ=dict(env,VLLM_FL_BIV150_BF16_SMALL_SPLITK='0'),abi=abi)
 assert default==enabled and enabled!=disabled
print('new_feature_flag_distinguishes_compile_cache_default_enabled')
