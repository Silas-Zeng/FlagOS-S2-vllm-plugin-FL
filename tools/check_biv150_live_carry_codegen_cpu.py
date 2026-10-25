"""Execute the real postcompile transform and routing with CPU-only stubs."""
import argparse
import ast
import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paired_codegen',ROOT/'vllm_fl/patches/biv150_live_carry_codegen.py')
own=importlib.util.module_from_spec(spec);spec.loader.exec_module(own)
norm=(ROOT/'docs/evidence/live-carry-service-20261025/postcompile-reference-norm.py.txt').read_text()
assert own.norm_hash(norm)==own.EXPECTED_NORM_SHA256
fixture='kernel = async_compile.triton("reference", '+repr(norm)+''')
class Runner:
    def __init__(self, partitions): self.partitions=partitions
    def call(self,args):
        x,r,w,q,s,marker,m,carry=args
        out=empty_strided_cuda((m,2048),(2048,1),torch.bfloat16)
        kernel.run(x,r,w,out,m,2048,stream=13)
        del w
        result=torch.ops.biv150_native.gate_silu_auto.default(out,q,s,marker,17)
        del out
        return result,carry
call=Runner([]).call
'''
change=own.transform(fixture);assert change and len(change['pairs'])==1
reuse_fixture=fixture.replace('del out','storage=out\n        del out\n        extern_kernels.mm(result,q,out=storage)')
reuse=own.transform(reuse_fixture);assert reuse and len(reuse['pairs'][0]['storage_reuse'])==1
for broken in (reuse_fixture.replace('extern_kernels.mm','consume(storage)\n        extern_kernels.mm'),
    reuse_fixture.replace('mm(result,q','mm(storage,q'),
    reuse_fixture.replace('extern_kernels.mm','extern_kernels.addmm')):
    assert own.transform(broken) is None
negative=0
for broken in (fixture.replace('1e-06','1e-05'),
    fixture.replace('del out','consume(out); del out'),fixture.replace('marker,17','marker,18'),
    fixture.replace('torch.bfloat16)','torch.float32)'),fixture.replace('del w','consume(x); del w'),
    fixture.replace('default(out,q','default(x,q')):
    assert own.transform(broken) is None;negative+=1

class Tensor:
    def __init__(self,shape,dtype='bf16',device='cuda:0',contiguous=True,strides=None):
        self.shape=shape;self.dtype=dtype;self.device=device;self.contiguous=contiguous
        self.is_cuda=device.startswith('cuda');self.strides=strides;self.value=None
    def is_contiguous(self):return self.contiguous
    def stride(self):return self.strides

def inputs(rows):
    x=Tensor((rows,2048));r=Tensor((rows,2048));w=Tensor((2048,));out=Tensor((rows,2048))
    q=Tensor((2048,12288),'i8');s=Tensor((12288,),'f32');marker=Tensor((12288,2048),strides=(0,0))
    return (x,r,w,out,rows,2048),(out,q,s,marker,17)

own._TORCH=NS(bfloat16='bf16',int8='i8',float32='f32',cuda=NS(current_stream=lambda device:NS(cuda_stream=13)))
policy=0
for rows in (0,1,8,16,17,31,32,40,63,64,65,128,256,2048):
    a,g=inputs(rows);assert own.supported(a,g)==(rows==64);policy+=1
    for tensor,key,value in ((a[0],'dtype','f32'),(a[1],'device','cuda:1'),(a[2],'contiguous',False),
        (a[3],'dtype','f32'),(g[1],'dtype','bf16'),(g[2],'contiguous',False),(g[3],'strides',(2048,1))):
        old=getattr(tensor,key);setattr(tensor,key,value)
        assert not own.supported(a,g);setattr(tensor,key,old);policy+=1

events=[];fused_result=object();native_result=object()
class Kernel:
    def run(self,*a,**kw):events.append(('norm',a,kw));a[3].value='normalized'
def gate(*args):
    events.append(('gate',args));assert args[0].value=='normalized';return native_result
def fused(x,r,w,q,s):events.append(('fused',(x,r,w,q,s)));return fused_result
module_name='vllm_fl.patches.biv150_live_carry_gate'
old_modules={name:sys.modules.get(name) for name in ('vllm_fl','vllm_fl.patches',module_name)}
for name in old_modules:sys.modules[name]=types.ModuleType(name)
sys.modules[module_name].gate_m64=fused
try:
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'original.py';path.write_text(fixture)
        mod=types.ModuleType('cpu_compiled_fixture')
        mod.__dict__.update(async_compile=NS(triton=lambda *args:Kernel()),
            empty_strided_cuda=lambda shape,strides,dtype:Tensor(shape,dtype),
            torch=NS(bfloat16='bf16',ops=NS(biv150_native=NS(gate_silu_auto=NS(default=gate)))))
        exec(compile(fixture,str(path),'exec'),vars(mod))
        baseline=mod.call
        with contextlib.redirect_stdout(io.StringIO()):assert own.patch_module(mod,path) is mod
        patched=mod.call;assert patched.__func__ is mod.Runner.call
        assert Path(inspect.getsourcefile(patched)).is_file() and inspect.getsource(patched)==change['function_source']
        assert path.read_text()==fixture
        for rows in (64,32,2048,1,64):
            a,g=inputs(rows);carry=object();events.clear()
            actual=patched([a[0],a[1],a[2],g[1],g[2],g[3],rows,carry])
            assert actual[1] is carry
            assert actual[0] is (fused_result if rows==64 else native_result)
            assert [e[0] for e in events]==(['fused'] if rows==64 else ['norm','gate'])
            assert all(t.value is None for t in (a[0],a[1],a[2],g[1],g[2],g[3]))
        a,g=inputs(64);events.clear();baseline([a[0],a[1],a[2],g[1],g[2],g[3],64,object()])
        assert [e[0] for e in events]==['norm','gate']
        assert own.patch_module(mod,path).call.__func__ is patched.__func__
        a,g=inputs(64);events.clear();own.paired(Kernel().run,gate,a,{'stream':14},g)
        assert [e[0] for e in events]==['norm','gate']
        saved=sys.modules[module_name].gate_m64
        def fail(*args):raise RuntimeError('injected execution failure')
        sys.modules[module_name].gate_m64=fail;events.clear()
        try:own.paired(Kernel().run,gate,a,{'stream':13},g);raise AssertionError('swallowed failure')
        except RuntimeError as error:assert str(error)=='injected execution failure'
        assert not events;sys.modules[module_name].gate_m64=saved
finally:
    for name,value in old_modules.items():
        if value is None:sys.modules.pop(name,None)
        else:sys.modules[name]=value

result=dict(cpu_only=True,passed=True,policy_cases=policy,negative_transform_cases=negative+3,
    sole_value_consumer_with_proven_full_write_reuse=True,
    real_transform_executed=True,selected_skips_original_norm=True,fallback_original_callables=True,
    original_bound_baseline_preserved=True,carry_object_identity=True,readable_generated_source=True,
    foreign_stream_falls_back=True,execution_failure_not_hidden=True)
print(json.dumps(result))
