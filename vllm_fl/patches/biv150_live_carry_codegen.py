"""Pair an audited generated norm producer with its sole native gate consumer.

Dispatch happens in the generated Python Runner, after Dynamo/FX partitioning.
All unsupported inputs execute the exact original kernel and gate callables.
"""
import ast
import copy
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import types

EXPECTED_CODECACHE_SHA256='d8dcb8ce11fa4be5a7f97aca9933b3ed986b08f9857e19b3328e4bd60bb2c352'
EXPECTED_NORM_SHA256='473b43506d9a4bdd98ef096830092c771bff0b0e1381606520f066d9b2aec4a8'
_HELPER='_biv150_generated_norm_gate_pair'
_TORCH=None
_SELECTIONS=0
_PATCHED=[]


def _canonical(node):
    if isinstance(node,ast.AST):return [type(node).__name__,[[k,_canonical(v)] for k,v in ast.iter_fields(node)]]
    if isinstance(node,list):return [_canonical(v) for v in node]
    return node


def norm_hash(source):
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1:return None
    fn=functions[0];fn.name='audited_norm';fn.decorator_list=[]
    return hashlib.sha256(json.dumps(_canonical(fn),separators=(',',':')).encode()).hexdigest()


def transform(source):
    """Return a standalone rewritten Runner.call, or None on nonmatching code."""
    if 'gate_silu_auto.default' not in source or 'class Runner' not in source:return None
    module=ast.parse(source);kernels=set()
    for stmt in module.body:
        if not (isinstance(stmt,ast.Assign) and len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name)
            and isinstance(stmt.value,ast.Call) and ast.unparse(stmt.value.func)=='async_compile.triton'):continue
        args=stmt.value.args
        if len(args)>=2 and isinstance(args[1],ast.Constant) and isinstance(args[1].value,str):
            if norm_hash(args[1].value)==EXPECTED_NORM_SHA256:kernels.add(stmt.targets[0].id)
    if not kernels:return None
    runners=[n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Runner']
    if len(runners)!=1:return None
    calls=[n for n in runners[0].body if isinstance(n,ast.FunctionDef) and n.name=='call']
    if len(calls)!=1 or calls[0].decorator_list:return None
    fn=copy.deepcopy(calls[0]);pairs=[]
    for parent in list(ast.walk(fn)):
        for _,body in ast.iter_fields(parent):
            if not isinstance(body,list) or not body or not all(isinstance(n,ast.stmt) for n in body):continue
            for i,stmt in list(enumerate(body)):
                if not (isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call)):continue
                norm=stmt.value
                if not (isinstance(norm.func,ast.Attribute) and norm.func.attr=='run'
                    and isinstance(norm.func.value,ast.Name) and norm.func.value.id in kernels
                    and len(norm.args)==6 and isinstance(norm.args[3],ast.Name)
                    and isinstance(norm.args[5],ast.Constant) and norm.args[5].value==2048
                    and len(norm.keywords)==1 and norm.keywords[0].arg=='stream'):continue
                j=i+1
                while j<len(body) and isinstance(body[j],ast.Delete):j+=1
                if j>=len(body) or not isinstance(body[j],ast.Assign):continue
                assignment=body[j];gate=assignment.value
                if not (len(assignment.targets)==1 and isinstance(assignment.targets[0],ast.Name)
                    and isinstance(gate,ast.Call) and ast.unparse(gate.func)=='torch.ops.biv150_native.gate_silu_auto.default'
                    and len(gate.args)==5 and not gate.keywords
                    and isinstance(gate.args[0],ast.Name) and gate.args[0].id==norm.args[3].id
                    and isinstance(gate.args[4],ast.Constant) and gate.args[4].value==17):continue
                output=norm.args[3].id
                uses=[n for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id==output]
                if len(uses)!=2:continue
                allocations=[n for n in ast.walk(fn) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id==output for t in n.targets)]
                if len(allocations)!=1:continue
                allocation=allocations[0].value
                if not (isinstance(allocation,ast.Call) and ast.unparse(allocation.func)=='empty_strided_cuda'
                    and len(allocation.args)==3 and not allocation.keywords
                    and ast.unparse(allocation.args[0])==f'({ast.unparse(norm.args[4])}, 2048)'
                    and ast.unparse(allocation.args[1])=='(2048, 1)'
                    and ast.unparse(allocation.args[2])=='torch.bfloat16'):continue
                replacement=ast.Call(func=ast.Name(id=_HELPER,ctx=ast.Load()),args=[
                    copy.deepcopy(norm.func),copy.deepcopy(gate.func),
                    ast.Tuple(elts=copy.deepcopy(norm.args),ctx=ast.Load()),
                    ast.Dict(keys=[ast.Constant(k.arg) for k in norm.keywords],values=[copy.deepcopy(k.value) for k in norm.keywords]),
                    ast.Tuple(elts=copy.deepcopy(gate.args),ctx=ast.Load())],keywords=[])
                body[i]=ast.copy_location(ast.Assign(targets=copy.deepcopy(assignment.targets),value=replacement),stmt)
                body[j]=ast.copy_location(ast.Pass(),assignment)
                pairs.append(dict(kernel=norm.func.value.id,norm_line=stmt.lineno,gate_line=assignment.lineno,
                    normalized_buffer=output,output=assignment.targets[0].id,sole_consumer=True))
    if not pairs:return None
    return dict(function_source=ast.unparse(ast.fix_missing_locations(fn))+'\n',pairs=pairs)


def supported(norm_args,gate_args):
    if len(norm_args)!=6 or len(gate_args)!=5 or type(norm_args[4]) is not int or norm_args[4]!=64:return False
    x,residual,weight,out,_,hidden=norm_args;_,wq,sw,marker,threshold=gate_args;t=_TORCH
    if hidden!=2048 or threshold!=17:return False
    tensors=(x,residual,weight,out,wq,sw,marker)
    if not all(v.is_cuda and v.device==x.device for v in tensors):return False
    if any(v.dtype!=t.bfloat16 for v in (x,residual,weight,out,marker)):return False
    if not (tuple(x.shape)==tuple(residual.shape)==tuple(out.shape)==(64,2048)
        and tuple(weight.shape)==(2048,) and all(v.is_contiguous() for v in (x,residual,weight,out))):return False
    return (tuple(wq.shape)==(2048,12288) and wq.dtype==t.int8 and wq.is_contiguous()
        and tuple(sw.shape)==(12288,) and sw.dtype==t.float32 and sw.is_contiguous()
        and tuple(marker.shape)==(12288,2048) and marker.stride()==(0,0))


def paired(norm_run,gate,norm_args,norm_kwargs,gate_args):
    global _SELECTIONS
    if supported(norm_args,gate_args):
        stream=_TORCH.cuda.current_stream(norm_args[0].device).cuda_stream
        if set(norm_kwargs)=={'stream'} and norm_kwargs['stream']==stream:
            from vllm_fl.patches.biv150_live_carry_gate import gate_m64
            _SELECTIONS+=1  # Host invocations/capture only, not Graph replays.
            return gate_m64(norm_args[0],norm_args[1],norm_args[2],gate_args[1],gate_args[2])
    norm_run(*norm_args,**norm_kwargs)
    return gate(*gate_args)


def patch_module(module,path):
    if getattr(module,'_biv150_live_carry_pair',False):return module
    source=Path(path).read_text();change=transform(source)
    if change is None:return module
    cls=getattr(module,'Runner',None);bound=getattr(module,'call',None)
    if (cls is None or not isinstance(bound,types.MethodType) or type(bound.__self__) is not cls
        or bound.__func__ is not cls.call):return module
    if _HELPER in vars(module) and vars(module)[_HELPER] is not paired:
        raise RuntimeError('Foreign generated norm/gate helper binding')
    # Persistent readable source, outside the immutable package and distinct from
    # the original cached code. Never overwrite that cached module's source.
    with tempfile.NamedTemporaryFile(mode='w',suffix='.py',prefix='biv150-live-carry-',dir=Path(path).parent,delete=False) as file:
        file.write(change['function_source']);generated=file.name
    vars(module)[_HELPER]=paired;local={}
    exec(compile(change['function_source'],generated,'exec'),vars(module),local)
    new=local['call'];new.__qualname__=cls.call.__qualname__
    cls.call=new;module.call=types.MethodType(new,bound.__self__)
    module._biv150_live_carry_pair=True
    evidence=dict(
        source_file=str(path),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        generated_file=generated,generated_sha256=hashlib.sha256(change['function_source'].encode()).hexdigest(),
        pairs=change['pairs'],original_transparent_carry=True)
    _PATCHED.append(evidence)
    print('BIV150_LIVE_CARRY_CODEGEN_PAIR '+json.dumps(evidence),flush=True)
    return module


def install():
    global _TORCH
    if os.environ.get('VLLM_FL_BIV150_LIVE_CARRY_GATE','1')!='1':return dict(installed=False,reason='disabled')
    import torch
    from torch._inductor.codecache import PyCodeCache
    from vllm.model_executor.models.llama import LlamaDecoderLayer
    _TORCH=torch
    if getattr(LlamaDecoderLayer,'_biv150_live_carry',False):
        return dict(installed=False,reason='legacy decoder rewrite is active')
    descriptor=PyCodeCache.__dict__['load_by_key_path']
    if not isinstance(descriptor,classmethod):return dict(installed=False,reason='unexpected cache entry descriptor')
    original=descriptor.__func__
    if getattr(original,'_biv150_live_carry_pair',False):return dict(installed=True,already=True)
    path=Path(inspect.getsourcefile(original))
    if (original.__module__!='torch._inductor.codecache' or original.__code__.co_filename!=str(path)
        or hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_CODECACHE_SHA256):
        return dict(installed=False,reason='unaudited or wrapped cache loader')
    @functools.wraps(original)
    def load(cls,key,path,linemap=None,attrs=None):
        module=original(cls,key,path,linemap=linemap,attrs=attrs)
        return patch_module(module,path)
    load._biv150_live_carry_pair=True;PyCodeCache.load_by_key_path=classmethod(load)
    return dict(installed=True,exact_rows=[64],epsilon=1e-6,
        original_transparent_carry=True,persistent_weight_copy=False,
        original_decoder_forward_unchanged=True,hook_target='PyCodeCache.load_by_key_path',
        expected_norm_sha256=EXPECTED_NORM_SHA256,expected_codecache_sha256=EXPECTED_CODECACHE_SHA256,
        hook_file=str(Path(__file__).resolve()),hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
