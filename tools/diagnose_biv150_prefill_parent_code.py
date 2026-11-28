#!/usr/bin/env python3
"""Default stdlib compile-only; --loaded imports parents but never installs/JITs/launches."""
import argparse,ast,copy,dis,hashlib,importlib,json,sys,types
from pathlib import Path
sys.dont_write_bytecode=True
NAMES=('biv150_prefill_multi2','biv150_prefill_q32','biv150_q_fragment_service')
FIELDS=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags',
        'co_code','co_names','co_varnames','co_filename','co_name','co_qualname','co_firstlineno',
        'co_linetable','co_exceptiontable','co_freevars','co_cellvars')
def sha(b):return hashlib.sha256(b).hexdigest()
def child(code,name):
    found=[c for c in code.co_consts if isinstance(c,types.CodeType)and c.co_name==name]
    assert len(found)==1,(name,len(found));return found[0]
def expected(raw,filename,name,partial):
    target='_eligible'if name.endswith('fragment_service')else'_Policy'
    if partial:
        n=next(n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==target)
        raw=ast.Module(body=[copy.deepcopy(n)],type_ignores=[])
    module=compile(raw,filename,'exec',flags=0,dont_inherit=True,optimize=sys.flags.optimize)
    code=child(module,target)
    return code if target=='_eligible'else child(code,'__getitem__')
def value(v):
    if isinstance(v,bytes):return dict(type='bytes',bytes=len(v),sha256=sha(v))
    if isinstance(v,types.CodeType):return dict(type='code',name=v.co_name,qualname=v.co_qualname)
    if isinstance(v,tuple):return dict(type='tuple',items=[value(x)for x in v])
    if type(v)in(str,int,float,bool,type(None)):return dict(type=type(v).__name__,value=v)
    return dict(type=type(v).__name__)
def diff(a,b,path='root'):
    result=[]
    for f in FIELDS:
        av,bv=getattr(a,f),getattr(b,f)
        if av!=bv:result.append(dict(path=path+'.'+f,actual=value(av),expected=value(bv)))
    if len(a.co_consts)!=len(b.co_consts):result.append(dict(path=path+'.co_consts.length',actual=len(a.co_consts),expected=len(b.co_consts)))
    for i,(av,bv)in enumerate(zip(a.co_consts,b.co_consts)):
        p=path+f'.co_consts[{i}]'
        if isinstance(av,types.CodeType)and isinstance(bv,types.CodeType):result.extend(diff(av,bv,p+'('+av.co_name+')'))
        elif type(av)is not type(bv)or av!=bv:result.append(dict(path=p,actual=value(av),expected=value(bv)))
    return result
def instruction_diffs(a,b,path='root'):
    output=[]
    aa,bb=list(dis.get_instructions(a)),list(dis.get_instructions(b))
    for av,bv in zip(aa,bb):
        if(av.opname,av.arg,av.argval)!=(bv.opname,bv.arg,bv.argval):
            output.append(dict(path=path,actual=dict(offset=av.offset,opname=av.opname,arg=av.arg,argrepr=av.argrepr),
                expected=dict(offset=bv.offset,opname=bv.opname,arg=bv.arg,argrepr=bv.argrepr)))
    for i,(av,bv)in enumerate(zip(a.co_consts,b.co_consts)):
        if isinstance(av,types.CodeType)and isinstance(bv,types.CodeType):output.extend(instruction_diffs(av,bv,path+f'.co_consts[{i}]('+av.co_name+')'))
    return output[:24]
def signature(code):
    r={k:value(getattr(code,k))for k in FIELDS}
    r['co_consts']=[signature(v)if isinstance(v,types.CodeType)else value(v)for v in code.co_consts];return r
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--loaded',action='store_true');a=p.parse_args()
root=a.source.resolve();report=dict(python=sys.version,executable=sys.executable,optimize=sys.flags.optimize,debug=__debug__,
    loaded_mode=a.loaded,script_installs_called=False,script_JIT_called=False,script_GPU_calls=False,records=[])
if a.loaded:sys.path.insert(0,str(root))
for name in NAMES:
    path=root/'vllm_fl/patches'/(name+'.py');raw=path.read_bytes();fn=None;module=None
    if a.loaded:
        module=importlib.import_module('vllm_fl.patches.'+name)
        fn=module._eligible if name.endswith('fragment_service')else module._Policy.__getitem__
    filename=fn.__code__.co_filename if fn else str(path)
    full,partial=expected(raw,filename,name,False),expected(raw,filename,name,True)
    actual=fn.__code__ if fn else full
    r=dict(name=name,pinned_source_path=str(path),pinned_source_sha256=sha(raw),
        module_future_features=[alias.name for n in ast.parse(raw).body if isinstance(n,ast.ImportFrom)and n.module=='__future__'for alias in n.names],
        actual_vs_full_equal=actual==full,actual_vs_old_partial_equal=actual==partial,
        full_vs_old_partial_equal=full==partial,actual_vs_full_differences=diff(actual,full),
        actual_vs_old_partial_differences=diff(actual,partial),
        full_vs_old_partial_instruction_differences=instruction_diffs(full,partial),
        actual_code=signature(actual),full_expected_code=signature(full),old_partial_expected_code=signature(partial))
    if module:
        loadedpath=Path(module.__file__).resolve()
        r.update(loaded_module_file=str(loadedpath),loaded_source_sha256=sha(loadedpath.read_bytes()),
            loaded_module_name=module.__name__,loader_type=type(module.__loader__).__name__,
            spec_origin=module.__spec__.origin,module_cached=getattr(module,'__cached__',None),
            function_module=fn.__module__,function_qualname=fn.__qualname__,
            function_closure=[type(c.cell_contents).__name__ for c in(fn.__closure__ or ())],
            function_defaults=value(fn.__defaults__),function_kwdefaults={k:value(v)for k,v in(fn.__kwdefaults__ or {}).items()},
            globals_file=fn.__globals__.get('__file__'),globals_name=fn.__globals__.get('__name__'))
    report['records'].append(r)
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(out=str(a.out),sha256=sha(a.out.read_bytes()),loaded=a.loaded,
    equal_full=[r['actual_vs_full_equal']for r in report['records']],equal_partial=[r['actual_vs_old_partial_equal']for r in report['records']])))
