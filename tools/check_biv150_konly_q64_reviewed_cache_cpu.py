#!/usr/bin/env python3
"""Exercise the actual installed cache and compiler hit-prefix without GPU imports."""
import argparse
import ast
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from abc import ABC, abstractmethod
from typing import Dict, Optional
import biv150_konly_q64_compile_source as c
import biv150_konly_q64_screen_contract as f
import biv150_konly_q64_reviewed_cache as r


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--execution-sources',type=Path,required=True);p.add_argument('--failed-v2',type=Path,required=True);a=p.parse_args()
    pre=f.preflight(a.raw,a.review,a.references);proof=f.execution_sources(a.execution_sources)
    cache=(a.execution_sources/'runtime/cache.py').read_bytes()
    c.need(c.sha(cache)==r.CACHE_SOURCE,'actual cache source SHA')
    knobs=types.SimpleNamespace(cache=types.SimpleNamespace(dir=None,manager_class=None),compilation=types.SimpleNamespace(always_compile=False,override=False,dump_ir=False,store_binary_only=False))
    ns=dict(json=json,os=os,ABC=ABC,abstractmethod=abstractmethod,Dict=Dict,Optional=Optional,base64=base64,knobs=knobs)
    tree=ast.parse(cache);tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('CacheManager','FileCacheManager','_base32','get_cache_manager')]
    exec(compile(tree,'actual-cache-readonly-subset','exec'),ns)
    # Execute the original contiguous compiler cache-hit/miss prefix. The key
    # has already been computed; real JIT/ASTSource key computation stays on device.
    comp=ast.parse((a.execution_sources/'spec/iluvatar/triton/compiler/compiler.py').read_bytes())
    function=next(n for n in comp.body if isinstance(n,ast.FunctionDef) and n.name=='compile')
    start=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='fn_cache_manager')
    end=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Expr) and ast.unparse(n.value).startswith('backend.add_stages('))
    original_prefix=function.body[start:end+1]
    c.need(original_prefix[0].lineno==284 and original_prefix[-1].lineno==324,'actual compiler prefix boundary')
    wrapper=ast.parse('def actual_prefix():\n    pass\n');wrapper.body[0].body=original_prefix
    compile_prefix=compile(ast.fix_missing_locations(wrapper),'actual-compiler-cache-prefix','exec')
    class Backend:
        def add_stages(self,*args):return 'restored-writer-stage-entry'
    original_add=Backend.add_stages;rejections=0
    def reject(call):
        nonlocal rejections
        try:call()
        except (ValueError,KeyError,TypeError,FileNotFoundError):rejections+=1;return
        raise AssertionError('invalid cache accepted')
    with tempfile.TemporaryDirectory(prefix='cjj-q64-cache-cpu-') as tmp:
        destination=Path(tmp)/'cache';groups=r.prepare(a.raw,destination,pre['records'],f.MANIFEST);knobs.cache.dir=str(destination)
        for role,g in groups.items():
            manager=ns['get_cache_manager'](g['hash']);r.validate_group(manager,g);r.validate_knobs(knobs,types.SimpleNamespace(FileCacheManager=ns['FileCacheManager']),g)
            def construct(src,group,h):
                c.need(group==g['paths'] and h==g['hash'],'actual prefix changed cache group')
                return types.SimpleNamespace(src=src,hash=h,metadata_group=group,kernel=Path(group[g['name']+'.cubin']).read_bytes())
            env=dict(ns,hash=g['hash'],src=types.SimpleNamespace(name=g['name'],language='triton'),backend=Backend(),CompiledKernel=construct,compilation_listener=None,target=None,options=types.SimpleNamespace(),env_vars={},__version__='3.6.0')
            exec(compile_prefix,env)
            with r.prohibit_backend_compile(Backend):
                kernel=env['actual_prefix']()
            c.need(Backend.add_stages is original_add and c.sha(kernel.kernel)==pre['records'][role]['artifacts']['cubin'],'cache hit exact cubin / restore')
            for setting in ('always_compile','override'):
                setattr(knobs.compilation,setting,True)
                reject(lambda:r.validate_knobs(knobs,types.SimpleNamespace(FileCacheManager=ns['FileCacheManager']),g))
                setattr(knobs.compilation,setting,False)
            knobs.cache.manager_class=Backend
            reject(lambda:r.validate_knobs(knobs,types.SimpleNamespace(FileCacheManager=ns['FileCacheManager']),g));knobs.cache.manager_class=None
            knobs.cache.dir=str(Path(tmp)/'other')
            reject(lambda:r.validate_knobs(knobs,types.SimpleNamespace(FileCacheManager=ns['FileCacheManager']),g));knobs.cache.dir=str(destination)
            # Actual cache silently omits absent children; our completeness check rejects all seven.
            for name,filename in g['paths'].items():
                path=Path(filename);body=path.read_bytes();path.unlink()
                reject(lambda:r.validate_group(manager,g));path.write_bytes(body+b'changed')
                reject(lambda:r.validate_group(manager,g));path.write_bytes(body)
            index=Path(manager.cache_dir)/('__grp__'+g['name']+'.json');index_bytes=index.read_bytes()
            redirected=dict(g['paths']);redirected[g['name']+'.cubin']=str(a.raw/'out'/role/'hoisted/actual.cubin')
            index.write_text(json.dumps({'child_paths':redirected}));reject(lambda:r.validate_group(manager,g));index.write_bytes(index_bytes)
            # A real miss in the actual compiler prefix must reach our throw before any stage.
            index.unlink()
            try:
                with r.prohibit_backend_compile(Backend):reject(env['actual_prefix'])
            finally:index.write_bytes(index_bytes)
            c.need(Backend.add_stages is original_add,'backend restoration after miss')
            env['hash']='0'*64
            with r.prohibit_backend_compile(Backend):reject(env['actual_prefix'])
            c.need(Backend.add_stages is original_add and Backend().add_stages()=='restored-writer-stage-entry','writer entry restored after changed key')
        actual=json.loads((a.failed_v2/'out/q32/record.json').read_text())
        c.need(actual['artifacts']['cubin']=='1100a3a36aa84607645570481016abc33fe36e2fb8e43e2cd327a36abe82fa19','real second stop binary')
        reject(lambda:f.check_loaded('q32',None,actual,pre['records']['q32']))
        summary={role:dict(hash=g['hash'],original_index_sha256=g['original_index_sha256'],sha256=g['sha256']) for role,g in groups.items()}
    # Exact inverse of the frontend-only cache changes keeps all math/timing/Graph code intact.
    current=(c.ROOT/'tools/probe_biv150_konly_q64_screen.py').read_text()
    anchors=[('import biv150_konly_q64_reviewed_cache as reviewed_cache\n',''),("        result['reviewed_cache']=reviewed_cache.prepare(a.raw,a.out/'triton-cache',pre['records'],f.MANIFEST);save()\n",''),("            kernel=reviewed_cache.warmup(fn,torch.cuda.current_device(),result['reviewed_cache'][role],\n                lambda:fn.warmup(grid=grid,**calls[role],**launch_options[role]));handles[role]=kernel","            kernel=fn.warmup(grid=grid,**calls[role],**launch_options[role]);handles[role]=kernel")]
    for before,after in anchors:c.need(current.count(before)==1,'exact frontend inverse anchor');current=current.replace(before,after)
    c.need(c.sha(current)=='0dbe005b40936548ebfdc30b5db8c4d761663bc677fe5c27728fc373efbbfe9a','5ce frontend exact inverse')
    c.need(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'CPU check imported framework')
    print(json.dumps(dict(completed=True,errors=[],negative_cases_rejected=rejections,actual_cache_prefix_lines=[284,324],original_prefix_AST_sha256=c.sha(ast.dump(ast.Module(body=original_prefix,type_ignores=[]))),cache_hit_loaded_original_cubin=True,cache_miss_stopped_before_backend_stages=True,backend_restored_after_success_and_failure=True,original_frontend_inverse=True,groups=summary,source_proof=proof,source_sha256={name:c.sha((c.ROOT/'tools'/name).read_bytes()) for name in ('biv150_konly_q64_reviewed_cache.py','check_biv150_konly_q64_reviewed_cache_cpu.py','biv150_konly_q64_screen_contract.py','probe_biv150_konly_q64_screen.py')},actual_GPU_cache_hit_verified=False,numeric_verified=False,performance_verified=False),sort_keys=True,indent=2))
if __name__=='__main__':main()
