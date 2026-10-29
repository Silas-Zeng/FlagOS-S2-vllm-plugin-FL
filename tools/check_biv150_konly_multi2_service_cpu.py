#!/usr/bin/env python3
"""Bounded host/proposal equivalence; mocks never grant device/service qualification."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace as NS
from check_biv150_konly_encoded_cpu import load, Tensor, Storage, Kernel, rejected
import check_biv150_konly_encoded_hoist_cpu as old_check

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'ce905346a0bd6938cd8478cde34c1bcc52eaeed7'
PACKAGE = ROOT/'vllm_fl/patches'

def original(name):
    return subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':vllm_fl/patches/biv150_konly_encoded_'+name+'.py']).decode()

def module(name, source):
    result=types.ModuleType(name);sys.modules[name]=result;exec(compile(source,name,'exec'),result.__dict__);return result

def same(a,b): return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

def method(tree,name):
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)

class DefaultScope(ast.NodeTransformer):
    """Specialize only the new fixed default; compare every other AST node."""
    def visit_FunctionDef(self,node):
        if node.name not in ('validate_meta','propose'): return node
        node=copy.deepcopy(node);node.args.kwonlyargs=[];node.args.kw_defaults=[]
        node.body=[n for n in node.body if not (
            isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)) and not (
            isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(
                isinstance(x,ast.Constant) and x.value=='Boolean multi2 scope' for x in n.value.args))]
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if any(isinstance(n,ast.Name) and n.id=='MULTI2_SCHEMA' for n in node.targets): return None
        return self.generic_visit(node)
    def visit_Name(self,node):
        return {'_multi2':ast.Constant(False),'profile':ast.Constant('q32'),
                'sequences':ast.Constant(1),'schema':ast.Name(id='SCHEMA',ctx=ast.Load())}.get(node.id,node)
    def visit_IfExp(self,node):
        node=self.generic_visit(node)
        return node.body if isinstance(node.test,ast.Constant) and node.test.value is True else node.orelse if isinstance(node.test,ast.Constant) and node.test.value is False else node
    def visit_Call(self,node):
        node=self.generic_visit(node);node.keywords=[k for k in node.keywords if k.arg!='_multi2'];return node

class DefaultAdapter(ast.NodeTransformer):
    VALUES={'PROFILE':'q32','BINDING_SCHEMA':'cjj-encoded-q32-live-binding-v1','LOG_PREFIX':'BIV150_KONLY_Q32_HOIST'}
    def visit_Assign(self,node):
        if any(isinstance(n,ast.Name) and n.id in self.VALUES for n in node.targets): return None
        return self.generic_visit(node)
    def visit_Attribute(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='self' and node.attr in self.VALUES:return ast.Constant(self.VALUES[node.attr])
        return self.generic_visit(node)
    def visit_BinOp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.op,ast.Add) and isinstance(node.left,ast.Constant) and isinstance(node.right,ast.Constant):return ast.Constant(node.left.value+node.right.value)
        return node

def run(compile_raw):
    a,h=load('q32_adapter'),load('hoist_proposal')
    old_a,old_h=module('original_adapter',original('q32_adapter')),module('original_proposal',original('hoist_proposal'))
    assert same(DefaultScope().visit(ast.parse((PACKAGE/'biv150_konly_encoded_hoist_proposal.py').read_text())),ast.parse(original('hoist_proposal')))
    assert same(DefaultAdapter().visit(ast.parse((PACKAGE/'biv150_konly_encoded_q32_adapter.py').read_text())),ast.parse(original('q32_adapter')))
    assert a.Adapter.__init__.__defaults__==old_a.Adapter.__init__.__defaults__ and a.Adapter.__init__.__kwdefaults__==old_a.Adapter.__init__.__kwdefaults__
    assert h.propose.__kwdefaults__==h.validate_meta.__kwdefaults__=={'_multi2':False}
    # A CPU package enables ordinary relative imports without importing vllm/Torch.
    name='_cjj_multi2_service_cpu';pkg=types.ModuleType(name);pkg.__path__=[str(PACKAGE)];sys.modules[name]=pkg
    sys.modules[name+'.biv150_konly_encoded_q32_adapter']=a;sys.modules[name+'.biv150_konly_encoded_hoist_proposal']=h
    spec=importlib.util.spec_from_file_location(name+'.biv150_konly_encoded_multi2_adapter',PACKAGE/'biv150_konly_encoded_multi2_adapter.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    files=old_check.fixture(a);text=files['out/unhoisted/actual.ttgir'].decode();meta=json.loads(files['out/proposal.json'])['proposal']['actual_metadata']
    assert h.propose(text,meta)==old_h.propose(text,meta)
    assert h.propose(None,{})==old_h.propose(None,{})
    for bad in (None,0,1,'true'):
        rejected(lambda:h.propose(text,meta,_multi2=bad));rejected(lambda:h.validate_meta(text,meta,_multi2=bad))
    # Same saved artifacts and stand-in binder as original tests. This is ONLY host protocol.
    fake_meta=copy.deepcopy(meta);fake_meta.update(schema=m.Proposal.SCHEMA,profile='multi2',grid=[66,2]);fake_meta['constants']['num_seqs']=2
    query=Tensor(Storage(1,0x1000000,2048*2048*2),'bfloat16',(2048,16,128),(2048,128,1))
    launch=dict(fake_meta['constants'],query_ptr=query,num_warps=16,num_stages=1)
    with tempfile.TemporaryDirectory(prefix='multi2-service-cpu-') as temp,contextlib.redirect_stdout(io.StringIO()):
        fn=old_check.MockFunction(files)
        adapter=m.Adapter(m.Proposal,load('hoist_native'),directory=temp,target=fake_meta['target'],pins=m.Proposal.PINS,implementation_digest='CPU-protocol',jit=old_check.MockJIT,device=0,fn=fn)
        opts,binding=adapter.prepare(fn,launch,(66,2))
        assert binding is not None and binding['schema']=='cjj-encoded-multi2-live-binding-v1'
        assert binding['actual_metadata']==fake_meta and not binding['numerical_verified']
        assert len(fn.calls)==2 and adapter.prepare(fn,launch,(66,2))[1] is binding
        assert adapter.prepare(fn,launch,(66,2),capture=True)==({},None) and len(fn.calls)==2
        for wrong in (dict(launch,num_seqs=1),dict(launch,num_seqs=True),dict(launch,IS_3D=True)):
            assert adapter.prepare(fn,wrong,(66,2))==({},None)
        assert adapter.prepare(fn,launch,(65,2))==({},None)
        adapter.after_launch(fn.after,fn,binding)
        rejected(lambda:adapter.after_launch(copy.copy(fn.after),fn,binding))
        assert adapter.counts['cold_audits']==1
    # The route body up to selecting fn is byte-identical in AST, including capture fallback.
    r=load('runtime');tree=ast.parse((PACKAGE/'biv150_konly_encoded_runtime.py').read_text());prior=ast.parse(original('runtime'))
    iv,old_iv=method(tree,'invoke'),method(prior,'invoke')
    def split(n):return next(i for i,v in enumerate(n.body) if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fn' for t in v.targets))
    assert same(ast.Module(body=iv.body[:split(iv)],type_ignores=[]),ast.Module(body=old_iv.body[:split(old_iv)],type_ignores=[]))
    assert same(next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Registry'),next(n for n in ast.walk(prior) if isinstance(n,ast.ClassDef) and n.name=='Registry'))
    tail=ast.fix_missing_locations(ast.Module(body=copy.deepcopy(iv.body[split(iv):-1]),type_ignores=[]));code=compile(tail,'<actual-dispatch-tail>','exec')
    for profile in ('q32','multi2','normalized2d128'):
        for enabled in (False,True):
            for fail_after in (False,True):
                calls=[];runtime=r.Runtime(NS(cuda=NS(is_current_stream_capturing=lambda:False)),None,{profile:Kernel()},NS());runtime.note=lambda *args:None
                def after(*args):
                    calls.append('after')
                    if fail_after:raise ValueError('postlaunch')
                probe=NS(prepare=lambda *args,**kwargs:(calls.append('prepare') or ({'ir_override':'CPU'},{})),after_launch=after)
                if enabled and profile in ('q32','multi2'):setattr(runtime,profile+'_hoist',probe)
                active=enabled and profile!='normalized2d128';scope=dict(runtime=runtime,profile=profile,selected=launch,target_grid=(66,2))
                if active and fail_after:rejected(lambda:exec(code,scope))
                else:exec(code,scope)
                assert len(runtime.kernels[profile].calls)==1 and calls==(['prepare','after'] if active else [])
    for profile in ('q32','multi2'):
        runtime=r.Runtime(NS(cuda=NS(is_current_stream_capturing=lambda:False)),None,{profile:Kernel()},NS());runtime.note=lambda *args:None
        setattr(runtime,profile+'_hoist',NS(prepare=lambda *a,**kw:({},None),after_launch=lambda *a:(_ for _ in ()).throw(AssertionError('fallback after_launch'))))
        exec(code,dict(runtime=runtime,profile=profile,selected=launch,target_grid=(66,2)))
        assert len(runtime.kernels[profile].calls)==1 and runtime.kernels[profile].calls[0][2]==launch
    # Default stats/log bytes and cache facts are unchanged for identical package inputs.
    old_r=module('original_runtime',original('runtime'))
    results=[]
    for implementation in (old_r,r):
        runtime=implementation.Runtime(NS(cuda=NS(is_current_stream_capturing=lambda:False)),None,{},NS())
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):runtime.note('multi2',NS(hash='CPU'),launch,(66,2))
        results.append((capture.getvalue(),runtime.stats(),runtime.kernel_records))
    assert results[0]==results[1]
    cache_source='vllm_fl/compile_semantics.py'
    old_cache=module('original_cache',subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':'+cache_source]).decode())
    new_cache=module('new_cache',(ROOT/cache_source).read_text());config=NS(model_config=NS(hf_config=NS(**new_cache._GEOMETRY)))
    env={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1'}
    def fingerprint(implementation,env):return implementation.build_fingerprint(config,'iluvatar',package_root=ROOT/'vllm_fl',environ=env,abi={'CPU':True})
    assert fingerprint(new_cache,env)==fingerprint(old_cache,env)==fingerprint(new_cache,dict(env,**{m.ENV:'0'}))
    assert fingerprint(new_cache,env)!=fingerprint(new_cache,dict(env,**{m.ENV:'1'}))
    # Off import behavior and generated-source identities; no new GPU source or accepted Native edits.
    service=load('service');assert service.install(object,environ={m.ENV:'1'})==dict(installed=False,reason='disabled')
    provenance=json.loads((PACKAGE/'biv150_konly_encoded_sources.json').read_text())
    for record in provenance['generated'].values():assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
    old_manifest=json.loads(subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':vllm_fl/patches/biv150_konly_encoded_sources.json']))
    normalized=copy.deepcopy(provenance);del normalized['generated']['multi2_adapter'];normalized['generated']['hoist_proposal']=old_manifest['generated']['hoist_proposal'];assert normalized==old_manifest
    for suffix in ('multi2','q32','writer','views','native','mixed_native','hoist_native'):
        assert (PACKAGE/('biv150_konly_encoded_'+suffix+'.py')).read_text()==original(suffix)
    changed=set(subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',PARENT,'--','vllm_fl','csrc']).decode().splitlines())
    allowed={'vllm_fl/compile_semantics.py'}|{'vllm_fl/patches/biv150_konly_encoded_'+n for n in ('hoist_proposal.py','q32_adapter.py','multi2_adapter.py','runtime.py','service.py','sources.json')}
    assert changed<=allowed
    actual=None
    if compile_raw:
        raw=Path(compile_raw);manifest=raw/'download-manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='8b6af5b1aa715d449b95ae4826793afb8d6627de4d990eed3cba4cb2765b2391'
        for row in json.loads(manifest.read_text())['files']:
            p=raw/row['relative'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        base=raw/'out/multi2';meta=json.loads((base/'proposal.json').read_text())['actual_metadata'];text=(base/'unhoisted/actual.ttgir').read_text()
        frozen=module('frozen_multi2_proposal',(raw/'out/sources/multi2_hoist_proposal.py').read_text())
        moved,proof=m.Proposal.propose(text,meta)
        assert (moved,proof)==frozen.propose(text,meta) and moved==(base/'override.ttgir').read_text() and proof==json.loads((base/'proposal.json').read_text())
        for key,bad in (('profile','q32'),('schema',h.SCHEMA),('query_rows',True),('grid',[65,2])):
            changed=copy.deepcopy(meta);changed[key]=bad;rejected(lambda:m.Proposal.propose(text,changed))
        actual=dict(original_compile_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),actual_old_multi2_proposal_exact=True,new_device_compilation=False)
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,default_Q32_complete_AST_and_actual_proposal_equal=True,
        fixed_multi2_namespace_only=True,default_runtime_stats_log_and_cache_facts_equal=True,enabled_cache_flag_separates=True,original_route_and_Registry_AST_equal=True,one_launch_no_postlaunch_retry=True,
        mock_binder_only=True,actual_compile_replay=actual,generated_GPU_source_changed=False,
        new_source_device_verified=False,precision_gate=False,service_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compile-raw');p.add_argument('--out',required=True);args=p.parse_args()
    result=run(args.compile_raw);Path(args.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(out=args.out,sha256=hashlib.sha256(Path(args.out).read_bytes()).hexdigest())))
