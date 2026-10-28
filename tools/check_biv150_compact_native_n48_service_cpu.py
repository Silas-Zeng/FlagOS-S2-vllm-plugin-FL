#!/usr/bin/env python3
"""Finite CPU integration checks; real source/ELFs, mocked tensor/dispatcher metadata."""
import argparse, ast, contextlib, copy, hashlib, importlib.util, io, json, os, subprocess, sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/private/tmp/cjj-native110-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_v_overlap_probe.so')
N48=Path('/private/tmp/cjj-native112-n48-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_s4_probe.so')
def sha(b):return hashlib.sha256(b).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def source_defs(path):
    text=path.read_text();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--base-library',type=Path,default=BASE);p.add_argument('--n48-library',type=Path,default=N48);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    q=load('_native111_q',ROOT/'vllm_fl/compact_native_qualification.py')
    assert q.preflight({}) is None
    identity=load('_native111_identity',ROOT/'vllm_fl/patches/biv150_konly_compact_identity.py');identity.verify_package()
    controller=load('_native111_controller',ROOT/'tools/run_biv150_compact_native_n48_service.py')
    helper_path=ROOT/'docs/evidence/compact-service-q32-combination-20261029/actual-owned-service-helper94db.py'
    assert sha(helper_path.read_bytes())==controller.HELPER_SHA
    helper=load('_native111_actual_helper',helper_path)
    evidence=ROOT/'docs/evidence/compact-service-native-n48-cpu-20261029'
    parent=evidence/'parent109-controller.py';assert sha(parent.read_bytes())==controller.PARENT_CONTROLLER_SHA
    before=source_defs(parent);after=source_defs(ROOT/'tools/run_biv150_compact_native_n48_service.py')
    changed={k for k in before if before[k]!=after.get(k)}
    assert changed=={'native_env','native_events','evidence','run','graph_bootstrap_environment'}
    assert after['graph_bootstrap_environment'].replace('compact Native7892/2ad','compact40f9')==before['graph_bootstrap_environment']
    unchanged=sorted(set(before)-changed)
    assert {'validate_score','full_workload_only_throughput','full_geometry_throughput','warmup_plan_metadata',
      'OwnedBenchmarkAdapter','SupplementalMath64Adapter','supplemental64_command','graceful_api_shutdown'}<=set(unchanged)
    flags=controller.native_env({controller.ENABLE:'1'},a.base_library,q32_hoist=True)
    assert flags[controller.HOIST_ENABLE]=='1' and flags[controller.N48_ENABLE]=='0'
    base_config=q.preflight(flags);assert base_config['sha256']==controller.K8_SHA and base_config['n48'] is None
    flags48=controller.native_env({controller.ENABLE:'1'},a.base_library,n48_library=a.n48_library,q32_hoist=True)
    assert flags48[controller.HOIST_ENABLE]=='1' and flags48[controller.N48_ENABLE]=='1'
    off=controller.native_env(dict(flags48,**{controller.ENABLE:'0'}));assert off[controller.HOIST_ENABLE]==off[controller.N48_ENABLE]==off[controller.K8_ENABLE]=='0'
    assert controller.N48_PATH not in off and controller.N48_HASH not in off
    rejected=[]
    def reject(name,fn,contains=None):
        try:fn()
        except (RuntimeError,ValueError) as exc:
            if contains:assert contains in str(exc),str(exc)
            rejected.append(dict(name=name,reason=str(exc)))
        else:raise AssertionError('Accepted '+name)
    assert q.N48_QUALIFICATION['state']=='component-reviewed'
    combined_config=q.preflight(flags48)
    assert combined_config['n48']['qualification']==q.pin('n48')
    assert combined_config['n48']['sha256']==controller.N48_SHA
    args=NS(source=ROOT,native_k8_library=a.base_library,native_k8_n48_library=a.n48_library,baseline=False,q32_hoist=True)
    assert controller.component_source_preflight(args)==combined_config
    assert controller.component_qualification_status(ROOT)==dict(state='component-reviewed',pending=False,service_verified=False)
    # Real historical pending bytes: negative fixture only, never a positive override.
    pending_file=ROOT/'docs/evidence/compact-service-native-n48-qualified-cpu-20261029/pending-qualification-26e3.py'
    frozen=subprocess.check_output(['git','-C',str(ROOT),'show','26e3b198e131dc3b8c3c6389e822f5c4dbbc0845:vllm_fl/compact_native_qualification.py'])
    assert pending_file.read_bytes()==frozen
    pending=load('_native111_historical_pending',pending_file)
    reject('historical N48 pending preflight',lambda:pending.preflight(dict(flags48,**{controller.HOIST_ENABLE:'0'})),'PENDING n48')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='native111-real-pending-CPU-') as tmp:
        root=Path(tmp);(root/'vllm_fl').mkdir();(root/'vllm_fl/compact_native_qualification.py').write_bytes(frozen)
        pending_args=NS(**dict(vars(args),source=root,q32_hoist=False))
        reject('controller pending stops before helper/idle/output',lambda:controller.run(pending_args),'PENDING n48')
        assert controller.component_qualification_status(root)==dict(state='pending',pending=True,service_verified=False)
    reject('N48 missing explicit SHA',lambda:q.preflight(dict(flags48,**{q.N48_SHA_ENV:None})),'explicit actual library SHA pair')
    reject('N48 wrong explicit SHA',lambda:q.preflight(dict(flags48,**{q.N48_SHA_ENV:'0'*64})),'explicit actual library SHA pair')
    reject('N48 wrong actual library',lambda:q.preflight(dict(flags48,**{q.N48_PATH_ENV:str(a.base_library)})),'actual compact library contents')
    reject('N48 without base',lambda:controller.native_env({controller.ENABLE:'1'},n48_library=a.n48_library))
    reject('Q32 baseline',lambda:controller.native_env({controller.ENABLE:'0'},q32_hoist=True))
    reject('wrong actual library',lambda:q.preflight(dict(flags,**{q.PATH_ENV:str(a.n48_library)})),'actual compact library contents')
    for path,pin in ((a.base_library,controller.K8_SHA),(a.n48_library,controller.N48_SHA)):
        assert sha(path.read_bytes())==pin and path.read_bytes()[:6]==b'\x7fELF\x02\x01'
    # Real helper94db resolves this overlay without importing/executing vllm_fl.
    resolution=helper.package_resolution(dict(os.environ,PYTHONPATH=str(ROOT)),ROOT)
    assert Path(resolution['origin']).resolve()==ROOT/'vllm_fl/__init__.py' and resolution['package_executed'] is False
    fingerprints=[]
    for native_on,hoist in ((False,False),(False,True),(True,False),(True,True)):
        env=controller.native_env({controller.ENABLE:'1','VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0'},a.base_library if native_on else None,q32_hoist=hoist)
        value=helper.expected_fingerprint(ROOT,env,{'CPU_metadata_fixture':True})
        assert value['konly_compact']['q32_hoist_enabled'] is hoist
        if native_on:assert value['konly_compact']['native_decode']['sha256']==controller.K8_SHA
        fingerprints.append(value)
    assert len({json.dumps(v,sort_keys=True) for v in fingerprints})==4
    for hoist in (False,True):
        env=controller.native_env({controller.ENABLE:'1','VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0'},a.base_library,n48_library=a.n48_library,q32_hoist=hoist)
        value=helper.expected_fingerprint(ROOT,env,{'CPU_metadata_fixture':True})
        assert value['konly_compact']['native_decode']['n48']['qualification']==q.pin('n48')
        assert value['konly_compact']['q32_hoist_enabled'] is hoist
        fingerprints.append(value)
    assert len({json.dumps(v,sort_keys=True) for v in fingerprints})==6
    assert 'vllm_fl' not in sys.modules and not {'torch','triton','numpy','vllm'}&set(sys.modules)
    # Existing CPU fixture reuses exact tensor/storage and compact role validation.
    package=ModuleType('vllm_fl');package.__path__=[str(ROOT/'vllm_fl')];sys.modules['vllm_fl']=package
    import check_biv150_compact_native_qualified_service_cpu as existing
    old=existing.old
    native=existing.compact_load('native')
    class RegisteredNamespace:
        def __init__(self,owner,path,op):self.owner,self.path,self.op=owner,path,op
        @property
        def produce(self):
            if self.path not in self.owner.loaded_libraries:raise AttributeError('namespace not registered')
            return self.op
    class DualOps:
        def __init__(self,schema):
            assert schema==native.SCHEMA
            self.loaded_libraries=set();self.dispatch_queries=[]
            self.op=old.Op(schema);self.n48_op=old.Op(native.N48_SCHEMA)
            self.cjj_native_k8_compact_v_overlap_probe=RegisteredNamespace(self,str(a.base_library.resolve()),self.op)
            self.cjj_native_k8_compact_n48_s4_probe=RegisteredNamespace(self,str(a.n48_library.resolve()),self.n48_op)
            self.cjj_native_k8_v_overlap_probe=self.cjj_native_k8_compact_v_overlap_probe
        def load_library(self,path):
            assert path in (str(a.base_library.resolve()),str(a.n48_library.resolve()))
            self.loaded_libraries.add(path)
        def has_kernel(self,operator,key):
            self.dispatch_queries.append((operator,key))
            namespace=getattr(self,operator.split('::')[0])
            return key=='CUDA' and namespace.path in self.loaded_libraries
    fixture,fixture_proof=existing.adapted_fixture();fixture.__globals__['Ops']=DualOps
    class NoPrefill:
        by_hash={}
        def validate_call(self,*args):raise AssertionError('Decode reached Q32')
        def prepare(self,*args,**kwargs):raise AssertionError('Decode reached Q32')
    # Use the unmodified shipping picker; reducer receives the same local segments.
    text=(ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text();tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick_segments']
    assignments=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_TARGET_ENV','TARGET_PROGRAMS') for t in n.targets)]
    ns={'os':NS(environ={})};exec(compile(ast.Module(body=assignments+nodes,type_ignores=[]),'<actual shipping picker/default environment>','exec'),ns)
    picked={n:ns['_pick_segments'](n,2,16) for n in (1,8,16,32,40,48,64)}
    assert picked[32]==picked[48]==4 and picked[64]==2 and picked[40]==16
    assert 'num_segments=segments,' in text and 'SEGMENTS=segments,' in text
    cases=[];fallbacks=[]
    for n in (1,8,16,32,40,48,64):
        s=picked[n]
        for pitch in (2048,2560):
            runtime,entry,kw,producer,kernels,ops,capture,_,_=fixture(native,n,s,pitch,a.base_library)
            runtime.q32_hoist=NoPrefill()
            output=io.StringIO()
            with contextlib.redirect_stdout(output):producer((n,2,s),**kw);capture[0]=True;producer((n,2,s),**kw);capture[0]=False
            assert kw['SEGMENTS']==s
            if n in (32,64):
                assert len(ops.op.calls)==2 and not kernels['decode'].calls
                assert all(call[-1]==s for call in ops.op.calls)
                assert ops.op.calls[0][1].stride(0)==12544 and ops.op.calls[0][3].data_ptr()-ops.op.calls[0][1].data_ptr()==4352
                assert len(controller.native_events(output.getvalue(),runtime.native_decode.library))==2
                reject('stride '+str((n,pitch)),lambda:producer((n,2,s),**dict(kw,block_table_stride=1)))
                ops.op.throw=True;reject('postlaunch no retry '+str((n,pitch)),lambda:producer((n,2,s),**kw))
                assert len(ops.op.calls)==3 and not kernels['decode'].calls
                cases.append(dict(batch=n,segments=s,pitch=pitch,real_adapter_metadata_path=True))
            else:
                assert not ops.op.calls and len(kernels['decode'].calls)==2
                assert all(call[0]==(n,2,s) and call[2]['SEGMENTS']==s for call in kernels['decode'].calls)
                fallbacks.append(dict(batch=n,segments=s,pitch=pitch))
    # Real source qualification, attach, _load and verify remain unchanged.
    # The fixture delegates only env inputs and records dispatcher metadata.
    def dual_fixture(n,s,pitch,*,bad_schema=False):
        proxy=NS(**vars(native))
        def attach(runtime,env):
            runtime.torch._C._dispatch_has_kernel_for_dispatch_key=runtime.torch.ops.has_kernel
            if bad_schema:runtime.torch.ops.n48_op._schema=native.SCHEMA
            return native.attach(runtime,{**env,q.N48_ENV:'1',q.N48_PATH_ENV:str(a.n48_library),q.N48_SHA_ENV:controller.N48_SHA})
        proxy.attach=attach
        return fixture(proxy,n,s,pitch,a.base_library)
    dual_cases=[]
    for n,s in ((32,4),(48,4),(64,2)):
        for pitch in (2048,2560):
            runtime,entry,kw,producer,kernels,ops,capture,_,_=dual_fixture(n,s,pitch)
            runtime.q32_hoist=NoPrefill();adapter=runtime.native_decode
            assert not set(vars(adapter))&{'verify','validate','try_produce','_load'}
            assert adapter.verify(runtime)['library']==combined_config
            assert ops.loaded_libraries=={str(a.base_library.resolve()),str(a.n48_library.resolve())}
            assert ops.dispatch_queries==[(native.OPERATOR,'CUDA'),(native.N48_OPERATOR,'CUDA')]
            assert str(adapter.ops['base']._schema)==native.SCHEMA and str(adapter.ops['n48']._schema)==native.N48_SCHEMA
            selected,other=(ops.n48_op,ops.op) if n==48 else (ops.op,ops.n48_op)
            output=io.StringIO()
            with contextlib.redirect_stdout(output):producer((n,2,s),**kw);capture[0]=True;producer((n,2,s),**kw);capture[0]=False
            assert len(selected.calls)==2 and not other.calls and not kernels['decode'].calls
            assert all(call[-1]==s for call in selected.calls)
            events=controller.native_events(output.getvalue(),adapter.library);assert len(events)==2
            assert all(e['event']['operator']==(native.N48_OPERATOR if n==48 else native.OPERATOR) for e in events)
            reject('dual stride '+str((n,pitch)),lambda:producer((n,2,s),**dict(kw,block_table_stride=1)))
            reject('dual output alias '+str((n,pitch)),lambda:producer((n,2,s),**dict(kw,segm_output_ptr=kw['query_ptr'])))
            generation=entry.generation;entry.generation=runtime.registry.generation+1
            reject('dual stale epoch '+str((n,pitch)),lambda:producer((n,2,s),**kw));entry.generation=generation
            if n==48:
                line='BIV150_KONLY_NATIVE_DECODE_SELECTED '+json.dumps(next(iter(adapter.records.values())))
                reject('N48 event mislabeled base',lambda:controller.native_events(line.replace(controller.N48_SHA,controller.K8_SHA),adapter.library))
                namespace=ops.cjj_native_k8_compact_n48_s4_probe;saved=namespace.op;namespace.op=old.Op(native.N48_SCHEMA)
                reject('N48 replaced dispatcher',lambda:adapter.verify(runtime),'dispatcher identity changed');namespace.op=saved
            selected.throw=True;reject('dual postlaunch no retry '+str((n,pitch)),lambda:producer((n,2,s),**kw))
            assert len(selected.calls)==3 and not other.calls and not kernels['decode'].calls
            dual_cases.append(dict(batch=n,segments=s,pitch=pitch,actual_host_attach_load_verify_path=True,
                registered_mock_schemas=[native.SCHEMA,native.N48_SCHEMA],dispatch_queries=ops.dispatch_queries,selected_recorded_calls=3))
    reject('wrong N48 registered schema',lambda:dual_fixture(48,4,2560,bad_schema=True),'exact registered produce ABI')
    dual_fallbacks=[]
    for n,s in ((1,16),(8,16),(16,16),(40,16),(48,2),(48,16)):
        runtime,entry,kw,producer,kernels,ops,capture,_,_=dual_fixture(n,s,2560)
        with contextlib.redirect_stdout(io.StringIO()):producer((n,2,s),**kw)
        assert not ops.op.calls and not ops.n48_op.calls and len(kernels['decode'].calls)==1
        assert kernels['decode'].calls[0][0]==(n,2,s) and kernels['decode'].calls[0][2]['SEGMENTS']==s
        dual_fallbacks.append(dict(batch=n,segments=s))
    # Full real native+hoist installer is exercised with the existing42-layer CPU
    # framework mocks; actual7892 and actual2ad pass the real component gates.
    original_ops=old.Ops;old.Ops=DualOps
    try:
        with contextlib.redirect_stdout(io.StringIO()):installation=existing.installation(a.base_library,hoist=True,n48_path=a.n48_library)
    finally:old.Ops=original_ops
    assert 'numpy' not in sys.modules
    assert sys.modules['torch'].__file__=='<CPU metadata, not imported Torch>' and isinstance(sys.modules['torch']._C,NS)
    assert getattr(sys.modules['triton'],'__file__',None) is None and getattr(sys.modules['vllm'],'__file__',None) is None
    report=dict(CPU_only=True,passed=True,actual_SO_files_read_not_loaded=True,base_SO_sha256=controller.K8_SHA,n48_SO_sha256=controller.N48_SHA,
      controller_parent_sha256=sha(parent.read_bytes()),controller_sha256=sha((ROOT/'tools/run_biv150_compact_native_n48_service.py').read_bytes()),
      unchanged_controller_functions=unchanged,changed_controller_functions=sorted(changed),helper_sha256=controller.HELPER_SHA,
      actual_helper_resolution=resolution,C3_six_modes_fingerprints=fingerprints,source_fixture_reuse=fixture_proof,
      actual_segment_picker_results=picked,base_dispatch_cases=cases,unchanged_fallback_cases=fallbacks,rejections=rejected,
      N48_qualification_bypass_used=False,N48_source_component_admission_accepted=True,
      N48_dual_library_cases=dual_cases,N48_enabled_unsupported_fallbacks=dual_fallbacks,
      historical_pending_fixture_sha256=sha(frozen),N48_qualification=q.pin('n48'),
      N48_fixture_is_not_component_or_service_evidence=True,base_N48_and_Q32_mock_installation=installation,
      official_CLI_Math105_benchmark_and_warmup_function_sources_unchanged=True,
      CUDA_CPP_codec_Q32_compute_sources_changed=False,GPU_executed=False,numeric_verified=False,service_verified=False,performance_verified=False)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(report=str(a.out),sha256=sha(a.out.read_bytes()),CPU_only=True,passed=True,N48_source_component_admitted=True)))
if __name__=='__main__':main()
