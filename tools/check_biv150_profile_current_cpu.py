"""CPU contracts: private caches, immutable tuning seed and full warm before trace."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import run_biv150_baseline_profile as profile
import biv150_profile_routes as routes


def route_contracts():
    def manager():
        return SimpleNamespace(active=True, **{key: 0 for key in routes.COUNTERS},
                               optimized={'synthetic-cpu-fixture': {}}, record_errors=[])
    fragment, q32, k128 = manager(), manager(), manager()
    original = SimpleNamespace(manager=fragment)
    q32_policy = SimpleNamespace(manager=q32, base=original, disabled=False)
    fragment.holder = {'kernel': q32_policy}
    fragment.kernel_key = 'kernel'
    modules = {routes.MODULES['qfragment']: SimpleNamespace(_MANAGER=fragment),
               routes.MODULES['q32']: SimpleNamespace(_MANAGER=q32)}
    worker = SimpleNamespace(model_runner=SimpleNamespace(
        _biv150_compute_order_buffers={'selected': 30}))
    before = routes.snapshot(worker, modules=modules)
    q32.selected_calls += 42
    worker.model_runner._biv150_compute_order_buffers['selected'] += 1
    after = routes.snapshot(worker, modules=modules)
    assert routes.validate(before, after, 'q32')['passed']
    assert not routes.validate(before, before, 'q32')['passed']
    mutations = [
        lambda x: x['managers']['q32'].update(active=False),
        lambda x: x['managers']['q32'].update(bound=False),
        lambda x: x['managers']['q32'].update(identity=-1),
        lambda x: x['managers']['q32'].update(policy_disabled=True),
        lambda x: x['managers']['q32'].update(record_errors=['unproven lowering']),
        lambda x: x['managers']['q32']['counters'].update(selected_calls=-1),
        lambda x: x['compute_order'].update(active=False),
        lambda x: x['compute_order'].update(runner_identity=-1),
        lambda x: x.update(pid=-1),
    ]
    for mutate in mutations:
        broken = copy.deepcopy(after)
        mutate(broken)
        assert not routes.validate(before, broken, 'q32')['passed']
    # Future K128 may handle all eligible shapes: Q32 can remain at zero delta.
    modules[routes.MODULES['k128']] = SimpleNamespace(_MANAGER=k128)
    fragment.holder['kernel'] = SimpleNamespace(manager=k128, base=q32_policy, disabled=False)
    kbefore = routes.snapshot(worker, modules=modules)
    k128.selected_calls += 42
    worker.model_runner._biv150_compute_order_buffers['selected'] += 1
    kafter = routes.snapshot(worker, modules=modules)
    assert routes.validate(kbefore, kafter, 'k128')['passed']
    assert not routes.validate(kbefore, kafter, 'q32')['passed']
    return before, after


def bootstrap_contract(bootstrap, observations, source):
    # Exercise the generated wrapper with explicit CPU-only fake modules. This
    # does not establish real CUDA profiler or strict-C3 compatibility.
    events=[]
    class Worker:
        def __init__(self,value):
            events.append('original_init')
            if value=='fail':raise RuntimeError('original failure')
            self.value=value
    service=SimpleNamespace(__file__=str(source/'tools/biv150_profile_service.py'),
        install=lambda:events.append('install'),
        bind_worker=lambda worker:events.append(('bind',worker)))
    module=SimpleNamespace(WorkerFL=Worker,__file__=str(source/'vllm_fl/worker/worker.py'))
    fake_modules={'torch':SimpleNamespace(__version__='synthetic-cpu-abi',
                    _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)),
                  'vllm_fl':SimpleNamespace(__file__=str(source/'vllm_fl/__init__.py')),
                  'biv150_profile_service':service}
    finders=list(sys.meta_path)
    try:
        with patch.dict(sys.modules,fake_modules),patch.dict(os.environ,
                {'BIV150_PROFILE_DIR':str(observations),'BIV150_PROFILE_BOOTSTRAP':''}):
            namespace={}
            exec((bootstrap/'sitecustomize.py').read_text(),namespace)
            namespace['_wrap_worker'](module)
            result=module.WorkerFL(7)
            assert result.value==7 and events==['install','original_init',('bind',result)]
            events.clear()
            try:module.WorkerFL('fail')
            except RuntimeError as error:assert str(error)=='original failure'
            else:raise AssertionError('Worker initialization failure swallowed')
            assert events==['install','original_init']
            record=json.loads((observations/('bootstrap-%d.json'%os.getpid())).read_text())
            assert record['torch_abi']=={'torch':'synthetic-cpu-abi','cxx11_abi':False}
    finally:
        sys.meta_path[:]=finders


def main():
    markers=('BIV150_NATIVE_W8A8_INSTALL','BIV150_STABLE_SMALL_O_INSTALL','BIV150_Q_FRAGMENT_INSTALL','BIV150_PREFILL_NORMALIZATION_INSTALL','BIV150_GQA_DECODE_INSTALL','BIV150_MIXED_DECODE_INSTALLED','BIV150_PREFILL_Q32_INSTALL','BIV150_SMALL_OUT_LOWERING_INSTALL')
    logs='\n'.join(marker+" {'installed': True, 'large_linear': True}" for marker in markers)
    assert len(profile.installed_runtime(logs))==8
    try:profile.installed_runtime(logs.replace("'installed': True","'installed': False",1))
    except RuntimeError:pass
    else:raise AssertionError('Inactive baseline silently accepted')
    try:profile.installed_runtime(logs,required_route='k128')
    except RuntimeError:pass
    else:raise AssertionError('Missing K128 installation silently accepted')
    assert len(profile.installed_runtime(logs+"\nBIV150_PREFILL_K128_INSTALL {'installed': True}",required_route='k128'))==9
    try:profile.installed_runtime(logs.replace("'large_linear': True", "'large_linear': False"))
    except RuntimeError:pass
    else:raise AssertionError('Disabled default large-linear provider accepted')
    before,after=route_contracts()
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary).resolve();out=root/'out';out.mkdir();seed=root/'seed';seed.mkdir()
        (seed/'fixture.db').write_bytes(b'fixed CPU fixture')
        record=dict(name='fixture.db',sha256=profile.sha256(seed/'fixture.db'))
        (seed/'manifest.json').write_text(json.dumps({'databases':[record]}))
        args=SimpleNamespace(out=out,source=Path(__file__).resolve().parents[1],library=root/'library.so',tuning_snapshot=seed,workload='4096')
        args.library.write_bytes(b'CPU fingerprint fixture, not a real native library')
        copied=profile.prepare_tuning(args)
        assert copied['databases']==[record] and (out/'flaggems-cache/config_cache/fixture.db').read_bytes()==b'fixed CPU fixture'
        with patch.dict(os.environ,{'TRITON_CACHE_DIR':'/forbidden-global','FLAGGEMS_CACHE_DIR':'/forbidden-global','VLLM_FL_BIV150_SMALL_OUT_LOWERING':'1'}):
            service,bench=profile.make_environment(args,root/'bootstrap')
        for env in [service,bench]:
            assert env['TRITON_CACHE_DIR']==str(out/'triton-cache')
            assert env['FLAGGEMS_CACHE_DIR']==str(out/'flaggems-cache')
            assert env['VLLM_FL_BIV150_NATIVE_W8A8']==env['VLLM_FL_BIV150_STABLE_SMALL_O']=='1'
            assert 'VLLM_FL_BIV150_SMALL_OUT_LOWERING' not in env
        abi={'torch':'synthetic-cpu-abi','cxx11_abi':False}
        fingerprint=profile.expected_source_fingerprint(args.source,service,abi)
        fp_log='BIV150_COMPILE_SEMANTICS '+json.dumps(fingerprint)
        assert profile.runtime_fingerprint(fp_log,args.source,service,abi)['verified']
        for changed_log,changed_env in [(fp_log.replace(fingerprint['sha256'],'0'*64),service),
                                        (fp_log,{**service,'VLLM_FL_BIV150_PREFILL_Q32':'0'})]:
            try:profile.runtime_fingerprint(changed_log,args.source,changed_env,abi)
            except RuntimeError:pass
            else:raise AssertionError('Stale source or effective flags accepted')
        bootstrap,launcher=profile.make_bootstrap(args)
        compile((bootstrap/'sitecustomize.py').read_text(),'<synthetic-sitecustomize>','exec')
        compile(launcher.read_text(),'<synthetic-launcher>','exec')
        bootstrap_contract(bootstrap,out/'observations',args.source)
        observation=out/'observations/worker-1-window-1.json'
        value=dict(errors=[],scope_diagnostics=dict(errors=[],error_total=0,callback_disabled=False),
            samples=[dict(observations=[dict(variant=kind,numeric_exact=True,graph_total_ms=1.,
                all_api_calls_succeeded=True,kernels=[{'synthetic_cpu_fixture':True}]) for kind in ['boundary','full']])],
            production_routes_before=before,production_routes_after=after)
        observation.write_text(json.dumps(value))
        assert profile.validate_profile_windows(observation.parent,set())['passed']
        value['production_routes_after']=before
        observation.write_text(json.dumps(value))
        assert not profile.validate_profile_windows(observation.parent,set())['passed']
        original=subprocess.check_output(['git','-C',str(args.source),'show',profile.MAIN+':tools/run_biv150_baseline_profile.py'],text=True)
        current=Path(profile.__file__).read_text()
        def cleanup(code):
            tree=ast.parse(code)
            return ast.dump(next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='Driver' for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='cleanup'),include_attributes=False)
        assert cleanup(original)==cleanup(current)
        events=[];driver=profile.Driver(args);driver.update=lambda phase,**kw:events.append((phase,kw))
        driver.stop_profile=lambda:None
        def fake_benchmark(load,**geometry):
            events.append(('benchmark',dict(geometry,profile=driver.current_benchmark['profile'])))
            return dict(completed=geometry['count'],num_prompts=geometry['count'],failed=0,max_concurrency=64,total_input_tokens=geometry['input_len']*geometry['count'],total_output_tokens=geometry['output_len']*geometry['count'])
        with patch.object(profile,'original_benchmark',return_value=fake_benchmark),patch.object(profile,'validate_profile_windows',return_value={'passed':True,'windows':[],'errors':[]}):
            driver.workloads()
        cases=[x[1] for x in events if x[0]=='benchmark']
        assert len(cases)==5 and [x['profile'] for x in cases]==[False,False,False,False,True]
        assert cases[-2]['input_len']==cases[-1]['input_len']==4096
        assert cases[-2]['output_len']==cases[-1]['output_len']==1024
        assert cases[-2]['count']==cases[-1]['count']==256 and cases[-2]['seed']==cases[-1]['seed']==0
        assert 'torch' not in sys.modules
    print(json.dumps(dict(passed=True,production_baseline=profile.MAIN,private_caches=True,full_workload_warmup_before_profile=True,owned_cleanup_ast_unchanged=True,runtime_installation_required=True,window_selected_routes_required=True,source_fingerprint_required=True,synthetic_cpu_fixtures_only=True,torch_imported=False)))


if __name__=='__main__':main()
