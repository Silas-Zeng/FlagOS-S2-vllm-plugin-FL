#!/usr/bin/env python3
"""Actual pinned dispatch-method ASTs, opaque CPU worker/app, no SDK imports."""
import ast,asyncio,copy,functools,json,os,shutil,subprocess,sys,tempfile,types
from pathlib import Path
import biv150_runtime_stats_snapshot as s
SDK=Path('/private/tmp/cjj-current-api-rpc-source-20261029')
checks={}
def ok(name,condition=True):assert condition,name;checks[name]=True
def refused(name,fn):
    try:fn()
    except (ValueError,TypeError,AttributeError,KeyError):ok(name);return
    raise AssertionError(name+' did not reject')
def fn(relative,cls,name,globals_=None):
    path=SDK/relative;tree=ast.parse(path.read_text());body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    node=next(n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    ns={} if globals_ is None else dict(globals_)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns);return ns[name]

def main():
    before=list(sys.meta_path);ok('default_OFF_no_import_hook',s.install({})=={'enabled':False} and sys.meta_path==before)
    refused('missing_token',lambda:s.configuration({s.ENABLE:'1',s.RUN_ID:'cpu-fixture'}))
    with tempfile.TemporaryDirectory(prefix='cjj-snapshot-bootstrap-cpu-') as tmp:
        Path(tmp,'sitecustomize.py').write_text('raise AssertionError("must not execute old startup")\n')
        env=dict(os.environ,PYTHONPATH=tmp,CJJ_RUNTIME_STATS_SNAPSHOT='1',CJJ_RUNTIME_STATS_TOKEN='A'*48,CJJ_RUNTIME_STATS_RUN_ID='cpu-fixture')
        proc=subprocess.run([sys.executable,'-S',str(Path(__file__).parent/'runtime_stats_bootstrap/sitecustomize.py')],env=env,text=True,capture_output=True)
        ok('existing_sitecustomize_refused_without_executing',proc.returncode!=0 and 'must not execute old startup' not in proc.stderr)
        ok('startup_failure_does_not_echo_token','A'*48 not in proc.stdout+proc.stderr)
    config=('A'*48,'cpu-fixture')
    enabled={s.ENABLE:'1',s.TOKEN:config[0],s.RUN_ID:config[1]}
    target=next(iter(s.TARGETS));sys.modules[target]=types.SimpleNamespace()
    try:refused('late_SDK_import',lambda:s.install(enabled))
    finally:sys.modules.pop(target,None)
    original_meta=list(sys.meta_path)
    try:
        ok('enabled_cold_hook_registered',s.install(enabled)['enabled'] and isinstance(sys.meta_path[0],s.Finder))
        refused('duplicate_bootstrap',lambda:s.install(enabled))
    finally:sys.meta_path[:]=original_meta
    # CPU-only opaque identity fixture; no claim to real Linux worker execution.
    s.process_identity=lambda:(123456,7890123)
    tree=ast.parse((SDK/'entrypoints/openai/api_server.py').read_text())
    text=(SDK/'entrypoints/openai/api_server.py').read_text()
    ok('actual_build_then_init_state_source',text.index('app = build_app(args, supported_tasks, model_config)')<text.index('await init_app_state(engine_client, app.state, args, supported_tasks)'))
    ok('actual_state_engine_client_assignment','state.engine_client = engine_client' in text)
    for relative,digest in s.SDK_PINS.items():ok('SDK_SHA_'+relative,s.sha(SDK/relative)==digest)
    wrapper_class=type('WorkerWrapperBase',(),{'__module__':'vllm.v1.worker.worker_base',
        '__init__':fn('v1/worker/worker_base.py','WorkerWrapperBase','__init__'),
        '__getattr__':fn('v1/worker/worker_base.py','WorkerWrapperBase','__getattr__')})
    wm=types.SimpleNamespace(__file__=str(SDK/'v1/worker/worker_base.py'),__name__='vllm.v1.worker.worker_base',WorkerWrapperBase=wrapper_class)
    # Exercise Loader/Finder sequencing using an opaque loader and actual file
    # pins. This does not import the SDK/framework or execute GPU code.
    from unittest.mock import patch
    events=[]
    class OpaqueLoader:
        def create_module(self,spec):events.append('create');return None
        def exec_module(self,module):
            events.append('original_exec')
            module.WorkerWrapperBase=type('WorkerWrapperBase',(),{})
        def get_source(self,name):return 'opaque fixture source'
    fake_spec=types.SimpleNamespace(loader=OpaqueLoader())
    finder=s.Finder(config)
    ok('finder_unrelated_no_intercept',finder.find_spec('unrelated.module') is None)
    with patch.object(s.importlib.machinery.PathFinder,'find_spec',return_value=fake_spec):
        wrapped=finder.find_spec('vllm.v1.worker.worker_base')
        temp=types.SimpleNamespace(__file__=wm.__file__,__name__=wm.__name__)
        wrapped.loader.create_module(wrapped);wrapped.loader.exec_module(temp)
        ok('cold_loader_then_fixed_method',events==['create','original_exec'] and hasattr(temp.WorkerWrapperBase,s.METHOD))
        ok('loader_introspection_delegated',wrapped.loader.get_source('fixture')=='opaque fixture source')
    with tempfile.TemporaryDirectory(prefix='cjj-snapshot-sdk-cpu-') as tmp:
        root=Path(tmp)
        for relative in s.SDK_PINS:
            target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SDK/relative,target)
        altered=root/'v1/worker/worker_base.py';altered.write_bytes(altered.read_bytes()+b'\n# CPU mutation\n')
        wrong=types.SimpleNamespace(__name__=wm.__name__,__file__=str(altered))
        refused('changed_SDK_source',lambda:s.sdk_check(wrong))
    receipt=s.install_worker(wm,config);wrapper=wrapper_class()
    refused('worker_missing',lambda:getattr(wrapper,s.METHOD)('warm-before-4096','cpu-fixture'))
    Worker=type('WorkerFL',(),{'__module__':'vllm_fl.worker.worker'});Runner=type('ModelRunnerFL',(),{'__module__':'vllm_fl.worker.model_runner'})
    wrapper.worker=Worker();refused('runner_missing',lambda:getattr(wrapper,s.METHOD)('warm-before-4096','cpu-fixture'))
    wrapper.worker.model_runner=Runner()
    refused('runtime_missing',lambda:getattr(wrapper,s.METHOD)('warm-before-4096','cpu-fixture'))
    production=s.ROOT/'vllm_fl/patches/biv150_konly_encoded_service.py'
    original=subprocess.check_output(['git','-C',str(s.ROOT),'show','a455a3de364051fbea80226870c239de3a48e716:vllm_fl/patches/biv150_konly_encoded_service.py'])
    ok('production_service_byte_exact_a455',production.read_bytes()==original)
    assignments={node.targets[0].attr:node.value for node in ast.walk(ast.parse(original)) if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Attribute) and node.targets[0].attr in ('actual_target','attribute_bindings')}
    # Evaluate the actual assignment expressions, not a separately invented
    # target shape or tuple arity for the runtime/runner binding fixture.
    actual_target=eval(compile(ast.Expression(assignments['actual_target']),str(production),'eval'),{'target':types.SimpleNamespace(backend='corex',arch=71,warp_size=64)})
    actual_bindings=eval(compile(ast.Expression(assignments['attribute_bindings']),str(production),'eval'),{'attributes':[(Runner,'fixture',None)]})
    ok('actual_runtime_target_and_binding_assignment',actual_target==['corex',71,64] and len(actual_bindings[0])==3)
    native=dict(enabled=True,library={'sha256':'fixture-not-an-SO'},selected_host_calls=1,overflow=0,
        records=[dict(shape=64,segments=2,capture=False,host_calls=1)],private_Triton_fallbacks={'shape_or_segments':2})
    stats=dict(selected={'decode':2,'q32':1,'native_decode':1},native_decode=native,q32_hoist=dict(enabled=True,admitted_host_calls=1))
    state=types.SimpleNamespace(registry=types.SimpleNamespace(armed=True,entries={'opaque':object()}),
        actual_target=actual_target,attribute_bindings=actual_bindings,
        final_export=None,installed_digest='cpu-fixture-not-runtime-proof',
        q32_hoist=types.SimpleNamespace(geometry_records={'q32-key':dict(admitted_host_calls=1,query_rows=2018)}),
        shape_records={'decode-key':dict(profile='decode',rows=30,host_calls=2),'q32-key':dict(profile='q32',rows=2018,host_calls=1)},
        shape_overflow=0,kernel_records={},stats=lambda:copy.deepcopy(stats))
    service=types.SimpleNamespace(__file__=str(s.ROOT/'vllm_fl/patches/biv150_konly_encoded_service.py'),_STATE=state)
    sys.modules[s.SERVICE]=service
    try:
        before_state=copy.deepcopy((stats,state.shape_records,state.q32_hoist.geometry_records,state.final_export))
        a=json.loads(getattr(wrapper,s.METHOD)('warm-before-4096','cpu-fixture'))
        ok('complete_JSON_no_secret',config[0] not in json.dumps(a) and a['stats']['native_decode']['selected_host_calls']==1)
        ok('no_mutation',before_state==(stats,state.shape_records,state.q32_hoist.geometry_records,state.final_export))
        state.final_export={};refused('already_finalized',lambda:getattr(wrapper,s.METHOD)('warm-before-4096','cpu-fixture'));state.final_export=None
        state.registry.armed=False;refused('unarmed_worker',lambda:getattr(wrapper,s.METHOD)('warm-before-4096','cpu-fixture'));state.registry.armed=True
        refused('wrong_phase',lambda:getattr(wrapper,s.METHOD)('arbitrary-phase','cpu-fixture'))
        refused('wrong_run',lambda:getattr(wrapper,s.METHOD)('warm-before-4096','other-run'))
        stats['selected']['decode']+=3;state.shape_records['decode61-key']=dict(profile='decode',rows=61,host_calls=3)
        b=json.loads(getattr(wrapper,s.METHOD)('warm-after-4096','cpu-fixture'))
        delta=s.counter_delta(a,b);ok('valid_host_window_delta',delta['shapes']['decode61-key']==3 and delta['selected']['decode']==3)
        def mutant(name,change):
            other=copy.deepcopy(b);change(other);refused(name,lambda:s.counter_delta(a,other))
        mutant('shape_overflow',lambda x:x.update(shape_overflow=1))
        mutant('native_overflow',lambda x:x['stats']['native_decode'].update(overflow=1))
        mutant('counter_regression',lambda x:x['shape_records']['decode-key'].update(host_calls=1))
        mutant('Q32_regression',lambda x:x['q32_geometry_records']['q32-key'].update(admitted_host_calls=0))
        mutant('worker_restart',lambda x:x.update(pid=x['pid']+1))
        mutant('record_disappeared',lambda x:x['shape_records'].pop('decode-key'))
        mutant('unexplained_count',lambda x:x['stats']['selected'].update(decode=100))
        # Dispatch the fixed string through actual source-extracted methods.
        run_method=fn('v1/serial_utils.py',None,'run_method',dict(partial=functools.partial))
        Executor=type('CPUExecutor',(),{'collective_rpc':fn('v1/executor/uniproc_executor.py','UniProcExecutor','collective_rpc',dict(run_method=run_method,AsyncModelRunnerOutput=type('OpaqueAsyncOutput',(),{})))})
        executor=Executor();executor.driver_worker=wrapper
        Core=type('CPUCore',(),{'collective_rpc':fn('v1/engine/core.py','EngineCore','collective_rpc')});core=Core();core.model_executor=executor
        wires=[]
        async def utility(self,method,*args):
            encoded=json.dumps([method,args]);ok('transport_contains_no_token',config[0] not in encoded);wires.append(encoded)
            name,values=json.loads(encoded);assert name=='collective_rpc';assert values[0]==s.METHOD
            return getattr(core,name)(*values)
        Client=type('CPUClient',(),{'call_utility_async':utility,'collective_rpc_async':fn('v1/engine/core_client.py','AsyncMPClient','collective_rpc_async')})
        Engine=type('CPUEngine',(),{'collective_rpc':fn('v1/engine/async_llm.py','AsyncLLM','collective_rpc')});engine=Engine();engine.engine_core=Client()
        class App:
            def __init__(self):self.state=types.SimpleNamespace();self.routes=[]
            def add_api_route(self,path,endpoint,**kw):self.routes.append(types.SimpleNamespace(path=path,endpoint=endpoint,options=kw))
        original_calls=[]
        def original(*args,**kw):original_calls.append((args,kw));return App()
        api=types.SimpleNamespace(__file__=str(SDK/'entrypoints/openai/api_server.py'),__name__='vllm.entrypoints.openai.api_server',build_app=original)
        class HTTPException(Exception):
            def __init__(self,*,status_code,detail):self.status_code=status_code;super().__init__(detail)
        class Response:
            def __init__(self,**kwargs):self.__dict__.update(kwargs)
        fastapi=types.SimpleNamespace(HTTPException=HTTPException,Request=type('Request',(),{}),Response=Response)
        sys.modules['fastapi']=fastapi
        try:
            s.install_api(api,config);app=api.build_app('same-args',same_kw=1)
            ok('original_build_once_args_preserved',original_calls==[(('same-args',),{'same_kw':1})])
            ok('no_early_engine_lookup',not hasattr(app.state,'engine_client'))
            ok('one_fixed_hidden_POST',len(app.routes)==1 and app.routes[0].path==s.ROUTE and app.routes[0].options==dict(methods=['POST'],include_in_schema=False))
            class Request:
                def __init__(self,body=None,host='127.0.0.1',token=config[0]):
                    self.app=app;self.client=types.SimpleNamespace(host=host);self.headers={'x-cjj-stats-token':token}
                    self.value=body if body is not None else {'phase':'warm-after-4096','run_id':'cpu-fixture'}
                async def body(self):return json.dumps(self.value).encode()
            async def verify():
                async def reject(name,request,status):
                    previous=len(wires)
                    try:await app.routes[0].endpoint(request)
                    except HTTPException as exc:ok(name,exc.status_code==status and len(wires)==previous);return
                    raise AssertionError(name+' accepted')
                await reject('API_engine_missing',Request(),503)
                app.state.engine_client=engine
                response=await app.routes[0].endpoint(Request());got=json.loads(response.content)
                ok('actual_AST_RPC_string_chain',got['pid']==a['pid'] and got['rpc_rank']==0 and len(wires)==1)
                ok('response_SHA',response.headers['x-cjj-stats-sha256']==s.hashlib.sha256(response.content.encode()).hexdigest())
                await reject('remote_peer',Request(host='203.0.113.10'),403)
                await reject('bad_token',Request(token='wrong'),403)
                await reject('nonASCII_token',Request(token='错误'),403)
                await reject('arbitrary_method_body',Request(body={'phase':'warm-after-4096','run_id':'cpu-fixture','method':'shutdown'}),400)
                await reject('bad_phase_type',Request(body={'phase':[],'run_id':'cpu-fixture'}),400)
                await reject('unknown_phase',Request(body={'phase':'execute_model','run_id':'cpu-fixture'}),400)
            asyncio.run(verify())
            refused('duplicate_API_install',lambda:s.install_api(api,config))
        finally:sys.modules.pop('fastapi',None)
        refused('duplicate_worker_install',lambda:s.install_worker(wm,config))
    finally:sys.modules.pop(s.SERVICE,None)
    ok('no_GPU_modules_imported',not {'torch','triton'}.intersection(sys.modules))
    return dict(passed=True,checks=checks,source_sha256=s.sha(s.__file__),test_sha256=s.sha(__file__),SDK_pins=s.SDK_PINS,
        actual_AST_dispatch_chain=['AsyncLLM.collective_rpc','AsyncMPClient.collective_rpc_async','EngineCore.collective_rpc','UniProcExecutor.collective_rpc','run_method','WorkerWrapperBase'],
        opaque_CPU_app_and_worker=True,transport='CPU JSON roundtrip mock; actual ZMQ/Msgpack/HTTP not executed',actual_Service_or_GPU_verified=False,token_exported=False)
if __name__=='__main__':print(json.dumps(main(),sort_keys=True,indent=2))
