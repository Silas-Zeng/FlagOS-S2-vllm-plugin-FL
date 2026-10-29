"""Opt-in, fixed readonly worker counter snapshot. Default import is stdlib-only.

No general RPC endpoint, pickle, CUDA API, counter reset, or per-forward hook.
"""
import functools
import hashlib
import hmac
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys
import time

ENABLE='CJJ_RUNTIME_STATS_SNAPSHOT'
TOKEN='CJJ_RUNTIME_STATS_TOKEN'
RUN_ID='CJJ_RUNTIME_STATS_RUN_ID'
METHOD='cjj_runtime_stats_snapshot_v1'
ROUTE='/_cjj/runtime-stats-snapshot-v1'
SCHEMA='cjj-existing-runtime-stats-snapshot-v1'
ROOT=Path(__file__).resolve().parents[1]
SDK_PINS=json.loads((Path(__file__).with_name('biv150_runtime_stats_sdk_pins.json')).read_text())
PHASES=frozenset(f'{kind}-{length}' for length in (4096,16384) for kind in ('warm-before','warm-after','score-after'))
MAX_JSON_BYTES=32*1024*1024
SERVICE='vllm_fl.patches.biv150_konly_encoded_service'
TARGETS={'vllm.entrypoints.openai.api_server','vllm.v1.worker.worker_base'}


def need(value,message):
    if not value:raise ValueError('Readonly runtime snapshot: '+message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(environ):
    if environ.get(ENABLE)!='1':return None
    token=environ.get(TOKEN,'');run=environ.get(RUN_ID,'')
    need(re.fullmatch(r'[A-Za-z0-9_-]{32,128}',token),'missing/invalid private token')
    need(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',run),'invalid run id')
    # No token value, prefix, digest or representation is ever logged/exported.
    return token,run


def sdk_check(module):
    path=Path(module.__file__).resolve();root=path.parents[2]
    need(module.__name__ in TARGETS,'unexpected module')
    expected='entrypoints/openai/api_server.py' if module.__name__.endswith('api_server') else 'v1/worker/worker_base.py'
    need(root/expected==path,'SDK module layout differs')
    for relative,digest in SDK_PINS.items():need(sha(root/relative)==digest,'SDK source SHA differs: '+relative)
    return dict(SDK_PINS)


def process_identity():
    stat=Path('/proc/self/stat')
    need(stat.is_file(),'Linux process identity unavailable')
    return os.getpid(),int(stat.read_text().rsplit(')',1)[1].split()[19])


def snapshot(wrapper,phase,run_id,*,expected_run,source_sha256,sdk_pins):
    need(phase in PHASES and run_id==expected_run,'phase/run mismatch')
    worker=vars(wrapper).get('worker');need(worker is not None,'worker not installed')
    runner=vars(worker).get('model_runner');need(runner is not None,'model runner not installed')
    module=sys.modules.get(SERVICE);state=None if module is None else vars(module).get('_STATE')
    need(state is not None,'encoded runtime not installed in this worker')
    source=ROOT/'vllm_fl/patches/biv150_konly_encoded_service.py'
    need(Path(module.__file__).resolve()==source.resolve(),'foreign encoded service module')
    need(type(worker).__module__=='vllm_fl.worker.worker' and type(worker).__name__=='WorkerFL','foreign worker type')
    need(type(runner).__module__=='vllm_fl.worker.model_runner' and type(runner).__name__=='ModelRunnerFL','foreign runner type')
    need(state.registry.armed and bool(state.registry.entries),'worker has no armed encoded allocation')
    need(state.actual_target==['corex',71,64],'runtime target differs')
    need(any(owner is type(runner) for owner,_,_ in state.attribute_bindings),'runtime/runner binding differs')
    need(state.final_export is None,'worker has already finalized')
    pid,ticks=process_identity()
    need(type(pid) is int and pid>0 and type(ticks) is int and ticks>0,'worker process identity invalid')
    q32=state.q32_hoist
    # Called through serialized worker utility dispatch after all client requests
    # complete. Serialize here, never hand live dict references to an IPC thread.
    payload=dict(schema=SCHEMA,phase=phase,run_id=run_id,pid=pid,start_ticks=ticks,
        rpc_rank=vars(wrapper).get('rpc_rank'),unix_time=time.time(),monotonic_ns=time.monotonic_ns(),
        implementation_sha256=state.installed_digest,tool_source_sha256=source_sha256,sdk_pins=sdk_pins,target=list(state.actual_target),
        service_module=str(source),worker_type=type(worker).__module__+'.'+type(worker).__name__,
        runner_type=type(runner).__module__+'.'+type(runner).__name__,
        stats=state.stats(),shape_records=state.shape_records,shape_overflow=state.shape_overflow,
        kernel_records=list(state.kernel_records.values()),
        q32_geometry_records={} if q32 is None else q32.geometry_records,
        graph_replays_counted=False,client_invocation_including_preflight=True)
    data=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
    need(len(data.encode())<=MAX_JSON_BYTES,'snapshot exceeds size bound')
    return data


def install_worker(module,config):
    pins=sdk_check(module);_,run=config;cls=module.WorkerWrapperBase
    need(not hasattr(cls,METHOD),'fixed method already exists')
    source_sha=sha(__file__)
    def fixed(self,phase,run_id):
        return snapshot(self,phase,run_id,expected_run=run,source_sha256=source_sha,sdk_pins=pins)
    setattr(cls,METHOD,fixed)
    return dict(installed=True,method=METHOD,tool_source_sha256=source_sha)


def install_api(module,config):
    pins=sdk_check(module);token,run=config;original=module.build_app
    need(not getattr(original,'_cjj_runtime_stats_wrapped',False),'API already wrapped')
    @functools.wraps(original)
    def build(*args,**kwargs):
        app=original(*args,**kwargs)
        need(not any(getattr(route,'path',None)==ROUTE for route in app.routes),'route already exists')
        # Imported only by opt-in API installation, after the actual SDK module.
        from fastapi import HTTPException,Request,Response
        async def endpoint(request):
            need_peer=request.client is not None and request.client.host in ('127.0.0.1','::1')
            provided=request.headers.get('x-cjj-stats-token','')
            if not need_peer or type(provided) is not str or not provided.isascii() or not hmac.compare_digest(provided,token):raise HTTPException(status_code=403,detail='Denied')
            body=await request.body()
            if len(body)>512:raise HTTPException(status_code=400,detail='Invalid snapshot request')
            try:value=json.loads(body)
            except (ValueError,UnicodeError):raise HTTPException(status_code=400,detail='Invalid snapshot request') from None
            if type(value) is not dict or set(value)!= {'phase','run_id'} or type(value.get('phase')) is not str or type(value.get('run_id')) is not str or value['phase'] not in PHASES or value['run_id']!=run:
                raise HTTPException(status_code=400,detail='Invalid snapshot request')
            engine=getattr(request.app.state,'engine_client',None)
            if engine is None:raise HTTPException(status_code=503,detail='Engine unavailable')
            # Static method only. No code/method/path/body forwarding. Native
            # UniProc timeout is not a hard interrupt; caller awaits this ack.
            results=await engine.collective_rpc(METHOD,timeout=30.0,args=(value['phase'],run),kwargs={})
            need(type(results) is list and len(results)==1 and type(results[0]) is str,'unexpected worker result')
            data=results[0];need(len(data.encode())<=MAX_JSON_BYTES,'worker result exceeds size bound')
            got=json.loads(data)
            need(got['schema']==SCHEMA and got['phase']==value['phase'] and got['run_id']==run,'worker response mismatch')
            need(got['sdk_pins']==pins and got['tool_source_sha256']==sha(__file__),'worker bootstrap identity mismatch')
            need(got['rpc_rank']==0 and type(got['pid']) is int and got['pid']>0 and type(got['start_ticks']) is int and got['start_ticks']>0,'worker identity missing')
            return Response(content=data,media_type='application/json',headers={'x-cjj-stats-sha256':hashlib.sha256(data.encode()).hexdigest()})
        # Annotation must resolve at FastAPI registration; no eager engine lookup.
        endpoint.__annotations__={'request':Request}
        app.add_api_route(ROUTE,endpoint,methods=['POST'],include_in_schema=False)
        return app
    build._cjj_runtime_stats_wrapped=True;module.build_app=build
    return dict(installed=True,route=ROUTE,tool_source_sha256=sha(__file__))


class Loader:
    def __init__(self,original,callback):self.original,self.callback=original,callback
    def __getattr__(self,name):return getattr(self.original,name)
    def create_module(self,spec):
        return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
    def exec_module(self,module):
        self.original.exec_module(module);self.callback(module)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self,config):self.config=config
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in TARGETS:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        need(spec is not None and hasattr(spec.loader,'exec_module'),'target loader unavailable')
        callback=install_api if fullname.endswith('api_server') else install_worker
        spec.loader=Loader(spec.loader,lambda module:callback(module,self.config));return spec


def install(environ=None):
    config=configuration(os.environ if environ is None else environ)
    if config is None:return dict(enabled=False)
    need(not TARGETS.intersection(sys.modules),'install must precede SDK imports')
    need(not any(isinstance(f,Finder) for f in sys.meta_path),'bootstrap already installed')
    sys.meta_path.insert(0,Finder(config));return dict(enabled=True,method=METHOD,route=ROUTE)


def counter_delta(before,after):
    """Fail closed on incomplete/reset windows. Does not alter either snapshot."""
    for key in ('schema','run_id','pid','start_ticks','rpc_rank','implementation_sha256','tool_source_sha256','sdk_pins'):
        need(before[key]==after[key],'window identity changed: '+key)
    need(before['schema']==SCHEMA and after['monotonic_ns']>=before['monotonic_ns'],'window order/schema')
    need(before['phase'].rsplit('-',1)[-1]==after['phase'].rsplit('-',1)[-1],'mixed benchmark lengths')
    need((before['phase'].rsplit('-',1)[0],after['phase'].rsplit('-',1)[0]) in
         (('warm-before','warm-after'),('warm-after','score-after')),'invalid phase pair')
    def diff(a,b):
        need(type(a) is int and type(b) is int and 0<=a<=b,'counter decreased/invalid');return b-a
    def mapping(a,b,field):
        need(set(a)<=set(b),'counter records disappeared');result={}
        for key,row in b.items():
            previous=a.get(key)
            if previous is not None:
                need({k:v for k,v in row.items() if k!=field}=={k:v for k,v in previous.items() if k!=field},'record identity changed')
            result[key]=diff(0 if previous is None else previous[field],row[field])
        return result
    def scalar_map(a,b):
        need(set(a)<=set(b),'counter keys disappeared');return {k:diff(a.get(k,0),v) for k,v in b.items()}
    omitted=diff(before['shape_overflow'],after['shape_overflow']);need(omitted==0,'shape overflow in window')
    shapes=mapping(before['shape_records'],after['shape_records'],'host_calls')
    selected=scalar_map(before['stats']['selected'],after['stats']['selected'])
    native_before=before['stats']['native_decode'];native_after=after['stats']['native_decode']
    need(native_before['enabled']==native_after['enabled'],'native binding changed')
    native={};fallback={}
    if native_after['enabled']:
        need(native_before['library']==native_after['library'],'native library changed')
        need(diff(native_before['overflow'],native_after['overflow'])==0,'native shape overflow in window')
        def rows(values):
            result={}
            for row in values:
                key=json.dumps({k:v for k,v in row.items() if k!='host_calls'},sort_keys=True)
                need(key not in result,'duplicate native record');result[key]=row
            return result
        native=mapping(rows(native_before['records']),rows(native_after['records']),'host_calls')
        need(sum(native.values())==diff(native_before['selected_host_calls'],native_after['selected_host_calls'])==selected.get('native_decode',0),'native counter reconciliation')
        fallback=scalar_map(native_before['private_Triton_fallbacks'],native_after['private_Triton_fallbacks'])
    else:need(selected.get('native_decode',0)==0,'native calls without enabled adapter')
    need(sum(shapes.values())==sum(v for k,v in selected.items() if k!='native_decode'),'TL shape reconciliation')
    q32=mapping(before['q32_geometry_records'],after['q32_geometry_records'],'admitted_host_calls')
    qb,qa=before['stats']['q32_hoist'],after['stats']['q32_hoist']
    need(qb['enabled']==qa['enabled'],'Q32 binding changed')
    if qa['enabled']:need(sum(q32.values())==diff(qb['admitted_host_calls'],qa['admitted_host_calls']),'Q32 geometry reconciliation')
    return dict(complete_host_window=True,lifetime_shape_records_complete=after['shape_overflow']==0,
        lifetime_native_records_complete=not native_after['enabled'] or native_after['overflow']==0,selected=selected,shapes=shapes,native=native,native_fallbacks=fallback,q32=q32,
        graph_replays_counted=False,client_invocation_including_preflight=True,device_time_or_model_steps=False)
