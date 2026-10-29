"""Smoke-only fixed readonly gather submit snapshot. Default import is stdlib-only.

No general RPC endpoint, pickle, CUDA API or counter reset. Helper counters are explicit.
"""
import functools
import hashlib
import hmac
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ENABLE='CJJ_GATHER_SNAPSHOT'
TOKEN='CJJ_GATHER_SNAPSHOT_TOKEN'
RUN_ID='CJJ_GATHER_SNAPSHOT_RUN_ID'
METHOD='cjj_hidden_gather_snapshot_v1'
ROUTE='/_cjj/hidden-gather-snapshot-v1'
SCHEMA='cjj-hidden-gather-submit-snapshot-v1'
ROOT=Path(__file__).resolve().parents[1]
SDK_PINS=json.loads((Path(__file__).with_name('biv150_runtime_stats_sdk_pins.json')).read_text())
PHASES=('smoke-before','smoke-after')
HELPER='vllm_fl/patches/biv150_dynamic_hidden_gather.py'
RUNNER='vllm_fl/worker/model_runner.py'
KERNEL='vllm_fl/patches/biv150_dynamic_hidden_gather_kernel.py'
COUNTERS=('original_query_start_loc_minus_one','original_compute_order_logits_copy')
MAX_JSON_BYTES=128*1024
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


def compiled_record(kernel, function):
    # Worker bootstrap need not put tools/ on sys.path; load this fixed local file.
    path=ROOT/'tools/compile_biv150_dynamic_hidden_gather.py'
    spec=importlib.util.spec_from_file_location('cjj_gather_compiled_metadata',path)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    need(getattr(kernel.src,'fn',None) is function,'compiled object/JIT binding differs')
    files={}
    for ext in ('source','ttir','ttgir','llir','asm','cubin'):
        raw=kernel.asm[ext];raw=raw.encode() if isinstance(raw,str) else raw
        need(type(raw) is bytes,'compiled artifact type')
        files[ext]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    launch=getattr(kernel._run,'launch',None);owner=getattr(launch,'__self__',None)
    need(getattr(owner,'__file__',None),'actual loaded handler missing')
    return dict(hash=str(kernel.hash),ABI=c.runtime_ABI(kernel),options=c.metadata(kernel),
        resources=c.resources(kernel),files=files,launcher_sha256=sha(owner.__file__))


def snapshot(wrapper,phase,run_id,*,expected_run,source_sha256,sdk_pins):
    need(phase in PHASES and run_id==expected_run,'phase/run mismatch')
    worker=vars(wrapper).get('worker');need(worker is not None,'worker not installed')
    runner=vars(worker).get('model_runner');need(runner is not None,'runner not installed')
    need(type(worker).__module__=='vllm_fl.worker.worker' and type(worker).__name__=='WorkerFL','foreign worker')
    need(type(runner).__module__=='vllm_fl.worker.model_runner' and type(runner).__name__=='ModelRunnerFL','foreign runner')
    module=sys.modules.get('vllm_fl.patches.biv150_dynamic_hidden_gather')
    need(module is not None and Path(module.__file__).resolve()==ROOT/HELPER,'actual gather helper path')
    need(os.environ.get(module.ENV)=='1' and os.environ.get(module.DIAGNOSTICS)=='1','gather/diagnostic disabled')
    source=sys.modules.get('vllm_fl.worker.model_runner');order=sys.modules.get('vllm_fl.patches.biv150_compute_order')
    need(source is not None and order is not None and Path(source.__file__).resolve()==ROOT/RUNNER and sha(source.__file__)==order.EXPECTED_RUNNER,'actual runner bytes/pin')
    service=sys.modules.get(SERVICE);encoded=None if service is None else vars(service).get('_STATE')
    need(encoded is not None and Path(service.__file__).resolve()==ROOT/'vllm_fl/patches/biv150_konly_encoded_service.py','actual encoded runtime')
    need(encoded.registry.armed and bool(encoded.registry.entries) and encoded.actual_target==['corex',71,64],'encoded allocation/target')
    need(any(owner is type(runner) for owner,_,_ in encoded.attribute_bindings) and encoded.final_export is None,'runner binding/finalized')
    stats=encoded.stats();native=stats['native_decode'];q32=stats['q32_hoist']
    need(native.get('enabled') is True and q32.get('enabled') is True and encoded.mixed_native is not None,'two Native/Q32 runtime bindings')
    state=vars(runner).get(module._STATE);counts={k:0 for k in COUNTERS};objects=[]
    if state is not None:
        need(state['codes']==module._codes(runner) and state['event'] is runner.prepare_inputs_event,'gather runner/event binding')
        observed=state.get('diagnostics');need(type(observed) is dict,'cold diagnostic state absent')
        need(observed['overflow']==0,'compiled object overflow')
        for key in COUNTERS:
            value=observed[key];need(type(value) is int and value>=0,'invalid submit count');counts[key]=value
        kernel_module=sys.modules.get('vllm_fl.patches.biv150_dynamic_hidden_gather_kernel')
        if observed['compiled']:
            need(kernel_module is not None and Path(kernel_module.__file__).resolve()==ROOT/KERNEL,'actual kernel module path')
            objects=[compiled_record(k,kernel_module.gather_hidden_bits) for k in observed['compiled'].values()]
    need(sum(counts.values())==0 or objects,'submissions without actual compiled object')
    pid,ticks=process_identity()
    payload=dict(schema=SCHEMA,phase=phase,run_id=run_id,pid=pid,start_ticks=ticks,rpc_rank=vars(wrapper).get('rpc_rank'),
        monotonic_ns=time.monotonic_ns(),tool_source_sha256=source_sha256,sdk_pins=sdk_pins,
        source_files={p:sha(ROOT/p) for p in (HELPER,RUNNER,KERNEL,'vllm_fl/patches/biv150_compute_order.py')},
        helper_module=str(ROOT/HELPER),runner_module=str(ROOT/RUNNER),kernel_module=str(ROOT/KERNEL),
        implementation_sha256=encoded.installed_digest,native_library=native['library'],mixed_library=encoded.mixed_native.library,
        cold_initialized=state is not None,counts=counts,compiled=objects,compiled_overflow=0,
        host_submit_not_GPU_completion=True,Graph_replays_counted=False,model_precision_verified=False)
    data=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
    need(len(data.encode())<=MAX_JSON_BYTES,'snapshot exceeds size bound');return data


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
    need(not getattr(original,'_cjj_hidden_gather_wrapped',False),'API already wrapped')
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
    build._cjj_hidden_gather_wrapped=True;module.build_app=build
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

