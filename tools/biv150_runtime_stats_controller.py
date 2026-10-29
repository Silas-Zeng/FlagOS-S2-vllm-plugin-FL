"""Optional full-warm/score observer. No benchmark, CUDA, or general RPC."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import urllib.request

import biv150_runtime_stats_snapshot as snapshot

FILES=('tools/biv150_runtime_stats_controller.py','tools/biv150_runtime_stats_snapshot.py',
       'tools/biv150_runtime_stats_sdk_pins.json','tools/runtime_stats_bootstrap/sitecustomize.py',
       'tools/build_biv150_runtime_stats_controller.py','tools/run_biv150_native117_stats_service.py',
       'tools/run_biv150_parent107_stats_service.py')
PHASES=tuple(f'{kind}-{length}' for length in (4096,16384) for kind in ('warm-before','warm-after','score-after'))
NATIVE_SHA='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
CONTROLLERS={
    'tools/run_biv150_parent107_stats_service.py':dict(library_sha256=NATIVE_SHA,
        controller_sha256='6556a609261372769bf9e3e638e2771ceef3dffabd90df7a06093035f2df9f36',
        operator='cjj_native_k8_v_overlap_probe::produce'),
    'tools/run_biv150_native117_stats_service.py':dict(
        library_sha256='5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b',
        controller_sha256='3208684e9c64b66855478e4dc3ed2548eac2bd342b1f98aabe55deee74d0e237',
        operator='cjj_native_k8_klookahead_probe::produce'),
}
NATIVE_SHAS=frozenset(x['library_sha256'] for x in CONTROLLERS.values())


def need(value,message):
    if not value:raise ValueError('Stats controller: '+message)


def plan(a):
    if not getattr(a,'runtime_stats_bootstrap',False):return dict(enabled=False)
    need(a.throughput and a.full_workload_warmup and a.warmup_plan=='full-only' and a.throughput_case=='both',
         'requires throughput, full-only, full-workload-warmup, both lengths')
    need(not a.baseline and a.q32_hoist and a.native_k8_library is not None and a.native_k8_sha256 in NATIVE_SHAS,
         'requires reviewed Native8b or Native5898 with Q32 ON; --baseline is not main A')
    need(not a.math105 and not a.supplemental_native_math105_64 and a.graph_coverage_bootstrap_dir is None,
         'throughput only, no Math105 or Graph bootstrap combination')
    return dict(enabled=True,phases=list(PHASES),readonly_existing_counters=True,
        fixed_service_and_benchmark_CLI=True,Graph_replays_counted=False,model_precision_verified=False,
        actual_worker_ACK_required=True,includes_benchmark_client_preflight=True)


def clean_environment(original,source):
    result=dict(original)
    for key in (snapshot.ENABLE,snapshot.TOKEN,snapshot.RUN_ID):result.pop(key,None)
    directory=(Path(source)/'tools/runtime_stats_bootstrap').resolve()
    # Parent controller constructs PYTHONPATH=source before the service/client
    # seam; remove only our exact inserted path, preserve every other entry.
    if 'PYTHONPATH' in result:
        result['PYTHONPATH']=os.pathsep.join(p for p in result['PYTHONPATH'].split(os.pathsep)
            if not p or Path(p).resolve()!=directory)
    return result


class Session:
    def __init__(self,a,controller,expected):
        need(plan(a)['enabled'],'not enabled')
        self.source=a.source.resolve();self.out=a.out.resolve();self.commit=a.commit
        self.entry=next((name for name in CONTROLLERS if Path(controller).resolve()==self.source/name),None)
        need(self.entry is not None,'unknown controller or not in the same source')
        binding=CONTROLLERS[self.entry]
        need(a.native_k8_sha256==binding['library_sha256'],'controller/library argument mismatch')
        need(Path(__file__).resolve()==self.source/FILES[0] and snapshot.ROOT==self.source,'observer/tool source differs')
        self.hashes={}
        for name in FILES:
            path=self.source/name
            need(path.resolve()==path,'symlink observer source')
            raw=path.read_bytes()
            frozen=subprocess.check_output(['git','-C',str(self.source),'show',self.commit+':'+name])
            need(raw==frozen,'observer is not frozen in source commit: '+name)
            self.hashes[name]=hashlib.sha256(raw).hexdigest()
        need(self.hashes[self.entry]==binding['controller_sha256'],'controller bytes differ from the reviewed entry')
        self.run_id=self.out.name
        # Validate the public run id without generating/storing any real token.
        snapshot.configuration({snapshot.ENABLE:'1',snapshot.TOKEN:'x'*32,snapshot.RUN_ID:self.run_id})
        encoded=expected['konly_encoded'];native=encoded['native_decode']
        need(encoded.get('q32_hoist_enabled') is True and native.get('enabled') is True and native.get('library_sha256')==binding['library_sha256'] and native.get('operator')==binding['operator'],'expected active semantics differ')
        self.implementation=encoded['implementation_sha256']
        self.library=dict(path=str(a.native_k8_library.resolve()),sha256=binding['library_sha256'])
        self.directory=self.out/'runtime-stats';self.directory.mkdir()
        self.rows=[];self.previous=None;self._token=None

    def public(self):
        return dict(enabled=True,run_id=self.run_id,source_commit=self.commit,source_hashes=self.hashes,
            implementation_sha256=self.implementation,native_library=self.library,controller=self.entry,
            phases=list(PHASES),snapshots=list(self.rows),Graph_replays_counted=False,
            model_precision_verified=False,includes_benchmark_client_preflight=True)

    def check_sources(self):
        for name,digest in self.hashes.items():need(snapshot.sha(self.source/name)==digest,'observer source changed: '+name)

    def service_environment(self,original):
        need(self._token is None,'service bootstrap already prepared')
        self.check_sources();result=clean_environment(original,self.source)
        need(result.get('CJJ_NATIVE_GRAPH_COVERAGE')!='1' and 'CJJ_NATIVE_GRAPH_BOOTSTRAP_DIR' not in result,'Graph bootstrap environment present')
        self._token=secrets.token_urlsafe(32)
        result.update({snapshot.ENABLE:'1',snapshot.TOKEN:self._token,snapshot.RUN_ID:self.run_id,
            'PYTHONPATH':str(self.source/'tools/runtime_stats_bootstrap')+os.pathsep+result.get('PYTHONPATH','')})
        return result

    def capture(self,phase,owned,server_owner):
        need(self._token is not None and len(self.rows)<len(PHASES) and phase==PHASES[len(self.rows)],'snapshot order/activation')
        self.check_sources()
        request=urllib.request.Request('http://127.0.0.1:9031'+snapshot.ROUTE,
            data=json.dumps(dict(phase=phase,run_id=self.run_id)).encode(),
            headers={'Content-Type':'application/json','x-cjj-stats-token':self._token},method='POST')
        # Bypass ambient HTTP proxy and redirects; never send the private header
        # to another host. An uncertain timeout aborts this controller, no retry.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        try:
            with opener.open(request,timeout=60) as response:
                raw=response.read(snapshot.MAX_JSON_BYTES+1)
                claimed=response.headers.get('x-cjj-stats-sha256')
                need(response.status==200 and response.geturl()==request.full_url,'RPC response status/location')
        except Exception:
            raise RuntimeError('Stats RPC failed; no retry or next benchmark; original cleanup follows') from None
        need(len(raw)<=snapshot.MAX_JSON_BYTES and self._token.encode() not in raw,'RPC size/private payload')
        digest=hashlib.sha256(raw).hexdigest();need(digest==claimed,'RPC response SHA')
        value=json.loads(raw)
        for key,want in dict(schema=snapshot.SCHEMA,phase=phase,run_id=self.run_id,rpc_rank=0,
             tool_source_sha256=self.hashes[FILES[1]],sdk_pins=snapshot.SDK_PINS,
             implementation_sha256=self.implementation,service_module=str(self.source/'vllm_fl/patches/biv150_konly_encoded_service.py'),
             worker_type='vllm_fl.worker.worker.WorkerFL',runner_type='vllm_fl.worker.model_runner.ModelRunnerFL',
             graph_replays_counted=False,client_invocation_including_preflight=True).items():
            need(value.get(key)==want,'RPC binding differs: '+key)
        need(value['stats']['native_decode'].get('enabled') is True and value['stats']['native_decode'].get('library')==self.library,'actual Native library differs')
        need(value['stats']['q32_hoist'].get('enabled') is True,'actual Q32 disabled')
        owned.remember();pid=value.get('pid');ticks=value.get('start_ticks')
        need(type(pid) is int and pid>0 and type(ticks) is int and ticks>0,'worker PID identity invalid')
        member=owned.members.get(pid);live=owned.read(pid);api=owned.read(server_owner['pid'])
        need(member is not None and live is not None and live['state']!='Z' and member['start_ticks']==live['start_ticks']==ticks,'worker not currently owned/live')
        need(api is not None and api['state']!='Z' and api['start_ticks']==server_owner['start_ticks'] and live['sid']==api['sid'],'worker/server session differs')
        if self.previous is not None:
            for key in ('pid','start_ticks','implementation_sha256','tool_source_sha256'):
                need(value[key]==self.previous[key],'worker changed across boundaries: '+key)
        path=self.directory/(phase+'.json')
        with path.open('xb') as stream:stream.write(raw)
        row=dict(phase=phase,file=str(path.relative_to(self.out)),sha256=digest,bytes=len(raw),
                 pid=pid,start_ticks=ticks,ack_verified=True,owned_live_worker_verified=True)
        self.rows.append(row)
        self._index()
        if not phase.startswith('warm-before-'):
            delta=snapshot.counter_delta(self.previous,value)
            delta_path=self.directory/(phase+'-delta.json')
            with delta_path.open('x') as stream:json.dump(delta,stream,sort_keys=True,indent=2);stream.write('\n')
            row['delta_file']=str(delta_path.relative_to(self.out));row['delta_sha256']=snapshot.sha(delta_path)
        self.previous=value;self._index()

    def _index(self):
        path=self.directory/'index.json';temp=self.directory/'index.next'
        temp.write_text(json.dumps(self.public(),sort_keys=True,indent=2)+'\n');temp.replace(path)

    def finish(self):
        need([row['phase'] for row in self.rows]==list(PHASES),'not all six snapshots acknowledged')
        self.check_sources()
        return dict(self.public(),all_six_ACK_verified=True,index_sha256=snapshot.sha(self.directory/'index.json'),
                    runtime_HTTP_GPU_validation_proven_by_this_CPU_tool=False)
