"""Two fixed plain-smoke snapshots. No throughput observation or generic RPC."""
import hashlib,json,os,secrets,subprocess,urllib.request
from pathlib import Path
import biv150_hidden_gather_service_bridge as b
import biv150_hidden_gather_snapshot as s
import biv150_hidden_gather_path_identity as identity
PRIVATE_KEYS=(s.ENABLE,s.TOKEN,s.RUN_ID,b.DIAGNOSTICS)
DEFAULT_ENVIRONMENT=dict(name=b.ENV,present=False,raw_value=None,effective_enabled=True)

def need(ok,message):
    if not ok:raise ValueError('Gather smoke observer: '+message)

def delta(before,after):
    for key in ('schema','run_id','pid','start_ticks','rpc_rank','implementation_sha256','tool_source_sha256','sdk_pins',
                'source_files','helper_module','runner_module','kernel_module','native_library','mixed_library','feature_environment'):
        need(before[key]==after[key],'worker/source changed: '+key)
    need(before['phase']=='smoke-before' and after['phase']=='smoke-after' and before['schema']==s.SCHEMA,'fixed phase/schema')
    need(after['monotonic_ns']>=before['monotonic_ns'],'snapshot order')
    need(after['feature_environment']==DEFAULT_ENVIRONMENT,'worker must use the exact default, not explicit ON/OFF')
    values={}
    for name in s.COUNTERS:
        x,y=before['counts'][name],after['counts'][name]
        need(type(x) is int and type(y) is int and 0<=x<=y,'counter reset/invalid');values[name]=y-x
    need(before['compiled_overflow']==after['compiled_overflow']==0,'compiled overflow')
    return dict(counts=values,ordinary_submitted=values[s.COUNTERS[0]]>0,Plan_submitted=values[s.COUNTERS[1]]>0,
                both_sources_submitted=all(v>0 for v in values.values()),host_submit_not_GPU_completion=True,
                default_worker_environment_verified=True,Graph_replays_counted=False,model_precision_verified=False)

def compiled_verdict(row):
    binding=row.get('launcher_binding');expected=b.PINS['C_function']
    proof=row.get('source_path_identity',{})
    if not isinstance(proof,dict):return False
    if not (proof.get('verified') is True and proof.get('rule')==identity.RULE and
            proof.get('reference_root')==identity.OLD_ROOT and
            proof.get('source_root')==str(b.ROOT) and proof.get('raw_files')==row['files'] and
            proof.get('canonical_files')==b.PINS['files']):return False
    if not isinstance(binding,dict):return False
    ints=('C_address','DSO_base','DSO_offset','method_flags')
    if not all(type(binding.get(k)) is int for k in ints):return False
    path=binding.get('DSO_path')
    if type(path) is not str or not Path(path).is_absolute():return False
    return (binding['C_address']>binding['DSO_base']>0 and
            binding['C_address']-binding['DSO_base']==binding['DSO_offset']==expected['DSO_offset'] and
            binding['method_flags']==expected['method_flags'] and binding.get('method_name')==expected['method_name'] and
            binding.get('DSO_sha256')==b.PINS['launcher'] and Path(path).name==expected['filename'] and
            Path(path).parent.name==expected['cache_directory'] and
            row['hash']==b.PINS['hash'] and
            row['ABI']==b.PINS['ABI'] and row['options']==b.PINS['options'] and
            row['resources']==b.PINS['resources'] and row['launcher_sha256']==b.PINS['launcher'])

class Session:
    def __init__(self,a,controller,expected):
        b.options(a);need(a.gather_diagnostics is True,'smoke diagnostics not requested')
        self.source=a.source.resolve();self.out=a.out.resolve();self.commit=a.commit
        need(Path(controller).resolve()==self.source/b.CONTROLLER and self.source==b.ROOT==s.ROOT,'same frozen controller/tools')
        self.hashes={}
        for name in b.TOOLS:
            raw=(self.source/name).read_bytes()
            need(raw==subprocess.check_output(['git','-C',str(self.source),'show',self.commit+':'+name]),'unfrozen observer '+name)
            self.hashes[name]=hashlib.sha256(raw).hexdigest()
        self.source_hashes={name:b.sha(self.source/name) for name in (s.HELPER,s.RUNNER,s.KERNEL,'vllm_fl/patches/biv150_compute_order.py')}
        self.implementation=expected['konly_encoded']['implementation_sha256']
        need(expected['konly_encoded']['q32_hoist_enabled'] is True,'Q32 disabled')
        self.native=dict(path=str(a.native_k8_library.resolve()),sha256=b.PARENT)
        self.mixed=dict(path=str(a.mixed_native_library.resolve()),sha256=b.MIXED)
        need(expected['konly_encoded']['native_decode']['library_sha256']==b.PARENT and
             expected['konly_encoded']['mixed_native']['qualification']==b.QUALIFICATION,'same two Native fingerprints')
        self.run_id=self.out.name;s.configuration({s.ENABLE:'1',s.TOKEN:'x'*32,s.RUN_ID:self.run_id})
        self.directory=self.out/'gather-smoke-snapshots';self.directory.mkdir()
        self.rows=[];self.previous=None;self.verdict=None;self._token=None

    def check_sources(self):
        for name,pin in {**self.hashes,**self.source_hashes}.items():need(b.sha(self.source/name)==pin,'source changed: '+name)

    def service_environment(self,original):
        need(self._token is None,'observer already prepared');self.check_sources();result=dict(original)
        for key in PRIVATE_KEYS:result.pop(key,None)
        need(result.get(b.ENV)=='1' and result.get('CJJ_NATIVE_GRAPH_COVERAGE')!='1' and
             'CJJ_NATIVE_GRAPH_BOOTSTRAP_DIR' not in result and result.get('CJJ_RUNTIME_STATS_SNAPSHOT')!='1','gather/other observation flags')
        self._token=secrets.token_urlsafe(32)
        result.update({s.ENABLE:'1',s.TOKEN:self._token,s.RUN_ID:self.run_id,b.DIAGNOSTICS:'1',
                       'PYTHONPATH':str(self.source/'tools/hidden_gather_bootstrap')+os.pathsep+result.get('PYTHONPATH','')})
        result.pop(b.ENV) # Only the plain-smoke observer removes the validated ON key.
        return result

    def capture(self,phase,owned,server_owner):
        need(self._token is not None and len(self.rows)<2 and phase==s.PHASES[len(self.rows)],'activation/phase order');self.check_sources()
        request=urllib.request.Request('http://127.0.0.1:9031'+s.ROUTE,data=json.dumps(dict(phase=phase,run_id=self.run_id)).encode(),
            headers={'Content-Type':'application/json','x-cjj-stats-token':self._token},method='POST')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        try:
            with opener.open(request,timeout=60) as response:
                raw=response.read(s.MAX_JSON_BYTES+1);claimed=response.headers.get('x-cjj-stats-sha256')
                need(response.status==200 and response.geturl()==request.full_url,'response status/location')
        except Exception:raise RuntimeError('Gather snapshot RPC failed; no retry; original cleanup follows') from None
        need(len(raw)<=s.MAX_JSON_BYTES and self._token.encode() not in raw,'RPC size/private payload')
        digest=hashlib.sha256(raw).hexdigest();need(digest==claimed,'RPC SHA');value=json.loads(raw)
        expected=dict(schema=s.SCHEMA,phase=phase,run_id=self.run_id,rpc_rank=0,tool_source_sha256=self.hashes['tools/biv150_hidden_gather_snapshot.py'],
            sdk_pins=s.SDK_PINS,implementation_sha256=self.implementation,source_files=self.source_hashes,
            helper_module=str(self.source/s.HELPER),runner_module=str(self.source/s.RUNNER),kernel_module=str(self.source/s.KERNEL),
            native_library=self.native,mixed_library=self.mixed,feature_environment=DEFAULT_ENVIRONMENT,
            host_submit_not_GPU_completion=True,Graph_replays_counted=False,model_precision_verified=False)
        for key,want in expected.items():need(value.get(key)==want,'RPC identity '+key)
        owned.remember();pid=value.get('pid');ticks=value.get('start_ticks')
        need(type(pid) is int and pid>0 and type(ticks) is int and ticks>0,'worker identity missing')
        member=owned.members.get(pid);live=owned.read(pid);api=owned.read(server_owner['pid'])
        need(member is not None and live is not None and live['state']!='Z' and member['start_ticks']==live['start_ticks']==ticks,'worker not currently owned')
        need(api is not None and api['state']!='Z' and api['start_ticks']==server_owner['start_ticks'] and live['sid']==api['sid'],'worker/API epoch/session')
        path=self.directory/(phase+'.json')
        with path.open('xb') as stream:stream.write(raw)
        self.rows.append(dict(phase=phase,file=str(path.relative_to(self.out)),sha256=digest,bytes=len(raw),pid=pid,start_ticks=ticks,ack_verified=True))
        self._index()
        need(value['compiled_overflow']==0 and all(compiled_verdict(row) for row in value['compiled']),'actual gather compilation differs; original ACK preserved')
        need(sum(value['counts'].values())==0 or value['compiled'],'submissions without compiled evidence')
        if self.previous is not None:
            self.verdict=delta(self.previous,value)
            self.verdict['actual_compile_identity_verified']=bool(value['compiled']) and all(compiled_verdict(row) for row in value['compiled'])
            self.verdict['default_worker_gather_verified']=self.verdict['default_worker_environment_verified'] and bool(any(self.verdict['counts'].values())) and self.verdict['actual_compile_identity_verified']
        self.previous=value;self._index()

    def public(self):
        return dict(enabled=True,source_commit=self.commit,source_hashes=self.hashes,run_id=self.run_id,
                    snapshots=list(self.rows),delta=self.verdict,worker_environment_required=DEFAULT_ENVIRONMENT,
                    host_submit_not_GPU_completion=True,model_precision_verified=False)

    def _index(self):
        path=self.directory/'index.json';temp=self.directory/'index.next'
        temp.write_text(json.dumps(self.public(),sort_keys=True,indent=2)+'\n');temp.replace(path)

    def finish(self):
        need([row['phase'] for row in self.rows]==list(s.PHASES),'two worker ACKs required');self.check_sources()
        return dict(self.public(),both_ACK_verified=True,index_sha256=b.sha(self.directory/'index.json'),
                    service_model_precision_verified=False,service_performance_verified=False)
