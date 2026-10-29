"""Explicit old-device/new-host bridge; CPU preparation never approves service."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE='64989ebfeca8066f6c28b4a14de33dfb044d9b6f'
FROZEN='01dfe84028560b1a982dac33bc788e875ce9cf1b'
MAIN='56f88737fcf3b940760b9ca3027a925cd5a7d77d'
PARENT='5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b'
MIXED='ebbc35d028bb7c719895c8b3569f3120387380888f23c4bb772b7176f0856bba'
STATIC='f75044d8f636157ab2a64392ba8e09ead013e671b11f9cfab0c25246f4ccbeb7'
CONSUMER='458cd7638ad5b9df8efb3d3f72f626a3414ee947da5cc12f0524c7c1fc97d653'
STEP='db7a21308aa07b89b8e35493a9c05d2ccf8fc3028c173d1dbd2df37824b0cc49'
PINS=dict(library_sha256=MIXED,actual_so_review_sha256=STATIC,consumer_review_sha256=CONSUMER,fullstep_review_sha256=STEP)
MODULE='vllm_fl/patches/biv150_konly_encoded_mixed_native.py'
PAIRS=[[29,4],[30,4],[31,4],[60,2],[61,2],[63,2]]
QUALIFICATION=dict(**PINS,physical_page_bytes=16384,rows_segments=PAIRS,service_verified=False)
SCOPE=dict(page_bytes=16384,pages=34276,query_pitch=2048,scratch_width=16,
           new_K_pitch=256,new_V_pitch=2560,rows_context={'4k':[61,4096],'16k':[30,16384]})
ENV='VLLM_FL_BIV150_KONLY_MIXED_NATIVE'
PATH_ENV=ENV+'_LIBRARY'
SHA_ENV=ENV+'_LIBRARY_SHA256'


def need(ok,why):
    if not ok:raise ValueError('PR118 service bridge: '+why)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])

def production(root,ref):
    """Same mode/link-text definition as 118 launcher; read-through is separate."""
    root=Path(root).resolve();records={};contents={}
    for row in git(root,'ls-tree','-r','-z',ref,'--','vllm_fl','csrc').split(b'\0'):
        if not row:continue
        meta,name=row.split(b'\t');mode,kind,oid=meta.split();name=name.decode();p=root/name
        need(kind==b'blob' and mode in (b'100644',b'100755',b'120000'),'production mode')
        need(p.is_symlink()==(mode==b'120000'),'actual production link type')
        if mode==b'120000':
            raw=os.readlink(os.fsencode(p));need(p.resolve().is_relative_to(root),'link leaves source')
        else:raw=p.read_bytes()
        need(raw==git(root,'cat-file','blob',oid.decode()),'actual production differs from frozen Git: '+name)
        records[name]=dict(mode=mode.decode(),sha256=hashlib.sha256(raw).hexdigest())
        contents[name]=sha(p)
    return dict(mode_and_link_text=records,mode_and_link_text_digest=digest(records),
                read_through_content_sha256=contents,read_through_content_digest=digest(contents))

def qualification(root):
    tree=ast.parse((Path(root)/MODULE).read_bytes())
    value=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='QUALIFICATION' for t in n.targets))
    need(value==QUALIFICATION,'exact four qualification pins/scope')
    return value

def current_identity(root,ref):
    root=Path(root).resolve();q=qualification(root)
    # No production changes beyond the four real component review pins.
    paths=git(root,'diff','--name-only',BASE,'--','vllm_fl','csrc').decode().splitlines()
    need(paths==[MODULE],'production changes beyond four-pin module')
    raw=(root/MODULE).read_bytes()
    for key,value in PINS.items():
        new=(repr(key)+':'+repr(value)).encode();old=(repr(key)+':None').encode()
        need(raw.count(new)==1,'pin inverse count');raw=raw.replace(new,old)
    need(raw==git(root,'show',BASE+':'+MODULE),'four-pin inverse differs')
    for name in ('biv150_konly_encoded_native.py','biv150_konly_encoded_sources.json'):
        p='vllm_fl/patches/'+name
        need((root/p).read_bytes()==git(root,'show',MAIN+':'+p),'accepted5898/C3 changed')
    return dict(production=production(root,ref),qualification=q,retained_native_parent_sha256=PARENT,
                accepted_main=MAIN,service_verified=False)

def prepare(root,ref,compile_summary,consumer_review,step_review):
    root=Path(root).resolve()
    need(sha(consumer_review)==CONSUMER and sha(step_review)==STEP,'actual Root review hashes')
    c=json.loads(Path(consumer_review).read_bytes());s=json.loads(Path(step_review).read_bytes())
    j=json.loads(Path(compile_summary).read_bytes())
    need(c['passed'] is c['mathematical_component_verified'] is s['passed'] is True,'actual reviewed components')
    need(c['source_commit']==s['source_commit']==j['source_commit']==FROZEN,'frozen actual source')
    need(j['completed'] is j['compile_passed'] is True and j['errors']==[],'actual compilation')
    need(c['recorded_binding']['compile_summary_sha256']==sha(compile_summary),'compile summary binding')
    need(c['recorded_binding']['assembly_review_sha256']==STATIC,'static review binding')
    need(c['recorded_binding']['library']['sha256']==j['library']['sha256']==s['mixed_SO']==MIXED,'actual final SO')
    need(s['consumer_independent_review_sha256']==CONSUMER and s['scope']==SCOPE,'actual step consumer/scope')
    library=Path(compile_summary).parent/'build/cjj_native_k8_mixed_probe.so'
    need(sha(library)==MIXED,'actual final SO bytes')
    import biv150_native_k8_mixed_factory as f
    import compile_biv150_native_k8_mixed as b
    need(f.ROOT==root,'factory root');proof=f.verify();plan=b.cpu_plan()
    need(f.BASE=='a455a3de364051fbea80226870c239de3a48e716' and f.HOST_BASE==MAIN,'GPU/host provenance')
    need({k:v for k,v in proof.items() if k!='host_reverse_base'}==j['source_transform'],'sole source proof delta')
    fields=('source_files','explicit_host_flags','explicit_cuda_flags','symbol_renames','frontend_sha256')
    need(all(plan[k]==j[k] for k in fields),'actual GPU inputs/options/frontend')
    for role in f.FILES:need(f.path(role).read_bytes()==(Path(compile_summary).parent/f.path(role).name).read_bytes(),'actual input bytes')
    need(hashlib.sha256(git(root,'show',FROZEN+':tools/biv150_native_k8_mixed_factory.py')).hexdigest()==j['factory_sha256'],'old factory')
    admissions={}
    for name in ('biv150_native_k8_mixed_numeric_contract.py','probe_biv150_native_k8_mixed_consumer.py','probe_biv150_native_k8_mixed_decode_step.py'):
        path='tools/'+name
        need((root/path).read_bytes()==git(root,'show',FROZEN+':'+path),'original admission source unchanged')
        admissions[path]=sha(root/path)
    return dict(schema='cjj-native118-service-source-bridge-v1',**current_identity(root,ref),
        frozen_actual_source=FROZEN,actual_compile_summary_sha256=sha(compile_summary),actual_SO_sha256=MIXED,
        actual_SO_static_review_sha256=STATIC,actual_consumer_review_sha256=CONSUMER,
        actual_decode_step_review_sha256=STEP,decode_step_scope=SCOPE,
        old_source_transform=j['source_transform'],host_source_transform=proof,
        old_factory_sha256=j['factory_sha256'],host_factory_sha256=plan['factory_sha256'],
        identical_actual_compile_fields={k:plan[k] for k in fields},
        preserved_admission_source_sha256=admissions,
        original_consumer_admission_modified=False,source_bridge_Root_approved=False,
        service_admission_authorized=False,model_precision_verified=False,
        limitation='Component evidence only. Natural mixed service and shared-QKV backing still require real smoke/model tests.')

def validate(root,ref,path,expected_sha,library):
    need(Path(path).is_absolute() and sha(path)==expected_sha,'explicit Root bridge bytes')
    j=json.loads(Path(path).read_bytes())
    need(j.get('schema')=='cjj-native118-service-source-bridge-v1','bridge schema')
    need(j.get('source_bridge_Root_approved') is True and j.get('service_admission_authorized') is True,'Root approval pending')
    for k,v in current_identity(root,ref).items():need(j.get(k)==v,'final production binding: '+k)
    for k,v in dict(frozen_actual_source=FROZEN,actual_SO_sha256=MIXED,actual_SO_static_review_sha256=STATIC,
                    actual_consumer_review_sha256=CONSUMER,actual_decode_step_review_sha256=STEP,
                    decode_step_scope=SCOPE,original_consumer_admission_modified=False,model_precision_verified=False).items():
        need(j.get(k)==v,'actual bridge identity/scope: '+k)
    need(Path(library).is_absolute() and sha(library)==MIXED,'explicit mixed library bytes')
    return j

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','compile-summary','consumer-review','step-review','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args()
    need(git(a.source,'rev-parse','HEAD').decode().strip()==a.commit and not git(a.source,'status','--porcelain'),'clean frozen source')
    a.out.write_text(json.dumps(prepare(a.source,a.commit,a.compile_summary,a.consumer_review,a.step_review),sort_keys=True,indent=2)+'\n')
    print(sha(a.out))

# Controller-only observation: existing host markers/final stats, no RPC or GPU.
def clear_environment(original):
    env=dict(original)
    for key in (ENV,PATH_ENV,SHA_ENV):env.pop(key,None)
    return env

def service_environment(original,library):
    env=clear_environment(original)
    need(env.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE')=='1' and
         env.get('VLLM_FL_BIV150_KONLY_NATIVE_DECODE')=='1' and
         env.get('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','0')!='1','encoded/native ON compact OFF')
    env.update({ENV:'1',PATH_ENV:str(Path(library).resolve()),SHA_ENV:MIXED})
    return env

def options(a):
    need(not a.baseline and a.native_k8_library is not None and a.native_k8_sha256==PARENT,'accepted5898 candidate required')
    need(a.graph_coverage_bootstrap_dir is None,'old5898 Graph bootstrap not mixed evidence')
    need(a.mixed_native_library is not None and a.mixed_native_library.is_absolute() and a.mixed_native_sha256==MIXED,'exact ebbc path/SHA required')
    need(a.mixed_source_bridge is not None and a.mixed_source_bridge.is_absolute() and
         type(a.mixed_source_bridge_sha256) is str and len(a.mixed_source_bridge_sha256)==64 and
         all(c in '0123456789abcdef' for c in a.mixed_source_bridge_sha256),'explicit Root bridge/SHA required')

def fingerprint(expected):
    need(expected.get('konly_encoded',{}).get('mixed_native')==dict(enabled=True,
         operator='cjj_native_k8_mixed_probe::produce',qualification=QUALIFICATION),'mixed compile fingerprint')

def mixed_events(logs):
    import re
    events=[];marker='BIV150_KONLY_MIXED_NATIVE_SELECTED '
    for line in logs.splitlines():
        if marker not in line:continue
        prefix,raw=line.split(marker,1);e=json.loads(raw)
        need(type(e.get('rows')) is int and type(e.get('segments')) is int and [e['rows'],e['segments']] in PAIRS,'mixed event exact N/S')
        need(e.get('library_sha256')==MIXED and e.get('host_selection_not_replay_count') is True,'mixed event library/claim')
        matches=re.findall(r'\bpid=(\d+)\b',prefix)
        events.append(dict(event=e,worker_pid=int(matches[-1]) if matches else None))
    return events

def smoke_window(log,start,end):
    with Path(log).open('rb') as stream:stream.seek(start);raw=stream.read(end-start)
    return dict(log_byte_start=start,log_byte_end=end,sha256=hashlib.sha256(raw).hexdigest(),
        mixed_events=mixed_events(raw.decode()),Graph_replays_counted=False,
        host_selection_only=True,service_verified=False)

def final_evidence(logs,out,expected,owned,server_owner,window,native_library,native_parser):
    """Bind shutdown payload PID/start to recorded owner and both actual libraries."""
    rows=[];marker='BIV150_KONLY_ENCODED_FINAL_STATS '
    for line in logs.splitlines():
        if marker not in line:continue
        claim=json.loads(line.split(marker,1)[1]);p=Path(claim['path'])
        pid=claim.get('pid');need(type(pid) is int and pid>0,'final worker pid')
        need(p.resolve()==(Path(out)/f'konly-encoded-final-{pid}.json').resolve(),'final stats output path')
        need(sha(p)==claim['sha256'] and p.stat().st_size==claim['bytes'],'final stats actual bytes')
        value=json.loads(p.read_bytes());ticks=value.get('start_ticks');member=owned.members.get(pid)
        need(type(ticks) is int and ticks>0 and member is not None and member['start_ticks']==ticks and
             member['sid']==server_owner['sid'],'final worker owned same session/start')
        need(value.get('pid')==pid and value.get('schema')=='cjj-konly-encoded-final-stats-v1' and
             value.get('implementation_sha256')==expected['konly_encoded']['implementation_sha256'],'final worker source/identity')
        stats=value['stats'];mixed=stats['mixed_native'];native=stats['native_decode']
        need(mixed.get('enabled') is True and mixed.get('library')==dict(path=expected['_mixed_library_path'],sha256=MIXED)
             and mixed.get('rows_segments')==PAIRS and mixed.get('service_verified') is False,'actual mixed runtime identity')
        calls=mixed.get('host_calls_not_graph_replays');need(type(calls) is int and calls>=0,'mixed host count')
        need(native.get('enabled') is True and native.get('library')==dict(path=str(Path(native_library).resolve()),sha256=PARENT),'same worker5898 identity')
        # Reuse the original117 geometry/ABI parser on this same-worker payload.
        text='\n'.join(('BIV150_KONLY_NATIVE_DECODE_CAPTURED ' if r['capture'] else 'BIV150_KONLY_NATIVE_DECODE_SELECTED ')+json.dumps(r) for r in native['records'])
        need(all(type(r.get('host_calls')) is int and r['host_calls']>0 for r in native['records']),'native record host counts')
        parsed=native_parser(text,native['library']);buckets=sorted({r['event']['shape'] for r in parsed})
        in_window=[e for e in window['mixed_events'] if e['worker_pid']==pid]
        rows.append(dict(pid=pid,start_ticks=ticks,sha256=claim['sha256'],mixed_lifetime_host_calls=calls,
                         smoke_mixed_events=in_window,native_same_worker_buckets=buckets,
                         verified=calls>0 and bool(in_window) and buckets==[32,64]))
    return dict(workers=rows,mixed_same_worker_smoke_verified=any(r['verified'] for r in rows),
        coverage_missing=not any(r['verified'] for r in rows),Graph_replays_counted=False,
        counts_are_lifetime_not_smoke_or_score_window=True,service_verified=False)

def mixed_smoke_requests(request_fn=None):
    """Smoke-only overlap opportunity; 120s first-token and 360s total per group.

    HTTP reads time out after at most30s. On total timeout, active readers stop
    at their next read boundary; executor shutdown may take up to another30s.
    First-token/early-finish gaps are observations, not operator failures.
    """
    import threading
    import time
    import urllib.request
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    deadline=None;abort=threading.Event()
    if request_fn is None:
        def request_fn(index,payload,on_first):
            need(not abort.is_set() and time.monotonic()<deadline,'smoke total stage timeout')
            req=urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',data=json.dumps(payload).encode(),
                                      headers={'Content-Type':'application/json','x-request-id':payload['user']})
            with urllib.request.urlopen(req,timeout=min(30,max(.1,deadline-time.monotonic()))) as response:
                if not payload.get('stream'):return json.load(response)
                chunks=[]
                for line in response:
                    need(not abort.is_set() and time.monotonic()<deadline,'smoke total stage timeout')
                    if not line.startswith(b'data: '):continue
                    raw=line[6:].strip()
                    if raw==b'[DONE]':break
                    value=json.loads(raw);chunks.append(value)
                    for choice in value.get('choices',[]):
                        delta=choice.get('delta',{})
                        if delta.get('content') or delta.get('reasoning_content'):on_first()
                return dict(chunks=chunks,usage=next((x['usage'] for x in reversed(chunks) if x.get('usage')),{}))
    groups=[]
    for keepers,arrivals in ((61,3),(30,2)):
        began=time.monotonic();deadline=began+360;first_deadline=began+120;abort.clear()
        gate=threading.Barrier(keepers);first=[threading.Event() for _ in range(keepers)]
        lock=threading.Lock();records=[];token_times={};tag=f'cjj-mixed-smoke-{keepers}-'+uuid.uuid4().hex
        def request(i,payload,on_first):
            row=dict(index=i,request_id=payload['user'],request=payload,started_seconds=time.monotonic()-began)
            try:
                row['response']=request_fn(i,payload,on_first);row['complete']=True
                row['response_ids']=sorted({x['id'] for x in row['response'].get('chunks',[row['response']]) if x.get('id')})
            except Exception as error:
                row.update(complete=False,error=type(error).__name__+': '+str(error))
            row['finished_seconds']=time.monotonic()-began
            with lock:records.append(row)
            return row
        def keeper(i):
            payload=dict(model='minicpm',messages=[dict(role='user',content=f'编号{i}，持续列出自然数和简短解释，从1开始。')],
                         user=tag+'-'+str(i),max_tokens=512,temperature=0,top_p=1,ignore_eos=True,stream=True,stream_options={'include_usage':True})
            def token():
                with lock:token_times.setdefault(i,time.monotonic()-began)
                first[i].set()
            gate.wait(timeout=30);return request(i,payload,token)
        def arrival(i):
            payload=dict(model='minicpm',messages=[dict(role='user',content=('1 2 3 4 '*1024)+f'。编号{i}，继续输出数字。')],
                         user=tag+'-'+str(keepers+i),max_tokens=16,temperature=0,top_p=1,ignore_eos=True)
            return request(keepers+i,payload,lambda:None)
        gaps=[]
        with ThreadPoolExecutor(max_workers=keepers+arrivals) as pool:
            live=[pool.submit(keeper,i) for i in range(keepers)]
            while not all(e.is_set() for e in first):
                if any(x.done() for x in live):gaps.append('keeper_completed_before_all_first_tokens');break
                if time.monotonic()>=first_deadline:gaps.append('first_token_barrier_timeout');break
                time.sleep(.01)
            active=[i for i,x in enumerate(live) if not x.done()]
            first_ids=[i for i,e in enumerate(first) if e.is_set()]
            if len(active)!=keepers:gaps.append('some_keepers_finished_before_injection')
            # No live keeper means no overlap opportunity; do not pretend a hit.
            incoming=[pool.submit(arrival,i) for i in range(arrivals)] if active and first_ids else []
            if not incoming:gaps.append('no_live_generated_keeper_for_injection')
            try:
                for task in live+incoming:task.result(timeout=max(.01,deadline-time.monotonic()))
            except TimeoutError:gaps.append('total_stage_timeout');abort.set()
            finally:
                if time.monotonic()>=deadline:abort.set()
        records.sort(key=lambda x:x['index'])
        for row in records:
            expected=512 if row['index']<keepers else 16
            if not row['complete'] or row['response'].get('usage',{}).get('completion_tokens')!=expected:
                gaps.append('incomplete_or_failed_smoke_request_'+str(row['index']))
        groups.append(dict(keepers=keepers,new_long_prompts=arrivals,request_id_prefix=tag,
            first_token_seconds={str(i):t for i,t in token_times.items()},first_tokens_at_injection=first_ids,
            active_keeper_indices_at_injection=active,new_requests_submitted=len(incoming),records=records,
            elapsed_seconds=time.monotonic()-began,first_token_timeout_seconds=120,total_timeout_seconds=360,
            coverage_gaps=gaps,host_overlap_opportunity_only=True,actual_mixed_bucket_proven=False))
    return dict(groups=groups,smoke_only=True,official_benchmark_unchanged=True,
                actual_mixed_hit_requires_post_op_marker=True,Graph_replays_counted=False)
