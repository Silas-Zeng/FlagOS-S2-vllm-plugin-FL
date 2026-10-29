"""Default-off multi2 hoist: explicit component-to-new-worker service bridge."""
import hashlib,json,re,time,uuid,threading,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import biv150_native118_service_bridge as old
MAIN='899ce77470b0b539aebc108fd122f668c2d5894d'
PAIR_SOURCE='fe2c7295e0f34a85269b1a48242d64451176e003'
COMPILE_SOURCE='9b4e032dc6390f4f19c07a7c996ed898ba908d9e'
COMPILE_REVIEW='f7d2ff77125dbe832d9cff98965858d638707c79fd706c00fceaff8c9152692a'
COMPILE_SUMMARY='12df8d1cbf08f58ae53562a9ba0abec9cde5389917ee129a642365f0cdfb625b'
PAIR_REVIEW='3eba51b9912c1659f0f0f54e29694c6b42f0de7ba57cc68d3d3db4afb2652273'
PAIR_SUMMARY='b0a4de1d63bb1c8c0aeaf1fae4ecf6d4911daa7542125436691466d6e4411409'
PAIR_CUBINS=dict(unhoisted='f0599daf8ec6c1daae817deae3e66839037171f1ded8944548c69100fd622267',hoisted='d8fba5d8f19305a0928ed9cf10e98256dfdeffbf5cc82dc71c12224f6e57b860')
ENV='VLLM_FL_BIV150_KONLY_SERVICE_MULTI2_HOIST'
LEGACY_TOOL='tools/biv150_native118_service_bridge.py'
LEGACY_SHA='e943ea850eae1f1bb2ddcbd0590dc25f23936f182ea38f2e8ca33365226ec176'
PARENT_CONTROLLER='tools/run_biv150_native118_service.py'
PARENT_CONTROLLER_SHA='637d29e30c86a56e0536548b5678067954c8900046dc2d1a9dd78a96f4fa7199'
BRIDGE='tools/biv150_multi2_hoist_service_bridge.py'
CONTROLLER='tools/run_biv150_multi2_hoist_service.py'
EDITS=[('import biv150_native118_service_bridge as mixed_bridge', 'import biv150_multi2_hoist_service_bridge as mixed_bridge'), ("tool=source/'tools/biv150_native118_service_bridge.py'", "tool=source/'tools/biv150_multi2_hoist_service_bridge.py'"), ("entry=source/'tools/run_biv150_native118_service.py'", "entry=source/'tools/run_biv150_multi2_hoist_service.py'"), ("a.commit+':tools/biv150_native118_service_bridge.py'", "a.commit+':tools/biv150_multi2_hoist_service_bridge.py'"), ("a.commit+':tools/run_biv150_native118_service.py'", "a.commit+':tools/run_biv150_multi2_hoist_service.py'"), ("        save('short_request_smoke')", "        if not (a.math105 or a.supplemental_native_math105_64 or a.throughput):\n            save('multi2_pair_smoke')\n            pair=mixed_bridge.pair_requests();m.atomic(out/'multi2-pair-smoke-requests.json',pair);owned.remember()\n        save('short_request_smoke')"), ("        state['smoke_passed'] = True; save('smoke_complete')", "        manifest['multi2_smoke_evidence']=mixed_bridge.window_evidence(out/'server.log',smoke_start,(out/'server.log').stat().st_size,out,expected)\n        m.atomic(out/'manifest.json',manifest)\n        state['smoke_passed'] = True; save('smoke_complete')")]

# Original functions retain their original globals; no old admission is patched.
MIXED=old.MIXED;PARENT=old.PARENT;QUALIFICATION=old.QUALIFICATION
sha=old.sha;git=old.git;smoke_window=old.smoke_window
mixed_smoke_requests=old.mixed_smoke_requests;final_evidence=old.final_evidence

def need(ok,why):
    if not ok:raise ValueError('PR123 service bridge: '+why)

def clear_environment(env):
    env=old.clear_environment(env);env.pop(ENV,None);return env

def service_environment(env,library):
    env=old.service_environment(env,library)
    need(env.get('VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST')=='1','retain Q32 ON')
    env[ENV]='1';return env

def options(a):
    old.options(a);need(a.q32_hoist is True,'Q32 remains ON')

def fingerprint(value):
    old.fingerprint(value)
    need(value['konly_encoded'].get('multi2_hoist_enabled') is True,'multi2 switch absent from actual C3')

def current_identity(root,ref):
    root=Path(root).resolve()
    need(git(root,'rev-parse','HEAD').decode().strip()==ref and not git(root,'status','--porcelain','--untracked-files=all'),'clean frozen source')
    git(root,'merge-base','--is-ancestor',MAIN,ref);git(root,'merge-base','--is-ancestor',PAIR_SOURCE,ref)
    need(git(root,'diff',COMPILE_SOURCE,ref,'--','vllm_fl','csrc')==b'','production must equal actual9b prepare source')
    need(git(root,'diff',PAIR_SOURCE,ref,'--','vllm_fl','csrc')==b'','production must equal actualfe2c numeric source')
    need(Path(old.__file__).resolve()==root/LEGACY_TOOL and sha(root/LEGACY_TOOL)==LEGACY_SHA,'fixed original support')
    for name in (LEGACY_TOOL,PARENT_CONTROLLER):need((root/name).read_bytes()==git(root,'show',MAIN+':'+name),'unchanged main899 protocol/support')
    need(sha(root/PARENT_CONTROLLER)==PARENT_CONTROLLER_SHA,'original controller SHA')
    inverse=(root/CONTROLLER).read_text()
    for before,after in reversed(EDITS):need(inverse.count(after)==1,'controller seam count');inverse=inverse.replace(after,before,1)
    need(inverse.encode()==(root/PARENT_CONTROLLER).read_bytes(),'complete controller inverse')
    for name in (BRIDGE,CONTROLLER):need((root/name).read_bytes()==git(root,'show',ref+':'+name),'tracked new tool')
    # These accepted runtime modules/libraries are unchanged, including their C3 identity.
    for name in ('biv150_konly_encoded_native.py','biv150_konly_encoded_mixed_native.py','biv150_konly_encoded_multi2.py','biv150_konly_encoded_q32.py'):
        path='vllm_fl/patches/'+name
        need((root/path).read_bytes()==git(root,'show',MAIN+':'+path),'unchanged accepted GPU/runtime: '+name)
    return dict(source_commit=ref,accepted_main=MAIN,production=old.production(root,ref),qualification=old.qualification(root),
        native_library_sha256=PARENT,mixed_library_sha256=MIXED,original118_bridge_sha256=LEGACY_SHA,
        parent_controller_sha256=PARENT_CONTROLLER_SHA,controller_sha256=sha(root/CONTROLLER),bridge_tool_sha256=sha(root/BRIDGE),
        production_equal_actual9b_and_fe2c=True,controller_inverse_to118=True,service_verified=False)

def record(root,ref):
    return dict(schema='cjj-multi2-hoist-service-source-bridge-v1',**current_identity(root,ref),
        component=dict(compile_source=COMPILE_SOURCE,compile_review_sha256=COMPILE_REVIEW,compile_summary_sha256=COMPILE_SUMMARY,
          pair_source=PAIR_SOURCE,pair_review_sha256=PAIR_REVIEW,pair_summary_sha256=PAIR_SUMMARY,actual_pair_cubin_sha256=PAIR_CUBINS,
          full_outputs=10,stages=5,query_rows=2048,num_sequences=2,query_pitch=2048,physical_page_bytes=16384,
          original_BF16_proxy_passed=False,quantized_reference_hard_gates_passed=True),
        default_off_switch=ENV,source_bridge_Root_approved=False,service_admission_authorized=False,
        new_worker_numeric_verified=False,model_precision_verified=False,service_throughput_verified=False,
        limitation='Actual9b pair is component evidence. Fresh service preparation retains the existing native audit and object/ABI checks, but gets its own actual binding and artifact identities. No automatic numeric/Graph/model/performance qualification is transferred. First-selected evidence is a host event, not invocation frequency or Graph replay count. BF16 proxy failures remain diagnostic.')

def prepare(root,ref,compile_review,pair_review,parent_library,mixed_library):
    need(sha(compile_review)==COMPILE_REVIEW and sha(pair_review)==PAIR_REVIEW,'original actual independent reports')
    c=json.loads(Path(compile_review).read_bytes());n=json.loads(Path(pair_review).read_bytes())
    need(c['review_passed'] is True and c['source_commit']==COMPILE_SOURCE and c['summary_sha256']==COMPILE_SUMMARY,'actual prepare evidence')
    need(n['review_completed'] is n['numeric_verified'] is n['all_available_full_outputs_bitwise'] is n['Graph_verified'] is n['side_C_verified'] is True,'complete pair hard gates')
    need(n['source_commit']==PAIR_SOURCE and n['summary_sha256']==PAIR_SUMMARY,'numeric source/summary')
    need(n['original_BF16_quality_passed'] is n['precision_gate'] is n['service_verified'] is False,'keep original proxy/service boundary')
    need({a:n['executable_binding'][a]['cubin_sha256'] for a in PAIR_CUBINS}==PAIR_CUBINS,'actual two tested cubins')
    need(sha(parent_library)==PARENT and sha(mixed_library)==MIXED,'accepted actual two libraries')
    return record(root,ref)

def validate(root,ref,path,expected_sha,library):
    need(Path(path).is_absolute() and sha(path)==expected_sha,'explicit new bridge SHA')
    value=json.loads(Path(path).read_bytes());expected=record(root,ref)
    need(value.get('source_bridge_Root_approved') is True and value.get('service_admission_authorized') is True,'Root approval pending')
    expected.update(source_bridge_Root_approved=True,service_admission_authorized=True)
    need(value==expected,'exact new source/component bridge')
    need(Path(library).is_absolute() and sha(library)==MIXED,'actual mixed library');return value

def pair_requests():
    """Only plain smoke: two queued prefill requests before any decode keeper."""
    tag='cjj-multi2-pair-'+uuid.uuid4().hex;gate=threading.Barrier(2);began=time.monotonic();deadline=began+180
    def one(i):
        request_id=tag+'-'+str(i);payload=dict(model='minicpm',messages=[dict(role='user',content='1 2 3 4 5 '*300+'请回答最后一个数。')],max_tokens=16,temperature=0,top_p=1)
        row=dict(request_id=request_id,request=payload,complete=False,response=None,error=None)
        try:
            gate.wait(timeout=10)
            request=urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Request-ID':request_id})
            row['submitted_seconds']=time.monotonic()-began
            with urllib.request.urlopen(request,timeout=max(.01,deadline-time.monotonic())) as response:
                chunks=[];size=0
                while True:
                    need(time.monotonic()<deadline,'pair response total deadline')
                    part=response.read1(65536)
                    if not part:break
                    size+=len(part);need(size<=4*1024*1024,'bounded pair JSON');chunks.append(part)
                value=json.loads(b''.join(chunks))
            row['response']=value;row['complete']=len(value.get('choices',[]))==1 and value.get('usage',{}).get('completion_tokens',0)>0
        except Exception as error:row['error']=type(error).__name__+': '+str(error)
        row['completed_seconds']=time.monotonic()-began;return row
    # Both worker futures are joined; context managers close HTTP responses on errors.
    # Each blocking read has <=180s socket timeout and the loop rechecks the total
    # deadline. Thus a late read may extend past180s, bounded by another180s.
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,range(2)))
    return dict(smoke_only=True,request_id_prefix=tag,records=rows,requests=2,concurrency=2,
        completed=sum(x['complete'] for x in rows),elapsed_seconds=time.monotonic()-began,
        deadline_seconds=180,maximum_blocking_read_overrun_seconds=180,
        intended_prompt_tokens_approximate=3000,actual_prompt_tokens=[(x['response'] or {}).get('usage',{}).get('prompt_tokens') for x in rows],
        coverage_gaps=[] if all(x['complete'] for x in rows) else ['incomplete_pair_requests'],
        scheduling_or_multi2_hit_guaranteed=False,formal_benchmark_unchanged=True)

def window_evidence(log,start,end,out,expected):
    """Read existing PREPARED/binding and post-op markers; never add hot hooks."""
    raw=Path(log).read_bytes();need(0<=start<=end<=len(raw),'bounded ready request log window')
    def events(data,marker):
        result=[]
        for line in data.decode(errors='replace').splitlines():
            if marker+' ' not in line:continue
            head,body=line.split(marker+' ',1);pid=re.search(r'EngineCore pid=(\d+)',head)
            value=json.loads(body);result.append(dict(pid=None if pid is None else int(pid[1]),event=value))
        return result
    prepared=events(raw[:end],'BIV150_KONLY_MULTI2_HOIST_PREPARED')
    selected=lambda data:[x for x in events(data,'BIV150_KONLY_ENCODED_SELECTED') if x['event'].get('profile')=='multi2']
    before=selected(raw[:start]);window=selected(raw[start:end]);bindings=[];root=Path(out).resolve()/'konly-multi2-hoist'
    for item in prepared:
        event=item['event'];folder=Path(event['artifact_directory']);need(folder.is_absolute() and folder.resolve().is_relative_to(root),'private fresh binding directory')
        value=json.loads((folder/'binding.json').read_bytes())
        need(value['schema']=='cjj-encoded-multi2-live-binding-v1' and value['artifact_directory']==str(folder),'actual binding identity')
        need(value['override_path']==str(folder/'override.ttgir') and value['native_binding_id']==folder.name,'fresh binding paths')
        need(value['implementation_sha256']==expected['konly_encoded']['implementation_sha256'],'fresh production implementation digest')
        need(value['hoisted']['hash']==event['hoisted_hash'] and value['unhoisted']['hash']==event['unhoisted_hash'],'actual pair hashes')
        need(value['override_sha256']==event['override_sha256']==sha(folder/'override.ttgir'),'actual override bytes')
        need(value['resources']==event['loaded_resources'] and event['actual_native_verified'] is True,'actual resource evidence')
        need(value['numerical_verified'] is value['service_verified'] is False,'fresh binding retains qualification boundary')
        need(value['actual_metadata']['profile']=='multi2' and value['actual_metadata']['constants']['num_seqs']==2,'fresh multi2 identity')
        files={}
        for arm in PAIR_CUBINS:
            snap=value[arm];need(json.loads((folder/arm/'record.json').read_bytes())==snap,'preserved actual snapshot')
            for ext,digest in snap['artifacts'].items():
                path=folder/arm/('actual.jit_python_source.py' if ext=='source' else 'actual.'+ext)
                need(sha(path)==digest,'actual preserved artifact '+arm+'/'+ext);files[str(path.relative_to(out))]=digest
        files[str((folder/'binding.json').relative_to(out))]=sha(folder/'binding.json');files[str((folder/'override.ttgir').relative_to(out))]=sha(folder/'override.ttgir')
        need(item['pid'] is not None and str(item['pid'])==folder.parent.name,'binding worker PID')
        matched=[x for x in window if x['pid']==item['pid'] and x['event']['hash']==value['hoisted']['hash'] and x['event'].get('I8_fragment_hoist_verified') is True and x['event'].get('implementation_sha256')==expected['konly_encoded']['implementation_sha256'] and x['event'].get('sequences')==2]
        bindings.append(dict(pid=item['pid'],binding=value,artifact_sha256=files,post_launch_window_matches=matched,
            cubin_equal_tested_pair={a:value[a]['artifacts']['cubin']==PAIR_CUBINS[a] for a in PAIR_CUBINS},
            fresh_worker_numeric_verified=False))
    hit=any(x['post_launch_window_matches'] for x in bindings)
    return dict(log_byte_start=start,log_byte_end=end,log_window_sha256=hashlib.sha256(raw[start:end]).hexdigest(),
        startup_only_first_selected=before,request_window_first_selected=window,actual_bindings=bindings,
        post_launch_hoisted_multi2_in_request_window=hit,coverage_gaps=[] if hit else ['no_new_verified_multi2_post_launch_marker_in_request_window'],
        no_new_runtime_observer=True,host_first_selection_only=True,Graph_replays_counted=False,
        fresh_worker_numeric_verified=False,model_precision_verified=False,service_throughput_verified=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','compile-review','pair-review','parent-library','mixed-library','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args();need(a.out.is_absolute() and not a.out.exists(),'new absolute receipt output')
    a.out.write_text(json.dumps(prepare(a.source,a.commit,a.compile_review,a.pair_review,a.parent_library,a.mixed_library),sort_keys=True,indent=2)+'\n');print(sha(a.out))
