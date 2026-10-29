"""Source/env/receipt and bounded two-request smoke checks; no real HTTP/GPU."""
import argparse,ast,contextlib,copy,importlib.util,io,json,runpy,shutil,sys,tempfile,threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import biv150_multi2_hoist_service_bridge as b
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-library','mixed-library','helper','compile-review','pair-review','actual-binding','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args();checks=[]
    def ck(name,v):assert v,name;checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,AssertionError):checks.append(name)
        else:raise AssertionError('accepted '+name)
    record=b.prepare(ROOT,a.commit,a.compile_review,a.pair_review,a.parent_library,a.mixed_library)
    ck('exact_production9b_fe2c',record['production_equal_actual9b_and_fe2c'] and not b.git(ROOT,'diff',b.PAIR_SOURCE,a.commit,'--','vllm_fl','csrc'))
    source=(ROOT/b.CONTROLLER).read_text();inverse=source
    for old,new in reversed(b.EDITS):ck('one_seam_'+old,inverse.count(new)==1);inverse=inverse.replace(new,old,1)
    ck('complete_controller_inverse637d',inverse.encode()==(ROOT/b.PARENT_CONTROLLER).read_bytes())
    c=load('cpu123_service_controller',ROOT/b.CONTROLLER);helper=load('cpu123_service_helper',a.helper)
    ck('same94db_and_e236',b.sha(a.helper)==c.HELPER_SHA and c.THROUGHPUT_FUNCTION_SHA=='e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d')
    ck('unchanged_helpers_and_119',all(getattr(b,n) is getattr(b.old,n) for n in ('smoke_window','mixed_smoke_requests','final_evidence')) and not b.git(ROOT,'diff',b.PAIR_SOURCE,a.commit,'--','tools/biv150_runtime_stats_snapshot.py','tools/runtime_stats_bootstrap'))
    with tempfile.TemporaryDirectory(prefix='multi2-service-cpu-') as folder:
        folder=Path(folder);receipt=folder/'receipt.json'
        def write(v):receipt.write_text(json.dumps(v,sort_keys=True)+'\n');return b.sha(receipt)
        h=write(record);reject('pending_receipt_rejected',lambda:b.validate(ROOT,a.commit,receipt,h,a.mixed_library))
        approved=copy.deepcopy(record);approved.update(source_bridge_Root_approved=True,service_admission_authorized=True)
        h=write(approved);ck('temporary_CPU_approval_only',b.validate(ROOT,a.commit,receipt,h,a.mixed_library)==approved)
        for key,value in [('source_bridge_Root_approved',False),('service_admission_authorized',False),('source_commit',b.MAIN),('production',{}),('component',{}),('native_library_sha256',b.MIXED),('mixed_library_sha256',b.PARENT),('qualification',{}),('new_worker_numeric_verified',True),('model_precision_verified',True),('service_throughput_verified',True)]:
            x=copy.deepcopy(approved);x[key]=value;h=write(x);reject('reject_'+key,lambda:b.validate(ROOT,a.commit,receipt,h,a.mixed_library))
        h=write(approved);reject('wrong_receipt_SHA',lambda:b.validate(ROOT,a.commit,receipt,'0'*64,a.mixed_library))
        reject('wrong_mixed_library',lambda:b.validate(ROOT,a.commit,receipt,h,a.parent_library))
        reject('legacy118_admission_refuses_new_source',lambda:b.old.current_identity(ROOT,a.commit))
        env=c.native_env({c.ENABLE:'1',c.HOIST_ENABLE:'1',b.ENV:'polluted'},a.parent_library)
        ck('inherited_switch_cleared',b.ENV not in env)
        env=b.service_environment(env,a.mixed_library);ck('new_and_old_features_ON',env[b.ENV]==env[c.K8_ENABLE]==env[c.HOIST_ENABLE]==env[b.old.ENV]=='1')
        off=c.native_env(dict(env,**{c.ENABLE:'0',c.HOIST_ENABLE:'0'}));ck('OFF_clears_multi2_and_mixed',b.ENV not in off and b.old.ENV not in off and off[c.K8_ENABLE]=='0')
        opts=NS(baseline=False,native_k8_library=a.parent_library,native_k8_sha256=b.PARENT,graph_coverage_bootstrap_dir=None,mixed_native_library=a.mixed_library,mixed_native_sha256=b.MIXED,mixed_source_bridge=receipt,mixed_source_bridge_sha256=h,q32_hoist=True)
        b.options(opts);ck('fixed_options',True);bad=copy.copy(opts);bad.q32_hoist=False;reject('Q32_OFF_rejected',lambda:b.options(bad))
        env.update(VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0');off.update(VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0')
        on=helper.expected_fingerprint(ROOT,env,{'torch':'2.10.0','cxx11_abi':False});b.fingerprint(on)
        off_fp=helper.expected_fingerprint(ROOT,off,{'torch':'2.10.0','cxx11_abi':False});ck('actual_OFF_no_encoded','konly_encoded' not in off_fp)
        parent=helper.expected_fingerprint(ROOT,{**env,b.ENV:'0'},{'torch':'2.10.0','cxx11_abi':False});reject('multi2_OFF_fingerprint_refused',lambda:b.fingerprint(parent))
        x,y=copy.deepcopy(on),copy.deepcopy(parent);x.pop('sha256');y.pop('sha256');x['konly_encoded'].pop('multi2_hoist_enabled');ck('ON_only_adds_multi2_feature',x==y)
        # Exercise the actual added if block, not a duplicate condition.
        run=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run')
        guard=next(n for n in ast.walk(run) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=='multi2_pair_smoke' for x in ast.walk(n)))
        calls=[];fake=NS(pair_requests=lambda:calls.append('pair') or {})
        block=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'smoke-only-actual-block','exec')
        for name in ('plain','math105','supplemental_native_math105_64','throughput'):
            flags=NS(math105=False,supplemental_native_math105_64=False,throughput=False)
            if name!='plain':setattr(flags,name,True)
            before=len(calls);exec(block,dict(a=flags,save=lambda _:None,mixed_bridge=fake,m=NS(atomic=lambda *_:None),out=folder,owned=NS(remember=lambda:None)))
            ck('pair_request_guard_'+name,len(calls)-before==(1 if name=='plain' else 0))
        # Real two threads and synchronization, fake transport only. Both responses
        # are closed on success/JSON error; one transport failure cannot orphan the other.
        for failure in ('none','json','transport'):
            rendezvous=threading.Barrier(2);opened=[];closed=[];requests=[];lock=threading.Lock()
            class Response:
                def __init__(self,i):self.i=i;self.n=0
                def __enter__(self):return self
                def __exit__(self,*_):closed.append(self.i)
                def read1(self,_):
                    self.n+=1
                    if self.n>1:return b''
                    return b'bad json' if failure=='json' and self.i==0 else json.dumps(dict(choices=[{}],usage=dict(completion_tokens=16,prompt_tokens=3017))).encode()
            def transport(req,timeout):
                with lock:i=len(requests);requests.append((req,timeout))
                rendezvous.wait(timeout=2)
                if failure=='transport' and i==0:raise TimeoutError('CPU injected')
                opened.append(i);return Response(i)
            with patch.object(b.urllib.request,'urlopen',transport):result=b.pair_requests()
            ck('concurrent_two_requests_'+failure,len(requests)==2 and all(0<x[1]<=180 for x in requests))
            ck('response_cleanup_'+failure,sorted(opened)==sorted(closed))
            ck('preserved_error_result_'+failure,result['completed']==(2 if failure=='none' else 1) and bool(result['coverage_gaps'])==(failure!='none'))
            ck('IDs_and_payload_'+failure,len({x[0].get_header('X-request-id') for x in requests})==2 and all(json.loads(x[0].data)['max_tokens']==16 for x in requests))
            ck('distinct_first_content_and_equal_digit_length_'+failure,{json.loads(x[0].data)['messages'][0]['content'] for x in requests}=={cycle*300+'请回答最后一个数。' for cycle in ('1 2 3 4 5 ','6 7 8 9 0 ')})
        # Actual compiled artifacts copied into a synthetic local evidence path;
        # only path fields are relocated. This does not claim a service execution.
        binding=json.loads(a.actual_binding.read_bytes());dest=folder/'konly-multi2-hoist'/'205019'/binding['native_binding_id'];shutil.copytree(a.actual_binding.parent,dest)
        binding.update(artifact_directory=str(dest),override_path=str(dest/'override.ttgir'));(dest/'binding.json').write_text(json.dumps(binding))
        need_impl=on['konly_encoded']['implementation_sha256'];ck('tested_binding_same_production_digest',binding['implementation_sha256']==need_impl)
        prep=dict(unhoisted_hash=binding['unhoisted']['hash'],hoisted_hash=binding['hoisted']['hash'],override_sha256=binding['override_sha256'],artifact_directory=str(dest),actual_native_verified=True,loaded_resources=binding['resources'])
        selected=dict(profile='multi2',hash=binding['hoisted']['hash'],I8_fragment_hoist_verified=True,implementation_sha256=need_impl,sequences=2)
        p_line='(EngineCore pid=205019) BIV150_KONLY_MULTI2_HOIST_PREPARED '+json.dumps(prep)+'\n'
        s_line='(EngineCore pid=205019) BIV150_KONLY_ENCODED_SELECTED '+json.dumps(selected)+'\n';log=folder/'server.log';log.write_text(p_line+s_line)
        observed=b.window_evidence(log,len(p_line.encode()),log.stat().st_size,folder,on);ck('fresh_binding_and_postlaunch_match',observed['post_launch_hoisted_multi2_in_request_window'] is True and observed['fresh_worker_numeric_verified'] is False)
        pre=b.window_evidence(log,log.stat().st_size,log.stat().st_size,folder,on);ck('startup_only_is_coverage_gap',not pre['post_launch_hoisted_multi2_in_request_window'] and len(pre['startup_only_first_selected'])==1)
        log.write_text('ready\n');gap=b.window_evidence(log,0,log.stat().st_size,folder,on);ck('no_multi2_is_coverage_gap_not_operator_failure',not gap['post_launch_hoisted_multi2_in_request_window'] and gap['coverage_gaps'])
        log.write_text(p_line+s_line);artifact=dest/'hoisted/actual.cubin';original=artifact.read_bytes();artifact.write_bytes(original+b'x');reject('changed_fresh_artifact_rejected',lambda:b.window_evidence(log,0,log.stat().st_size,folder,on));artifact.write_bytes(original)
        with patch.object(sys,'argv',[str(ROOT/b.CONTROLLER)]),contextlib.redirect_stdout(io.StringIO()) as buf:runpy.run_path(str(ROOT/b.CONTROLLER),run_name='__main__')
        plan=json.loads(buf.getvalue());ck('default_plan_no_service',plan['cpu_plan'] is True and plan['gpu_executed'] is False)
    ck('no_GPU_framework_import',not {'torch','triton','vllm','numpy'}.intersection(sys.modules))
    report=dict(passed=True,CPU_only=True,checks=checks,check_count=len(checks),source_commit=a.commit,script_sha256=b.sha(__file__),bridge_sha256=b.sha(ROOT/b.BRIDGE),controller_sha256=b.sha(ROOT/b.CONTROLLER),controller_inverse_exact=True,production_unchanged_from_actual_pair=True,C3_ON=on,C3_OFF=off_fp,Root_approval=False,new_worker_numeric_verified=False,service_executed=False,model_precision_verified=False,service_throughput_verified=False)
    assert not a.out.exists();a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(b.sha(a.out))
if __name__=='__main__':main()
