"""Frozen PR120 source bridge and original protocol/C3 checks; no GPU/service."""
import argparse,ast,contextlib,copy,hashlib,importlib.util,io,json,subprocess,sys,tarfile,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import biv150_native120_service_bridge as b
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-library','mixed-library','helper','component-review','component-summary','old-bridge','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--production-ref',required=True);a=p.parse_args();checks=[]
    def ck(name,v):assert v,name;checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,AssertionError):checks.append(name)
        else:raise AssertionError('accepted '+name)
    before={path:b.sha(path) for path in (a.parent_library,a.mixed_library,a.helper,a.component_review,a.component_summary,a.old_bridge)}
    record=b.prepare(ROOT,a.production_ref,a.component_review,a.component_summary,a.parent_library,a.mixed_library)
    ck('exact_new_source_record',record['source_commit']==a.production_ref and record['accepted_main']==b.MAIN and record['source_bridge_Root_approved'] is record['service_admission_authorized'] is False)
    ck('no_storage_switch_added',record['storage_specific_environment_switch'] is False and b.git(ROOT,'diff',b.COMPONENT_SOURCE,a.production_ref,'--','vllm_fl','csrc')==b'')
    source=(ROOT/b.CONTROLLER).read_text();parent=(ROOT/b.PARENT_CONTROLLER).read_text();inverse=source
    for previous,new in reversed(b.EDITS):ck('one_controller_seam:'+new,inverse.count(new)==1);inverse=inverse.replace(new,previous)
    ck('entire_controller_inverse_exact',inverse==parent and b.sha(ROOT/b.PARENT_CONTROLLER)==b.PARENT_CONTROLLER_SHA)
    candidate=load('cpu120_controller',ROOT/b.CONTROLLER);baseline=load('cpu118_controller',ROOT/b.PARENT_CONTROLLER)
    ck('exact_e236_protocol',candidate.THROUGHPUT_FUNCTION_SHA==baseline.THROUGHPUT_FUNCTION_SHA=='e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d')
    # The full inverse covers Math105, original67+plain96, warmup, CLI, Graph,
    # request parsing and cleanup. Only bridge/entry names differ.
    ck('same_existing_semantic_functions',all(getattr(b,key) is getattr(b.old,key) for key in ['options','clear_environment','service_environment','fingerprint','smoke_window','mixed_smoke_requests','final_evidence']))
    ck('same_libraries',b.sha(a.parent_library)==b.PARENT and b.sha(a.mixed_library)==b.MIXED)
    with tempfile.TemporaryDirectory(prefix='pr120-bridge-CPU-') as folder:
        folder=Path(folder);path=folder/'new-bridge.json'
        def write(v):path.write_text(json.dumps(v,sort_keys=True)+'\n');return b.sha(path)
        h=write(record);reject('pending_new_bridge_rejected',lambda:b.validate(ROOT,a.production_ref,path,h,a.mixed_library))
        # Ephemeral flags only exercise CPU admission. No approved artifact leaves this directory.
        approved=copy.deepcopy(record);approved.update(source_bridge_Root_approved=True,service_admission_authorized=True)
        h=write(approved);ck('positive_new_bridge_CPU_only',b.validate(ROOT,a.production_ref,path,h,a.mixed_library)==approved)
        for key,value in [('source_bridge_Root_approved',False),('service_admission_authorized',False),('source_commit',b.MAIN),('production',{}),('host_component',{}),('native_adapter_sha256',b.BASE_NATIVE),('source_manifest_sha256',b.BASE_MANIFEST),('native_library_sha256','0'*64),('mixed_library_sha256',b.PARENT),('qualification',{}),('model_precision_verified',True),('service_throughput_verified',True)]:
            altered=copy.deepcopy(approved);altered[key]=value;h=write(altered);reject('reject_bridge_'+key,lambda h=h:b.validate(ROOT,a.production_ref,path,h,a.mixed_library))
        h=write(approved);reject('wrong_bridge_SHA',lambda:b.validate(ROOT,a.production_ref,path,'0'*64,a.mixed_library))
        reject('wrong_mixed_library_bytes',lambda:b.validate(ROOT,a.production_ref,path,h,a.parent_library))
        reject('actual_old_c14_bridge_ineligible',lambda:b.validate(ROOT,a.production_ref,a.old_bridge,b.sha(a.old_bridge),a.mixed_library))
        reject('old118_current_identity_correctly_refuses_candidate',lambda:b.old.current_identity(ROOT,a.production_ref))
        env=candidate.native_env({candidate.ENABLE:'1',candidate.HOIST_ENABLE:'1'},a.parent_library);env=b.service_environment(env,a.mixed_library)
        ck('both_Native_and_mixed_ON',env[candidate.K8_ENABLE]=='1' and env[b.old.ENV]=='1' and env[candidate.HOIST_ENABLE]=='1')
        off=candidate.native_env(dict(env,**{candidate.ENABLE:'0',candidate.HOIST_ENABLE:'0'}))
        ck('diagnostic_OFF_clears_native_and_mixed',off[candidate.K8_ENABLE]=='0' and all(k not in off for k in [b.old.ENV,b.old.PATH_ENV,b.old.SHA_ENV]))
        opts=NS(baseline=False,native_k8_library=a.parent_library,native_k8_sha256=b.PARENT,graph_coverage_bootstrap_dir=None,mixed_native_library=a.mixed_library,mixed_native_sha256=b.MIXED,mixed_source_bridge=path,mixed_source_bridge_sha256=h)
        b.options(opts);ck('candidate_options_unchanged',True)
        bad=copy.copy(opts);bad.baseline=True;reject('baseline_switch_not_armA',lambda:b.options(bad))
        helper=load('cpu120_helper94db',a.helper);ck('helper94db_unchanged',b.sha(a.helper)==candidate.HELPER_SHA)
        # Isolated Native5898+mixedebbc CPU fingerprint. Other native libraries
        # are disabled here only; real controller keeps original full service env.
        env.update(VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0')
        off.update(VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0')
        actual_b=helper.expected_fingerprint(ROOT,env,{'torch':'2.10.0','cxx11_abi':False});b.fingerprint(actual_b)
        off_b=helper.expected_fingerprint(ROOT,off,{'torch':'2.10.0','cxx11_abi':False});ck('OFF_C3_has_no_encoded','konly_encoded' not in off_b)
        archive=b.git(ROOT,'archive',b.MAIN,'vllm_fl','csrc');shadow=folder/'main899';shadow.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(shadow,filter='data')
        actual_a=helper.expected_fingerprint(shadow,env,{'torch':'2.10.0','cxx11_abi':False});b.fingerprint(actual_a)
        ck('actual_C3_A_B_differ',actual_a['sha256']!=actual_b['sha256'] and actual_a['konly_encoded']['implementation_sha256']!=actual_b['konly_encoded']['implementation_sha256'])
        stripped=[]
        for value in (actual_a,actual_b):
            v=copy.deepcopy(value);v.pop('sha256');v['konly_encoded'].pop('implementation_sha256');stripped.append(v)
        ck('same_C3_features_libraries_and_qualification',stripped[0]==stripped[1])
        reject('wrong5898_fingerprint_rejected',lambda:helper.expected_fingerprint(ROOT,{**env,candidate.K8_HASH:'0'*64},{'torch':'2.10.0','cxx11_abi':False}))
        reject('wrong_mixed_fingerprint_rejected',lambda:helper.expected_fingerprint(ROOT,{**env,b.old.SHA_ENV:'0'*64},{'torch':'2.10.0','cxx11_abi':False}))
        with patch.object(sys,'argv',[str(ROOT/b.CONTROLLER)]),contextlib.redirect_stdout(io.StringIO()) as out:candidate.main()
        plan=json.loads(out.getvalue());ck('default_controller_no_service_or_GPU',plan['cpu_plan'] is True and plan['gpu_executed'] is False and plan['formal_throughput'] is False)
    ck('inputs_unchanged',all(b.sha(path)==digest for path,digest in before.items()))
    ck('no_GPU_framework_imported',not {'torch','triton','vllm','numpy'}.intersection(sys.modules))
    report=dict(passed=True,CPU_only=True,source_commit=a.production_ref,checks=checks,check_count=len(checks),new_bridge_pending=True,controller_inverse_exact=True,
        parent_controller_sha256=b.PARENT_CONTROLLER_SHA,controller_sha256=b.sha(ROOT/b.CONTROLLER),bridge_sha256=b.sha(ROOT/b.BRIDGE),CPU_script_sha256=b.sha(Path(__file__)),
        component_review_sha256=b.COMPONENT_REVIEW,component_summary_sha256=b.COMPONENT_SUMMARY,baseline=b.MAIN,production_candidate=b.COMPONENT_SOURCE,
        C3=dict(A_main899=actual_a,B_candidate=actual_b,B_diagnostic_OFF=off_b,scope='CPU Native5898+mixedebbc only; W8A8/GQA disabled in this fixture. Real service controller retains full5c/0c66 and derives its full fingerprint normally.'),
        native_math_changed=False,storage_switch_added=False,Root_approval=False,service_executed=False,model_precision_verified=False,service_throughput_verified=False)
    assert not a.out.exists();a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(b.sha(a.out))
if __name__=='__main__':main()
