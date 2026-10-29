"""Execute actual PR123 run preflight; stop at first idle, never query a device."""
import argparse,copy,hashlib,importlib.machinery,importlib.util,json,subprocess,sys,tempfile,types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import biv150_multi2_hoist_service_bridge as b
ROOT=Path(__file__).resolve().parents[1]
GQA='0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8'
class IdleReached(Exception):pass

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('helper','gate-library','gqa-library','parent-library','mixed-library','tuning-snapshot','pending-bridge','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args();checks=[];events=[]
    def ck(name,value):assert value,name;checks.append(name)
    ck('clean_frozen_ref',b.git(ROOT,'rev-parse','HEAD').decode().strip()==a.commit and not b.git(ROOT,'status','--porcelain'))
    ck('production_and_old118_unchanged_from_actual_pair_source',not b.git(ROOT,'diff',b.PAIR_SOURCE,a.commit,'--','vllm_fl','csrc',b.LEGACY_TOOL,b.PARENT_CONTROLLER))
    entry=ROOT/b.CONTROLLER;source=entry.read_text();inverse=source
    ck('five_identity_and_two_smoke_only_edits',len(b.EDITS)==7)
    for before,after in reversed(b.EDITS):ck('inverse_'+after,inverse.count(after)==1);inverse=inverse.replace(after,before,1)
    ck('entire_controller_inverse637d',inverse.encode()==(ROOT/b.PARENT_CONTROLLER).read_bytes())
    spec=importlib.util.spec_from_file_location('cpu120_actual_run',entry);controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    for path,digest in [(a.helper,controller.HELPER_SHA),(a.gate_library,controller.NATIVE_SHA),(a.gqa_library,GQA),(a.parent_library,b.PARENT),(a.mixed_library,b.MIXED)]:
        ck('actual_input_'+path.name,b.sha(path)==digest)
    pending=json.loads(a.pending_bridge.read_bytes());ck('actual_new_pending_bridge',pending==b.record(ROOT,a.commit))
    files=[a.helper,a.gate_library,a.gqa_library,a.parent_library,a.mixed_library,a.pending_bridge,entry,ROOT/b.BRIDGE]
    files += sorted(x for x in a.tuning_snapshot.iterdir() if x.is_file())
    before={str(x):b.sha(x) for x in files}
    original_exec=importlib.machinery.SourceFileLoader.exec_module;original_popen=subprocess.Popen
    def stop_idle():events.append('first_real_helper_idle_reached');raise IdleReached()
    def intercepted_exec(loader,module):
        original_exec(loader,module)
        if module.__name__=='encoded_smoke_owned_helpers':
            ck('actual_imported94db',Path(module.__file__).resolve()==a.helper.resolve() and b.sha(module.__file__)==controller.HELPER_SHA)
            module.idle=stop_idle  # Sole helper substitution, before any idle implementation executes.
    def git_only(command,*args,**kwargs):
        if not isinstance(command,list) or command[:3]!=['git','-C',str(ROOT)]:raise AssertionError('non-local-Git subprocess blocked: '+repr(command))
        events.append(command);return original_popen(command,*args,**kwargs)
    with tempfile.TemporaryDirectory(prefix='multi2_hoist-run-preflight-CPU-') as folder:
        folder=Path(folder);bridge=folder/'CPU-only-admission.json';output=folder/'not-created'
        temporary=copy.deepcopy(pending);temporary.update(source_bridge_Root_approved=True,service_admission_authorized=True)
        bridge.write_text(json.dumps(temporary,sort_keys=True)+'\n')
        args=NS(source=ROOT,out=output,helper=a.helper,gate_library=a.gate_library,tuning_snapshot=a.tuning_snapshot,
            native_gqa_library=a.gqa_library,native_gqa_sha256=GQA,native_k8_library=a.parent_library,native_k8_sha256=b.PARENT,
            mixed_native_library=a.mixed_library,mixed_native_sha256=b.MIXED,mixed_source_bridge=bridge,mixed_source_bridge_sha256=b.sha(bridge),
            commit=a.commit,base=b.MAIN,baseline=False,q32_hoist=True,graph_coverage_bootstrap_dir=None,math105=False,supplemental_native_math105_64=False,
            throughput=False,full_workload_warmup=False,warmup_plan='legacy',throughput_case='both')
        with patch.object(importlib.machinery.SourceFileLoader,'exec_module',intercepted_exec),patch.object(subprocess,'Popen',git_only):
            try:controller.run(args)
            except IdleReached:ck('actual_run_reaches_first_idle_after_all_identity_and_snapshot_checks',True)
            else:raise AssertionError('run did not stop at idle')
            ck('no_output_created',not output.exists())
            ck('runtime_read_correct_git_blobs',all(['git','-C',str(ROOT),'show',a.commit+':'+name] in events for name in (b.BRIDGE,b.CONTROLLER)))
            positive_events=copy.deepcopy(events)
            # In-memory fault injection only: files/receipts stay frozen. Each
            # regression executes actual run with just one old Git lookup restored.
            for new,old,error in [("a.commit+':tools/biv150_multi2_hoist_service_bridge.py'","a.commit+':tools/biv150_native118_service_bridge.py'",'Mixed bridge tool differs from frozen source'),
                                  ("a.commit+':tools/run_biv150_multi2_hoist_service.py'","a.commit+':tools/run_biv150_native118_service.py'",'Mixed controller is not frozen source entry')]:
                ck('one_fault_site_'+new,source.count(new)==1)
                fault=types.ModuleType('cpu120_in_memory_old_path');fault.__file__=str(entry)
                exec(compile(source.replace(new,old,1),str(entry),'exec'),fault.__dict__)
                count=events.count('first_real_helper_idle_reached')
                try:fault.run(args)
                except RuntimeError as exc:ck('old_path_rejected_'+old,str(exc)==error)
                else:raise AssertionError('old path accepted')
                ck('old_path_rejected_before_idle_'+old,events.count('first_real_helper_idle_reached')==count)
    ck('real_inputs_unchanged',all(b.sha(path)==digest for path,digest in before.items()))
    ck('no_GPU_framework_import',not {'torch','triton','vllm','numpy'}.intersection(sys.modules))
    ck('frozen_source_still_clean',not b.git(ROOT,'status','--porcelain'))
    report=dict(passed=True,CPU_only=True,source_commit=a.commit,pair_source=b.PAIR_SOURCE,
        checks=checks,check_count=len(checks),controller_sha256=b.sha(entry),bridge_sha256=b.sha(ROOT/b.BRIDGE),script_sha256=b.sha(__file__),
        inputs=before,positive_run_events=positive_events,first_idle_implementation_executed=False,service_executed=False,GPU_executed=False,
        actual_run_preflight_verified=True,Root_approval=False,temporary_flags_only=True,model_precision_verified=False,service_throughput_verified=False,
        limit='Actual frozen run executes helper/protocol/four SO/new bridge/Git object/snapshot checks. Only helper.idle is replaced by a stop. This neither queries CUDA nor imports service packages or proves natural service startup. Two fault-injected in-memory old Git paths are separately refused; frozen production/raw unchanged.')
    assert a.out.is_absolute() and not a.out.exists();a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(b.sha(a.out))
if __name__=='__main__':main()
