"""PR122 exact gather-host/source bridge; old118 approval does not qualify new bytes."""
import hashlib,json,os
from pathlib import Path
import biv150_native118_service_bridge as old
import biv150_hidden_gather_snapshot as snapshot

MAIN='899ce77470b0b539aebc108fd122f668c2d5894d'
COMPONENT='a10145e173e27fd27c798452701b101c8cdcf69d' # unchanged production-math parent
NUMERIC_COMPONENT='73470bbcc28f63cba715a64b94766fbd550301b4'
CONTROLLER='tools/run_biv150_hidden_gather_service.py'
BRIDGE='tools/biv150_hidden_gather_service_bridge.py'
LEGACY='tools/biv150_native118_service_bridge.py'
LEGACY_SHA='e943ea850eae1f1bb2ddcbd0590dc25f23936f182ea38f2e8ca33365226ec176'
PARENT_CONTROLLER='tools/run_biv150_native118_service.py'
PARENT_SHA='637d29e30c86a56e0536548b5678067954c8900046dc2d1a9dd78a96f4fa7199'
ROOT=Path(__file__).resolve().parents[1]
ENV='VLLM_FL_BIV150_DYNAMIC_HIDDEN_GATHER'
DIAGNOSTICS='CJJ_HIDDEN_GATHER_DIAGNOSTICS'
PINS=json.loads((ROOT/'tools/biv150_hidden_gather_sme1_pins.json').read_text())
PRODUCTION=('vllm_fl/compile_semantics.py','vllm_fl/patches/biv150_compute_order.py',snapshot.HELPER,snapshot.KERNEL,snapshot.RUNNER)
TOOLS=(BRIDGE,CONTROLLER,'tools/biv150_hidden_gather_service_observer.py','tools/biv150_hidden_gather_snapshot.py',
       'tools/hidden_gather_bootstrap/sitecustomize.py','tools/biv150_hidden_gather_sme1_pins.json',
       'tools/biv150_hidden_gather_launcher_address.py','tools/biv150_hidden_gather_path_identity.py',
       'tools/biv150_hidden_gather_diagnostic_edits.json','tools/biv150_hidden_gather_controller_edits.json',
       'tools/compile_biv150_dynamic_hidden_gather.py','tools/biv150_runtime_stats_sdk_pins.json')
MIXED=old.MIXED;PARENT=old.PARENT;QUALIFICATION=old.QUALIFICATION
sha=old.sha;git=old.git;fingerprint=old.fingerprint;smoke_window=old.smoke_window
mixed_smoke_requests=old.mixed_smoke_requests;final_evidence=old.final_evidence

def need(ok,why):
    if not ok:raise ValueError('PR122 source bridge: '+why)

def inverse(raw,edits):
    for before,after in reversed(edits):
        need(raw.count(after.encode())==1,'one source inverse seam');raw=raw.replace(after.encode(),before.encode(),1)
    return raw

def current_identity(root,ref):
    root=Path(root).resolve();need(root==ROOT and snapshot.ROOT==ROOT,'same-source tools')
    need(git(root,'rev-parse','HEAD').decode().strip()==ref and not git(root,'status','--porcelain','--untracked-files=all'),'exact clean frozen source')
    git(root,'merge-base','--is-ancestor',COMPONENT,ref);git(root,'merge-base','--is-ancestor',MAIN,ref)
    git(root,'merge-base','--is-ancestor',NUMERIC_COMPONENT,ref)
    need(git(root,'diff','--name-only',MAIN,ref,'--','vllm_fl','csrc').decode().splitlines()==list(PRODUCTION),'only five gather production files')
    for name in PRODUCTION:
        raw=(root/name).read_bytes()
        if name==snapshot.HELPER:
            raw=inverse(raw,json.loads((root/'tools/biv150_hidden_gather_diagnostic_edits.json').read_text()))
        need(raw==git(root,'show',COMPONENT+':'+name),'component bytes plus only diagnostic inverse: '+name)
    need(sha(root/snapshot.KERNEL)==PINS['kernel_source_sha256'],'qualified unchanged GPU source')
    for name,pin in ((LEGACY,LEGACY_SHA),(PARENT_CONTROLLER,PARENT_SHA)):
        need(sha(root/name)==pin and (root/name).read_bytes()==git(root,'show',MAIN+':'+name),'unchanged118 protocol/support')
    need(Path(old.__file__).resolve()==root/LEGACY,'actual118 support import')
    edits=json.loads((root/'tools/biv150_hidden_gather_controller_edits.json').read_text())
    need(inverse((root/CONTROLLER).read_bytes(),edits)==(root/PARENT_CONTROLLER).read_bytes(),'whole controller reverse to637d')
    for name in TOOLS:need((root/name).read_bytes()==git(root,'show',ref+':'+name),'actual frozen tool '+name)
    return dict(source_commit=ref,production=old.production(root,ref),main_dependency=MAIN,component_source=COMPONENT,
        numeric_component_source=NUMERIC_COMPONENT,
        source_sha256={name:sha(root/name) for name in PRODUCTION},tool_sha256={name:sha(root/name) for name in TOOLS},
        original118_bridge_sha256=LEGACY_SHA,parent_controller_sha256=PARENT_SHA,
        only_optional_diagnostic_host_delta_from_component=True,controller_inverse_to118=True,
        accepted_native_sha256=PARENT,accepted_mixed_sha256=MIXED,qualification=QUALIFICATION,
        unchanged_main899_other_production=True,service_verified=False)

def record(root,ref):
    return dict(schema='cjj-hidden-gather-service-source-bridge-v1',**current_identity(root,ref),
        actual_component=PINS,source_bridge_Root_approved=False,service_admission_authorized=False,
        model_precision_verified=False,service_throughput_verified=False,
        limitation='Actual helper component contains constructed Plan/controlled model. New natural runner/C3/ordinary and Plan submissions/model quality remain separate. Diagnostics count host launch returns, never GPU completion.')

def prepare(root,ref,review,summary,parent_library,mixed_library):
    need(sha(review)==PINS['review_sha256'] and sha(summary)==PINS['summary_sha256'],'actual component report/summary SHA')
    r=json.loads(Path(review).read_bytes());s=json.loads(Path(summary).read_bytes())
    need(r['passed'] is True and r['source_commit']==s['source_commit']==PINS['component_source']==NUMERIC_COMPONENT and r['summary_sha256']==PINS['summary_sha256'],'component review source')
    need(s['completed'] is True and s['errors']==[] and s['actual_helper_entry_verified'] is s['actual_helper_cold_verified'] is True,'complete actual helper')
    need(s['service_verified'] is s['Plan_make_plan_or_C3_verified'] is False,'component scope not service')
    need(r['actual_GPU_hash']==PINS['hash'] and r['actual_cubin_sha256']==PINS['cubin'] and r['actual_C_pointer_binding']['DSO_sha256']==PINS['launcher'],'actual compiled and C-address identity')
    need(r['boundaries']['component_numeric_passed'] is True and r['boundaries']['current_numeric_process_C_binding_proven'] is True and
         r['boundaries']['historical_service_C_binding_proven'] is False and s['service_actual_object_numeric_verified'] is True,'new actual numeric scope only')
    need(r['helper_launches']==8 and r['direct_guard_launches']==4 and s['production_warmups_completed']==5,'original complete numeric coverage')
    need(s['service_actual_object']['ABI']==PINS['ABI'] and s['service_actual_object']['options']==PINS['options'] and
         s['service_actual_object']['resources']==PINS['resources'] and s['service_actual_object']['files']==PINS['files'],'exact new SME1 GPU group')
    need(sha(parent_library)==PARENT and sha(mixed_library)==MIXED,'same accepted actual libraries')
    return record(root,ref)

def validate(root,ref,path,expected_sha,library):
    need(Path(path).is_absolute() and sha(path)==expected_sha,'explicit new Root receipt SHA')
    actual=json.loads(Path(path).read_bytes());expected=record(root,ref)
    need(actual.get('source_bridge_Root_approved') is True and actual.get('service_admission_authorized') is True,'Root service approval pending')
    expected.update(source_bridge_Root_approved=True,service_admission_authorized=True)
    need(actual==expected,'new exact source/component bridge; no prior118/120 receipt')
    need(Path(library).is_absolute() and sha(library)==MIXED,'actual mixed library');return actual

def clear_environment(original):
    env=old.clear_environment(original)
    for key in (ENV,DIAGNOSTICS,snapshot.ENABLE,snapshot.TOKEN,snapshot.RUN_ID,
                'CJJ_RUNTIME_STATS_SNAPSHOT','CJJ_RUNTIME_STATS_TOKEN','CJJ_RUNTIME_STATS_RUN_ID'):
        env.pop(key,None)
    paths={ROOT/'tools/hidden_gather_bootstrap',ROOT/'tools/runtime_stats_bootstrap'}
    if 'PYTHONPATH' in env:env['PYTHONPATH']=os.pathsep.join(x for x in env['PYTHONPATH'].split(os.pathsep) if not x or Path(x).resolve() not in paths)
    return env

def service_environment(original,library):
    env=old.service_environment(clear_environment(original),library);env[ENV]='1';return env

def options(a):
    old.options(a)
    need(a.q32_hoist is True and a.native_gqa_library is not None and a.native_gqa_library.is_absolute() and
         a.native_gqa_sha256=='0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8','fixed Q32/0c66 configuration')
    diagnostic=getattr(a,'gather_diagnostics',False);plain=not(a.math105 or a.supplemental_native_math105_64 or a.throughput)
    need(type(diagnostic) is bool and diagnostic==plain,'plain smoke requires diagnostics; Math/performance must disable it')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','component-review','component-summary','parent-library','mixed-library','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args()
    need(a.out.is_absolute() and not a.out.exists(),'new absolute receipt output')
    a.out.write_text(json.dumps(prepare(a.source,a.commit,a.component_review,a.component_summary,a.parent_library,a.mixed_library),sort_keys=True,indent=2)+'\n');print(sha(a.out))
