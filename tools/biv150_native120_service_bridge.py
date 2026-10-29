"""PR120 host-only service admission. New source needs its own Root receipt."""
import hashlib,json
from pathlib import Path
import subprocess
import biv150_native118_service_bridge as old

MAIN='899ce77470b0b539aebc108fd122f668c2d5894d'
COMPONENT_SOURCE='071232aa3641a378fc709d21b5e5c4b6c90a6e7f'
COMPONENT_REVIEW='bd03aa5f2dd7d805eb87929a69707329a7a261361c33af054671e7fb5bbea591'
COMPONENT_SUMMARY='7f1339b615ee8043d893a765ff9e24c502a19128b53766aa42cdf20b55298351'
COMPONENT_REVIEW_SCRIPT='694ce9ad685e13d2fbda38b12012d016acf3b6c867bd4afdc80933005c4e8ed2'
MODULE='vllm_fl/patches/biv150_konly_encoded_native.py'
MANIFEST='vllm_fl/patches/biv150_konly_encoded_sources.json'
BASE_NATIVE='4a2f387d835dc9addcf38a88714bbb01d8c0e526a9b196c946ae616c4749df89'
CANDIDATE_NATIVE='ce5feb765a635ded45924bfafce109a6b6ccfe04cc00495599b1c5170fe039b7'
BASE_MANIFEST='5d7e4b28022f7a0d1cc5a1ef881d12da05a41c6c503ab0a38e0a39cd14230878'
CANDIDATE_MANIFEST='749c7ba477cb1445458ee8c6806c8432dc3804026cb511845ff3ae1e0baac56e'
LEGACY_TOOL='tools/biv150_native118_service_bridge.py'
LEGACY_SHA='e943ea850eae1f1bb2ddcbd0590dc25f23936f182ea38f2e8ca33365226ec176'
PARENT_CONTROLLER='tools/run_biv150_native118_service.py'
PARENT_CONTROLLER_SHA='637d29e30c86a56e0536548b5678067954c8900046dc2d1a9dd78a96f4fa7199'
CONTROLLER='tools/run_biv150_native120_service.py'
BRIDGE='tools/biv150_native120_service_bridge.py'
EDITS=[('import biv150_native118_service_bridge as mixed_bridge','import biv150_native120_service_bridge as mixed_bridge'),
 ("tool=source/'tools/biv150_native118_service_bridge.py'","tool=source/'tools/biv150_native120_service_bridge.py'"),
 ("entry=source/'tools/run_biv150_native118_service.py'","entry=source/'tools/run_biv150_native120_service.py'"),
 ("a.commit+':tools/biv150_native118_service_bridge.py'","a.commit+':tools/biv150_native120_service_bridge.py'"),
 ("a.commit+':tools/run_biv150_native118_service.py'","a.commit+':tools/run_biv150_native120_service.py'")]
# These unchanged118 functions keep their original module globals. No rebinding
# of old.current_identity/prepare/validate and no reuse of an old Root receipt.
MIXED=old.MIXED;PARENT=old.PARENT;QUALIFICATION=old.QUALIFICATION
clear_environment=old.clear_environment;service_environment=old.service_environment
options=old.options;fingerprint=old.fingerprint;smoke_window=old.smoke_window
mixed_smoke_requests=old.mixed_smoke_requests;final_evidence=old.final_evidence
sha=old.sha;git=old.git

def need(ok,message):
    if not ok:raise ValueError('PR120 host service bridge: '+message)

def restore(source):
    changes=[('    backing=storage(t);_,start,end=backing;pointer=int(t.data_ptr());offset=int(t.storage_offset())','    _,start,end=storage(t);pointer=int(t.data_ptr());offset=int(t.storage_offset())'),
      ('    return backing\n',''),
      ('        # Only this call: tensors keeps every object alive; retain all bounds\n        # checks in order and reuse their observed whole-storage intervals.\n        backing={id(t):bounds(t) for t in tensors}\n        def observed_overlap(a,b):\n            _,x,y=backing[id(a)];_,u,v=backing[id(b)];return x<v and u<y','        for t in tensors:bounds(t)'),
      ('observed_overlap(src,k)','overlaps(src,k)'),('observed_overlap(out,src)','overlaps(out,src)'),('observed_overlap(out,other)','overlaps(out,other)')]
    for new,previous in changes:need(source.count(new)==1,'native inverse count');source=source.replace(new,previous,1)
    return source

def current_identity(root,ref):
    root=Path(root).resolve()
    need(git(root,'rev-parse','HEAD').decode().strip()==ref and not git(root,'status','--porcelain','--untracked-files=all'),'clean exact frozen source')
    git(root,'merge-base','--is-ancestor',MAIN,ref)
    need(git(root,'diff','--name-only',MAIN,ref,'--','vllm_fl','csrc').decode().splitlines()==[MODULE,MANIFEST],'only two exact host/manifest changes')
    for name,base_hash,new_hash in ((MODULE,BASE_NATIVE,CANDIDATE_NATIVE),(MANIFEST,BASE_MANIFEST,CANDIDATE_MANIFEST)):
        need(hashlib.sha256(git(root,'show',MAIN+':'+name)).hexdigest()==base_hash and sha(root/name)==new_hash,'exact main899/candidate bytes')
        need((root/name).read_bytes()==git(root,'show',COMPONENT_SOURCE+':'+name),'same actual component production bytes')
    need(restore((root/MODULE).read_text()).encode()==git(root,'show',MAIN+':'+MODULE),'exact native source inverse')
    manifest=(root/MANIFEST).read_text();need(manifest.count(CANDIDATE_NATIVE)==1,'one manifest digest')
    need(manifest.replace(CANDIDATE_NATIVE,BASE_NATIVE).encode()==git(root,'show',MAIN+':'+MANIFEST),'exact manifest inverse')
    need(Path(old.__file__).resolve()==root/LEGACY_TOOL and sha(root/LEGACY_TOOL)==LEGACY_SHA,'fixed imported118 support module')
    for name in (LEGACY_TOOL,PARENT_CONTROLLER):need((root/name).read_bytes()==git(root,'show',MAIN+':'+name),'unchanged118 support/protocol source')
    need(sha(root/PARENT_CONTROLLER)==PARENT_CONTROLLER_SHA,'parent controller identity')
    inverse=(root/CONTROLLER).read_text()
    for before,after in reversed(EDITS):need(inverse.count(after)==1,'controller identity seam count');inverse=inverse.replace(after,before,1)
    need(inverse.encode()==(root/PARENT_CONTROLLER).read_bytes(),'controller inverse changes protocol')
    for name in (BRIDGE,CONTROLLER):need((root/name).read_bytes()==git(root,'show',ref+':'+name),'new tracked bridge/controller source')
    qualification=old.qualification(root)
    need((root/old.MODULE).read_bytes()==git(root,'show',MAIN+':'+old.MODULE),'accepted mixed source unchanged')
    return dict(source_commit=ref,accepted_main=MAIN,production=old.production(root,ref),qualification=qualification,
        native_library_sha256=PARENT,mixed_library_sha256=MIXED,native_adapter_sha256=CANDIDATE_NATIVE,
        source_manifest_sha256=CANDIDATE_MANIFEST,original118_bridge_sha256=LEGACY_SHA,
        parent_controller_sha256=PARENT_CONTROLLER_SHA,controller_sha256=sha(root/CONTROLLER),bridge_tool_sha256=sha(root/BRIDGE),
        native_inverse_to_main899=True,manifest_inverse_to_main899=True,controller_inverse_to118=True,
        storage_specific_environment_switch=False,service_verified=False)

def record(root,ref):
    return dict(schema='cjj-native120-host-validation-service-source-bridge-v1',**current_identity(root,ref),
        host_component=dict(source_commit=COMPONENT_SOURCE,summary_sha256=COMPONENT_SUMMARY,review_sha256=COMPONENT_REVIEW,
          review_script_sha256=COMPONENT_REVIEW_SCRIPT,owner_pid=180272,owner_start_ticks=363320134,
          timing_groups=8,samples=96,real_CUDA_Tensor_host_validate_only=True),
        unchanged_main899_math_and_libraries=True,source_bridge_Root_approved=False,service_admission_authorized=False,
        model_precision_verified=False,service_throughput_verified=False,
        limitation='New host validation bytes require this new Root receipt and natural service validation; component host timing is not model performance. No independent storage env; compare main899 production with the exact071 production using Native5898+mixedebbc ON in both arms.')

def prepare(root,ref,component_review,component_summary,parent_library,mixed_library):
    need(sha(component_review)==COMPONENT_REVIEW and sha(component_summary)==COMPONENT_SUMMARY,'actual host component original receipts')
    r=json.loads(Path(component_review).read_bytes());s=json.loads(Path(component_summary).read_bytes())
    need(r['passed'] is True and r['source_commit']==s['source_commit']==COMPONENT_SOURCE,'component source/pass')
    need(r['summary']['sha256']==COMPONENT_SUMMARY and r['complete_samples']==96 and r['timing_groups']==8,'full samples binding')
    need(s['completed'] is True and s['actual_behavior_verified'] is True and s['errors']==[],'actual paired behavior')
    need(s['C3_service_verified'] is s['model_precision_verified'] is s['performance_verified'] is False,'component does not qualify service')
    need(sha(parent_library)==PARENT and sha(mixed_library)==MIXED,'same two actual libraries')
    return record(root,ref)

def validate(root,ref,path,expected_sha,library):
    need(Path(path).is_absolute() and sha(path)==expected_sha,'explicit new Root bridge SHA')
    value=json.loads(Path(path).read_bytes());expected=record(root,ref)
    need(value.get('source_bridge_Root_approved') is True and value.get('service_admission_authorized') is True,'Root service approval pending')
    expected.update(source_bridge_Root_approved=True,service_admission_authorized=True)
    need(value==expected,'exact new host/source/component bridge; old118 receipt is ineligible')
    need(Path(library).is_absolute() and sha(library)==MIXED,'actual fixed mixed library')
    return value

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','component-review','component-summary','parent-library','mixed-library','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args()
    need(a.out.is_absolute() and not a.out.exists(),'new absolute receipt output')
    a.out.write_text(json.dumps(prepare(a.source,a.commit,a.component_review,a.component_summary,a.parent_library,a.mixed_library),sort_keys=True,indent=2)+'\n');print(sha(a.out))
