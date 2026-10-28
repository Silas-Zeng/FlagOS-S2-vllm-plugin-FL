"""PR106 CPU identity adapter around the unchanged 3eb fullstep post-hoc audit."""
import argparse,hashlib,importlib.util,json,runpy,subprocess,sys
from pathlib import Path
PARENT=Path('/private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028')
SOURCE=Path('/private/tmp/flagos-cjj-native-k8-v-overlap-fullstep-frozen-569-20261028')
RAW=Path('/private/tmp/cjj-native-k8-v-overlap-fullstep-v1-raw-20261028')
AUDITOR=Path('/private/tmp/check_cjj_native_k8_fullstep_actual_20261028.py')
COMMIT='569c2e929b51ab639a7b154d826516fa0b4fac33'
LIB='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
CONSUMER='4474281381486c0b813be4d2eee0b045a43858e70a498a8846d9436804fb066a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(s,old,new):
 assert s.count(old)==1,(old,s.count(old));return s.replace(old,new)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--regenerate',action='store_true');a=parser.parse_args()
 assert not a.out.exists()
 assert sha(AUDITOR)=='1c527358d545110a462866c2cc675334e5ea9baf9ccf4a047bec1bcd1f6c060f'
 assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
 assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'])
 transport=json.loads((RAW/'download-manifest.json').read_text());assert transport['raw_files']==192
 owner=transport['original_owner'];assert owner['pid']==14358 and owner['same_start'] is True and owner['state']=='Z' and type(owner['start_ticks']) is int
 # Run the existing original/new four-case consumer array+FP64 replay, mapping
 # its report output to a fresh local file only. Never alter original archives.
 replay=Path('/private/tmp/replay_cjj_native_k8_v_overlap_consumer_actual_20261028.py')
 new_report=a.out.with_name(a.out.stem+'-consumer-replay.json');assert not new_report.exists()
 replay_source=replay.read_text();replay_source=replace(replay_source,"SOURCE=Path('/private/tmp/flagos-cjj-native-k8-v-overlap-20261028')",'SOURCE=Path('+repr(str(SOURCE))+')')
 replay_source=replace(replay_source,"OUT=Path('/private/tmp/cjj-native-k8-v-overlap-consumer-actual-replay-20261028.json')",'OUT=Path('+repr(str(new_report))+')')
 exec(compile(replay_source,str(replay),'exec'),{'__name__':'cjj_existing_consumer_replay','__file__':str(replay)})
 replay_result=json.loads(new_report.read_text())
 sys.path.insert(0,str(SOURCE/'tools'));import probe_biv150_native_k8_v_overlap_fullstep as variant
 summary=json.loads((RAW/'out/fullstep-summary.json').read_text())
 assert summary['K8_library_sha256']==LIB and summary['BF16_library_sha256']=='aace693bdb302d2685ec1b5e11e859c8a4a77d7f45c0f4dacddc6b5e49b015b8'
 assert summary['actual_variant_consumer']==replay_result['new_consumer']
 assert summary['actual_consumer']==replay_result['old_consumer']
 assert summary['actual_consumer']['K8_library_sha256']!=LIB
 assert summary['variant']['compile_admission']==json.loads((Path('/private/tmp/cjj-native-k8-v-overlap-consumer-v1-raw-20261028/out/consumer-summary.json')).read_text())['variant']['compile_admission']
 assert summary['variant']['source_proof']==variant.consumer.factory.verify()
 assert summary['variant']['host_adapter']==variant.adapted_source(PARENT,'fullstep')[1]
 assert summary['variant']['consumer_receipt_adapter']==variant.adapted_source(PARENT,'receipt')[1]
 assert summary['variant']['qualification_inherited_from_old_library'] is False
 text=AUDITOR.read_text()
 # Identity substitutions only: every numerical/cache/guard/Graph/side/resource
 # and timing-arithmetic check below is the existing auditor's code.
 text=replace(text,"dict(pid=706887,start_ticks=347999537,same_start=True,state='Z')",repr(owner))
 text=replace(text,"len(paths)==manifest['raw_files']==192 and total==manifest['raw_bytes']==144847098","len(paths)==manifest['raw_files']==192 and total==manifest['raw_bytes']=="+str(transport['raw_bytes']))
 text=replace(text,"raw/'exports/cjj-native-k8-fullstep-v1-20261028-owner.json'","raw/'exports/cjj-native-k8-v-overlap-fullstep-v1-20261028-owner.json'")
 text=replace(text,"owner['source_commit']==COMMIT and owner['pid']==706887 and owner['start_ticks']==347999537", "owner['source_commit']=="+repr(COMMIT)+" and owner['pid']==14358 and owner['start_ticks']=="+str(owner['start_ticks']))
 text=replace(text,"for key,value in json.loads(json.dumps(f.plan())).items():", "for key,value in json.loads(json.dumps(f.plan())).items():\n        if key=='K8_library_sha256':value="+repr(LIB))
 text=replace(text,'source_commit=COMMIT,source_root=str(source),raw_root=str(raw),','source_commit='+repr(COMMIT)+',parent_fixture_commit=COMMIT,source_root=str(source),raw_root=str(raw),')
 old_argv=sys.argv;sys.argv=[str(AUDITOR),'--raw-root',str(RAW),'--source-root',str(PARENT),'--out',str(a.out)]+(['--regenerate'] if a.regenerate else [])
 try:
  ns={'__name__':'cjj_existing_3eb_posthoc_audit','__file__':str(AUDITOR)};exec(compile(text,str(AUDITOR),'exec'),ns);ns['main']()
 finally:sys.argv=old_argv
 report=json.loads(a.out.read_text());report.update(PR106_identity_adapter_sha256=sha(__file__),original_3eb_posthoc_auditor_sha256=sha(AUDITOR),reused_original_3eb_numerical_checks=True,variant_source_root=str(SOURCE),new_and_old_consumers_full_arrays_replayed=True,new_consumer=summary['actual_variant_consumer'],old_consumer=summary['actual_consumer'],variant_provenance=summary['variant'],baseline_is_old_component_aace_not_current_main_0c66=True,cross_run_22e_comparison_is_not_paired_or_model_gain=True,new_model_service_or_formal_score_verified=False)
 a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(report=str(a.out),sha256=sha(a.out),reused_original_auditor=True,formal_model_gain=False),sort_keys=True))
if __name__=='__main__':main()
