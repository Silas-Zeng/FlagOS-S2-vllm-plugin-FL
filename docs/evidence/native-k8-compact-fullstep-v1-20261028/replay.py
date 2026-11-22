"""Explicit CPU replay; defaults to a plan and never creates checkouts or loads a SO."""
import argparse,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCRIPT=HERE/'review_compact_fullstep.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path);p.add_argument('--regenerate',action='store_true')
 defaults={
 'tested-source':'/private/tmp/flagos-cjj-compact-fullstep-frozen-fa9-20261028',
 'parent-source':'/private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028',
 'raw-root':'/private/tmp/cjj-native-k8-compact-fullstep-v1-raw-20261028',
 'compact-consumer-raw':'/private/tmp/cjj-native-k8-compact-consumer-v1-raw-20261028',
 'compile-out':'/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out',
 'codec-summary':'/private/tmp/cjj-native-k8-codec-v1-raw-20261028/out/codec-summary.json',
 'original-library':'/private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so'}
 for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=Path(value))
 a=p.parse_args();assert sha(SCRIPT)=='f0922d1c5dff75619a3ccec0536e0559305756b26d0ff08d5b76c3571f6520f5'
 assert sha(HERE/'original_fullstep_auditor.py')=='1c527358d545110a462866c2cc675334e5ea9baf9ccf4a047bec1bcd1f6c060f'
 mapping={old:str(getattr(a,key.replace('-','_')).resolve()) for key,old in defaults.items()}
 mapping['/private/tmp/check_cjj_native_k8_fullstep_actual_20261028.py']=str(HERE/'original_fullstep_auditor.py')
 text=SCRIPT.read_text()
 for old,new in mapping.items():
  assert text.count(repr(old))==1,old;text=text.replace(repr(old),repr(new))
 if not a.run:
  print(json.dumps(dict(CPU_plan=True,actual_review_executed=False,path_literals_checked=True,requires_full_external_raw_and_dependency_archives=True,metadata_archive_alone_is_insufficient=True,path_map=mapping,qualification_granted=False),sort_keys=True,indent=2));return
 assert a.out is not None and not a.out.exists() and not a.out.with_suffix('.paths.json').exists(),'new output required'
 # Only declared path literals changed; all source, numeric and artifact checks
 # remain in the preserved reviewed script, which validates the exact commits.
 old_argv=sys.argv;sys.argv=[str(SCRIPT),'--out',str(a.out.resolve())]+(['--regenerate'] if a.regenerate else [])
 try:
  ns={'__name__':'shared_compact108_review','__file__':str(SCRIPT)};exec(compile(text,str(SCRIPT)+'::<paths only>','exec'),ns);ns['main']()
 finally:sys.argv=old_argv
 receipt=dict(wrapper_sha256=sha(Path(__file__)),preserved_review_script_sha256=sha(SCRIPT),path_map=mapping,review_output=str(a.out.resolve()),review_output_sha256=sha(a.out),no_original_raw_mutations=True)
 a.out.with_suffix('.paths.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
