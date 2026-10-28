#!/usr/bin/env python3
"""Reuse PR108's complete array/cache/writer/ABBA audit; PR110 identities only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'native-k8-compact-fullstep-v1-20261028'
SCRIPT=OLD/'review_compact_fullstep.py'
AUDIT=OLD/'original_fullstep_auditor.py'
SOURCE='4d1166792a5247b774d6b1977e60466574e6a34c'
LIBRARY='7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4'
CONSUMER='6f719fe84e83e547d0e42f7c2be5bdb408e64cdb107cc455994a681705e34c7c'
REVIEW='90f6e18e142813e732694f453779b876048daaa8c122fd3e2d985dc2969cdb17'
MANIFEST='944578c93b8c5db2bed3fa7d1285c5bb5d7132259f9c99eecd52057defa49962'
SUMMARY='f12332f0b2397ed17baaadbad5e5d189c326188e1b6b9a05c8d4084ef7667478'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--regenerate',action='store_true');p.add_argument('--out',type=Path)
    defaults={'tested-source':'/private/tmp/flagos-cjj-compact-native-v-overlap-validation-20261029',
        'parent-source':'/private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028',
        'raw-root':'/private/tmp/cjj-native110-fullstep-v1-raw-20261029',
        'compact-consumer-raw':'/Users/chenjunjie/Documents/flagos/artifacts/cjj-native110-consumer-v1-raw-20261029',
        'compile-out':'/private/tmp/cjj-native110-compile-v1-raw-20261029/out',
        'codec-summary':'/private/tmp/cjj-native-k8-codec-v1-raw-20261028/out/codec-summary.json',
        'original-library':'/private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so'}
    for name,value in defaults.items():p.add_argument('--'+name,type=Path,default=Path(value))
    a=p.parse_args();original=SCRIPT.read_text()
    assert sha(SCRIPT)=='f0922d1c5dff75619a3ccec0536e0559305756b26d0ff08d5b76c3571f6520f5'
    assert sha(AUDIT)=='1c527358d545110a462866c2cc675334e5ea9baf9ccf4a047bec1bcd1f6c060f'
    path_defaults={'tested-source':'/private/tmp/flagos-cjj-compact-fullstep-frozen-fa9-20261028',
        'parent-source':defaults['parent-source'],'raw-root':'/private/tmp/cjj-native-k8-compact-fullstep-v1-raw-20261028',
        'compact-consumer-raw':'/private/tmp/cjj-native-k8-compact-consumer-v1-raw-20261028',
        'compile-out':'/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out',
        'codec-summary':defaults['codec-summary'],'original-library':defaults['original-library']}
    edits=[(repr(old),repr(str(getattr(a,key.replace('-','_')).resolve()))) for key,old in path_defaults.items()]
    edits.append((repr('/private/tmp/check_cjj_native_k8_fullstep_actual_20261028.py'),repr(str(AUDIT))))
    edits.extend([
        ("COMMIT='fa92060af9399e60938ddee6dd7e0a603610cd8c'",'COMMIT='+repr(SOURCE)),
        ('sys.path.insert(0,str(SOURCE/\'tools\'));import probe_biv150_native_k8_compact_fullstep as c',
         'sys.path.insert(0,str(SOURCE/\'tools\'));import probe_biv150_native_k8_compact_v_overlap_fullstep as variant\n c=types.SimpleNamespace(**vars(variant.base_module()))\n c.LIBRARY='+repr(LIBRARY)+';c.CONSUMER='+repr(CONSUMER)+'\n c.adapted_source=variant.adapted_source;c.replay_consumer=variant.replay_consumer'),
        ("summary['namespace']=='cjj_native_k8_compact_probe'","summary['namespace']=='cjj_native_k8_compact_v_overlap_probe'"),
        ("k8_library=compiled/'build/cjj_native_k8_compact_probe.so')",
         "k8_library=compiled/'build/cjj_native_k8_compact_v_overlap_probe.so',assembly_review=Path("+repr(str((a.compact_consumer_raw/'out/actual-so-review-input.json').resolve()))+"),assembly_review_sha256="+repr(REVIEW)+",variant_consumer_sha256="+repr(CONSUMER)+",variant_archive_manifest_sha256='4435696c50214a8ba60f6a4e4fbf7ff1970ce3f8fb202cb13f7b82b4c4c5613a')"),
        ("ast.parse((SOURCE/'tools/probe_biv150_native_k8_compact_fullstep.py').read_text())",'ast.parse(variant.execute_source()[0])'),
        ("dict(pid=38381,start_ticks=354308715,same_start=True,state='Z')","dict(pid=61630,start_ticks=355834758,same_start=True,state='Z')"),
        ("len(paths)==manifest['raw_files']==181 and total==manifest['raw_bytes']==145937040","len(paths)==manifest['raw_files']==181 and total==manifest['raw_bytes']==145951897"),
        ("raw/'exports/cjj-native-k8-compact-fullstep-v1-20261028-owner.json'","raw/'exports/cjj-native-k8-compact-v-overlap-fullstep-v1-20261029-owner.json'"),
        ("and owner['pid']==38381 and owner['start_ticks']==354308715","and owner['pid']==61630 and owner['start_ticks']==355834758"),
        ("compact_frontend_sha256=sha(SOURCE/'tools/probe_biv150_native_k8_compact_fullstep.py')","compact_frontend_sha256=sha(SOURCE/'tools/probe_biv150_native_k8_compact_v_overlap_fullstep.py')"),
    ])
    text=original
    for old,new in edits:
        assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
    restored=text
    for old,new in reversed(edits):
        assert restored.count(new)==1,(new,restored.count(new));restored=restored.replace(new,old)
    assert restored==original;ast.parse(text)
    proof=dict(original_PR108_adapter_sha256=sha(SCRIPT),original_3eb_auditor_sha256=sha(AUDIT),
        adapted_review_sha256=hashlib.sha256(text.encode()).hexdigest(),inverse_byte_exact=True,
        edits=[dict(old=x,new=y) for x,y in edits],numeric_reference_writer_cache_and_ABBA_bodies_unchanged=True,
        exact_variant_execute_source_used_for_nested_plan=True)
    if not a.run:
        print(json.dumps(dict(CPU_plan=True,review_executed=False,proof=proof,qualification_granted=False),sort_keys=True,indent=2));return
    assert a.out is not None and not a.out.exists()
    assert sha(a.raw_root/'download-manifest.json')==MANIFEST and sha(a.raw_root/'out/fullstep-summary.json')==SUMMARY
    old_argv=sys.argv;sys.argv=[str(SCRIPT),'--out',str(a.out.resolve())]+(['--regenerate'] if a.regenerate else [])
    try:
        ns=dict(__name__='compact110_identity_only_review',__file__=str(Path(__file__).resolve()))
        exec(compile(text,str(SCRIPT)+'::<PR110 identity adapter>','exec'),ns);ns['main']()
    finally:sys.argv=old_argv
    report=json.loads(a.out.read_text());report['variant_identity_adapter']=proof
    report['variant_identity_adapter_sha256']=sha(__file__)
    report['scope_limits'].append('The 1.21669/1.22020 denominators are oldaace component costs. Old40f9 PR108 ratios are cross-run historical context only, not a same-run comparison or latest-main model gain.')
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    assert not {'torch','triton'}.intersection(sys.modules)
    print(json.dumps(dict(report=str(a.out),sha256=sha(a.out),passed=report['passed'],regenerated=report['cold_counter_and_gold_regenerated'],
        event_ratios={r['case']:r['event_BF16_over_K8'] for r in report['records']},qualification=report['qualification']),sort_keys=True))

if __name__=='__main__':main()
