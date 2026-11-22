#!/usr/bin/env python3
"""Reuse PR108 fullstep and saved-array replay; only candidate identities differ."""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PARENT='46b44a1326973ccb6b874452841726298507414c'
OLD='probe_biv150_native_k8_compact_fullstep'
SELF='probe_biv150_native_k8_compact_v_overlap_fullstep.py'
NAMESPACE='cjj_native_k8_compact_v_overlap_probe'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_module():
    path=ROOT/'tools'/(OLD+'.py')
    assert path.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':tools/'+path.name])
    module=importlib.import_module(OLD);assert Path(module.__file__).resolve()==path
    return module


def replace_exact(original,edits):
    changed=original
    for old,new in edits:
        assert changed.count(old)==1,old
        changed=changed.replace(old,new)
    restored=changed
    for old,new in reversed(edits):
        assert restored.count(new)==1,new
        restored=restored.replace(new,old)
    assert restored==original
    ast.parse(changed)
    return changed,dict(parent_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_sha256=hashlib.sha256(changed.encode()).hexdigest(),inverse_byte_exact=True,
        edits=[dict(old=a,new=b) for a,b in edits])


def adapted_source():
    source,parent_proof=base_module().adapted_source()
    changed,proof=replace_exact(source,[('torch.ops.cjj_native_k8_compact_probe.produce',
                                       'torch.ops.'+NAMESPACE+'.produce')])
    return changed,dict(parent_compact_adapter=parent_proof,variant_namespace_adapter=proof)


def replay_source(a,owner,manifest):
    old=base_module();path=ROOT/old.REPLAY_FILE
    text=path.read_text();assert sha(path)==old.REPLAY_SHA
    prefix,_=text.split("path=Path('/private/tmp/cjj-compact100-consumer-actual-review-20261028.json')",1)
    edits=[
        (repr('/private/tmp/flagos-cjj-native-k8-compact-consumer-frozen-51d-20261028'),repr(str(ROOT))),
        (repr('/private/tmp/cjj-native-k8-compact-consumer-v1-raw-20261028'),repr(str(a.compact_consumer_raw))),
        (repr('/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out'),repr(str(a.compile_summary.parent))),
        (repr('/private/tmp/cjj-native-k8-codec-v1-raw-20261028/out/codec-summary.json'),repr(str(a.codec_summary))),
        (repr('/private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so'),repr(str(a.original_library))),
        ('import probe_biv150_native_k8_compact_consumer as frontend',
         'import probe_biv150_native_k8_compact_v_overlap_consumer as variant_frontend\nfrontend=variant_frontend.replay_frontend()'),
        ('import biv150_native_k8_compact_consumer_contract as contract','contract=variant_frontend.modules()[2]'),
        ("(manifest['raw_files'],manifest['raw_bytes'])==(138,32787475)",
         "(manifest['raw_files'],manifest['raw_bytes'])=="+repr((manifest['raw_files'],manifest['raw_bytes']))),
        ("dict(pid=14175,start_ticks=353418626,same_start=True,state='Z')",repr(manifest['original_owner'])),
        ("owner['pid']==14175 and owner['start_ticks']==353418626 and owner['source_commit']=='51da6fd73e7fff1513081fb330323ce72c00856c'",
         "owner['pid']=="+repr(owner['pid'])+" and owner['start_ticks']=="+repr(owner['start_ticks'])+" and owner['source_commit']=="+repr(owner['source_commit'])),
        ("len(manifest['files'])==138 and sum(x['bytes'] for x in manifest['files'])==32787475",
         "len(manifest['files'])=="+repr(manifest['raw_files'])+" and sum(x['bytes'] for x in manifest['files'])=="+repr(manifest['raw_bytes'])),
        ("review_sha=='3b3e94c9af183b77df5be794746f2b783c1b925f643b5d1d450e13da3a3ddc28'",'review_sha=='+repr(a.assembly_review_sha256)),
        ("COMPILE/'build/cjj_native_k8_compact_probe.so'",'Path('+repr(str(a.k8_library))+')'),
        ('raw_files=138,raw_bytes=32787475',"raw_files=manifest['raw_files'],raw_bytes=manifest['raw_bytes']"),
    ]
    changed,proof=replace_exact(prefix,edits)
    # All numerical loops, metrics, tolerances, array loads, Graph/side and
    # original BF16 proxy checks remain the same byte string.
    start=prefix.index('@functools.lru_cache');end=prefix.index('\nreport=dict(')
    assert prefix[start:end]==changed[changed.index('@functools.lru_cache'):changed.index('\nreport=dict(')]
    proof['entire_array_numeric_replay_body_byte_identical']=True
    return changed,proof


def replay_consumer(a):
    # Called only after prepare(parent), so original3eb imports retain ownership.
    import probe_biv150_native_k8_compact_v_overlap_consumer as consumer
    old=base_module();summary=a.compact_consumer_raw/'out/consumer-summary.json'
    inventory=a.compact_consumer_raw/'download-manifest.json'
    for digest in (a.variant_consumer_sha256,a.variant_archive_manifest_sha256,a.assembly_review_sha256):
        assert re.fullmatch('[0-9a-f]{64}',digest or '')
    assert sha(summary)==a.variant_consumer_sha256 and sha(inventory)==a.variant_archive_manifest_sha256
    manifest=json.loads(inventory.read_text());owners=list((a.compact_consumer_raw/'exports').glob('*owner.json'))
    assert len(owners)==1;owner=json.loads(owners[0].read_text())
    assert type(owner['pid']) is int and owner['pid']>0 and type(owner['start_ticks']) is int and owner['start_ticks']>0
    observed=manifest['original_owner']
    assert observed['pid']==owner['pid'] and observed['start_ticks']==owner['start_ticks']
    assert observed['same_start'] is True and observed['state'] in ('Z','X') and manifest['model_rerun'] is False
    assert type(manifest['raw_files']) is int and manifest['raw_files']>0
    assert type(manifest['raw_bytes']) is int and manifest['raw_bytes']>0
    assert re.fullmatch('[0-9a-f]{40}',owner['source_commit'])
    d=json.loads(summary.read_text())
    # The future owner commit must contain exactly this executed adapter, factory,
    # compile frontend and unchanged parent/reference files, not a borrowed run.
    assert d['variant_adapter']==consumer.plan()['variant_adapter']
    for name,digest in {**d['frontend_sources'],**d['reference_source_pins']}.items():
        local=ROOT/'tools'/name;assert sha(local)==digest
        assert local.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',owner['source_commit']+':tools/'+name])
    assert (a.compact_consumer_raw/'out/actual-so-review-input.json').read_bytes()==a.assembly_review.read_bytes()
    admission=consumer.modules()[2].admission(a.compile_summary,a.assembly_review,a.assembly_review_sha256,a.k8_library)
    assert d['compact_admission']==admission
    text,proof=replay_source(a,owner,manifest)
    ns=dict(__file__=str(ROOT/old.REPLAY_FILE),__name__='cjj_compact_overlap_saved_array_replay')
    exec(compile(text,'<existing-compact-replay-candidate-identities-only>','exec'),ns)
    assert ns['admission']==admission
    result=ns['report'];result['variant_replay_adapter']=proof
    result['variant_archive_manifest_sha256']=a.variant_archive_manifest_sha256
    result['variant_fullstep_frontend_sha256']=sha(ROOT/'tools'/SELF)
    assert not {'torch','triton'}.intersection(sys.modules)
    return result


def execute_source():
    return replace_exact(inspect.getsource(base_module().execute),[
        ('native40f9','native deferred V wait'),
        ("namespace='cjj_native_k8_compact_probe'","namespace='"+NAMESPACE+"'")])


def execute(a):
    assert a.root_owned_exclusive and not a.out.exists()
    assert all(p.is_absolute() for p in vars(a).values() if isinstance(p,Path))
    old=base_module();old.prepare(a.parent_source)
    import probe_biv150_native_k8_compact_v_overlap_consumer as consumer
    binding=consumer.modules()[2].admission(a.compile_summary,a.assembly_review,a.assembly_review_sha256,a.k8_library)
    assert sha(a.compact_consumer_raw/'out/consumer-summary.json')==a.variant_consumer_sha256
    source,_=execute_source()
    ns=dict(vars(old),LIBRARY=binding['library_sha256'],CONSUMER=a.variant_consumer_sha256,
            replay_consumer=replay_consumer,adapted_source=adapted_source)
    exec(compile(source,'<existing-compact-fullstep-candidate-identities-only>','exec'),ns)
    return ns['execute'](a)


def plan():
    result=base_module().plan()
    result.update(library_sha256=None,consumer_summary_sha256=None,
        namespace=NAMESPACE,parent_compact_fullstep=PARENT,transform=adapted_source()[1],
        execution_adapter=execute_source()[1],new_actual_SO_review_and_consumer_archive_required=True,
        current40f9_qualification_unchanged=True,old_case_math_writer_timing_and_saved_arrays_unchanged=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true')
    paths=('parent-source','compact-consumer-raw','compile-summary','assembly-review','codec-summary','original-library','consumer-summary','k8-library','bf16-library','out')
    for key in paths:p.add_argument('--'+key,type=Path)
    for key in ('consumer-proof-sha256','variant-consumer-sha256','variant-archive-manifest-sha256','assembly-review-sha256'):p.add_argument('--'+key)
    p.add_argument('--case',choices=('4k','16k'),action='append');a=p.parse_args()
    if not a.gpu:
        if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Device arguments require --gpu')
        print(json.dumps(plan(),indent=2,sort_keys=True));return 0
    if not all(vars(a)[k.replace('-','_')] for k in (*paths,'root-owned-exclusive','consumer-proof-sha256','variant-consumer-sha256','variant-archive-manifest-sha256','assembly-review-sha256')):
        p.error('Root ownership, actual candidate compile/review/consumer/archive SHA and original references required')
    if a.case and len(a.case)!=len(set(a.case)):p.error('Duplicate case')
    return execute(a)


if __name__=='__main__':raise SystemExit(main())
