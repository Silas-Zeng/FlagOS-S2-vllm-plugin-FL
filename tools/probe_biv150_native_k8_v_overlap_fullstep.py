#!/usr/bin/env python3
"""CPU default; original complete-step ABBA after a fresh four-case consumer pass.

Reuses the frozen 3eb writer, producer/reducer tests and timing body. Actual SO
review and complete saved-array consumer replay are mandatory before Torch.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
import probe_biv150_native_k8_v_overlap_consumer as consumer

PINS={
 'probe_biv150_native_k8_fullstep':'79587c5b1ac19f0012b1cc49c6929f6be7f4c5567d9ec12b6cb4ab81f3ae2d0d',
 'biv150_native_k8_fullstep_fixture':'4fa0ac8b28855fb5a4f398467fc0a67c239d5c286aad25efbd7183230ce9d528',
 'biv150_native_k8_consumer_receipt':'509bab849580754b5015b4cd0760297f15678d3b9feeca35f6b76eff4599ad31',
 'biv150_native_k8_source_factory':'7cc9a3ebded9bed5d8f069efae04a7a142714413ca1a82f60bd7aa4e59cc3af9',
 'biv150_native_tl_pair_contract':'15827c408c98caad0612727f30bd3acee0b45956a3d387df3bfc422b4b627fb4',
 'biv150_konly_strided_writer_contract':'cdc1384ff4443bd577759592fa9aa3ea0ff79032507c7d6e63f4abc8db2569b9',
}


def parent_files(root):
    root=consumer.parent_files(root)
    for name,digest in PINS.items():
        path=root/'tools'/(name+'.py');assert consumer.sha(path)==digest,name
        if name in sys.modules:
            assert Path(sys.modules[name].__file__).resolve()==path.resolve(),'foreign loaded module: '+name
    return root


def adapted_source(root,kind):
    root=parent_files(root)
    file,fn=('probe_biv150_native_k8_fullstep','run') if kind=='fullstep' else ('biv150_native_k8_consumer_receipt','require_consumer')
    assert kind in ('fullstep','receipt')
    text=(root/'tools'/(file+'.py')).read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==fn)
    original=ast.get_source_segment(text,node);changed=original
    if kind=='fullstep':
        edits=(('f.sha(a.k8_library.read_bytes())==f.LIBRARY_SHA','f.sha(a.k8_library.read_bytes())==a.variant_sha256',1),
               ('torch.ops.cjj_native_k8_qk_probe.produce','torch.ops.cjj_native_k8_v_overlap_probe.produce',1),
               ('result=dict(f.plan(),actual_consumer=pre,','result=dict(variant_plan(),actual_consumer=pre,',1))
    else:
        edits=(("expected_sha=='c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128' and f.sha(raw)==expected_sha",
                'f.sha(raw)==expected_sha',1),('f.LIBRARY_SHA','variant_sha256',2))
    for old,new,count in edits:
        assert changed.count(old)==count;changed=changed.replace(old,new)
    inverse=changed
    for old,new,count in reversed(edits):
        assert inverse.count(new)==count;inverse=inverse.replace(new,old)
    assert inverse==original
    return changed,dict(parent_commit=consumer.PARENT,parent_file=file+'.py',
                        parent_function_sha256=consumer.factory.sha(original.encode()),
                        adapted_function_sha256=consumer.factory.sha(changed.encode()),inverse_byte_exact=True,
                        substitutions=[dict(old=a,new=b,count=n) for a,b,n in edits])


def variant_provenance(j,binding,adapter):
    assert j['library_sha256']==binding['library_sha256']
    assert j['variant']['compile_admission']==binding
    assert j['variant']['source_proof']==consumer.factory.verify()
    assert j['variant']['host_adapter']==adapter
    assert j['variant']['qualification_inherited_from_old_reader'] is False


def load_module(root,file,name):
    assert name not in sys.modules
    spec=importlib.util.spec_from_file_location(name,root/'tools'/(file+'.py'))
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    parent_files(root);return result


def execute(a):
    parent,binding=consumer.admission(a)
    parent=parent_files(parent)
    assert a.variant_consumer_summary.is_absolute() and a.variant_consumer_summary.is_file()
    assert re.fullmatch('[0-9a-f]{64}',a.variant_consumer_sha256 or '')
    assert consumer.sha(a.variant_consumer_summary)==a.variant_consumer_sha256
    variant=json.loads(a.variant_consumer_summary.read_text())
    variant_provenance(variant,binding,consumer.adapted_run(parent)[1])
    assert a.consumer_summary.is_absolute() and a.bf16_library.is_absolute()
    a.variant_sha256=binding['library_sha256'];a.k8_library=a.library
    fullstep_source,fullstep_adapter=adapted_source(parent,'fullstep')
    receipt_source,receipt_adapter=adapted_source(parent,'receipt')
    sys.path.insert(0,str(parent/'tools'))
    receipt=load_module(parent,'biv150_native_k8_consumer_receipt','cjj_v_overlap_fullstep_original_receipt')
    namespace=dict(vars(receipt),variant_sha256=a.variant_sha256)
    exec(compile(receipt_source,'<V-overlap-original-consumer-receipt-identity-adapter>','exec'),namespace)
    # This runs all original full-array/FP64/input/Graph/side receipt checks;
    # summary booleans alone never admit fullstep, and no Torch is imported yet.
    actual_variant=namespace['require_consumer'](a.variant_consumer_summary,a.variant_consumer_sha256)
    assert not {'torch','triton'}.intersection(sys.modules)
    module=load_module(parent,'probe_biv150_native_k8_fullstep','cjj_v_overlap_fullstep_original_probe')
    def variant_plan():
        plan=module.f.plan();plan['K8_library_sha256']=a.variant_sha256
        plan['actual_variant_consumer']=actual_variant
        plan['variant']=dict(compile_admission=binding,source_proof=consumer.factory.verify(),
                             host_adapter=fullstep_adapter,consumer_receipt_adapter=receipt_adapter,
                             qualification_inherited_from_old_library=False)
        return plan
    namespace=dict(vars(module),variant_plan=variant_plan)
    exec(compile(fullstep_source,'<V-overlap-original-fullstep-identity-adapter>','exec'),namespace)
    return namespace['run'](a)  # Retains original c2e consumer and aace BF16 gates.


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true')
    paths=('parent-source','compile-summary','assembly-review','library','codec-summary','out',
           'consumer-summary','variant-consumer-summary','bf16-library')
    digests=('assembly-review-sha256','codec-proof-sha256','consumer-proof-sha256','variant-consumer-sha256')
    for name in paths:p.add_argument('--'+name,type=Path)
    for name in digests:p.add_argument('--'+name)
    p.add_argument('--case',choices=('4k','16k'),action='append');a=p.parse_args()
    if a.gpu:
        if not a.root_owned_exclusive or not all(getattr(a,x.replace('-','_')) for x in paths+digests):
            p.error('Root ownership, actual SO review, codec, old and fresh consumer receipts, BF16 library and new out required')
        if a.case and len(set(a.case))!=len(a.case):p.error('Duplicate case')
        return execute(a)
    if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Device arguments require --gpu')
    print(json.dumps(dict(parent_commit=consumer.PARENT,additional_parent_pins=PINS,
                         source_proof=consumer.factory.verify(),readonly_CPU_default=True,
                         fresh_consumer_full_arrays_replay_required=True,original_fullstep_body_reused=True,
                         numerical_or_timing_executed=False,GPU_executed=False,precision_gate=False),sort_keys=True,indent=2))
    assert not {'torch','triton','numpy'}.intersection(sys.modules);return 0


if __name__=='__main__':raise SystemExit(main())
