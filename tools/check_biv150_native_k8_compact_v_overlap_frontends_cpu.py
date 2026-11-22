#!/usr/bin/env python3
"""Narrow source/parameter checks only; no synthetic candidate SO or GPU receipts."""
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types

import probe_biv150_native_k8_compact_v_overlap_consumer as consumer
import probe_biv150_native_k8_compact_v_overlap_fullstep as fullstep


def check(parent,old_compile):
    checks=[];old,factory,contract,binding=consumer.modules()
    original=consumer.frozen_module('biv150_native_k8_compact_consumer_contract')
    assert contract.LiveEpoch is original.LiveEpoch
    assert contract.admission.__code__ is original.admission.__code__
    assert contract.review_template.__code__ is original.review_template.__code__
    assert factory.FORMAT==old.factory.FORMAT and factory.PAGE==old.factory.PAGE
    assert factory.admit_fixture is old.factory.admit_fixture
    assert contract.REVIEW_FLAGS==original.REVIEW_FLAGS+consumer.EXTRA_REVIEW
    checks.append('original_layout_LiveEpoch_admission_and_resource_code_unchanged')
    run,runproof=consumer.adapted_run();assert runproof['inverse_byte_exact'] and len(runproof['edits'])==2
    assert binding['inverse_byte_exact'] and len(binding['edits'])==2
    assert json.loads(json.dumps(consumer.plan()['variant_adapter']))==consumer.plan()['variant_adapter']
    checks.append('consumer_run_only_namespace_and_provenance_tuple;binding_only_compile_frontend_flags;JSON_identity_roundtrip')
    # Exercise the unchanged review flag assertion against the extended tuple.
    # No fake compile/SO/hash is created and no admission-success receipt is made.
    tree=ast.parse(inspect.getsource(original.admission))
    gate=next(n for n in ast.walk(tree) if isinstance(n,ast.Assert) and 'REVIEW_FLAGS' in ast.unparse(n.test))
    code=compile(ast.fix_missing_locations(ast.Module(body=[gate],type_ignores=[])),'<actual-review-flag-gate>','exec')
    negatives=[]
    for flag in consumer.EXTRA_REVIEW:
        for value in (None,False,'absent'):
            review={key:True for key in contract.REVIEW_FLAGS}
            if value=='absent':review.pop(flag)
            else:review[flag]=value
            try:exec(code,dict(j=review,REVIEW_FLAGS=contract.REVIEW_FLAGS))
            except (AssertionError,KeyError):negatives.append(flag+':'+str(value))
            else:raise AssertionError('unknown/missing/false review passed')
    # Template defaults may be inspected with an empty binding, without
    # pretending that a candidate artifact or actual review exists.
    ns=dict(contract.review_template.__globals__,compile_binding=lambda path:{})
    template=types.FunctionType(contract.review_template.__code__,ns)(None)
    assert all(template[key] is False for key in contract.REVIEW_FLAGS)
    assert all(value is None for row in template['static_resources'].values() for value in row.values())
    assert 'library_sha256' not in template
    checks.append('extended_review_flags_missing_false_unknown_rejected;resources_default_unknown')
    try:contract.compile_binding(old_compile)
    except AssertionError:checks.append('real40f9_compile_rejected_as_new_candidate_before_framework')
    else:raise AssertionError('Borrowed40f9 compile accepted')
    source,transform=fullstep.adapted_source()
    assert transform['variant_namespace_adapter']['inverse_byte_exact']
    assert len(transform['variant_namespace_adapter']['edits'])==1
    assert fullstep.execute_source()[1]['inverse_byte_exact']
    plan=fullstep.plan();assert plan['library_sha256'] is None and plan['consumer_summary_sha256'] is None
    assert plan['fixed_pages']==34276 and plan['new_KV_pitch']==256
    assert plan['cases']=={'4k':[64,4096,2,2048],'16k':[32,16384,4,2560]}
    checks.append('fullstep_namespace_only;34276_pages_original_writer_shapes_timing_and_saved_arrays_preserved')
    a=types.SimpleNamespace(compact_consumer_raw=Path('/CPU_GENERATION_ONLY/raw'),compile_summary=Path('/CPU_GENERATION_ONLY/build/compile-summary.json'),codec_summary=Path('/CPU_GENERATION_ONLY/codec-summary.json'),original_library=Path('/CPU_GENERATION_ONLY/original.so'),assembly_review_sha256=None,k8_library=Path('/CPU_GENERATION_ONLY/unknown-candidate.so'))
    owner=dict(pid=1,start_ticks=2,source_commit='940d09984da201992194dbc25421e9f9f87198b4')
    inventory=dict(raw_files=1,raw_bytes=1,original_owner=dict(pid=1,start_ticks=2,same_start=True,state='Z'))
    replay,proof=fullstep.replay_source(a,owner,inventory)
    assert proof['inverse_byte_exact'] and proof['entire_array_numeric_replay_body_byte_identical']
    assert len(proof['edits'])==14
    checks.append('saved_array_replay_identity_header_only;entire_numeric_body_byte_identical_not_executed')
    root=consumer.candidate.ROOT
    for name in ('compile_biv150_native_k8_compact_v_overlap.py','biv150_native_k8_compact_v_overlap_factory.py'):
        assert (root/'tools'/name).read_bytes()==consumer.candidate.blob('940d09984da201992194dbc25421e9f9f87198b4','tools/'+name)
    checks.append('existing_compile_entry_factory_and_candidate_sources_unchanged')
    for name in (consumer.SELF,fullstep.SELF):
        p=root/'tools'/name
        output=subprocess.run([sys.executable,'-B',str(p)],capture_output=True,text=True,check=True)
        json.loads(output.stdout)
        rejected=subprocess.run([sys.executable,'-B',str(p),'--gpu'],capture_output=True,text=True)
        assert rejected.returncode==2 and 'Root' in rejected.stderr
    checks.append('default_CPU_plans_and_missing_real_receipt_GPU_arguments_rejected')
    # Reproduce actual import order in a separate stdlib-only interpreter: the
    # original3eb math/fixture modules are prepared before the PR100 adapter.
    script="""import sys
from pathlib import Path
import probe_biv150_native_k8_compact_v_overlap_fullstep as s
old=s.base_module();old.prepare(Path(sys.argv[1]))
import probe_biv150_native_k8_compact_v_overlap_consumer as c
c.plan();old.prepare(Path(sys.argv[1]));assert not {'torch','triton','numpy'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable,'-B','-c',script,str(parent)],cwd=root/'tools',check=True,capture_output=True,text=True)
    checks.append('frozen3eb_then_new_adapter_import_order_valid_without_framework')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,checks=checks,rejected_review_fields=negatives,
        candidate_library_sha256=None,no_fake_candidate_SO_or_receipt_written=True,
        original40f9_compile_readonly_negative=True,source_generation_fixture_not_executed=True,
        default_CPU_only=True,new_compile_entry=False,new_numerical_framework=False,
        GPU_executed=False,SSH_executed=False,actual_SO_reviewed=False,
        four_case_numeric_verified=False,fullstep_verified=False,performance_verified=False,
        precision_gate=False,source_pins={name:fullstep.sha(root/'tools'/name) for name in
            (consumer.SELF,fullstep.SELF,Path(__file__).name)})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True)
    p.add_argument('--old40f9-compile-summary',type=Path,required=True);a=p.parse_args()
    print(json.dumps(check(a.parent_source,a.old40f9_compile_summary),indent=2,sort_keys=True))
