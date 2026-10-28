"""Exact Root114 ef73 admission; original four-case aligned-cache scope only."""
import json
from pathlib import Path
import types

import biv150_native_k8_compact_n48_consumer_contract as old
import biv150_native_k8_compact_n48_kpair_factory as candidate

SO='ef732ef832b6399bf156d24c705b15e2e9e766947c0a456745673953ffbfb8fd'
SOURCE='52e1d2680d41e44f025b79fccb85867c8e783d39'
ROOT_REVIEW='5fc79ec23dc50108eea06ee34e3c56d7f55bd2ca5d6f3774b2531919e71b1f61'
INDEPENDENT_REVIEW='49d64203b9eae093000b7dae9f176869fc460ac45f646002ee5d223ebfeeca53'
REFERENCE_LIBRARY_SHA=old.REFERENCE_LIBRARY_SHA
REFERENCE_CONSUMER_SHA=old.REFERENCE_CONSUMER_SHA
reference_consumer=old.reference_consumer
original=old.original


def contracts():
    base,proof=old.namespace_contract(candidate,'compile_biv150_native_k8_compact_n48_kpair')
    reference=old.contracts()[1]
    def admission(summary,review_path,review_sha,library):
        binding=base.compile_binding(summary)
        raw=Path(review_path).read_bytes()
        assert review_sha==ROOT_REVIEW and candidate.sha(raw)==ROOT_REVIEW,'exact Root114 numeric admission required'
        review=json.loads(raw)
        assert review['source_commit']==SOURCE and review['SO_sha256']==SO==binding['library_sha256']
        assert candidate.sha(Path(library).read_bytes())==SO
        assert review['compile_summary_sha256']==binding['compile_summary_sha256']
        assert review['inspection_sha256']==binding['inspection_sha256']
        assert review['completed'] is True and review['errors']==[] and review['manual_Root_approval'] is True
        assert all(review[k] is True for k in ('actual_static_review_passed','actual_GPU_images_equal_to_built_objects',
          'actual_K_vector_lowering_verified','actual_codec_text_unchanged',
          'actual_full_metadata_ABI_equal_except_names_and_register_counts','compile_flags_and_compiler_unchanged'))
        assert review['instructions_bound']==3168
        admitted=[r for r in review['producer_reviews'] if r['host_admitted'] is True]
        assert len(admitted)==1 and 'kpair_producerILj4E' in admitted[0]['symbol']
        s4=admitted[0]
        resources={'.vgpr_count':97,'.sgpr_count':69,'.group_segment_fixed_size':8192,
          '.private_segment_fixed_size':0,'.vgpr_spill_count':0,'.sgpr_spill_count':0,
          '.warp_size':64,'.kernarg_segment_size':144,'.kernarg_segment_align':8}
        assert {k:s4['resource'][k] for k in resources}==resources
        assert all(s4[k] is True for k in ('actual_branch_zero_selects_only_vector_and_nonzero_only_scalar',
          'actual_vector_lowering_verified','math_sequence_and_semantic_flags_equal','scalar_fallback_static_code_retained'))
        assert len(s4['paths'])==2 and {r['path'] for r in s4['paths']}=={'full','tail'}
        for path in s4['paths']:
            assert all(path[k] is True for k in ('all_K_paths_rejoin_barrier','P_packing_before_wait','wait_dominates_tail_and_PV'))
        assert all(review[k] is False for k in ('GPU_numeric_verified','hardware_alignment_verified','Graph_verified',
          'side_stream_verified','fullstep_verified','occupancy_verified','performance_verified','service_verified'))
        assert 'Only isolated numeric/Graph/side tests' in review['Root_approval_scope']
        pins={r['path']:r['sha256'] for r in review['source_inverse']}
        for role in candidate.FILES:
            p=candidate.path(role);assert pins[str(p.relative_to(candidate.ROOT))]==candidate.sha(p.read_bytes())
        scope=dict(rows=48,segments=4,Q_pitches=[2048,2560],namespace=candidate.NAMESPACE,shared_bytes=8192,
          consumer_only=True,service_or_model_verified=False,default_key_base_mod8=0,
          whole_cache_4B_non8B_verified=False,unused_S2_admitted=False)
        return dict(**binding,review_sha256=review_sha,review=review,scope=scope,
          qualification_inherited_from_original22e=False,qualification_inherited_from_2ad=False,
          review_schema='original Root114 actual K-pair branch/IR/SO report; no PR113 schema translation')
    def review_template(summary):
        return dict(**base.compile_binding(summary),required_original_Root_review_sha256=ROOT_REVIEW,
          independent_report_sha256=INDEPENDENT_REVIEW,manual_Root_approval=False,
          notes='Use the already issued original Root114 receipt; this description is not approval or qualification.')
    result=types.SimpleNamespace(**vars(base));result.admission=admission;result.review_template=review_template
    return result,reference,dict(candidate_compile_binding=proof,Root_review_sha256=ROOT_REVIEW,
      original_Root_review_schema_preserved=True,candidate_scope='N48/S4/8192 aligned original four cases only',
      reference_contract_unchanged=True,whole_cache_4B_non8B_pending=True)


def record_default_alignment(key,scale,value,observe):
    """Record actual addresses; refuse to silently add the untested shifted case."""
    pointers={name:observe(t)['pointer'] for name,t in (('key',key),('scale',scale),('value',value))}
    assert all(p%8==0 for p in pointers.values()),'original default whole-cache alignment only; 4B-not8B remains pending'
    assert pointers['scale']==pointers['key'] and pointers['value']==pointers['key']+4352
    return dict(pointers=pointers,pointer_mod8={k:v%8 for k,v in pointers.items()},
      layer_offset_bytes=128,page_stride_bytes=12544,extra_whole_cache_shift_bytes=0,
      shifted_4B_non8B_exercised=False,scalar_fallback_device_qualification=False)


def record_executed_source(out,result):
    import probe_biv150_native_k8_compact_n48_kpair_consumer as adapter
    text,proof=adapter.adapted_source();raw=text.encode();name='executed-consumer-source.py'
    (out/name).write_bytes(raw)
    result['executed_consumer_source']=dict(relative=name,bytes=len(raw),sha256=candidate.sha(raw),source_adapter=proof)
