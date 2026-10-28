"""Narrow actual5cf6/Root-b481 admission; unchanged compact layout and live epoch."""
import json
from pathlib import Path
import types

import biv150_native_k8_compact_n48_consumer_contract as old
import biv150_native_k8_compact_n48_shared6k_factory as candidate

SO='5cf698e8a60272d86b2545364f580faf4c40114e0044e91557c1a137f2521bed'
SOURCE='1dc7ce5c2dd9050712aa9d5e7cba59fe3699f8d4'
ROOT_REVIEW='b481ccf7ec5da72b3036b91d212ebdf02c97d906d93f0914d083df7b35d2d87f'
INDEPENDENT_REVIEW='48b8e9d8ca1ee2a226727727dcedafa2663086952dcfa37578209bc7a54e5654'
REFERENCE_LIBRARY_SHA=old.REFERENCE_LIBRARY_SHA
REFERENCE_CONSUMER_SHA=old.REFERENCE_CONSUMER_SHA
reference_consumer=old.reference_consumer
original=old.original


def contracts():
    base,proof=old.namespace_contract(candidate,'compile_biv150_native_k8_compact_n48_shared6k')
    reference=old.contracts()[1]
    def admission(summary,review_path,review_sha,library):
        binding=base.compile_binding(summary)
        raw=Path(review_path).read_bytes()
        assert review_sha==ROOT_REVIEW and candidate.sha(raw)==ROOT_REVIEW,'exact Root113 numeric admission required'
        review=json.loads(raw)
        assert review['source_commit']==SOURCE and review['SO_sha256']==SO==binding['library_sha256']
        assert candidate.sha(Path(library).read_bytes())==SO
        assert review['compile_summary_sha256']==binding['compile_summary_sha256']
        assert review['inspection_sha256']==binding['inspection_sha256']
        assert review['completed'] is True and review['errors']==[]
        assert review['actual_static_review_passed'] is True and review['manual_Root_approval'] is True
        assert review['instructions_bound']==2764 and review['actual_GPU_images_equal_to_built_objects'] is True
        assert review['actual_full_metadata_ABI_equal_except_names_and_shared'] is True and review['actual_codec_text_unchanged'] is True
        assert review['actual_S4_resources']=={'.vgpr_count':100,'.sgpr_count':66,'.group_segment_fixed_size':6144,
          '.private_segment_fixed_size':0,'.vgpr_spill_count':0,'.sgpr_spill_count':0,'.warp_size':64,
          '.kernarg_segment_size':144,'.kernarg_segment_align':8}
        assert all(review[k] is False for k in ('GPU_numeric_verified','hardware_SME_alignment_verified','Graph_verified',
          'side_stream_verified','fullstep_verified','occupancy_verified','performance_verified','model_verified','service_verified'))
        assert 'isolated numeric/Graph/side tests only' in review['Root_approval_scope']
        pins={r['path']:r['sha256'] for r in review['source_inverse']}
        for role in candidate.FILES:
            p=candidate.path(role);assert pins[str(p.relative_to(candidate.ROOT))]==candidate.sha(p.read_bytes())
        scope=dict(rows=48,segments=4,Q_pitches=[2048,2560],namespace=candidate.NAMESPACE,shared_bytes=6144,
          consumer_only=True,service_or_model_verified=False)
        return dict(**binding,review_sha256=review_sha,review=review,scope=scope,
          qualification_inherited_from_original22e=False,qualification_inherited_from_2ad=False,
          review_schema='original Root113 actual-SO report, not the legacy S2/S4 8192 template')
    def review_template(summary):
        return dict(**base.compile_binding(summary),required_original_Root_review_sha256=ROOT_REVIEW,
          independent_report_sha256=INDEPENDENT_REVIEW,manual_Root_approval=False,
          notes='Use the already issued original Root receipt; this description is not an approval template or qualification.')
    result=types.SimpleNamespace(**vars(base));result.admission=admission;result.review_template=review_template
    return result,reference,dict(candidate_compile_binding=proof,Root_review_sha256=ROOT_REVIEW,
      original_Root_review_schema_preserved=True,candidate_scope='N48/S4/6144 only',reference_contract_unchanged=True)


def record_executed_source(out,result):
    import probe_biv150_native_k8_compact_n48_shared6k_consumer as adapter
    text,proof=adapter.adapted_source();raw=text.encode();name='executed-consumer-source.py'
    (out/name).write_bytes(raw)
    result['executed_consumer_source']=dict(relative=name,bytes=len(raw),sha256=candidate.sha(raw),source_adapter=proof)
