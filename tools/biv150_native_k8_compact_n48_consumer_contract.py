"""Actual SO and reference consumer gates, all before device imports."""
import inspect
import json
from pathlib import Path
import types

import biv150_native_k8_compact_n48_factory as candidate
import biv150_native_k8_compact_v_overlap_factory as reference
import biv150_native_k8_compact_consumer_contract as original
import biv150_native_k8_compact_factory as compact
from probe_biv150_native_k8_compact_v_overlap_consumer import EXTRA_REVIEW,replace_exact

REFERENCE_LIBRARY_SHA='7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4'
REFERENCE_CONSUMER_SHA='6f719fe84e83e547d0e42f7c2be5bdb408e64cdb107cc455994a681705e34c7c'


def namespace_contract(factory_module,compile_module,*,candidate_only=False):
    # Keep existing compile/SO artifact checks and all review flags unchanged.
    facade=types.SimpleNamespace(**vars(compact))
    facade.verify=factory_module.verify;facade.path=factory_module.path
    facade.NAMESPACE=factory_module.NAMESPACE;facade.FILES=factory_module.FILES
    source,proof=replace_exact(inspect.getsource(original.compile_binding),[
      ('import probe_biv150_native_k8_compact as frontend','import '+compile_module+' as frontend'),
      ('frontend.build_module().CUDA_FLAGS','frontend.CUDA_FLAGS')])
    ns=dict(vars(original),f=facade,REVIEW_FLAGS=original.REVIEW_FLAGS+EXTRA_REVIEW)
    exec(compile(source,'<N48-existing-actual-SO-binding>','exec'),ns)
    scope_proofs={}
    for name in ('review_template','admission'):
        if candidate_only:
            old=inspect.getsource(getattr(original,name))
            edits=[("for key in ('S2','S4')","for key in ('S4',)")] if name=='review_template' else [("assert set(j['static_resources'])=={'S2','S4'}","assert set(j['static_resources'])=={'S4'}")]
            adapted,scope_proofs[name]=replace_exact(old,edits)
            exec(compile(adapted,'<N48-S4-only-existing-resource-scope>','exec'),ns)
        else:
            fn=getattr(original,name);ns[name]=types.FunctionType(fn.__code__,ns,name)
    if candidate_only:
        base_template,base_admission=ns['review_template'],ns['admission']
        wanted_scope=dict(rows=48,segments=4,namespace=candidate.NAMESPACE,consumer_only=True,service_or_model_verified=False)
        def scoped_template(summary):
            result=base_template(summary);result['scope']=dict(wanted_scope);result['manual_Root_approval']=False;return result
        def scoped_admission(summary,review_path,review_sha,library):
            review=json.loads(Path(review_path).read_text())
            assert review['scope']==wanted_scope,'only actual N48/S4 consumer scope'
            assert review['manual_Root_approval'] is True,'Root actual-SO approval required'
            return base_admission(summary,review_path,review_sha,library)
        ns.update(review_template=scoped_template,admission=scoped_admission)
    proof=dict(compile_binding=proof,S4_only_source_adapters=scope_proofs)
    return types.SimpleNamespace(**ns),proof


def contracts():
    a,ap=namespace_contract(candidate,'compile_biv150_native_k8_compact_n48',candidate_only=True)
    b,bp=namespace_contract(reference,'compile_biv150_native_k8_compact_v_overlap')
    return a,b,dict(candidate_binding=ap,reference_binding=bp)


def reference_consumer(summary,expected_sha,reference_admission,library_sha):
    """Only an actual, complete PR110 four-case consumer can authorize control."""
    path=Path(summary);raw=path.read_bytes()
    assert expected_sha==REFERENCE_CONSUMER_SHA and candidate.sha(raw)==expected_sha,'actual reference consumer SHA required'
    result=json.loads(raw)
    for flag in ('completed','implementation_verified','full_field_bitwise_verified','Graph_verified','side_stream_verified'):
        assert result[flag] is True,flag
    assert result['errors']==[]
    assert library_sha==REFERENCE_LIBRARY_SHA,'Root actual PR110 7892 full library SHA required'
    assert result['compact_admission']==reference_admission
    assert result['compact_admission']['library_sha256']==library_sha
    assert result['compact_admission']['source_proof']==reference.verify()
    assert result['eager_producer_calls']==result['eager_reducer_calls']==24
    assert result['capture_pairs']==result['graph_replay_pairs']==8
    assert 'cjj_native_k8_compact_v_overlap_probe::produce' in result['actual_schemas']['compact']
    records=result['records']
    assert len(records)==4 and {tuple(r['case']) for r in records}=={(64,2,2048),(64,2,2560),(32,4,2048),(32,4,2560)}
    phases={'A-producer','A-after','graphB-after','eagerB-producer','eagerB-after','sideC-after'}
    for record in records:
        assert record['completed'] is True
        assert set(record['cross_arm_full_bitwise'])==phases
        for checks in record['cross_arm_full_bitwise'].values():
            assert set(checks)=={'partial','maximum','summation','output'} and all(x is True for x in checks.values())
        for arm in ('original','compact'):
            value=record['arms'][arm]
            assert value['completed'] is True and {x['stage'] for x in value['checks']}==phases
            assert all(all(v is True for v in item['checks'].values()) for item in value['checks'])
            assert value['closed_epoch_rejected'] if arm=='compact' else True
    seen=set();artifact_rows={}
    assert result['artifact_manifest'],'actual reference snapshots are required'
    for row in result['artifact_manifest']:
        assert row['relative'] not in seen;seen.add(row['relative'])
        original.artifact(path.parent,row);artifact_rows[row['relative']]=row
    for record in records:
        for arm in ('original','compact'):
            snapshots=record['arms'][arm]['snapshots']
            assert phases<=set(snapshots)
            for stage in phases:
                assert set(snapshots[stage])=={'query','cache','lengths','table','partial','maximum','summation','output'}
                for meta in snapshots[stage].values():
                    saved=artifact_rows[meta['path']]
                    assert saved['sha256']==meta['stored_sha256'] and saved['bytes']==meta['stored_bytes']
    return dict(summary_sha256=expected_sha,library_sha256=library_sha,actual_four_cases_verified=True,
      complete_artifact_manifest_rehashed=True,Graph_and_side_verified=True,
      original_BF16_proxy_passed=result['original_BF16_proxy_passed'],
      qualification='two actual N32/S4 calls as a numeric control only; not a performance denominator')
