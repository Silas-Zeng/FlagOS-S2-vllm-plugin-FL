#!/usr/bin/env python3
"""CPU source/ABI adapter checks. Mock handles never qualify actual device resources."""
import hashlib,json,sys,types
from pathlib import Path

import probe_biv150_native_k8_compact_n48_kpair_fullstep as p
import biv150_native_k8_compact_n48_kpair_fullstep_source as s


def main():
    plan=p.plan();assert plan['pending_fields']==p.pending() and plan['fullstep_execution_allowed']==(not p.pending())
    assert plan['pages']==44769 and plan['query_pitch']==plan['new_KV_pitch']==2560
    assert plan['warm_replays']==128 and plan['timing_replays']==64 and plan['unroll']==8
    assert p.ORDER==('old8k','kpair','kpair','old8k')
    body,proof=s.adapted_source();assert set(proof['seven_unchanged_body_hashes'])==set(s.PRESERVED)
    assert 'reader=' not in body and 'reader._gqa_decode' not in body and 'ops[arm](' in body
    assert 'torch.ops.load_library(str(path))' in body and 'pre[\'arm_schemas\'][arm]' in body
    import biv150_native_k8_compact_n48_shared6k_fullstep_source as previous113
    previous,previous_proof=previous113.adapted_source()
    alignment_line="            rec['default_cache_alignment']={arm:record_default_alignment(z['key'],z['scale'],z['value'],lambda t:dict(pointer=int(t.data_ptr()))) for arm,z in cases.items()}\n"
    assert body.count(alignment_line)==1
    inverse=body.replace(alignment_line,'').replace('cjj_native_k8_compact_n48_kpair_probe','cjj_native_k8_compact_n48_shared6k_probe').replace('kpair','new6k').replace('Kpair','New6k')
    assert inverse==previous,'full113 driver changed outside exact producer identity/arm labels/alignment observation'
    handles,hproof=s.handles_source()
    assert handles.replace('kpair','new6k')==previous113.handles_source()[0]
    ns=dict(vars(p.old),f=p.f);exec(compile(handles,'<CPU-only handle-contract mock>','exec'),ns)
    names=['key_source','value_source','slots','code_full','scale_full','value_full','key_row_stride','value_row_stride','key_rows','value_rows','slot_count','cache_pages']
    dtype=['*bf16','*bf16','*i64','*i8','*fp32','*bf16']+['i64']*6
    writerfn=types.SimpleNamespace(arg_names=names);reducerfn=types.SimpleNamespace(arg_names=[])
    writer=types.SimpleNamespace(encode_append=writerfn);base=types.SimpleNamespace(_reduce_partials=reducerfn)
    def fresh():
        return {role:types.SimpleNamespace(n_regs=32,n_spills=0,src=types.SimpleNamespace(fn=fn,signature=dict(zip(names,dtype)) if fn is writerfn else {})) for role,fn in
          [('initial_encoder',writerfn),('old8k_append',writerfn),('kpair_append',writerfn),('reducer',reducerfn)]}
    def check(h):
        rec={'actual_compiled':{k:{} for k in h}}
        ns['verify_actual_handles'](h,None,rec,writer,None,base)
        assert rec['stride2560_actual_JIT_and_resources_verified'] is True
    check(fresh());rejected=[]
    def reject(name,fn):
        try:fn()
        except (AssertionError,KeyError) as exc:rejected.append(dict(case=name,error=type(exc).__name__))
        else:raise AssertionError('invalid adapter input accepted: '+name)
    actual_pins=dict(p.NEW_CONSUMER)
    try:
        p.NEW_CONSUMER.update({k:None for k in actual_pins})
        reject('unknown_real114_consumer_evidence',lambda:p.actual_consumer_gate(None))
    finally:p.NEW_CONSUMER.update(actual_pins)
    h=fresh();h['tl_producer']=h['reducer'];reject('unexpected_TL_handle',lambda:check(h))
    h=fresh();h['kpair_append'].n_spills=1;reject('new_writer_runtime_spill',lambda:check(h))
    h=fresh();h['old8k_append'].n_regs=None;reject('old_writer_unknown_registers',lambda:check(h))
    h=fresh();h['kpair_append'].src.signature['key_row_stride']='i32';reject('wrong_2560_writer_typed_ABI',lambda:check(h))
    h=fresh();h['kpair_append'].src.fn=types.SimpleNamespace(arg_names=names);reject('foreign_writer_function_identity',lambda:check(h))
    alignment=p.contract.record_default_alignment(4096,4096,4096+4352,lambda pointer:dict(pointer=pointer))
    assert alignment['pointer_mod8']==dict(key=0,scale=0,value=0)
    reject('whole_cache_4B_non8B_unqualified',lambda:p.contract.record_default_alignment(4100,4100,4100+4352,lambda pointer:dict(pointer=pointer)))
    reject('wrong_V_offset',lambda:p.contract.record_default_alignment(4096,4096,4096+4360,lambda pointer:dict(pointer=pointer)))
    assert len(rejected)==8 and not {'torch','triton','numpy'}&set(sys.modules)
    print(json.dumps(dict(completed=True,errors=[],CPU_only=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      direct113_original_source_sha256=hashlib.sha256(previous.encode()).hexdigest(),
      complete_direct113_driver_inverse_byte_exact=True,actual_alignment_observation_only_addition=True,
      parent_source_pins=s.PINS,executed_fullstep_sha256=proof['executed_sha256'],seven_original_bodies_preserved=proof['seven_unchanged_body_hashes'],
      existing_consumer_gate_adapter=plan['candidate_consumer_gate_adapter'],remaining_JIT_adapter=hproof,
      mock_handle_positive_case_passed=True,mock_handles_are_not_actual_resources=True,refusals=rejected,
      actual_consumer_pins=dict(p.NEW_CONSUMER),fullstep_execution_allowed=plan['fullstep_execution_allowed'],
      GPU_executed=False,GPU_numeric_verified=False,Graph_verified=False,side_stream_verified=False,
      actual_resources_verified=False,performance_verified=False),indent=2,sort_keys=True))
if __name__=='__main__':main()
