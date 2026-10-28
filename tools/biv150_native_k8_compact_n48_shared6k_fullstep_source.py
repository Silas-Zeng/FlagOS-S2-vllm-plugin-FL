"""Bounded identity adaptation: existing112 fullstep, actual2ad versus actual5cf6."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import biv150_native_k8_compact_n48_fullstep_source as old_source
import probe_biv150_native_k8_compact_n48_fullstep as old_frontend
from probe_biv150_native_k8_compact_v_overlap_fullstep import replace_exact

ROOT=old_source.ROOT
PINS={
 'tools/probe_biv150_native_k8_compact_n48_fullstep.py':'ad87f323c0b96b13920ab1fd725079a18dbc6153820db2b48efe02cb6d372443',
 'tools/biv150_native_k8_compact_n48_fullstep_source.py':'fa2b389b925fe6526c7cf42f179dab48794773386449481009815db6efbed9e9',
 'tools/biv150_native_k8_compact_n48_fullstep_fixture.py':'95968c46dc749b6559dc1be8945dedc2cf81fefcb1dfdc058086b2c196c3014f'}
LABELS=(("'tl'","'old8k'"),("'native'","'new6k'"),
 ('event_ratio_TL_over_Native','event_ratio_old8k_over_new6k'),
 ('host_wall_ratio_TL_over_Native','host_wall_ratio_old8k_over_new6k'),
 ('TL_GPU_full_encode','Old8k_GPU_full_encode'),('Native_GPU_full_encode','New6k_GPU_full_encode'))
PRESERVED=('timed','capture','reducer','poison','fingerprints','verify_inputs','step')


def sha(raw):return hashlib.sha256(raw).hexdigest()
def verify_pins():
    for path,pin in PINS.items():assert sha((ROOT/path).read_bytes())==pin,path


def adapted_source():
    verify_pins();original,parent=old_source.adapted_source();edits=[]
    def edit(a,b):edits.append((a,b))
    old_load="""torch.ops.load_library(str(a.k8_library))
        assert str(a.k8_library.resolve()) in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
        assert str(torch.ops.cjj_native_k8_compact_n48_s4_probe.produce.default._schema)==pre['candidate_actual_schema']
        result['actual_native_library']=dict(path=str(a.k8_library),sha256=f.sha(a.k8_library.read_bytes()),loaded_library_set_confirmed=True,actual_schema=pre['candidate_actual_schema'])"""
    edit(old_load,"""ops={}
        for arm,path,name in (('old8k',a.old_library,'cjj_native_k8_compact_n48_s4_probe'),('new6k',a.k8_library,'cjj_native_k8_compact_n48_shared6k_probe')):
            torch.ops.load_library(str(path))
            assert str(path.resolve()) in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
            op=getattr(torch.ops,name).produce
            assert str(op.default._schema)==pre['arm_schemas'][arm]
            ops[arm]=op
        result['actual_native_libraries']={arm:dict(path=str(path),sha256=f.sha(path.read_bytes()),loaded_library_set_confirmed=True,actual_schema=pre['arm_schemas'][arm]) for arm,path in (('old8k',a.old_library),('new6k',a.k8_library))}""")
    edit("        reader=module(materialize_reader(out),'cjj_fullstep_actual_compact_TL48')\n        result['actual_TL_reader_and_quant_helper']=verify_loaded_reader(reader)\n",'        # Both producers are the admitted Native SOs; no TL JIT is launched.\n')
    edit(',production_compact_reader=reader.__file__',",producer_implementations='two admitted Native SOs'")
    edit(old_source.nested(original,'producer'),"""def producer(arm):
                z=cases[arm]
                ops[arm](q,z['key'],z['scale'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634,s)""")
    edit('verify_actual_handles(handles,compiled,rec,writer,reader,base)','verify_actual_handles(handles,compiled,rec,writer,None,base)')
    # Keep CPU plan false, running-but-unconfirmed state unknown, completed device result true.
    edit('result.update(completed=True,numerical_verified=True','result.update(GPU_executed=True,completed=True,numerical_verified=True')
    changed,proof=replace_exact(original,edits)
    for x,y in LABELS:
        assert x in changed;changed=changed.replace(x,y)
    unlabelled=changed
    # Private op/setup already uses descriptive arm names; normalize only original
    # seven function bodies for equality, not their new explicit loading records.
    unchanged={}
    for name in PRESERVED:
        old=old_source.nested(original,name);new=old_source.nested(changed,name)
        for x,y in reversed(LABELS):new=new.replace(y,x)
        assert old==new,name;unchanged[name]=sha(old.encode())
    # The six declared edits are exact at their application boundary. Reapplying
    # labels to that bounded result must regenerate the complete executed body.
    bounded,_=replace_exact(original,edits)
    for x,y in LABELS:bounded=bounded.replace(x,y)
    assert bounded==unlabelled;ast.parse(changed)
    proof.update(parent_N48=parent,original_sha256=sha(original.encode()),executed_sha256=sha(changed.encode()),
      label_changes=LABELS,seven_unchanged_body_hashes=unchanged,
      GPU_math_changed=False,TL_runtime_producer_removed=True,two_actual_native_libraries=True,
      GPU_execution_flag_fix='CPU plan false; in-progress unknown; completed actual device result true')
    return changed,json.loads(json.dumps(proof))


def handles_source():
    verify_pins();original=inspect.getsource(old_frontend.verify_actual_handles)
    reader="""    reader_signature=typed_signature(handles['tl_producer'])
    assert [reader_signature[n] for n in ('segm_output_ptr','segm_max_ptr','segm_expsum_ptr','query_ptr','query_code_ptr','query_scale_ptr','key_scale_ptr','key_cache_ptr','value_cache_ptr','seq_lens_ptr','block_table_ptr')]==['*fp32']*3+['*bf16','*i8','*fp32','*fp32','*i8','*bf16','*i32','*i32']
"""
    expected="""    expected=dict(initial_encoder=writer.encode_append,tl_append=writer.encode_append,native_append=writer.encode_append,
      tl_producer=reader._gqa_decode_konly_i8,reducer=base._reduce_partials)"""
    replacement="""    expected=dict(initial_encoder=writer.encode_append,old8k_append=writer.encode_append,new6k_append=writer.encode_append,
      reducer=base._reduce_partials)"""
    edits=[(expected,replacement),(reader,'    # Both Native producer identities are admitted by their actual SO receipts.\n'),
      ("for role in ('tl_append','native_append'):","for role in ('old8k_append','new6k_append'):")]
    text,proof=replace_exact(original,edits)
    # The removal is explicit; all remaining gates are the exact source outside
    # those three changes, including typed writer ABI and runtime n_regs/spills.
    ast.parse(text)
    return text,proof
