"""Reversible adaptation of the existing compact fullstep, not new GPU math."""
import ast
import hashlib
import inspect
import subprocess
from pathlib import Path

import probe_biv150_native_k8_compact_fullstep as old
from probe_biv150_native_k8_compact_v_overlap_fullstep import replace_exact

ROOT=old.ROOT
PRODUCTION='fc25c86a13fc273bb7368b83af4ca264524b4e12'
READER='vllm_fl/patches/biv150_konly_compact_decode.py'
READER_SHA='ea955c009c7b3810ac6e6332ca4ac3d01ef535281f4332bcc708f016aea5e8b7'
HELPER='vllm_fl/patches/biv150_konly_encoded_writer.py'
HELPER_SHA='16afc46585f950f4b9c06c831b0e0dbf5515ee941a476a9ed300c492ebf947f9'


def production_bytes():
    raw=old.blob(PRODUCTION,READER)
    assert old.sha(raw)==READER_SHA and old.sha((ROOT/HELPER).read_bytes())==HELPER_SHA
    assert old.blob(PRODUCTION,HELPER)==(ROOT/HELPER).read_bytes()
    return raw


def materialize_reader(out):
    path=out/'frozen-sources'/Path(READER).name
    assert path.parent.is_dir() and not path.exists();path.write_bytes(production_bytes());return path


def nested(source,name):
    tree=ast.parse(source);run=tree.body[0]
    nodes=[n for n in ast.walk(run) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1;node=nodes[0]
    return ast.get_source_segment(source,node)


def adapted_source():
    source,parent_proof=old.adapted_source();edits=[]
    def edit(a,b):edits.append((a,b))
    edit('pre=require_consumer(a.consumer_summary,a.consumer_proof_sha256)','pre=actual_consumer_gate(a)')
    edit('assert f.sha(a.k8_library.read_bytes())==COMPACT_LIBRARY and f.sha(a.bf16_library.read_bytes())==f.BF16_SHA','assert f.sha(a.k8_library.read_bytes())==f.LIBRARY_SHA')
    edit('result=dict(compact_plan(),actual_consumer=pre,completed=False,errors=[],records=[],performance_acceptance=False)',
      'result=dict(compact_plan(),actual_consumer=pre,completed=False,errors=[],records=[],performance_acceptance=False)\n    save_frontend_sources(out,result)')
    edit('for path in (a.bf16_library,a.k8_library):torch.ops.load_library(str(path))',"torch.ops.load_library(str(a.k8_library))\n        assert str(a.k8_library.resolve()) in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}\n        assert str(torch.ops.cjj_native_k8_compact_n48_s4_probe.produce.default._schema)==pre['candidate_actual_schema']\n        result['actual_native_library']=dict(path=str(a.k8_library),sha256=f.sha(a.k8_library.read_bytes()),loaded_library_set_confirmed=True,actual_schema=pre['candidate_actual_schema'])")
    edit("        support=module(f.ROOT/'tools/kernels/biv150_pair_append_support.py','cjj_fullstep_94_support')\n        writer=module(materialize(out,'writer'),'cjj_fullstep_compact91_writer')",
      "        writer=module(materialize(out,'writer'),'cjj_fullstep_compact91_writer')")
    edit(",support94=support.__file__",",production_compact_reader=reader.__file__")
    edit("contract=compact_contract(module(f.ROOT/'tools/biv150_konly_strided_writer_contract.py','cjj_fullstep_83_contract'))",
      "contract=compact_contract(module(f.ROOT/'tools/biv150_konly_strided_writer_contract.py','cjj_fullstep_83_contract'))\n        reader=module(materialize_reader(out),'cjj_fullstep_actual_compact_TL48')\n        result['actual_TL_reader_and_quant_helper']=verify_loaded_reader(reader)")
    edit("nk=inputviews['new_k'];nv=inputviews['new_v'];","nk=inputviews['new_k'].as_strided((n,2,128),(f.SOURCE_PITCH,128,1));nv=inputviews['new_v'].as_strided((n,2,128),(f.SOURCE_PITCH,128,1));")
    edit("result['records'].append(rec);checkpoint()","result['records'].append(rec);rec['page_budget']=f.budget(f.LIVE[case],length);rec['phase_page_proofs']={str(p):{k:v for k,v in f.host_plan(case,p).items() if k!='table'} for p in (0,1,2)};checkpoint()")
    edit("                page_bytes=16384 if arm=='bf16' else 12544\n                full=","                page_bytes=12544\n                full=")
    old_views="""if arm=='bf16':
                    viewed=resident.view(torch.bfloat16).reshape(f.PAGES,2,16,2,128);key,value=viewed.unbind(1);scale=dummy
                else:
                    key=resident.view(torch.int8).as_strided((f.PAGES,16,2,128),(12544,256,128,1));scale=resident.view(torch.float32)
                    value=full.view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(6272,256,128,1),storage_offset=(f.GUARD+4352)//2)"""
    edit(old_views,"""key=resident.view(torch.int8).as_strided((f.PAGES,16,2,128),(12544,256,128,1));scale=resident.view(torch.float32)
                value=full.view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(6272,256,128,1),storage_offset=(f.GUARD+4352)//2)""")
    edit("if arm=='k8':assert key.data_ptr()",'assert key.data_ptr()')
    reset=nested(source,'reset')
    edit(reset,"""def reset(arm):
                arenas[arm]['raw'].fill_(165)
                handles['initial_encoder']=writer.encode_append[(f.PAGES*16,2)](initial_values['key'],initial_values['value'],initial_values['slots'],arenas[arm]['raw'].view(torch.int8),cases[arm]['scale'],arenas[arm]['raw'].view(torch.bfloat16),256,256,f.PAGES*16,f.PAGES*16,f.PAGES*16,f.PAGES,num_warps=1,num_stages=1)""")
    edit("return contract.qualify(descriptor(nk),descriptor(nv),descriptor(slots),descriptor(arenas['k8']['raw']),f.PAGES,contract.SlotSnapshot(tuple(int(x) for x in z['slots']),'component-owned-cpu-preupload',p))",
      "return {arm:contract.qualify(descriptor(nk),descriptor(nv),descriptor(slots),descriptor(arena['raw']),f.PAGES,contract.SlotSnapshot(tuple(int(x) for x in z['slots']),'component-owned-cpu-preupload',p)) for arm,arena in arenas.items()}")
    edit(nested(source,'append'),"""def append(arm):
                z=cases[arm];handles[arm+'_append']=writer.encode_append[(n,2)](nk,nv,slots,z['key'],z['scale'],arenas[arm]['raw'].view(torch.bfloat16),nk.stride(0),nv.stride(0),nk.shape[0],nv.shape[0],slots.shape[0],f.PAGES,num_warps=1,num_stages=1)""")
    edit(nested(source,'producer'),"""def producer(arm):
                z=cases[arm]
                if arm=='bf16':handles['tl_producer']=reader._gqa_decode_konly_i8[(n,2,s)](**tl_arguments(q,z,lens,table,s))
                else:torch.ops.cjj_native_k8_compact_n48_s4_probe.produce(q,z['key'],z['scale'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634,s)""")
    edit(nested(source,'compiled'),"""def compiled(k):
            return compiled_receipt(k,blob)""")
    edit("z=arenas[arm];page_bytes=16384 if arm=='bf16' else 12544","z=arenas[arm];page_bytes=12544")
    edit("initial=cold[start:start+count] if arm=='bf16' else packed[start:start+count]","initial=packed[start:start+count]")
    edit("initial.view(np.uint8).reshape(count,16384).copy() if p is None else f.patched_chunk(arm,start,initial,controls[p])","initial.copy() if p is None else f.patched_chunk(arm,start,initial,controls[p])")
    edit("expected=compact_chunk(expected) if arm=='k8' else expected","expected=compact_chunk(expected)")
    edit("gold[arm,p]=f.golden(cold,packed,controls[p],s,arm)","gold[arm,p]=f.golden(cold,packed,controls[p],s,arm) if arm=='bf16' else gold['bf16',p]")
    # Validate both stride2560 writer handles; retain the old exact typed ABI.
    typed_start=source.index('            signature={};names=')
    typed_end=source.index('            # Dense state restored',typed_start)
    old_typed=source[typed_start:typed_end]
    edit(old_typed,"""            verify_actual_handles(handles,compiled,rec,writer,reader,base)
""")
    # Labels are changed after the address/producer adaptations; this does not
    # turn a BF16 control into an encoded reader implicitly.
    changed,proof=replace_exact(source,edits)
    labels=[("'bf16'","'tl'"),("'k8'","'native'"),('event_ratio_BF16_over_K8','event_ratio_TL_over_Native'),('host_wall_ratio_BF16_over_K8','host_wall_ratio_TL_over_Native'),('BF16_GPU_pack','TL_GPU_full_encode'),('K8_GPU_full_encode','Native_GPU_full_encode')]
    before=changed
    for old_,new in labels:
        assert old_ in changed and new not in changed;changed=changed.replace(old_,new)
    restored=changed
    for old_,new in reversed(labels):restored=restored.replace(new,old_)
    assert restored==before;ast.parse(changed)
    proof.update(parent_compact=parent_proof,arm_label_renames=labels,
      label_inverse_byte_exact=True,final_source_sha256=old.sha(changed.encode()),GPU_math_changed=False)
    return changed,proof
