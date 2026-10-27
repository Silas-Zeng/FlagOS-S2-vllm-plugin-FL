#!/usr/bin/env python3
"""Replay original target bytes and prepare one CPU I8-Q hoist; no JIT/device entry."""
import argparse
import json
from pathlib import Path
import re
import sys

import biv150_konly_q32_i8_hoist as hoist
import biv150_konly_prefill_compile_contract as contract


def reject(call):
    try: call()
    except (ValueError,AssertionError,StopIteration): return
    raise AssertionError('Malformed I8 hoist fixture accepted')


def run(emit=None,extract=None):
    files,summary,record=hoist.original()
    assert not summary['runtime_binding_qualified'] and not summary['precision_gate']
    assert not record['resources']['loaded_resource_fields_known']
    contract.check_descriptors(summary['plan'],summary['tensor_descriptors'])
    raw=files['out/ir/candidate/actual.ttgir'].decode()
    llvm=files['out/ir/candidate/actual.llir'].decode()
    assembly=files['out/ir/candidate/actual.asm'].decode()
    changed,proof=hoist.transform(raw)
    old=hoist.llvm_observation(llvm)
    assert old['native_I8_QK_calls']==16 and old['BF16_PV_calls']==32
    assert len(old['Q_shared_reads_inside_KV'])==16 and not old['all_Q_operand_reads_outside_KV']
    assert len(old['RN32_divisions_dominating_KV'])==48
    assert assembly.count('ml_matrix_mad_i32_i8_m16n16k32')==16
    assert assembly.count('ml_matrix_mad_f32_bf16_m16n16k16')==32
    lines=raw.splitlines(keepends=True);before=proof['before'];qload=lines[before['Q_operand_load']-1]
    qalloc=lines[before['q_allocation']-1];qcast=lines[before['q_cast']-1]
    qglobal=lines[before['q_global']-1];qk=lines[before['QK']-1];pv=lines[before['PV']-1]
    mutations=[
        raw.replace(qload,qload.replace('kWidth = 4','kWidth = 2'),1),
        raw.replace(qload,qload.replace('256x128xi8','256x128xbf16'),1),
        raw.replace(qload,qload+qload,1),
        raw.replace(qload,'  '+qload,1),
        raw.replace(qalloc,qalloc+'    ttg.local_dealloc %5 : !ttg.memdesc<256x128xi8, #shared, #smem>\n',1),
        raw.replace(qalloc,qalloc+'    ttg.local_store %4, %5 : tensor<256x128xi8, #blocked>\n',1),
        raw.replace(qk,qk.replace('%10,','%Q_156,'),1),
        raw.replace(qk,qk.replace('%cst_15, inputPrecision','%arg27, inputPrecision'),1),
        raw.replace(qk,qk.replace('-> tensor<256x64xi32','-> tensor<256x64xf32'),1),
        raw.replace(pv,pv.replace('xbf16','xf32'),1),
        raw.replace('kernel_cjj_konly_q32','kernel_cjj_konly_multi2'),
        raw.replace('tt.precise_divf','arith.divf',1),
        raw.replace(qcast,'',1).replace(qload,qcast+qload,1),
        raw.replace(qglobal,'',1).replace(qload,qglobal+qload,1),
        raw.replace(qload,qload+'      %extra = ttg.convert_layout %10 : tensor<256x128xi8, #blocked>\n',1),
        raw.replace('    %acc:3 = scf.for ','    %other = scf.for %x = 0 to 1 step 1 {\n    }\n    %acc:3 = scf.for ',1),
    ]
    for mutant in mutations:
        reject(lambda mutant=mutant:hoist.grammar(mutant))
    # Math outside the small parsed span is protected by the exact original hash.
    reject(lambda:hoist.transform(raw.replace('math.exp','math.exp2',1)))
    reject(lambda:hoist.transform(changed))
    reject(lambda:hoist.transform(files['out/ir/baseline/actual.ttgir'].decode()))
    rejected=len(mutations)+3
    llvm_mutants=[
        llvm.replace('matrix.mad.i32x4.i8x8','matrix.mad.i32x4.other',1),
        llvm.replace('matrix.mad.f32x4.bf16x4','matrix.mad.f32x4.other',1),
        llvm.replace('load <2 x i8>, ptr addrspace(3) %1651','load <4 x i8>, ptr addrspace(3) %1651',1),
        llvm.replace('!57 = !DILocation(line: 238,','!57 = !DILocation(line: 239,',1),
        llvm.replace('label %.lr.ph108','label %unknown_target',1),
        llvm.replace('fdiv float','other.float',1),
    ]
    first=re.search(r'i8x8\(<8 x i8> (%\w+), <8 x i8> (%\w+)',llvm)
    llvm_mutants.append(llvm[:first.start()]+llvm[first.start():].replace(first.group(0),
                       f'i8x8(<8 x i8> {first.group(2)}, <8 x i8> {first.group(1)}',1))
    llvm_mutants.append(llvm[:first.start()]+llvm[first.start():].replace(first.group(0),
                       f'i8x8(<8 x i8> zeroinitializer, <8 x i8> {first.group(2)}',1))
    for mutant in llvm_mutants:
        assert mutant!=llvm
        reject(lambda mutant=mutant:hoist.llvm_observation(mutant));rejected+=1
    # A source/TTGIR-only success must never reuse the old failing target LLVM.
    mixed=hoist.audit(changed,llvm,assembly)
    assert not mixed['actual_I8_hoist_verified'] and not mixed['ready_for_numeric_or_timing']
    assert 'LLVM Q operand shared reads remain inside KV loop' in mixed['errors']
    raw_audit=hoist.audit(raw,llvm,assembly)
    assert not raw_audit['actual_I8_hoist_verified']
    if emit:
        with emit.open('x') as out: out.write(changed)
        sidecar=dict(proof=proof,original_kernel_hash=record['hash'],driver_commit=hoist.DRIVER,
                     reader_commit=hoist.READER,required_typed_ABI=record['typed_ABI'],
                     required_requested_options=record['requested_options'],required_arguments=record['arguments'],
                     original_source_files=summary['source_files'],target=summary['target'],
                     actual_override_compile_completed=False,ready_for_numeric_or_timing=False)
        with emit.with_suffix(emit.suffix+'.json').open('x') as out: json.dump(sidecar,out,indent=2,sort_keys=True);out.write('\n')
    if extract:
        extract.mkdir(parents=True,exist_ok=False)
        for name,value in files.items():
            destination=extract/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(value)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(cpu_only=True,passed=True,archive_sha256=hoist.ARCHIVE_SHA,all40_original_hashes_checked=True,
                owner=dict(pid=626170,start_ticks=343488382,same_start=True,state='Z'),
                original_driver_commit=hoist.DRIVER,original_reader_commit=hoist.READER,
                original_completed=True,original_hoist_verified=False,original_registers_spills_unknown=True,
                transform=proof,original_LLVM=old,old_LLVM_after_CPU_TTGIR_move=mixed,rejected_fixtures=rejected,
                source_reader_ABI_driver_d6_unchanged=True,actual_override_compile_completed=False,
                ASM_hoist_auditor_supported=False,actual_I8_hoist_verified=False,runtime_binding_qualified=False,
                precision_gate=False,performance_verified=False,service_installed=False,capacity_reclaimed=False,
                kernel_launches=0,JIT_executed=False,torch_imported=False,triton_imported=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path);parser.add_argument('--emit-override',type=Path)
    parser.add_argument('--extract-original',type=Path)
    args=parser.parse_args();text=json.dumps(run(args.emit_override,args.extract_original),indent=2,sort_keys=True)+'\n'
    if args.out:
        with args.out.open('x') as out: out.write(text)
    print(text,end='')


if __name__=='__main__': main()
