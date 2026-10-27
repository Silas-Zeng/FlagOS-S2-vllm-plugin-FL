#!/usr/bin/env python3
"""Rehash/extract42 original files and audit the witnessed post-hoist target IR."""
import argparse
import json
from pathlib import Path

import biv150_konly_q32_actual_hoist as actual


def rejected(fn):
    try: fn()
    except (ValueError,StopIteration,KeyError): return
    raise AssertionError('Malformed positive-hoist witness accepted')


def run(extract=None):
    files,summary,record=actual.original()
    ttgir,llir,assembly=(files['out/ir/candidate/actual.'+e].decode() for e in ('ttgir','llir','asm'))
    proof=actual.audit(ttgir,llir,assembly)
    # SSA reprinting is legitimate; opcode, operands and region placement are not.
    rejected(lambda:actual.audit(ttgir.replace('tt.precise_divf','arith.divf',1),llir,assembly,known_hash=False))
    rejected(lambda:actual.audit(ttgir.replace('tt.dot %6,','tt.dot %5,',1),llir,assembly,known_hash=False))
    rejected(lambda:actual.audit(ttgir,llir.replace('matrix.mad.i32x4.i8x8','matrix.mad.i32x4.other',1),assembly,known_hash=False))
    for number in (1739,2128,2251):
        rows=llir.splitlines(keepends=True);rows[number-1]=''
        rejected(lambda rows=rows:actual.audit(ttgir,''.join(rows),assembly,known_hash=False))
    loop=proof['ASM']['KV_loop_body'];rows=assembly.splitlines(keepends=True)
    for instruction in ('\tml_mov_b32 v32, 0x0\n','\tml_slb_load_b8 v32, v51, noreg, #0, null\n'):
        mutant=''.join(rows[:loop]+[instruction]+rows[loop:])
        rejected(lambda mutant=mutant:actual.asm_observation(mutant))
    for token in ('\tsl_barrier \n','v[32:33]'):
        # Explicit specialized operands/publish fence must be observed, not inferred from counts.
        if token=='v[32:33]':
            mutant=assembly.replace(token,'v[30:31]',1)
        else:
            mutant=assembly.replace(token,'',assembly.count(token))
        rejected(lambda mutant=mutant:actual.asm_observation(mutant))
    oldfiles,oldsummary,oldrecord=actual.old.original()
    rejected(lambda:actual.audit(*(oldfiles['out/ir/candidate/actual.'+e].decode() for e in ('ttgir','llir','asm'))))
    assert not summary['actual_I8_operand_hoist_verified'] and not oldsummary['actual_I8_operand_hoist_verified']
    assert not record['resources']['loaded_resource_fields_known']
    assert record['resources']['n_regs'] is None and record['resources']['n_spills'] is None
    assert (oldrecord['resources']['shared'],record['resources']['shared'])==(57344,33792)
    if extract is not None:
        target=Path(extract);target.mkdir(parents=True,exist_ok=False)
        for name,value in files.items():
            destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(value)
    return dict(archive_sha256=actual.ARCHIVE_SHA,original_raw_files=42,original_raw_bytes=3151371,
                original_owner=dict(pid=626927,start_ticks=343979040,state='Z'),
                raw_plus_manifest_files=43,raw_all_SHA_verified=True,CPU_rejected_faults=11,
                candidate_hash=record['hash'],baseline_is_accepted_service_main=False,
                original_driver_hoist_false_preserved=True,original_frontend_unsupported_preserved=True,
                old_resources=oldrecord['resources'],actual_resources=record['resources'],
                actual_mechanism=proof,capacity_reclaimed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract');args=parser.parse_args()
    print(json.dumps(run(args.extract),sort_keys=True,indent=2))
