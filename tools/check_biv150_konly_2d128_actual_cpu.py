#!/usr/bin/env python3
"""Replay PR87 actual compile evidence and reject semantic IR mutations; CPU only."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import biv150_konly_2d128_actual as actual


def run():
    files,summary,raw=actual.original(); manifest=json.loads((actual.DIRECTORY/'archive-manifest.json').read_text())
    actual.need(actual.sha((actual.DIRECTORY/'raw-v1.tar.gz').read_bytes())==manifest['archive_sha256'],'archive SHA')
    actual.need(actual.sha(actual.encode(summary))==manifest['summary_sha256'],'recomputed summary SHA')
    actual.need((actual.DIRECTORY/'summary.json').read_bytes()==actual.encode(summary),'byte-identical actual summary')
    actual.need(manifest['members']==[dict(file=n,bytes=len(b),sha256=actual.sha(b)) for n,b in sorted(files.items())],'archive member SHA list')
    texts={e:files['out/ir/candidate/actual.'+e].decode() for e in ('ttgir','llir','asm')}
    rejected={}
    for name in ('Q_operand_outside_loop','scale_offset','scale_mask','native_I8','BF16_PV','Q_byte_read','LLVM_Q_origin'):
        t=dict(texts)
        if name=='Q_operand_outside_loop':
            ls=t['ttgir'].splitlines(True);i=next(i for i,s in enumerate(ls) if '%10 = ttg.local_load %5' in s);row=ls.pop(i)
            j=next(i for i,s in enumerate(ls) if '%acc:3 = scf.for' in s);ls.insert(j,row);t['ttgir']=''.join(ls)
        elif name=='scale_offset':t['ttgir']=t['ttgir'].replace('%cst_17 = arith.constant dense<1024>','%cst_17 = arith.constant dense<1023>')
        elif name=='scale_mask':t['ttgir']=t['ttgir'].replace('%key_code_scale_266, %tile_mask_211, %cst_19','%key_code_scale_266, %tile_mask_211, %cst_18')
        elif name=='native_I8':t['asm']=t['asm'].replace('ml_matrix_mad_i32_i8_m16n16k32','ml_matrix_mad_f32_bf16_m16n16k16',1)
        elif name=='BF16_PV':t['ttgir']=t['ttgir'].replace('%acc_300 = tt.dot %acc_298, %V_load_299','%acc_300 = tt.dot %acc_299, %V_load_299').replace('128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma1','128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1')
        elif name=='Q_byte_read':t['asm']=t['asm'].replace('ml_slb_load_b8 ','ml_slb_load_b16 ',1)
        else:t['llir']=t['llir'].replace('load <2 x i8>, ptr addrspace(3)','load <2 x i16>, ptr addrspace(3)',1)
        try:actual.mechanism(*(t[e] for e in ('ttgir','llir','asm')),known=False)
        except (ValueError,StopIteration) as exc:rejected[name]=str(exc)
        else:raise AssertionError('accepted semantic IR fault '+name)
    return dict(summary=summary,semantic_IR_faults_rejected=rejected,GPU_executed=False,JIT_executed=False,
                source_and_ABI_changed=False,precision_gate=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);a=p.parse_args();result=run()
    if a.out:a.out.write_bytes(actual.encode(result))
    print(json.dumps(dict(actual_compile_replay_passed=True,semantic_faults=len(result['semantic_IR_faults_rejected']),
                         actual_I8_operand_hoist_verified=False,GPU_executed=False)))


if __name__=='__main__':main()
