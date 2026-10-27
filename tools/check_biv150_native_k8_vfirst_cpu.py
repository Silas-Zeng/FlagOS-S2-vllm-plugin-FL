#!/usr/bin/env python3
"""No frameworks: prove one source-order edit plus isolated names, not scheduling."""
import ast
import json
import sys
import biv150_native_k8_vfirst_factory as f

def run():
    proof=f.verify();text=f.path('qk.cu').read_text();rejected=[]
    for name,bad in [('wait_count',text.replace('__ivcorex_sl_waitcnt(8ULL)','__ivcorex_sl_waitcnt(7ULL)')),
                     ('P_round',text.replace('__float2bfloat16_rn(p0)','__float2bfloat16_rn(p0*.99f)')),
                     ('scale_base',text.replace('page)*4096+1024+col','page)*4096+2048+col')),
                     ('restore_original_issue_order',text.replace(f.NEW,f.OLD)),
                     ('K_byte_layout',text.replace('panel*1024+bank*256','panel*1024+bank*252'))]:
        assert bad!=text
        try:f.inverse(bad,'qk.cu')
        except AssertionError:rejected.append(name)
        else:raise AssertionError('mutant accepted: '+name)
    adapters={kind:f.host_source(kind)[1] for kind in ('compile','numeric','fullstep')}
    for kind in adapters:ast.parse(f.host_source(kind)[0])
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,source_proof=proof,host_adapters=adapters,rejected=rejected,
                actual_compilation_or_IR_or_device=False,precision_gate=False)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
