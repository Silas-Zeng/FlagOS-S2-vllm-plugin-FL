#!/usr/bin/env python3
"""Stdlib contracts/faults for the bounded eager entry; no device or framework."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import bench_biv150_konly_2d128_numeric as entry
import biv150_konly_2d128_actual as actual
import biv150_konly_2d128_numeric_admission as admission
import biv150_konly_2d128_numeric_contract as numeric


def run():
    payload=entry.preflight();files,proof,raw=actual.original();accepted=[];rejected=[]
    for role in ('baseline','candidate'):
        record=next(r for r in raw['records'] if r['label']==role)
        ir={e:files['out/ir/'+role+'/actual.'+e] for e in ('source','ttir','ttgir','llir','asm','cubin')}
        entry.verify_compilation(record,record,role,ir,ir);accepted.append(role)
        for field in ('typed_ABI','actual_options','jit_version','hash'):
            bad=deepcopy(record);bad[field]={'fault':True}
            try:entry.verify_compilation(bad,record,role,ir,ir)
            except ValueError:rejected.append(role+'_'+field)
            else:raise AssertionError('binding fault accepted')
    hard={n:True for n in ('finite','baseline_reference','CPU_codec_cache_bytes','output_canaries','read_only_inputs','actual_IR','quantized_implementation')}
    numeric.policy(hard,True,False);numeric.policy(hard,False,True)
    for name in hard:
        try:numeric.policy(dict(hard,**{name:False}),True,True)
        except ValueError:rejected.append('hard_'+name)
        else:raise AssertionError('hard gate bypass')
    try:numeric.policy(hard,False,False)
    except ValueError:rejected.append('BF16_failure_requires_diagnostic')
    else:raise AssertionError('precision incorrectly accepted')
    g=numeric.gates([1.],[1.02],[1.],[1.02])
    actual.need(g['quantized_implementation_passed'] and not g['original_BF16_quality_passed'] and not g['candidate_numeric_passed'],'separate approximation/implementation gates')
    g=numeric.gates([1.],[float('nan')],[1.],[1.])
    actual.need(not g['quantized_implementation_passed'] and not g['original_BF16_quality_passed'],'nonfinite hard gate')
    # Mock the actual container/host-PID protocol, not the CUDA initializer.
    state=dict(identity=dict(pid=17,start_ticks=123),native=[],closed=True)
    def initialize():state['native']=[456];return object()
    def make():return admission.Admission(lambda:None,lambda:('frozen-source',),query=lambda:list(state['native']),
        identity=lambda:dict(state['identity']),closed=lambda:state['closed'],initialize=initialize)
    a=make();token=a.begin()
    for phase in admission.PHASES:a.check(phase,token)
    actual.need(a.state=='completed','full exact phase order')
    for name in ('foreign_PID','port_open','stale_owner','wrong_phase','wrong_token'):
        state.update(native=[],closed=True,identity=dict(pid=17,start_ticks=123));a=make();token=a.begin()
        if name=='foreign_PID':state['native']=[456,789]
        elif name=='port_open':state['closed']=False
        elif name=='stale_owner':state['identity']['start_ticks']=124
        try:a.check('candidate0' if name=='wrong_phase' else 'compile','bad' if name=='wrong_token' else token)
        except ValueError:rejected.append(name)
        else:raise AssertionError('admission fault accepted')
    actual.need(not {'torch','triton','numpy'}.intersection(sys.modules),'CPU checker imported framework')
    tree=ast.parse(Path(entry.__file__).read_text())
    actual.need(not any(isinstance(n,ast.Import) and any(a.name=='time' for a in n.names) for n in ast.walk(tree)),'no timing import')
    return dict(CPU_only=True,plan=numeric.plan(),preflight=payload[0],real_record_binding_accepted=accepted,
        rejected=rejected,admission_phases=admission.PHASES,original_BF16_and_quantized_gates_separate=True,
        expected_reader_launches=4,expected_encoder_launches=0,actual_I8_operand_hoist_verified=False,
        GPU_executed=False,JIT_executed=False,eager_numeric_passed=False,precision_gate=False,
        full_numerical_matrix_verified=False,model_or_service_verified=False,performance_measured=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);a=p.parse_args();result=run()
    if a.out:a.out.write_bytes(actual.encode(result))
    print(json.dumps(dict(CPU_contract_passed=True,rejections=len(result['rejected']),GPU_executed=False)))


if __name__=='__main__':main()
