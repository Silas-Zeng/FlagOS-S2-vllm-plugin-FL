#!/usr/bin/env python3
"""Small CPU preflight/policy/namespace/ABI refusals; no framework imports."""
import copy
import json
import sys

import bench_biv150_konly_q32_numeric as runner
import biv150_konly_numeric_admission as native
import biv150_konly_q32_numeric_contract as numeric


def reject(fn):
    try: fn()
    except (ValueError,KeyError): return
    raise AssertionError('Malformed numeric boundary accepted')


def run():
    pre,*_=runner.preflight();assert numeric.plan()['klens']==[6144]
    assert numeric.plan()['qlens']==[2048] and numeric.plan()['grid']==(65,2)
    assert pre['actual_mechanism_verified'] and not pre['actual_binding_verified']
    assert not pre['Graph_verified'] and not pre['side_stream_verified']
    baseline_bad=numeric.gates([1.1],[1.],[1.],[1.])
    assert not baseline_bad['baseline_hard_passed']
    candidate_bad=numeric.gates([1.],[1.02],[1.],[1.02])
    assert candidate_bad['baseline_hard_passed'] and not candidate_bad['candidate_numeric_passed']
    hard={k:True for k in ('finite','baseline_reference','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise')}
    reject(lambda:numeric.policy(hard,False,False))
    assert not numeric.policy(hard,False,True)['precision_gate']
    for name in hard:
        mutant=dict(hard,**{name:False});reject(lambda mutant=mutant:numeric.policy(mutant,False,True))
    reject(lambda:numeric.policy(dict(hard,unknown=True),True,True))
    # A real queried native host PID can differ from container PID. Never trust
    # namespace equality/token alone; exact queried singleton must persist.
    def admission(values):
        sequence=iter(values)
        return native.Admission(lambda:None,lambda:dict(commit='frozen'),query=lambda:next(sequence),
            identity=lambda:dict(pid=7,start_ticks=11),closed=lambda:True,initialize=lambda:object())
    owned=admission([[],[701],[701]]);token=owned.begin();owned.check('compile',token)
    assert owned.native==701 and owned.owner['pid']==7
    for pids in ([7],[702],[],[701,702],['701'],[None]):
        bad=admission([[],[701],pids]);key=bad.begin();reject(lambda bad=bad,key=key:bad.check('compile',key))
    bad=admission([[],[701],[701]]);bad.begin();reject(lambda:bad.check('compile','stale'))
    for pids in ([],[701,702],['701'],[None],[True]):
        bad=admission([[],pids]);reject(bad.begin)
    old=runner.actual.old.original()[2]
    runner.bind_record(old,old,'unhoisted')
    for field in ('hash','typed_ABI','files'):
        mutant=copy.deepcopy(old)
        if field=='hash': mutant[field]='unproven'
        elif field=='typed_ABI': mutant[field]={}
        else: mutant['files']['asm']['sha256']='unproven'
        reject(lambda mutant=mutant:runner.bind_record(mutant,old,'unhoisted'))
    mutant=copy.deepcopy(old);mutant['files']['ttir']['sha256']='unproven'
    assert 'ttir' in runner.differences(mutant,old)
    reject(lambda:runner.bind_record(mutant,old,'unhoisted'))
    assert not any(name in sys.modules for name in ('torch','triton','numpy'))
    return dict(cpu_only=True,geometry=numeric.plan(),mechanism_replayed=True,
                finite_baseline_failure_not_bypassable=True,unknown_ABI_or_IR_refused=True,
                native_namespace_distinction_tested=True,original_thresholds_preserved=True,
                framework_imported=False,actual_numeric_executed=False,Graph_verified=False,
                side_stream_verified=False,precision_gate=False,timing_attempted=False)


if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
