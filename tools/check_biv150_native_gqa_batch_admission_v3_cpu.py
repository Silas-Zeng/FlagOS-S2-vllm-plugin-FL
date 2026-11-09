"""CPU replay of actual rejected v1 and controlled-window admission faults."""
import copy
import json
from pathlib import Path
from biv150_native_gqa_batch_admission import Admission,parse_observation
from check_biv150_native_gqa_batch_admission_cpu import run as prior_checks

ROOT=Path(__file__).resolve().parents[1]


def reject(fn):
    try:fn()
    except RuntimeError:return
    raise AssertionError('bad context-window admission accepted')


def snapshot(memory=68,pids=''):
    return parse_observation(f'{memory}MiB / 32768MiB\nNo running processes',pids)


def run():
    assert prior_checks()['original_numeric_loop_AST_identical']
    old=json.loads((ROOT/'docs/evidence/native-gqa-batch-coverage-20261028/v1-review/numeric-v1-summary.json').read_text())
    assert not old['completed'] and not old['passed'] and old['records']==[] and old['GPU_executed']
    assert old['diagnostic_admission']['tiny_cuda_initializations']==1
    actual=old['diagnostic_admission']['observations'];owner=old['diagnostic_admission']['container_process_identity']
    keys=('memory_used_mib','memory_total_mib','visible_native_pids','native_query_pids','full_table_native_pids','native_pid_visibility','no_visible_pid_is_idle_proof')
    states=[{k:x[k] for k in keys} for x in actual]
    def new(values):
        it=iter(copy.deepcopy(values))
        return Admission(True,reader=lambda:next(it),identity_reader=lambda:owner,
                         port_reader=lambda:dict(port9031_probe_code=111,port9031_closed=True))
    a=new([*states,states[-1]])
    a.check('pre-CUDA');a.check('post-tiny-initialization');a.check()
    assert a.report['context_window_candidates_unverified']==[2554078]
    assert a.report['driver_host_pid'] is None and not a.report['driver_host_pid_mapping_verified']
    assert not a.report['candidates_are_owned_process_proof'] and not a.report['candidates_rule_out_races']
    # Empty and singleton windows are both observational, never ownership proof.
    empty=new([snapshot(),snapshot(206),snapshot(206)])
    empty.check('pre-CUDA');empty.check('post-tiny-initialization');empty.check()
    assert empty.report['context_window_candidates_unverified']==[]
    bad=0
    for p in ('2554078',str(owner['pid'])):
        x=new([snapshot(68,p)]);reject(lambda:x.check('pre-CUDA'));bad+=1
    for post in (snapshot(206,'1\n2'),snapshot(4097),snapshot(2000,'1')):
        x=new([snapshot(),post]);x.check('pre-CUDA');reject(lambda:x.check('post-tiny-initialization'));bad+=1
    for first,later in [('1','2'),('1',''),('','1')]:
        x=new([snapshot(),snapshot(206,first),snapshot(206,later)])
        x.check('pre-CUDA');x.check('post-tiny-initialization');reject(lambda:x.check());bad+=1
    x=new([snapshot(206,'1')]);reject(lambda:x.check('post-tiny-initialization'));bad+=1
    x=new([snapshot(),snapshot(206,'1'),snapshot(206,'1')]);x.check('pre-CUDA');x.check('post-tiny-initialization')
    reject(lambda:x.check('post-tiny-initialization'));bad+=1
    x=new([snapshot(),snapshot(206,'1'),snapshot(206,'1')]);x.check('pre-CUDA');x.check('post-tiny-initialization')
    x.port_reader=lambda:dict(port9031_probe_code=0,port9031_closed=False);reject(lambda:x.check());bad+=1
    x=new([snapshot(),snapshot(206,'1'),snapshot(206,'1')]);x.check('pre-CUDA');x.check('post-tiny-initialization')
    x.identity_reader=lambda:{**owner,'start_ticks':owner['start_ticks']+1};reject(lambda:x.check());bad+=1
    assert not any(n in __import__('sys').modules for n in ('torch','triton','numpy'))
    return dict(cpu_passed=True,prior_admission_and_numeric_AST_checks_passed=True,
        replayed_v1_observations_only=True,real_new_device_run_performed=False,
        original_v1_rejection_and_zero_attention_preserved=True,
        actual_v1_window_candidate_replayed=[2554078],candidate_owned_identity_claimed=False,
        driver_host_PID_mapping_verified=False,Root_external_serial_exclusivity_required=True,
        empty_window_supported=True,hard_transition_mutants_rejected=bad,
        math_or_numeric_gate_changed=False,GPU_executed=False,SSH_executed=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
