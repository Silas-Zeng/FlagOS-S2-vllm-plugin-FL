"""CPU-only admission counterexamples. Never invokes ixsmi or imports Torch."""
import ast
import copy
import json
from pathlib import Path
import subprocess
from biv150_native_gqa_batch_admission import Admission as RealAdmission,parse_observation


def Admission(*args,**kwargs):
    return RealAdmission(*args,**kwargs,port_reader=lambda:dict(port9031_probe_code=111,port9031_closed=True))

ROOT=Path(__file__).resolve().parents[1]
OLD='dc9d451c977ee4f12cfb725def8742671373f4f2'
PATH='tools/bench_biv150_native_gqa_batch.py'


def reject(fn):
    try:fn()
    except RuntimeError:return
    raise AssertionError('bad admission accepted')


def snapshot(used=64,pids=''):
    return parse_observation(f'GPU0 memory {used}MiB / 32768MiB\nNo running processes',pids)


def run():
    owner=dict(pid=123,start_ticks=456,scope='CPU fixture container, not driver')
    reject(lambda:Admission(False,reader=lambda:None,identity_reader=lambda:owner))
    low=Admission(True,reader=lambda:snapshot(),identity_reader=lambda:owner)
    first=low.check('pre-CUDA');assert first['native_pid_visibility']=='unavailable'
    assert low.report['driver_host_pid'] is None and not low.report['exclusive_by_ixsmi']
    high=Admission(True,reader=lambda:snapshot(28554),identity_reader=lambda:owner)
    reject(lambda:high.check('pre-CUDA'));assert high.report['observations'][0]['memory_used_mib']==28554
    # A visible container-number match is not namespace/host ownership proof.
    for pids in ('999','123','123\n999'):
        a=Admission(True,reader=lambda:snapshot(10,pids),identity_reader=lambda:owner)
        reject(lambda:a.check('pre-CUDA'))
    for text in ('', 'No running processes','No running processes found.'):
        a=Admission(True,reader=lambda:snapshot(10,text),identity_reader=lambda:owner);a.check('pre-CUDA')
    visible=parse_observation('GPU0 64MiB / 32768MiB\nProcesses:\n| 0 999 python 64MiB |','')
    a=Admission(True,reader=lambda:visible,identity_reader=lambda:owner);reject(lambda:a.check('pre-CUDA'))
    reject(lambda:parse_observation('GPU memory unknown',''))
    reject(lambda:parse_observation('1MiB / 32768MiB','pid,unknown'))
    reject(lambda:parse_observation('40000MiB / 32768MiB',''))
    values=iter([snapshot(10),snapshot(900),snapshot(950),snapshot(2050)])
    a=Admission(True,reader=lambda:next(values),identity_reader=lambda:owner)
    a.check('pre-CUDA');a.check('post-tiny-initialization');a.check();reject(lambda:a.check())
    a=Admission(True,reader=lambda:snapshot(4097),identity_reader=lambda:owner);reject(lambda:a.check())
    a=Admission(True,reader=lambda:snapshot(10),identity_reader=lambda:owner)
    a.check('pre-CUDA');a.identity_reader=lambda:{**owner,'start_ticks':457};reject(lambda:a.check())
    def failure():raise RuntimeError('actual query failed')
    a=Admission(True,reader=failure,identity_reader=lambda:owner);reject(lambda:a.check('pre-CUDA'))
    old=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show',OLD+':'+PATH]));new=ast.parse((ROOT/PATH).read_bytes())
    # Entire actual numeric loop (not a reconstructed oracle) is bytecode-AST identical.
    def numeric(tree):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        return next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='n')
    assert ast.dump(numeric(old),include_attributes=False)==ast.dump(numeric(new),include_attributes=False)
    text=(ROOT/PATH).read_text();assert text.index("guard.check('pre-CUDA')")<text.index('        import torch')
    assert text.index("connect_ex(('127.0.0.1',9031))")<text.index("guard.check('pre-CUDA')")
    assert not any(name in __import__('sys').modules for name in ('torch','triton','numpy'))
    return dict(cpu_passed=True,original_numeric_loop_AST_identical=True,
        invisible_high_memory_rejected=True,invisible_low_memory_recorded_unavailable=True,
        visible_foreign_and_container_number_match_rejected=True,unknown_query_rejected=True,
        memory_jump_and_running_cap_rejected=True,pid_start_change_rejected=True,
        pre_CUDA_port_and_memory_guard_order_verified=True,
        Root_external_serial_exclusivity_required=True,ixsmi_exclusivity_proven=False,
        driver_host_pid_mapping_verified=False,GPU_executed=False,SSH_executed=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
