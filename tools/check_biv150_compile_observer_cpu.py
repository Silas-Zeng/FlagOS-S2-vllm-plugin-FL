"""CPU-only actual callable-identity/monitoring/fork/thread/interval checks."""
import importlib.util,json,tempfile,os,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('observer',ROOT/'biv150_compile_observer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as temp:
    destination=Path(temp);sentinel=object();calls=[];original_error=RuntimeError('original exception')
    def compile(src,options=None):
        calls.append((src,options))
        if src.name=='fail':raise original_error
        if src.name=='recursive':compile(SimpleNamespace(name='inner'))
        return sentinel
    package=SimpleNamespace(compile=compile);implementation=SimpleNamespace(compile=compile)
    already_bound=package.compile;code=compile.__code__;filename=code.co_filename
    handle=m.Handle(package,implementation,destination,{'cpu_fixture':True})
    try:
        late_bound=package.compile
        assert package.compile is implementation.compile is already_bound is late_bound is compile
        assert compile.__code__ is code and compile.__code__.co_filename==filename
        assert late_bound(SimpleNamespace(name='mm_kernel'),options={'num_warps':8,'num_stages':2}) is sentinel
        assert calls[-1][1]=={'num_warps':8,'num_stages':2}
        try:late_bound(SimpleNamespace(name='fail'))
        except RuntimeError as error:assert error is original_error
        else:raise AssertionError('exception swallowed')
        assert already_bound(SimpleNamespace(name='already_bound_observed')) is sentinel
        assert compile(SimpleNamespace(name='recursive')) is sentinel
        # The inherited code-local monitor must work in a real child process.
        child=os.fork()
        if child==0:
            try:
                assert compile(SimpleNamespace(name='_small_o_kernel_fork_child')) is sentinel
                os._exit(0)
            except BaseException:os._exit(101)
        assert os.waitpid(child,0)[1]==0
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert all(value is sentinel for value in pool.map(lambda i:compile(SimpleNamespace(name='thread-'+str(i))),range(24)))
        records=[json.loads(line) for p in destination.glob('*.jsonl') for line in p.read_text().splitlines()]
        calls_recorded=[r for r in records if r['event']=='compile_call']
        assert len(calls_recorded)==30,len(calls_recorded)
        assert any(r['kernel']=='fail' and r['error_type']=='RuntimeError' for r in calls_recorded)
        assert len([r for r in calls_recorded if r['kernel'].startswith('thread-')])==24
        assert any(r['event']=='installation' and r['pid']==child and r.get('inherited_from_pid')==os.getpid() for r in records)
        assert any(r['kernel']=='_small_o_kernel_fork_child' and r['pid']==child for r in calls_recorded)
        assert all(r['end_ns']>=r['start_ns'] and r['elapsed_ns']==r['end_ns']-r['start_ns'] for r in calls_recorded)
    finally:result=handle.uninstall()
    assert result=={'aliases_unchanged':[True,True],'record_errors':0},result
    count=sum(len(p.read_text().splitlines()) for p in destination.glob('*.jsonl'))
    assert compile(SimpleNamespace(name='after_uninstall')) is sentinel
    assert sum(len(p.read_text().splitlines()) for p in destination.glob('*.jsonl'))==count
    try:m.Handle(package,SimpleNamespace(compile=lambda x:x),destination,{})
    except RuntimeError:pass
    else:raise AssertionError('alien alias accepted')
    assert package.compile is compile
with tempfile.TemporaryDirectory() as temp:
    package=SimpleNamespace(compile=compile);implementation=SimpleNamespace(compile=compile)
    emit=m.Handle.emit;slots={i:sys.monitoring.get_tool(i) for i in (3,4)}
    def fail_emit(*args):raise OSError('record unavailable')
    m.Handle.emit=fail_emit
    try:m.Handle(package,implementation,temp,{})
    except OSError:pass
    else:raise AssertionError('installation write failure ignored')
    finally:m.Handle.emit=emit
    assert {i:sys.monitoring.get_tool(i) for i in (3,4)}==slots
    assert package.compile is implementation.compile is compile
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'compile-1.jsonl';records=[{'event':'installation','pid':1}]
    for a,b,k in [(0,10,'mm'),(5,15,'mm'),(14,20,'index')]:records.append({'event':'compile_call','pid':1,'kernel':k,'start_ns':a,'end_ns':b,'elapsed_ns':b-a})
    p.write_text(''.join(json.dumps(r)+'\n' for r in records));s=m.summarize(temp,[{'label':'score','start_ns':3,'end_ns':18}])['windows'][0]
    assert s['compile_calls_overlapping']==3 and s['host_interval_union_seconds']==15/1e9 and s['host_interval_sum_seconds']==21/1e9,s
    p.write_text('bad-json\n')
    try:m.summarize(temp,[{'label':'score','start_ns':0,'end_ns':1}])
    except ValueError:pass
    else:raise AssertionError('corrupt evidence treated as zero compile')
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'compile-1.jsonl'
    rows=[{'event':'installation','pid':1,'provenance':{'jit':m.JIT_SHA,'compiler':m.COMPILER_SHA}},
          {'event':'compile_call','pid':1,'kernel':'_small_o_kernel'}]
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert m.verify_worker_coverage(temp,'(EngineCore pid=1) ready')['engine_pids']==[1]
    for log in ['(EngineCore pid=2) ready','no worker evidence']:
        try:m.verify_worker_coverage(temp,log)
        except ValueError:pass
        else:raise AssertionError('worker coverage gap hidden')
    p.write_text(json.dumps(rows[0])+'\n')
    try:m.verify_worker_coverage(temp,'(EngineCore pid=1) ready')
    except ValueError:pass
    else:raise AssertionError('no real kernel positive control accepted')
print(json.dumps({'passed':True,'gpu_executed':False,'actual_callable_and_code_unchanged':True,'return_identity_kwargs_exception_preserved':True,'already_bound_calls_observed':True,'nested_threaded_and_real_fork_observed':True,'monitoring_slot_rollback':True,'overlap_union_clipping':True,'corrupt_evidence_rejected':True,'worker_positive_required':True}))
