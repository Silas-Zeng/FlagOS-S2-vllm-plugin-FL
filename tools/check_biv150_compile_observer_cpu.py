"""CPU-only observer return/exception/alias/interval-union checks."""
import importlib.util,json,tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('observer',ROOT/'biv150_compile_observer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as temp:
    destination=Path(temp);sentinel=object();calls=[]
    def compile(source,**kwargs):
        calls.append((source,kwargs))
        if source.name=='fail':raise RuntimeError('original exception')
        return sentinel
    package=SimpleNamespace(compile=compile);implementation=SimpleNamespace(compile=compile)
    already_bound=package.compile
    handle=m.Handle(package,implementation,destination,{'cpu_fixture':True})
    late_bound=package.compile
    assert late_bound(SimpleNamespace(name='mm_kernel'),options={'num_warps':8,'num_stages':2}) is sentinel
    assert calls[-1][1]=={'options':{'num_warps':8,'num_stages':2}}
    try:late_bound(SimpleNamespace(name='fail'))
    except RuntimeError as e:assert str(e)=='original exception'
    else:raise AssertionError('exception swallowed')
    records=[json.loads(line) for p in destination.glob('*.jsonl') for line in p.read_text().splitlines()];assert len([r for r in records if r['event']=='compile_call'])==2
    assert already_bound(SimpleNamespace(name='unobserved_before_install')) is sentinel
    assert len([json.loads(line) for p in destination.glob('*.jsonl') for line in p.read_text().splitlines() if json.loads(line)['event']=='compile_call'])==2
    real_pid=m.os.getpid
    try:
        m.os.getpid=lambda:1234567
        assert late_bound(SimpleNamespace(name='_small_o_kernel')) is sentinel
    finally:m.os.getpid=real_pid
    inherited=[json.loads(line) for p in destination.glob('*.jsonl') for line in p.read_text().splitlines()]
    assert any(r['event']=='installation' and r['pid']==1234567 and r.get('inherited_from_pid')==real_pid() for r in inherited)
    assert handle.uninstall()=={'aliases_restored':[True,True],'record_errors':0};assert package.compile is implementation.compile is compile
    try:m.Handle(package,SimpleNamespace(compile=lambda x:x),destination,{})
    except RuntimeError:pass
    else:raise AssertionError('alien alias replaced')
    assert package.compile is compile
with tempfile.TemporaryDirectory() as temp:
    package=SimpleNamespace(compile=compile);implementation=SimpleNamespace(compile=compile)
    emit=m.Handle.emit
    def fail_emit(*args):raise OSError('record unavailable')
    m.Handle.emit=fail_emit
    try:m.Handle(package,implementation,temp,{})
    except OSError:pass
    else:raise AssertionError('installation write failure ignored')
    finally:m.Handle.emit=emit
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
print(json.dumps({'passed':True,'gpu_executed':False,'return_identity_and_kwargs':True,'original_exception_preserved':True,'late_binding_observed':True,'already_bound_limitation_proved':True,'alias_guard_and_restore':True,'overlap_union_clipping':True,'corrupt_evidence_rejected':True}))
