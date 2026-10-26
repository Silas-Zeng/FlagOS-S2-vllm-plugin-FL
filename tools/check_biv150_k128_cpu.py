"""CPU fault checks for K128 preparation, Q32 fallback and effective cache keys."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'vllm_fl/patches/biv150_prefill_k128.py'
    class ColdFailure(RuntimeError):pass
    class Parent:
        def _compile(self,*args,**kwargs):return self.compiled
    ns={'q32':NS(_VerifiedManager=Parent,_UnavailableBeforeLaunch=ColdFailure),
        'qfragment':NS(_eligible=lambda k:k.get('eligible',True))}
    classes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef)]
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),ns)
    loaded=[]
    compiled=NS(hash='verified',n_regs=127,n_spills=0,metadata=NS(shared=65536),
                _init_handles=lambda:loaded.append(True))
    class Launch:
        def __init__(self,value,error=None):self.value,self.error,self.calls=value,error,[]
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                self.calls.append((grid,args,kwargs))
                if self.error:raise self.error
                return self.value
            return call
    capture=[False]
    def setup(error=None):
        old,new=Launch('known-Q32'),Launch(compiled,error)
        manager=NS(wrapper=new,async_mode=NS(get=lambda:None),raw=NS(pre_run_hooks=[]),
                   candidate=NS(pre_run_hooks=[]),device=None,optimized={'verified':{}},
                   attention=NS(torch=NS(cuda=NS(is_current_stream_capturing=lambda:capture[0]))))
        kwargs=dict(query_ptr=NS(shape=[1985],device=NS(index=0)),num_seqs=1,
                    BLOCK_M=128,BLOCK_Q=16,num_warps=8,num_stages=1,TILE_SIZE=64)
        return ns['_Policy'](old,manager),old,new,kwargs
    with contextlib.redirect_stdout(io.StringIO()):
        policy,old,new,kwargs=setup();assert policy[(125,2)](**kwargs) is compiled
        selected=new.calls[0][2]
        assert new.calls[0][0]==(63,2)
        assert (selected['BLOCK_M'],selected['BLOCK_Q'],selected['TILE_SIZE'],selected['num_warps'],selected['maxnreg'])==(256,32,128,16,128)
        assert kwargs['TILE_SIZE']==64 and 'maxnreg' not in kwargs and not old.calls
        for change in ({'num_seqs':2},{'eligible':False},{'ir_override':'foreign'},{'maxnreg':128}):
            p,o,n,k=setup();k.update(change);assert p[(125,2)](**k)=='known-Q32' and not n.calls
        p,o,n,k=setup();k['query_ptr'].shape=[1023];assert p[(125,2)](**k)=='known-Q32' and not n.calls
        capture[0]=True;p,o,n,k=setup();assert p[(125,2)](**k)=='known-Q32' and not n.calls;capture[0]=False
        p,o,n,k=setup(ColdFailure('proof failed'));assert p[(125,2)](**k)=='known-Q32' and p.disabled
        assert p[(125,2)](**k)=='known-Q32' and len(n.calls)==1
        p,o,n,k=setup(RuntimeError('device error'))
        try:p[(125,2)](**k)
        except RuntimeError:pass
        else:raise AssertionError('device execution error swallowed')
        assert not o.calls
        m=ns['_Manager']();m.compiled=compiled
        assert m._compile(None,options={'maxnreg':128}) is compiled and loaded
        for field,value in (('n_regs',129),('n_spills',1)):
            original=getattr(compiled,field);setattr(compiled,field,value)
            try:m._compile(None,options={'maxnreg':128})
            except ColdFailure:pass
            else:raise AssertionError('invalid resources accepted')
            setattr(compiled,field,original)
        for options in (None,{}, {'maxnreg':127}):
            try:m._compile(None,options=options)
            except ColdFailure:pass
            else:raise AssertionError('unaudited budget accepted')
    spec=importlib.util.spec_from_file_location('_k128_semantics',root/'vllm_fl/compile_semantics.py')
    semantics=importlib.util.module_from_spec(spec);spec.loader.exec_module(semantics)
    config=NS(model_config=NS(hf_config=NS(**semantics._GEOMETRY)))
    def fp(**extra):return semantics.build_fingerprint(config,'iluvatar',package_root=root/'vllm_fl',
        environ={'VLLM_FL_BIV150_NATIVE_W8A8':'0',**extra},abi={'cpu_contract':1})
    flag='VLLM_FL_BIV150_PREFILL_K128'
    assert fp()==fp(**{flag:'1'}) and fp()!=fp(**{flag:'0'})
    for parent in ('VLLM_FL_BIV150_PREFILL_Q32','VLLM_FL_BIV150_Q_FRAGMENT','VLLM_FL_BIV150_PREFILL_NORMALIZE'):
        assert fp(**{parent:'0',flag:'1'})==fp(**{parent:'0',flag:'0'})
    base='241220723a09061c182764f8f738e5ec01190367'
    def old(path):return subprocess.check_output(['git','-C',str(root),'show',base+':'+path],text=True)
    runner=(root/'vllm_fl/worker/model_runner.py').read_text()
    hook='    from vllm_fl.patches.biv150_prefill_k128 import install as _install_prefill_k128\n\n    print("BIV150_PREFILL_K128_INSTALL", _install_prefill_k128(), flush=True)\n\n'
    assert runner.count(hook)==1 and runner.replace(hook,'')==old('vllm_fl/worker/model_runner.py')
    order=(root/'vllm_fl/patches/biv150_compute_order.py').read_text()
    prior=old('vllm_fl/patches/biv150_compute_order.py')
    for name in ('_main_stack','_cold'):
        def function(text):return next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert function(order)==function(prior),name
    expected=next(ast.literal_eval(n.value) for n in ast.parse(order).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPECTED_RUNNER' for t in n.targets))
    assert hashlib.sha256(runner.encode()).hexdigest()==expected
    print(json.dumps(dict(passed=True,gpu_executed=False,known_q32_fallback=True,
        no_retry_after_device_error=True,resource_budget_before_launch=True,
        effective_cache_flags=True,runner_inverse_exact=True,strict_compute_order_functions_unchanged=True)))


if __name__=='__main__':main()
