"""仅组件实例跳过已经完成的原temperature；原Sampler代码对象全保留。"""
import hashlib,importlib,inspect
from pathlib import Path

ROWS=None

def configure(rows):
    global ROWS
    assert type(rows) is int and rows in (31,41,42,43)
    ROWS=rows

LINEAR_SHA='d431a4fa144ed67dc3deab46dd5beace3612858075f85c39e602447ba3bf768a'
CONFIG_TABLE='linear_kernel_91df7045e2de13a4c456e8d13079eb7c'


def baseline_config():
    assert ROWS in (31,41,42,43)
    return dict(BLOCK_M=32,BLOCK_N=64,BLOCK_K=64,num_warps=8,num_stages=1,num_ctas=1) if ROWS==31 else dict(BLOCK_M=64,BLOCK_N=256,BLOCK_K=32,num_warps=16,num_stages=2,num_ctas=1)


def baseline_before_launch(original,args,kwargs):
    """Read the actual pinned selector; a miss must never start autotuning here."""
    expected=baseline_config()
    assert tuple(args[4:7])==(ROWS,130560,2048)
    assert kwargs==dict(BIAS=False,GROUP_M=8,SPLIT_K=1)
    assert original._has_flagtune_tuner is False,'dynamic FlagTune selection not admitted'
    tuner=original.fn
    assert getattr(tuner._run_mode,'value',tuner._run_mode)=='normal'
    assert tuner.config_table_name==CONFIG_TABLE and list(tuner.keys)==['M','N','K']
    assert len(tuner.configs)>1,'single-config selector bypasses the checked cache'
    bound=dict(zip(original.arg_names,args));bound.update(kwargs)
    key=tuner.get_key({k:v for k,v in bound.items() if k in tuner.arg_names})
    assert key==((32 if ROWS==31 else 64),130560,2048)+('torch.bfloat16',)*4
    config=tuner.cache.get(key)
    assert config is not None,'baseline config cache miss: stop before tuning or launch'
    assert config.pre_hook is None,'unreviewed baseline config hook'
    constants=config.all_kwargs()
    assert all(constants.get(k)==v for k,v in expected.items()),'baseline cached config outside the reviewed tile'
    assert getattr(tuner,'shared_config_pre_hook',None) is None
    assert all(c.pre_hook is None for c in tuner.configs if c.all_kwargs()==constants),'matching pool config has an unreviewed hook'
    x=args[0]
    assert tuple(x.shape)==(ROWS,2048) and tuple(x.stride())==(2048,1) and x.element_size()==2
    assert x.data_ptr()%16==0
    available=x.untyped_storage().nbytes()-x.storage_offset()*x.element_size()
    required=(expected['BLOCK_M']+1)*2048*2  # complete tile plus one complete trailing guard row
    assert available>=required,'baseline hidden backing does not cover the full physical tile'
    return dict(key=list(key),table=tuner.config_table_name,constants=constants,
                physical_rows_from_input=available//4096,required_rows_from_input=expected['BLOCK_M']+1,
                logical_rows=ROWS,cache_hit=True,autotune_allowed=False)


class Capture:
    def __init__(self,original):self.original,self.records,self.prelaunch=original,[],[]
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            admission=baseline_before_launch(self.original,args,kwargs)
            expected=baseline_config()
            checked=[]
            def checked_grid(meta):
                # Both the cold JIT and warm LibEntry evaluate this before launching.
                assert all(meta.get(k)==expected[k] for k in ('BLOCK_M','BLOCK_N','BLOCK_K')),'actual baseline launch tile changed'
                assert meta.get('EVEN_K') is True and meta.get('SPLIT_K')==1
                value=tuple(grid(meta)) if callable(grid) else tuple(grid)
                assert value==(2040 if ROWS==31 else 510,1),'actual baseline launch grid changed'
                checked.append(dict(BLOCK_M=meta['BLOCK_M'],BLOCK_N=meta['BLOCK_N'],BLOCK_K=meta['BLOCK_K'],grid=list(value)))
                return value
            value=self.original[checked_grid](*args,**kwargs)
            assert checked,'pinned runtime did not evaluate the prelaunch grid guard'
            admission['actual_checked_grids']=checked
            assert isinstance(value,tuple) and len(value)==2,'真实LibEntry ABI变化'
            kernel,constants=value
            assert tuple(args[4:7])==(ROWS,130560,2048)
            assert kwargs['BIAS'] is False and kwargs['GROUP_M']==8 and kwargs['SPLIT_K']==1
            assert constants.get('EVEN_K') is True,'非原EVEN_K head，不改变循环数学'
            assert all(constants.get(k)==v for k,v in expected.items()),'实际原head配置不在本次复用范围'
            resolved=dict(zip(self.original.arg_names,args));resolved.update(kwargs);resolved.update(constants)
            actual_grid=tuple(grid(resolved)) if callable(grid) else tuple(grid)
            assert actual_grid==(2040 if ROWS==31 else 510,1),'原head实际grid变化'
            self.records.append(dict(kernel=kernel,constants=dict(constants),kwargs=dict(kwargs),grid=actual_grid,args=tuple(args)))
            self.prelaunch.append(admission)
            return value
        return launch


def actual_linear_module(flag_gems,run):
    entries=[v for v in flag_gems.current_work_registrar.config if v[0]=='linear']
    assert len(entries)==1 and entries[0][1] is flag_gems.linear
    fn=inspect.unwrap(flag_gems.linear);module=importlib.import_module(fn.__module__)
    path=Path(inspect.getsourcefile(fn))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==LINEAR_SHA
    (run/'source/actual-installed-linear.py').write_bytes(path.read_bytes())
    return module


def launch(candidate,record,x,w,temp,out):
    assert tuple(x.shape)==(ROWS,2048) and tuple(w.shape)==(130560,2048)
    assert tuple(out.shape)==(ROWS,130560) and tuple(temp.shape)==(ROWS,)
    assert x.is_contiguous() and w.is_contiguous() and out.is_contiguous() and temp.is_contiguous()
    args=list(record['args']);args[:4]=[x,w,w,out]
    args.append(temp)
    options=dict(record['constants']);options.update(record['kwargs'])
    assert 'RAGGED_M' not in options,'original head unexpectedly supplied the new specialization'
    options['RAGGED_M']=True
    return candidate.head_temperature_kernel[record['grid']](*args,**options)


class PreScaledSampler:
    def __init__(self,sampler,torch):
        self.sampler,self.torch=sampler,torch
        self.pending=self.temperature=None
        self.observer=None
        sampler.apply_temperature=self.consume
    def consume(self,value,temperature,all_random):
        assert all_random and self.pending is value and self.temperature is temperature
        self.pending=None
        if self.observer:self.observer(value)
        return value
    def __call__(self,value,sm):
        assert self.pending is None and value.dtype==self.torch.float32
        assert sm.all_random and not sm.all_greedy and sm.temperature is not None
        assert sm.no_penalties and sm.max_num_logprobs is None and not sm.logprob_token_ids
        assert sm.allowed_token_ids_mask is None and not sm.bad_words_token_ids and sm.spec_token_ids is None
        assert not sm.logitsprocs.argmax_invariant and not sm.logitsprocs.non_argmax_invariant
        holder=sm.thinking_budget_state_holder
        assert holder is None or not holder.has_tracked_requests()
        self.pending,self.temperature=value,sm.temperature
        try:
            result=self.sampler(value,sm)
            assert self.pending is None,'原Sampler未消费已完成temperature'
            return result
        finally:self.pending=self.temperature=None
