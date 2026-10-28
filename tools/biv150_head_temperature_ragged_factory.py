"""仅组件实例跳过已经完成的原temperature；原Sampler代码对象全保留。"""
import hashlib,importlib,inspect
from pathlib import Path

ROWS=None

def configure(rows):
    global ROWS
    assert type(rows) is int and rows in (31,41,42,43)
    ROWS=rows

LINEAR_SHA='d431a4fa144ed67dc3deab46dd5beace3612858075f85c39e602447ba3bf768a'


class Capture:
    def __init__(self,original):self.original,self.records=original,[]
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            value=self.original[grid](*args,**kwargs)
            assert isinstance(value,tuple) and len(value)==2,'真实LibEntry ABI变化'
            kernel,constants=value
            assert tuple(args[4:7])==(ROWS,130560,2048)
            assert kwargs['BIAS'] is False and kwargs['GROUP_M']==8 and kwargs['SPLIT_K']==1
            assert constants.get('EVEN_K') is True,'非原EVEN_K head，不改变循环数学'
            expected=dict(BLOCK_M=32,BLOCK_N=64,BLOCK_K=64,num_warps=8,num_stages=1,num_ctas=1) if ROWS==31 else dict(BLOCK_M=64,BLOCK_N=256,BLOCK_K=32,num_warps=16,num_stages=2,num_ctas=1)
            assert all(constants.get(k)==v for k,v in expected.items()),'实际原head配置不在本次复用范围'
            resolved=dict(zip(self.original.arg_names,args));resolved.update(kwargs);resolved.update(constants)
            actual_grid=tuple(grid(resolved)) if callable(grid) else tuple(grid)
            assert actual_grid==(2040 if ROWS==31 else 510,1),'原head实际grid变化'
            self.records.append(dict(kernel=kernel,constants=dict(constants),kwargs=dict(kwargs),grid=actual_grid,args=tuple(args)))
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
