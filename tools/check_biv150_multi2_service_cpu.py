"""Real installer/policy/manager source with fake JIT/tensors; no Torch or GPU."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
_async_compile=NS(active_mode=NS(get=lambda:None))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class JIT:
    def __init__(self,fn,**options):
        self.fn=fn;self.version=options.get('version');self.pre_run_hooks=[]
        self.do_not_specialize=[];self.do_not_specialize_on_alignment=[]
        self.debug=False;self.noinline=False;self._repr=None;self.launch_metadata=None
        self.device_caches={};self.label='unlabelled';self.calls=[];self.error=None
    def run(self):pass
    def __getitem__(self,grid):
        def invoke(*args,**kwargs):
            self.calls.append((grid,args,kwargs))
            if self.error:raise self.error
            return NS(hash=self.label)
        return invoke


DEVICE=NS(type='cuda',index=0)
class Tensor:
    def __init__(self,shape,strides,address,dtype='torch.bfloat16'):
        self.shape,self.strides,self.address,self.dtype=shape,strides,address,dtype
        self.device=DEVICE;self.ndim=len(shape)
    def stride(self,axis=None):return self.strides if axis is None else self.strides[axis]
    def untyped_storage(self):return NS(data_ptr=lambda:self.address)


def values(rows=1026,count=2):
    return dict(IS_3D=False,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,num_query_heads=16,
        num_queries_per_kv=8,BLOCK_M=128,BLOCK_Q=16,BLOCK_SIZE=16,TILE_SIZE=64,
        NUM_SEGMENTS_PER_SEQ=1,KV_QUANT_MODE=0,USE_CAUSAL=True,SLIDING_WINDOW=0,
        CHUNK_LOOKBACK=-1,CHUNK_SIZE=-1,num_warps=8,num_stages=1,num_seqs=count,
        query_ptr=Tensor((rows,16,128),(2048,128,1),1),
        output_ptr=Tensor((rows,16,128),(2048,128,1),2),
        key_cache_ptr=Tensor((2048,16,2,128),(8192,256,128,1),3),
        value_cache_ptr=Tensor((2048,16,2,128),(8192,256,128,1),3),
        query_start_len_ptr=Tensor((count+1,),(1,),4,'torch.int32'),
        seq_lens_ptr=Tensor((count,),(1,),5,'torch.int32'),
        block_tables_ptr=Tensor((count,8192),(8192,1),6,'torch.int32'))


def environment():
    fragment=load('_service_real_fragment','vllm_fl/patches/biv150_q_fragment_service.py')
    package,patches=ModuleType('vllm_fl'),ModuleType('vllm_fl.patches')
    package.__path__=[];patches.__path__=[];patches.biv150_q_fragment_service=fragment
    with patch.dict(sys.modules,{'vllm_fl':package,'vllm_fl.patches':patches}):
        q32=load('_service_real_q32','vllm_fl/patches/biv150_prefill_q32.py')
        patches.biv150_prefill_q32=q32
        multi=load('_service_real_multi2','vllm_fl/patches/biv150_prefill_multi2.py')
    raw=JIT(lambda:None);raw.label='raw'
    base_candidate=JIT(raw.fn);base_candidate.label='bm128'
    capture=[False]
    attention=NS(torch=NS(cuda=NS(is_current_stream_capturing=lambda:capture[0])))
    backend=NS(TritonAttentionImpl=NS(forward=lambda:None))
    base=fragment.Manager(attention,backend,{'kernel':raw},'kernel',raw,base_candidate)
    base.install();fragment._MANAGER=base
    jit=NS(JITFunction=JIT)
    with patch.object(q32.importlib,'import_module',return_value=jit):
        assert q32.install()['installed']
    parent=q32._MANAGER;parent.candidate.label='q32'
    def install():
        with patch.object(multi.importlib,'import_module',return_value=jit):
            return multi.install()
    def activate(manager,label):
        # Only the device binder/compiler is replaced. Real Manager/_Launch,
        # descriptor gates, installers and both policy classes remain intact.
        manager.candidate.label=label;manager.optimized[label]={}
        manager._ensure=lambda device:None
    activate(base,'bm128');activate(parent,'q32')
    return NS(fragment=fragment,q32=q32,multi=multi,base=base,parent=parent,
              install=install,jit=jit,activate=activate,capture=capture)


def policy_and_fault_checks():
    env=environment();m=env.multi;b=env.base;p=env.parent
    before=b.holder['kernel']
    with patch.dict(os.environ,{'VLLM_FL_BIV150_PREFILL_MULTI2':'0'}):
        assert env.install()=={'installed':False,'reason':'disabled'}
    assert b.holder['kernel'] is before and m._MANAGER is None
    with patch.dict(os.environ):
        os.environ.pop('VLLM_FL_BIV150_PREFILL_MULTI2',None)
        installed=env.install()
    assert installed['installed'] and type(b.holder['kernel']) is m._Policy
    policy=b.holder['kernel'];wide=m._MANAGER;env.activate(wide,'multi2')
    assert policy.base is before and before.manager is p and before.base is b.wrapper
    assert len({id(x.device_caches) for x in (b.raw,b.candidate,p.candidate,wide.candidate)})==4
    assert env.install()['installed'] and b.holder['kernel'] is policy
    routed=[]
    for rows,count,label in ((1023,2,'bm128'),(1024,2,'multi2'),(1025,2,'multi2'),
                              (1026,2,'multi2'),(2048,2,'multi2'),(2033,1,'q32'),(2048,3,'bm128')):
        item=values(rows,count);original=dict(item)
        got=policy[(rows//16+count,2)](**item)
        assert got.hash==label and item==original
        if label=='multi2':
            grid,args,selected=wide.candidate.calls[-1]
            assert grid==(rows//32+count,2) and not args
            assert selected['BLOCK_M']==256 and selected['BLOCK_Q']==32 and selected['num_warps']==16
        routed.append(dict(rows=rows,sequences=count,route=label))
    guards=[]
    for name,edit,expected in (
            ('alias',lambda k:k.update(output_ptr=k['query_ptr']),'raw'),
            ('unsupported semantic',lambda k:k.update(USE_SINKS=True),'raw'),
            ('foreign override',lambda k:k.update(ir_override='foreign'),'raw')):
        item=values();edit(item);n=len(wide.candidate.calls)
        assert policy[(66,2)](**item).hash==expected and len(wide.candidate.calls)==n
        guards.append(name)
    env.capture[0]=True;n=len(wide.candidate.calls)
    assert policy[(66,2)](**values()).hash=='raw' and len(wide.candidate.calls)==n
    env.capture[0]=False;guards.append('capture')
    with patch.object(wide.async_mode,'get',return_value=object()):
        assert policy[(66,2)](**values()).hash=='raw'
    guards.append('async')
    wide.candidate.pre_run_hooks.append(object())
    assert policy[(66,2)](**values()).hash=='bm128'
    wide.candidate.pre_run_hooks.clear();guards.append('candidate hook')
    wide.raw.pre_run_hooks.append(object())
    assert policy[(66,2)](**values()).hash=='raw'
    wide.raw.pre_run_hooks.clear();guards.append('raw hook')
    wide.active=False;assert policy[(66,2)](**values()).hash=='bm128';wide.active=True
    guards.append('inactive manager')
    assert policy[(66,2)](object(),**values()).hash=='raw'
    guards.append('positional call')
    wide.device=1;assert policy[(66,2)](**values()).hash=='bm128';wide.device=None
    guards.append('device mismatch')
    wide.candidate.error=RuntimeError('already launched device failure')
    old_calls=len(b.candidate.calls)+len(b.raw.calls)+len(p.candidate.calls)
    try:policy[(66,2)](**values())
    except RuntimeError as error:assert 'device failure' in str(error)
    else:raise AssertionError('Device error hidden')
    assert old_calls==len(b.candidate.calls)+len(b.raw.calls)+len(p.candidate.calls)
    assert not policy.disabled
    wide.candidate.error=m._UnavailableBeforeLaunch('proof unavailable')
    assert policy[(66,2)](**values()).hash=='bm128' and policy.disabled
    attempts=len(wide.candidate.calls)
    assert policy[(66,2)](**values()).hash=='bm128' and len(wide.candidate.calls)==attempts
    # Real verified-manager method, stopped at a fake prelaunch compile result.
    compiled=NS(hash='proved');wide.optimized={'proved':{}}
    with patch.object(env.fragment.Manager,'_compile',return_value=compiled):
        assert wide._compile() is compiled
        wide.optimized={}
        try:wide._compile()
        except m._UnavailableBeforeLaunch:pass
        else:raise AssertionError('Unproved result accepted')
    with patch.object(env.fragment.Manager,'_compile',side_effect=RuntimeError('compile only')):
        try:wide._compile()
        except m._UnavailableBeforeLaunch:pass
        else:raise AssertionError('Compile error escaped prelaunch classification')
    b.holder['kernel']=before
    assert not env.install()['installed'] and b.holder['kernel'] is before
    return dict(default_install=installed,real_policy_routes=routed,guards=guards,
        private_caches_distinct=True,unchanged_q32_parent=True,idempotent_install=True,
        prelaunch_only_fallback=True,device_error_propagates_without_retry=True,
        disabled_after_proof_failure=True,rebound_install_rejected=True)


def installer_rejections():
    rejected=[]
    for name,change in (
        ('broken parent',lambda e:setattr(e.base.holder['kernel'],'base',object())),
        ('disabled parent',lambda e:setattr(e.base.holder['kernel'],'disabled',True)),
        ('unknown parent source',lambda e:setattr(e.multi,'_EXPECTED_Q32','0'*64)),
        ('private cache alias',lambda e:setattr(e.jit,'JITFunction',lambda *a,**k:NS(device_caches=e.base.raw.device_caches)))):
        e=environment();change(e);before=e.base.holder['kernel']
        assert not e.install()['installed'] and e.base.holder['kernel'] is before
        rejected.append(name)
    return rejected


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        with patch.dict(os.environ,{'VLLM_FL_BIV150_PREFILL_MULTI2':'1','VLLM_FL_BIV150_PREFILL_Q32':'1'}):
            policy=policy_and_fault_checks();rejected=installer_rejections()
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    result=dict(passed=True,gpu_executed=False,precision_gate=False,policy=policy,
        installer_rejections=rejected,
        limitation='Real source installers, policies, manager launch gates; explicit fake tensors/JIT/compiler, no SDK lowering, C3 live stack, numerical, resource or model proof.')
    path=ROOT/'docs/evidence/prefill-multi2-20261027/service-cpu.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
