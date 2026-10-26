"""Synthetic CPU routing checks; no target compiler, Torch, C3 or kernel proof."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from types import ModuleType
from unittest.mock import patch

import bench_biv150_prefill_multi2 as probe


def scenario(present=False,disabled=False,missing_dependency=False,broken_chain=False):
    def manager(rows,warps,tile):
        return NS(block_m=rows,num_warps=warps,tile_size=tile,num_stages=1,
                  selected_calls=0,active=True,holder={},kernel_key='kernel')
    base,qm,km=manager(128,8,64),manager(256,16,64),manager(256,16,128)
    class Launch:
        def __init__(self,owner):self.owner=owner
        def __getitem__(self,grid):
            def invoke(**kwargs):
                self.owner.selected_calls+=1
                return NS(grid=grid,kwargs=kwargs)
            return invoke
    for m in [base,qm,km]:m.wrapper=Launch(m)
    base.holder['kernel']=base.wrapper
    class Q32Policy:
        def __init__(self,previous,owner):
            self.base,self.manager=previous,owner
            self.disabled=False
        def __getitem__(self,grid):
            def invoke(**kwargs):
                if kwargs['num_seqs']==1 and not self.disabled:
                    m=self.manager
                    return m.wrapper[grid](**dict(kwargs,BLOCK_M=m.block_m,
                        num_warps=m.num_warps,TILE_SIZE=m.tile_size,num_stages=1))
                return self.base[grid](**kwargs)
            return invoke
    class K128Policy(Q32Policy):pass
    def install_q32():
        base.holder['kernel']=Q32Policy(base.wrapper,qm)
        return {'installed':True}
    def install_k128():
        if disabled:return {'installed':False,'reason':'disabled'}
        previous=base.holder['kernel']
        base.holder['kernel']=K128Policy(Launch(base) if broken_chain else previous,km)
        return {'installed':True}
    q32=NS(_MANAGER=qm,_Policy=Q32Policy,install=install_q32)
    k128=NS(_MANAGER=km,_Policy=K128Policy,install=install_k128)
    def imported(name):
        if name=='vllm_fl.patches.biv150_prefill_q32':return q32
        if name=='vllm_fl.patches.biv150_prefill_k128':
            if missing_dependency:raise ModuleNotFoundError(name='different_missing_dependency')
            if present:return k128
            raise ModuleNotFoundError(name=name)
        raise AssertionError(name)
    def function(**kwargs):return base.holder['kernel'][(129,2)](**kwargs)
    with patch.object(probe.importlib,'import_module',side_effect=imported):
        fn,active,states,tap=probe.comparison_baseline(NS(install_baseline=lambda:(function,base,{})))
    assert active is base
    for name in probe.DEFAULT_CASES:
        qlens,klens=probe.CASES[name]
        probe.validate_case(qlens,klens)
        out=fn(num_seqs=2,BLOCK_M=128,BLOCK_Q=16,num_warps=8,TILE_SIZE=64,num_stages=1,IS_3D=False)
        assert tap.kwargs==out.kwargs and out.kwargs['BLOCK_M']==128
    with patch.object(probe,'kernel_configuration',return_value={}):
        proof=probe.selected_configuration(tap,active)
        assert len(proof['preserved_default_parents'])==(2 if present else 1)
        qm.selected_calls=1
        try:probe.selected_configuration(tap,active)
        except RuntimeError:pass
        else:raise AssertionError('Stale or wrongly selected parent accepted')
    assert km.selected_calls==0 and base.selected_calls==4
    return dict(k128_present=present,bm128_fallback_calls=base.selected_calls,
                q32_preserved=True,k128_preserved=present)


def real_source_scope():
    """Load unmodified policy/descriptor source with explicit fake tensor metadata."""
    root=Path(__file__).resolve().parents[1]
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,root/path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    fragment=load('_multi2_real_fragment','vllm_fl/patches/biv150_q_fragment_service.py')
    package,patches=ModuleType('vllm_fl'),ModuleType('vllm_fl.patches')
    package.__path__=[];patches.__path__=[]
    patches.biv150_q_fragment_service=fragment
    with patch.dict(sys.modules,{'vllm_fl':package,'vllm_fl.patches':patches}):
        q32=load('_multi2_real_q32','vllm_fl/patches/biv150_prefill_q32.py')
    device=NS(type='cuda',index=0)
    class Tensor:
        def __init__(self,shape,strides,address,dtype='torch.bfloat16'):
            self.shape,self.strides,self.address,self.dtype=shape,strides,address,dtype
            self.ndim,self.device=len(shape),device
        def stride(self,axis=None):return self.strides if axis is None else self.strides[axis]
        def untyped_storage(self):return NS(data_ptr=lambda:self.address)
    calls=[]
    class Launch:
        def __init__(self,label):self.label=label
        def __getitem__(self,grid):
            def invoke(**kwargs):
                calls.append((self.label,grid,kwargs))
                return NS(hash='selected')
            return invoke
    manager=NS(wrapper=Launch('q32'),async_mode=NS(get=lambda:None),
        raw=NS(pre_run_hooks=[]),candidate=NS(pre_run_hooks=[]),device=None,
        attention=NS(torch=NS(cuda=NS(is_current_stream_capturing=lambda:False))),
        optimized={'selected':{}})
    policy=q32._Policy(Launch('bm128'),manager)
    def values(qlens):
        count=len(qlens)
        return dict(IS_3D=False,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,num_query_heads=16,
            num_queries_per_kv=8,BLOCK_M=128,BLOCK_Q=16,BLOCK_SIZE=16,TILE_SIZE=64,
            NUM_SEGMENTS_PER_SEQ=1,KV_QUANT_MODE=0,USE_CAUSAL=True,SLIDING_WINDOW=0,
            CHUNK_LOOKBACK=-1,CHUNK_SIZE=-1,num_warps=8,num_stages=1,num_seqs=count,
            query_ptr=Tensor((sum(qlens),16,128),(2048,128,1),1),
            output_ptr=Tensor((sum(qlens),16,128),(2048,128,1),2),
            key_cache_ptr=Tensor((2048,16,2,128),(8192,256,128,1),3),
            value_cache_ptr=Tensor((2048,16,2,128),(8192,256,128,1),3),
            query_start_len_ptr=Tensor((count+1,),(1,),4,'torch.int32'),
            seq_lens_ptr=Tensor((count,),(1,),5,'torch.int32'),
            block_tables_ptr=Tensor((count,8192),(8192,1),6,'torch.int32'))
    for qlens,klens in probe.CASES.values():
        item=values(qlens)
        assert fragment._eligible(item)
        candidate=dict(item,BLOCK_M=256,BLOCK_Q=32,num_warps=16)
        assert fragment._eligible(candidate,256,num_warps=16)
        assert not fragment._eligible(dict(item,USE_SINKS=True))
        assert not fragment._eligible(dict(item,output_ptr=item['query_ptr']))
        assert not fragment._eligible(dict(item,query_start_len_ptr=Tensor((2,),(1,),4,'torch.int32')))
        policy[(129,2)](**item)
        assert calls[-1][0]=='bm128'
        assert calls[-1][2]==item
        contract=probe.case_contract(qlens,klens)
        assert contract['first_visible_kv']==[k-q+1 for q,k in zip(qlens,klens)]
    assert not policy.disabled and not policy.reported
    two_sequence_calls=len(calls)
    # Suppress the production selection marker from this explicitly fake launch.
    with patch('builtins.print'):
        policy[(129,2)](**values([2033]))
    assert calls[-1][0]=='q32' and calls[-1][1]==(64,2)
    assert calls[-1][2]['BLOCK_M']==256 and calls[-1][2]['BLOCK_Q']==32
    assert calls[-1][2]['num_warps']==16 and not policy.disabled
    return dict(actual_descriptor_source=True,actual_q32_policy_source=True,
        two_sequence_natural_fallbacks=two_sequence_calls,single_sequence_q32_preserved=True,
        tensor_metadata_is_fake=True,target_compiler_executed=False)


def preflight_checks():
    for connected,report,passes in ((True,'No running processes found',False),
            (False,'active process',False),(False,'unknown report',False),
            (False,'No running processes found',True)):
        with patch.object(probe.socket,'socket') as socket,patch.object(
                probe.subprocess,'check_output',return_value=report):
            socket.return_value.__enter__.return_value.connect_ex.return_value=0 if connected else 1
            try:probe.require_idle_probe_host()
            except RuntimeError:assert not passes
            else:assert passes
    return dict(service_busy_rejected=True,gpu_busy_rejected=True,
                unknown_gpu_report_rejected=True,idle_allowed=True,real_host_inspected=False)


def main():
    cases=[scenario(),scenario(present=True)]
    real_scope=real_source_scope()
    preflight=preflight_checks()
    rejected=[]
    for name,kwargs,kind in [
        ('disabled K128',dict(present=True,disabled=True),RuntimeError),
        ('broken parent chain',dict(present=True,broken_chain=True),RuntimeError),
        ('different missing dependency',dict(missing_dependency=True),ModuleNotFoundError)]:
        try:scenario(**kwargs)
        except kind:rejected.append(name)
        else:raise AssertionError(name+' silently accepted')
    for qlens,klens in [([2048],[4096]),([1,2032],[4096,4096]),([512,512],[4096,4096]),([1945,88],[100,4096])]:
        try:probe.validate_case(qlens,klens)
        except ValueError:pass
        else:raise AssertionError('Outside two-sequence probe scope')
    assert 'torch' not in sys.modules
    value=dict(passed=True,gpu_executed=False,synthetic_cpu_fixtures=True,
               precision_gate=False,cases=cases,rejected=rejected,query_cases=probe.CASES,
               default_gpu_cases=probe.DEFAULT_CASES,real_source_scope=real_scope,
               preflight=preflight,case_contracts={name:probe.case_contract(*case)
                                                  for name,case in probe.CASES.items()},
               limitation='Real source with fake tensor metadata plus simulated policy chain; no real install, C3, numerical, resource or speed proof.')
    folder=Path(__file__).resolve().parents[1]/'docs/evidence/prefill-multi2-20261027'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'cpu-routing.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
