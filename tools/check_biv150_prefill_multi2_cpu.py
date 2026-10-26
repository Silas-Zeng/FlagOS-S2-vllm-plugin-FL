"""Synthetic CPU routing checks; no target compiler, Torch, C3 or kernel proof."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
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
    for qlens,klens in probe.CASES.values():
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


def main():
    cases=[scenario(),scenario(present=True)]
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
               cases=cases,rejected=rejected,query_cases=probe.CASES,
               limitation='Routing and scope only; no real install, C3, numerical, resource or speed proof.')
    folder=Path(__file__).resolve().parents[1]/'docs/evidence/prefill-multi2-20261027'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'cpu-routing.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
