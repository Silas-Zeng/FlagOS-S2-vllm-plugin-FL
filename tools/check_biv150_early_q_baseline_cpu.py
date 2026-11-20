"""CPU fault tests: current-policy tap must follow Q32 and a future K128 main."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS


def main():
    source=Path(__file__).with_name('bench_biv150_prefill_bm64_qhoist.py')
    nodes=[n for n in ast.parse(source.read_text()).body
           if (isinstance(n,ast.ClassDef) and n.name=='Tap')
           or (isinstance(n,ast.FunctionDef) and n.name=='comparison_baseline')]
    results=[]
    def scenario(mode,present=False,disabled=False,missing_dependency=False):
        class Launch:
            def __getitem__(self,grid):
                return lambda **kwargs:NS(grid=grid,kwargs=kwargs)
        old=NS(wrapper=Launch(),holder={},kernel_key='kernel',block_m=128,
               num_warps=8,tile_size=64,num_stages=1)
        old.holder['kernel']=old.wrapper
        class Q32Policy:
            def __init__(self,manager):self.manager=manager
            def __getitem__(self,grid):
                def call(**kwargs):
                    m=self.manager
                    return m.wrapper[grid](**dict(kwargs,BLOCK_M=m.block_m,
                        num_warps=m.num_warps,TILE_SIZE=m.tile_size,num_stages=m.num_stages))
                return call
        class K128Policy(Q32Policy):pass
        def manager(tile):return NS(wrapper=Launch(),block_m=256,num_warps=16,tile_size=tile,num_stages=1)
        q32=NS(_MANAGER=manager(64),_Policy=Q32Policy)
        k128=NS(_MANAGER=manager(128),_Policy=K128Policy)
        def install_q32():
            old.holder['kernel']=Q32Policy(q32._MANAGER)
            return {'installed':True}
        def install_k128():
            if disabled:return {'installed':False,'reason':'disabled'}
            old.holder['kernel']=K128Policy(k128._MANAGER)
            return {'installed':True}
        q32.install=install_q32;k128.install=install_k128
        def imported(name):
            if name=='vllm_fl.patches.biv150_prefill_q32':return q32
            if name=='vllm_fl.patches.biv150_prefill_k128':
                if missing_dependency:raise ModuleNotFoundError(name='unrelated_dependency')
                if present:return k128
                raise ModuleNotFoundError(name=name)
            raise AssertionError(name)
        namespace={'importlib':NS(import_module=imported)}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        def function(**kwargs):return old.holder['kernel'][(65,2)](**kwargs)
        fixture=NS(install_baseline=lambda:(function,old,{}))
        fn,active,states,tap=namespace['comparison_baseline'](fixture,mode)
        if mode!='bm128':assert type(old.holder['kernel']) is (K128Policy if mode=='current' and present else Q32Policy)
        result=fn(BLOCK_M=128,num_warps=8,TILE_SIZE=64,num_stages=1)
        expected_tile=128 if mode=='current' and present else 64
        assert result.kwargs['TILE_SIZE']==tap.kwargs['TILE_SIZE']==expected_tile
        assert active.wrapper is tap if mode!='bm128' else old.holder['kernel'] is tap
        assert result.kwargs['BLOCK_M']==(128 if mode=='bm128' else 256)
        results.append(dict(mode=mode,k128_present=present,actual_tile=expected_tile,inner_tap=True))
    scenario('current',False)
    scenario('current',True)
    scenario('q32',True)
    scenario('bm128',False)
    rejected=[]
    for name,kwargs,exception in [('disabled current K128',{'present':True,'disabled':True},RuntimeError),
                                  ('unrelated missing dependency',{'present':True,'missing_dependency':True},ModuleNotFoundError)]:
        try:scenario('current',**kwargs)
        except exception:rejected.append(name)
        else:raise AssertionError('Silently weakened current baseline')
    record=dict(passed=True,gpu_executed=False,cases=results,rejected=rejected,
                limits='Synthetic routing/identity contract only, not target module installation or kernel validation.')
    root=Path(__file__).resolve().parents[1]
    (root/'docs/evidence/prefill-early-q-20261027/current-baseline-cpu.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
