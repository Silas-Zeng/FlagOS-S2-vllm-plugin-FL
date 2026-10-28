"""No-framework checks for continuation of the exact archived ten-gate worker."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'docs/evidence/head-temperature-ragged-20261029/legacy'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def functions(source):
    found={}
    def walk(body,prefix=''):
        for n in body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                name=prefix+n.name;found[name]=n;walk(n.body,name+'.')
            elif isinstance(n,(ast.Try,ast.With,ast.For,ast.If)):
                walk(n.body,prefix)
                if hasattr(n,'orelse'):walk(n.orelse,prefix)
                if isinstance(n,ast.Try):walk(n.finalbody,prefix)
    walk(ast.parse(source).body)
    return found

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args()
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    spec=importlib.util.spec_from_file_location('_head_front',ROOT/'tools/bench_biv150_head_temperature_ragged.py');front=importlib.util.module_from_spec(spec);spec.loader.exec_module(front)
    assert not front.plan()['gpu_executed']
    assert all(c['RAGGED_M'] is True for c in front.plan()['rows'].values())
    cases=[(False,False,40000,[]),(True,True,40000,[]),(True,False,29999,[]),(True,False,40000,[123])]
    for c in cases:
        try:front.admit(*c)
        except ValueError:pass
        else:raise AssertionError('bad pre-CUDA admission accepted')
    valid=front.admit(True,False,40000,[])
    assert 'not idle proof' in valid['process_visibility']
    old=functions((LEGACY/'r32-worker.py').read_text());new=functions((ROOT/'tools/biv150_head_temperature_ragged_worker.py').read_text())
    unchanged=['save','worker.head','worker.bits','worker.rng','worker.reset','worker.equal_rng','worker.complete_a','worker.complete_b','worker.observe']
    for n in unchanged:assert ast.dump(old[n],include_attributes=False)==ast.dump(new[n],include_attributes=False),n
    old_check=old['worker.check'];new_check=new['worker.check']
    extra=[n for n in new_check.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='case' and isinstance(t.slice,ast.Constant) and t.slice.value=='CPU_full_storage_canaries' for t in n.targets)]
    assert len(extra)==1
    new_check.body.remove(extra[0]);assert ast.dump(old_check,include_attributes=False)==ast.dump(new_check,include_attributes=False)
    old_factory=functions((LEGACY/'r32-factory.py').read_text());new_factory=functions((ROOT/'tools/biv150_head_temperature_ragged_factory.py').read_text())
    assert ast.dump(old_factory['PreScaledSampler'],include_attributes=False)==ast.dump(new_factory['PreScaledSampler'],include_attributes=False)
    assert (ROOT/'tools/biv150_head_temperature_ragged_timing.py').read_bytes()==(LEGACY/'r32-reused_helpers.py').read_bytes()
    worker=(ROOT/'tools/biv150_head_temperature_ragged_worker.py').read_text()
    assert "check('row_varying_temperature')" in worker
    # Hidden is still logical M, but the admitted baseline reads a full tile.
    # The complete prefix and every row after logical M remain canaries.
    guard_layout=[]
    for rows in (31,41,42,43):
        for row_bytes in (4,4096,522240):
            padding=max(1,16//row_bytes);start=padding*row_bytes;end=(padding+rows)*row_bytes
            tile=(32 if rows==31 else 64) if row_bytes==4096 else rows
            tail=padding+tile-rows
            assert start%16==0 and start>0 and end<(rows+padding+tail)*row_bytes
            assert end-start==rows*row_bytes
            if row_bytes==4096:
                raw=bytearray([71])*((rows+padding+tail)*row_bytes)
                expected=bytes(raw[end:])
                assert len(expected)==tail*row_bytes
                for physical in range(padding+rows,rows+padding+tail):
                    for offset in (0,row_bytes//2,row_bytes-1):
                        pos=physical*row_bytes+offset;raw[pos]^=1
                        assert bytes(raw[end:])!=expected
                        raw[pos]^=1
            guard_layout.append(dict(rows=rows,row_bytes=row_bytes,padding=padding,tail_canary_rows=tail,physical_rows=rows+padding+tail))
    assert 'atol=.001,rtol=.01' in worker and 'rows=list(range(row_count));cols=[0,1,127,65535,130559]' in worker
    assert "for arm in 'ABBA'" in worker and 'pair,8' in worker
    assert 'authority[:padding].clone()' in worker and 'authority[-tail:].clone()' in worker
    assert "reshape(item['storage'].shape[0],-1)" in worker
    assert "if timing_mode else buffer('candidate_output'" in worker
    fpath=ROOT/'tools/biv150_head_temperature_ragged_factory.py'
    fspec=importlib.util.spec_from_file_location('_factory_cpu',fpath);fm=importlib.util.module_from_spec(fspec);fspec.loader.exec_module(fm)
    from types import SimpleNamespace as NS
    factory_cases=[];prelaunch_refusals=[]
    for rows in (31,41,42,43):
        fm.configure(rows)
        config=dict(BLOCK_M=32,BLOCK_N=64,BLOCK_K=64,num_warps=8,num_stages=1,num_ctas=1,EVEN_K=True) if rows==31 else dict(BLOCK_M=64,BLOCK_N=256,BLOCK_K=32,num_warps=16,num_stages=2,num_ctas=1,EVEN_K=True)
        grid=(2040 if rows==31 else 510,1)
        def tensor(shape):return NS(shape=shape,is_contiguous=lambda:True,stride=lambda:(2048,1),element_size=lambda:2,data_ptr=lambda:4096,
            storage_offset=lambda:2048,untyped_storage=lambda:NS(nbytes=lambda:(config['BLOCK_M']+2)*4096))
        x,w,out,temp=tensor((rows,2048)),tensor((130560,2048)),tensor((rows,130560)),tensor((rows,))
        args=(x,w,w,out,rows,130560,2048,2048,1,2048,1,130560,1,0)
        launches=[]
        class Entry:
            arg_names=('input_ptr','weight_ptr','bias_ptr','output_ptr','M','N','K','stride_im','stride_ik','stride_wn','stride_wk','stride_om','stride_on','stride_bn','BIAS','GROUP_M','SPLIT_K')
            _has_flagtune_tuner=False
            def __init__(self):
                self.actual=dict(config)
                cached=NS(pre_hook=None,all_kwargs=lambda:dict(config))
                self.fn=NS(_run_mode='normal',config_table_name=fm.CONFIG_TABLE,keys=['M','N','K'],configs=[cached,cached],arg_names=self.arg_names,
                    get_key=lambda bound:(((bound['M']+31)//32)*32,bound['N'],bound['K'])+('torch.bfloat16',)*4,cache=NS(get=lambda key:cached))
            def __getitem__(self,g):
                def call(*a,**kw):
                    g(dict(self.actual,**kw));launches.append(True)
                    return object(),dict(self.actual)
                return call
        cap=fm.Capture(Entry());cap[grid](*args,BIAS=False,GROUP_M=8,SPLIT_K=1)
        calls=[]
        class Candidate:
            def __getitem__(self,g):
                def call(*a,**kw):calls.append((g,a,kw));return object()
                return call
        fm.launch(NS(head_temperature_kernel=Candidate()),cap.records[0],x,w,temp,out)
        assert calls[0][0]==grid and calls[0][1][0] is x and calls[0][1][1] is w and calls[0][1][3] is out and calls[0][1][-1] is temp
        assert calls[0][1][4:7]==(rows,130560,2048)
        assert calls[0][2]['RAGGED_M'] is True
        assert {k:v for k,v in calls[0][2].items() if k!='RAGGED_M'}==dict(config,BIAS=False,GROUP_M=8,SPLIT_K=1)
        assert len(launches)==1 and cap.prelaunch[0]['autotune_allowed'] is False
        for bad in ('cache_miss','wrong_cached_tile','dynamic_flagtune','forced_policy','wrong_table','physical_table_as_logical','wrong_key','single_config','short_backing','changed_launch_tile','changed_launch_grid','shared_hook','matching_pool_hook'):
            e=Entry();bad_args=list(args);bad_grid=grid
            if bad=='cache_miss':e.fn.cache.get=lambda key:None
            elif bad=='wrong_cached_tile':e.fn.cache.get=lambda key:NS(pre_hook=None,all_kwargs=lambda:dict(config,BLOCK_M=256))
            elif bad=='dynamic_flagtune':e._has_flagtune_tuner=True
            elif bad=='forced_policy':e.fn._run_mode='force_policy'
            elif bad=='wrong_table':e.fn.config_table_name='unknown'
            elif bad=='physical_table_as_logical':e.fn.config_table_name=fm.CONFIG_TABLE+'-0f015d35c97293819201f206510edb40'
            elif bad=='wrong_key':e.fn.get_key=lambda bound:(rows,)
            elif bad=='single_config':e.fn.configs=[None]
            elif bad=='short_backing':bad_args[0]=NS(**{**vars(x),'untyped_storage':lambda:NS(nbytes=lambda:(rows+2)*4096)})
            elif bad=='changed_launch_tile':e.actual['BLOCK_M']=256
            elif bad=='changed_launch_grid':bad_grid=(1,1)
            elif bad=='shared_hook':e.fn.shared_config_pre_hook=lambda x:None
            elif bad=='matching_pool_hook':e.fn.configs=[NS(pre_hook=lambda x:None,all_kwargs=lambda:dict(config)),e.fn.configs[0]]
            before=len(launches)
            try:fm.Capture(e)[bad_grid](*bad_args,BIAS=False,GROUP_M=8,SPLIT_K=1)
            except AssertionError:pass
            else:raise AssertionError(('unsafe baseline admitted',rows,bad))
            assert len(launches)==before,(rows,bad,'device stand-in already launched')
            prelaunch_refusals.append(dict(rows=rows,reason=bad,launches=0))
        factory_cases.append(rows)
    try:fm.configure(48)
    except AssertionError:pass
    else:raise AssertionError('factory admitted pad bucket48')
    sdk=json.loads((LEGACY/'r32-installed-manifest.json').read_text());receipt=json.loads((LEGACY/'r32-archive-receipt.json').read_text());assert sha(LEGACY/'r32-installed-manifest.json')==next(v['sha256'] for v in receipt['files'] if v['path'].endswith('/source/installed-manifest.json'))
    front_file=ROOT/'tools/bench_biv150_head_temperature_ragged.py';compile_file=ROOT/'tools/compile_biv150_head_temperature_ragged.py'
    cli=[]
    for tool,bad in [(front_file,[['--run-gpu'],['--worker'],['--run-gpu','--root-owned-exclusive'],['--row','31']]),(compile_file,[['--compile-only'],['--compile-only','--root-owned-exclusive'],['--row','41']])]:
        for argv in bad:
            # Restore only this repository tools directory under -I.
            command=[sys.executable,'-I','-B','-W','error','-c','import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name="__main__")',str(ROOT/'tools'),str(tool),*argv]
            r=subprocess.run(command,capture_output=True,text=True);assert r.returncode==2,(argv,r.stderr);cli.append(dict(tool=tool.name,argv=argv,rejected_before_framework=True))
    result=dict(gpu_executed=False,legacy_numeric_helpers_AST_unchanged=unchanged,
        legacy_check_AST_unchanged_except_additional_canary=True,original_sampler_proxy_AST_unchanged=True,
        original_timing_helper_byte_identical=True,FP64_thresholds_unchanged=True,
        FP64_all_real_rows_five_columns=True,guard_CPU_complete_rows_and_cloned_boundaries=True,guard_layout=guard_layout,row_varying_temperature_added=True,
        SDK_pin_count=len(sdk['files']),factory_actual_argument_cases=factory_cases,factory_and_compile_plan_RAGGED_M_true=True,
        pad48_factory_rejected=True,admission_refusals=len(cases),CLI_refusals=cli,
        baseline_prelaunch_refusals=prelaunch_refusals,physical_hidden_tail_all_rows_checked=True,
        fresh_IR_loaded_resources_numerics_RNG=False,service_105_throughput=False,
        sources={p.name:sha(p) for p in sorted((ROOT/'tools').glob('*head_temperature_ragged*.py'))})
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    text=json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+'\n'
    if a.out:
        with a.out.open('x') as f:f.write(text)
    print(text,end='')

if __name__=='__main__':main()
