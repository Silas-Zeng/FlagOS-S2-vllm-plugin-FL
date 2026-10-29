"""Reuse ce905 CPU gates; test only new same-JIT cache-transition boundaries."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import biv150_konly_multi2_service_numeric_source as b


def main(a):
    raw=subprocess.check_output(['git','-C',str(b.c.ROOT),'show',b.production.FROZEN+':tools/check_biv150_konly_multi2_numeric_cpu.py']).decode()
    b.c.need(b.c.sha(raw)=='20e7aaf2d7bd46fa0d240a9183ded0d7c8c6a3d62c87e4c7488508aa8446ddb6','frozen original CPU checks')
    text,proof=b.transform(raw,[
        ('import biv150_konly_multi2_hoist_source as c\nimport biv150_konly_multi2_numeric_source as s\nimport biv150_konly_multi2_numeric_contract as f',
         'import biv150_konly_multi2_service_numeric_source as b\nc,f=b.c,b.f\ns=NS(adapted_driver=b.driver_source,original=b.source.original,inherited_contract=b.source.inherited_contract)'),
        ("restored==s.original('probe_biv150_konly_q64_screen.py')","restored==b.source.adapted_driver()[0]"),
        ("'exact_executed_116_driver_inverse'","'exact_executed_ce905_driver_inverse'"),
        ("'actual_44_originals_and_compile_review'","'actual_52_originals_and_compile_review'"),
        ("a.raw/'out/multi2'/role/'actual.cubin'","a.raw/'out'/b.BINDING/role/'actual.cubin'"),
        ("'no_production_diff_main899'","'no_production_diff_frozen_d5d'"),
    ])
    reused=b.c.module_from_source('_cjj_reused_numeric_cpu',text).main(a)
    checks={}
    def ok(name,value):b.c.need(value,name);checks[name]=True
    def bad(name,call):
        try:call()
        except ValueError:checks[name]=True;return
        raise AssertionError(name+' accepted')
    # Execute the actual new memory-map statements, with fake opaque objects;
    # the inherited SDK/loader/compile guard is AST-compared, not pretended run.
    tree=ast.parse(b.cache_source());warm=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='warmup')
    first=next(i for i,n in enumerate(warm.body) if isinstance(n,ast.If))
    before_nodes=warm.body[first:first+2]
    after_nodes=warm.body[-4:-1]
    ns={'c':b.c}
    def make(name,args,nodes,ret):
        f=ast.parse('def '+name+'('+args+'):\n pass').body[0]
        f.body=nodes+[ast.Return(value=ast.Name(id=ret,ctx=ast.Load()))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'actual-cache-transition','exec'),ns)
    make('before','memory_cache,expected_memory_cache',before_nodes,'before')
    make('after','memory_cache,before,kernel',after_nodes,'added')
    one,two,foreign=object(),object(),object();empty={}
    first_map=ns['before'](empty,None);empty['old']=one;ns['after'](empty,first_map,one)
    second_map=ns['before'](empty,{'old':one});empty['new']=two;ns['after'](empty,second_map,two)
    ok('exact_empty_old_old_plus_new_transition',empty=={'old':one,'new':two})
    for label,mem,expected in [
        ('first_nonempty',{'old':one},None),('no_expected_object',{},{}),
        ('unexpected_key',{'wrong':one},{'old':one}),('cloned_prior_object',{'old':foreign},{'old':one}),
        ('missing_prior',{}, {'old':one}),('extra_before',{'old':one,'x':foreign},{'old':one}),
        ('two_expected',{'old':one,'x':two},{'old':one,'x':two})]:
        bad('reject_'+label,lambda mem=mem,expected=expected:ns['before'](mem,expected))
    for label,mem in [('cleared',{'new':two}),('replaced_old',{'old':foreign,'new':two}),
        ('no_growth',{'old':one}),('multiple_growth',{'old':one,'new':two,'x':foreign}),
        ('wrong_new_object',{'old':one,'new':foreign})]:
        bad('reject_'+label,lambda mem=mem:ns['after'](mem,{'old':one},two))
    parent=ast.parse(b.parent_cache_source());functions=lambda t:{n.name:ast.dump(n) for n in t.body if isinstance(n,ast.FunctionDef)}
    for name in ('prepare','validate_group','validate_knobs','prohibit_backend_compile'):
        ok('unchanged_cache_'+name,functions(parent)[name]==functions(tree)[name])
    original=ast.parse(b.original('biv150_konly_multi2_numeric_contract.py'));new=ast.parse(b.contract_source()[0])
    for name in ('phase_geometry','reference_sources','load_reference','references'):
        ok('unchanged_math_'+name,functions(original)[name]==functions(new)[name])
    driver=b.driver_source()[0]
    ok('normal_production_fn_once',driver.count('fn=bridge.production_fn()')==1 and 'cjj_private_multi2_compile_candidate' not in driver)
    ok('original_override_and_no_prepare',"a.raw/'out'/bridge.BINDING/'override.ttgir'" in driver and '.prepare(fn' not in driver)
    ok('previous_map_only_after_loaded_gate',driver.index('resources=f.check_loaded')<driver.index("if role=='unhoisted':"))
    ok('backend_guard_retained','with prohibit_backend_compile(type(backend)):\n        kernel=call()' in b.cache_source())
    ok('no_framework_or_GPU',not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules))
    return dict(completed=True,errors=[],check_count=reused['check_count']+len(checks),reused_cpu=reused,
        new_checks=checks,plan=b.plan(),CPU_only=True,GPU_executed=False,numeric_verified=False,
        script_sha256=b.c.sha(Path(__file__).read_bytes()),CPU_scope='Inherited numeric CPU gates plus exact new cache-map statements with opaque fake objects. Full warmup/SDK/production import and real handles require device run.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path);p.add_argument('--review',type=Path);a=p.parse_args()
    print(json.dumps(main(a),indent=2,sort_keys=True,allow_nan=False))
