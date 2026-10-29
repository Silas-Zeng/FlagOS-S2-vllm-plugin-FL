#!/usr/bin/env python3
"""CPU source/metadata contract only; no actual SDK import or warmup execution."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
sys.dont_write_bytecode = True
import biv150_konly_q32_s2_compile_source as c


def run():
    checks=[]
    def check(ok,label): c.need(ok,label); checks.append(label)
    def reject(call,label):
        try:call()
        except (ValueError,AttributeError):checks.append(label)
        else:raise AssertionError('not rejected: '+label)
    plan=c.plan()
    check(not subprocess.check_output(['git','-C',str(c.ROOT),'diff',c.BASE,'--','vllm_fl','csrc']),
          'main899 production unchanged')
    check(c.candidate_source()==c.original('biv150_konly_encoded_q32.py'),'complete original Q32 math/source bytes')
    parent=c.parent('probe_biv150_konly_q64_compile.py'); source=c.driver_source()
    check(len(c.driver_edits(parent))==24,'24 exact-count edits and complete byte inverse')
    tree=ast.parse(source); old=ast.parse(parent)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    warms=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='warmup']
    check(len(warms)==1 and all(k.arg!='ir_override' for k in warms[0].keywords),'exactly one warmup without override keyword')
    check(not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('propose','grammar','alpha','prepare','capture','replay')
                  for n in calls),'no hoist/proof/Graph or performance calls')
    for name in ('allocate','save','git','main'):
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        check(ast.dump(a)==ast.dump(b),'parent function unchanged: '+name)
    check("for bq in (32,):" in source and "for label, kernel in (('raw-stage2',first),):" in source,
          'single geometry and single actual handle')
    check(c.geometry(2048,32)==(dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,IS_3D=False),(65,2)),
          'fixed N1 Q2048 BQ32 BM256 K64 grid')
    for q,bq in ((1024,32),(2049,32),(True,32),(2048,64),(2048,True)):
        reject(lambda q=q,bq=bq:c.geometry(q,bq),'reject geometry '+repr((q,bq)))
    reject(lambda:c.driver().run(NS(root_owned_exclusive=False)),'ownership false rejects before device work')
    reject(lambda:c.driver().run(NS(root_owned_exclusive=True,commit='0'*40,
           out=Path('/private/tmp/cjj-q32-s2-cpu-must-not-create-20261029'))),'wrong commit rejects before device work')
    snapshot=c.module_from_source('cpu_private_s2_snapshot',c.snapshot_source())
    original=c.module_from_source('cpu_original_s1_snapshot',c.original('biv150_konly_encoded_q32_adapter.py'))
    check(c.snapshot_source().replace('== (16, 2, 64)','== (16, 1, 64)')
          ==c.original('biv150_konly_encoded_q32_adapter.py'),'snapshot helper single scope edit inverse')
    # Explicitly synthetic metadata/artifacts, not a real ELF or compiler receipt.
    fn_node=next(n for n in ast.parse(c.candidate_source()).body if isinstance(n,ast.FunctionDef) and n.name==c.NAME)
    fn=NS(src=ast.get_source_segment(c.candidate_source(),fn_node)+'\n')
    kernel=NS(src=NS(fn=fn,signature={'q':'*bf16'},constants={'num_seqs':1},attrs={}),hash='synthetic-cpu-only',
        asm=dict(ttir='fake ttir',ttgir='fake ttgir',llir='fake llir',asm='fake asm',cubin=b'\x7fELF\x02\x01FAKE'),
        metadata=NS(num_warps=16,num_stages=2,warp_size=64,name=c.NAME))
    value=snapshot.kernel_snapshot(kernel,fn)
    check(value['source_digest']==c.FUNCTION_SHA and value['launch_geometry']==[16,2,64],
          'synthetic snapshot retains original source and exact stage2 metadata')
    reject(lambda:original.kernel_snapshot(kernel,fn),'unchanged production recorder rejects stage2')
    for stages in (1,3,None):
        kernel.metadata.num_stages=stages
        reject(lambda:snapshot.kernel_snapshot(kernel,fn),'private recorder rejects stage '+repr(stages))
    kernel.metadata.num_stages=2;kernel.src.fn=object()
    reject(lambda:snapshot.kernel_snapshot(kernel,fn),'recorder rejects different selected JIT object')
    # Execute only the actual prewarmup override-manager guard with CPU fakes.
    assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='compiler_knobs')
    guard=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and any(isinstance(a,ast.Constant) and a.value=='actual compiler override manager must be disabled' for a in n.value.args))
    body=compile(ast.fix_missing_locations(ast.Module(body=[assign,guard],type_ignores=[])),'<prewarmup guard only>','exec')
    def guard_fake(value, *, missing_knobs=False, missing_compile=False):
        # Cold JIT has no .compile: actual SDK assigns it lazily in create_binder.
        # Its own knobs deliberately differ; only the imported compiler is checked.
        jit_globals={'__package__':'cpu_fixture.runtime','knobs':NS(compilation=NS(override=True))}
        exec('def create_binder(self): raise AssertionError("binder must not execute")',jit_globals)
        fn=type('ColdJit',(),{'create_binder':jit_globals['create_binder']})()
        assert not hasattr(fn,'compile')
        compiler_globals={} if missing_knobs else {'knobs':NS(compilation=NS(override=value))}
        exec('def compile(): raise AssertionError("compiler must not execute")',compiler_globals)
        compiler=NS() if missing_compile else NS(compile=compiler_globals['compile'])
        imports=[]
        def import_same_package(name, *, package):
            assert (name,package)==('..compiler','cpu_fixture.runtime')
            imports.append((name,package));return compiler
        fake_c=NS(need=c.need,importlib=NS(import_module=import_same_package))
        exec(body,dict(fn=fn,c=fake_c))
        assert imports==[('..compiler','cpu_fixture.runtime')] and not hasattr(fn,'compile')
    guard_fake(False);checks.append('cold SDK lifecycle: same relative compiler import, no binder/compile call')
    for value in (True,None,0):reject(lambda value=value:guard_fake(value),'reject unknown/enabled manager '+repr(value))
    reject(lambda:guard_fake(False,missing_knobs=True),'reject compiler without knobs')
    reject(lambda:guard_fake(False,missing_compile=True),'reject package without compile export')
    check(hasattr(c.importlib,'import_module'),'stdlib relative import API available')
    check("hasattr(first.metadata,'ir_override') and first.metadata.ir_override is None" in source,
          'actual postcompile metadata must explicitly have no override')
    check(not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules),'no framework imports')
    return dict(schema='cjj-q32-stage2-raw-compile-cpu-v1',completed=True,errors=[],check_count=len(checks),checks=checks,
        plan=plan,source_files={p.name:c.sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*q32_s2_compile*.py'))},
        fixtures='synthetic CPU metadata only; original warmup body has not been executed',
        GPU_executed=False,target_compilation=False,hoist_qualified=False,numerical_verified=False,performance_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=run();text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.report:args.report.write_text(text)
    print(text,end='')
