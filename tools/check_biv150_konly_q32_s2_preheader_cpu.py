#!/usr/bin/env python3
"""Narrow real-IR relocation/inverse checks. No target verifier or JIT is invoked."""
import argparse,ast,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.dont_write_bytecode=True
import biv150_konly_q32_s2_preheader_source as c


def run(out):
    checks=[]
    def ck(ok,name):c.need(ok,name);checks.append(name)
    def reject(fn,name):
        try:fn()
        except (ValueError,AttributeError,FileNotFoundError):checks.append(name)
        else:raise AssertionError('accepted '+name)
    proposal=c.reviewed(out);original=(out/'raw-q32-stage2/raw-stage2/actual.ttgir').read_text()
    result=proposal['override'];proof=proposal['proof'];restored=result
    for before,after in reversed(proof['edits']):
        ck(restored.count(after)==1,'unique IR inverse site');restored=restored.replace(after,before,1)
    ck(restored==original,'whole S2 TTGIR exact inverse including all pipeline and debug locations')
    old=original.splitlines();new=result.splitlines()
    ck(new[424].startswith('    %cjj_s2_q_operand = ttg.local_load %5') and new[425]==old[424],
       'direct preheader only, after complete K-scale-V prologue')
    ck('%cjj_s2_q_operand,' in new[440] and len(new)==len(old),'same dot sole operand alias and module line count')
    # Remove exactly the Q-read and normalize only its alias: all other bytes must match.
    stripped=result.replace(new[424]+'\n','',1).replace('%cjj_s2_q_operand,','%10,',1)
    ck(stripped==original.replace(old[438]+'\n','',1),'all non-Q operations complete bytes unchanged')
    for text,label in [(original.replace('mutable','mutable, BAD',1),'changed pipeline'),
                       (original.replace('%V_load_386','%BAD_V',1),'changed V carry'),
                       (original.replace('num = 0','num = 1',1),'changed async wait'),
                       (original.replace('memdesc<256x128xi8, #shared, #smem>','memdesc<256x128xi8, #shared, #smem, mutable>',1),'mutable Q'),
                       (result,'already transformed or S1-like input')]:reject(lambda text=text:c.hoist(text),'reject '+label)
    parent=c.raw.driver_source();source=c.driver_source();a=ast.parse(parent);b=ast.parse(source)
    for name in ('allocate','save','git'):
        get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        ck(ast.dump(get(a))==ast.dump(get(b)),'unchanged full descriptor/resource support '+name)
    rev=source
    for before,after in reversed(c.driver_edits()):rev=rev.replace(after,before,1)
    ck(rev==parent,'whole generated frontend exact inverse to cb908 raw driver')
    calls=[n for n in ast.walk(b) if isinstance(n,ast.Call)]
    ck(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='warmup' for n in calls)==1,'one actual warmup site')
    ck(not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('capture','replay','synchronize','fill_','zero_','normal_') for n in calls),'no candidate launch/data initialization or timing/Graph added')
    ck("launch['ir_override'] = str(override.resolve())" in source and "first.metadata.ir_override == str(override.resolve())" in source,'own override path before warmup and actual metadata binding')
    ck("not args.out.resolve().is_relative_to(args.reviewed_out.resolve())" in source,'new output cannot mutate readonly original tree')
    ck(source.index('reviewed = c.reviewed')<source.index('import torch') and "c.reviewed(args.reviewed_out)==reviewed" in source,'original evidence pinned before CUDA and unchanged after compile')
    # Same real relative compiler lookup/false-manager check as already accepted raw124.
    ck("compiler_knobs.compilation.override is False" in source and "create_binder.__globals__['__package__']" in source,'original cold-JIT override guard retained')
    reject(lambda:c.driver().run(NS(root_owned_exclusive=False)),'false Root ownership before device')
    reject(lambda:c.driver().run(NS(root_owned_exclusive=True,commit='0'*40,out=Path('/private/tmp/cjj-s2-preheader-no-create'))),'wrong freeze before device')
    plan=c.plan();ck(plan['num_seqs']==1 and plan['query_rows']==2048 and plan['num_stages']==2 and plan['planned_warmups']==1,'single original N1/Q2048 S2 scope')
    ck(all(plan[k] is False for k in ('new_actual_Q_hoist_verified','pipeline_verified','actual_resources_verified','numerical_verified','Graph_verified','performance_verified','service_verified')),'actual compilation/synchronization/resources/numeric remain pending')
    ck(not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules),'stdlib only no framework import')
    return dict(schema='cjj-q32-s2-preheader-CPU-v1',passed=True,count=len(checks),checks=checks,plan=plan,
        proof=proof,actual_input_sha256=c.ORIGINALS,CPU_read_real_IR=True,
        target_MLIR_verifier_executed=False,target_compilation=False,GPU_executed=False,
        source_files={p.name:c.sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*q32_s2_preheader*.py'))})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewed-out',type=Path,required=True);p.add_argument('--report',type=Path)
    a=p.parse_args();value=run(a.reviewed_out);text=json.dumps(value,sort_keys=True,indent=2)+'\n'
    if a.report:a.report.write_text(text)
    print(text,end='')
