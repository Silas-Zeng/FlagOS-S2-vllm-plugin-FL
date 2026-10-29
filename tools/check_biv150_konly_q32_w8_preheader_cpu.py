#!/usr/bin/env python3
"""Small inverse/entry check over the actual bound IR, no mirrored device suite."""
import argparse,ast,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.dont_write_bytecode=True
import biv150_konly_q32_w8_preheader_source as c


def check(path):
    before=path.read_text();after,proof=c.hoist(before);checks=[]
    assert proof['moved_operations']==1 and proof['exact_byte_inverse'];checks.append('actual input and fixed one-line override SHA / exact inverse')
    assert proof['all_other_text_math_masks_memory_and_layout_unchanged'];checks.append('allocation312 / unique memory+operand uses / no other text change')
    try:c.hoist(before.replace('num-warps" = 8','num-warps" = 16',1))
    except ValueError:checks.append('unknown or w16 input refused')
    else:raise AssertionError('changed input accepted')
    source=c.driver_source();old=c.raw.driver_source();tree=ast.parse(source);parent=ast.parse(old)
    for name in ('allocate','save','git'):
        get=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(get(tree))==ast.dump(get(parent));checks.append('original function unchanged '+name)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    warm=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='warmup']
    assert len(warm)==1 and any(k.arg=='ir_override'for k in warm[0].keywords);checks.append('one warmup with explicit fixed override; no base recompile')
    assert "Path(reviewed['source'])" in source and "result['tensor_descriptors'] == reviewed['tensor_descriptors']" in source;checks.append('original source/debug root and actual descriptors retained')
    assert source.index('c.preserve_unqualified(kernel, fn,')<source.index('snapshot = adapter.kernel_snapshot(kernel,fn)')<source.index('c.actual_ttgir(first,override)');checks.append('actual artifacts saved before unknown metadata/IR rejection')
    assert "compiler_knobs.compilation.override is False"in source and "'ir_override' not in launch"in source;checks.append('inherited external-override guards retained')
    try:c.driver().run(NS(root_owned_exclusive=False))
    except ValueError:checks.append('no Root ownership rejects before CUDA/output')
    else:raise AssertionError('ownership required')
    assert not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules);checks.append('no framework/device imports')
    return dict(schema='cjj-w8-exact-preheader-CPU-v1',passed=True,checks=checks,check_count=len(checks),
        plan=c.plan(),proof=proof,actual_input_file_sha256=c.sha(before),GPU_executed=False,
        actual_override_compiled=False,actual_Q_publish_bank_sync_verified=False,numerical_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actual-ttgir',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();text=json.dumps(check(a.actual_ttgir),sort_keys=True,indent=2)+'\n'
    if a.out:a.out.write_text(text)
    print(text,end='')
