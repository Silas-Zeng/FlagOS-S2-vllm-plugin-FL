#!/usr/bin/env python3
"""Narrow CPU source/parameter checks; no Torch/Triton or target compilation."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
sys.dont_write_bytecode = True
import biv150_konly_q32_w8_compile_source as c


def run():
    checks = []
    def check(ok, label): c.need(ok, label); checks.append(label)
    def reject(call, label):
        try: call()
        except (ValueError, AttributeError): checks.append(label)
        else: raise AssertionError('not rejected: ' + label)
    plan = c.plan()
    check(not subprocess.check_output(['git','-C',str(c.ROOT),'diff',c.BASE,'--','vllm_fl','csrc']),
          'main899 production unchanged')
    check(c.candidate_source() == c.original('biv150_konly_encoded_q32.py'), 'all GPU source bytes unchanged')
    source, parent = c.driver_source(), c._parent.driver_source()
    restored = source
    for before, after in reversed(c.driver_edits()): restored = restored.replace(after, before, 1)
    check(restored == parent, 'complete existing zero-launch driver recovered by inverse')
    tree, old = ast.parse(source), ast.parse(parent)
    for name in ('allocate','save','git','main'):
        get = lambda t: next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name == name)
        check(ast.dump(get(tree)) == ast.dump(get(old)), 'unchanged original function ' + name)
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    check(sum(isinstance(n.func,ast.Attribute) and n.func.attr == 'warmup' for n in calls) == 1,
          'one actual warmup site; no second/override warmup')
    check(not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('propose','grammar','prepare','replay','capture') for n in calls),
          'no legacy hoist/Graph/numeric/performance machinery')
    check('launch.update(num_warps=8, num_stages=1)' in source and 'threads=8*64' in source,
          'only w8/s1 launch and correct512-thread resource label')
    check('c.preserve_unqualified(kernel, fn,' in source and source.index('c.preserve_unqualified(kernel, fn,')
          < source.index('snapshot = adapter.kernel_snapshot(kernel,fn)'), 'raw actual artifacts precede recorder rejection')
    check("type(resource['n_spills']) is int and resource['n_spills'] >= 0" in source,
          'actual nonzero spill is recorded, never replaced with zero')
    for q,bq in ((True,32),(2049,32),(2048,16),(2048,True)):
        reject(lambda q=q,bq=bq:c.geometry(q,bq), 'reject unsupported geometry ' + repr((q,bq)))
    class Tensor:
        def __init__(self, strides): self.strides = strides
        def stride(self, i): return self.strides[i]
    t = {k:Tensor(s) for k,s in dict(q=(2048,128,1),out=(2048,128,1),k=(16384,256,128,1),
         v=(8192,256,128,1),kscale=(1,),table=(8192,1),lengths=(1,),cu_q=(1,),k_descale=(),v_descale=()).items()}
    config, grid = c.geometry(2048,32)
    args = c.kwargs(dict(profile='q32',sequences=1,segments=1,config=config),t,True)
    check(grid == (65,2) and args['BLOCK_M'] == 256 and args['BLOCK_Q'] == 32 and args['TILE_SIZE'] == 64,
          'fixed N1 Q32 grid and original source constants')
    check(args['key_cache_ptr'] is t['k'] and args['value_cache_ptr'] is t['v'] and args['key_code_scale_ptr'] is t['kscale'],
          'original private K/full-base scale/BF16 V roles')
    check(tuple(args['stride_k_cache_'+str(i)] for i in range(4)) == (16384,256,128,1)
          and tuple(args['stride_v_cache_'+str(i)] for i in range(4)) == (8192,256,128,1)
          and args['KV_QUANT_MODE'] == 0 and args['num_seqs'] == 1, 'same byte/BF16 stride units and mode0')
    snapshot = c.module_from_source('cpu_w8_recorder',c.snapshot_source())
    fn_node = next(n for n in ast.parse(c.candidate_source()).body if isinstance(n,ast.FunctionDef) and n.name == c.NAME)
    fn = NS(src=ast.get_source_segment(c.candidate_source(),fn_node)+'\n')
    kernel = NS(src=NS(fn=fn,signature={'q':'*bf16'},constants={},attrs={}),hash='synthetic-CPU-only',
        asm=dict(ttir='fake',ttgir='fake',llir='fake',asm='fake',cubin=b'\x7fELF\x02\x01FAKE'),
        metadata=NS(num_warps=8,num_stages=1,warp_size=64,name=c.NAME))
    check(snapshot.kernel_snapshot(kernel,fn)['launch_geometry'] == [8,1,64], 'private recorder w8/s1 synthetic metadata')
    for field,value in (('num_warps',16),('num_stages',2),('num_stages',True),('warp_size',32)):
        old_value = getattr(kernel.metadata,field);setattr(kernel.metadata,field,value)
        reject(lambda:snapshot.kernel_snapshot(kernel,fn), 'reject recorder '+field+'='+repr(value))
        setattr(kernel.metadata,field,old_value)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)/'unqualified';c.preserve_unqualified(kernel,fn,folder)
        check((folder/'actual.cubin').read_bytes() == kernel.asm['cubin'] and (folder/'actual.llir').read_text() == 'fake',
              'unqualified artifacts retained independently of qualification')
    check(c.memory_observation('68MiB / 32768MiB')['empty_PID_table_is_not_exclusivity_proof'], 'low memory observation is not exclusivity proof')
    reject(lambda:c.memory_observation('28500MiB / 32768MiB'), 'empty namespace table cannot bypass high memory')
    reject(lambda:c.memory_observation('No running processes'), 'unknown memory refuses')
    reject(lambda:c.driver().run(NS(root_owned_exclusive=False)), 'default ownership refuses before CUDA')
    check(not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules), 'CPU paths import no framework')
    return dict(schema='cjj-q32-I8-w8-compile-CPU-v1',completed=True,errors=[],check_count=len(checks),checks=checks,
        plan=plan,framework_imported=False,GPU_executed=False,target_compilation=False,
        note='Synthetic recorder/stride fixtures check host parameters, not actual ABI, binary, hoist or numerics')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path)
    args=parser.parse_args();text=json.dumps(run(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.report: args.report.write_text(text)
    print(text,end='')
