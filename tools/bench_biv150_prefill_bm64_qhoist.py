"""Compare bounded query tiles/pipelines with verified Q-fragment hoisting.

The default control is the actual main2412207 Q32 policy (BM256/w16/K64/s1).
Use --baseline bm128 only to reproduce older experiments. Only diagnostic
private JIT caches are changed. No installed or serving source is edited.
Graphs time the warmed, proved kernels, bypassing capture fallback deliberately;
these are component measurements, never official service scores.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

CASES = {
    'fresh1024': ([1024], [1024]),
    'fresh2048': ([2048], [2048]),
    'minimum16k': ([1024], [16384]),
    'chunk4k': ([2048], [4096]),
    'chunk16k': ([2048], [16384]),
    'suffix4k': ([1985], [4096]),
    'suffix16k': ([2017], [16384]),
    'mixed4k': ([1] * 63 + [1985], [4096] * 63 + [4096]),
    'mixed16k': ([1] * 31 + [2017], [16384] * 31 + [16384]),
}


def q_allocation(text, rows=64):
    """Q and V can both be64x128; only Q's lhs-dot load dominates the loop."""
    loop = text.index('= scf.for ')
    candidates = []
    for match in re.finditer(rf'(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<{rows}x128xbf16', text):
        if match.start() < loop and re.search(
            r'ttg.local_load ' + re.escape(match[1]) + r'(?=\s| :).+dot_op<\{opIdx = 0', text
        ):
            candidates.append(match[1])
    return candidates


def adapted_manager(original, rows=64, warps=8, kv_tile=64, stages=1):
    source = Path(original.__file__).read_text()
    first = '''allocations = re.findall(
        r"(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<128x128xbf16",
        text,
    )'''
    second = '''alloc = re.findall(
        r"(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<128x128xbf16", ttgir
    )'''
    parameterized = 'block_m' in inspect.signature(original.transform_ir).parameters
    if parameterized:
        first = first.replace('r"(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<128x128xbf16"', 'rf"(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<{block_m}x128xbf16"')
        second = second.replace('r"(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<128x128xbf16"', 'rf"(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<{block_m}x128xbf16"')
    edits = {
        first: 'allocations = _probe_q_allocation(text)',
        second: 'alloc = _probe_q_allocation(ttgir)',
        '"BLOCK_M": 128,': f'"BLOCK_M": {rows},',
        '"BLOCK_Q": 16,': f'"BLOCK_Q": {rows//8},',
        '"num_warps": 8,': f'"num_warps": {warps},',
        '"num_stages": 1,': f'"num_stages": {stages},',
        '"TILE_SIZE": 64,': f'"TILE_SIZE": {kv_tile},',
        '!= (8, 1)': f'!= ({warps}, {stages})',
    }
    if parameterized:
        del edits['"BLOCK_M": 128,']
        del edits['"BLOCK_Q": 16,']
    if 'num_warps' in inspect.signature(original.Manager).parameters:
        for key in ('"num_warps": 8,', '"num_stages": 1,', '"TILE_SIZE": 64,', '!= (8, 1)'):
            del edits[key]
    for old, new in edits.items():
        if source.count(old) != 1:
            raise RuntimeError('Audited manager anchor changed: ' + old)
        source = source.replace(old, new)
    module = types.ModuleType('_biv150_bm64_private_manager')
    module.__file__ = __file__
    module._probe_q_allocation = lambda text: q_allocation(text,rows)
    sys.modules[module.__name__] = module
    exec(compile(source, '<biv150-bm64-private-manager>', 'exec'), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


class Tap:
    def __init__(self, target):
        self.target = target
        self.grid = self.kwargs = self.compiled = None

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            if args:
                raise RuntimeError('Unexpected positional kernel arguments')
            self.grid, self.kwargs = grid, kwargs.copy()
            self.compiled = self.target[grid](**kwargs)
            return self.compiled
        return invoke


def comparison_baseline(fixture, mode):
    function, manager, states = fixture.install_baseline()
    if mode == 'q32':
        module = importlib.import_module('vllm_fl.patches.biv150_prefill_q32')
        states['biv150_prefill_q32'] = module.install()
        if not states['biv150_prefill_q32'].get('installed'):
            raise RuntimeError('Current Q32 baseline installation failed')
        policy = manager.holder[manager.kernel_key]
        private = module._MANAGER
        if (type(policy) is not module._Policy or policy.manager is not private
                or (private.block_m, private.num_warps, private.tile_size,
                    private.num_stages) != (256, 16, 64, 1)):
            raise RuntimeError('Current serving policy/configuration changed')
        # Policy reads its private manager.wrapper at every eligible invocation.
        # Replacing the outer holder here would silently bypass that policy.
        tap = Tap(private.wrapper)
        private.wrapper = tap
        return function, private, states, tap
    tap = Tap(manager.wrapper)
    manager.holder[manager.kernel_key] = tap
    return function, manager, states, tap


def selected_configuration(tap, manager):
    expected = dict(BLOCK_M=manager.block_m, BLOCK_Q=manager.block_m // 8,
                    num_warps=manager.num_warps, TILE_SIZE=manager.tile_size,
                    num_stages=manager.num_stages, IS_3D=False)
    if tap.kwargs is None or any(tap.kwargs.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Selected kernel does not match comparison configuration')
    if tap.compiled is None or tap.compiled.hash not in manager.optimized:
        raise RuntimeError('Q-hoist was not selected')
    return dict(expected, grid=list(tap.grid), hash=tap.compiled.hash,
                loaded_resources=importlib.import_module(
                    'vllm_fl.patches.biv150_q_fragment_service')._resources(tap.compiled))


def candidate_function(attention, backend, qmodule, rows=64, warps=8, kv_tile=64, stages=1):
    jit = importlib.import_module('triton.runtime.jit')
    from vllm_fl.patches.biv150_prefill_normalized import make_normalized
    normalized, _ = make_normalized(attention)
    source = inspect.getsource(inspect.unwrap(attention.unified_attention))
    for old, new in [('BLOCK_M = 128', f'BLOCK_M = {rows}'), ('BLOCK_Q = 16', f'BLOCK_Q = {rows//8}'), ('biv150_prefill_tile = 64\n        launch_num_warps = 8', f'biv150_prefill_tile = 64\n        launch_num_warps = {warps}')]:
        if source.count(old) != 1:
            raise RuntimeError('Audited launch anchor changed: ' + old)
        source = source.replace(old, new)
    assert source.count('biv150_prefill_tile = 64') == 1
    source = source.replace('biv150_prefill_tile = 64', f'biv150_prefill_tile = {kv_tile}')
    assert source.count('launch_num_stages = 1') == 1
    source = source.replace('launch_num_stages = 1', f'launch_num_stages = {stages}')
    namespace = dict(normalized.__globals__)
    exec(compile(source, '<biv150-bm64-launch>', 'exec'), namespace)
    function = namespace['unified_attention']
    raw = namespace['kernel_unified_attention']
    clone = jit.JITFunction(
        raw.fn, version=raw.version, do_not_specialize=list(raw.do_not_specialize),
        do_not_specialize_on_alignment=list(raw.do_not_specialize_on_alignment),
        debug=raw.debug, noinline=raw.noinline, repr=raw._repr,
        launch_metadata=raw.launch_metadata,
    )
    extra = {'block_m': rows} if 'block_m' in inspect.signature(qmodule.Manager).parameters else {}
    if 'num_warps' in inspect.signature(qmodule.Manager).parameters:
        extra.update(num_warps=warps, num_stages=stages, tile_size=kv_tile)
    manager = qmodule.Manager(attention, backend, namespace, 'kernel_unified_attention', raw, clone, **extra)
    manager.install()
    return function, manager


def main(args):
    if not args.run_gpu:
        print(json.dumps({'cases': CASES, 'baseline':args.baseline,
                          'baseline_config':[256,32,16,64,1] if args.baseline=='q32' else [128,16,8,64,1],
                          'candidate_config':[args.bm,args.bm//8,args.warps,args.kv_tile,args.stages]}))
        return
    if args.baseline == 'q32' and any(len(CASES[name][0]) != 1 or min(CASES[name][0]) < 1024
                                     for name in args.case or ['chunk4k','chunk16k']):
        raise RuntimeError('Current Q32 comparison requires its actual single-sequence >=1024-row scope')
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',9031)) != 0, 'Service running'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True), 'GPU busy'
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import biv150_prefill_probe_fixtures as fixture
    fixture.torch = torch
    torch.set_num_threads(4)
    baseline, baseline_manager, installations, baseline_tap = comparison_baseline(fixture,args.baseline)
    qoriginal = importlib.import_module('vllm_fl.patches.biv150_q_fragment_service')
    if args.baseline == 'q32':
        # First test the same exact inverse/lowered guards as production, without
        # quietly introducing a second transform while changing the pipeline.
        qmodule, adapted_sha = qoriginal, None
    else:
        qmodule, adapted_sha = adapted_manager(qoriginal,args.bm,args.warps,args.kv_tile,args.stages)
    attention = importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    backend = importlib.import_module('vllm.v1.attention.backends.triton_attn')
    candidate, candidate_manager = candidate_function(attention, backend, qmodule,args.bm,args.warps,args.kv_tile,args.stages)
    taps = {'baseline':baseline_tap}
    for label, manager in [('candidate',candidate_manager)]:
        tap = Tap(manager.wrapper)
        manager.holder[manager.kernel_key] = tap
        taps[label] = tap
    result = dict(completed=False, official_score=False, service_installed=False,
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  baseline_mode=args.baseline, baseline_config=[baseline_manager.block_m,baseline_manager.num_warps,baseline_manager.tile_size,baseline_manager.num_stages],
                  production_transform_unchanged=args.baseline=='q32',
                  adapted_manager_sha256=adapted_sha, candidate_tile=[args.bm,args.bm//8], candidate_warps=args.warps, candidate_kv_tile=args.kv_tile, candidate_stages=args.stages, installations=installations, cases=[], errors=[])
    def save():
        temp=args.out/'result.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.out/'result.json')
    save()
    try:
        for name in args.case or ['chunk4k','chunk16k']:
            qlens, klens = CASES[name]
            row=dict(name=name, qlens=qlens, klens=klens, checks=[], rounds=[], completed=False)
            result['cases'].append(row);save()
            host=fixture.cpu_inputs(qlens,klens,20261025,0)
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                device={k:v.cuda() for k,v in host.items()}
                key,value=device['packed'].unbind(1)
                out=torch.empty_like(device['q'])
            stream.synchronize()
            def call(function):
                function(q=device['q'],k=key,v=value,out=out,cu_seqlens_q=device['cu_q'],
                         max_seqlen_q=max(qlens),seqused_k=device['lengths'],max_seqlen_k=max(klens),
                         softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=device['table'],
                         softcap=0,q_descale=None,k_descale=device['k_scale'],v_descale=device['v_scale'])
            graphs={}
            for pattern in (0,1):
                host=fixture.cpu_inputs(qlens,klens,20261025,pattern)
                with torch.cuda.stream(stream):
                    for k,v in host.items():device[k].copy_(v)
                    out.fill_(float('nan'));call(baseline)
                stream.synchronize();expected=out.cpu().clone()
                with torch.cuda.stream(stream):
                    out.fill_(float('nan'));call(candidate)
                stream.synchronize();actual=out.cpu().clone()
                indices,reference=fixture.reference(host,qlens,klens,pattern==0,False)
                check=dict(pattern=pattern,
                           direct=fixture.compare(actual,expected,.005,.005),
                           candidate_cpu=fixture.compare(actual[indices],reference,.005,.005),
                           baseline_cpu=fixture.compare(expected[indices],reference,.005,.005),
                           reference_guard=fixture.reference_guard(actual[indices],expected[indices],reference),graphs=[])
                row['checks'].append(check);save()
                if not check['direct']['passed'] or not check['reference_guard']['passed']:
                    torch.save(dict(input_q=host['q'],actual=actual,baseline=expected,reference_rows=indices,reference=reference),args.out/(name+'-failure.pt'))
                    raise RuntimeError(name+': numerical gate failed')
                for label,manager in [('baseline',baseline_manager),('candidate',candidate_manager)]:
                    tap=taps[label]
                    row[label+'_selection'] = selected_configuration(tap,manager)
                    save()
                    if label not in graphs:
                        with torch.cuda.stream(stream):
                            for _ in range(3):manager.candidate[tap.grid](**tap.kwargs)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=stream):
                                for _ in range(4):manager.candidate[tap.grid](**tap.kwargs)
                        stream.synchronize();graphs[label]=graph
                    with torch.cuda.stream(stream):
                        out.fill_(float('nan'));graphs[label].replay()
                    stream.synchronize()
                    comparison=fixture.compare(out.cpu(),expected if label=='baseline' else actual,0,0)
                    check['graphs'].append(dict(route=label,**comparison))
                    if not comparison['passed']:raise RuntimeError(label+': changed-input Graph differs')
            for index,order in enumerate([['baseline','candidate'],['candidate','baseline'],['baseline','candidate']]):
                timed=dict(index=index,order=order)
                for label in order:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        graphs[label].replay();graphs[label].replay();begin.record()
                        for _ in range(16):graphs[label].replay()
                        end.record()
                    end.synchronize();timed[label+'_ms']=begin.elapsed_time(end)/64
                timed['speedup']=timed['baseline_ms']/timed['candidate_ms'];row['rounds'].append(timed)
            row['median_speedup']=statistics.median(r['speedup'] for r in row['rounds'])
            row['baseline_proof']=baseline_manager.snapshot()
            row['candidate_proof']=candidate_manager.snapshot()
            row['completed']=True;save()
            print(json.dumps(dict(case=name,median_speedup=row['median_speedup'],rounds=row['rounds'])),flush=True)
        result['completed']=True
    except Exception as error:
        import traceback
        result['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        # Keep failure resources and transformed IR before TemporaryDirectory
        # cleanup, including configurations rejected before a timed case ends.
        result['terminal_manager_proofs'] = {}
        for label, manager in [('baseline', baseline_manager), ('candidate', candidate_manager)]:
            try:
                result['terminal_manager_proofs'][label] = manager.snapshot()
                shutil.copytree(manager.artifact_dir, args.out/'compiler-ir'/label,
                                dirs_exist_ok=True)
                compiled = taps[label].compiled
                if compiled is not None:
                    for name in ('ttgir','llir'):
                        (args.out/'compiler-ir'/label/('actual.'+name)).write_text(compiled.asm[name])
                    result['terminal_manager_proofs'][label]['actual_loaded_resources'] = qoriginal._resources(compiled)
            except Exception as diagnostic_error:
                result.setdefault('diagnostic_copy_errors', []).append(
                    dict(route=label, error=repr(diagnostic_error)))
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--baseline',choices=['q32','bm128'],default='q32')
    parser.add_argument('--bm',type=int,choices=[64,128,256,512],default=256)
    parser.add_argument('--warps',type=int,choices=[4,8,16],default=16)
    parser.add_argument('--kv-tile',type=int,choices=[64,128],default=64)
    parser.add_argument('--stages',type=int,choices=[1,2],default=2)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--case',action='append',choices=CASES)
    args=parser.parse_args()
    if args.run_gpu and args.out is None:parser.error('--out required for GPU execution')
    main(args)
