"""Two-sequence prefill probe under the real current policy chain; opt-in GPU only.

Private component experiment; no serving registration or production change.
CPU/query cases derive from shared PR64 protocol032d6d3; early-Q is not used.
"""
import argparse,hashlib,importlib,inspect,json,shutil,socket,statistics,subprocess,sys,time,types
from pathlib import Path

CASES = {
    'double4k-big-first': ([1945,88], [4096,4096]),
    'double4k-small-first': ([88,1945], [4096,4096]),
    'double16k-big-first': ([1994,39], [16384,16384]),
    'double16k-small-first': ([39,1994], [16384,16384]),
}

def validate_case(qlens,klens):
    if (len(qlens)!=2 or len(klens)!=2 or min(qlens)<=1 or max(qlens)<1024
            or sum(qlens)<1024 or any(k<q for k,q in zip(klens,qlens))):
        raise ValueError('Two non-decode sequences with a >=1024-row query are required')

class Tap:
    def __init__(self, target, maxnreg=None):
        self.target = target
        self.maxnreg = maxnreg
        self.grid = self.kwargs = self.compiled = None

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            if args:
                raise RuntimeError('Unexpected positional kernel arguments')
            if self.maxnreg is not None:
                if 'maxnreg' in kwargs:
                    raise RuntimeError('Conflicting register constraint')
                kwargs = dict(kwargs, maxnreg=self.maxnreg)
            self.grid, self.kwargs = grid, kwargs.copy()
            self.compiled = self.target[grid](**kwargs)
            return self.compiled
        return invoke

def kernel_configuration(tap, manager):
    expected = dict(BLOCK_M=manager.block_m, BLOCK_Q=manager.block_m // 8,
                    num_warps=manager.num_warps, TILE_SIZE=manager.tile_size,
                    num_stages=manager.num_stages, IS_3D=False)
    if tap.kwargs is None or any(tap.kwargs.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Selected kernel does not match comparison configuration')
    if tap.compiled is None or tap.compiled.hash not in manager.optimized:
        raise RuntimeError('Q-hoist was not selected')
    occupancy = {}
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        query = lib.cuOccupancyMaxActiveBlocksPerMultiprocessor
        query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                          ctypes.c_int, ctypes.c_size_t]
        blocks = ctypes.c_int()
        threads = manager.num_warps * tap.compiled.metadata.warp_size
        rc = query(ctypes.byref(blocks), ctypes.c_void_p(tap.compiled.function),
                   threads, tap.compiled.metadata.shared)
        occupancy = dict(returncode=rc, active_blocks=blocks.value if rc == 0 else None,
                         threads=threads, dynamic_shared=tap.compiled.metadata.shared,
                         limitation='driver reported capacity, not measured active residency')
    except Exception as error:
        occupancy = dict(error=repr(error))
    return dict(expected, grid=list(tap.grid), hash=tap.compiled.hash,
                requested_maxnreg=tap.kwargs.get('maxnreg'), driver_occupancy=occupancy,
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

def comparison_baseline(fixture):
    function,manager,states=fixture.install_baseline()
    q32=importlib.import_module('vllm_fl.patches.biv150_prefill_q32')
    states['biv150_prefill_q32']=q32.install()
    if not states['biv150_prefill_q32'].get('installed'):
        raise RuntimeError('Current default Q32 must remain installed')
    qpolicy=manager.holder[manager.kernel_key]
    qm=q32._MANAGER
    if (type(qpolicy) is not q32._Policy or qpolicy.manager is not qm
            or qpolicy.base is not manager.wrapper
            or (qm.block_m,qm.num_warps,qm.tile_size,qm.num_stages)!=(256,16,64,1)):
        raise RuntimeError('Unknown Q32 default/fallback binding')
    parents=[('q32',qpolicy,qm)]
    name='vllm_fl.patches.biv150_prefill_k128'
    try:
        newer=importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name!=name:raise
    else:
        states['biv150_prefill_k128']=newer.install()
        if not states['biv150_prefill_k128'].get('installed'):
            raise RuntimeError('Present default K128 must remain installed')
        outer=manager.holder[manager.kernel_key]
        km=newer._MANAGER
        if (type(outer) is not newer._Policy or outer.manager is not km
                or outer.base is not qpolicy
                or (km.block_m,km.num_warps,km.tile_size,km.num_stages)!=(256,16,128,1)):
            raise RuntimeError('Unknown K128 default/fallback binding')
        parents.append(('k128',outer,km))
    if (manager.block_m,manager.num_warps,manager.tile_size,manager.num_stages)!=(128,8,64,1):
        raise RuntimeError('Unknown current multi-sequence fallback')
    for _,policy,parent in parents:
        if policy.disabled or not parent.active or parent.selected_calls!=0:
            raise RuntimeError('Current default parent is disabled or unexpectedly pre-used')
    tap=Tap(manager.wrapper)
    tap.parents=parents
    tap.outer=manager.holder[manager.kernel_key]
    tap.holder,tap.kernel_key=manager.holder,manager.kernel_key
    tap.qpolicy=qpolicy
    # Keep both real parents active; observe their normal two-sequence fallback.
    qpolicy.base=tap
    return function,manager,states,tap


def selected_configuration(tap,manager):
    result=kernel_configuration(tap,manager)
    if hasattr(tap,'parents'):
        if tap.holder[tap.kernel_key] is not tap.outer or tap.qpolicy.base is not tap:
            raise RuntimeError('Default policy chain changed during the probe')
        parents=[]
        for name,policy,parent in tap.parents:
            if (policy.manager is not parent or policy.disabled or not parent.active
                    or parent.selected_calls!=0):
                raise RuntimeError('Parent disabled/rebound or incorrectly selected on two sequences')
            parents.append(dict(name=name,active=parent.active,disabled=policy.disabled,
                                selected_calls=parent.selected_calls))
        result['preserved_default_parents']=parents
    return result


def main(args):
    chosen=args.case or list(CASES)
    for name in chosen:validate_case(*CASES[name])
    if not args.run_gpu:
        print(json.dumps(dict(gpu_executed=False,baseline='actual current policy fallback',
            cases={name:CASES[name] for name in chosen},candidate=[args.bm,args.warps,args.kv_tile,args.stages],
            production_changed=False)))
        return
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',9031)) != 0, 'Service running'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True), 'GPU busy'
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    package=importlib.import_module('vllm_fl')
    if Path(package.__file__).resolve()!=Path(__file__).resolve().parents[1]/'vllm_fl/__init__.py':
        raise RuntimeError('Actual loaded package is outside this diagnostic worktree')
    import biv150_prefill_probe_fixtures as fixture
    fixture.torch = torch
    torch.set_num_threads(4)
    baseline, baseline_manager, installations, baseline_tap = comparison_baseline(fixture)
    qoriginal = importlib.import_module('vllm_fl.patches.biv150_q_fragment_service')
    qmodule, adapted_sha = qoriginal, None
    attention = importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    backend = importlib.import_module('vllm.v1.attention.backends.triton_attn')
    candidate, candidate_manager = candidate_function(attention, backend, qmodule,args.bm,args.warps,args.kv_tile,args.stages)
    taps = {'baseline':baseline_tap}
    for label, manager in [('candidate',candidate_manager)]:
        tap = Tap(manager.wrapper,args.maxnreg)
        manager.holder[manager.kernel_key] = tap
        taps[label] = tap
    result = dict(completed=False, official_score=False, service_installed=False,
                  source_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  baseline_mode='current-multi2', baseline_config=[baseline_manager.block_m,baseline_manager.num_warps,baseline_manager.tile_size,baseline_manager.num_stages],
                  production_transform_unchanged=True,
                  query_case_provenance='Representative orderings from aggregate logs; KV lengths chosen, not a captured step.',
                  protocol_source_commit='032d6d317289cfd156ff8227904f6de264deab3d',
                  candidate_maxnreg=args.maxnreg,
                  adapted_manager_sha256=adapted_sha, candidate_tile=[args.bm,args.bm//8], candidate_warps=args.warps, candidate_kv_tile=args.kv_tile, candidate_stages=args.stages, installations=installations, cases=[], errors=[])
    def save():
        temp=args.out/'result.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.out/'result.json')
    save()
    try:
        for name in chosen:
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
    parser.add_argument('--bm',type=int,choices=[256],default=256)
    parser.add_argument('--warps',type=int,choices=[16],default=16)
    parser.add_argument('--kv-tile',type=int,choices=[64,128],default=64)
    parser.add_argument('--stages',type=int,choices=[1],default=1)
    parser.add_argument('--maxnreg',type=int,choices=[128],default=None)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--case',action='append',choices=CASES)
    args=parser.parse_args()
    if args.run_gpu and args.out is None:parser.error('--out required for GPU execution')
    main(args)
