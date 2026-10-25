"""Real-weight M2048 BF16 BLAS probe against the worker's FlagGems mm.out.

No model hooks or serving defaults change. CPU FP64 is an independent oracle;
all numerical discrepancies are retained even when diagnostic timing proceeds.
This is a warmed component test, not service accuracy or a throughput score.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import socket
import statistics
import subprocess
import traceback


def weights(model, layers, gate_layers):
    from safetensors import safe_open
    locations = {}
    for path in sorted(model.glob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for key in handle.keys():
                locations[key] = path
    cases = [(layer, name, projections) for layer in layers for name, projections in (
        ('qkv', ('q_proj', 'k_proj', 'v_proj')), ('o', ('o_proj',)), ('down', ('down_proj',)))]
    cases += [(layer, 'gate_up', ('gate_proj', 'up_proj')) for layer in gate_layers]
    for layer, name, projections in cases:
        parts, identities = [], []
        for projection in projections:
            section = 'mlp' if projection in ('gate_proj', 'up_proj', 'down_proj') else 'self_attn'
            key = f'model.layers.{layer}.{section}.{projection}.weight'
            with safe_open(str(locations[key]), framework='pt', device='cpu') as handle:
                tensor = handle.get_tensor(key)
            assert tensor.dtype == torch.bfloat16 and tensor.ndim == 2
            parts.append(tensor)
            identities.append(dict(key=key, sha256=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()))
        tensor = torch.cat(parts) if len(parts) > 1 else parts[0]
        assert tuple(tensor.shape) == dict(qkv=(2560,2048),o=(2048,2048),down=(2048,6144),gate_up=(12288,2048))[name]
        yield layer, name, tensor, identities


def compare(actual, reference):
    actual, reference = actual.cpu().float(), reference.cpu().float()
    difference = (actual-reference).abs()
    finite = bool(torch.isfinite(actual).all())
    bad = (difference > .005+.005*reference.abs()) | ~torch.isfinite(actual)
    return dict(finite=finite, tolerance_passed=not bool(bad.any()),
        violations=int(bad.sum()), elements=bad.numel(), max_abs=float(difference.max()),
        numeric_differences=int((actual != reference).sum()),
        relative_l2=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(reference).clamp_min(1e-30)))


def main(args):
    with socket.socket() as port:
        assert port.connect_ex(('127.0.0.1',9031)) != 0, 'A service is running'
    assert 'No running processes found' in subprocess.check_output(
        ['/usr/local/corex/bin/ixsmi'], text=True), 'GPU is busy'
    args.out.mkdir(parents=True, exist_ok=False)
    global torch
    import torch
    import flag_gems
    from biv150_benchmark_utils import graph_time
    from vllm_fl.utils import use_flaggems, get_flag_gems_whitelist_blacklist
    assert use_flaggems()
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert not whitelist and not blacklist, 'Reaudit worker dispatch policy'
    native_path = Path(inspect.getsourcefile(inspect.unwrap(flag_gems.mm_out)))
    native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
    assert native_hash == '4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    flag_gems.enable(record=False, once=True)
    torch.ops.load_library(str(args.library.resolve()))
    op = torch.ops.biv150_native_bf16_probe.linear_out
    result = dict(component_only=True, service_installed=False, official_score=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        baseline='torch.mm(out=caller_buffer) with current worker FlagGems policy',
        flaggems_mm_source=str(native_path), flaggems_mm_sha256=native_hash,
        dispatch_table=torch._C._dispatch_dump_table('aten::mm.out'),
        rows=args.rows, layers=args.layers, gate_layers=args.gate_layers,
        synthetic_activations=True, oracle='all output elements: CPU FP64 then BF16',
        timing_policy='fixed caller-owned output buffers; 3 alternating rounds, 4+5 graph warmups',
        timing_limit='Repeated same-weight graph timings; not a cold weight-ring or service measurement',
        precision_policy='finite results may be timed despite tolerance failures; every failure is retained',
        records=[])
    save = lambda: (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    with torch.inference_mode():
        for layer, name, host, identities in weights(args.model, args.layers, args.gate_layers):
            weight, host64 = host.cuda(), host.double()
            n, k = host.shape
            for rows in args.rows:
                record = dict(layer=layer, geometry=name, m=rows, n=n, k=k,
                    weights=identities, patterns=[], full_cpu_passed=False,
                    eager_cpu_complete=False, graph_cpu_complete=False)
                result['records'].append(record)
                try:
                    x = torch.empty((rows,k),device='cuda',dtype=torch.bfloat16)
                    baseline_out = torch.empty((rows,n),device='cuda',dtype=torch.bfloat16)
                    candidate_out = torch.empty_like(baseline_out)
                    baseline = lambda: torch.mm(x,weight.t(),out=baseline_out)
                    candidate = lambda: op(x,weight,candidate_out)
                    for pattern in ('zero','tiny','normal','large','outlier'):
                        x.normal_()
                        if pattern == 'zero': x.zero_()
                        elif pattern == 'tiny': x.mul_(1e-4)
                        elif pattern == 'large': x.mul_(100)
                        elif pattern == 'outlier': x[:,k//2] = 32
                        reference = (x.cpu().double() @ host64.t()).to(torch.bfloat16)
                        baseline_out.fill_(float('nan')); candidate_out.fill_(float('nan'))
                        baseline(); candidate(); torch.cuda.synchronize()
                        b, c = compare(baseline_out, reference), compare(candidate_out, reference)
                        direct = compare(candidate_out,baseline_out)
                        record['patterns'].append(dict(name=pattern,baseline_cpu=b,candidate_cpu=c,direct=direct))
                        if not c['tolerance_passed'] and 'failure_fixture' not in record:
                            path=args.out/f'failure-l{layer}-{name}-m{rows}.pt'
                            torch.save(dict(x=x.cpu(),weight_identity=identities,reference=reference,
                                candidate=candidate_out.cpu(),baseline=baseline_out.cpu(),pattern=pattern),path)
                            record['failure_fixture']=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    record['eager_cpu_complete'] = True
                    record['eager_cpu_passed'] = all(p['candidate_cpu']['tolerance_passed'] for p in record['patterns'])
                    assert all(p['candidate_cpu']['finite'] for p in record['patterns']), 'Nonfinite candidate'
                    # Verify actual non-default stream execution and graph input changes.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        x.normal_(); candidate(); graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream): candidate()
                    stream.synchronize()
                    record['changed_input_graph_checks'] = []
                    with torch.cuda.stream(stream):
                        for factor in (0.,.5,4.):
                            x.normal_().mul_(factor); candidate_out.fill_(float('nan'))
                            graph.replay(); stream.synchronize()
                            actual=candidate_out.cpu().clone(); candidate_out.fill_(float('nan'))
                            candidate(); stream.synchronize()
                            exact=bool(torch.equal(actual,candidate_out.cpu()))
                            check=compare(actual,(x.cpu().double()@host64.t()).to(torch.bfloat16))
                            record['changed_input_graph_checks'].append(dict(factor=factor,eager_numeric_equal=exact,cpu=check))
                            assert check['finite'] and bool(torch.isfinite(candidate_out).all()), 'Nonfinite Graph/eager output'
                    record['graph_cpu_complete'] = True
                    record['graph_cpu_passed'] = all(v['cpu']['tolerance_passed'] for v in record['changed_input_graph_checks'])
                    record['graph_eager_numeric_equal'] = all(v['eager_numeric_equal'] for v in record['changed_input_graph_checks'])
                    torch.cuda.current_stream().wait_stream(stream)
                    del graph
                    x.normal_(); candidate_out.fill_(float('nan')); baseline_out.fill_(float('nan'))
                    candidate(); baseline(); torch.cuda.synchronize()
                    timing_reference=(x.cpu().double()@host64.t()).to(torch.bfloat16)
                    record['timing_candidate_cpu']=compare(candidate_out,timing_reference)
                    record['timing_baseline_cpu']=compare(baseline_out,timing_reference)
                    assert record['timing_candidate_cpu']['finite'], 'Nonfinite timed input'
                    record['full_cpu_passed'] = (record['eager_cpu_passed'] and record['graph_cpu_passed']
                                                and record['timing_candidate_cpu']['tolerance_passed'])
                    rounds=[]
                    for index in range(3):
                        calls=[('baseline_ms',baseline),('candidate_ms',candidate)]
                        if index % 2: calls.reverse()
                        timing={label:graph_time(fn,repeats=9,unroll=4)[0] for label,fn in calls}
                        timing['speedup']=timing['baseline_ms']/timing['candidate_ms']; rounds.append(timing)
                    record.update(rounds=rounds, median_speedup=statistics.median(r['speedup'] for r in rounds),
                        min_round_speedup=min(r['speedup'] for r in rounds))
                except Exception as error:
                    record.update(error=repr(error),traceback=traceback.format_exc(),full_cpu_passed=False)
                save()
                print(json.dumps({key:record.get(key) for key in ('layer','geometry','m','full_cpu_passed','median_speedup','error')}),flush=True)
            del weight,host64
    result['completed'] = True
    result['expected_records']=(3*len(args.layers)+len(args.gate_layers))*len(args.rows)
    assert len(result['records']) == result['expected_records']
    result['records_with_errors']=sum('error' in row for row in result['records'])
    save()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'))
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rows',type=int,nargs='+',default=[2048])
    parser.add_argument('--layers',type=int,nargs='+',default=[20])
    parser.add_argument('--gate-layers',type=int,nargs='*',default=[0])
    args=parser.parse_args()
    assert all(1 <= row <= 2048 for row in args.rows)
    assert all(0 <= layer < 42 for layer in args.layers)
    assert all(0 <= layer < 4 or 38 <= layer < 42 for layer in args.gate_layers)
    main(args)
