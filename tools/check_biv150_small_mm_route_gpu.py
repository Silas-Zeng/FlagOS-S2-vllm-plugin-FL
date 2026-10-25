"""Real MiniCPM weight CPU FP64 oracle for functional/out BF16 small-row GEMM.

Isolated real selected-vendor binding probe. No serving or throughput claim.
"""
import argparse
import hashlib
import inspect
import importlib
import os
import shutil
import json
import socket
import subprocess
import sys
from pathlib import Path


def main(a):
    if not a.run_gpu:
        print(json.dumps(dict(gpu_executed=False,rows=a.rows,projections=a.projections,
                             routes=['functional','out','direct_functional','direct_out'],
                             oracle='independent CPU FP64',snapshot='copied to private config_cache',
                             gate='selected direct_out; no transfer of legacy binding qualification')))
        return
    assert a.source and a.commit and a.out and a.tuning_snapshot
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==a.commit
    assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    a.out.mkdir(exist_ok=False, parents=True)
    records=json.loads((a.tuning_snapshot/'manifest.json').read_text())['databases']
    destination=a.out/'flaggems-cache/config_cache';destination.mkdir(parents=True)
    for record in records:
        name=record['name'];assert Path(name).name==name and name.endswith('.db')
        original=a.tuning_snapshot/name
        assert hashlib.sha256(original.read_bytes()).hexdigest()==record['sha256']
        shutil.copy2(original,destination/name)
    os.environ['FLAGGEMS_CACHE_DIR']=str(a.out/'flaggems-cache')
    os.environ['TRITON_CACHE_DIR']=str(a.out/'triton-cache')
    sys.path.insert(0, str(a.source))
    import torch
    import flag_gems
    from safetensors import safe_open
    from vllm_fl.utils import use_flaggems, get_flag_gems_whitelist_blacklist
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert use_flaggems() and not whitelist and not blacklist
    torch.set_num_threads(2)
    baseline_file = Path(inspect.getsourcefile(inspect.unwrap(flag_gems.mm_out)))
    assert hashlib.sha256(baseline_file.read_bytes()).hexdigest() == '4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    flag_gems.enable(record=False, once=True)
    from vllm_fl.patches import biv150_stable_small_o as small
    module=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
    original_launch=module._launch_mm
    installation=small._install_mm_route()
    assert installation.get('installed'),installation
    direct_launch=module._launch_mm
    assert direct_launch is not original_launch
    definitions = {
        'o': ['model.layers.0.self_attn.o_proj.weight'],
        'qkv': ['model.layers.0.self_attn.' + p + '_proj.weight' for p in ['q', 'k', 'v']],
        'down': ['model.layers.0.mlp.down_proj.weight'],
        'gate_edge': ['model.layers.0.mlp.' + p + '_proj.weight' for p in ['gate', 'up']],
        'logits': ['lm_head.weight'],
    }
    result = {'complete': False, 'diagnostic_only': True, 'rows': a.rows,
              'source_commit': subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'binding':'mm','installation':installation,'tuning_snapshot':records,
              'records': []}
    result['candidate_operator_sha256'] = hashlib.sha256((a.source / 'vllm_fl/patches/biv150_stable_small_o.py').read_bytes()).hexdigest()
    def save():
        (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    def digest(v):
        return hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    torch.manual_seed(20261025)
    with torch.inference_mode():
        for projection in a.projections:
            parts = []
            for key in definitions[projection]:
                for path in sorted(a.model.glob('*.safetensors')):
                    with safe_open(str(path), framework='pt', device='cpu') as h:
                        if key in h.keys():
                            parts.append(h.get_tensor(key))
                            break
                else:
                    raise KeyError(key)
            weight = torch.cat(parts, dim=0).contiguous()
            del parts
            n, k = weight.shape
            assert (n,k) in [(2048,2048),(12288,2048),(2560,2048),(2048,6144)]
            inputs = [torch.randn(a.rows, k).to(torch.bfloat16) for _ in range(2)]
            gold = [x.double() @ weight.double().t() for x in inputs]
            w = weight.cuda()
            x = torch.empty((a.rows, k), device='cuda', dtype=torch.bfloat16)
            out = torch.empty((a.rows, n), device='cuda', dtype=torch.bfloat16)
            metadata = dict(projection=projection, n=n, k=k,
                            weight_keys=definitions[projection], weight_sha256=digest(weight),
                            input_sha256=[digest(v) for v in inputs])
            torch.cuda.synchronize()
            default = torch.cuda.default_stream()
            for stream_name in ['default', 'side']:
                stream = default if stream_name == 'default' else torch.cuda.Stream()
                stream.wait_stream(default)
                for route in ['functional','out','direct_functional','direct_out']:
                    module._launch_mm=direct_launch if route.startswith('direct_') else original_launch
                    small._OBSERVED_ROWS.clear()
                    row = dict(**metadata, route=route, stream=stream_name, calls=0,
                               violation_calls=0, nonfinite_calls=0, max_abs_error=0.0,
                               graph_eager_different=0, patterns=[])
                    def call():
                        return torch.nn.functional.linear(x,w) if route.endswith('functional') else torch.mm(x,w.t(),out=out)
                    def check(value, pattern, hashes):
                        value = value.cpu().clone()
                        delta = (value.double() - gold[pattern]).abs()
                        bad = (delta > 0.005 + 0.005 * gold[pattern].abs()) | ~torch.isfinite(value)
                        row['calls'] += 1
                        row['violation_calls'] += int(bool(bad.any()))
                        row['nonfinite_calls'] += int(not bool(torch.isfinite(value).all()))
                        row['max_abs_error'] = max(row['max_abs_error'], float(delta.max()))
                        hashes.add(digest(value))
                        return value
                    with torch.cuda.stream(stream):
                        x.copy_(inputs[0])
                        for _ in range(3):
                            call()
                        stream.synchronize()
                        capture = torch.cuda.Stream()
                        capture.wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=capture):
                            graph_out = call()
                        stream.wait_stream(capture)
                        for pattern, host in enumerate(inputs):
                            hashes = set()
                            for _ in range(16):
                                x.copy_(host)
                                out.fill_(float('nan'))
                                value = call()
                                check(value, pattern, hashes)
                            for _ in range(8):
                                x.copy_(host)
                                graph_out.fill_(float('nan'))
                                graph.replay()
                                actual = check(graph_out, pattern, hashes)
                                eager = call().cpu()
                                row['graph_eager_different'] += int(not torch.equal(actual, eager))
                            row['patterns'].append(dict(input=pattern, output_hashes=len(hashes)))
                        del graph, graph_out, value, actual, eager
                    row['selected_by_vendor_wrapper']=(a.rows,n,k) in small._OBSERVED_ROWS
                    if route=='direct_out':assert row['selected_by_vendor_wrapper'],'actual out path never entered vendor patch'
                    row['passed'] = (row['violation_calls'] == row['nonfinite_calls'] == 0
                                     and row['graph_eager_different'] == 0
                                     and all(v['output_hashes'] == 1 for v in row['patterns']))
                    result['records'].append(row)
                    save()
                    print(json.dumps({key: row[key] for key in ['projection', 'route', 'stream', 'calls', 'violation_calls', 'nonfinite_calls', 'graph_eager_different', 'passed']}), flush=True)
                default.wait_stream(stream)
            torch.cuda.synchronize()
            del w, x, out, weight, gold
    module._launch_mm=original_launch
    result['vendor_restored']=module._launch_mm is original_launch
    owned=[v for v in result['records'] if v['route'].startswith('direct_') and v['selected_by_vendor_wrapper']]
    result['owned_routes_passed']=bool(owned) and all(v['passed'] for v in owned)
    result['all_direct_functional_passed']=all(v['passed'] for v in result['records'] if v['route']=='direct_functional')
    result['complete'] = True
    save()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--source',type=Path)
    p.add_argument('--commit')
    p.add_argument('--tuning-snapshot',type=Path)
    p.add_argument('--model', type=Path, default=Path('/workspace/MiniCPM5-2B'))
    p.add_argument('--out',type=Path)
    p.add_argument('--rows', type=int, default=2, choices=list(range(1,9)))
    p.add_argument('--projections', nargs='+', choices=['o','qkv','down','gate_edge'], default=['o','qkv','down','gate_edge'])
    main(p.parse_args())
