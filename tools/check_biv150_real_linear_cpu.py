"""Real MiniCPM weight CPU FP64 oracle for functional/out BF16 small-row GEMM.

Readonly model fixture, isolated process, no serving patch or throughput claim.
"""
import argparse
import hashlib
import inspect
import json
import socket
import subprocess
import sys
from pathlib import Path


def main(a):
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    a.out.mkdir(exist_ok=False, parents=True)
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
    if a.candidate:
        from vllm_fl.patches import biv150_stable_small_o as small
        small._register()
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
              'candidate': a.candidate,
              'records': []}
    if a.candidate:
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
            if a.candidate:
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
                for route in ['functional', 'out'] + (['stable_functional', 'stable_out'] if a.candidate else []):
                    row = dict(**metadata, route=route, stream=stream_name, calls=0,
                               violation_calls=0, nonfinite_calls=0, max_abs_error=0.0,
                               graph_eager_different=0, patterns=[])
                    def call():
                        if route == 'stable_functional':
                            return torch.ops.biv150_stable_small_o.linear(x,w)
                        if route == 'stable_out':
                            return small._small_o_into(x,w,out)
                        return torch.nn.functional.linear(x, w) if route == 'functional' else torch.mm(x, w.t(), out=out)
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
                    row['passed'] = (row['violation_calls'] == row['nonfinite_calls'] == 0
                                     and row['graph_eager_different'] == 0
                                     and all(v['output_hashes'] == 1 for v in row['patterns']))
                    result['records'].append(row)
                    save()
                    print(json.dumps({key: row[key] for key in ['projection', 'route', 'stream', 'calls', 'violation_calls', 'nonfinite_calls', 'graph_eager_different', 'passed']}), flush=True)
                default.wait_stream(stream)
            torch.cuda.synchronize()
            del w, x, out, weight, gold
    result['complete'] = True
    save()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--candidate', action='store_true')
    p.add_argument('--model', type=Path, default=Path('/workspace/MiniCPM5-2B'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rows', type=int, default=2, choices=list(range(1,9)))
    p.add_argument('--projections', nargs='+', choices=['o', 'qkv', 'down', 'gate_edge', 'logits'], default=['o', 'qkv', 'down', 'gate_edge', 'logits'])
    main(p.parse_args())
