"""Independent CPU INT64 oracle for small-row native gate, eager and changed Graph.

Diagnostic only. Does not install a serving patch or claim model accuracy.
"""
import argparse
import hashlib
import json
import socket
from pathlib import Path

import torch


def reference(x, wq, scales, cols):
    # All quantization and integer dot products below execute on the CPU.
    f = x.cpu().float()
    first_scale = f.abs().amax(1).clamp_min(1e-6) / 127
    z = f / first_scale[:, None]
    q1 = torch.where(z >= 0, (z + 0.5).floor(), (z - 0.5).ceil()).clamp(-127, 127)
    second_scale = first_scale / 254
    z2 = (f - q1 * first_scale[:, None]) / second_scale[:, None]
    q2 = torch.where(z2 >= 0, (z2 + 0.5).floor(), (z2 - 0.5).ceil()).clamp(-127, 127)
    columns = torch.cat((cols, cols + 6144))
    weights = wq[:, columns].long()
    dot1, dot2 = q1.long() @ weights, q2.long() @ weights
    raw = (dot1.double() * first_scale.double()[:, None]
           + dot2.double() * second_scale.double()[:, None]) * scales[columns].double()
    gate, up = raw.float().to(torch.bfloat16).float().chunk(2, -1)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def main(args):
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1', 9031)) != 0, 'service active'
    torch.set_num_threads(2)
    torch.ops.load_library(str(args.library))
    torch.manual_seed(20261025)
    # Quantize synthetic weights on the CPU; never use the candidate quantizer
    # or GPU GEMM to construct an oracle.
    weight = torch.randn(12288, 2048).mul_(0.02).to(torch.bfloat16).float()
    scales_cpu = weight.abs().amax(1).clamp_min(1e-6) / 127
    wq_cpu = (weight / scales_cpu[:, None]).round().clamp(-127, 127).to(torch.int8).t().contiguous()
    del weight
    wq, scales = wq_cpu.cuda(), scales_cpu.cuda()
    marker = torch.zeros((), device='cuda', dtype=torch.bfloat16).expand(12288, 2048)
    cols = torch.tensor(sorted(set([0, 1, 31, 127, 1023, 6143] + list(range(17, 6144, 97)))))
    result = {'complete': False, 'diagnostic_only': True,
              'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'oracle': 'CPU-only quantization + INT64 dot + FP64 scaling; BF16 rounding and CPU SiLU',
              'columns_per_half': len(cols), 'records': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for rows in [1, 2, 3, 4, 8, 16]:
        inputs = []
        for pattern in ['normal', 'outlier']:
            x = torch.randn(rows, 2080).to(torch.bfloat16)[:, :2048]
            if pattern == 'outlier':
                x[:, 0] = 32
            inputs.append((pattern, x, reference(x, wq_cpu, scales_cpu, cols)))
        for stream_name in ['default', 'side']:
            stream = torch.cuda.default_stream() if stream_name == 'default' else torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Preserve a non-contiguous input stride to exercise contiguous().
                static = torch.empty((rows, 2080), dtype=torch.bfloat16, device='cuda')[:, :2048]
                static.copy_(inputs[0][1])
                for _ in range(3):
                    torch.ops.biv150_native.gate_silu_auto(static, wq, scales, marker, 17)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(stream)
                with torch.cuda.graph(graph, stream=capture_stream):
                    replayed = torch.ops.biv150_native.gate_silu_auto(static, wq, scales, marker, 17)
                stream.wait_stream(capture_stream)
                record = {'rows': rows, 'stream': stream_name, 'calls': 0, 'violations': 0,
                          'nonfinite_calls': 0, 'max_abs_error': 0.0, 'patterns': []}
                for pattern, x, gold in inputs:
                    hashes = set()
                    for mode, repeats in [('eager', 16), ('graph', 8)]:
                        for _ in range(repeats):
                            static.copy_(x)
                            if mode == 'graph':
                                replayed.fill_(float('nan'))
                                graph.replay()
                                y = replayed
                            else:
                                y = torch.ops.biv150_native.gate_silu_auto(static, wq, scales, marker, 17)
                            cpu = y.cpu()
                            checked = cpu[:, cols]
                            delta = (checked.float() - gold.float()).abs()
                            violations = delta > (0.005 + 0.005 * gold.float().abs())
                            record['violations'] += int(violations.sum())
                            record['nonfinite_calls'] += int(not torch.isfinite(cpu).all())
                            record['max_abs_error'] = max(record['max_abs_error'], float(delta.max()))
                            record['calls'] += 1
                            hashes.add(hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
                    record['patterns'].append({'name': pattern, 'output_hashes': len(hashes)})
                result['records'].append(record)
                args.out.write_text(json.dumps(result, indent=2))
                print(json.dumps(record), flush=True)
            torch.cuda.current_stream().wait_stream(stream)
    result['complete'] = True
    result['passed'] = all(r['violations'] == r['nonfinite_calls'] == 0 and
                           all(p['output_hashes'] == 1 for p in r['patterns'])
                           for r in result['records'])
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    main(parser.parse_args())
