#!/usr/bin/env python3
"""Default CPU source check; device work requires explicit --gpu and root scheduling."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys

from biv150_unchanged_max_factory import MAIN, NAME, build
from biv150_unchanged_max_ir import audit_artifacts

JIT = Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py')
JIT_SHA = '66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
BASE_HASHES = {64: '641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff',
               32: 'e29a0c84d6b99f1ad44189fbdb74434b2d7da5675af2e2e2422941739916adb9'}
PRIMITIVE = '''import triton
import triton.language as tl
@triton.jit
def probe(x, zero, product, alpha, N:tl.constexpr):
    i = tl.arange(0, 32)
    a = tl.math.exp2(tl.load(zero+i, i<N, other=0.0))
    v = tl.load(x+i, i<N, other=0.0)
    tl.store(alpha+i, a, i<N)
    tl.store(product+i, v*a, i<N)
'''


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def resources(kernel, out, label):
    md = kernel.metadata
    record = dict(phase='post_launch', hash=kernel.hash,
                  n_regs=getattr(kernel, 'n_regs', None), n_spills=getattr(kernel, 'n_spills', None),
                  metadata={k: getattr(md, k, None) for k in
                            ('shared', 'num_warps', 'num_stages', 'use_sme', 'warp_size')}, files={})
    record['registers_known'] = all(isinstance(record[k], int) and not isinstance(record[k], bool)
                                    for k in ('n_regs', 'n_spills'))
    for name, value in kernel.asm.items():
        if isinstance(value, (str, bytes)):
            data = value.encode() if isinstance(value, str) else value
            target = out/(label+'.'+name)
            target.write_bytes(data)
            record['files'][name] = dict(path=target.name, bytes=len(data),
                                         sha256=hashlib.sha256(data).hexdigest())
    return record


def cpu_bits(tensor, torch):
    host = tensor.detach().cpu().contiguous()
    return host.view(torch.int16 if host.element_size() == 2 else torch.int32)


def primitive_gate(torch, load_module, out):
    path = out/'identity-primitive.py'
    path.write_text(PRIMITIVE)
    module = load_module(path, 'unchanged_max_identity_probe')
    patterns = [0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x80000001,
                0x807fffff, 0x80800000, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff]
    values = torch.tensor([b if b < 2**31 else b-2**32 for b in patterns], dtype=torch.int32).view(torch.float32)
    x = values.cuda()
    zeros = torch.tensor([0, -2**31]*6, dtype=torch.int32).view(torch.float32).cuda()
    product, alpha = torch.empty_like(x), torch.empty_like(x)
    kernel = module.probe[(1,)](x, zeros, product, alpha, len(patterns), num_warps=1, num_stages=1)
    torch.cuda.synchronize()
    artifact = resources(kernel, out, 'identity-primitive')
    actual = kernel.asm.get('asm', '')
    present = isinstance(actual, str) and 'ml_exp_f32' in actual and 'ml_mul_f32' in actual
    report = dict(resources=artifact, real_exp_and_mul_present=present,
                  alpha_one_exact=torch.equal(cpu_bits(alpha, torch), cpu_bits(torch.ones_like(values), torch)),
                  signed_zero_subnormal_normal_identity_exact=torch.equal(cpu_bits(product, torch), cpu_bits(values, torch)))
    report['passed'] = present and report['alpha_one_exact'] and report['signed_zero_subnormal_normal_identity_exact']
    atomic(out/'identity-primitive.json', report)
    if not report['passed']:
        raise RuntimeError('Native exp2(±0)/mul-one exact falsifier failed; no attention timing')
    return report


def reference_check(c, baseline, candidate, torch):
    checks = []
    for index in (0, 1, c['q'].shape[0]-1):
        n = int(c['lengths'][index].cpu())
        if n:
            ids = c['table'][index, :(n+15)//16].long()
            k = c['k'][ids].reshape(-1, 2, 128)[:n].cpu().double()
            v = c['v'][ids].reshape(-1, 2, 128)[:n].cpu().double()
            q = c['q'][index].cpu().double().reshape(2, 8, 128)
            p = torch.einsum('hgd,thd->hgt', q, k).mul(128**-.5).softmax(-1)
            ref = torch.einsum('hgt,thd->hgd', p, v).reshape(16, 128)
        else:
            ref = torch.zeros((16, 128), dtype=torch.float64)
        denominator = max(float(ref.square().mean().sqrt()), 1e-30)
        errors = []
        for output in (baseline, candidate):
            delta = output[index].detach().cpu().double()-ref
            rms, maximum = float(delta.square().mean().sqrt()), float(delta.abs().max())
            assert rms != 0.0 or maximum == 0.0, 'Contradictory CPU RMS/max metric'
            errors.append(rms/denominator)
        checks.append(dict(row=index, length=n, baseline_relative_rmse=errors[0],
                           candidate_relative_rmse=errors[1], passed=errors[1] <= max(.005, 1.2*errors[0])))
    return checks


def run_gpu(args, generated, proof):
    source = args.source.resolve()
    assert args.commit and args.out, '--gpu requires frozen --commit and fresh --out'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == args.commit
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', MAIN, '--', 'vllm_fl', 'csrc'], check=True)
    assert hashlib.sha256(JIT.read_bytes()).hexdigest() == JIT_SHA
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0, 'Existing service; nothing stopped'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True)
    args.out.mkdir(parents=False, exist_ok=False)
    import torch
    from types import SimpleNamespace
    from biv150_benchmark_utils import inputs, load_module, main_call, reduce_call, graph_time
    torch.set_num_threads(4)
    props = torch.cuda.get_device_properties(0)
    assert 'BI-V150' in props.name and props.multi_processor_count == 16
    base = load_module(source/'vllm_fl/patches/biv150_gqa_decode.py', 'unchanged_max_frozen_base')
    path = args.out/'generated-gqa.py'
    path.write_text(generated)
    own = load_module(path, 'unchanged_max_private')
    proxy = SimpleNamespace(LOG2E=base.LOG2E, _gqa_decode_attention=getattr(own, NAME))
    result = dict(source_commit=args.commit, main=MAIN, proof=proof, component_only=True,
                  gpu_executed=True, official_score=False, precision_gate=False, model_evaluated=False, records=[])
    atomic(args.out/'result.json', result)
    try:
        result['primitive'] = primitive_gate(torch, load_module, args.out)
        for batch, length in ((64, 4096), (32, 16384)):
            row = dict(batch=batch, kv=length, passed=False, timing_attempted=False)
            result['records'].append(row)
            c = inputs(batch, length)  # Preserve the original input factory and seed.
            packed = torch.empty((c['k'].shape[0], 2, 16, 2, 128), device='cuda', dtype=torch.bfloat16)
            k, v = packed.unbind(1)
            k.copy_(c['k']); v.copy_(c['v']); c.update(k=k, v=v)
            table = torch.full((batch, 8192), -1, device='cuda', dtype=torch.int32)
            table[:, :c['table'].shape[1]].copy_(c['table'])
            c['table'] = table[:, :c['table'].shape[1]]
            segments = base._pick_segments(batch, 2, 16)
            for key in ('scratch', 'maxima', 'sums'):
                c[key] = c[key].narrow(2, 0, segments)
            originals = {key: c[key].clone() for key in ('q', 'k', 'v')}

            def run(selected):
                compiled = main_call(selected, c, segments)
                reduce_call(base, c, segments, 4)
                return compiled

            def snapshot(selected):
                for key in ('scratch', 'maxima', 'sums', 'out'):
                    c[key].fill_(float('nan'))
                compiled = run(selected)
                torch.cuda.synchronize()
                bits = {key: cpu_bits(c[key], torch) for key in ('scratch', 'maxima', 'sums', 'out')}
                return compiled, bits, c['out'].detach().cpu().clone()

            old, _, _ = snapshot(base)
            new, _, _ = snapshot(proxy)
            row.update(segments=segments, packed_shape=list(packed.shape), kv_stride=list(k.stride()),
                       baseline=resources(old, args.out, f'b{batch}-baseline'),
                       candidate=resources(new, args.out, f'b{batch}-candidate'))
            assert old.hash == BASE_HASHES[batch], 'Baseline differs from real current packed specialization'
            for kernel in (old, new):
                assert kernel.metadata.num_warps == 1 and kernel.metadata.num_stages == 1
            comparisons = {}
            for field in ('n_regs', 'n_spills', 'shared', 'num_warps', 'num_stages', 'use_sme', 'warp_size'):
                values = [row[k].get(field) if field in ('n_regs', 'n_spills')
                          else row[k]['metadata'].get(field) for k in ('baseline', 'candidate')]
                known = all(isinstance(v, int) and not isinstance(v, bool) for v in values)
                comparisons[field] = ('unknown' if not known else
                                      ('known_equal' if values[0] == values[1] else 'known_changed'))
            row['resource_comparison'] = comparisons
            row['mechanism'] = audit_artifacts(new.asm, proof)
            atomic(args.out/'result.json', result)
            row['numeric_checks'] = []
            for mode in ('original', 'ragged', 'zero_qk', 'one_head_update', 'tiny_v'):
                for key in originals:
                    c[key].copy_(originals[key])
                c['lengths'].fill_(length)
                if mode == 'ragged':
                    for index, n in enumerate((0, 1, 15, 16, 17, 63, 100, length-1)):
                        c['lengths'][index] = n
                if mode in ('zero_qk', 'tiny_v', 'one_head_update'):
                    c['q'].zero_(); c['k'].zero_()
                if mode == 'tiny_v':
                    tiny = torch.tensor([0., -0., 2.**-126, -2.**-126, 2.**-133, -2.**-133],
                                        dtype=torch.bfloat16, device='cuda')
                    c['v'].copy_(tiny[torch.arange(128, device='cuda') % len(tiny)][None, None, None, :])
                if mode == 'one_head_update':
                    c['q'][:, 0, 0] = 1
                    pages = length//16
                    ids = c['table'][:, :pages].reshape(-1).long()
                    progression = (torch.arange(pages, device='cuda')/128).expand(batch, pages).reshape(-1)
                    c['k'][ids, :, 0, 0] = progression[:, None]
                _, gold_bits, gold = snapshot(base)
                _, got_bits, got = snapshot(proxy)
                exact = {key: torch.equal(gold_bits[key], got_bits[key]) for key in gold_bits}
                checks = reference_check(c, gold, got, torch)
                check = dict(mode=mode, producer_and_output_bits=exact,
                             finite=bool(torch.isfinite(got).all()), reference=checks)
                row['numeric_checks'].append(check)
                atomic(args.out/'result.json', result)
                assert all(exact.values()) and check['finite'] and all(x['passed'] for x in checks), \
                    'Direct bitwise/reference falsifier failed; no timing'
            for key in originals:
                c[key].copy_(originals[key])
            c['lengths'].fill_(length)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(proxy)
            c['q'].normal_()
            c['lengths'][:8].copy_(torch.tensor([0, 1, 15, 16, 17, 63, 100, length-1], device='cuda', dtype=torch.int32))
            for key in ('scratch', 'maxima', 'sums', 'out'):
                c[key].fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            replay = {key: cpu_bits(c[key], torch) for key in ('scratch', 'maxima', 'sums', 'out')}
            row['changed_graph_finite'] = bool(torch.isfinite(c['out'].detach().cpu()).all())
            _, eager, _ = snapshot(proxy)
            _, gold, _ = snapshot(base)
            row['changed_graph_exact'] = all(torch.equal(replay[k], eager[k]) and torch.equal(replay[k], gold[k]) for k in replay)
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for key in ('scratch', 'maxima', 'sums', 'out'):
                    c[key].fill_(float('nan'))
                run(proxy)
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            row['side_stream_exact'] = all(torch.equal(cpu_bits(c[k], torch), gold[k]) for k in gold)
            row['side_stream_finite'] = bool(torch.isfinite(c['out'].detach().cpu()).all())
            atomic(args.out/'result.json', result)
            assert all(row[k] for k in ('changed_graph_exact', 'changed_graph_finite',
                                       'side_stream_exact', 'side_stream_finite')), 'Graph/side exact gate failed; no timing'
            for key in originals:
                c[key].copy_(originals[key])
            c['lengths'].fill_(length)
            row['timing_attempted'] = True
            row['timings'] = []
            for iteration in range(3):
                choices = [('baseline_ms', base), ('candidate_ms', proxy)]
                if iteration % 2:
                    choices.reverse()
                timing = {name: graph_time(lambda selected=selected: run(selected), repeats=15, unroll=8)[0]
                          for name, selected in choices}
                timing['speedup'] = timing['baseline_ms']/timing['candidate_ms']
                row['timings'].append(timing)
            row.update(passed=True, median_speedup=statistics.median(x['speedup'] for x in row['timings']))
            atomic(args.out/'result.json', result)
            del graph, originals, c, k, v, packed, table
            torch.cuda.empty_cache()
    except BaseException as error:
        result['error'] = repr(error)
        atomic(args.out/'result.json', result)
        raise
    result['completed'] = True
    atomic(args.out/'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--commit')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    generated, proof = build((args.source/'vllm_fl/patches/biv150_gqa_decode.py').read_text())
    if not args.gpu:
        print(json.dumps(dict(proof=proof, cpu_only=True, torch_imported='torch' in sys.modules,
                              triton_imported='triton' in sys.modules, gpu_executed=False)))
        return
    run_gpu(args, generated, proof)


if __name__ == '__main__':
    main()
