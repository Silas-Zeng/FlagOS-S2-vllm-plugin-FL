#!/usr/bin/env python3
"""Component-only KV pointer rank experiment. No production installation.

Keep the same packed4D allocation, addresses and kernel strides; pass a zero-copy
[pages,32,128] view only at the producer's pointer ABI. The pinned frontend's
SME predicate may then admit K/V. Generated code and device results decide this,
not the predicate alone. Run only after all other GPU work has ended.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys

MAIN = '241220723a09061c182764f8f738e5ec01190367'
JIT = '/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py'
JIT_SHA = '66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
MODEL_4K_HASH = '641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff'


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


class PointerKernel:
    def __init__(self, kernel, k, v):
        self.kernel = kernel
        self.original = (k, v)
        self.views = tuple(t.view(t.shape[0], 32, 128) for t in (k, v))
        for original, view in zip(self.original, self.views):
            assert tuple(original.shape[1:]) == (16, 2, 128)
            assert tuple(original.stride()[1:]) == (256, 128, 1)
            assert view.data_ptr() == original.data_ptr()
            assert view.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
            assert tuple(view.stride()) == (original.stride(0), 128, 1)

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        def call(**kw):
            assert kw['key_cache_ptr'] is self.original[0]
            assert kw['value_cache_ptr'] is self.original[1]
            kw['key_cache_ptr'], kw['value_cache_ptr'] = self.views
            return launch(**kw)
        return call


def resources(kernel, out, label):
    metadata = kernel.metadata
    record = dict(hash=kernel.hash, n_regs=kernel.n_regs, n_spills=kernel.n_spills,
                  metadata={k: getattr(metadata, k, None) for k in
                            ('shared', 'num_warps', 'num_stages', 'use_sme', 'warp_size')}, files={})
    for name, value in kernel.asm.items():
        if not isinstance(value, (str, bytes)):
            continue
        data = value.encode() if isinstance(value, str) else value
        path = out / (label + '.' + name)
        path.write_bytes(data)
        record['files'][name] = dict(path=path.name, bytes=len(data),
                                    sha256=hashlib.sha256(data).hexdigest())
    return record


def main(args):
    source = args.source.resolve()
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == args.commit
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', MAIN, '--', 'vllm_fl', 'csrc'], check=True)
    assert hashlib.sha256(Path(JIT).read_bytes()).hexdigest() == JIT_SHA
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
    module = load_module(source / 'vllm_fl/patches/biv150_gqa_decode.py', 'pointer_view_frozen_gqa')
    # The pinned frontend computes SME only AFTER a per-JITFunction cache miss.
    # Rank is absent from that cache key. Preserve the original source but use
    # an independent JITFunction so a baseline launch cannot mask this probe.
    candidate_module = load_module(source / 'vllm_fl/patches/biv150_gqa_decode.py',
                                   'pointer_view_candidate_gqa')
    assert candidate_module._gqa_decode_attention is not module._gqa_decode_attention
    result = dict(component_only=True, official_score=False, production_unchanged=True,
                  source_commit=args.commit, production_baseline=MAIN, sdk_jit_sha256=JIT_SHA,
                  kernel_source_sha256=hashlib.sha256((source / 'vllm_fl/patches/biv150_gqa_decode.py').read_bytes()).hexdigest(),
                  records=[])
    atomic(args.out / 'result.json', result)
    for batch, length in ((64, 4096), (32, 16384)):
        row = dict(batch=batch, kv=length, passed=False)
        result['records'].append(row)
        try:
            c = inputs(batch, length)
            packed = torch.empty((c['k'].shape[0], 2, 16, 2, 128), device='cuda', dtype=torch.bfloat16)
            k, v = packed.unbind(1)
            k.copy_(c['k']); v.copy_(c['v']); c.update(k=k, v=v)
            table = torch.full((batch, 8192), -1, device='cuda', dtype=torch.int32)
            table[:, :c['table'].shape[1]].copy_(c['table'])
            c['table'] = table[:, :c['table'].shape[1]]
            segments = module._pick_segments(batch, 2, 16)
            for key in ('scratch', 'maxima', 'sums'):
                c[key] = c[key].narrow(2, 0, segments)
            proxy = SimpleNamespace(LOG2E=module.LOG2E,
                                    _gqa_decode_attention=PointerKernel(candidate_module._gqa_decode_attention, k, v))
            def run(selected):
                kernel = main_call(selected, c, segments)
                reduce_call(module, c, segments, 4)
                return kernel
            row.update(segments=segments, packed_shape=list(packed.shape), kv_stride=list(k.stride()),
                       pointer_view_stride=list(proxy._gqa_decode_attention.views[0].stride()), zero_copy=True)
            old = run(module); gold = c['out'].clone()
            new = run(proxy); torch.cuda.synchronize()
            row['baseline'] = resources(old, args.out, f'b{batch}-baseline')
            row['candidate'] = resources(new, args.out, f'b{batch}-candidate')
            if batch == 64:
                assert old.hash == MODEL_4K_HASH, 'Fixture differs from actual current model kernel'
            assert row['baseline']['metadata']['use_sme'] == 14
            assert row['candidate']['metadata']['use_sme'] == 62, row['candidate']['metadata']
            row['uniform_exact'] = bool(torch.equal(c['out'], gold))
            row['numeric_checks'] = []
            for ragged in (False, True):
                c['q'].normal_()
                c['lengths'].fill_(length)
                if ragged:
                    for index, n in enumerate((0, 1, 15, 16, 17, 63, 100, length - 1)):
                        c['lengths'][index] = n
                c['out'].fill_(float('nan')); run(module); baseline = c['out'].clone()
                c['out'].fill_(float('nan')); run(proxy); torch.cuda.synchronize()
                assert torch.isfinite(c['out']).all().item()
                for index in (0, 1, batch - 1):
                    n = int(c['lengths'][index].cpu())
                    if n:
                        ids = c['table'][index, :(n + 15)//16].long()
                        kk = c['k'][ids].reshape(-1, 2, 128)[:n].cpu().double()
                        vv = c['v'][ids].reshape(-1, 2, 128)[:n].cpu().double()
                        qq = c['q'][index].cpu().double().reshape(2, 8, 128)
                        pp = torch.einsum('hgd,thd->hgt', qq, kk).mul(128**-.5).softmax(-1)
                        ref = torch.einsum('hgt,thd->hgd', pp, vv).reshape(16, 128)
                    else:
                        ref = torch.zeros((16, 128), dtype=torch.float64)
                    denom = max(float(ref.square().mean().sqrt()), 1e-30)
                    errors = [float((t[index].cpu().double()-ref).square().mean().sqrt())/denom
                              for t in (baseline, c['out'])]
                    row['numeric_checks'].append(dict(ragged=ragged, row=index, length=n,
                        baseline_relative_rmse=errors[0], candidate_relative_rmse=errors[1],
                        passed=errors[1] <= max(.005, 1.2*errors[0])))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(proxy)
                c['q'].normal_(); c['out'].fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
                changed = c['out'].clone(); run(module); torch.cuda.synchronize()
                row['numeric_checks'].append(dict(changed_input_graph=True, ragged=ragged,
                    finite=bool(torch.isfinite(changed).all().item()), exact=bool(torch.equal(changed, c['out'])),
                    passed=bool(torch.isfinite(changed).all().item()) and bool(torch.allclose(changed, c['out'], rtol=.005, atol=1e-5))))
                del graph, changed, baseline
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run(proxy)
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            row['side_stream_finite'] = bool(torch.isfinite(c['out']).all().item())
            c['lengths'].fill_(length)
            row['timings'] = []
            for iteration in range(3):
                choices = [('baseline_ms', module), ('candidate_ms', proxy)]
                if iteration % 2:
                    choices.reverse()
                timing = {name: graph_time(lambda selected=selected: run(selected), repeats=15, unroll=8)[0]
                          for name, selected in choices}
                timing['speedup'] = timing['baseline_ms']/timing['candidate_ms']
                row['timings'].append(timing)
            row.update(median_speedup=statistics.median(x['speedup'] for x in row['timings']),
                       passed=all(x['passed'] for x in row['numeric_checks']) and row['side_stream_finite'])
            del c, proxy, packed, k, v, table, gold
            torch.cuda.empty_cache()
        except BaseException as error:
            row['error'] = repr(error)
            atomic(args.out / 'result.json', result)
            raise
        atomic(args.out / 'result.json', result)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
