#!/usr/bin/env python3
"""Component-only GQA constant strides, preserving signed int64 offsets.

Reuse the original while-loop/math and physical layouts. Specialize actual
host strides without narrowing address arithmetic. No serving installation.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
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


_STRIDES = ('block_table_stride', 'query_stride_token', 'k_stride_block',
            'k_stride_slot', 'k_stride_head', 'v_stride_block', 'v_stride_slot')


def specialize_stride_source(text):
    tree = ast.parse(text)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_gqa_decode_attention']
    if len(found) != 1:
        raise ValueError('Unique audited GQA function required')
    fn = found[0]
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    if any(name + '_i64' in names for name in _STRIDES):
        raise ValueError('Signed stride alias collision')
    if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
           and n.id in _STRIDES for n in ast.walk(fn)):
        raise ValueError('Stride parameter reassignment')
    lines = text.splitlines(keepends=True)
    begin = min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1
    original = ''.join(lines[begin:fn.end_lineno])
    header, body = original.split('\n):\n', 1)
    original_header, original_body = header, body
    for name in _STRIDES:
        old = name + ': tl.int64'
        if header.count(old) != 1:
            raise ValueError('Unique signed int64 stride declaration required: ' + name)
        header = header.replace(old, name + ': tl.constexpr', 1)
        body = re.sub(r'\b' + name + r'\b', name + '_i64', body)
    aliases = ''.join('    ' + name + '_i64 = tl.full((), ' + name + ', tl.int64)\n'
                      for name in _STRIDES)
    candidate = header + '\n):\n' + aliases + body
    restored_header, restored_body = header, body
    for name in _STRIDES:
        restored_header = restored_header.replace(name + ': tl.constexpr', name + ': tl.int64', 1)
        restored_body = re.sub(r'\b' + name + r'_i64\b', name, restored_body)
    if (restored_header, restored_body) != (original_header, original_body):
        raise ValueError('Exact byte inverse failed')
    generated = ('import triton\nimport triton.language as tl\n'
                 'LN2 = tl.constexpr(0.6931471805599453)\n\n' + candidate)
    compile(generated, '<gqa-stride-constants>', 'exec')
    return generated, dict(original_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        exact_byte_inverse=True, specialized_strides=list(_STRIDES),
        address_arithmetic_signed_int64=True, original_while_loop_and_float_math=True,
        layout_and_reducer_unchanged=True)


def compilation_constants(kernel):
    values = {}
    for key, value in kernel.src.constants.items():
        if isinstance(key, tuple):
            if len(key) != 1:
                raise ValueError('Unexpected nested constexpr key')
            key = key[0]
        name = kernel.src.fn.arg_names[key] if isinstance(key, int) else key
        while type(value).__name__ == 'constexpr':
            value = value.value
        values[str(name)] = value if isinstance(value, (int, float, bool, str, type(None))) else repr(value)
    return values


def resources(kernel, out, label):
    metadata = kernel.metadata
    record = dict(hash=kernel.hash, n_regs=kernel.n_regs, n_spills=kernel.n_spills,
                  source_constants=compilation_constants(kernel),
                  source_signature={str(k):str(v) for k,v in kernel.src.signature.items()},
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
    generated, proof = specialize_stride_source((source / 'vllm_fl/patches/biv150_gqa_decode.py').read_text())
    generated_path = args.out / 'generated-gqa-stride-constants.py'
    generated_path.write_text(generated)
    atomic(args.out / 'source-transform-proof.json', proof)
    candidate_module = load_module(generated_path, 'gqa_stride_constants_private')
    result = dict(component_only=True, official_score=False, production_unchanged=True,
                  source_commit=args.commit, production_baseline=MAIN, sdk_jit_sha256=JIT_SHA, source_transform=proof,
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
                _gqa_decode_attention=candidate_module._gqa_decode_attention)
            def run(selected):
                kernel = main_call(selected, c, segments)
                reduce_call(module, c, segments, 4)
                return kernel
            row.update(segments=segments, packed_shape=list(packed.shape), kv_stride=list(k.stride()),
                       stride_specialized=True, same_allocation_and_strides=True)
            old = run(module); gold = c['out'].clone()
            new = run(proxy); torch.cuda.synchronize()
            row['baseline'] = resources(old, args.out, f'b{batch}-baseline')
            row['candidate'] = resources(new, args.out, f'b{batch}-candidate')
            expected = dict(block_table_stride=c['table'].stride(0),
                query_stride_token=c['q'].stride(0), k_stride_block=k.stride(0),
                k_stride_slot=k.stride(1), k_stride_head=k.stride(2),
                v_stride_block=v.stride(0), v_stride_slot=v.stride(1))
            row['actual_stride_constants'] = {name:row['candidate']['source_constants'].get(name)
                                              for name in _STRIDES}
            assert row['actual_stride_constants'] == expected, 'Actual compilation lost constexpr strides' 
            if batch == 64:
                assert old.hash == MODEL_4K_HASH, 'Fixture differs from actual current model kernel'
            assert row['baseline']['metadata']['use_sme'] == 14
            assert row['candidate']['metadata']['use_sme'] == row['baseline']['metadata']['use_sme']
            assert row['candidate']['metadata']['num_stages'] == 1
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
                row['numeric_checks'].append(dict(ragged=ragged, all_output_exact=True,
                    passed=bool(torch.equal(c['out'],baseline))))
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
                    passed=bool(torch.isfinite(changed).all().item()) and bool(torch.equal(changed,c['out']))))
                del graph, changed, baseline
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run(proxy)
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            side_result = c['out'].clone(); run(module); torch.cuda.synchronize()
            row['side_stream_finite'] = bool(torch.isfinite(side_result).all().item())
            row['side_stream_exact'] = bool(torch.equal(side_result,c['out']))
            del side_result
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
                       passed=row['uniform_exact'] and all(x['passed'] for x in row['numeric_checks'])
                       and row['side_stream_finite'] and row['side_stream_exact'])
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
    parser.add_argument('--commit')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-source', action='store_true')
    args=parser.parse_args()
    if args.check_source:
        _,proof=specialize_stride_source((args.source/'vllm_fl/patches/biv150_gqa_decode.py').read_text())
        print(json.dumps(dict(**proof,torch_imported='torch' in sys.modules,gpu_executed=False)))
    else:
        if not args.commit or not args.out: parser.error('--commit and --out required for GPU experiment')
        main(args)
