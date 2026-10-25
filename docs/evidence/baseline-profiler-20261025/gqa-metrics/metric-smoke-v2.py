#!/usr/bin/env python3
"""UNEXECUTED capability workload for a future, owner-serialized ixsys smoke.

Only the shipping GQA producer is timed. 3840 calls should take about two
seconds at the observed 4K cost; the count stays identical across collectors.
Import is inert. No service hooks or production files are changed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import time

EXPECTED_HASH = '641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replays', type=int, default=120)
    parser.add_argument('--unroll', type=int, default=32)
    args = parser.parse_args()
    if args.replays <= 0 or args.unroll <= 0 or args.out.exists():
        raise ValueError('Need positive counts and a new output file')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 9031)) == 0:
            raise RuntimeError('Service is active; owner must serialize this smoke')
    # GPU-idle checking belongs outside ixsys: its metric collector may itself
    # retain a GPU context before starting this process.
    source = args.source_root / 'vllm_fl/patches/biv150_gqa_decode.py'
    helper = args.source_root / 'tools/biv150_benchmark_utils.py'
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, helper)}
    if hashes[str(source)] != '2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1':
        raise RuntimeError('Shipping GQA source changed; reaudit before running')
    if hashes[str(helper)] != 'ff0cd964aad34e6d19609fa4e1d76e7d46b4f98c98a4d925e33d658efd4e2879':
        raise RuntimeError('Benchmark helper changed; reaudit before running')
    import torch
    torch.manual_seed(20261025)
    module = load(source, 'biv150_metric_smoke_shipping_gqa')
    bench = load(helper, 'biv150_metric_smoke_helpers')
    case = bench.inputs(64, 4096, mixed=False)
    # Frozen service BlockTable allocation uses max_model_len131072 / page16.
    # get_device_tensor slices rows only; it retains this 8192-element row stride.
    table = torch.full((64, 8192), -1, device='cuda', dtype=torch.int32)
    table[:, :case['table'].shape[1]].copy_(case['table'])
    case['table'] = table
    # Match the service's K/V page packing; preserve the shuffled page table.
    packed = torch.empty((case['k'].shape[0], 2, 16, 2, 128), device='cuda', dtype=torch.bfloat16)
    key, value = packed.unbind(1)
    key.copy_(case['k']); value.copy_(case['v'])
    case.update(k=key, v=value)
    runtime_strides = dict(query=list(case['q'].stride()), key=list(key.stride()),
                           value=list(value.stride()), block_table=list(table.stride()))
    expected_runtime_strides = dict(query=[2048, 128, 1], key=[8192, 256, 128, 1],
                                    value=[8192, 256, 128, 1], block_table=[8192, 1])
    if runtime_strides != expected_runtime_strides:
        raise RuntimeError(f'Frozen service allocation stride mismatch: {runtime_strides}')
    segments = module._pick_segments(64, 2, 16)
    if segments != 2:
        raise RuntimeError('Shipping 4K segment selection changed')
    # The service passes narrowed scratch views, not the full builder tensors.
    # Although their strides agree, their shapes/contiguity affect Corex's
    # automatic get_corex_sme launch option (14 for service versus 15 for full).
    for name in ('scratch', 'maxima', 'sums'):
        case[name] = case[name].narrow(2, 0, segments)
    constexprs = dict(
        queries_per_kv=8, query_stride_head=case['q'].stride(1),
        v_stride_head=value.stride(2),
        segment_stride_token=case['scratch'].stride(0),
        segment_stride_head=case['scratch'].stride(1),
        segment_stride_segment=case['scratch'].stride(2),
        max_stride_token=case['maxima'].stride(0),
        max_stride_head=case['maxima'].stride(1),
        HEAD_SIZE=128, PAGE_SIZE=16, ROWS=8, SEGMENTS=segments,
    )
    expected_constexprs = dict(
        queries_per_kv=8, query_stride_head=128, v_stride_head=128,
        segment_stride_token=32768, segment_stride_head=2048,
        segment_stride_segment=128, max_stride_token=256, max_stride_head=16,
        HEAD_SIZE=128, PAGE_SIZE=16, ROWS=8, SEGMENTS=2,
    )
    if constexprs != expected_constexprs:
        raise RuntimeError(f'Frozen service constexpr mismatch: {constexprs}')
    compiled = bench.main_call(module, case, segments)
    if str(compiled.hash) != EXPECTED_HASH:
        raise RuntimeError(f'Compiled hash {compiled.hash} != frozen service {EXPECTED_HASH}; no timed region was run')
    bench.reduce_call(module, case, segments, 4)
    torch.cuda.synchronize()
    reference = case['out'].clone()
    if not bool(torch.isfinite(reference).all()):
        raise RuntimeError('Shipping baseline produced nonfinite output')
    for _ in range(8):
        bench.main_call(module, case, segments)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.unroll):
            bench.main_call(module, case, segments)
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Start/stop gates the CUDA trace. Metric sampling is system scope and may
    # cover setup too: counter rows must be restricted to the NVTX workload.
    torch.cuda.profiler.start()
    host_start = time.monotonic_ns()
    with torch.cuda.nvtx.range('biv150_gqa_metric_window'):
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
    host_stop = time.monotonic_ns()
    torch.cuda.profiler.stop()
    elapsed_ms = start.elapsed_time(end)
    # This check is outside the sampled region. It verifies replay did not
    # change the baseline result; it is not an independent precision oracle.
    bench.reduce_call(module, case, segments, 4)
    torch.cuda.synchronize()
    numeric_equal = bool(torch.isfinite(case['out']).all() and torch.equal(case['out'], reference))
    record = dict(diagnostic_only=True, official_score=False, service_installed=False,
                  production_timing=False, source_hashes=hashes,
                  compiled_hash=str(compiled.hash), n_regs=compiled.n_regs, n_spills=compiled.n_spills,
                  shared_bytes=compiled.metadata.shared, grid=[64, 2, 2], warps=1, stages=1,
                  query_stride=list(case['q'].stride()), key_stride=list(key.stride()),
                  value_stride=list(value.stride()),
                  block_table_shape=list(case['table'].shape), block_table_stride=list(case['table'].stride()),
                  scratch_shapes={name: list(case[name].shape) for name in ('scratch', 'maxima', 'sums')},
                  use_sme=compiled.metadata.use_sme,
                  launch_constexprs=constexprs, batch=64, kv_bound=4096,
                  producer_only=True, includes_reducer_in_timed_region=False,
                  graph_unroll=args.unroll, replays=args.replays, timed_calls=args.unroll * args.replays,
                  gpu_elapsed_ms=elapsed_ms, gpu_ms_per_call=elapsed_ms / (args.unroll * args.replays),
                  host_start_monotonic_ns=host_start, host_stop_monotonic_ns=host_stop,
                  host_elapsed_ms=(host_stop - host_start) / 1e6,
                  replay_numeric_equal=numeric_equal,
                  limitations=['Synthetic inputs; one repeatedly reused KV working set.',
                               'No independent numerical oracle; this executes the unchanged baseline.',
                               'Compiled hash is gated to the frozen service; runtime inputs remain synthetic.',
                               'System-scope metrics are not per-kernel counters; isolate the NVTX region.',
                               'No bandwidth/occupancy conclusion until requested metrics return valid samples.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record), flush=True)
    if not numeric_equal:
        raise RuntimeError('Baseline graph replay output changed')


if __name__ == '__main__':
    main()
