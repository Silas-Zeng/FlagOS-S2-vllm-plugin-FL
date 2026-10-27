#!/usr/bin/env python3
"""Packaged stock-QK8 service-shape micro. Default CPU; no timing/model/service hook."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from types import SimpleNamespace

from biv150_stock_service_micro_common import (atomic, host_bits, host_bytes, loaded, m16_probe,
                                                module, output_buffers, output_check, poison_outputs, preserve)
from biv150_stock_service_abi import compare
from biv150_konly_decode_ir import audit
from biv150_konly_decode_numeric_contract import reference_gates
from biv150_konly_decode_numeric_oracle import attention_references, quantize_cpu

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / 'vllm_fl/patches/biv150_stock_qk8_diagnostic.py'
MAIN = '56f1838192abc940b3fb9accef490022ac125000'
JIT = Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py')
JIT_SHA = '66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'


class Warmup:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def compile_only(**kwargs):
            return self.kernel.warmup(**kwargs, grid=grid)
        return compile_only


def cases():
    return [(b, 129) for b in (1, 8, 16, 32, 40, 64)] + [(1, 4096)]


def control(q0, table0, batch, length, iteration, torch):
    q = q0.clone()
    table = table0.roll(iteration, 0)
    lengths = [length] * batch
    if iteration == 1:
        q = -q.roll(1, 1)
        for row in range(batch):
            lengths[row] = (17, 0, 1, 15, 16, 63, 100, length - 1)[row % 8]
    elif iteration == 2:
        q.zero_()
    elif iteration == 3:
        lengths[0] = 0
    return dict(q=q, table=table, lengths=lengths, slots=[-1] * batch,
                new_k=torch.zeros((batch, 2, 128), dtype=torch.bfloat16),
                new_v=torch.zeros((batch, 2, 128), dtype=torch.bfloat16))


def run_device(args, hook):
    assert args.commit and args.out, 'Explicit device mode requires exact --commit and fresh --out'
    assert subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == args.commit
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    assert hashlib.sha256((ROOT / 'tools/biv150_benchmark_utils.py').read_bytes()).hexdigest() == 'ff0cd964aad34e6d19609fa4e1d76e7d46b4f98c98a4d925e33d658efd4e2879'
    assert hashlib.sha256(JIT.read_bytes()).hexdigest() == JIT_SHA
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True), 'GPU busy; do not queue'
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0, 'Service9031 active'
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(source_commit=args.commit, **hook.source_contract(), completed=False, passed=False,
                  compile_only=args.compile_only, gpu_executed=False, precision_gate=False,
                  diagnostic_only=True, model105_passed=False, full_prefill_quantization=False,
                  timing_attempted=False, service_installed=False, records=[], errors=[])
    def save():
        atomic(args.out / 'result.json', result)
    save()
    try:
        import os
        os.environ['TRITON_CACHE_DIR'] = str(args.out / 'triton-cache')
        os.environ[hook.ENABLE] = '0'  # This is a direct micro, never install the service hook.
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch
        import triton
        from biv150_benchmark_utils import inputs, main_call, reduce_call
        torch.set_num_threads(4)
        torch.set_flush_denormal(False)  # CPU oracles only.
        target = triton.runtime.driver.active.get_current_target()
        assert (target.backend, target.arch, target.warp_size) == ('corex', 71, 64)
        result['target'] = [target.backend, target.arch, target.warp_size]
        gqa = importlib.import_module('vllm_fl.patches.biv150_gqa_decode')
        assert Path(gqa.__file__).resolve() == ROOT / 'vllm_fl/patches/biv150_gqa_decode.py'
        assert gqa.TARGET_PROGRAMS == 256
        raw, candidate = hook.prepare_candidate(gqa)
        assert gqa._gqa_decode_attention is raw, 'Micro unexpectedly changed GQA binding'
        source = (ROOT / 'vllm_fl/patches/biv150_stock_qk8_kernel.py').read_text()
        (args.out / 'packaged-kernel.py.txt').write_text(source)
        (args.out / 'packaged-codec.py.txt').write_bytes((ROOT / 'vllm_fl/patches/biv150_stock_qk8_codec.py').read_bytes())
        probe = module(ROOT / 'tools/kernels/biv150_stock_service_codec_probe.py', 'cjj_stock_service_m16')
        if args.compile_only:
            a = torch.empty((141, 128), device='cuda', dtype=torch.bfloat16)
            b = torch.empty_like(a, dtype=torch.int8)
            c = torch.empty(141, device='cuda', dtype=torch.float32)
            kernel = probe.probe_stock_m16.warmup(a, b, c, 141, grid=(9,), num_warps=1, num_stages=1)
            result['codec_compile'], _ = preserve(kernel, args.out, 'codec-M16')
        else:
            result['stage'] = 'M16_codec'
            result['codec'] = m16_probe(probe, args.out, torch, np)
            result['gpu_executed'] = True
        save()
        certificates = []
        for batch, length in cases():
            segments = gqa._pick_segments(batch, 2, 16)
            name = 'b' + str(batch) + '-k' + str(length)
            record = dict(batch=batch, kv=length, segments=segments, input_seed=20261023 + batch + length,
                          controls=[], resources={}, passed=False)
            result['records'].append(record)
            result['stage'] = name
            save()
            pages = batch * ((length + 15) // 16)
            if args.compile_only:
                q = torch.empty((batch, 16, 128), device='cuda', dtype=torch.bfloat16)
                backing = torch.empty(pages * 16384 + 512, device='cuda', dtype=torch.uint8)
            else:
                fixture = inputs(batch, length)
                q = fixture['q']
                q0, table0 = q.cpu(), fixture['table'].cpu()
                backing = torch.full((pages * 16384 + 512,), 90, device='cuda', dtype=torch.uint8)
            packed = backing[256:-256].view(torch.bfloat16).reshape(pages, 2, 16, 2, 128)
            key, value = packed.unbind(1)
            if not args.compile_only:
                key.copy_(fixture['k']); value.copy_(fixture['v'])
                frozen_k, frozen_v = key.cpu(), value.cpu()
                codes_cpu, scales_cpu, codec_info = quantize_cpu(frozen_k, torch, np)
                stock_hash = hashlib.sha256(host_bytes(packed, torch)).hexdigest()
                record['stock_cache_sha256'] = stock_hash
                record['cpu_codec'] = codec_info
            table = torch.empty((batch, 8192), device='cuda', dtype=torch.int32)[:, :(length + 15) // 16]
            lengths = torch.empty(batch, device='cuda', dtype=torch.int32)
            ios, backs = {}, {}
            for label in ('baseline', 'candidate'):
                if args.compile_only:
                    io = {k: torch.empty((batch, 16, 16, 128) if k == 'scratch' else (batch, 16, 16), dtype=torch.float32, device='cuda').narrow(2, 0, segments)
                          for k in ('scratch', 'maxima', 'sums')}
                    io['out'] = torch.empty_like(q)
                else:
                    io, backs[label] = output_buffers(batch, segments, torch)
                ios[label] = dict(q=q, k=key, v=value, table=table, lengths=lengths, **io)
            kernels, artifacts = {}, {}
            def capture(label, compiled):
                resource, strings = preserve(compiled, args.out, name + '-' + label) if args.compile_only else loaded(compiled, args.out, name + '-' + label)
                record['resources'][label] = resource; artifacts[label] = strings; save()
                if label == 'candidate':
                    resource['actual_ir'] = audit(strings, source); save()
                    assert resource['actual_ir']['passed'], 'Actual packaged I8/PV/C-chain mechanism rejected'
                if label in ('baseline', 'candidate'):
                    assert compiled.metadata.num_warps == 1 and compiled.metadata.num_stages == 1
                if label == 'reducer':
                    assert compiled.metadata.num_warps == 4
            proxies = {label: SimpleNamespace(LOG2E=gqa.LOG2E, _gqa_decode_attention=Warmup(kernel) if args.compile_only else kernel)
                       for label, kernel in (('baseline', raw), ('candidate', candidate))}
            if args.compile_only:
                for label in proxies:
                    capture(label, main_call(proxies[label], ios[label], segments))
                capture('reducer', reduce_call(SimpleNamespace(_reduce_partials=Warmup(gqa._reduce_partials)), ios['candidate'], segments, 4))
                record['actual_typed_abi'] = compare(artifacts['baseline'], artifacts['candidate'])
                record['compile_passed'] = True; save(); continue
            def upload(c):
                q.copy_(c['q']); table.copy_(c['table']); lengths.copy_(torch.tensor(c['lengths'], dtype=torch.int32))
            def step(label):
                kernels[label] = main_call(proxies[label], ios[label], segments)
                kernels['reducer'] = reduce_call(gqa, ios[label], segments, 4)
            def verify(label, c):
                torch.cuda.synchronize()
                assert hashlib.sha256(host_bytes(packed, torch)).hexdigest() == stock_hash, 'Stock cache was changed'
                assert bool((backing[:256].cpu() == 90).all() and (backing[-256:].cpu() == 90).all())
                assert torch.equal(host_bits(q, torch), host_bits(c['q'], torch))
                assert torch.equal(table.cpu(), c['table']) and lengths.cpu().tolist() == c['lengths']
                return output_check(ios[label], backs[label], c['lengths'], segments, torch)
            controls, eager = [], []
            for iteration in range(4):
                c = control(q0, table0, batch, length, iteration, torch); controls.append(c); upload(c)
                selected = sorted({0, min(1, batch - 1), batch - 1}) if iteration != 1 else sorted(set(range(min(batch, 8))) | {batch - 1})
                refs, _ = attention_references(frozen_k, frozen_v, codes_cpu, scales_cpu, c, selected, torch, np)
                snapshots, outputs = {}, {}
                for label in proxies:
                    poison_outputs(backs[label]); step(label)
                    snapshots[label], outputs[label] = verify(label, c)
                    if label not in record['resources']:
                        capture(label, kernels[label])
                    if 'reducer' not in record['resources']:
                        capture('reducer', kernels['reducer'])
                if 'actual_typed_abi' not in record:
                    record['actual_typed_abi'] = compare(artifacts['baseline'], artifacts['candidate']); save()
                checks = [dict(row=r['row'], length=r['length'], integer_bound=r['integer_bound'],
                               **reference_gates(outputs['baseline'][r['row']].float().reshape(-1).tolist(),
                                   outputs['candidate'][r['row']].float().reshape(-1).tolist(),
                                   r['original'].reshape(-1).tolist(), r['quantized'].reshape(-1).tolist())) for r in refs]
                np.savez_compressed(args.out / (name + '-control' + str(iteration) + '.npz'),
                    q_bits=c['q'].view(torch.int16).numpy().view(np.uint16), table=c['table'].numpy(),
                    lengths=np.array(c['lengths'], dtype=np.int32), selected_rows=np.array(selected, dtype=np.int32),
                    original_reference=np.stack([r['original'].numpy() for r in refs]), quantized_reference=np.stack([r['quantized'].numpy() for r in refs]),
                    **{k + '_output_bits': v.view(torch.int16).numpy().view(np.uint16) for k, v in outputs.items()})
                record['controls'].append(dict(iteration=iteration, checks=checks)); save()
                assert all(v['baseline_hard_passed'] and v['quantized_operand']['pointwise_passed'] for v in checks), 'Baseline/quantized-operand hard gate failed'
                eager.append(snapshots['candidate'])
            upload(controls[0])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step('candidate')
            for iteration in (1, 2, 3, 0):
                upload(controls[iteration]); poison_outputs(backs['candidate']); graph.replay()
                got, _ = verify('candidate', controls[iteration])
                assert all(torch.equal(got[k], eager[iteration][k]) for k in got), 'Changed Q/table/length Graph differs from eager'
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                poison_outputs(backs['candidate']); step('candidate')
            torch.cuda.current_stream().wait_stream(side)
            got, _ = verify('candidate', controls[0])
            assert all(torch.equal(got[k], eager[0][k]) for k in got), 'Side differs from eager'
            row = dict(batch=batch, kv=length, segments=segments, quantized_operand_gate_passed=True,
                       baseline_gate_passed=True, source_cache_unchanged=True, canaries=True,
                       actual_ir_passed=True, typed_abi_match=True, eager_warm_passed=True,
                       graph_changed_q_table_lengths_exact=True, side_exact=True,
                       original_bf16_proxy_passed=all(v['candidate_numeric_passed'] for c in record['controls'] for v in c['checks']))
            certificates.append(row); record.update(passed=True, implementation_hard_gates=row); save()
            assert gqa._gqa_decode_attention is raw and not hook.stats()['installed']
        if args.compile_only:
            result.update(completed=True, compile_passed=True, status='compile_only_no_micro_certificate')
        else:
            certificate = dict(schema=hook.SCHEMA, **hook.source_contract(), target=result['target'], source_commit=args.commit,
                               gpu_executed=True, passed=True, m16_codec_bits_exact=True, cases=certificates,
                               diagnostic_only=True, model105_passed=False, full_prefill_quantization=False, timing_attempted=False)
            hook.certificate_contract(certificate)
            atomic(args.out / 'micro-certificate.json', certificate)
            proxy_passed = all(r['original_bf16_proxy_passed'] for r in certificates)
            result.update(completed=True, passed=proxy_passed, implementation_hard_gates_passed=True,
                          status='packaged_decode_micro_hard_gates_only', original_bf16_proxy_passed=proxy_passed,
                          micro_certificate='micro-certificate.json')
        save()
    except Exception as error:
        result['errors'].append(dict(stage=result.get('stage'), type=type(error).__name__, message=str(error)))
        (args.out / 'error-traceback.txt').write_text(traceback.format_exc()); save(); raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--compile-only', action='store_true')
    group.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--commit'); parser.add_argument('--out', type=Path)
    args = parser.parse_args(); hook = module(HOOK, 'cjj_stock_service_default_cpu')
    result = run_device(args, hook) if args.compile_only or args.run_gpu else dict(cpu_only=True,
        source=hook.source_contract(), cases=[dict(batch=b, kv=k) for b, k in cases()], timing_prepared=False,
        compile_entry_prepared=True, numeric_entry_prepared=True, certificate_available=False,
        torch_imported='torch' in sys.modules, triton_imported='triton' in sys.modules, numpy_imported='numpy' in sys.modules,
        service_installed=False, gpu_executed=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
