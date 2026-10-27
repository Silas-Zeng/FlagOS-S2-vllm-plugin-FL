#!/usr/bin/env python3
"""Opt-in compile/numeric only for stock-cache decode quantization. No timing or hook."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from types import SimpleNamespace

from bench_biv150_konly_decode import (BASE_HASHES, CASES, JIT, JIT_SHA, ROOT, Warmup, atomic, module, preserve)
from bench_biv150_konly_decode_numeric import (cache, cache_guard, exact_bytes, host_bits, host_bytes,
                                               loaded, output_buffers, output_check, poison_outputs, source_controls)
from biv150_konly_decode_factory import MAIN, NAME as PACKED_NAME, build as build_packed
from biv150_konly_decode_ir import audit
from biv150_konly_decode_numeric_contract import codec_vectors, reference_gates
from biv150_konly_decode_numeric_oracle import attention_references, encoded_cache_bytes
from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_decode_sdk import INSTALLED_MATH, MATH_SHA, check_frontend
from biv150_konly_stock_diagnostic_factory import NAME, build, validate_stock_contract


class Extra:
    def __init__(self, kernel, **extras):
        self.kernel, self.extras = kernel, extras

    def __getitem__(self, grid):
        def launch(**kwargs):
            return self.kernel[grid](**kwargs, **self.extras)
        return launch


def m16_probe(probe, out, torch, np):
    vectors = codec_vectors()
    host = torch.tensor(vectors, dtype=torch.bfloat16)
    expected = [quantize_v2(row) for row in vectors]
    qc = torch.tensor([v[0] for v in expected], dtype=torch.int8)
    qs = torch.tensor([v[1] for v in expected], dtype=torch.float32)
    n = len(vectors)
    source = host.cuda()
    # Guard a whole accidentally unmasked M16 tile, including N141's last tile.
    cb = torch.full((n * 128 + 4096,), 85, dtype=torch.int8, device='cuda')
    sb = torch.full((n + 128,), -98765.5, dtype=torch.float32, device='cuda')
    codes, scales = cb[2048:-2048].view(n, 128), sb[64:-64]
    codes.fill_(-128)
    scales.fill_(math.nan)
    kernel = probe.probe_stock_m16[((n + 15) // 16,)](source, codes, scales, n, num_warps=1, num_stages=1)
    torch.cuda.synchronize()
    resource, _ = loaded(kernel, out, 'codec-M16')
    np.savez_compressed(out / 'codec-M16-witness.npz', source_bits=host.view(torch.int16).numpy().view(np.uint16),
                        actual_codes=codes.cpu().numpy(), actual_scale_bits=host_bits(scales, torch).numpy().view(np.uint32),
                        expected_codes=qc.numpy(), expected_scale_bits=host_bits(qs, torch).numpy().view(np.uint32))
    atomic(out / 'codec-M16-progress.json', dict(passed=False, resources=resource, witness='codec-M16-witness.npz'))
    assert torch.equal(codes.cpu(), qc) and torch.equal(host_bits(scales, torch), host_bits(qs, torch)), 'M16 codec bits mismatch'
    assert bool((cb[:2048].cpu() == 85).all() and (cb[-2048:].cpu() == 85).all())
    assert bool((sb[:64].cpu() == -98765.5).all() and (sb[-64:].cpu() == -98765.5).all())
    assert torch.equal(host_bits(source, torch), host_bits(host, torch))
    atomic(out / 'codec-M16-progress.json', dict(passed=True, resources=resource, witness='codec-M16-witness.npz'))
    return dict(passed=True, m=16, vectors=n, codes_and_scale_bits_exact=True, source_unchanged=True,
                canaries=True, resources=resource)


def run_device(args, generated, proof):
    assert args.commit and args.out, 'Explicit device mode needs frozen --commit and fresh --out'
    assert subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == args.commit
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', MAIN, '--', 'vllm_fl', 'csrc'], check=True)
    assert hashlib.sha256(JIT.read_bytes()).hexdigest() == JIT_SHA
    assert hashlib.sha256(INSTALLED_MATH.read_bytes()).hexdigest() == MATH_SHA
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True), 'GPU busy; do not queue'
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0, 'Service9031 active'
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(source_commit=args.commit, completed=False, passed=False, compile_only=args.compile_only,
                  diagnostic_only=True, scope='stock_bf16_decode_quantization_only', throughput_candidate=False,
                  service_installed=False, full_prefill_quantization_represented=False, precision_gate=False,
                  timing_attempted=False, proof=proof, records=[], resources={}, errors=[])
    def save():
        atomic(args.out / 'result.json', result)
    save()
    try:
        import os
        os.environ['TRITON_CACHE_DIR'] = str(args.out / 'triton-cache')
        import numpy as np
        import torch
        import triton
        from biv150_benchmark_utils import inputs, main_call, reduce_call
        torch.set_num_threads(4)
        torch.set_flush_denormal(False)  # CPU oracle only; no claim about target GPU FTZ.
        target = triton.runtime.driver.active.get_current_target()
        assert (target.backend, target.arch, target.warp_size) == ('corex', 71, 64)
        result['frontend'] = check_frontend(ROOT)
        base = module(ROOT / 'vllm_fl/patches/biv150_gqa_decode.py', 'cjj_stock_diag_base')
        assert base.TARGET_PROGRAMS == 256
        support = module(ROOT / 'tools/kernels/biv150_konly_decode_support.py', 'cjj_konly_decode_support')
        probe = module(ROOT / 'tools/kernels/biv150_konly_stock_codec_falsifier.py', 'cjj_stock_diag_probe')
        (args.out / 'support-source.py.txt').write_bytes((ROOT / 'tools/kernels/biv150_konly_decode_support.py').read_bytes())
        paths = {}
        sources = {'stock': generated, 'packed': build_packed((ROOT / 'vllm_fl/patches/biv150_gqa_decode.py').read_text(), 'inline')[0]}
        for label, text in sources.items():
            path = args.out / ('generated-' + label + '.py')
            path.write_text(text)
            paths[label] = module(path, 'cjj_stock_diag_' + label)
        if not args.compile_only:
            result['stage'] = 'M16_codec'
            result['codec'] = m16_probe(probe, args.out, torch, np)
            save()  # A failed M16 hard gate cannot proceed to attention.
        batch, length = CASES[args.case]
        segments = base._pick_segments(batch, 2, 16)
        pages = batch * ((length + 15) // 16)
        result['geometry'] = dict(batch=batch, kv=length, pages=pages, segments=segments, table_pitch=8192)
        if args.compile_only:
            q = torch.empty((batch, 16, 128), device='cuda', dtype=torch.bfloat16)
            cold = torch.empty((pages, 2, 16, 2, 128), device='cuda', dtype=torch.bfloat16)
            raw = torch.empty((pages * 16384,), device='cuda', dtype=torch.uint8)
            encoded = dict(raw=raw, code=torch.as_strided(raw.view(torch.int8), (pages, 16, 2, 128), (16384, 256, 128, 1)),
                           scale=raw.view(torch.float32), value=torch.as_strided(raw.view(torch.bfloat16), (pages, 16, 2, 128), (8192, 256, 128, 1), 4096))
        else:
            fixture = inputs(batch, length)
            q = fixture['q']
            q0, table0 = q.cpu(), fixture['table'].cpu()
            cold = torch.empty((pages, 2, 16, 2, 128), device='cuda', dtype=torch.bfloat16)
            cold[:, 0].copy_(fixture['k'])
            cold[:, 1].copy_(fixture['v'])
            encoded = cache(pages, torch)
            encoded['raw'].fill_(0xa5)
        key, value = cold.unbind(1)
        validate_stock_contract(payload_format='stock-bf16', query_dtype=str(q.dtype).split('.')[-1],
                                key_dtype=str(key.dtype).split('.')[-1], value_dtype=str(value.dtype).split('.')[-1], dimension_stride=key.stride(3))
        table = torch.empty((batch, 8192), dtype=torch.int32, device='cuda')[:, :(length + 15) // 16]
        lengths = torch.empty(batch, dtype=torch.int32, device='cuda')
        cases, backs = {}, {}
        for label in ('baseline', 'stock', 'packed'):
            if args.compile_only:
                io = {k: torch.empty((batch, 16, 16, 128) if k == 'scratch' else (batch, 16, 16), device='cuda', dtype=torch.float32).narrow(2, 0, segments)
                      for k in ('scratch', 'maxima', 'sums')}
                io['out'] = torch.empty_like(q)
            else:
                io, backs[label] = output_buffers(batch, segments, torch)
            cases[label] = dict(q=q, k=encoded['code'] if label == 'packed' else key,
                                v=encoded['value'] if label == 'packed' else value, table=table, lengths=lengths, **io)
        qcode_unused = torch.empty_like(q, dtype=torch.int8)
        qscale_unused = torch.empty((batch, 16), device='cuda', dtype=torch.float32)
        if not args.compile_only:
            qcode_unused.fill_(-128)
            qscale_unused.fill_(math.nan)
        # The inline variant never reads these parameters, but its static pointer
        # dtype contract still applies and must not be satisfied with BF16 Q.
        extra = dict(query_code_ptr=qcode_unused, query_scale_ptr=qscale_unused, key_scale_ptr=encoded['scale'])
        proxies = {'baseline': base,
                   'stock': SimpleNamespace(LOG2E=base.LOG2E, _gqa_decode_attention=getattr(paths['stock'], NAME)),
                   'packed': SimpleNamespace(LOG2E=base.LOG2E, _gqa_decode_attention=Extra(getattr(paths['packed'], PACKED_NAME), **extra))}
        def capture(label, kernel):
            resource, strings = preserve(kernel, args.out, label) if args.compile_only else loaded(kernel, args.out, label)
            result['resources'][label] = resource
            save()  # Preserve raw evidence before any semantic assertion can fail.
            if label == 'baseline-producer':
                assert str(kernel.hash) == BASE_HASHES[batch], 'Current-main BF16 control specialization changed'
            if label in ('stock-producer', 'packed-producer'):
                resource['actual_ir'] = audit(strings, sources[label.split('-')[0]])
                save()
                assert resource['actual_ir']['passed'], 'Actual decoder I8/PV/C-chain gate failed'
            if label.endswith('producer'):
                assert kernel.metadata.num_warps == 1 and kernel.metadata.num_stages == 1
            if label == 'original-reducer':
                assert kernel.metadata.num_warps == 4
            save()
        if args.compile_only:
            codec_source = torch.empty((141, 128), device='cuda', dtype=torch.bfloat16)
            codec_codes = torch.empty((141, 128), device='cuda', dtype=torch.int8)
            codec_scales = torch.empty(141, device='cuda', dtype=torch.float32)
            capture('codec-M16', probe.probe_stock_m16.warmup(codec_source, codec_codes, codec_scales, 141, grid=(9,), num_warps=1, num_stages=1))
            capture('initial-encoder', support.encode_initial.warmup(key, value, encoded['code'], encoded['scale'], encoded['value'], pages, grid=(pages * 32,), num_warps=1, num_stages=1))
            for label in proxies:
                kernel = base._gqa_decode_attention if label == 'baseline' else getattr(paths[label], NAME if label == 'stock' else PACKED_NAME)
                proxy = SimpleNamespace(LOG2E=base.LOG2E, _gqa_decode_attention=Warmup(kernel, **(extra if label == 'packed' else {})))
                capture(label + '-producer', main_call(proxy, cases[label], segments))
            capture('original-reducer', reduce_call(SimpleNamespace(_reduce_partials=Warmup(base._reduce_partials)), cases['stock'], segments, 4))
        else:
            result['stage'] = 'initial_encoder'
            frozen_k, frozen_v = key.cpu(), value.cpu()
            cold_sha = hashlib.sha256(host_bytes(cold, torch)).hexdigest()
            expected, codes_cpu, scales_cpu, oracle = encoded_cache_bytes(frozen_k, frozen_v, torch, np)
            encoder = support.encode_initial[(pages * 32,)](key, value, encoded['code'], encoded['scale'], encoded['value'], pages, num_warps=1, num_stages=1)
            torch.cuda.synchronize()
            capture('initial-encoder', encoder)
            result['initial_encoder_exact'] = exact_bytes(host_bytes(encoded['raw'], torch), expected, 'packed comparison encoder', np)
            result['cpu_encoder_oracle'] = oracle
            kernels = {'initial-encoder': encoder}
            encoded_sha = hashlib.sha256(host_bytes(encoded['raw'], torch)).hexdigest()
            controls, snapshots = [], []
            def step(label):
                kernels[label + '-producer'] = main_call(proxies[label], cases[label], segments)
                kernels['original-reducer'] = reduce_call(base, cases[label], segments, 4)
            def verify(label, control):
                torch.cuda.synchronize()
                assert hashlib.sha256(host_bytes(cold, torch)).hexdigest() == cold_sha, 'Stock cache was modified'
                assert hashlib.sha256(host_bytes(encoded['raw'], torch)).hexdigest() == encoded_sha and cache_guard(encoded, torch)
                assert torch.equal(host_bits(q, torch), host_bits(control['q'], torch))
                assert torch.equal(table.cpu(), control['table']) and lengths.cpu().tolist() == control['lengths']
                assert bool((qcode_unused.cpu() == -128).all() and torch.isnan(qscale_unused.cpu()).all()), 'Inline unexpectedly wrote Qtemp'
                return output_check(cases[label], backs[label], control['lengths'], segments, torch)
            for iteration in range(3):
                result['stage'] = 'attention-' + str(iteration)
                control = source_controls(q0, table0, batch, length, iteration, torch)
                control['slots'] = [-1] * batch  # No append writer in this stock-cache bridge.
                controls.append(control)
                q.copy_(control['q'])
                table.copy_(control['table'])
                lengths.copy_(torch.tensor(control['lengths'], dtype=torch.int32))
                selected = [0, 1, batch - 1] if iteration == 0 else list(range(8)) + [batch - 1]
                refs, _ = attention_references(frozen_k, frozen_v, codes_cpu, scales_cpu, control, selected, torch, np)
                outputs, snap = {}, {}
                for label in proxies:
                    poison_outputs(backs[label])
                    step(label)
                    snap[label], outputs[label] = verify(label, control)
                    if label + '-producer' not in result['resources']:
                        capture(label + '-producer', kernels[label + '-producer'])
                    if 'original-reducer' not in result['resources']:
                        capture('original-reducer', kernels['original-reducer'])
                checks = {label: [dict(row=r['row'], length=r['length'], integer_bound=r['integer_bound'],
                                      **reference_gates(outputs['baseline'][r['row']].float().reshape(-1).tolist(), outputs[label][r['row']].float().reshape(-1).tolist(),
                                                        r['original'].reshape(-1).tolist(), r['quantized'].reshape(-1).tolist())) for r in refs] for label in ('stock', 'packed')}
                assert all(v['baseline_hard_passed'] and v['quantized_operand']['pointwise_passed'] for rows in checks.values() for v in rows), 'Baseline or quantized-operand implementation hard gate failed'
                np.savez_compressed(args.out / ('numeric-witness-' + str(iteration) + '.npz'), q_bits=control['q'].view(torch.int16).numpy().view(np.uint16),
                                    table=control['table'].numpy(), lengths=np.array(control['lengths'], dtype=np.int32), selected_rows=np.array(selected, dtype=np.int32),
                                    original_reference=np.stack([r['original'].numpy() for r in refs]), quantized_reference=np.stack([r['quantized'].numpy() for r in refs]),
                                    **{label + '_output_bits': output.view(torch.int16).numpy().view(np.uint16) for label, output in outputs.items()})
                result['records'].append(dict(iteration=iteration, checks=checks, stock_packed_equal={k: torch.equal(snap['stock'][k], snap['packed'][k]) for k in snap['stock']}))
                snapshots.append(snap)
                save()
            result['stage'] = 'changed_graph_and_side'
            q.copy_(controls[0]['q']); table.copy_(controls[0]['table']); lengths.copy_(torch.tensor(controls[0]['lengths'], dtype=torch.int32))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step('stock')
            for iteration in (1, 2, 0):
                control = controls[iteration]
                q.copy_(control['q']); table.copy_(control['table']); lengths.copy_(torch.tensor(control['lengths'], dtype=torch.int32))
                poison_outputs(backs['stock']); graph.replay()
                got, _ = verify('stock', control)
                assert all(torch.equal(got[k], snapshots[iteration]['stock'][k]) for k in got), 'Changed metadata Graph differs from eager'
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                poison_outputs(backs['stock']); step('stock')
            torch.cuda.current_stream().wait_stream(side)
            got, _ = verify('stock', controls[0])
            assert all(torch.equal(got[k], snapshots[0]['stock'][k]) for k in got), 'Side stream differs from eager'
            result['graph_changed_inputs_poison_exact'] = result['side_eager_exact'] = True
            result['graph_changed_fields'] = ['Q', 'block_table', 'lengths']
            result['kv_append_or_changed_cache_writer_tested'] = False
            result['stock_cache_unchanged_sha256'] = cold_sha
        result['compile_passed'] = True
        result['quantized_operand_device_gate_verified'] = not args.compile_only
        result['candidate_original_precision_passed'] = (not args.compile_only and all(v['candidate_numeric_passed'] for r in result['records'] for v in r['checks']['stock']))
        result['completed'] = True
        result['passed'] = result['candidate_original_precision_passed']
        result['status'] = 'compile_only_not_numeric_qualified' if args.compile_only else 'decode_only_accuracy_diagnostic_no_timing'
        result['stage'] = 'completed'
        save()
    except Exception as error:
        result['errors'].append(dict(stage=result.get('stage'), type=type(error).__name__, message=str(error)))
        (args.out / 'error-traceback.txt').write_text(traceback.format_exc())
        save()
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--compile-only', action='store_true')
    mode.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--case', choices=CASES, default='4k')
    parser.add_argument('--commit')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    generated, proof = build((ROOT / 'vllm_fl/patches/biv150_gqa_decode.py').read_text())
    result = run_device(args, generated, proof) if args.compile_only or args.run_gpu else dict(cpu_only=True, proof=proof,
             compile_entry_prepared=True, numeric_entry_prepared=True, timing_entry_prepared=False,
             torch_imported='torch' in sys.modules, triton_imported='triton' in sys.modules, numpy_imported='numpy' in sys.modules,
             service_installed=False, gpu_executed=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
