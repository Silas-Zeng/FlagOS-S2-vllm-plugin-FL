"""Diagnostic INT8 KV writer + fused GQA; never registers a serving implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
from types import SimpleNamespace
import traceback

CASES = {'steady4k': (64, 4096, False), 'steady16k': (32, 16384, False),
         'ragged': (8, 257, True)}
SOURCE_SHA = '2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
SERVICE_4K_HASH = '641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff'


def main(args):
    chosen = args.case or list(CASES)
    if not args.run_gpu:
        print(json.dumps(dict(cases={k: CASES[k] for k in chosen}, service_installed=False,
                              quantized_storage_fraction=(128 + 4) / 256,
                              model_accuracy='not tested', throughput='not tested')))
        return
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0, 'Service active'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True), 'GPU busy'
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import biv150_benchmark_utils as bench
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    source = root / 'vllm_fl/patches/biv150_gqa_decode.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    base = bench.load_module(source, 'int8_kv_shipping_gqa')
    kernel_path = root / 'tools/kernels/biv150_int8_kv.py'
    own = bench.load_module(kernel_path, 'int8_kv_experimental_kernels')
    assert base.TARGET_PROGRAMS == 256
    result = dict(completed=False, service_installed=False, official_score=False,
                  model_accuracy_tested=False, capacity_reclaimed=False,
                  source_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                  operator_sha256=hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
                  baseline_source_sha256=SOURCE_SHA, cases=[], errors=[])

    def save():
        path = args.out / 'result.tmp';path.write_text(json.dumps(result, indent=2) + '\n');path.replace(args.out / 'result.json')

    def compare(a, b):
        a, b = a.float(), b.float()
        error = (a-b).abs()
        return dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                    max_abs=float(error.max()), mse=float((error*error).mean()),
                    tol_005_violations=int((error > .005 + .005*b.abs()).sum()))

    def reference(case, *, quantized=False):
        # Independent CPU FP64 attention. Quantized oracle first reconstructs
        # exactly the intended BF16 operands, separating implementation bugs
        # from the intentionally approximate cache representation.
        answers, selected = [], sorted(set([0, min(1, case['q'].shape[0]-1), case['q'].shape[0]-1]))
        for seq in selected:
            length = int(case['lengths'][seq].cpu())
            if length == 0:
                answers.append(torch.zeros(16, 128, dtype=torch.float64));continue
            ids = case['table'][seq, :(length+15)//16].long()
            key = case['k'][ids].cpu().reshape(-1, 2, 128)[:length]
            value = case['v'][ids].cpu().reshape(-1, 2, 128)[:length]
            if quantized:
                ks = case['key_scale'][ids].cpu().reshape(-1, 2)[:length]
                vs = case['value_scale'][ids].cpu().reshape(-1, 2)[:length]
                key = (key.float() * ks[..., None]).to(torch.bfloat16)
                value = (value.float() * vs[..., None]).to(torch.bfloat16)
            q = case['q'][seq].cpu().double().reshape(2, 8, 128)
            scores = torch.einsum('ghd,tgd->ght', q, key.double()) * 128**-.5
            answer = torch.einsum('ght,tgd->ghd', torch.softmax(scores, -1), value.double())
            answers.append(answer.reshape(16, 128))
        return selected, torch.stack(answers)

    def resources(compiled):
        return dict(hash=str(compiled.hash), n_regs=compiled.n_regs,
                    n_spills=compiled.n_spills, shared=compiled.metadata.shared,
                    use_sme=getattr(compiled.metadata, 'use_sme', None))

    save()
    try:
        for name in chosen:
            batch, length, mixed = CASES[name]
            c = bench.inputs(batch, length, mixed)
            packed = torch.empty((c['k'].shape[0], 2, 16, 2, 128), device='cuda', dtype=torch.bfloat16)
            key, value = packed.unbind(1);key.copy_(c['k']);value.copy_(c['v']);c.update(k=key, v=value)
            table = torch.full((batch, 8192), -1, device='cuda', dtype=torch.int32)
            table[:, :c['table'].shape[1]].copy_(c['table']);c['table'] = table
            segments = base._pick_segments(batch, 2, 16)
            for field in ('scratch', 'maxima', 'sums'):c[field] = c[field].narrow(2, 0, segments)
            qp = torch.empty_like(packed, dtype=torch.int8)
            scales = torch.empty(packed.shape[:-1], device='cuda', dtype=torch.float32)
            own.quantize_packed_cache[(packed.numel()//128,)](packed, qp, scales, DIM=128, num_warps=1)
            quant = dict(c);quant['k'], quant['v'] = qp.unbind(1)
            quant['key_scale'], quant['value_scale'] = scales.unbind(1)
            class Producer:
                def __getitem__(self, grid):
                    def run(**kwargs):
                        return own._gqa_decode_attention_int8[grid](
                            **kwargs, key_scale_ptr=quant['key_scale'], value_scale_ptr=quant['value_scale'],
                            scale_stride_block=quant['key_scale'].stride(0), scale_stride_slot=quant['key_scale'].stride(1))
                    return run
            candidate = SimpleNamespace(LOG2E=base.LOG2E, _gqa_decode_attention=Producer())
            host_lengths = c['lengths'].cpu().tolist()
            slots = torch.tensor([int(table[i, (n-1)//16].cpu())*16 + (n-1)%16 if n else -1
                                  for i, n in enumerate(host_lengths)], device='cuda', dtype=torch.int64)
            new_key = torch.randn(batch, 2, 128, device='cuda', dtype=torch.bfloat16)
            new_value = torch.randn_like(new_key)
            assert new_key.stride() == new_value.stride()
            assert c['k'].stride() == c['v'].stride() == quant['k'].stride() == quant['v'].stride()

            def append(case, encode):
                return own.append_token_kv[(batch, 2)](
                    new_key, new_value, slots, case['k'], case['v'], quant['key_scale'], quant['value_scale'],
                    source_stride_token=new_key.stride(0), source_stride_head=new_key.stride(1),
                    cache_stride_block=case['k'].stride(0), cache_stride_slot=case['k'].stride(1),
                    cache_stride_head=case['k'].stride(2), scale_stride_block=quant['key_scale'].stride(0),
                    scale_stride_slot=quant['key_scale'].stride(1), TOTAL_SLOTS=packed.shape[0]*16,
                    DIM=128, QUANTIZE=encode, num_warps=1)

            def run(encode, writer=True):
                case, module = (quant, candidate) if encode else (c, base)
                if writer:append(case, encode)
                compiled = bench.main_call(module, case, segments)
                bench.reduce_call(base, case, segments, 4)
                return compiled

            row = dict(name=name, batch=batch, length=length, segments=segments,
                       key_stride=list(key.stride()), table_stride=list(table.stride()),
                       original_cache_bytes=packed.numel()*packed.element_size(),
                       quantized_cache_bytes=qp.numel()+scales.numel()*scales.element_size(),
                       dual_cache_retained_in_probe=True, numeric=[], rounds=[], completed=False)
            result['cases'].append(row);save()
            graphs = {}
            # Explicit non-default capture stream, warmed before capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            if mixed:
                # Padding and invalid slots must leave both bytes and scales
                # intact. Keep this full-copy fault check on the small case.
                saved_slots = slots.clone()
                saved_bytes, saved_scales = qp.clone(), scales.clone()
                slots.fill_(-1)
                append(quant, True)
                slots.fill_(packed.shape[0]*16)
                append(quant, True)
                torch.cuda.synchronize()
                assert torch.equal(qp, saved_bytes) and torch.equal(scales, saved_scales)
                slots.copy_(saved_slots)
                stream.wait_stream(torch.cuda.current_stream())
                row['invalid_slots_no_write'] = True
            for encode in (False, True):
                with torch.cuda.stream(stream):
                    compiled = run(encode)
                stream.synchronize()
                if not encode and name == 'steady4k':assert str(compiled.hash) == SERVICE_4K_HASH
                row['int8_resources' if encode else 'baseline_resources'] = resources(compiled)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):run(encode)
                graphs[encode] = graph
            for pattern in range(2):
                if pattern:
                    c['q'].copy_(torch.randn_like(c['q']))
                    new_key.copy_(torch.randn_like(new_key) * .5)
                    new_value.copy_(torch.randn_like(new_value) * 2)
                stream.wait_stream(torch.cuda.current_stream())
                originals, decoded = None, None
                for encode in (False, True):
                    with torch.cuda.stream(stream):run(encode)
                    stream.synchronize();eager = c['out'].cpu().clone()
                    c['out'].fill_(float('nan'));torch.cuda.synchronize()
                    graphs[encode].replay();torch.cuda.synchronize()
                    replay = c['out'].cpu().clone()
                    assert torch.isfinite(replay).all() and torch.equal(eager, replay), 'Graph stale output'
                    indices, oracle = reference(quant if encode else c, quantized=encode)
                    check = compare(replay[indices], oracle)
                    assert check['finite'] and check['tol_005_violations'] == 0, (name, encode, check)
                    row['numeric'].append(dict(pattern=pattern, quantized=encode, cpu=check, changed_graph_equal=True))
                    if encode:decoded = replay
                    else:originals = replay
                row['numeric'].append(dict(pattern=pattern, quantization_error=compare(decoded, originals),
                                           model_accuracy_claim=False))
                # Representation invariants on finite fixture values.
                assert torch.isfinite(scales).all() and bool((scales > 0).all())
                assert not bool((qp == -128).any())
                if mixed:
                    # Both routes have appended the same new source tokens.
                    decoded_cache = qp.float() * scales[..., None]
                    bound = .5001 * scales[..., None] + 1e-6 * packed.float().abs()
                    assert bool(((decoded_cache - packed.float()).abs() <= bound).all())
                    row['pointwise_quantization_bound'] = True
                save()
            # Include writer + producer + unchanged reducer; report producer /
            # reducer-only cost separately. Writer control is ours, not vLLM's.
            for index, order in enumerate(((False, True), (True, False), (False, True))):
                timings = {}
                for encode in order:
                    timings['int8' if encode else 'baseline'] = bench.graph_time(lambda e=encode: run(e))[0]
                row['rounds'].append(dict(index=index, order=list(order), milliseconds=timings,
                                          speedup=timings['baseline']/timings['int8']))
                save()
            row['speedup_median'] = statistics.median(v['speedup'] for v in row['rounds'])
            row['attention_only_ms'] = {str(encode): bench.graph_time(lambda e=encode: run(e, writer=False))[0]
                                        for encode in (False, True)}
            row['completed'] = True;save()
            print(json.dumps(dict(case=name, speedup=row['speedup_median'], official_score=False)), flush=True)
            del graphs
        result['completed'] = True
    except Exception as error:
        result['errors'].append(dict(error=repr(error), traceback=traceback.format_exc()));raise
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-gpu', action='store_true');p.add_argument('--out', type=Path)
    p.add_argument('--case', action='append', choices=sorted(CASES))
    a = p.parse_args()
    if a.run_gpu and a.out is None:p.error('--out is required for GPU work')
    main(a)
