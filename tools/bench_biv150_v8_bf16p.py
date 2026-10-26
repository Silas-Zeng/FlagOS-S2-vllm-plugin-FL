"""Diagnostic BF16 QK + V8 storage + BF16 weighted P/PV; no serving registration."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
from types import SimpleNamespace
import traceback

from biv150_v8_bf16p_validation import consumption_gate, mapped_byte_offsets, validate_codec_vector

CASES = {'steady4k': (64, 4096, False), 'steady16k': (32, 16384, False),
         'ragged': (8, 257, True), 'ragged-long': (8, 257, True)}
SOURCE_SHA = '2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
SERVICE_HASHES = {'steady4k': '641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff',
                  'steady16k': 'e29a0c84d6b99f1ad44189fbdb74434b2d7da5675af2e2e2422941739916adb9'}


def main(args):
    chosen = args.case or list(CASES)
    if not args.run_gpu:
        print(json.dumps(dict(cases={k: CASES[k] for k in chosen}, service_installed=False,
                              theoretical_payload_fraction=12416/16384, physical_capacity_reclaimed=False,
                              allocation_bytes_per_page=16384, payload_bytes_per_page=12416,
                              hypothetical_aligned_compact_bytes_per_page=12544, precision_gate=False,
                              component_consumption_relative_rmse_limit=0.02,
                              oracle_from_frozen_cpu_bf16=True, graph_writer_poison_check=True,
                              model_accuracy='not tested', throughput='not tested')))
        return
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0, 'Service active'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True), 'GPU busy'
    root = Path(__file__).resolve().parents[1]
    assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip(), 'dirty source'
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import biv150_benchmark_utils as bench
    torch.set_num_threads(4)
    source = root / 'vllm_fl/patches/biv150_gqa_decode.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    base = bench.load_module(source, 'v8_bf16p_shipping_gqa')
    kernel_path = root / 'tools/kernels/biv150_v8_bf16p.py'
    own = bench.load_module(kernel_path, 'v8_bf16p_experimental_kernels')
    assert base.TARGET_PROGRAMS == 256
    result = dict(completed=False, service_installed=False, official_score=False,
                  model_accuracy_tested=False, precision_gate=False, capacity_reclaimed=False,
                  probability_dtype='bfloat16', pv_dtype='bfloat16', candidate_warps=args.warps,
                  component_consumption_relative_rmse_limit=0.02,
                  oracle_from_frozen_cpu_bf16=True, graph_writer_poison_check=True,
                  source_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                  operator_sha256=hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
                  baseline_source_sha256=SOURCE_SHA, cases=[], errors=[])

    def save():
        path = args.out / 'result.tmp'
        path.write_text(json.dumps(result, indent=2) + '\n')
        path.replace(args.out / 'result.json')

    def compare(a, b):
        a, b = a.double(), b.double()
        error = (a-b).abs()
        return dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                    max_abs=float(error.max()), mse=float((error*error).mean()),
                    tol_005_violations=int((error > .005 + .005*b.abs()).sum()))

    def cpu_encode(values):
        # CPU-only, independent FP32 maxabs/127 (not the GPU scale/codec).
        assert values.device.type == 'cpu' and values.dtype == torch.bfloat16
        wide = values.float()
        maximum = wide.abs().amax(-1)
        # Correctly round division to FP32 independently of a backend's
        # constant-reciprocal lowering, then apply the specified normal clamp.
        scale = torch.where(maximum == 0, 1., (maximum.double()/127).float().clamp_min(2**-126))
        ratio = wide.double()/scale.double()[..., None]
        codes = (ratio.sign()*torch.floor(ratio.abs()+.5)).clamp(-127, 127).to(torch.int8)
        return codes, scale

    def reference(cpu, *, quantized=False):
        # Only frozen CPU BF16 + CPU new-token overlays enter this FP64 oracle.
        # No writer output, device V scale, recurrence or P8 path is reused.
        answers = []
        selected = list(range(cpu['q'].shape[0])) if cpu['q'].shape[0] <= 8 else [0, 1, cpu['q'].shape[0]-1]
        for seq in selected:
            length = cpu['lengths'][seq]
            if not length:
                answers.append(torch.zeros(16, 128, dtype=torch.float64))
                continue
            ids = cpu['table'][seq, :(length+15)//16].long()
            key = cpu['k'][ids].reshape(-1, 2, 128)[:length].double()
            original_value = cpu['v'][ids].reshape(-1, 2, 128)[:length]
            q = cpu['q'][seq].double().reshape(2, 8, 128)
            weights = torch.einsum('ghd,tgd->ght', q, key).mul(128**-.5).softmax(-1)
            if quantized:
                codes, scales = cpu_encode(original_value)
                value = codes.double()*scales.double()[:, :, None]
            else:
                value = original_value.double()
            answers.append(torch.einsum('ght,tgd->ghd', weights, value).reshape(16, 128))
        return selected, torch.stack(answers)

    def consume_rows(actual, original):
        checks = [dict(row=index, **consumption_gate(a.reshape(-1).double().tolist(),
                                                    b.reshape(-1).double().tolist()))
                  for index, (a, b) in enumerate(zip(actual, original))]
        assert all(check['passed'] for check in checks), ('V8 consumption precision gate', checks)
        return checks

    def resources(compiled):
        destination = args.out/'compiler-ir'/str(compiled.hash)
        destination.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name, value in compiled.asm.items():
            if name not in ('ttgir', 'llir', 'asm', 'cubin'):
                continue
            raw = value.encode() if isinstance(value, str) else value
            (destination/('actual.'+name)).write_bytes(raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        return dict(hash=str(compiled.hash), n_regs=compiled.n_regs,
                    n_spills=compiled.n_spills, shared=compiled.metadata.shared,
                    num_warps=compiled.metadata.num_warps, num_stages=compiled.metadata.num_stages,
                    use_sme=getattr(compiled.metadata, 'use_sme', None), asm_sha256=hashes)

    save()
    try:
        for name in chosen:
            batch, length, mixed = CASES[name]
            c = bench.inputs(batch, length, mixed)
            if mixed:
                lengths = ([0, 1, 15, 16, 17, 31, 32, 33] if name == 'ragged'
                           else [63, 64, 65, 127, 128, 129, 256, 257])
                c['lengths'].copy_(torch.tensor(lengths, device='cuda', dtype=torch.int32))
            packed = torch.empty((c['k'].shape[0], 2, 16, 2, 128), device='cuda', dtype=torch.bfloat16)
            key, value = packed.unbind(1)
            key.copy_(c['k']); value.copy_(c['v']); c.update(k=key, v=value)
            table = torch.full((batch, 8192), -1, device='cuda', dtype=torch.int32)
            table[:, :c['table'].shape[1]].copy_(c['table']); c['table'] = table
            # Preserve immutable original BF16 inputs before any codec/writer.
            frozen_key, frozen_value = key.cpu().clone(), value.cpu().clone()
            host_table = table.cpu().clone()
            host_lengths = tuple(c['lengths'].cpu().tolist())
            host_slots = tuple(int(host_table[i, (n-1)//16])*16+(n-1)%16 if n else -1
                               for i, n in enumerate(host_lengths))
            valid_tokens = [i for i, slot in enumerate(host_slots) if 0 <= slot < packed.shape[0]*16]
            target_pages = [host_slots[i]//16 for i in valid_tokens]
            target_columns = [host_slots[i]%16 for i in valid_tokens]
            check_pages = (list(range(packed.shape[0])) if mixed else
                           sorted(set([0, packed.shape[0]//2, packed.shape[0]-1, *target_pages])))
            expected_cpu = dict(k=frozen_key.clone(), v=frozen_value.clone(),
                                table=host_table, lengths=host_lengths)
            fixture_path = args.out/(name+'-cpu-inputs.pt')
            fixture = dict(original_bf16_key=frozen_key[check_pages],
                           original_bf16_value=frozen_value[check_pages], page_ids=check_pages,
                           cache_coverage='full' if mixed else 'sampled_pages',
                           slot_mapping=host_slots, table=host_table, lengths=host_lengths, patterns=[])
            torch.save(fixture, fixture_path)
            slots = torch.tensor(host_slots, device='cuda', dtype=torch.int64)
            new_key = torch.randn(batch, 2, 128, device='cuda', dtype=torch.bfloat16)
            new_value = torch.randn_like(new_key)
            segments = base._pick_segments(batch, 2, 16)
            for field in ('scratch', 'maxima', 'sums'):
                c[field] = c[field].narrow(2, 0, segments)
            encoded_storage = packed.clone()
            raw_bytes = encoded_storage.view(torch.int8).reshape(-1)
            encoded_v = torch.as_strided(raw_bytes, (packed.shape[0], 16, 2, 128), (16384, 256, 128, 1), 8192)
            padding = torch.as_strided(raw_bytes, (packed.shape[0], 3968), (16384, 1), 12416)
            padding_before = padding.clone()
            scales = torch.as_strided(encoded_storage.view(torch.float32).reshape(-1),
                                      (packed.shape[0], 16, 2), (4096, 2, 1), 3072)
            scales.fill_(float('nan'))
            quant = dict(c, k=encoded_storage[:, 0], v=encoded_v, vscale=scales)
            assert c['q'].dtype == quant['k'].dtype == torch.bfloat16
            assert encoded_v.dtype == torch.int8 and scales.dtype == torch.float32
            assert scales.stride() == (4096, 2, 1)
            assert new_key.stride() == new_value.stride()
            assert c['k'].stride() == c['v'].stride() == quant['k'].stride()
            assert quant['v'].stride() == (16384, 256, 128, 1)
            own.encode_cache_v[(packed.shape[0]*32,)](value, encoded_v, scales,
                src_block=value.stride(0), src_slot=value.stride(1), src_head=value.stride(2),
                dst_block=encoded_v.stride(0), dst_slot=encoded_v.stride(1), dst_head=encoded_v.stride(2),
                SCALE_BLOCK=scales.stride(0), DIM=128, num_warps=1)

            class Producer:
                def __getitem__(self, grid):
                    def launch(**kwargs):
                        kwargs['num_warps'] = args.warps
                        return own.gqa_decode_v8_bf16p[grid](**kwargs, value_scale_ptr=scales,
                            scale_stride_block=scales.stride(0), scale_stride_slot=scales.stride(1),
                            scale_stride_head=scales.stride(2))
                    return launch
            candidate = SimpleNamespace(LOG2E=base.LOG2E, _gqa_decode_attention=Producer())

            def append(case, encode):
                return own.append_token_kv[(batch, 2)](new_key, new_value, slots, case['k'], case['v'], scales,
                    source_token=new_key.stride(0), source_head=new_key.stride(1),
                    key_block=case['k'].stride(0), key_slot=case['k'].stride(1), key_head=case['k'].stride(2),
                    value_block=case['v'].stride(0), value_slot=case['v'].stride(1), value_head=case['v'].stride(2),
                    SCALE_BLOCK=scales.stride(0), TOTAL_SLOTS=packed.shape[0]*16, DIM=128, ENCODE=encode, num_warps=1)

            def run(encode, writer=True):
                case, module = (quant, candidate) if encode else (c, base)
                if writer:
                    append(case, encode)
                compiled = bench.main_call(module, case, segments)
                bench.reduce_call(base, case, segments, 4)
                return compiled

            def codec_check(original, actual_codes, actual_scales):
                originals = original.reshape(-1, 128).tolist()
                codes = actual_codes.cpu().reshape(-1, 128).tolist()
                actual = actual_scales.cpu().reshape(-1).tolist()
                checked = [validate_codec_vector(v, code, scale) for v, code, scale in zip(originals, codes, actual)]
                assert len(checked) == len(originals) == len(actual)
                return dict(vectors=len(checked), max_scale_ulps=max(x['scale_ulps'] for x in checked),
                            half_boundary_code_differences=sum(x['half_boundary_code_differences'] for x in checked),
                            max_half_tie_distance_fp32_ulps=max(x['max_half_tie_distance_fp32_ulps'] for x in checked),
                            half_boundary_examples=[example for x in checked for example in x['half_boundary_examples']][:8])

            def codec_pages():
                return codec_check(expected_cpu['v'][check_pages], encoded_v[check_pages], scales[check_pages])

            cache_bytes = {False: packed.view(torch.int8).reshape(-1), True: raw_bytes}
            allowed_bytes = {encode: torch.tensor(mapped_byte_offsets(host_slots, packed.shape[0], encode),
                                                  device='cuda', dtype=torch.int64) for encode in (False, True)}

            def check_writer(encode, before, cpu_key, cpu_value):
                case = quant if encode else c
                assert torch.equal(case['k'][target_pages, target_columns].cpu(), cpu_key[valid_tokens]), 'Missing/stale K writer'
                if encode:
                    detail = codec_check(cpu_value[valid_tokens], encoded_v[target_pages, target_columns],
                                         scales[target_pages, target_columns])
                else:
                    assert torch.equal(case['v'][target_pages, target_columns].cpu(), cpu_value[valid_tokens]), 'Missing/stale BF16 V writer'
                    detail = dict(vectors=len(valid_tokens)*2)
                # Exempt only mapped K/V/(scale) bytes, then compare the complete
                # allocation: all other live slots, K, scales and padding matter.
                current = cache_bytes[encode]
                allowed = allowed_bytes[encode]
                before.index_copy_(0, allowed, current.index_select(0, allowed))
                assert torch.equal(before, current), 'Writer changed unmapped cache bytes'
                return dict(**detail, target_writer_checked=True, unmapped_bytes_unchanged=True,
                            unmapped_bytes_checked=current.numel()-allowed.numel())

            row = dict(name=name, batch=batch, length=length, segments=segments,
                       key_stride=list(key.stride()), baseline_v_stride=list(value.stride()),
                       candidate_v_stride=list(encoded_v.stride()), table_stride=list(table.stride()),
                       original_cache_bytes=packed.numel()*packed.element_size(),
                       candidate_allocated_bytes=encoded_storage.numel()*encoded_storage.element_size(),
                       candidate_payload_bytes=quant['k'].numel()*2+quant['v'].numel()+scales.numel()*4,
                       probability_dtype='bfloat16', candidate_warps=args.warps,
                       dual_cache_retained_in_probe=True, capacity_reclaimed=False, precision_gate=False,
                       v_scale_stride=list(scales.stride()), v_scale_byte_offset=12288,
                       cpu_fixture=fixture_path.name, codec_checked_page_ids=check_pages,
                       codec_coverage='full' if mixed else 'sampled_pages_and_all_writer_targets',
                       numeric=[], rounds=[], completed=False)
            result['cases'].append(row); save()
            row['initial_codec'] = codec_pages()
            assert torch.equal(quant['k'], key) and torch.equal(padding, padding_before)
            graphs = {}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            if mixed:
                saved_slots = slots.clone()
                for encode in (False, True):
                    before = cache_bytes[encode].clone()
                    slots.fill_(-1); append(quant if encode else c, encode)
                    slots.fill_(packed.shape[0]*16); append(quant if encode else c, encode)
                    torch.cuda.synchronize()
                    assert torch.equal(cache_bytes[encode], before), 'Invalid slot wrote cache'
                slots.copy_(saved_slots)
                stream.wait_stream(torch.cuda.current_stream())
                row['invalid_slots_no_write'] = True
            for encode in (False, True):
                with torch.cuda.stream(stream):
                    compiled = run(encode)
                stream.synchronize()
                if not encode and name in SERVICE_HASHES:
                    assert str(compiled.hash) == SERVICE_HASHES[name]
                row['v8_bf16p_resources' if encode else 'baseline_resources'] = resources(compiled)
                assert compiled.metadata.num_warps == (args.warps if encode else 1)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    run(encode)
                graphs[encode] = graph
            for pattern in range(2):
                # Always change both inputs AFTER capture/warmup, including pattern0.
                c['q'].copy_(torch.randn_like(c['q']))
                new_key.copy_(torch.randn_like(new_key)*(pattern+.5))
                new_value.copy_(torch.randn_like(new_value)*(pattern+1))
                cpu_key, cpu_value = new_key.cpu().clone(), new_value.cpu().clone()
                expected_cpu['q'] = c['q'].cpu().clone()
                expected_cpu['k'][target_pages, target_columns] = cpu_key[valid_tokens]
                expected_cpu['v'][target_pages, target_columns] = cpu_value[valid_tokens]
                fixture['patterns'].append(dict(query=expected_cpu['q'], new_key=cpu_key, new_value=cpu_value))
                torch.save(fixture, fixture_path)
                indices, bf16_oracle = reference(expected_cpu)
                originals, decoded = None, None
                for encode in (False, True):
                    # Poison mapped K/V/scale bytes before REPLAY. No eager call
                    # can prepopulate the new token and hide a missing graph writer.
                    before = cache_bytes[encode].clone()
                    cache_bytes[encode].index_fill_(0, allowed_bytes[encode], -91)
                    for field in ('out', 'scratch', 'maxima', 'sums'):
                        c[field].fill_(float('nan'))
                    torch.cuda.synchronize()
                    graphs[encode].replay(); torch.cuda.synchronize()
                    replay_writer = check_writer(encode, before, cpu_key, cpu_value)
                    replay = c['out'].cpu().clone()
                    assert torch.isfinite(replay).all(), 'Nonfinite graph output'
                    # Only now run eager, and independently check its writes too.
                    before = cache_bytes[encode].clone()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run(encode)
                    stream.synchronize()
                    eager_writer = check_writer(encode, before, cpu_key, cpu_value)
                    eager = c['out'].cpu().clone()
                    assert torch.equal(eager, replay), 'Changed-input graph/eager mismatch'
                    _, oracle = reference(expected_cpu, quantized=encode)
                    check = compare(replay[indices], oracle)
                    assert check['finite'] and check['tol_005_violations'] == 0, (name, encode, check)
                    record = dict(pattern=pattern, quantized=encode, cpu=check, changed_graph_equal=True,
                                  graph_writer=replay_writer, eager_writer=eager_writer)
                    if encode:
                        decoded = replay
                        record['independent_bf16_consumption_gate'] = consume_rows(replay[indices], bf16_oracle)
                    else:
                        originals = replay
                    row['numeric'].append(record); save()
                # A hard per-sequence 2% relative-RMS gate covers EVERY output,
                # while the independent FP64 BF16 gate covers all ragged rows
                # and sampled normal-range steady rows. This is not model105.
                row['numeric'].append(dict(pattern=pattern,
                    all_rows_bf16_consumption_gate=consume_rows(decoded, originals),
                    quantization_error=compare(decoded, originals), model_accuracy_claim=False))
                row['last_codec'] = codec_pages()
                assert torch.equal(c['k'], quant['k']), 'K differs between candidates'
                assert torch.equal(padding, padding_before), 'V encoding touched allocation padding'
                save()
            for index, order in enumerate(((False, True), (True, False), (False, True))):
                timings = {}
                for encode in order:
                    timings['v8_bf16p' if encode else 'baseline'] = bench.graph_time(lambda e=encode: run(e))[0]
                row['rounds'].append(dict(index=index, order=list(order), milliseconds=timings,
                                          speedup=timings['baseline']/timings['v8_bf16p']))
                save()
            row['speedup_median'] = statistics.median(v['speedup'] for v in row['rounds'])
            row['attention_only_ms'] = {str(encode): bench.graph_time(lambda e=encode: run(e, writer=False))[0]
                                        for encode in (False, True)}
            row['component_checks_passed'] = True
            row['completed'] = True; save()
            print(json.dumps(dict(case=name, speedup=row['speedup_median'], component_checks_passed=True,
                                  precision_gate=False, official_score=False)), flush=True)
            del graphs, c, quant, packed, key, value, encoded_storage, raw_bytes, encoded_v, scales
            del frozen_key, frozen_value, expected_cpu, fixture, cache_bytes, padding, padding_before
            del before, allowed_bytes, slots, new_key, new_value, candidate, stream
            torch.cuda.empty_cache()
        result['completed'] = True
    except Exception as error:
        result['errors'].append(dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-gpu', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--case', action='append', choices=sorted(CASES))
    p.add_argument('--warps', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    if a.run_gpu and a.out is None:
        p.error('--out is required for GPU work')
    main(a)
