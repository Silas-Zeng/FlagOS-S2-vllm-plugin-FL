"""Diagnostic BF16 K + software FP8 V writer/consumer; never registers a serving implementation."""
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
                              theoretical_payload_fraction=0.75, physical_capacity_reclaimed=False,
                              model_accuracy='not tested', throughput='not tested')))
        return
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0, 'Service active'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True), 'GPU busy'
    root=Path(__file__).resolve().parents[1]
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip(),'dirty source'
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import biv150_benchmark_utils as bench
    from biv150_fp8_v_reference import decode_e4m3,encode_nearest_reference
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    source = root / 'vllm_fl/patches/biv150_gqa_decode.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    base = bench.load_module(source, 'fp8_v_shipping_gqa')
    kernel_path = root / 'tools/kernels/biv150_fp8_v_attention.py'
    own = bench.load_module(kernel_path, 'fp8_v_experimental_kernels')
    assert base.TARGET_PROGRAMS == 256
    result = dict(completed=False, service_installed=False, official_score=False,
                  model_accuracy_tested=False, capacity_reclaimed=False, decoder=args.decoder,
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
        # exactly the intended BF16 K and reconstructed V operands, separating implementation bugs
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
                lookup = torch.tensor([decode_e4m3(code) for code in range(256)],dtype=torch.bfloat16)
                value = lookup[value.long()]
            q = case['q'][seq].cpu().double().reshape(2, 8, 128)
            scores = torch.einsum('ghd,tgd->ght', q, key.double()) * 128**-.5
            answer = torch.einsum('ght,tgd->ghd', torch.softmax(scores, -1), value.double())
            answers.append(answer.reshape(16, 128))
        return selected, torch.stack(answers)

    def resources(compiled):
        return dict(hash=str(compiled.hash), n_regs=compiled.n_regs,
                    n_spills=compiled.n_spills, shared=compiled.metadata.shared,
                    num_warps=compiled.metadata.num_warps, num_stages=compiled.metadata.num_stages,
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
            encoded_storage=packed.clone()
            raw_bytes=encoded_storage.view(torch.uint8).reshape(-1)
            encoded_v=torch.as_strided(raw_bytes,(packed.shape[0],16,2,128),(16384,256,128,1),8192)
            padding=torch.as_strided(raw_bytes,(packed.shape[0],4096),(16384,1),12288)
            padding_before=padding.clone()
            quant=dict(c,k=encoded_storage[:,0],v=encoded_v)
            lookup=torch.tensor([decode_e4m3(code) for code in range(256)],dtype=torch.bfloat16,device='cuda')
            own.encode_cache_v[(packed.shape[0]*32,)](value,encoded_v,
                src_block=value.stride(0),src_slot=value.stride(1),src_head=value.stride(2),
                dst_block=encoded_v.stride(0),dst_slot=encoded_v.stride(1),dst_head=encoded_v.stride(2),DIM=128,num_warps=1)
            class Producer:
                def __getitem__(self,grid):
                    def run(**kwargs):
                        return own.gqa_decode_fp8_v[grid](**kwargs,value_lut_ptr=lookup,USE_LUT=args.decoder=='lut')
                    return run
            candidate=SimpleNamespace(LOG2E=base.LOG2E,_gqa_decode_attention=Producer())
            host_lengths = c['lengths'].cpu().tolist()
            slots = torch.tensor([int(table[i, (n-1)//16].cpu())*16 + (n-1)%16 if n else -1
                                  for i, n in enumerate(host_lengths)], device='cuda', dtype=torch.int64)
            new_key = torch.randn(batch, 2, 128, device='cuda', dtype=torch.bfloat16)
            new_value = torch.randn_like(new_key)
            assert new_key.stride() == new_value.stride()
            assert c['k'].stride()==c['v'].stride()==quant['k'].stride()
            assert quant['v'].stride()==(16384,256,128,1)

            def append(case, encode):
                return own.append_token_kv[(batch,2)](new_key,new_value,slots,case['k'],case['v'],
                    source_token=new_key.stride(0),source_head=new_key.stride(1),
                    key_block=case['k'].stride(0),key_slot=case['k'].stride(1),key_head=case['k'].stride(2),
                    value_block=case['v'].stride(0),value_slot=case['v'].stride(1),value_head=case['v'].stride(2),
                    TOTAL_SLOTS=packed.shape[0]*16,DIM=128,ENCODE=encode,num_warps=1)

            def run(encode, writer=True):
                case, module = (quant, candidate) if encode else (c, base)
                if writer:append(case, encode)
                compiled = bench.main_call(module, case, segments)
                bench.reduce_call(base, case, segments, 4)
                return compiled

            row = dict(name=name, batch=batch, length=length, segments=segments,
                       key_stride=list(key.stride()), baseline_v_stride=list(value.stride()), candidate_v_stride=list(encoded_v.stride()), table_stride=list(table.stride()),
                       original_cache_bytes=packed.numel()*packed.element_size(),
                       candidate_allocated_bytes=encoded_storage.numel()*encoded_storage.element_size(),
                       candidate_payload_bytes=quant['k'].numel()*2+quant['v'].numel(),decoder=args.decoder,
                       dual_cache_retained_in_probe=True, numeric=[], rounds=[], completed=False)
            result['cases'].append(row);save()
            graphs = {}
            # Explicit non-default capture stream, warmed before capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            if mixed:
                # Padding and invalid slots must leave both cache bytes
                # intact. Keep this full-copy fault check on the small case.
                saved_slots = slots.clone()
                saved_bytes=raw_bytes.clone()
                slots.fill_(-1)
                append(quant, True)
                slots.fill_(packed.shape[0]*16)
                append(quant, True)
                torch.cuda.synchronize()
                assert torch.equal(raw_bytes,saved_bytes)
                slots.copy_(saved_slots)
                stream.wait_stream(torch.cuda.current_stream())
                row['invalid_slots_no_write'] = True
            for encode in (False, True):
                with torch.cuda.stream(stream):
                    compiled = run(encode)
                stream.synchronize()
                if not encode and name == 'steady4k':assert str(compiled.hash) == SERVICE_4K_HASH
                row['fp8_v_resources' if encode else 'baseline_resources'] = resources(compiled)
                assert compiled.metadata.num_warps==1
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
                assert not bool(((encoded_v&127)==127).any()),'finite source encoded NaN'
                assert torch.equal(c['k'],quant['k']),'K changed'
                assert torch.equal(padding,padding_before),'V encoding touched allocation padding'
                if mixed:
                    bits=c['v'].cpu().contiguous().view(torch.uint16).reshape(-1).tolist()
                    expected=torch.tensor([encode_nearest_reference(v) for v in bits],dtype=torch.uint8).reshape_as(encoded_v.cpu())
                    assert torch.equal(encoded_v.cpu(),expected),'cache writer differs from CPU nearest-value oracle'
                    row['pointwise_code_exact']=True

                save()
            # Include writer + producer + unchanged reducer; report producer /
            # reducer-only cost separately. Writer control is ours, not vLLM's.
            for index, order in enumerate(((False, True), (True, False), (False, True))):
                timings = {}
                for encode in order:
                    timings['fp8_v' if encode else 'baseline'] = bench.graph_time(lambda e=encode: run(e))[0]
                row['rounds'].append(dict(index=index, order=list(order), milliseconds=timings,
                                          speedup=timings['baseline']/timings['fp8_v']))
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
    p.add_argument('--decoder',choices=('bits','lut'),default='bits')
    a = p.parse_args()
    if a.run_gpu and a.out is None:p.error('--out is required for GPU work')
    main(a)
