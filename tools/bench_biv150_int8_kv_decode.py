"""Diagnostic only: INT8/FP8 KV storage with dequantization fused into GQA decode.

No serving installation. Offline quantization excluded from attention timing;
this is an upper-bound feasibility probe, not an end-to-end throughput result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import torch
from biv150_kv_append_probe import append_quantized
from biv150_benchmark_utils import load_module, inputs, full_call, graph_time


def quantize(tensor, dtype):
    limit = 127 if dtype == torch.int8 else torch.finfo(dtype).max
    scale = tensor.float().abs().amax(-1).clamp_min(1e-8) / limit
    normalized = (tensor.float() / scale[..., None]).clamp(-limit, limit)
    quantized = (normalized.round() if dtype == torch.int8 else normalized).to(dtype)
    return quantized, scale.contiguous()


class ScaledLaunch:
    def __init__(self, kernel, ks, vs):
        self.kernel, self.ks, self.vs = kernel, ks, vs
        self.last = None
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.last = self.kernel[grid](*args, k_scale_ptr=self.ks, v_scale_ptr=self.vs, **kwargs)
            return self.last
        return launch


def dense_reference(c, keys, values, row):
    n = int(c['lengths'][row].item())
    if not n:
        return torch.zeros_like(c['out'][row]).float()
    pages = c['table'][row, : (n + 15) // 16].long()
    k = keys[pages].reshape(-1, 2, 128)[:n].float()
    v = values[pages].reshape(-1, 2, 128)[:n].float()
    q = c['q'][row].float().reshape(2, 8, 128)
    p = torch.einsum('hgd,thd->hgt', q, k).mul(128 ** -.5).softmax(-1)
    return torch.einsum('hgt,thd->hgd', p, v).reshape(16, 128)


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(('127.0.0.1', 9031)) != 0, 'Service running; no overlapping GPU probe'
    torch.backends.cuda.matmul.allow_tf32 = False
    original = args.source.read_text()
    text = original.replace('def _gqa_decode_attention(', 'def _gqa_decode_attention_int8_kv(')
    needle = '    queries_per_kv: tl.constexpr,\n'
    assert text.count(needle) == 1
    text = text.replace(needle, '    k_scale_ptr,\n    v_scale_ptr,\n' + needle)
    for tail in (False, True):
        if tail:
            before = '        K = tl.load(key_cache_ptr + k_offsets, mask=column_valid[None, :], other=0.0)\n        V = tl.load(value_cache_ptr + v_offsets, mask=column_valid[:, None], other=0.0)'
            scales = ('        ks = tl.load(k_scale_ptr + scale_offsets, mask=column_valid, other=0.0)\n'
                      '        vs = tl.load(v_scale_ptr + scale_offsets, mask=column_valid, other=0.0)\n')
        else:
            before = '        K = tl.load(key_cache_ptr + k_offsets)\n        V = tl.load(value_cache_ptr + v_offsets)'
            scales = ('        ks = tl.load(k_scale_ptr + scale_offsets)\n'
                      '        vs = tl.load(v_scale_ptr + scale_offsets)\n')
        assert text.count(before) == 1
        after = before + ('\n        scale_offsets = page * PAGE_SIZE * 2 + column_slot * 2 + kv_head_idx\n' + scales +
                         '        K = (K.to(tl.float32) * ks[None, :]).to(tl.bfloat16)\n'
                         '        V = (V.to(tl.float32) * vs[:, None]).to(tl.bfloat16)')
        text = text.replace(before, after)
    assert not (args.inline_cache and args.page_packed_cache)
    if args.page_packed_cache:
        text = text.replace('scale_offsets = page * PAGE_SIZE * 2 + column_slot * 2 + kv_head_idx',
                            'scale_offsets = page * (k_stride_block // 4) + column_slot * 2 + kv_head_idx')
    if args.inline_cache:
        assert text.count('scale_offsets = page * PAGE_SIZE * 2 + column_slot * 2 + kv_head_idx') == 2
        text = text.replace('scale_offsets = page * PAGE_SIZE * 2 + column_slot * 2 + kv_head_idx',
                            'scale_offsets = page * (k_stride_block // 4) + column_slot * (k_stride_slot // 4) + kv_head_idx * (k_stride_head // 4)')
    if args.scale_placement == 'scores':
        text = text.replace('K = (K.to(tl.float32) * ks[None, :]).to(tl.bfloat16)',
                            'K = K.to(tl.bfloat16)')
        text = text.replace('V = (V.to(tl.float32) * vs[:, None]).to(tl.bfloat16)',
                            'V = V.to(tl.bfloat16)')
        assert text.count('scores = tl.dot(Q, K) * softmax_scale_log2e') == 2
        text = text.replace('scores = tl.dot(Q, K) * softmax_scale_log2e',
                            'scores = tl.dot(Q, K) * softmax_scale_log2e * ks[None, :]')
        assert text.count('probability.to(V.dtype), V') == 2
        text = text.replace('probability.to(V.dtype), V',
                            '(probability * vs[None, :]).to(V.dtype), V')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generated = args.out.parent / ('quant-kv-generated-' + args.cache_dtype + '-' + args.scale_placement + ('-inline' if args.inline_cache else '-pagepacked' if args.page_packed_cache else '-compact') + '.py')
    generated.write_text(text)
    baseline = load_module(args.source, 'biv150_int8_kv_reference')
    candidate = load_module(generated, 'biv150_int8_kv_candidate')
    records = []
    for batch, kv, mixed in [(64,4096,False), (32,16384,False), (31,16896,False),
                             (8,1024,True), (1,100,False)]:
        c = inputs(batch, kv, mixed=mixed)
        segments = baseline._pick_segments(batch, 2, 16)
        update_baseline = lambda: None
        if args.append_updates:
            from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
            torch.manual_seed(943002+batch+kv)
            new_key=torch.randn(batch,2,128,device='cuda',dtype=torch.bfloat16)
            new_value=torch.randn_like(new_key)
            last=(c['lengths'].long()-1).clamp_min(0)
            slots=c['table'].gather(1,(last//16)[:,None]).squeeze(1).long()*16+last%16
            slots=torch.where(c['lengths']>0,slots,-1)
            one=torch.ones((),device='cuda',dtype=torch.float32)
            update_baseline=lambda:triton_reshape_and_cache_flash(new_key,new_value,c['k'],c['v'],slots,'auto',one,one)
        def baseline_chain():
            update_baseline()
            full_call(baseline,c,segments)
        baseline_chain()
        bf16_result = c['out'].clone()
        bf16_ms = graph_time(baseline_chain,repeats=11,unroll=4)[0]
        dtype = torch.int8 if args.cache_dtype == 'int8' else torch.float8_e4m3fn
        k8, ks = quantize(c['k'],dtype); v8, vs = quantize(c['v'],dtype)
        if args.inline_cache:
            # Reuse the original BF16 allocation geometry. Bytes0:128 hold payload,
            # bytes128:132 hold FP32 scale. No capacity/memory saving is claimed.
            raw = torch.empty((k8.shape[0],2,16,2,256),device='cuda',dtype=torch.uint8)
            kk = raw[:,0,:,:,:128].view(dtype)
            vv = raw[:,1,:,:,:128].view(dtype)
            sk = raw[:,0,:,:,128:132].view(torch.float32).squeeze(-1)
            sv = raw[:,1,:,:,128:132].view(torch.float32).squeeze(-1)
            kk.copy_(k8); vv.copy_(v8); sk.copy_(ks); sv.copy_(vs)
            k8,v8,ks,vs = kk,vv,sk,sv
        if args.page_packed_cache:
            # BF16 block allocation retained, but payload is compact within each
            # page and scales follow the payload contiguously, avoiding head holes.
            raw=torch.empty((k8.shape[0],2,8192),device='cuda',dtype=torch.uint8)
            kk=raw[:,0,:4096].view(dtype).reshape(-1,16,2,128)
            vv=raw[:,1,:4096].view(dtype).reshape(-1,16,2,128)
            sk=raw[:,0,4096:4224].view(torch.float32).reshape(-1,16,2)
            sv=raw[:,1,4096:4224].view(torch.float32).reshape(-1,16,2)
            kk.copy_(k8);vv.copy_(v8);sk.copy_(ks);sv.copy_(vs)
            k8,v8,ks,vs=kk,vv,sk,sv
        launch = ScaledLaunch(candidate._gqa_decode_attention_int8_kv, ks, vs)
        candidate._gqa_decode_attention = launch
        qc = dict(c, k=k8, v=v8)
        def candidate_chain():
            if args.append_updates:
                append_quantized(new_key,new_value,slots,k8,v8,ks,vs)
            full_call(candidate,qc,segments)
        row = dict(append_updates=args.append_updates,batch=batch,kv=kv,mixed=mixed,segments=segments,bf16_ms=bf16_ms,cache_dtype=args.cache_dtype,
                   bf16_cache_bytes=c['k'].nbytes+c['v'].nbytes,
                   quantized_cache_bytes=k8.nbytes+v8.nbytes+ks.nbytes+vs.nbytes,
                   inline_cache=args.inline_cache,page_packed_cache=args.page_packed_cache,
                   allocated_cache_bytes=(raw.untyped_storage().nbytes() if (args.inline_cache or args.page_packed_cache) else
                       k8.nbytes+v8.nbytes+ks.nbytes+vs.nbytes))
        try:
            candidate_chain()
            int8_result = qc['out'].clone()
            row['quantized_ms'] = graph_time(candidate_chain,repeats=11,unroll=4)[0]
            # Recheck BF16 timing after candidate for drift visibility.
            row['bf16_after_ms'] = graph_time(baseline_chain,repeats=11,unroll=4)[0]
            row['speedup'] = bf16_ms / row['quantized_ms']
            row['max_output_diff_from_bf16'] = (int8_result.float()-bf16_result.float()).abs().max().item()
            row['rmse_from_bf16'] = (int8_result.float()-bf16_result.float()).square().mean().sqrt().item()
            kd = (k8.float()*ks[...,None]).to(torch.bfloat16)
            vd = (v8.float()*vs[...,None]).to(torch.bfloat16)
            row['dense_checks'] = []
            for i in sorted({0,min(1,batch-1),batch-1}):
                dense = dense_reference(c,kd,vd,i).to(torch.bfloat16).float()
                torch.testing.assert_close(int8_result[i].float(),dense,atol=.005,rtol=.005)
                original_dense = dense_reference(c,c['k'],c['v'],i)
                row['dense_checks'].append(dict(row=i,dequantized_dense_pass=True,
                    max_quantization_error=(int8_result[i].float()-original_dense).abs().max().item()))
            row.update(n_regs=getattr(launch.last,'n_regs',None),
                       shared=getattr(getattr(launch.last,'metadata',None),'shared',None))
        except Exception as error:
            row['error'] = str(error)[:1500]
        records.append(row)
        print(json.dumps(row),flush=True)
        args.out.write_text(json.dumps(dict(probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            append_source_sha256=hashlib.sha256((Path(__file__).parent/'biv150_kv_append_probe.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            generated_sha256=hashlib.sha256(text.encode()).hexdigest(),offline_quantization_timed=False,
            service_installed=False,scale_placement=args.scale_placement,cache_dtype=args.cache_dtype,inline_cache=args.inline_cache,page_packed_cache=args.page_packed_cache,records=records),indent=2))
        del c,qc,k8,v8,ks,vs,bf16_result
        torch.cuda.empty_cache()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--page-packed-cache',action='store_true')
    p.add_argument('--append-updates',action='store_true')
    p.add_argument('--inline-cache',action='store_true')
    p.add_argument('--cache-dtype',choices=['int8','fp8'],default='int8')
    p.add_argument('--scale-placement',choices=['vectors','scores'],default='vectors')
    main(p.parse_args())
