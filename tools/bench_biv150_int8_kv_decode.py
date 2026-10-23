"""Diagnostic only: INT8 KV storage with dequantization fused into GQA decode.

No serving installation. Offline quantization excluded from attention timing;
this is an upper-bound feasibility probe, not an end-to-end throughput result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import torch
from biv150_benchmark_utils import load_module, inputs, full_call, graph_time


def quantize(tensor):
    scale = tensor.float().abs().amax(-1).clamp_min(1e-8) / 127
    quantized = (tensor.float() / scale[..., None]).round().clamp(-127, 127).to(torch.int8)
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generated = args.out.parent / 'int8-kv-generated.py'
    generated.write_text(text)
    baseline = load_module(args.source, 'biv150_int8_kv_reference')
    candidate = load_module(generated, 'biv150_int8_kv_candidate')
    records = []
    for batch, kv, mixed in [(64,4096,False), (32,16384,False), (31,16896,False),
                             (8,1024,True), (1,100,False)]:
        c = inputs(batch, kv, mixed=mixed)
        segments = baseline._pick_segments(batch, 2, 16)
        full_call(baseline, c, segments)
        bf16_result = c['out'].clone()
        bf16_ms = graph_time(lambda: full_call(baseline,c,segments),repeats=11,unroll=4)[0]
        k8, ks = quantize(c['k']); v8, vs = quantize(c['v'])
        launch = ScaledLaunch(candidate._gqa_decode_attention_int8_kv, ks, vs)
        candidate._gqa_decode_attention = launch
        qc = dict(c, k=k8, v=v8)
        row = dict(batch=batch,kv=kv,mixed=mixed,segments=segments,bf16_ms=bf16_ms,
                   bf16_cache_bytes=c['k'].nbytes+c['v'].nbytes,
                   int8_cache_bytes=k8.nbytes+v8.nbytes+ks.nbytes+vs.nbytes)
        try:
            full_call(candidate, qc, segments)
            int8_result = qc['out'].clone()
            row['int8_ms'] = graph_time(lambda:full_call(candidate,qc,segments),repeats=11,unroll=4)[0]
            # Recheck BF16 timing after candidate for drift visibility.
            row['bf16_after_ms'] = graph_time(lambda:full_call(baseline,c,segments),repeats=11,unroll=4)[0]
            row['speedup'] = bf16_ms / row['int8_ms']
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
        args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            generated_sha256=hashlib.sha256(text.encode()).hexdigest(),offline_quantization_timed=False,
            service_installed=False,records=records),indent=2))
        del c,qc,k8,v8,ks,vs,bf16_result
        torch.cuda.empty_cache()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
