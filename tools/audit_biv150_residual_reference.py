"""Reproduce the initial RNG sequence and audit both kernels against CPU oracles.

Diagnostic only: no timings, model dispatch changes or relaxed tolerances.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

import torch
from bench_biv150_native_gate_residual import cpu_reference


def close_report(got, gold):
    got, gold = got.float(), gold.float()
    delta = (got - gold).abs()
    bad = delta > 0.005 + 0.005 * gold.abs()
    return dict(passed=not bool(bad.any()), violations=int(bad.sum()),
                max_abs=float(delta.max()), count=got.numel())


def cpu_quantize(x):
    """CPU FP32, native round-away and contracted residual subtraction.

    FP64 exactly represents each product of these FP32 operands before its
    final FP32 rounding, emulating the assembly's single-round FMA.
    """
    assert x.device.type == 'cpu'
    f = x.float()
    s1 = f.abs().amax(1).clamp_min(1e-6) / 127
    s2 = s1 / 254
    def away(v):
        return torch.where(v >= 0, (v + 0.5).floor(), (v - 0.5).ceil()).clamp(-127,127)
    first = away(f / s1[:, None])
    residual = (f.double() - first.double() * s1.double()[:, None]).float()
    second = away(residual / s2[:, None])
    return first.to(torch.int8), second.to(torch.int8), s1, s2


def cpu_output(first, second, s1, s2, weights, scales):
    # INT64 dot is exact; assembly rounds the residual product, then FMA.
    a = (first.long() @ weights.long()).float()
    b = (second.long() @ weights.long()).float()
    residual = b * s2[:,None]
    linear = (a.double() * s1.double()[:,None] + residual.double()).float() * scales
    gate, up = linear.to(torch.bfloat16).float().chunk(2,-1)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(('127.0.0.1',9031)) != 0
    assert 'No running processes found' in subprocess.check_output(
        ['/usr/local/corex/bin/ixsmi'],text=True)
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['BIV150_NATIVE_W8A8_LIBRARY'] = str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight
    load_library()
    torch.manual_seed(20261024)
    weight = torch.randn(12288,2048,device='cuda',dtype=torch.bfloat16)*0.02
    q, scales = quantize_weight(weight)
    cols = torch.tensor([0,1,31,127,1023,6143])
    both = torch.cat((cols,cols+6144)).to('cuda')
    weights_cpu, scales_cpu = q[:,both].cpu(), scales[both].cpu()
    report = dict(diagnostic_only=True, tolerance=dict(atol=0.005,rtol=0.005),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),records=[])
    for m in (1,2,4,8,15,16):
        for pitch in (2048,2080):
            x = torch.randn(m,pitch,device='cuda',dtype=torch.bfloat16)[:,:2048]
            for threads in (64,128,256):
                record=dict(m=m,pitch=pitch,threads=threads,patterns=[])
                legacy_pass=True
                for pattern in ('zero','tiny','normal','large','outlier'):
                    x.normal_()
                    if pattern=='zero': x.zero_()
                    elif pattern=='tiny': x.mul_(1e-9)
                    elif pattern=='large': x.mul_(100)
                    elif pattern=='outlier': x[:,0]=32
                    oldq, olds = torch.ops.biv150_native.quantize_residual_probe(x,0)
                    newq, news = torch.ops.biv150_native.quantize_residual_probe(x,threads)
                    before = torch.ops.biv150_native.gate_silu_residual(x,q,scales)
                    after = torch.ops.biv150_native.gate_residual_cached_probe(x,q,scales,threads)
                    _, legacy = cpu_reference(x,q,scales)
                    before_cpu, after_cpu = before[:,cols.to('cuda')].cpu(), after[:,cols.to('cuda')].cpu()
                    first, second, s1, s2 = cpu_quantize(x.cpu())
                    gold = cpu_output(first,second,s1,s2,weights_cpu,scales_cpu)
                    oldq_cpu, olds_cpu = oldq.cpu(), olds.cpu()
                    row=dict(pattern=pattern, raw_old_new_exact=torch.equal(oldq,newq) and torch.equal(olds,news),
                             full_gate_old_new_exact=torch.equal(before,after),
                             cpu_main_exact=torch.equal(first,oldq_cpu[:m]),
                             cpu_residual_exact=torch.equal(second,oldq_cpu[16:16+m]),
                             cpu_scales_exact=torch.equal(s1,olds_cpu[:m]) and torch.equal(s2,olds_cpu[16:16+m]),
                             old_legacy=close_report(before_cpu,legacy),new_legacy=close_report(after_cpu,legacy),
                             old_contracted_cpu=close_report(before_cpu,gold),new_contracted_cpu=close_report(after_cpu,gold))
                    row['cpu_residual_differences']=int((second!=oldq_cpu[16:16+m]).sum())
                    record['patterns'].append(row)
                    if not row['old_legacy']['passed']:
                        legacy_pass=False
                        fixture=args.out/f'fixture-m{m}-p{pitch}-t{threads}-{pattern}.pt'
                        torch.save(dict(x=x.cpu(),native_q=oldq_cpu,native_scales=olds_cpu,weights=weights_cpu,
                                        weight_scales=scales_cpu,cols=cols,old=before_cpu,new=after_cpu,
                                        legacy=legacy,contracted_cpu=gold,metadata=row),fixture)
                        row['fixture']=fixture.name
                        row['fixture_sha256']=hashlib.sha256(fixture.read_bytes()).hexdigest()
                        break # Preserve the initial failed-record RNG control flow.
                if legacy_pass:
                    x.normal_() # Original timing setup.
                    for factor in (0.0,0.5,4.0): x.normal_().mul_(factor)
                report['records'].append(record)
                (args.out/'result.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(record),flush=True)
    rows=[p for r in report['records'] for p in r['patterns']]
    report['summary']=dict(records=len(report['records']),patterns=len(rows),
        legacy_failed_patterns=sum(not r['old_legacy']['passed'] for r in rows),
        contracted_failed_patterns=sum(not r['old_contracted_cpu']['passed'] for r in rows),
        raw_cpu_failed_patterns=sum(not all(r[k] for k in ('cpu_main_exact','cpu_residual_exact','cpu_scales_exact')) for r in rows),
        bit_identity_failures=sum(not r['raw_old_new_exact'] or not r['full_gate_old_new_exact'] for r in rows))
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary']),flush=True)
    if report['summary']['bit_identity_failures'] or report['summary']['raw_cpu_failed_patterns'] or report['summary']['contracted_failed_patterns']:
        raise RuntimeError('Reference audit remains unresolved; no acceptance claim')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
