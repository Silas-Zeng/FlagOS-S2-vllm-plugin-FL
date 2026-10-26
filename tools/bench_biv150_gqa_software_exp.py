#!/usr/bin/env python3
"""Component-only bounded <=0 software exp2 GQA experiment.

Run only on an idle, coordinated BI-V150. No serving install, package edit,
production source edit, or official-score claim. CPU mode imports no torch.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import sys

from biv150_software_exp2 import (BASELINE as MAIN, source_factory,
                                  probe_values)

JIT = '/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py'
JIT_SHA = '66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
MODEL_HASH = {4096: '641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff',
              16384: 'e29a0c84d6b99f1ad44189fbdb74434b2d7da5675af2e2e2422941739916adb9'}

def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


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


def relative_rmse(a, b):
    diff = a.double() - b.double()
    denom = max(float(b.double().square().mean().sqrt()), 1e-30)
    return float(diff.square().mean().sqrt())/denom


def primitive_check(candidate, torch, out):
    # Device primitive must pass before any attention timing is attempted.
    values = probe_values(32768)
    x = torch.tensor(values, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)
    kernel = candidate._software_exp2_probe[(math.ceil(len(values)/256),)](
        x, y, len(values), 256, num_warps=4)
    got = y.cpu().tolist()
    worst_rel = worst_abs = 0.0
    exact_integers = True
    for value, actual in zip(values, got):
        assert math.isfinite(actual) and actual >= 0, (value, actual)
        if value < -126:
            assert actual == 0.0, (value, actual)
            continue
        expected = math.exp2(value)
        error = abs(actual-expected)
        worst_abs = max(worst_abs, error)
        worst_rel = max(worst_rel, error/expected)
        if value == int(value):
            exact_integers &= actual == expected
    assert worst_rel <= 2e-6, worst_rel
    assert exact_integers
    invalid = torch.tensor([1.0, float('inf'), float('nan')], dtype=torch.float32, device='cuda')
    invalid_out = torch.empty_like(invalid)
    candidate._software_exp2_probe[(1,)](invalid, invalid_out, 3, 256, num_warps=4)
    assert torch.isnan(invalid_out).all().item()
    return dict(inputs=len(values), max_relative_error=worst_rel,
        max_absolute_error=worst_abs, exact_integer_powers=exact_integers,
        invalid_domain_returns_nan=True, explicit_flush_below=-126,
        passed=True, resources=resources(kernel, out, 'primitive'))


def main(args):
    source = args.source.resolve()
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip() == args.commit
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'], text=True).strip()
    subprocess.run(['git','-C',str(source),'diff','--exit-code',MAIN,'--','vllm_fl','csrc'], check=True)
    assert hashlib.sha256(Path(JIT).read_bytes()).hexdigest() == JIT_SHA
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',9031)) != 0, 'Existing service; nothing stopped'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True)
    args.out.mkdir(parents=False, exist_ok=False)
    import torch
    from types import SimpleNamespace
    from biv150_benchmark_utils import inputs, load_module, main_call, reduce_call, graph_time
    torch.set_num_threads(4)
    props = torch.cuda.get_device_properties(0)
    assert 'BI-V150' in props.name and props.multi_processor_count == 16
    module = load_module(source/'vllm_fl/patches/biv150_gqa_decode.py','software_exp_frozen_gqa')
    generated, proof = source_factory((source/'vllm_fl/patches/biv150_gqa_decode.py').read_text(), args.exp_scope)
    generated_path = args.out/'generated-gqa-software-exp.py'
    generated_path.write_text(generated)
    atomic(args.out/'source-transform-proof.json', proof)
    candidate = load_module(generated_path,'gqa_software_exp_private')
    proxy = SimpleNamespace(LOG2E=module.LOG2E,
                            _gqa_decode_attention=candidate._gqa_decode_attention)
    result = dict(component_only=True, official_score=False, precision_gate=False,
        production_unchanged=True, source_commit=args.commit, production_baseline=MAIN,
        sdk_jit_sha256=JIT_SHA, source_transform=proof, records=[])
    atomic(args.out/'result.json',result)
    try:
        result['primitive'] = primitive_check(candidate, torch, args.out)
    except BaseException as error:
        result['primitive'] = dict(passed=False,error=repr(error))
        atomic(args.out/'result.json',result)
        raise
    atomic(args.out/'result.json',result)
    for batch,length in ((64,4096),(32,16384)):
        row = dict(batch=batch,kv=length,passed=False)
        result['records'].append(row)
        try:
            c=inputs(batch,length)
            packed=torch.empty((c['k'].shape[0],2,16,2,128),device='cuda',dtype=torch.bfloat16)
            k,v=packed.unbind(1)
            k.copy_(c['k']); v.copy_(c['v']); c.update(k=k,v=v)
            table=torch.full((batch,8192),-1,device='cuda',dtype=torch.int32)
            table[:,:c['table'].shape[1]].copy_(c['table'])
            c['table']=table[:,:c['table'].shape[1]]
            segments=module._pick_segments(batch,2,16)
            for key in ('scratch','maxima','sums'):
                c[key]=c[key].narrow(2,0,segments)
            def run(selected):
                kernel=main_call(selected,c,segments)
                reduce_call(module,c,segments,4)
                return kernel
            row.update(segments=segments, packed_shape=list(packed.shape),kv_stride=list(k.stride()),
                       table_stride=list(c['table'].stride()),same_allocation_and_strides=True)
            c['q'].zero_()  # exact exp2(0)=1: uniform attention is an invariant
            old=run(module); gold=c['out'].clone()
            new=run(proxy); torch.cuda.synchronize()
            row['baseline']=resources(old,args.out,f'b{batch}-baseline')
            row['candidate']=resources(new,args.out,f'b{batch}-candidate')
            assert old.hash == MODEL_HASH[length], 'Fixture differs from actual model kernel'
            assert row['baseline']['metadata']['use_sme']==14
            assert row['candidate']['metadata']['use_sme']==row['baseline']['metadata']['use_sme']
            assert row['candidate']['metadata']['num_stages']==1
            row['uniform_exact']=bool(torch.equal(c['out'],gold))
            assert row['uniform_exact'], 'exp2(0) uniform invariant failed'
            row['numeric_checks']=[]
            for amplitude in (1,8):
                for ragged in (False,True):
                    c['q'].normal_().mul_(amplitude)
                    c['lengths'].fill_(length)
                    if ragged:
                        for index,n in enumerate((0,1,15,16,17,63,100,length-1)):
                            c['lengths'][index]=n
                    c['out'].fill_(float('nan')); run(module); baseline=c['out'].clone()
                    c['out'].fill_(float('nan')); run(proxy); torch.cuda.synchronize()
                    finite=bool(torch.isfinite(c['out']).all().item())
                    delta=relative_rmse(c['out'],baseline)
                    row['numeric_checks'].append(dict(amplitude=amplitude,ragged=ragged,
                        finite=finite,candidate_vs_baseline_relative_rmse=delta,
                        max_abs=float((c['out'].float()-baseline.float()).abs().max()),
                        passed=finite and delta<=.002))
                    for index in (0,1,batch-1):
                        n=int(c['lengths'][index].cpu())
                        if n:
                            ids=c['table'][index,:(n+15)//16].long()
                            kk=k[ids].reshape(-1,2,128)[:n].cpu().double()
                            vv=v[ids].reshape(-1,2,128)[:n].cpu().double()
                            qq=c['q'][index].cpu().double().reshape(2,8,128)
                            pp=torch.einsum('hgd,thd->hgt',qq,kk).mul(128**-.5).softmax(-1)
                            ref=torch.einsum('hgt,thd->hgd',pp,vv).reshape(16,128)
                        else:
                            ref=torch.zeros((16,128),dtype=torch.float64)
                        errors=[relative_rmse(t[index].cpu(),ref) for t in (baseline,c['out'])]
                        row['numeric_checks'].append(dict(amplitude=amplitude,ragged=ragged,
                            row=index,length=n,baseline_relative_rmse=errors[0],
                            candidate_relative_rmse=errors[1],
                            passed=errors[1]<=max(.005,1.2*errors[0])))
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run(proxy)
                    c['q'].normal_().mul_(amplitude); c['out'].fill_(float('nan'))
                    graph.replay(); torch.cuda.synchronize(); changed=c['out'].clone()
                    run(module); torch.cuda.synchronize()
                    delta=relative_rmse(changed,c['out'])
                    row['numeric_checks'].append(dict(changed_input_graph=True,amplitude=amplitude,
                        ragged=ragged,finite=bool(torch.isfinite(changed).all().item()),
                        relative_rmse=delta,passed=bool(torch.isfinite(changed).all().item()) and delta<=.002))
                    del graph,changed,baseline
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run(proxy)
            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
            side_result=c['out'].clone();run(proxy);torch.cuda.synchronize()
            row['side_stream_finite']=bool(torch.isfinite(side_result).all().item())
            row['side_stream_exact']=bool(torch.equal(side_result,c['out']))
            del side_result
            row['passed']=(row['uniform_exact'] and all(x['passed'] for x in row['numeric_checks'])
                           and row['side_stream_finite'] and row['side_stream_exact'])
            atomic(args.out/'result.json',result)
            assert row['passed'],'Component numeric checks failed; skip timing'
            c['lengths'].fill_(length);c['q'].normal_()
            row['timings']=[]
            for iteration in range(args.iterations):
                choices=[('baseline_ms',module),('candidate_ms',proxy)]
                if iteration%2:choices.reverse()
                timing={name:graph_time(lambda selected=selected:run(selected),repeats=15,unroll=8)[0]
                        for name,selected in choices}
                timing['speedup']=timing['baseline_ms']/timing['candidate_ms']
                row['timings'].append(timing)
            row['median_speedup']=statistics.median(x['speedup'] for x in row['timings'])
            del c,packed,k,v,table,gold
            torch.cuda.empty_cache()
        except BaseException as error:
            row['error']=repr(error)
            atomic(args.out/'result.json',result)
            raise
        atomic(args.out/'result.json',result)
        print(json.dumps(row),flush=True)
    result['component_numeric_passed']=all(row['passed'] for row in result['records'])
    atomic(args.out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--commit')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--exp-scope',choices=('probability','both'),default='probability')
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--check-source',action='store_true')
    args=parser.parse_args()
    if args.iterations<1:parser.error('--iterations must be positive')
    if args.check_source:
        _,proof=source_factory((args.source/'vllm_fl/patches/biv150_gqa_decode.py').read_text(),args.exp_scope)
        print(json.dumps(dict(**proof,torch_imported='torch' in sys.modules,gpu_executed=False)))
    else:
        if not args.commit or not args.out:parser.error('--commit and --out required for GPU experiment')
        main(args)
