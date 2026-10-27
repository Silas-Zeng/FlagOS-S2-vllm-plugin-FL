#!/usr/bin/env python3
"""CPU by default. Explicit compile/GPU primitive validation only; no timing."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess

from biv150_konly_i8_qk_ir import audit_artifacts
from biv150_konly_i8_qk_reference import PAGE_BYTES, f32_bits
from check_biv150_konly_i8_qk_cpu import CASES, KERNEL, ROOT, host_case, run_checks

JIT = Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py')
JIT_SHA = '66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
GUARD = 16
INTEGER_CANARY = 123456789
SCORE_CANARY = -98765.5


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def signed_bits(value):
    bits = f32_bits(value)
    return bits if bits < 2**31 else bits - 2**32


def preserve(kernel, destination, phase):
    kernel_hash = str(kernel.hash)
    assert re.fullmatch('[0-9a-f]{64}', kernel_hash), 'unexpected kernel hash identity'
    destination = destination / kernel_hash
    destination.mkdir(parents=True, exist_ok=True)
    files, strings = {}, {}
    for name, value in kernel.asm.items():
        if not isinstance(value, (str, bytes)):
            continue
        raw = value.encode() if isinstance(value, str) else value
        path = destination / ('actual.' + name)
        path.write_bytes(raw)
        files[name] = dict(path=kernel_hash+'/'+path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if isinstance(value, str):
            strings[name] = value
    md = kernel.metadata
    resources = dict(phase=phase, hash=str(kernel.hash), n_regs=getattr(kernel, 'n_regs', None),
                     n_spills=getattr(kernel, 'n_spills', None),
                     metadata={k:getattr(md, k, None) for k in
                               ('shared','num_warps','num_stages','warp_size','use_sme')})
    resources['loaded_resources_known'] = phase == 'post_launch' and all(
        isinstance(v, int) and not isinstance(v, bool)
        for v in (resources['n_regs'], resources['n_spills'], resources['metadata']['shared']))
    return dict(files=files, resources=resources, actual_ir=audit_artifacts(strings),
                all_required_artifacts_present=all(k in files and files[k]['bytes'] > 0
                                                  for k in ('ttgir','llir','asm','cubin')))


def run_device(args, cpu):
    assert args.commit and args.out, 'device work requires frozen --commit and fresh --out'
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip() == args.commit
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'], text=True).strip()
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','56f1838192abc940b3fb9accef490022ac125000',
                    '--','vllm_fl','csrc'], check=True)
    assert JIT.is_file() and hashlib.sha256(JIT.read_bytes()).hexdigest() == JIT_SHA, 'unaudited SDK JIT'
    if args.run_gpu:
        assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True), 'GPU busy; no queue/retry'
        with socket.socket() as guard:
            assert guard.connect_ex(('127.0.0.1',9031)) != 0, 'service9031 is running'
    args.out.mkdir(parents=True, exist_ok=False)
    atomic(args.out/'cpu.json', cpu)
    (args.out/'primitive-source.py.txt').write_bytes(KERNEL.read_bytes())
    os.environ['TRITON_CACHE_DIR'] = str(args.out/'triton-cache')
    result = dict(passed=False, gpu_executed=False, compile_only=args.compile_only, source_commit=args.commit,
                  scope='private_primitive_only', official_score=False, model_precision_gate=False,
                  kernel_sha256=hashlib.sha256(KERNEL.read_bytes()).hexdigest(), sdk_jit_sha256=JIT_SHA,
                  cpu=cpu, checks=[], errors=[], compiled_hashes=[], capacity_reclaimed=False, allocated_page_bytes=PAGE_BYTES,
                  gpu_quantizer_implemented=False, performance_verified=False, service_verified=False)
    def save(): atomic(args.out/'result.json', result)
    save()
    try:
        import torch
        import triton
        target = triton.runtime.driver.active.get_current_target()
        result['target'] = dict(backend=target.backend, arch=target.arch, warp_size=target.warp_size)
        assert (target.backend, target.arch, target.warp_size) == ('corex',71,64), 'wrong target'
        spec = importlib.util.spec_from_file_location('private_konly_i8_qk_probe', KERNEL)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.set_num_threads(2)
        device = torch.device('cuda', torch.cuda.current_device())
        packed = torch.empty(3*PAGE_BYTES, dtype=torch.uint8, device=device)
        qcodes = torch.empty(CASES*1024, dtype=torch.int8, device=device)
        qscales = torch.empty(CASES*8, dtype=torch.float32, device=device)
        pages = torch.empty(CASES, dtype=torch.int32, device=device)
        heads, tails = torch.empty_like(pages), torch.empty_like(pages)
        istore = torch.empty(CASES*128+2*GUARD, dtype=torch.int32, device=device)
        sstore = torch.empty(CASES*128+2*GUARD, dtype=torch.float32, device=device)
        integers, scores = istore[GUARD:-GUARD], sstore[GUARD:-GUARD]
        code_view, scale_view = packed.view(torch.int8), packed.view(torch.float32)
        vview = torch.as_strided(packed.view(torch.bfloat16), (3,16,2,128), (8192,256,128,1), 4096)
        assert vview.storage_offset()*2 == 8192 and tuple(vview.stride()) == (8192,256,128,1)
        arguments = (qcodes,qscales,code_view,scale_view,pages,heads,tails,integers,scores)
        def call():
            return module.konly_i8_qk_probe[(CASES,)](*arguments,num_warps=1,num_stages=1)
        if args.compile_only:
            compiled = module.konly_i8_qk_probe.warmup(*arguments,grid=(CASES,),num_warps=1,num_stages=1)
            result['compiler'] = preserve(compiled,args.out/'compiler-ir','pre_launch')
            result['status'] = 'compile_only_not_device_qualified'
            save()
            return result
        patterns = [host_case(p) for p in range(3)]
        def load(host):
            packed.copy_(torch.tensor(list(host['packed']), dtype=torch.uint8))
            qcodes.copy_(torch.tensor(host['qcodes'], dtype=torch.int8))
            qscales.copy_(torch.tensor(host['qscales'], dtype=torch.float32))
            for tensor, key in ((pages,'pages'),(heads,'heads'),(tails,'tails')):
                tensor.copy_(torch.tensor(host[key], dtype=torch.int32))
            istore.fill_(INTEGER_CANARY); sstore.fill_(SCORE_CANARY)
            integers.fill_(-999999); scores.fill_(math.nan)
        def check(host, stream, stream_name, pattern, mode):
            stream.synchronize()
            raw = integers.cpu().tolist()
            bits = scores.cpu().view(torch.int32).tolist()
            expected_i = [v for e in host['expected'] for row in e['integer'] for v in row]
            expected_s = [v for e in host['expected'] for row in e['fp32'] for v in row]
            assert raw == expected_i, 'I8 dot differs from independent integer oracle'
            assert bits == [signed_bits(v) for v in expected_s], 'scaled scores differ from explicit IEEE FP32 order'
            assert istore[:GUARD].cpu().tolist() == istore[-GUARD:].cpu().tolist() == [INTEGER_CANARY]*GUARD
            assert sstore[:GUARD].cpu().tolist() == sstore[-GUARD:].cpu().tolist() == [SCORE_CANARY]*GUARD
            assert bytes(packed.cpu().tolist()) == host['packed'], 'read-only packed payload/V/padding was modified'
            actual_s = scores.cpu().tolist()
            errors = []
            for case,e in enumerate(host['expected']):
                for row in range(8):
                    for col in range(host['tails'][case]):
                        ref = e['quantized_operand_fp64'][row][col]
                        errors.append(actual_s[case*128+row*16+col] - ref)
            maximum = max(map(abs, errors), default=0.0)
            rms = math.sqrt(math.fsum(x*x for x in errors)/max(1,len(errors)))
            assert rms != 0 or maximum == 0, 'contradictory RMS/max statistics'
            result['checks'].append(dict(stream=stream_name, pattern=pattern, mode=mode,
                integer_exact=True, score_fp32_bits_exact=True, quantized_operand_fp64_max_abs=maximum,
                quantized_operand_fp64_rms=rms, canaries=True, packed_bytes_and_original_v_unchanged=True,
                dynamic_page_head_tail_metadata=True))
            save()
        default = torch.cuda.current_stream()
        for stream_name in ('default','side'):
            stream = default if stream_name == 'default' else torch.cuda.Stream()
            stream.wait_stream(default)
            with torch.cuda.stream(stream):
                for pattern,host in enumerate(patterns):
                    load(host)
                    result['gpu_executed'] = True
                    compiled = call()
                    stream.synchronize()
                    result['compiler'] = preserve(compiled,args.out/'compiler-ir','post_launch')
                    kernel_hash = str(compiled.hash)
                    if kernel_hash not in result['compiled_hashes']:
                        result['compiled_hashes'].append(kernel_hash)
                    save()
                    assert len(result['compiled_hashes']) == 1, 'changed-input replay unexpectedly recompiled'
                    assert result['compiler']['all_required_artifacts_present'], 'incomplete actual compiler artifacts'
                    assert result['compiler']['actual_ir']['passed'], 'actual mechanism IR gate failed'
                    check(host,stream,stream_name,pattern,'eager')
                capture = torch.cuda.Stream()
                capture.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture): call()
                stream.wait_stream(capture)
                for pattern in (0,1,2,0,1,2):
                    load(patterns[pattern]); graph.replay()
                    check(patterns[pattern],stream,stream_name,pattern,'changed-poisoned-graph')
            default.wait_stream(stream)
        result['numeric_passed'] = len(result['checks']) == 18
        result['passed'] = (result['numeric_passed'] and result['compiler']['actual_ir']['passed']
                            and result['compiler']['resources']['loaded_resources_known'])
        result['status'] = 'primitive_only' if result['passed'] else 'resource_evidence_unknown'
        save()
        return result
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__, message=str(error)))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run-gpu', action='store_true')
    mode.add_argument('--compile-only', action='store_true')
    parser.add_argument('--commit')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    cpu = run_checks()
    if args.run_gpu or args.compile_only:
        print(json.dumps(run_device(args,cpu)))
    else:
        print(json.dumps(cpu))


if __name__ == '__main__':
    main()
