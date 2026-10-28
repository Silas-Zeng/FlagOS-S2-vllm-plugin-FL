#!/usr/bin/env python3
"""CPU plan by default; explicit Root-owned Q32/Q64 warm compiles, zero launches."""
import argparse
import errno
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
sys.dont_write_bytecode = True
import biv150_konly_q64_compile_source as c
from biv150_konly_q64_compile_arguments import kwargs


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', '-C', str(c.ROOT), *args], text=True).strip()


def allocate(torch):
    # Same Q32 compile fixture descriptors; no initializer or device copy.
    def empty(shape, dtype): return torch.empty(shape, device='cuda', dtype=dtype)
    raw = empty((773 * 16384,), torch.uint8)
    t = dict(q=empty((2048,16,128), torch.bfloat16), out=empty((2048,16,128), torch.bfloat16),
        k=torch.as_strided(raw.view(torch.int8), (773,16,2,128), (16384,256,128,1), 0),
        v=torch.as_strided(raw.view(torch.bfloat16), (773,16,2,128), (8192,256,128,1), 4096),
        kscale=raw.view(torch.float32), table=empty((1,8192), torch.int32),
        lengths=empty((1,), torch.int32), cu_q=empty((2,), torch.int32),
        k_descale=empty((), torch.float32), v_descale=empty((), torch.float32))
    return t, raw


def run(args):
    plan = c.plan()
    c.need(args.root_owned_exclusive, 'Root serial ownership acknowledgement required')
    c.need(args.commit and args.out and args.out.is_absolute() and not args.out.exists(), 'exact commit/new absolute output')
    c.need(not args.out.resolve().is_relative_to(c.ROOT), 'output outside checkout required')
    c.need(git('rev-parse', 'HEAD') == args.commit and not git('status', '--porcelain'), 'exact clean frontend freeze')
    c.need(not git('diff', c.BASE, '--', 'vllm_fl', 'csrc'), 'production must equal a455')
    c.need(not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules), 'cold admission')
    args.out.mkdir(parents=True)
    result = dict(plan, commit=args.commit, mode='compile-only', completed=False, compile_passed=False,
        errors=[], records={}, JIT_warmups_attempted=0, attention_launches=0, encoder_launches=0,
        writer_launches=0, initialized_tensor_elements=0, CUDA_API_entered=False,
        context_or_allocations_attempted=False, actual_native_audit_pending=True)
    def checkpoint(): save(args.out/'summary.json', result)
    checkpoint()
    try:
        h32 = c.proposal(32)
        base = Path('/usr/local/lib/python3.12/site-packages')
        installed = dict(sdk=base/'vllm/v1/attention/ops/triton_unified_attention.py',
            helpers=base/'vllm/v1/attention/ops/triton_attention_helpers.py',
            jit=base/'triton/spec/iluvatar/triton/runtime/jit.py')
        for key, path in installed.items(): c.need(c.sha(path.read_bytes()) == h32.PINS[key], 'installed pin ' + key)
        smi = subprocess.run(['/usr/local/corex/bin/ixsmi','-i','0'], capture_output=True, text=True, timeout=15)
        pids = subprocess.run(['/usr/local/corex/bin/ixsmi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=15)
        save(args.out/'prelaunch-observation.json', dict(smi=vars(smi), pids=vars(pids)))
        c.need(smi.returncode == pids.returncode == 0 and 'No running processes' in smi.stdout
            and pids.stdout.strip() in ('','No running processes','No running processes found','No running processes found.'),
            'visible GPU process observation unavailable or busy; exclusivity is Root acknowledgement')
        with socket.socket() as sock: port = sock.connect_ex(('127.0.0.1',9031))
        c.need(port == errno.ECONNREFUSED, 'service 9031 not confirmed closed')
        for key in ('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE',
                    'VLLM_FL_BIV150_KONLY_NATIVE_DECODE','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'):
            c.need(os.environ.get(key,'0') != '1', 'component rejects service activation ' + key)
        os.environ['TRITON_CACHE_DIR'] = str(args.out/'triton-cache')
        import torch
        from vllm.triton_utils import triton
        result['CUDA_initialized_after_torch_import'] = bool(torch.cuda.is_initialized())
        result.update(CUDA_API_entered=True, GPU_executed=True)
        checkpoint()
        target = triton.runtime.driver.active.get_current_target()
        target = dict(backend=target.backend, arch=target.arch, warp_size=target.warp_size)
        c.need(target == dict(backend='corex', arch=71, warp_size=64), 'actual CoreX target')
        folder = args.out/'sources'; folder.mkdir()
        source = folder/'candidate_q64.py'; source.write_text(c.candidate_source())
        (folder/'q64_hoist_proposal.py').write_text(c.proposal_source())
        patch = c.ROOT/'vllm_fl/patches'
        modules = {32:c.load_file('cjj_q64_screen_control', patch/'biv150_konly_encoded_q32.py'),
                   64:c.load_file('cjj_q64_screen_candidate', source)}
        adapter = c.load_file('cjj_q64_screen_snapshot', patch/'biv150_konly_encoded_q32_adapter.py')
        # Only use the old SSA-normalization helper. Its Q32-specific native
        # instruction/count audit is deliberately NOT used to qualify Q64.
        native = c.module_from_source('cjj_q64_screen_alpha', c.original('biv150_konly_encoded_hoist_native.py'))
        result['context_or_allocations_attempted'] = True; checkpoint()
        tensors, owner = allocate(torch)
        result['tensor_descriptors'] = {k:dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
            storage_offset=t.storage_offset(), storage_bytes=t.untyped_storage().nbytes()) for k,t in tensors.items()}
        for bq in (32,64):
            fn = getattr(modules[bq], 'kernel_cjj_konly_q32' if bq == 32 else c.NAME)
            pins = adapter.Adapter.installed_pins(fn, modules[bq].__file__, {
                'vllm.v1.attention.ops.triton_unified_attention':h32.PINS['sdk'],
                'vllm.v1.attention.ops.triton_attention_helpers':h32.PINS['helpers']})
            c.need(pins == h32.PINS, 'actual codec/JIT pins')
            for key, module_name in (('sdk','vllm.v1.attention.ops.triton_unified_attention'),
                                     ('helpers','vllm.v1.attention.ops.triton_attention_helpers')):
                c.need(Path(sys.modules[module_name].__file__).resolve() == installed[key].resolve(), 'actual import ' + key)
            c.need(Path(inspect.getfile(type(fn))).resolve() == installed['jit'].resolve(), 'actual JIT path')
            config, grid = c.geometry(2048,bq)
            launch = kwargs(dict(profile='q' + str(bq), sequences=1, segments=1, config=config), tensors, True)
            launch.update(num_warps=16, num_stages=1)
            role = 'q' + str(bq); record = {}; result['records'][role] = record
            h = c.proposal(bq)
            out = args.out/role; out.mkdir()
            def preserve(kernel, label):
                snapshot = adapter.kernel_snapshot(kernel,fn)
                adapter.Adapter._preserve(out/label, kernel, snapshot)
                # GPUTarget inside metadata is not a primitive ABI value. Keep
                # the original representation; exact typed ABI is in snapshot.
                (out/label/'compiler-metadata.txt').write_text(repr(kernel.metadata)+'\n')
                record[label] = snapshot; checkpoint()
                return snapshot
            result['JIT_warmups_attempted'] += 1; result['JIT_executed'] = True; checkpoint()
            first = fn.warmup(grid=grid, **launch)
            before = preserve(first,'unhoisted')
            meta = c.metadata(h,before,2048,bq,target)
            override, proof = h.propose(first.asm['ttgir'],meta)
            path = out/'override.ttgir'; path.write_text(override)
            save(out/'proposal.json',proof); record['proposal'] = proof
            result['JIT_warmups_attempted'] += 1; checkpoint()
            second = fn.warmup(grid=grid, **launch, ir_override=str(path))
            after = preserve(second,'hoisted')
            c.need(all(before[k] == after[k] for k in ('source_digest','function_name','ABI','launch_geometry')),
                'typed ABI/source differs beyond one operand move')
            c.need(first.asm['ttir'] == second.asm['ttir'], 'TTIR changed beyond override')
            c.need(native.alpha(override) == native.alpha(second.asm['ttgir']), 'actual override differs beyond scoped SSA names')
            record['actual_hoisted_TTGIR_structure'] = h.grammar(second.asm['ttgir'],after['function_name'],hoisted=True)
            record['handle_resources'] = {}
            for label, kernel in (('unhoisted',first),('hoisted',second)):
                kernel._init_handles()
                resource = dict(n_regs=getattr(kernel,'n_regs',None), n_spills=getattr(kernel,'n_spills',None),
                    shared=getattr(kernel.metadata,'shared',None), threads=16*64,
                    phase='handles_initialized_zero_reader_launches', occupancy_claimed=False)
                record['handle_resources'][label] = resource; checkpoint()
                c.need(type(resource['n_regs']) is int and resource['n_regs'] > 0
                    and type(resource['n_spills']) is int and resource['n_spills'] == 0,
                    'unknown or spilled actual handle; stop before numeric')
                c.need(type(resource['shared']) is int and resource['shared'] >= 0, 'unknown shared resource')
            record['actual_backend_hoist_and_resource_cost_review_pending'] = True
            checkpoint()
        result.update(completed=True, compile_passed=True,
            actual_handle_resources_recorded=True, actual_resources_verified=False,
            CUDA_initialized_after_compile=bool(torch.cuda.is_initialized()),
            GPU_field_means='CUDA API/context/allocation work; attention/writer/encoder launches remain zero')
        checkpoint(); return result
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)))
        checkpoint(); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--root-owned-exclusive', action='store_true')
    parser.add_argument('--commit'); parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.compile_only: result = run(args)
    else:
        c.need(not any((args.commit,args.out,args.root_owned_exclusive)), 'device options require --compile-only')
        result = c.plan()
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__ == '__main__': main()
