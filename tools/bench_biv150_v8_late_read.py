"""CPU preview by default. Explicit compile-only proof must precede device diagnostics."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

from biv150_v8_late_read_checks import (CANDIDATE_KERNEL, DEPENDENCY, FROZEN_BENCH,
    FROZEN_KERNEL, ROOT, adapted_benchmark_tree, lowering_check, sha, source_inverse)

CASES = {'ragged': (8, 257), 'ragged-long': (8, 257),
         'steady4k': (64, 4096), 'steady16k': (32, 16384)}


def head():
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def device_preflight():
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip(), 'Dirty source'
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0, 'Service active'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True), 'GPU busy'
    source_inverse()


def save(path, record):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(path)


def resources(compiled, directory):
    # warmup compiles; initializing a module handle obtains backend register and
    # spill metadata. There is deliberately no kernel invocation in this mode.
    compiled._init_handles()
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, value in compiled.asm.items():
        if not isinstance(value, (str, bytes)):
            continue
        raw = value.encode() if isinstance(value, str) else value
        path = directory/('actual.'+name)
        path.write_bytes(raw)
        files[name] = dict(path=str(path.resolve()), bytes=len(raw), sha256=sha(raw))
    return dict(hash=str(compiled.hash), n_regs=compiled.n_regs, n_spills=compiled.n_spills,
        metadata={key: getattr(compiled.metadata, key, None) for key in
                  ('shared', 'num_warps', 'num_stages', 'use_sme', 'warp_size', 'triton_version')}, files=files)


def read_ir(record):
    result = {}
    for name in ('ttgir', 'llir', 'asm'):
        entry = record['files'][name]
        raw = Path(entry['path']).read_bytes()
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], 'Changed compiled artifact'
        result[name] = raw.decode()
    return result


def inspect_case(row):
    previous = read_ir(row['frozen_v8'])
    late = read_ir(row['late_v8'])
    proof = lowering_check(late['ttgir'], late['llir'], late['asm'], previous=previous)
    equal = all(row['late_v8']['metadata'][key] == row['frozen_v8']['metadata'][key]
                for key in ('shared', 'num_warps', 'num_stages', 'warp_size'))
    if not equal:
        proof['reasons'].append('Shared bytes/warp/stage configuration differs from frozen V8')
        proof['mechanism_passed'] = False
    proof['resource_delta'] = {key: row['late_v8'][key]-row['frozen_v8'][key]
                               for key in ('n_regs', 'n_spills')}
    return proof


def compile_only(args):
    device_preflight()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import biv150_benchmark_utils as bench
    base = bench.load_module(ROOT/'vllm_fl/patches/biv150_gqa_decode.py', 'v8_bf16p_shipping_gqa')
    original = bench.load_module(FROZEN_KERNEL, 'late_read_frozen_v8')
    # Match the adapted runtime probe's module identity as well as its source.
    own = bench.load_module(CANDIDATE_KERNEL, 'v8_bf16p_experimental_kernels')
    assert base.TARGET_PROGRAMS == 256
    # Tiny pointer stubs only: stride/constexpr ABI matches the frozen packed
    # fixture, including its NARROWED (not recontiguous) scratch. No values are
    # initialized or consumed; these are never launched by compile-only mode.
    pointers = {name: torch.empty(1, device='cuda', dtype=dtype) for name, dtype in
                [('bf16', torch.bfloat16), ('i8', torch.int8), ('f32', torch.float32), ('i32', torch.int32)]}
    record = dict(completed=False, compile_only=True, kernel_launches=0, source_commit=head(),
        dependency=DEPENDENCY, source=source_inverse(), precision_gate=False,
        numeric_passed=False, official_score=False, cases=[], errors=[])
    destination = args.out/'compile-proof.json'
    save(destination, record)
    try:
        for name in args.case:
            batch, length = CASES[name]
            segments = base._pick_segments(batch, 2, 16)
            kw = dict(segm_output_ptr=pointers['f32'], segm_max_ptr=pointers['f32'],
                segm_expsum_ptr=pointers['f32'], query_ptr=pointers['bf16'], key_cache_ptr=pointers['bf16'],
                value_cache_ptr=pointers['bf16'], seq_lens_ptr=pointers['i32'], block_table_ptr=pointers['i32'],
                softmax_scale_log2e=128**-.5*base.LOG2E, queries_per_kv=8,
                block_table_stride=8192, query_stride_token=2048, query_stride_head=128,
                k_stride_block=8192, k_stride_slot=256, k_stride_head=128,
                v_stride_block=8192, v_stride_slot=256, v_stride_head=128,
                segment_stride_token=32768, segment_stride_head=2048, segment_stride_segment=128,
                max_stride_token=256, max_stride_head=16,
                HEAD_SIZE=128, PAGE_SIZE=16, ROWS=8, SEGMENTS=segments, num_warps=1, num_stages=1)
            grid = (batch, 2, segments)
            row = dict(case=name, nominal_length=length, segments=segments, grid=list(grid))
            compiled = base._gqa_decode_attention.warmup(grid=grid, **kw)
            row['bf16'] = resources(compiled, args.out/(name+'-bf16'))
            kw.update(value_cache_ptr=pointers['i8'], value_scale_ptr=pointers['f32'],
                v_stride_block=16384, scale_stride_block=4096, scale_stride_slot=2, scale_stride_head=1)
            for label, module in [('frozen_v8', original), ('late_v8', own)]:
                compiled = module.gqa_decode_v8_bf16p.warmup(grid=grid, **kw)
                row[label] = resources(compiled, args.out/(name+'-'+label))
            row['lowering'] = inspect_case(row)
            record['cases'].append(row)
            save(destination, record)
        record['mechanism_passed'] = all(row['lowering']['mechanism_passed'] for row in record['cases'])
        record['completed'] = True
        save(destination, record)
        assert record['mechanism_passed'], 'Late-V mechanism absent or staging changed: stop before device execution/timing'
    except Exception as error:
        record['errors'].append(repr(error)); save(destination, record)
        raise


def validated_proof(path, chosen):
    record = json.loads(path.read_text())
    assert record['completed'] and record['compile_only'] and record['kernel_launches'] == 0
    assert record['source_commit'] == head() and record['dependency'] == DEPENDENCY
    assert record['source'] == source_inverse(), 'Proof is from different source'
    assert record['mechanism_passed'] is True and not record['errors'], 'Mechanism failed: do not launch'
    by_case = {row['case']: row for row in record['cases']}
    assert set(chosen) <= by_case.keys(), 'Unproven case requested'
    for name in chosen:
        assert inspect_case(by_case[name])['mechanism_passed'], 'Rechecked actual lowering failed'
    return record


class GuardedKernel:
    """Check actual runtime specialization BEFORE first launch; no new math."""
    def __init__(self, kernel, proof_path):
        self.kernel = kernel
        self.proof = validated_proof(proof_path, [])
        self.allowed = {row['late_v8']['hash']: row['late_v8'] for row in self.proof['cases']}
        self.checked = set()

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        def call(**kwargs):
            signature = (tuple(grid), tuple(sorted((key,
                (str(value.dtype), value.data_ptr()) if hasattr(value, 'data_ptr') else value)
                for key, value in kwargs.items())))
            if signature not in self.checked:
                compiled = self.kernel.warmup(grid=grid, **kwargs)
                assert str(compiled.hash) in self.allowed, 'Actual runtime specialization lacks compile proof; stop before launch'
                expected = self.allowed[str(compiled.hash)]
                for name in ('ttgir', 'llir', 'asm'):
                    value = compiled.asm[name]
                    assert sha(value.encode() if isinstance(value, str) else value) == expected['files'][name]['sha256'], 'Runtime IR differs from proof'
                self.checked.add(signature)
            return launch(**kwargs)
        return call


def main(args):
    source_inverse()
    if not args.compile_only and not args.run_gpu:
        print(json.dumps(dict(dependency=DEPENDENCY, candidate='late_v_scale_read',
            cases=args.case, source_inverse=True, compilation='not run', mechanism_passed=False,
            numeric_passed=False, precision_gate=False, official_score=False,
            next_step='Root-only compile-only -> actual IR/ASM gate -> explicit original guarded diagnostics',
            frozen73_preserved=True)))
        return
    if args.compile_only:
        return compile_only(args)
    validated_proof(args.ir_proof, args.case)
    # Execute the independently reversible adaptation of the frozen4a probe.
    # It adds a pre-launch mechanism gate and changes the kernel filename only.
    namespace = {'__file__': str(FROZEN_BENCH), '__name__': 'late_read_frozen_guarded_probe',
                 'GuardedKernel': GuardedKernel}
    exec(compile(adapted_benchmark_tree(), str(FROZEN_BENCH), 'exec'), namespace)
    namespace['main'](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--compile-only', action='store_true', help='Explicit JIT/module-handle creation, no kernel launch; root schedules this')
    mode.add_argument('--run-gpu', action='store_true', help='Requires exact-source successful compile proof; all original4a guards remain')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--ir-proof', type=Path)
    parser.add_argument('--case', action='append', choices=sorted(CASES))
    parser.add_argument('--warps', type=int, choices=(1,), default=1)
    parser.add_argument('--numeric-diagnostic', action='store_true',
                        help='Same4a finite-candidate-only diagnostic policy; baseline/codec/Graph remain hard')
    args = parser.parse_args()
    args.case = args.case or ['ragged', 'ragged-long', 'steady4k', 'steady16k']
    if (args.compile_only or args.run_gpu) and args.out is None:
        parser.error('--out required for explicit compilation/device work')
    if args.run_gpu and args.ir_proof is None:
        parser.error('--ir-proof required before device work')
    if args.compile_only and args.numeric_diagnostic:
        parser.error('--numeric-diagnostic applies only to guarded device diagnostics')
    main(args)
