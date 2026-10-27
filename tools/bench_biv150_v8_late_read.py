"""CPU preview by default. Explicit compile-only proof must precede device diagnostics."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

from biv150_v8_late_read_checks import (CANDIDATE_KERNEL, DEPENDENCY, FROZEN_BENCH,
    FROZEN_KERNEL, ROOT, AFTER_QK_CONTRACT, adapted_benchmark_tree, after_qk_check,
    lowering_check, runtime_descriptors, sha, source_inverse)

CASES = {'ragged': (8, 257), 'ragged-long': (8, 257),
         'steady4k': (64, 4096), 'steady16k': (32, 16384)}
OPTION_KEYS = ('enable_fp_fusion','enable_reflect_ftz','default_dot_input_precision',
               'maxnreg','backend_name','arch','sanitize_overflow','debug')


def compiler_options(compiled):
    return {key:getattr(compiled.metadata,key,None) for key in OPTION_KEYS}


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
        options=compiler_options(compiled),
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


def inspect_after_qk_case(row):
    previous,late = read_ir(row['frozen_v8']),read_ir(row['late_v8'])
    proof = after_qk_check(late['ttgir'],late['llir'],late['asm'],previous=previous)
    batch,length = CASES[row['case']]
    if row['descriptors'] != runtime_descriptors(batch,length,row['segments']):
        proof['reasons'].append('Actual descriptor plan differs from frozen runtime')
    for key in ('shared','num_warps','num_stages','warp_size','use_sme','triton_version'):
        if row['late_v8']['metadata'][key] != row['frozen_v8']['metadata'][key]:
            proof['reasons'].append('Frozen/late metadata differs: '+key)
    metadata = row['late_v8']['metadata']
    if metadata['num_warps'] != 1 or metadata['num_stages'] != 1 or metadata['warp_size'] != 64:
        proof['reasons'].append('Only audited w1/s1/warp64 contract is supported')
    if metadata['shared'] != 4096 or any(row[label]['n_spills'] != 0 for label in ('late_v8','frozen_v8')):
        proof['reasons'].append('Only audited4096B shared/zero-spill V8 staging is supported')
    options = row['late_v8'].get('options',{})
    if set(options) != set(OPTION_KEYS) or options != row['frozen_v8'].get('options'):
        proof['reasons'].append('Missing or unequal compiler options; fresh explicit compile proof required')
    elif any(options[key] is None for key in OPTION_KEYS if key != 'maxnreg'):
        proof['reasons'].append('Compiler option metadata unavailable')
    if row['late_v8']['hash'] == row['frozen_v8']['hash']:
        proof['reasons'].append('Late/frozen actual compiled hashes are identical')
    proof['diagnostic_mechanism_passed'] = not proof['reasons']
    return proof


def compile_tensors(torch, batch, length, segments):
    """Uninitialized real-shape views; metadata exactly matches frozen4a.

    Allocation/view/narrow do not launch a kernel. One packed allocation is
    reused for baseline/V8 compilation, with their respective actual views;
    the compile-only mode never compares or consumes uninitialized values.
    """
    plan = runtime_descriptors(batch, length, segments)
    packed = torch.empty(tuple(plan['packed_shape']), device='cuda', dtype=torch.bfloat16)
    key, value = packed.unbind(1)
    raw_bytes = packed.view(torch.int8).reshape(-1)
    blocks = plan['packed_shape'][0]
    encoded_v = torch.as_strided(raw_bytes, (blocks,16,2,128), (16384,256,128,1), 8192)
    scales = torch.as_strided(packed.view(torch.float32).reshape(-1),
                             (blocks,16,2), (4096,2,1), 3072)
    tensors = dict(q=torch.empty(batch,16,128,device='cuda',dtype=torch.bfloat16),
        k=key, v=value, v8=encoded_v, scale=scales,
        scratch=torch.empty(batch,16,16,128,device='cuda',dtype=torch.float32).narrow(2,0,segments),
        maxima=torch.empty(batch,16,16,device='cuda',dtype=torch.float32).narrow(2,0,segments),
        sums=torch.empty(batch,16,16,device='cuda',dtype=torch.float32).narrow(2,0,segments),
        table=torch.empty(batch,8192,device='cuda',dtype=torch.int32),
        lengths=torch.empty(batch,device='cuda',dtype=torch.int32))
    for name,tensor in tensors.items():
        expected = plan['tensors'][name]
        actual = dict(shape=list(tensor.shape), stride=list(tensor.stride()),
            dtype=str(tensor.dtype).removeprefix('torch.'), storage_offset=tensor.storage_offset(),
            is_contiguous=tensor.is_contiguous())
        assert all(actual[key] == expected[key] for key in actual), ('Descriptor mismatch',name,actual,expected)
    return tensors, plan


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
    record = dict(completed=False, compile_only=True, kernel_launches=0, source_commit=head(),
        dependency=DEPENDENCY, source=source_inverse(), precision_gate=False,
        numeric_passed=False, official_score=False, cases=[], errors=[],
        selected_mechanism_contract=AFTER_QK_CONTRACT if args.after_qk_diagnostic else 'strict_after_softmax',
        after_qk_diagnostic=bool(args.after_qk_diagnostic))
    destination = args.out/'compile-proof.json'
    save(destination, record)
    try:
        for name in args.case:
            batch, length = CASES[name]
            segments = base._pick_segments(batch, 2, 16)
            tensors, descriptors = compile_tensors(torch, batch, length, segments)
            q,k,v = (tensors[name] for name in ('q','k','v'))
            scratch,maxima = (tensors[name] for name in ('scratch','maxima'))
            kw = dict(segm_output_ptr=scratch, segm_max_ptr=maxima,
                segm_expsum_ptr=tensors['sums'], query_ptr=q, key_cache_ptr=k,
                value_cache_ptr=v, seq_lens_ptr=tensors['lengths'], block_table_ptr=tensors['table'],
                softmax_scale_log2e=128**-.5*base.LOG2E, queries_per_kv=8,
                block_table_stride=tensors['table'].stride(0), query_stride_token=q.stride(0), query_stride_head=q.stride(1),
                k_stride_block=k.stride(0), k_stride_slot=k.stride(1), k_stride_head=k.stride(2),
                v_stride_block=v.stride(0), v_stride_slot=v.stride(1), v_stride_head=v.stride(2),
                segment_stride_token=scratch.stride(0), segment_stride_head=scratch.stride(1), segment_stride_segment=scratch.stride(2),
                max_stride_token=maxima.stride(0), max_stride_head=maxima.stride(1),
                HEAD_SIZE=128, PAGE_SIZE=16, ROWS=8, SEGMENTS=segments, num_warps=1, num_stages=1)
            grid = (batch, 2, segments)
            row = dict(case=name, nominal_length=length, segments=segments, grid=list(grid), descriptors=descriptors)
            compiled = base._gqa_decode_attention.warmup(grid=grid, **kw)
            row['bf16'] = resources(compiled, args.out/(name+'-bf16'))
            kw.update(value_cache_ptr=tensors['v8'], value_scale_ptr=tensors['scale'],
                v_stride_block=tensors['v8'].stride(0), v_stride_slot=tensors['v8'].stride(1), v_stride_head=tensors['v8'].stride(2),
                scale_stride_block=tensors['scale'].stride(0), scale_stride_slot=tensors['scale'].stride(1), scale_stride_head=tensors['scale'].stride(2))
            for label, module in [('frozen_v8', original), ('late_v8', own)]:
                compiled = module.gqa_decode_v8_bf16p.warmup(grid=grid, **kw)
                row[label] = resources(compiled, args.out/(name+'-'+label))
            row['lowering'] = inspect_case(row)
            if args.after_qk_diagnostic:
                row['after_qk'] = inspect_after_qk_case(row)
            record['cases'].append(row)
            save(destination, record)
            del kw,tensors,q,k,v,scratch,maxima
        record['mechanism_passed'] = all(row['lowering']['mechanism_passed'] for row in record['cases'])
        if args.after_qk_diagnostic:
            record['diagnostic_mechanism_passed'] = all(row['after_qk']['diagnostic_mechanism_passed'] for row in record['cases'])
        record['completed'] = True
        save(destination, record)
        accepted = record['diagnostic_mechanism_passed'] if args.after_qk_diagnostic else record['mechanism_passed']
        assert accepted, 'Selected Late-V contract failed: stop before device execution/timing'
    except Exception as error:
        record['errors'].append(repr(error)); save(destination, record)
        raise


def validated_proof(path, chosen, *, after_qk=False):
    record = json.loads(path.read_text())
    assert record['completed'] and record['compile_only'] and record['kernel_launches'] == 0
    assert record['source_commit'] == head() and record['dependency'] == DEPENDENCY
    assert record['source'] == source_inverse(), 'Proof is from different source'
    assert not record['errors'], 'Compile proof has errors: do not launch'
    if after_qk:
        assert record.get('after_qk_diagnostic') is True and record.get('selected_mechanism_contract') == AFTER_QK_CONTRACT
        assert record.get('diagnostic_mechanism_passed') is True, 'After-QK diagnostic contract failed'
    else:
        assert not record.get('after_qk_diagnostic',False), 'Partial proof cannot license default strict mode'
        assert record['mechanism_passed'] is True, 'Strict mechanism failed: do not launch'
    by_case = {row['case']: row for row in record['cases']}
    assert set(chosen) <= by_case.keys(), 'Unproven case requested'
    strict_results = []
    for name in by_case:
        strict = inspect_case(by_case[name])['mechanism_passed']
        assert strict == by_case[name]['lowering']['mechanism_passed'], 'Stored strict result differs from actual IR'
        strict_results.append(strict)
        if after_qk:
            assert inspect_after_qk_case(by_case[name])['diagnostic_mechanism_passed'], 'Rechecked after-QK contract failed'
        else:
            assert strict, 'Rechecked strict lowering failed'
    assert by_case and record['mechanism_passed'] == all(strict_results), 'Top-level strict state differs from actual IR'
    return record


class GuardedKernel:
    """Check actual runtime specialization BEFORE first launch; no new math."""
    def __init__(self, kernel, proof_path, *, after_qk=False):
        self.kernel = kernel
        self.after_qk = after_qk
        self.proof = validated_proof(proof_path, [], after_qk=after_qk)
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
                if self.after_qk:
                    compiled._init_handles()
                    assert compiler_options(compiled) == expected['options'], 'Runtime compiler options differ from proof'
                    assert all(getattr(compiled.metadata,key,None) == value for key,value in expected['metadata'].items()), 'Runtime metadata differs from proof'
                    assert compiled.n_regs == expected['n_regs'] and compiled.n_spills == expected['n_spills'], 'Runtime resources differ from proof'
                self.checked.add(signature)
            return launch(**kwargs)
        return call


def main(args):
    source_inverse()
    if args.run_gpu and args.after_qk_diagnostic:
        assert args.numeric_diagnostic, 'After-QK device work requires explicit numeric-diagnostic too'
    if not args.compile_only and not args.run_gpu:
        print(json.dumps(dict(dependency=DEPENDENCY, candidate='late_v_scale_read',
            cases=args.case, source_inverse=True, compilation='not run', mechanism_passed=False,
            numeric_passed=False, precision_gate=False, official_score=False,
            after_qk_diagnostic=bool(args.after_qk_diagnostic),
            selected_mechanism_contract=AFTER_QK_CONTRACT if args.after_qk_diagnostic else 'strict_after_softmax',
            next_step='Root-only compile-only -> actual IR/ASM gate -> explicit original guarded diagnostics',
            frozen73_preserved=True)))
        return
    if args.compile_only:
        return compile_only(args)
    proof = validated_proof(args.ir_proof,args.case,after_qk=args.after_qk_diagnostic)
    # Execute the independently reversible adaptation of the frozen4a probe.
    # It adds a pre-launch mechanism gate and changes the kernel filename only.
    namespace = {'__file__': str(FROZEN_BENCH), '__name__': 'late_read_frozen_guarded_probe',
                 'GuardedKernel': GuardedKernel}
    exec(compile(adapted_benchmark_tree(), str(FROZEN_BENCH), 'exec'), namespace)
    try:
        namespace['main'](args)
    finally:
        if args.out.is_dir():
            save(args.out/'mechanism-contract.json',dict(
                selected_mechanism_contract=proof.get('selected_mechanism_contract','strict_after_softmax'),
                strict_mechanism_passed=proof['mechanism_passed'],
                after_qk_diagnostic=bool(args.after_qk_diagnostic),
                diagnostic_mechanism_passed=proof.get('diagnostic_mechanism_passed',False),
                compile_proof_sha256=sha(args.ir_proof.read_bytes()),source_commit=head(),
                numeric_gate_unchanged=True,precision_gate=False,official_score=False))


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
    parser.add_argument('--after-qk-diagnostic',action='store_true',
                        help='Separate explicit last-QK-MMA contract; never marks strict after-softmax proof passed')
    args = parser.parse_args()
    args.case = args.case or ['ragged', 'ragged-long', 'steady4k', 'steady16k']
    if (args.compile_only or args.run_gpu) and args.out is None:
        parser.error('--out required for explicit compilation/device work')
    if args.run_gpu and args.ir_proof is None:
        parser.error('--ir-proof required before device work')
    if args.compile_only and args.numeric_diagnostic:
        parser.error('--numeric-diagnostic applies only to guarded device diagnostics')
    if args.run_gpu and args.after_qk_diagnostic and not args.numeric_diagnostic:
        parser.error('--after-qk-diagnostic device work requires explicit --numeric-diagnostic')
    main(args)
