"""Falsify source/probe/lowering guards using only stdlib CPU and saved b8 bytes."""
import ast
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import biv150_v8_late_read_checks as checks
import bench_biv150_v8_late_read as probe
import check_biv150_v8_bf16p_cpu as frozen_checks

EVIDENCE = checks.ROOT/'docs/evidence/v8-late-read-20261027'


def originals():
    manifest = json.loads((EVIDENCE/'b8-ir-manifest.json').read_text())
    with tarfile.open(EVIDENCE/'b8-ir-originals.tar.gz') as archive:
        entries = archive.getmembers()
        assert len(entries) == len(manifest['files']) == 19
        files = {item.name: archive.extractfile(item).read() for item in entries}
    for item in manifest['files']:
        raw = files[item['path']]
        assert len(raw) == item['bytes'] and checks.sha(raw) == item['sha256']
    prefix = 'workspace/experiments/minicpm-s2/cjj-v8-bf16p-ragged-v2-20261027/compiler-ir/d2481b9f7db900083e1fdbc307add5f2f66b5a34fe3b1ef6c577db69e9c9e760/'
    return {name: files[prefix+'actual.'+name].decode() for name in ('ttgir', 'llir', 'asm')}


def synthetic_late_parser_fixture(old):
    """Reorder TEXT for a positive parser fixture; NOT executable/proven GPU IR.

    In particular physical register allocation is not repaired. This fixture
    only checks that the guard recognizes its intended ordering and rejects
    meaningful mutants, rather than being a guard that always returns false.
    """
    result = dict(old)
    lines = old['llir'].splitlines()
    groups = checks.matrix_groups(lines, '@llvm.bi.matrix.mad.f32x4.bf16x4(')
    for path in (1, 0):
        qk, pv = groups[path]
        begin = 0 if path == 0 else groups[path-1][1][-1]+1
        selected = [i for i in range(begin, pv[0]) if 'extractelement <4 x i8>' in lines[i]]
        end = max(i for i in range(qk[-1], pv[0]) if '@llvm.nvvm.ex2.' in lines[i])
        moved = [lines[i] for i in selected]
        lines = [line for i,line in enumerate(lines) if i not in selected[:]]
        insert = end+1-sum(i <= end for i in selected)
        lines[insert:insert] = moved
    result['llir'] = '\n'.join(lines)+'\n'
    lines = old['asm'].splitlines()
    groups = checks.matrix_groups(lines, 'ml_matrix_mad_f32_bf16_m16n16k16 ')
    records = checks.asm_records(old['asm'])
    source_paths = checks.paths(checks.producer(checks.FROZEN_KERNEL.read_bytes()))
    for path in (1, 0):
        qk, pv = groups[path]
        begin = 0 if path == 0 else groups[path-1][1][-1]+1
        load = next(n for n in source_paths[path] if checks.assignment(n, 'V'))
        selected = [r for r in records if begin < r['line'] <= pv[0] and (
            r['opcode'] in ('ml_cvt_i16_i8', 'ml_cvt_bf16_i16') or (
                r['opcode'] in ('ml_bfe2_b32', 'ml_and_b32') and r['source_file'] == checks.FROZEN_KERNEL.name
                and load.lineno <= r['source_line'] <= load.end_lineno))]
        end = max(i for i in range(qk[-1], pv[0]) if 'ml_exp_f32 ' in lines[i])
        deleted = {r['line']-1 for r in selected}
        moved = []
        for r in selected:
            moved += ['\t.loc\t1 '+str(r['source_line'])+' 1', lines[r['line']-1]]
        moved.append('\t.loc\t1 0 0')
        lines = [line for i,line in enumerate(lines) if i not in deleted]
        insert = end+1-sum(i <= end for i in deleted)
        lines[insert:insert] = moved
    result['asm'] = '\n'.join(lines)+'\n'
    return result


def check():
    rejected = []
    def reject(label, action):
        try:
            action()
        except (AssertionError, StopIteration):
            rejected.append(label)
        else:
            raise AssertionError('Mutant accepted: '+label)
    source = checks.CANDIDATE_KERNEL.read_text()
    for label, before, after in [
        ('pv_math', 'probability * value_scale[None, :]', 'probability + value_scale[None, :]'),
        ('tail_zero', 'mask=column_valid[:, None], other=0.0', 'mask=column_valid[:, None], other=1.0'),
        ('fake_dependency', 'V = tl.load(value_cache_ptr + v_offsets)',
         'V = tl.load(value_cache_ptr + v_offsets) + running_sum[0]*0'),
        ('cache_write', '        running_max = new_max',
         '        tl.store(value_cache_ptr + v_offsets, V)\n        running_max = new_max'),
        ('qk_dtype', 'tl.dot(Q, K)', 'tl.dot(Q.to(tl.float32), K)'),
        ('writer_alias', 'import encode_cache_v, append_token_kv', 'import encode_cache_v, append_token_kv as other'),
        ('page_step', 'position += PAGE_SIZE', 'position += PAGE_SIZE*2'),
        ('module_side_effect', 'import triton\n', 'import triton\nunexpected = 0\n')]:
        assert before in source
        reject(label, lambda b=before,a=after: checks.source_inverse(source.replace(b,a).encode()))
    checks.adapted_benchmark_tree()
    old = originals()
    options = dict(source=checks.FROZEN_KERNEL.read_bytes(), source_name=checks.FROZEN_KERNEL.name)
    old_proof = checks.lowering_check(old['ttgir'], old['llir'], old['asm'], previous=old, **options)
    assert not old_proof['mechanism_passed']
    assert len(old_proof['reasons']) == 4 and all('before softmax' in r for r in old_proof['reasons'])
    late = synthetic_late_parser_fixture(old)
    positive = checks.lowering_check(late['ttgir'], late['llir'], late['asm'], previous=old, **options)
    assert positive['mechanism_passed'], positive['reasons']
    for label, mutated in [
        ('early_llvm_extract', dict(late, llir=old['llir'])),
        ('early_asm_unpack', dict(late, asm=old['asm'])),
        ('wrong_mma_dtype', {name: text.replace('bf16', 'i8') for name,text in late.items()}),
        ('missing_asm', dict(late, asm='')),
        ('extra_store', dict(late, asm=late['asm']+'\n\tml_slb_store_b16 v1, v2, noreg, #0, null\n')),
        ('missing_barrier', dict(late, asm=late['asm'].replace('sl_barrier', 'unsupported_sync', 1)))]:
        proof = checks.lowering_check(mutated['ttgir'], mutated['llir'], mutated['asm'], previous=old, **options)
        assert not proof['mechanism_passed'], label
        rejected.append(label)
    # Existing4a fixture falsifiers and baseline/default diagnostic policy are
    # reused, not loosened for this new source-scheduling experiment.
    baseline = frozen_checks.numeric_diagnostic_check()
    writer = frozen_checks.falsification_check()
    compile_fn = next(n for n in ast.parse(Path(probe.__file__).read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'compile_only')
    assert not any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute)
                   and n.value.attr == 'gqa_decode_v8_bf16p' for n in ast.walk(compile_fn))
    warmups = [n for n in ast.walk(compile_fn) if isinstance(n, ast.Call) and
               isinstance(n.func, ast.Attribute) and n.func.attr == 'warmup']
    assert len(warmups) == 2  # one baseline site, one shared frozen/late site
    # Mock only artifact IO, so actual-runtime specialization mismatch and
    # changed IR are shown to fail before invoking even a fake launcher.
    old_loader = probe.validated_proof
    allowed_ir = dict(ttgir='typed', llir='llvm', asm='assembly')
    allowed = dict(hash='allowed', files={name: dict(sha256=checks.sha(text.encode()))
                                         for name,text in allowed_ir.items()})
    probe.validated_proof = lambda path, chosen: dict(cases=[dict(late_v8=allowed)])
    class FakeKernel:
        launches = warmups = 0
        compiled_hash = 'allowed'
        ir = allowed_ir
        def warmup(self, **kwargs):
            self.warmups += 1
            return SimpleNamespace(hash=self.compiled_hash, asm=self.ir)
        def __getitem__(self, grid):
            def launch(**kwargs):
                self.launches += 1
            return launch
    try:
        kernel = FakeKernel()
        guarded = probe.GuardedKernel(kernel, None)
        guarded[(1, 2, 16)](num_warps=1)
        guarded[(1, 2, 16)](num_warps=1)
        assert kernel.launches == 2 and kernel.warmups == 1
        wrong = FakeKernel(); wrong.compiled_hash = 'unproven'
        reject('unproven_runtime_hash', lambda: probe.GuardedKernel(wrong, None)[(1, 2, 16)](num_warps=1))
        assert wrong.launches == 0
        changed = FakeKernel(); changed.ir = dict(allowed_ir, asm='changed')
        reject('changed_runtime_ir', lambda: probe.GuardedKernel(changed, None)[(1, 2, 16)](num_warps=1))
        assert changed.launches == 0
    finally:
        probe.validated_proof = old_loader
    return dict(source=checks.source_inverse(), frozen_benchmark_ast_inverse=True,
        archive=dict(files=19, bytes=966065, archive_bytes=(EVIDENCE/'b8-ir-originals.tar.gz').stat().st_size,
                     sha256=checks.sha((EVIDENCE/'b8-ir-originals.tar.gz').read_bytes())),
        original_b8_lowering_rejected=old_proof,
        synthetic_parser_fixture_passed=True, synthetic_fixture_is_executable_ir=False,
        rejected_mutants=rejected, numeric_policy=baseline, writer_codec=writer,
        compile_only_has_no_kernel_launch_site=True, gpu_executed=False, jit_executed=False,
        actual_specialization_hash_and_ir_checked_before_launch=True,
        new_candidate_compiled=False, mechanism_passed=False, precision_gate=False, official_score=False)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
