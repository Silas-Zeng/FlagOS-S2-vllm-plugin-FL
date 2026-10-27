"""Falsify source/probe/lowering guards using only stdlib CPU and saved b8 bytes."""
import ast
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import math
import tempfile

import biv150_v8_late_read_checks as checks
import bench_biv150_v8_late_read as probe
import check_biv150_v8_bf16p_cpu as frozen_checks

EVIDENCE = checks.ROOT/'docs/evidence/v8-late-read-20261027'


class MetadataTensor:
    """Independent CPU view arithmetic, no storage allocation or GPU imports."""
    sizes = dict(bfloat16=2, int8=1, float32=4, int32=4)
    def __init__(self, shape, dtype, stride=None, offset=0, storage=None):
        self.shape, self.kind = tuple(shape), dtype
        if stride is None:
            reverse, step = [], 1
            for size in reversed(shape):
                reverse.append(step); step *= size
            stride = tuple(reversed(reverse))
        self.steps, self.offset = tuple(stride), offset
        self.storage = math.prod(shape)*self.sizes[dtype] if storage is None else storage
        end = (offset+sum((n-1)*s for n,s in zip(shape,stride))+1)*self.sizes[dtype]
        assert end <= self.storage
    @property
    def dtype(self):
        return 'torch.'+self.kind
    def stride(self, dim=None):
        return self.steps if dim is None else self.steps[dim]
    def storage_offset(self):
        return self.offset
    def is_contiguous(self):
        step = 1
        for n,s in zip(reversed(self.shape),reversed(self.steps)):
            if n > 1 and s != step:
                return False
            step *= n
        return True
    def unbind(self, dim):
        shape = self.shape[:dim]+self.shape[dim+1:]
        strides = self.steps[:dim]+self.steps[dim+1:]
        return tuple(MetadataTensor(shape,self.kind,strides,self.offset+i*self.steps[dim],self.storage)
                     for i in range(self.shape[dim]))
    def view(self, dtype):
        old, new = self.sizes[self.kind], self.sizes[dtype]
        assert self.steps[-1] == 1 and self.shape[-1]*old % new == 0
        assert all(s*old % new == 0 for s in self.steps[:-1])
        shape = self.shape[:-1]+(self.shape[-1]*old//new,)
        stride = tuple(s*old//new for s in self.steps[:-1])+(1,)
        return MetadataTensor(shape,dtype,stride,self.offset*old//new,self.storage)
    def reshape(self, size):
        assert size == -1 and self.is_contiguous()
        return MetadataTensor((math.prod(self.shape),),self.kind,(1,),self.offset,self.storage)
    def narrow(self, dim, start, length):
        assert 0 <= start <= self.shape[dim]-length
        shape = list(self.shape); shape[dim] = length
        return MetadataTensor(shape,self.kind,self.steps,self.offset+start*self.steps[dim],self.storage)


class MetadataTorch:
    bfloat16, int8, float32, int32 = 'bfloat16', 'int8', 'float32', 'int32'
    allocated_bytes = 0
    def empty(self, *shape, device, dtype):
        assert device == 'cuda'  # describes intended constructor, never accesses it
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        tensor = MetadataTensor(shape,dtype)
        self.allocated_bytes += tensor.storage
        return tensor
    def as_strided(self, tensor, shape, stride, offset):
        return MetadataTensor(shape,tensor.kind,stride,offset,tensor.storage)


def descriptor_check():
    records = []
    for name, batch, length, segments in [('ragged',8,257,16), ('ragged-long',8,257,16),
                                        ('steady4k',64,4096,2), ('steady16k',32,16384,4)]:
        metadata = MetadataTorch()
        tensors,plan = probe.compile_tensors(metadata,batch,length,segments)
        assert metadata.allocated_bytes == plan['unique_allocation_bytes']
        assert len(tensors['k'].shape) == len(tensors['v8'].shape) == 4
        assert len(tensors['scale'].shape) == 3
        assert not tensors['k'].is_contiguous() and not tensors['v8'].is_contiguous()
        assert not tensors['scale'].is_contiguous()
        assert tensors['scratch'].is_contiguous() == (segments == 16)
        assert tensors['maxima'].is_contiguous() == (segments == 16)
        assert tensors['scratch'].stride() == (32768,2048,128,1)
        assert tensors['maxima'].stride() == (256,16,1)
        assert tensors['k'].storage_offset() == 0 and tensors['v'].storage_offset() == 4096
        assert tensors['v8'].storage_offset() == 8192 and tensors['scale'].storage_offset() == 3072
        records.append(dict(case=name, descriptor_plan=plan, no_storage_or_gpu_access=True))
    return records


def actual_after_qk_check():
    """Read-only original3f1 artifacts: strict stays false; test new hypothesis."""
    directory = EVIDENCE/'compile-v1'
    manifest = json.loads((directory/'archive-manifest.json').read_text())
    files = {}
    for archive in manifest['archives']:
        raw = (directory/archive['path']).read_bytes()
        assert len(raw) == archive['bytes'] and checks.sha(raw) == archive['sha256']
        with tarfile.open(directory/archive['path']) as tar:
            files.update({member.name:tar.extractfile(member).read() for member in tar.getmembers()})
    for item in manifest['original']['files']:
        assert checks.sha(files[item['relative']]) == item['sha256'] and len(files[item['relative']]) == item['bytes']
    def ir(label):
        return {name:files[f'out/ragged-{label}/actual.{name}'].decode() for name in ('ttgir','llir','asm')}
    late,old = ir('late_v8'),ir('frozen_v8')
    original = json.loads(files['out/compile-proof.json'])
    assert original['source_commit'] == '3f1a2586cecc71c3badefc165a9bf9b2321a6db3'
    assert original['mechanism_passed'] is False and original['kernel_launches'] == 0
    assert original['source'] == checks.source_inverse()
    strict = checks.lowering_check(late['ttgir'],late['llir'],late['asm'],previous=old)
    assert strict['mechanism_passed'] is False and len(strict['reasons']) == 2
    partial = checks.after_qk_check(late['ttgir'],late['llir'],late['asm'],previous=old)
    assert partial['diagnostic_mechanism_passed'] is True and partial['strict_mechanism_passed'] is False
    source_options = dict(source=checks.FROZEN_KERNEL.read_bytes(),source_name=checks.FROZEN_KERNEL.name)
    assert not checks.after_qk_check(old['ttgir'],old['llir'],old['asm'],previous=old,**source_options)['diagnostic_mechanism_passed']
    rejected = ['original_before_qk']
    # A one-instruction move is a parser mutant, not executable GPU IR.
    records = checks.asm_records(late['asm'])
    for path_index,key in [(0,'v_packed_loads'),(1,'scale_loads'),(0,'v_expand'),(1,'v_casts')]:
        path = strict['evidence']['asm_paths'][path_index]
        selected = path[key][0]
        rec = next(r for r in records if r['line'] == selected)
        lines = late['asm'].splitlines()
        moved = lines.pop(selected-1)
        insert = path['qk'][0]-1
        lines[insert:insert] = [f'\t.loc\t1 {rec["source_line"]} 1',moved,'\t.loc\t1 0 0']
        proof = checks.after_qk_check(late['ttgir'],late['llir'],'\n'.join(lines),previous=old)
        assert not proof['diagnostic_mechanism_passed'], key
        rejected.append(f'early_asm_path{path_index}_{key}')
    llvm_lines = late['llir'].splitlines()
    lp = strict['evidence']['llvm_paths'][0]
    moved = llvm_lines.pop(lp['v_packed_loads'][0]-1)
    llvm_lines.insert(lp['qk'][0]-1,moved)
    assert not checks.after_qk_check(late['ttgir'],'\n'.join(llvm_lines),late['asm'],previous=old)['diagnostic_mechanism_passed']
    rejected.append('early_llvm_load')
    first_store = next(r['text'] for r in records if r['opcode']=='ml_slb_store_b16')
    assert '#0' in first_store
    changed_offset = late['asm'].replace(first_store,first_store.replace('#0','#2',1),1)
    for label,assembly in [('barrier',late['asm'].replace('sl_barrier','wrong_barrier',1)),
                           ('store_offset_same_opcode_count',changed_offset)]:
        assert not checks.after_qk_check(late['ttgir'],late['llir'],assembly,previous=old)['diagnostic_mechanism_passed']
        rejected.append(label)
    # A new partial compile record cannot be consumed through default strict mode.
    with tempfile.TemporaryDirectory(prefix='late-qk-policy-') as directory:
        path = Path(directory)/'proof.json'
        path.write_text(json.dumps(dict(completed=True,compile_only=True,kernel_launches=0,
            source_commit=probe.head(),dependency=checks.DEPENDENCY,source=checks.source_inverse(),
            errors=[],mechanism_passed=False,after_qk_diagnostic=True,
            selected_mechanism_contract=checks.AFTER_QK_CONTRACT,diagnostic_mechanism_passed=True,cases=[])))
        try:probe.validated_proof(path,[])
        except AssertionError:rejected.append('partial_proof_in_default_strict_mode')
        else:raise AssertionError('Default strict mode accepted a partial proof')
    return dict(original_raw_compile_head=original['source_commit'],original_raw_strict_false=True,
        original_raw_proof_sha256=checks.sha(files['out/compile-proof.json']),
        source_and_descriptors_match=original['source']==checks.source_inverse() and
          original['cases'][0]['descriptors']==checks.runtime_descriptors(8,257,16),
        partial_mechanical_predicate=partial,rejected_mutants=rejected,
        legacy_options_not_serialized=True,fresh_explicit_compile_proof_required=True,
        local_gpu_or_jit_executed=False)


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
    probe.validated_proof = lambda path, chosen, **kwargs: dict(cases=[dict(late_v8=allowed)])
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
        options = dict(enable_fp_fusion=True,enable_reflect_ftz=True,
            default_dot_input_precision='tf32',maxnreg=None,backend_name='corex',arch='sm71',
            sanitize_overflow=True,debug=False)
        metadata = dict(shared=4096,num_warps=1,num_stages=1,use_sme=79,warp_size=64,triton_version='3.6.0')
        partial_allowed = dict(allowed,options=options,metadata=metadata,n_regs=137,n_spills=0)
        probe.validated_proof = lambda path,chosen,**kwargs: dict(cases=[dict(late_v8=partial_allowed)])
        class OptionKernel(FakeKernel):
            def __init__(self, *, option_updates=None, metadata_updates=None, n_regs=137):
                self.options = dict(options,**(option_updates or {}))
                self.metadata = dict(metadata,**(metadata_updates or {}))
                self.n_regs = n_regs
            def warmup(self, **kwargs):
                self.warmups += 1
                return SimpleNamespace(hash='allowed',asm=allowed_ir,n_regs=self.n_regs,n_spills=0,
                    metadata=SimpleNamespace(**self.options,**self.metadata),_init_handles=lambda:None)
        partial = OptionKernel()
        probe.GuardedKernel(partial,None,after_qk=True)[(1,2,16)](num_warps=1)
        assert partial.launches == 1
        for label,kernel in [('changed_runtime_option',OptionKernel(option_updates=dict(enable_fp_fusion=False))),
                             ('changed_runtime_metadata',OptionKernel(metadata_updates=dict(num_warps=2))),
                             ('changed_runtime_resource',OptionKernel(n_regs=143))]:
            reject(label,lambda k=kernel:probe.GuardedKernel(k,None,after_qk=True)[(1,2,16)](num_warps=1))
            assert kernel.launches == 0
    finally:
        probe.validated_proof = old_loader
    return dict(source=checks.source_inverse(), frozen_benchmark_ast_inverse=True,
        actual_after_qk=actual_after_qk_check(),
        runtime_descriptor_cpu_view_arithmetic=descriptor_check(),
        archive=dict(files=19, bytes=966065, archive_bytes=(EVIDENCE/'b8-ir-originals.tar.gz').stat().st_size,
                     sha256=checks.sha((EVIDENCE/'b8-ir-originals.tar.gz').read_bytes())),
        original_b8_lowering_rejected=old_proof,
        synthetic_parser_fixture_passed=True, synthetic_fixture_is_executable_ir=False,
        rejected_mutants=rejected, numeric_policy=baseline, writer_codec=writer,
        compile_only_has_no_kernel_launch_site=True, gpu_executed=False, jit_executed=False,
        actual_specialization_hash_and_ir_checked_before_launch=True,
        owner_compiled_3f1=True, this_cpu_check_compiled_candidate=False,
        current_after_qk_entrypoint_device_tested=False,
        mechanism_passed=False, precision_gate=False, official_score=False)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
