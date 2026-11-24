"""Private mixed decode readers: exact source mapping, no framework import or hook."""
import ast
import hashlib

import biv150_konly_decode_factory as dependency

MAIN = 'dac1b8fb48e7983e48283032efeee9b592c5fbe5'
REDUCER_SHA = '3685996879558c702d535bc62fea2dcd32150710c2286734797ada35450e75c5'
BF16_PREFIX = 'import triton\nimport triton.language as tl\nLN2 = tl.constexpr(0.6931471805599453)\n\n'
MODES = ('bf16', 'k8')
NAMES = dict(bf16='_mixed_bf16_decode', k8='_mixed_konly_i8_decode')
OLD_NAMES = dict(bf16='_gqa_decode_attention', k8=dependency.NAME)
META_ABI = '    query_start_len_ptr,\n    NUM_SEQS: tl.constexpr,\n    TOTAL_QUERY_TOKENS: tl.constexpr,\n'
PRODUCER_GUARD = '''    tl.static_assert(query_start_len_ptr.dtype.element_ty == tl.int32)
    if seq_idx >= NUM_SEQS:
        return
    q_start = tl.load(query_start_len_ptr + seq_idx).to(tl.int64)
    q_end = tl.load(query_start_len_ptr + seq_idx + 1).to(tl.int64)
    if ((q_end - q_start) != 1) | (q_start < 0) | (q_end > TOTAL_QUERY_TOKENS):
        return
'''
BF16_GUARD = '''    tl.static_assert(ROWS == 8 and HEAD_SIZE == 128 and PAGE_SIZE == 16 and queries_per_kv == 8)
    tl.static_assert(query_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(key_cache_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(value_cache_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(seq_lens_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(block_table_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(segm_output_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_max_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_expsum_ptr.dtype.element_ty == tl.float32)
'''
REDUCER_GUARD = '''    tl.static_assert(query_start_len_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(seq_lens_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(output_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(segm_output_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_max_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_expsum_ptr.dtype.element_ty == tl.float32)
    if token_idx >= NUM_SEQS:
        return
    q_start = tl.load(query_start_len_ptr + token_idx).to(tl.int64)
    q_end = tl.load(query_start_len_ptr + token_idx + 1).to(tl.int64)
    if ((q_end - q_start) != 1) | (q_start < 0) | (q_end > TOTAL_QUERY_TOKENS):
        return
'''


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def span(text, name):
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([fn.lineno] + [n.lineno for n in fn.decorator_list]) - 1
    return ''.join(text.splitlines(keepends=True)[start:fn.end_lineno])


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Exact source mapping span changed: ' + old[:90])
    return text.replace(old, new, 1)


def mapped_producer(original, mode):
    if mode not in MODES:
        raise ValueError('Explicit BF16 or packed K8 arm required')
    text = replace_once(original, 'def ' + OLD_NAMES[mode] + '(', 'def ' + NAMES[mode] + '(')
    text = replace_once(text, '    query_ptr,\n', '    query_ptr,\n' + META_ABI)
    text = replace_once(text, '    seq_idx = tl.program_id(0)\n',
                        '    seq_idx = tl.program_id(0)\n' + (BF16_GUARD if mode == 'bf16' else '') + PRODUCER_GUARD)
    # token_idx must remain the compact sequence row for all partial stores.
    # Only the Q address uses the actual token offset read from cu_q.
    return replace_once(text, '        token_idx * query_stride_token\n', '        q_start * query_stride_token\n')


def restore_producer(text, mode):
    if mode not in MODES:
        raise ValueError('Unknown reader arm')
    text = replace_once(text, 'def ' + NAMES[mode] + '(', 'def ' + OLD_NAMES[mode] + '(')
    text = replace_once(text, META_ABI, '')
    text = replace_once(text, (BF16_GUARD if mode == 'bf16' else '') + PRODUCER_GUARD, '')
    text = replace_once(text, '        q_start * query_stride_token\n', '        token_idx * query_stride_token\n')
    if mode == 'bf16' and sha(text) != dependency.KERNEL_SHA:
        raise ValueError('BF16 control math or compact partial addresses changed')
    if mode == 'k8':
        dependency.restore(dependency.PROLOGUE + text, 'inline')
    return text


def mapped_reducer(original):
    text = replace_once(original, 'def _reduce_partials(', 'def _reduce_mixed_partials(')
    text = replace_once(text, '    seq_lens_ptr,\n', '    seq_lens_ptr,\n' + META_ABI)
    text = replace_once(text, '    token_idx = tl.program_id(0)\n', '    token_idx = tl.program_id(0)\n' + REDUCER_GUARD)
    return replace_once(text, 'output_ptr + token_idx * output_stride_token', 'output_ptr + q_start * output_stride_token')


def restore_reducer(text):
    text = replace_once(text, 'def _reduce_mixed_partials(', 'def _reduce_partials(')
    text = replace_once(text, META_ABI, '')
    text = replace_once(text, REDUCER_GUARD, '')
    text = replace_once(text, 'output_ptr + q_start * output_stride_token', 'output_ptr + token_idx * output_stride_token')
    if sha(text) != REDUCER_SHA:
        raise ValueError('Original reducer math or compact partial addresses changed')
    return text


def build(main_source):
    original = dependency.original_kernel(main_source)
    original_reducer = span(main_source, '_reduce_partials')
    if sha(original_reducer) != REDUCER_SHA:
        raise ValueError('Original reducer bytes changed')
    reserved = {'q_start', 'q_end', 'query_start_len_ptr', 'NUM_SEQS', 'TOTAL_QUERY_TOKENS'}
    for source in (original, original_reducer):
        tree = ast.parse(source)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        if names & reserved:
            raise ValueError('Mixed metadata temporary collides with original source')
    packed, inherited = dependency.build(main_source, 'inline')
    packed_fn = packed[len(dependency.PROLOGUE):]
    reducer = mapped_reducer(original_reducer)
    assert restore_reducer(reducer) == original_reducer
    modules, proofs = {}, {}
    for mode, base in (('bf16', original), ('k8', packed_fn)):
        producer = mapped_producer(base, mode)
        restored = restore_producer(producer, mode)
        assert restored == base
        if mode == 'k8':
            assert dependency.restore(dependency.PROLOGUE + restored, 'inline') == original
        prefix = BF16_PREFIX if mode == 'bf16' else dependency.PROLOGUE
        modules[mode] = prefix + producer + '\n\n' + reducer + '\n'
        compile(modules[mode], '<mixed-source-syntax-only>', 'exec')
        tree = ast.parse(modules[mode])
        pvs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n) == 'tl.dot(probability.to(V.dtype), V)']
        assert len(pvs) == 2
        proofs[mode] = dict(generated_sha256=sha(modules[mode]), producer_inverse_sha256=sha(restored),
            exact_producer_mapping_inverse=True, exact_shared_reducer_inverse=True,
            partial_index='compact sequence row, never q_start', query_and_output_index='cu_q[sequence]',
            metadata_difference_in_int64=True, producer_reducer_decode_guards_before_data_access=True,
            original_softmax_pv_accumulator_and_reduction_order_preserved=True,
            typed_compiled_abi_verified=False, actual_ir_verified=False, loaded_resources_known=False,
            gpu_numeric_verified=False, native_accumulator_component_independence_assumed=False)
    return modules, dict(main=MAIN, main_source_sha256=dependency.SOURCE_SHA, original_kernel_sha256=dependency.KERNEL_SHA,
        original_reducer_sha256=REDUCER_SHA, arms=proofs, inherited_k8_contract=inherited,
        num_seqs_max=64, total_query_capacity_from_shape=True, private_only=True,
        kv_writer_or_allocator_modified=False, prefill_reader_modified=False,
        service_installed=False, gpu_executed=False, performance_claim=False, precision_gate=False,
        future_coordination='Paired prefill reader must skip qlen==1 rows; disjoint writes, no BF16 fallback on encoded payload')
