"""Stock-BF16-cache Q/K quantization source bridge; no hook, writer, JIT or device work."""
import ast
import hashlib
from pathlib import Path

from biv150_konly_decode_factory import (KERNEL_SHA, MAIN, Q_OLD, SCORE_NEW, SCORE_OLD,
                                         SOURCE_SHA, original_kernel, sha)
from biv150_konly_decode_quant_reference import FORMAT

NAME = '_gqa_decode_stock_konly_diagnostic'
SUPPORT_SHA = '409ca094e324dea321a9dec8286c19d066ad303113ad879af5a94eaba038a018'
PROLOGUE = ('import triton\nimport triton.language as tl\n'
            'from cjj_konly_decode_support import quantize_bf16_rne\n'
            'LN2 = tl.constexpr(0.6931471805599453)\n\n')
Q_NEW = Q_OLD + '    Q, query_scale = quantize_bf16_rne(Q)\n'
K_FULL = '        K = tl.load(key_cache_ptr + k_offsets)\n'
K_TAIL = '        K = tl.load(key_cache_ptr + k_offsets, mask=column_valid[None, :], other=0.0)\n'
K_QUANT = ('        key_code_rows, key_scale = quantize_bf16_rne(tl.trans(K))\n'
           '        K = tl.trans(key_code_rows)\n')
GUARDS = '''    tl.static_assert(ROWS == 8 and HEAD_SIZE == 128 and PAGE_SIZE == 16 and queries_per_kv == 8)
    tl.static_assert(segm_output_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_max_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_expsum_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(seq_lens_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(block_table_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(query_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(key_cache_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(value_cache_ptr.dtype.element_ty == tl.bfloat16)
'''


def restore(candidate):
    if not candidate.startswith(PROLOGUE):
        raise ValueError('Private diagnostic prologue changed')
    text = candidate[len(PROLOGUE):]
    for old, new, count in ((GUARDS, '', 1), (Q_NEW, Q_OLD, 1),
                            (K_FULL + K_QUANT, K_FULL, 1), (K_TAIL + K_QUANT, K_TAIL, 1),
                            (SCORE_NEW, SCORE_OLD, 2)):
        if text.count(old) != count:
            raise ValueError('Diagnostic transform span missing or duplicated')
        text = text.replace(old, new)
    if text.count('def ' + NAME + '(') != 1:
        raise ValueError('Diagnostic name changed')
    text = text.replace('def ' + NAME + '(', 'def _gqa_decode_attention(', 1)
    if sha(text) != KERNEL_SHA:
        raise ValueError('Original ABI/address/mask/softmax/PV expression changed')
    return text


def build(source):
    original = original_kernel(source)
    support = (Path(__file__).parent / 'kernels/biv150_konly_decode_support.py').read_bytes()
    if hashlib.sha256(support).hexdigest() != SUPPORT_SHA:
        raise ValueError('Frozen ae00 codec helper changed')
    reserved = {'query_scale', 'key_scale', 'key_code_rows', 'score_integer', 'score_query_scaled'}
    if reserved.intersection(n.id for n in ast.walk(ast.parse(original)) if isinstance(n, ast.Name)):
        raise ValueError('Diagnostic temporary collision')
    candidate = original.replace(Q_OLD, Q_NEW, 1)
    candidate = candidate.replace(K_FULL, K_FULL + K_QUANT, 1).replace(K_TAIL, K_TAIL + K_QUANT, 1)
    candidate = candidate.replace(SCORE_OLD, SCORE_NEW)
    candidate = candidate.replace('    seq_idx = tl.program_id(0)\n', GUARDS + '    seq_idx = tl.program_id(0)\n', 1)
    candidate = PROLOGUE + candidate.replace('def _gqa_decode_attention(', 'def ' + NAME + '(', 1)
    assert restore(candidate) == original
    # Parse only: never import this source or its Triton helper in a CPU workflow.
    generated = next(n for n in ast.parse(candidate).body if isinstance(n, ast.FunctionDef))
    parent = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef))
    assert ast.dump(generated.args) == ast.dump(parent.args)
    return candidate, dict(main=MAIN, source_sha256=SOURCE_SHA, original_kernel_sha256=KERNEL_SHA,
                          generated_sha256=sha(candidate), codec=FORMAT, support_sha256=SUPPORT_SHA,
                          same_raw_parameter_schema=True, compiled_typed_abi_verified=False,
                          exact_byte_inverse=True, stock_bf16_key_loads_unchanged=True,
                          original_address_masks_and_v_pv_preserved=True,
                          q_codec_position='once_per_active_split_CTA_after_original_BF16_Q_load',
                          k_codec_position='after_each_original_BF16_K_load; transpose16x128_then_back',
                          full_and_tail_sites=2, logical_q_rows=8, logical_k_rows=16, head_size=128,
                          no_cache_store_added=True, cache_writer_allocator_fallback_changed=False,
                          diagnostic_only=True, throughput_candidate=False,
                          full_prefill_quantization_represented=False,
                          actual_diagnostic_ir_verified=False, actual_m16_codec_verified=False,
                          device_numeric_verified=False, service_installed=False, gpu_executed=False)


def validate_stock_contract(*, payload_format, query_dtype, key_dtype, value_dtype, dimension_stride):
    # A BF16 wrapper alone cannot distinguish an encoded payload. The caller must
    # establish stock provenance; this CPU descriptor check cannot inspect storage.
    if payload_format != 'stock-bf16':
        raise ValueError('Only positively identified stock BF16 cache is supported')
    if (query_dtype, key_dtype, value_dtype) != ('bfloat16',) * 3 or dimension_stride != 1:
        raise ValueError('Audited contiguous-head BF16 operands required')
    return True
