"""Reversible private GQA I8-QK source factory; stdlib only, no JIT/serving hook."""
import ast
import hashlib
from pathlib import Path

MAIN='56f1838192abc940b3fb9accef490022ac125000'
SOURCE_SHA='2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
KERNEL_SHA='b505e712ea21e10563ffb4a14f01a13a9c77fa32308095a80ebbc417b96e69ce'
NAME='_gqa_decode_konly_i8'
PROLOGUE='import triton\nimport triton.language as tl\nfrom cjj_konly_decode_support import quantize_bf16_rne\nLN2 = tl.constexpr(0.6931471805599453)\n\n'
ABI_OLD='    query_ptr,\n    key_cache_ptr,\n'
ABI_NEW='    query_ptr,\n    query_code_ptr,\n    query_scale_ptr,\n    key_scale_ptr,\n    key_cache_ptr,\n'
Q_OLD='    Q = tl.load(query_ptr + query_offsets, mask=row_valid[:, None], other=0.0)\n'
Q_INLINE=Q_OLD+'    Q, query_scale = quantize_bf16_rne(Q)\n'
Q_PRE='    Q = tl.load(query_code_ptr + query_offsets, mask=row_valid[:, None], other=0)\n    query_scale = tl.load(query_scale_ptr + token_idx * 16 + head_idx, mask=row_valid, other=1.0)\n'
SCORE_OLD='        scores = tl.dot(Q, K) * softmax_scale_log2e\n'
SCORE_NEW='''        score_integer = tl.dot(Q, K, tl.zeros([ROWS, PAGE_SIZE], dtype=tl.int32), out_dtype=tl.int32)
        score_query_scaled = score_integer.to(tl.float32) * query_scale[:, None]
        scores = score_query_scaled * key_scale[None, :]
        scores = scores * softmax_scale_log2e
'''
SCALE_FULL='''        key_scale = tl.load(key_scale_ptr + page * 4096 + 1024 + column_slot * 2 + kv_head_idx)
'''
SCALE_TAIL='''        key_scale = tl.load(key_scale_ptr + page * 4096 + 1024 + column_slot * 2 + kv_head_idx,
                            mask=column_valid, other=1.0)
'''
K_FULL='        K = tl.load(key_cache_ptr + k_offsets)\n'
K_TAIL='        K = tl.load(key_cache_ptr + k_offsets, mask=column_valid[None, :], other=0.0)\n'
GUARDS='''    tl.static_assert(ROWS == 8 and HEAD_SIZE == 128 and PAGE_SIZE == 16 and queries_per_kv == 8)
    tl.static_assert(segm_output_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_max_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_expsum_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(seq_lens_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(block_table_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(query_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(query_code_ptr.dtype.element_ty == tl.int8)
    tl.static_assert(query_scale_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(key_scale_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(key_cache_ptr.dtype.element_ty == tl.int8)
    tl.static_assert(value_cache_ptr.dtype.element_ty == tl.bfloat16)
'''


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def original_kernel(text):
    if sha(text)!=SOURCE_SHA: raise ValueError('Unaudited current-main GQA source')
    tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gqa_decode_attention')
    start=min(fn.lineno,*[n.lineno for n in fn.decorator_list])-1
    kernel=''.join(text.splitlines(keepends=True)[start:fn.end_lineno])
    if sha(kernel)!=KERNEL_SHA: raise ValueError('Original kernel bytes changed')
    return kernel


def restore(candidate,mode):
    if mode not in ('inline','preq'): raise ValueError('Unknown Q quantization position')
    if not candidate.startswith(PROLOGUE): raise ValueError('Private prologue changed')
    text=candidate[len(PROLOGUE):]
    for old,new,count in [(GUARDS,'',1),(ABI_NEW,ABI_OLD,1),
                          (Q_INLINE if mode=='inline' else Q_PRE,Q_OLD,1),
                          (SCORE_NEW,SCORE_OLD,2),(SCALE_FULL,'',1),(SCALE_TAIL,'',1)]:
        if text.count(old)!=count: raise ValueError('Private transform span missing/duplicated')
        text=text.replace(old,new)
    text=text.replace('def '+NAME+'(', 'def _gqa_decode_attention(',1)
    if sha(text)!=KERNEL_SHA: raise ValueError('Non-QK source/ABI/math changed')
    return text


def build(text,mode):
    original=original_kernel(text)
    if mode not in ('inline','preq'): raise ValueError('Only inline/preq are supported')
    reserved={'query_code_ptr','query_scale_ptr','key_scale_ptr','query_scale','key_scale',
              'score_integer','score_query_scaled'}
    if reserved.intersection(n.id for n in ast.walk(ast.parse(original)) if isinstance(n,ast.Name)):
        raise ValueError('Temporary variable collision')
    candidate=original.replace(ABI_OLD,ABI_NEW,1).replace(Q_OLD,Q_INLINE if mode=='inline' else Q_PRE,1)
    candidate=candidate.replace(K_FULL,K_FULL+SCALE_FULL,1).replace(K_TAIL,K_TAIL+SCALE_TAIL,1)
    candidate=candidate.replace(SCORE_OLD,SCORE_NEW)
    candidate=candidate.replace('    seq_idx = tl.program_id(0)\n',GUARDS+'    seq_idx = tl.program_id(0)\n',1)
    candidate=PROLOGUE+candidate.replace('def _gqa_decode_attention(', 'def '+NAME+'(',1)
    assert restore(candidate,mode)==original
    compile(candidate,'<private-gqa-source-syntax-only>','exec')
    tree=ast.parse(candidate)
    pvs=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n)=='tl.dot(probability.to(V.dtype), V)']
    assert len(pvs)==2
    return candidate,dict(mode=mode,main=MAIN,source_sha256=SOURCE_SHA,generated_sha256=sha(candidate),
        exact_byte_inverse=True,codec='cjj-konly-rn32-rne-v2',
        support_sha256=hashlib.sha256((Path(__file__).parent/'kernels/biv150_konly_decode_support.py').read_bytes()).hexdigest(),
        logical_q_rows=8,qk_accumulator='int32',v_pv_dtype='bfloat16',
        original_softmax_pv_reducer_preserved=True,full_and_tail_qk_sites=2,unchanged_pv_sites=2,
        q_quantization_position='inside_each_active_split_CTA' if mode=='inline' else 'one_prep_per_token_and_kvhead',
        k_byte_strides=[16384,256,128,1],v_bf16_strides=[8192,256,128,1],
        scale_f32_strides=[4096,2,1],scale_base_f32_elements=1024,
        allocated_page_bytes=16384,capacity_reclaimed=False,actual_decoder_ir_verified=False,
        gpu_executed=False,precision_gate=False,service_installed=False)


def validate_binding(*,mode,k_strides,v_strides,scale_strides,q_strides,source_storage,destination_storage,format_id):
    if mode not in ('inline','preq'): raise ValueError('Unknown mode')
    if tuple(k_strides)!=(16384,256,128,1) or tuple(v_strides)!=(8192,256,128,1):
        raise ValueError('K byte strides and V BF16 strides have different units')
    if tuple(scale_strides)!=(4096,2,1) or tuple(q_strides)!=(2048,128,1):
        raise ValueError('Private scale/query layout required')
    if source_storage is None or destination_storage is None or source_storage==destination_storage:
        raise ValueError('Initial encoding requires separate source and destination storage')
    if format_id!='cjj-konly-rn32-rne-v2': raise ValueError('Payload cannot be inferred from BF16 dtype metadata')
    return True
