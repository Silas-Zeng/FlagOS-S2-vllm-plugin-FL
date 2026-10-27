"""Private V-only 2D source factory and CPU metadata contract; never imports/JITs Torch/Triton."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

SDK_SHA = '54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f'
HELPER_SHA = '59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1'
NORMALIZER_SHA = '957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1'
NORMALIZED_SHA = 'fa6ddc420a95102d116714fcbea41bc81aa9c09379d68518af65755c015b8744'
FORMAT = 'cjj-v8-token-head-fp32-v1'
NAME = 'kernel_biv150_v8_prefill_private'
GUARDS = '''    tl.static_assert(KV_QUANT_MODE == 0, "Private V8 keeps mode NONE")
    tl.static_assert(not IS_3D, "Private V8 factory is 2D only")
    tl.static_assert(not USE_TD and not USE_TD_QO, "Private V8 pointer path only")
    tl.static_assert(not Q_IS_FP8 and not USE_FP8, "Private V8 uses BF16 Q/output")
    tl.static_assert(query_ptr.dtype.element_ty == tl.bfloat16, "BF16 Q required")
    tl.static_assert(key_cache_ptr.dtype.element_ty == tl.bfloat16, "BF16 K required")
    tl.static_assert(value_cache_ptr.dtype.element_ty == tl.int8, "Real I8 Vcode view required")
    tl.static_assert(v_scale_cache_ptr.dtype.element_ty == tl.float32, "Real FP32 Vscale view required")
'''
VLOAD = '''        # Private V-only format: physical page / slot / KV head, FP32 units.
        private_v_scale_idx = (
            physical_block_idx * stride_vs_blk
            + (seq_offset % BLOCK_SIZE) * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        private_v_scales = tl.load(
            v_scale_cache_ptr + private_v_scale_idx, mask=tile_mask, other=1.0
        )
'''
OLD_PV = '            acc += tl.dot(P.to(V.dtype), V)'
NEW_PV = '            acc += tl.dot((P * private_v_scales[None, :]).to(V.dtype), V)'


class ContractError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ContractError(message)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def pinned(path, expected):
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected, 'Unaudited source: ' + str(path))
    return raw.decode()


def function_source(text, name):
    node = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == name)
    start = min([node.lineno] + [x.lineno for x in node.decorator_list])
    return ''.join(text.splitlines(keepends=True)[start - 1:node.end_lineno])


def normalized_source(sdk, normalizer):
    # Execute only the pinned stdlib stride helper, never make_normalized/install.
    tree = ast.parse(normalizer)
    nodes = [x for x in tree.body if
             isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_STRIDES' for t in x.targets)
             or isinstance(x, ast.FunctionDef) and x.name == '_specialize_strides']
    namespace = {'ast': ast, 're': __import__('re')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-stride-helper>', 'exec'), namespace)
    raw = function_source(sdk, 'kernel_unified_attention')
    old = 'L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)'
    need(raw.count(old) == 1, 'Normalization anchor changed')
    text = raw.replace(old, 'L = tl.zeros([BLOCK_M], dtype=tl.float32)')
    text = text.replace('def kernel_unified_attention(', 'def kernel_biv150_prefill_normalized(', 1)
    text = namespace['_specialize_strides'](text)
    need(digest(text) == NORMALIZED_SHA, 'Audited normalized stride source changed')
    return text


def restore(candidate):
    need(candidate.count(GUARDS) == candidate.count(VLOAD) == candidate.count(NEW_PV) == 1,
         'Private transform is missing or duplicated')
    return candidate.replace(GUARDS, '', 1).replace(VLOAD, '', 1).replace(NEW_PV, OLD_PV, 1).replace(
        'def ' + NAME + '(', 'def kernel_biv150_prefill_stride(', 1)


def build(attention_source, helper_source, normalization_source):
    sdk = pinned(attention_source, SDK_SHA)
    pinned(helper_source, HELPER_SHA)
    normalizer = pinned(normalization_source, NORMALIZER_SHA)
    normalized = normalized_source(sdk, normalizer)
    need(normalized.count(OLD_PV) == 1, 'NONE PV is not unique')
    anchor = '        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)\n'
    need(normalized.count(anchor) == 1, 'V cast anchor changed')
    candidate = normalized.replace('def kernel_biv150_prefill_stride(', 'def ' + NAME + '(', 1)
    header, body = candidate.split('\n):\n', 1)
    candidate = header + '\n):\n' + GUARDS + body
    candidate = candidate.replace(anchor, anchor + VLOAD, 1).replace(OLD_PV, NEW_PV, 1)
    recovered = restore(candidate)
    need(recovered == normalized, 'Non-V source bytes changed')
    need(ast.dump(ast.parse(recovered), include_attributes=False) ==
         ast.dump(ast.parse(normalized), include_attributes=False), 'Non-V AST changed')
    original_args = ast.parse(normalized).body[0].args
    candidate_args = ast.parse(candidate).body[0].args
    need(ast.dump(original_args) == ast.dump(candidate_args), 'Raw kernel ABI changed')
    return dict(source=candidate, normalized_source=normalized,
                metadata=dict(format=FORMAT, sdk_sha256=SDK_SHA, helper_sha256=HELPER_SHA,
                              normalizer_sha256=NORMALIZER_SHA, normalized_sha256=NORMALIZED_SHA,
                              candidate_sha256=digest(candidate), exact_inverse=True, same_raw_abi=True,
                              kv_quant_mode=0, production_installed=False, gpu_executed=False))


@dataclass(frozen=True)
class View:
    dtype: str
    shape: tuple
    strides: tuple
    offset: int  # Elements of this view dtype, never bytes.
    storage_bytes: int
    storage_id: str


def validate_views(views, rows, pages):
    need(type(rows) is type(pages) is int and min(rows, pages) >= 0, 'Invalid view geometry')
    expected = dict(q=('bf16', (rows, 16, 128)), out=('bf16', (rows, 16, 128)),
                    k=('bf16', (pages, 16, 2, 128)), v=('i8', (pages, 16, 2, 128)),
                    scales=('f32', (pages, 16, 2)))
    widths = {'bf16': 2, 'i8': 1, 'f32': 4}
    for name, (dtype, shape) in expected.items():
        v = views[name]
        need(v.dtype == dtype and v.shape == shape, 'Wrong typed private view: ' + name)
        need(type(v.storage_bytes) is int and v.storage_bytes >= 0, 'Invalid storage length')
        need(len(v.strides) == len(shape) and all(type(s) is int and s > 0 for s in v.strides), 'Invalid strides: ' + name)
        need(type(v.offset) is int and v.offset >= 0, 'Invalid element offset: ' + name)
        if all(shape):
            last = v.offset + sum((size - 1) * stride for size, stride in zip(shape, v.strides))
            need((last + 1) * widths[dtype] <= v.storage_bytes, 'View exceeds storage: ' + name)
    k, v, s = (views[x] for x in ('k', 'v', 'scales'))
    need(k.storage_id == v.storage_id == s.storage_id, 'Not one declared V8 physical cache')
    need(k.storage_bytes == v.storage_bytes == s.storage_bytes, 'Cache storage lengths differ')
    pitch = v.strides[0]
    need(pitch >= 16384 and pitch % 256 == 0 and k.strides == (pitch // 2, 256, 128, 1)
         and v.strides == (pitch, 256, 128, 1) and s.strides == (pitch // 4, 2, 1), 'Unknown V8 page/slot/head layout')
    need(k.offset * 2 + 8192 == v.offset and k.offset * 2 + 12288 == s.offset * 4, 'Cache dtype/byte offset mismatch')
    need(k.offset * 2 % 256 == 0, 'Cache base alignment changed')
    for name in ('q', 'out'):
        view = views[name]
        need(view.strides[2] == 1 and view.strides[1] >= 128
             and view.strides[0] >= 16 * view.strides[1], 'Unsupported Q/O overlap or dim stride')
    need(len({views[n].storage_id for n in ('q', 'out', 'k')}) == 3, 'Q/output/cache storage aliases')


def bind_private(raw, views, scale_handle, *, cache_format, rows, pages, num_seqs,
                 max_query_len, sdk_selected_2d, grid):
    """Metadata-only raw binding, with no fallback. Caller must derive views from actual tensors.

    This is not a service hook. It repairs the mode0 dummy scale binding only for
    an explicitly private raw call. Encoded-cache rejection must never retry an old reader.
    """
    need(cache_format == FORMAT, 'Undeclared encoded cache format')
    validate_views(views, rows, pages)
    need(type(num_seqs) is int and num_seqs >= 0 and rows >= 0, 'Invalid sequence/row metadata')
    need(num_seqs > 0 or rows == 0, 'Queries without sequences')
    need(sdk_selected_2d is True, 'Original dispatcher did not select 2D')
    need(raw.get('KV_QUANT_MODE') == 0 and raw.get('IS_3D') is False, 'Private path requires NONE/2D')
    for flag in ('USE_TD', 'USE_TD_QO', 'Q_IS_FP8', 'USE_FP8', 'USE_ALIBI_SLOPES', 'USE_ALIBI_SQRT',
                 'USE_QQ_BIAS', 'USE_SOFTCAP', 'USE_SINKS', 'USE_PER_SEQ_CAUSAL', 'USE_MM_PREFIX'):
        need(raw.get(flag) is False, 'Unsupported private feature: ' + flag)
    need(raw.get('USE_CAUSAL') is True and raw.get('SLIDING_WINDOW') == 0
         and raw.get('CHUNK_LOOKBACK') == -1, 'Unsupported causal/window contract')
    config = (128, 16, 64, 8, 1) if max_query_len > 1 else (16, 2, 32, None, None)
    need(tuple(raw.get(n) for n in ('BLOCK_M', 'BLOCK_Q', 'TILE_SIZE', 'num_warps', 'num_stages')) == config,
         'Changed original SDK tile/launch qualification')
    need(tuple(grid) == (rows // config[1] + num_seqs, 2), 'Dropped per-sequence query tail grid')
    need(raw.get('num_seqs') == num_seqs and raw.get('BLOCK_SIZE') == 16
         and raw.get('HEAD_SIZE') == raw.get('HEAD_SIZE_PADDED') == 128
         and raw.get('num_queries_per_kv') == 8 and raw.get('num_query_heads') == 16, 'Wrong raw geometry')
    for key, view in (('stride_k_cache_', views['k']), ('stride_v_cache_', views['v'])):
        need(tuple(raw.get(key + str(i)) for i in range(4)) == view.strides, 'Wrong typed cache strides')
    for key, view in (('query_stride_', views['q']), ('output_stride_', views['out'])):
        need(tuple(raw.get(key + str(i)) for i in range(2)) == view.strides[:2], 'Wrong Q/O strides')
    need(scale_handle is not None and scale_handle is not raw.get('value_cache_ptr'), 'Mode0 dummy Vscale pointer')
    bound = dict(raw, v_scale_cache_ptr=scale_handle,
                 stride_vs_blk=views['scales'].strides[0], stride_vs_slot=views['scales'].strides[1],
                 stride_vs_head=views['scales'].strides[2])
    need(all(bound[k] == raw[k] for k in raw if k not in ('v_scale_cache_ptr', 'stride_vs_blk', 'stride_vs_slot', 'stride_vs_head')),
         'Private binding changed non-V arguments')
    return bound, dict(no_launch=rows == 0, cache_format=FORMAT, ordinary_bf16_fallback_allowed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attention-source', type=Path, required=True)
    p.add_argument('--helper-source', type=Path, required=True)
    p.add_argument('--normalization-source', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'vllm_fl/patches/biv150_prefill_normalized.py')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = build(a.attention_source, a.helper_source, a.normalization_source)
    a.out.mkdir(parents=False, exist_ok=False)
    (a.out / 'private_v8_prefill.py').write_text(result['source'])
    (a.out / 'manifest.json').write_text(json.dumps(result['metadata'], indent=2) + '\n')
    print(json.dumps(result['metadata']))
