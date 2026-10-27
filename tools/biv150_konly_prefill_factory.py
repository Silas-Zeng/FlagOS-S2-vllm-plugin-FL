"""CPU-only, reversible K-only I8 QK readers. No serving hook or device entry point."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MAIN = '56f1838192abc940b3fb9accef490022ac125000'
DEPENDENCY = 'ae00f58629ad85850fec1b8b8d2da28757af5f8b'
FORMAT = 'cjj-konly-rn32-rne-v2'
PINS = {
    'sdk.py': '54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f',
    'helpers.py': '59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1',
    'normalizer.py': '957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1',
    'codec-support-ae00.py': '409ca094e324dea321a9dec8286c19d066ad303113ad879af5a94eaba038a018',
    'biv150_konly_decode_quant_reference.py': '38807ab17bf28e64afee6fb581af92d4e715856d954dc42db4fdd1dd6d91bc18',
    'biv150_konly_i8_qk_reference.py': '63d8f901b3648dad52f9dceab251e1d451def2aaa96f2bc30bd0aaff102e64e3',
}
ARCHIVE_SHA = '9c5b271774ea17e9434a5e1f305345555a537af31ac74126f2a73a2e80e55733'
PROFILES = {
    'normalized2d128': (False, 128, 16, 64, 8, 1, None),
    'q32': (False, 256, 32, 64, 16, 1, 1),
    'multi2': (False, 256, 32, 64, 16, 1, 2),
    # Actual SDK host leaves these launch options unspecified; do not invent them.
    'raw3d': (True, 16, 2, 16, None, None, None),
}
ABI_OLD = '    query_ptr,\n    key_cache_ptr,\n'
ABI_NEW = ABI_OLD + '    key_code_scale_ptr,\n'
Q_ANCHOR = '    block_table_offset = seq_idx * block_table_stride'
Q_ADD = '    QCode, query_code_scale = quantize_bf16_rne(Q)\n\n'
K_OLD = '        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)\n'
K_NEW = '''        K = K_load
        key_code_scale = _cjj_konly_scale(
            key_code_scale_ptr, physical_block_idx, seq_offset % BLOCK_SIZE,
            kv_head_idx, tile_mask)
'''
S_OLD = '''        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (score_scale * k_token_head_scales[None, :])
        else:
            S += score_scale * tl.dot(Q, K)
'''
S_NEW = '''        score_integer = tl.dot(QCode, K,
            tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.int32), out_dtype=tl.int32)
        score_query_scaled = score_integer.to(tl.float32) * query_code_scale[:, None]
        score_key_scaled = score_query_scaled * key_code_scale[None, :]
        S += score_key_scaled * score_scale
'''
SCALE_HELPER = '''@triton.jit
def _cjj_konly_scale(ptr, pages, slots, head, mask):
    offset = pages.to(tl.int64) * 4096 + 1024 + slots.to(tl.int64) * 2 + head.to(tl.int64)
    return tl.load(ptr + offset, mask=mask, other=1.0)
'''
IMPORTS = '''from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_unified_attention import (
    _cast_kv_tile, _load_q_td, _load_kv_tile_td, _store_output_td, float8_info)
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score, apply_softcap, cdiv_fn, compute_kv_seq_mask,
    compute_tile_loop_bounds, find_seq_idx, init_softmax_M, load_qq_bias_tile,
    resolve_seq_and_query_len, softmax_step, store_segm_reduce_scalars)
\n'''


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def inputs():
    directory = ROOT / 'docs/evidence/konly-prefill-reader-20261027'
    raw = (directory / 'source-inputs.tar.gz').read_bytes()
    if sha(raw) != ARCHIVE_SHA:
        raise ValueError('Frozen input archive changed')
    manifest = json.loads((directory / 'source-inputs.json').read_text())
    if manifest['main'] != MAIN or manifest['dependency_commit'] != DEPENDENCY:
        raise ValueError('Frozen root/dependency changed')
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members = archive.getmembers()
        if len(members) != len(PINS) or {m.name for m in members} != set(PINS):
            raise ValueError('Input archive membership changed')
        result = {}
        for member in members:
            if not member.isfile():
                raise ValueError('Only regular frozen source inputs are allowed')
            value = archive.extractfile(member).read()
            if sha(value) != PINS[member.name] or manifest['files'][member.name] != dict(bytes=len(value), sha256=sha(value)):
                raise ValueError('Frozen source bytes changed')
            result[member.name] = value.decode()
    for name in ('biv150_konly_decode_quant_reference.py', 'biv150_konly_i8_qk_reference.py'):
        if sha((ROOT / 'tools' / name).read_bytes()) != PINS[name]:
            raise ValueError('Vendored ae00 CPU oracle changed')
    return result


def extract(text, name):
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min(fn.lineno, *[n.lineno for n in fn.decorator_list]) - 1
    return ''.join(text.splitlines(keepends=True)[start:fn.end_lineno])


def canonical(source, profile):
    if profile not in PROFILES or any(sha(source[k]) != v for k, v in PINS.items()):
        raise ValueError('Unknown profile or unaudited source')
    raw = extract(source['sdk.py'], 'kernel_unified_attention')
    if profile == 'raw3d':
        return raw
    # Execute ONLY the pinned stdlib stride transformer, never the SDK/Triton module.
    tree = ast.parse(source['normalizer.py'])
    nodes = [n for n in tree.body if
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_STRIDES' for t in n.targets)
             or isinstance(n, ast.FunctionDef) and n.name == '_specialize_strides']
    scope = dict(ast=ast, re=re)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-stdlib-normalizer>', 'exec'), scope)
    raw = raw.replace('L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)', 'L = tl.zeros([BLOCK_M], dtype=tl.float32)')
    raw = raw.replace('def kernel_unified_attention(', 'def kernel_biv150_prefill_normalized(', 1)
    return scope['_specialize_strides'](raw)


def guards(profile):
    is3d, bm, bq, tile, *_ = PROFILES[profile]
    text = f'''    tl.static_assert(IS_3D == {is3d} and BLOCK_M == {bm} and BLOCK_Q == {bq} and TILE_SIZE == {tile})
    tl.static_assert(HEAD_SIZE == 128 and HEAD_SIZE_PADDED == 128 and BLOCK_SIZE == 16)
    tl.static_assert(num_query_heads == 16 and num_queries_per_kv == 8 and KV_QUANT_MODE == 0)
    tl.static_assert(not USE_TD and not USE_TD_QO and not Q_IS_FP8 and not USE_FP8)
    tl.static_assert(USE_CAUSAL and not USE_PER_SEQ_CAUSAL and not USE_MM_PREFIX)
    tl.static_assert(not USE_ALIBI_SLOPES and not USE_ALIBI_SQRT and not USE_QQ_BIAS)
    tl.static_assert(not USE_SOFTCAP and not USE_SINKS and SLIDING_WINDOW == 0)
    tl.static_assert(CHUNK_LOOKBACK == -1 and CHUNK_SIZE == -1)
    tl.static_assert(query_ptr.dtype.element_ty == tl.bfloat16 and output_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(value_cache_ptr.dtype.element_ty == tl.bfloat16)
    tl.static_assert(key_cache_ptr.dtype.element_ty == tl.int8 and key_code_scale_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(seq_lens_ptr.dtype.element_ty == tl.int32 and query_start_len_ptr.dtype.element_ty == tl.int32)
    tl.static_assert(block_tables_ptr.dtype.element_ty == tl.int32)
'''
    if is3d:
        text += '''    tl.static_assert(segm_output_ptr.dtype.element_ty == tl.float32)
    tl.static_assert(segm_max_ptr.dtype.element_ty == tl.float32 and segm_expsum_ptr.dtype.element_ty == tl.float32)
'''
    else:
        text += '    tl.static_assert(NUM_SEGMENTS_PER_SEQ == 1)\n'
    return text


def prefix(source):
    return IMPORTS + extract(source['codec-support-ae00.py'], 'quantize_bf16_rne') + '\n\n' + SCALE_HELPER + '\n\n'


def name(profile):
    return 'kernel_cjj_konly_' + profile


def restore(generated, profile, source):
    pre = prefix(source)
    if not generated.startswith(pre):
        raise ValueError('Private helper/import bytes changed')
    text = generated[len(pre):]
    original_name = 'kernel_unified_attention' if profile == 'raw3d' else 'kernel_biv150_prefill_stride'
    for old, new in [(guards(profile), ''), (ABI_NEW, ABI_OLD), (Q_ADD, ''), (K_NEW, K_OLD), (S_NEW, S_OLD),
                     ('def ' + name(profile) + '(', 'def ' + original_name + '(')]:
        if text.count(old) != 1:
            raise ValueError('Missing/duplicate private transform span')
        text = text.replace(old, new, 1)
    if text != canonical(source, profile):
        raise ValueError('Non-QK source/ABI/math changed')
    return text


def build(profile, source=None):
    source = inputs() if source is None else source
    raw = canonical(source, profile)
    reserved = {'key_code_scale_ptr', 'QCode', 'query_code_scale', 'key_code_scale',
                'score_integer', 'score_query_scaled', 'score_key_scaled'}
    if reserved.intersection(n.id for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.Name)):
        raise ValueError('Temporary variable collision')
    generated = raw
    for old, new in [(ABI_OLD, ABI_NEW), (Q_ANCHOR, Q_ADD + Q_ANCHOR), (K_OLD, K_NEW), (S_OLD, S_NEW)]:
        if generated.count(old) != 1:
            raise ValueError('Audited transform anchor changed')
        generated = generated.replace(old, new, 1)
    first = '    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2\n'
    generated = generated.replace(first, guards(profile) + first, 1)
    old_name = 'kernel_unified_attention' if profile == 'raw3d' else 'kernel_biv150_prefill_stride'
    generated = prefix(source) + generated.replace('def ' + old_name + '(', 'def ' + name(profile) + '(', 1)
    restore(generated, profile, source)
    compile(generated, '<cpu-syntax-only-private-konly>', 'exec')
    return generated, dict(profile=profile, source_sha256=PINS, main=MAIN, dependency_commit=DEPENDENCY,
        canonical_sha256=sha(raw), generated_sha256=sha(generated), exact_byte_inverse=True, exact_AST_inverse=True,
        codec=FORMAT, original_BF16_Q_V_PV_and_softmax_preserved=True,
        source_Q_quantization_outside_KV_loop=True, actual_I8_operand_hoist_verified=False,
        private_binding_only=True, unknown_format_fallback='reject',
        source_generation_only=True, factory_imports_no_framework=True,
        kernel_launches=0, JIT_executed=False,
        precision_gate=False, performance_verified=False, service_installed=False, capacity_reclaimed=False)
