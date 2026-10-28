"""CPU-only, reversible Q64 geometry experiment; never imported by production."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a455a3de364051fbea80226870c239de3a48e716'
PINS = {
    'biv150_konly_encoded_q32.py': 'ac0ee4caaeed8dfd211a57cd4d3d130d8437a2a0129c1c915e1b19ed4fc66b4b',
    'biv150_konly_encoded_hoist_proposal.py': 'c0135ba632270b5d769cf3e53d5a5fc0dbde62a6e782daedc97422d2bbeff658',
    'biv150_konly_encoded_q32_adapter.py': 'bb19f67d4e9ff80ca8510fe01ca5b1ad023d639ec267e5b51cc2d0d17bbfa04b',
    'biv150_konly_encoded_hoist_native.py': '2027e1ce4a489f6b47c97a5253c1482d2dc767e62e9b10ec029e3f192c559ef2',
}
NAME = 'kernel_cjj_konly_q64_compile_probe'
KERNEL_EDITS = (
    ('kernel_cjj_konly_q32(', NAME + '(', 1),
    ('BLOCK_M == 256 and BLOCK_Q == 32', 'BLOCK_M == 512 and BLOCK_Q == 64', 1),
)
PROPOSAL_EDITS = (
    ('q32', 'q64', 2), ('Q32', 'Q64', 5),
    ('tensor<256x', 'tensor<512x', 8),
    ('!ttg.memdesc<256x', '!ttg.memdesc<512x', 1),
    ('BLOCK_M=256,BLOCK_Q=32', 'BLOCK_M=512,BLOCK_Q=64', 1),
    ('q//32+1', 'q//64+1', 1),
)


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest()


def original(name):
    path = ROOT / 'vllm_fl/patches' / name
    raw = path.read_bytes()
    need(sha(raw) == PINS[name], 'a455 source changed: ' + name)
    return raw.decode()


def transform(text, edits):
    old = text
    for before, after, count in edits:
        need(text.count(before) == count and after not in text, 'nonunique source edit: ' + before)
        text = text.replace(before, after)
    restored = text
    for before, after, count in reversed(edits):
        need(restored.count(after) == count, 'inverse count: ' + after)
        restored = restored.replace(after, before)
    need(restored == old, 'exact source inverse failed')
    return text


def candidate_source():
    return transform(original('biv150_konly_encoded_q32.py'), KERNEL_EDITS)


def proposal_source():
    return transform(original('biv150_konly_encoded_hoist_proposal.py'), PROPOSAL_EDITS)


def module_from_source(name, text):
    module = types.ModuleType(name)
    exec(compile(text, '<' + name + '>', 'exec'), vars(module))
    return module


def proposal(bq):
    need(type(bq) is int and bq in (32, 64), 'only Q32 control or Q64 candidate')
    text = (original('biv150_konly_encoded_hoist_proposal.py') if bq == 32 else proposal_source())
    return module_from_source('cjj_private_compile_hoist_q' + str(bq), text)


def load_file(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometry(q, bq):
    need(type(q) is int and 1024 <= q <= 2048, 'isolated single-sequence 1024..2048 query scope')
    need(type(bq) is int and bq in (32, 64), 'only Q32/Q64; no parameter sweep')
    return dict(BLOCK_M=bq * 8, BLOCK_Q=bq, TILE_SIZE=64, IS_3D=False), (q // bq + 1, 2)


def metadata(h, snapshot, q, bq, target):
    config, grid = geometry(q, bq)
    return dict(schema=h.SCHEMA, enabled=True, profile='q' + str(bq), format=h.FORMAT,
        target=target, launch_options=dict(num_warps=16, num_stages=1),
        constants=dict(config, num_seqs=1, num_query_heads=16, num_queries_per_kv=8, KV_QUANT_MODE=0),
        query_rows=q, grid=list(grid), kernel_hash=snapshot['hash'], function_name=snapshot['function_name'],
        source_digest=snapshot['source_digest'], ttgir_sha256=snapshot['artifacts']['ttgir'], pins=dict(h.PINS))


def plan():
    for name in PINS: original(name)
    source = candidate_source()
    base_tree, tree = ast.parse(original('biv150_konly_encoded_q32.py')), ast.parse(source)
    # Normalize only the private symbol and the two values in the geometry
    # assertion, then compare the COMPLETE Python AST, including codec/masks.
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == NAME)
    fn.name = 'kernel_cjj_konly_q32'
    assertion = next(n for n in fn.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == 'tl.static_assert' and 'BLOCK_M' in ast.unparse(n))
    for n in ast.walk(assertion):
        # TILE_SIZE remains 64; only named BM/BQ comparisons are adapted.
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name):
            if n.left.id == 'BLOCK_M': n.comparators[0].value = 256
            if n.left.id == 'BLOCK_Q': n.comparators[0].value = 32
    need(ast.dump(base_tree) == ast.dump(tree), 'non-geometry AST difference')
    return dict(schema='cjj-konly-q64-compile-screen-v1', base=BASE,
        source_pins=PINS, candidate_source_sha256=sha(source), proposal_source_sha256=sha(proposal_source()),
        kernel_edits=list(KERNEL_EDITS), proposal_geometry_edits=list(PROPOSAL_EDITS),
        complete_AST_equal_after_geometry_inverse=True, exact_byte_inverse=True,
        query_rows=2048, pages=773, logical_page_bytes=16384, source_pitch=2048,
        configs={str(bq):dict(constants=geometry(2048, bq)[0], grid=list(geometry(2048, bq)[1])) for bq in (32, 64)},
        launch_options=dict(num_warps=16, num_stages=1),
        GPU_executed=False, JIT_executed=False, actual_I8_hoist_verified=False,
        actual_resources_verified=False, numerical_verified=False, Graph_verified=False,
        performance_verified=False, service_verified=False, precision_gate=False)
