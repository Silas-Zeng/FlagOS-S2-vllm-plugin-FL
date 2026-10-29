"""Private two-sequence operand-hoist source; CPU-only construction, no service hook."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = '56f88737fcf3b940760b9ca3027a925cd5a7d77d'
DEPENDENCY = '844a69f10539c551500e29e2304061aef7d8671b'
NAME = 'kernel_cjj_konly_multi2'
FUNCTION_SHA = '39ebc5ce6ee4546f45932026f935c670e8c568aa94c6a3bc26e94b9ddab50e18'
FILES = {
    'biv150_konly_encoded_multi2.py': '7046cf538bf06278982041d04b129c65c74bdb65531e528366fc62626c6adb19',
    'biv150_konly_encoded_hoist_proposal.py': 'c0135ba632270b5d769cf3e53d5a5fc0dbde62a6e782daedc97422d2bbeff658',
    'biv150_konly_encoded_q32_adapter.py': 'bb19f67d4e9ff80ca8510fe01ca5b1ad023d639ec267e5b51cc2d0d17bbfa04b',
    'biv150_konly_encoded_hoist_native.py': '2027e1ce4a489f6b47c97a5253c1482d2dc767e62e9b10ec029e3f192c559ef2',
}
PARENT_FILES = {
    'probe_biv150_konly_q64_compile.py': '749a662b7e203640c43b4282ae70abed318792dff302c7ef88ecab534fc02c7a',
    'biv150_konly_q64_compile_arguments.py': 'b82f517b94b196822f00037773b1c80806716ac77d9ef44881829ef1fb8fa52f',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def original(name):
    data = (ROOT / 'vllm_fl/patches' / name).read_bytes()
    need(sha(data) == FILES[name], 'main56 source pin ' + name)
    return data.decode()


def parent(name):
    data = subprocess.check_output(['git', '-C', str(ROOT), 'show', DEPENDENCY + ':tools/' + name])
    need(sha(data) == PARENT_FILES[name], 'frozen compile helper pin ' + name)
    return data.decode()


def transform(text, edits):
    result = text
    for before, after in edits:
        need(before != after and result.count(before) == 1, 'unique source transformation ' + repr(before))
        result = result.replace(before, after)
    inverse = result
    for before, after in reversed(edits):
        need(inverse.count(after) == 1, 'unique inverse transformation ' + repr(after))
        inverse = inverse.replace(after, before)
    need(inverse == text, 'byte-exact inverse transformation')
    return result


def candidate_source():
    # Original function, including all geometry and math; the loader supplies a private module name.
    return original('biv150_konly_encoded_multi2.py')


PROPOSAL_EDITS = (
    ("SCHEMA='cjj-konly-service-q32-hoist-v1'", "SCHEMA='cjj-konly-private-multi2-hoist-v1'"),
    ("meta['profile']=='q32'", "meta['profile']=='multi2'"),
    ('num_seqs=1', 'num_seqs=2'),
    ("list(meta['grid'])==[q//32+1,2]", "list(meta['grid'])==[q//32+2,2]"),
    ("'private Q32 format/schema'", "'private multi2 format/schema'"),
    ("'single-sequence Q32 selected constants'", "'two-sequence private Q32 selected constants'"),
    ("'complete single-sequence parent grid'", "'complete two-sequence parent grid'"),
)


def proposal_source():
    return transform(original('biv150_konly_encoded_hoist_proposal.py'), PROPOSAL_EDITS)


def module_from_source(name, text):
    module = types.ModuleType(name)
    module.__file__ = str(Path(__file__))
    exec(compile(text, module.__file__, 'exec'), module.__dict__)
    return module


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def proposal(bq):
    need(type(bq) is int and bq == 32, 'private multi2 BQ32 only')
    return module_from_source('cjj_private_multi2_hoist_proposal', proposal_source())


def geometry(query_rows, bq):
    need(type(query_rows) is int and 1024 <= query_rows <= 2048, 'private total query rows 1024..2048')
    need(type(bq) is int and bq == 32, 'private BQ32 only')
    return dict(BLOCK_M=256, BLOCK_Q=32, TILE_SIZE=64, IS_3D=False), (query_rows // 32 + 2, 2)


def metadata(h, snapshot, query_rows, bq, target):
    config, grid = geometry(query_rows, bq)
    need(snapshot['function_name'] == NAME and snapshot['source_digest'] == FUNCTION_SHA,
         'actual unchanged multi2 JIT source identity')
    return dict(schema=h.SCHEMA, enabled=True, profile='multi2', format=h.FORMAT, target=target,
        launch_options=dict(num_warps=16, num_stages=1),
        constants=dict(config, num_seqs=2, num_query_heads=16, num_queries_per_kv=8, KV_QUANT_MODE=0),
        query_rows=query_rows, grid=list(grid), kernel_hash=snapshot['hash'],
        function_name=snapshot['function_name'], source_digest=snapshot['source_digest'],
        ttgir_sha256=snapshot['artifacts']['ttgir'], pins=h.PINS)


def kwargs(p, t, candidate):
    # Byte-for-byte original ABI mapper, not a new numerical implementation.
    module = module_from_source('cjj_private_multi2_compile_arguments', parent('biv150_konly_q64_compile_arguments.py'))
    return module.kwargs(p, t, candidate)


DRIVER_EDITS = (
    ('Root-owned Q32/Q64 warm compiles', 'Root-owned private multi2 warm compiles'),
    ('import biv150_konly_q64_compile_source as c', 'import biv150_konly_multi2_hoist_source as c'),
    ('from biv150_konly_q64_compile_arguments import kwargs', 'from biv150_konly_multi2_hoist_source import kwargs'),
    ('# Same Q32 compile fixture descriptors;', '# Original compile descriptors with exactly two sequence rows;'),
    ('table=empty((1,8192), torch.int32)', 'table=empty((2,8192), torch.int32)'),
    ('lengths=empty((1,), torch.int32), cu_q=empty((2,), torch.int32)',
     'lengths=empty((2,), torch.int32), cu_q=empty((3,), torch.int32)'),
    ("'production must equal a455'", "'production must equal main56'"),
    ("source = folder/'candidate_q64.py'", "source = folder/'candidate_multi2.py'"),
    ("(folder/'q64_hoist_proposal.py')", "(folder/'multi2_hoist_proposal.py')"),
    ("modules = {32:c.load_file('cjj_q64_screen_control', patch/'biv150_konly_encoded_q32.py'),\n                   64:c.load_file('cjj_q64_screen_candidate', source)}",
     "modules = {32:c.load_file('cjj_private_multi2_compile_candidate', source)}"),
    ("'cjj_q64_screen_snapshot'", "'cjj_private_multi2_compile_snapshot'"),
    ('# instruction/count audit is deliberately NOT used to qualify Q64.',
     '# instruction/count audit is deliberately NOT used to qualify multi2.'),
    ("'cjj_q64_screen_alpha'", "'cjj_private_multi2_compile_alpha'"),
    ('for bq in (32,64):', 'for bq in (32,):'),
    ("fn = getattr(modules[bq], 'kernel_cjj_konly_q32' if bq == 32 else c.NAME)", 'fn = getattr(modules[bq], c.NAME)'),
    ("dict(profile='q' + str(bq), sequences=1, segments=1, config=config)",
     "dict(profile='multi2', sequences=2, segments=1, config=config)"),
    ("role = 'q' + str(bq)", "role = 'multi2'"),
)


def driver_source():
    return transform(parent('probe_biv150_konly_q64_compile.py'), DRIVER_EDITS)


def driver():
    return module_from_source('cjj_private_multi2_compile_driver', driver_source())


def plan():
    for name in FILES:
        original(name)
    for name in PARENT_FILES:
        parent(name)
    source = candidate_source()
    fn = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == NAME)
    need(sha(ast.get_source_segment(source, fn) + '\n') == FUNCTION_SHA, 'original JIT function digest')
    # Evaluate source inverses now, before any SDK import or device action.
    proposal_source(); driver_source()
    config, grid = geometry(2048, 32)
    return dict(schema='cjj-private-multi2-operand-hoist-compile-v1', base_commit=BASE,
        compile_driver_dependency=DEPENDENCY, production_sources=FILES, frozen_helpers=PARENT_FILES,
        driver_sha256=sha(driver_source()), proposal_sha256=sha(proposal_source()),
        source_digest=FUNCTION_SHA, candidate_source_sha256=sha(source),
        profile='multi2', num_seqs=2, total_query_rows=2048, config=config, grid=list(grid),
        num_warps=16, num_stages=1, logical_page_bytes=16384, K_tile=64,
        change='one initialized immutable Q operand local_load moves to its same-parent KV preheader',
        candidate_source_byte_identical_to_main=True, production_modified=False, default_enabled=False,
        compiler_actions=['unhoisted warmup', 'hoisted warmup', 'load exact handles and record resources'],
        initialized_tensor_elements=0, attention_launches=0, writer_launches=0, encoder_launches=0,
        GPU_executed=False, numerical_verified=False, Graph_verified=False, resources_verified=False,
        backend_Q_operand_residency_verified=False, performance_verified=False, service_verified=False,
        q32_service_qualification_inherited=False,
        scope='private compile only; actual multi2 backend/ELF/loop resource review precedes device numeric')
