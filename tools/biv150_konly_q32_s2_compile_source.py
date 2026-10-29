"""Private raw Q32/stage2 compiler scout; no hoist or device execution by default."""
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
NAME = 'kernel_cjj_konly_q32'
FUNCTION_SHA = '308a02ba6bef2e50c4c3d439809fd73eb2ca1e9fe87d47d4772f5fcef7ebde23'
FILES = {
    'biv150_konly_encoded_q32.py': 'ac0ee4caaeed8dfd211a57cd4d3d130d8437a2a0129c1c915e1b19ed4fc66b4b',
    'biv150_konly_encoded_q32_adapter.py': 'bb19f67d4e9ff80ca8510fe01ca5b1ad023d639ec267e5b51cc2d0d17bbfa04b',
    'biv150_konly_encoded_hoist_proposal.py': 'c0135ba632270b5d769cf3e53d5a5fc0dbde62a6e782daedc97422d2bbeff658',
}
PARENT_FILES = {
    'probe_biv150_konly_q64_compile.py': '749a662b7e203640c43b4282ae70abed318792dff302c7ef88ecab534fc02c7a',
    'biv150_konly_q64_compile_arguments.py': 'b82f517b94b196822f00037773b1c80806716ac77d9ef44881829ef1fb8fa52f',
}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def original(name):
    data = (ROOT / 'vllm_fl/patches' / name).read_bytes()
    need(sha(data) == FILES[name], 'main56 source pin ' + name)
    return data.decode()


def parent(name):
    data = subprocess.check_output(['git','-C',str(ROOT),'show',DEPENDENCY + ':tools/' + name])
    need(sha(data) == PARENT_FILES[name], 'frozen helper blob ' + name)
    return data.decode()


def transform(text, edits):
    result = text
    for before, after in edits:
        need(before != after and result.count(before) == 1, 'unique source edit ' + repr(before[:100]))
        result = result.replace(before, after, 1)
    restored = result
    for before, after in reversed(edits):
        need(after and restored.count(after) == 1, 'unique source inverse ' + repr(after[:100]))
        restored = restored.replace(after, before, 1)
    need(restored == text, 'byte-exact driver inverse')
    return result


def candidate_source(): return original('biv150_konly_encoded_q32.py')


def snapshot_source():
    # A private recorder scope change, not a production Adapter admission.
    return transform(original('biv150_konly_encoded_q32_adapter.py'),
        [('== (16, 1, 64)', '== (16, 2, 64)')])


def expected_sdk_pins():
    tree = ast.parse(original('biv150_konly_encoded_hoist_proposal.py'))
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'PINS')
    need(isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
         and value.func.id == 'dict' and not value.args and all(k.arg for k in value.keywords),
         'pinned SDK identity literal')
    return {k.arg: ast.literal_eval(k.value) for k in value.keywords}


def module_from_source(name, text):
    module = types.ModuleType(name); module.__file__ = str(Path(__file__))
    exec(compile(text, module.__file__, 'exec'), module.__dict__)
    return module


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometry(query_rows, bq):
    need(type(query_rows) is int and query_rows == 2048 and type(bq) is int and bq == 32,
         'single Q2048/BQ32 scout only')
    return dict(BLOCK_M=256, BLOCK_Q=32, TILE_SIZE=64, IS_3D=False), (65, 2)


def kwargs(p, tensors, candidate):
    return module_from_source('cjj_q32_s2_original_arguments',
        parent('biv150_konly_q64_compile_arguments.py')).kwargs(p, tensors, candidate)


def driver_edits(text):
    # Remove the complete second warmup/proposal interval of the frozen driver.
    start = text.index('            meta = c.metadata(')
    stop = text.index("            record['handle_resources'] = {}", start)
    removed = text[start:stop]
    return [
        ('Root-owned Q32/Q64 warm compiles, zero launches.', 'Root-owned raw Q32/stage2 single warm compile, zero launches.'),
        ('import biv150_konly_q64_compile_source as c', 'import biv150_konly_q32_s2_compile_source as c'),
        ('from biv150_konly_q64_compile_arguments import kwargs', 'from biv150_konly_q32_s2_compile_source import kwargs'),
        ("'production must equal a455'", "'production must equal main56'"),
        ('h32 = c.proposal(32)', 'expected_pins = c.expected_sdk_pins()'),
        ('h32.PINS[key]', 'expected_pins[key]'),
        ("h32.PINS['sdk']", "expected_pins['sdk']"),
        ("h32.PINS['helpers']", "expected_pins['helpers']"),
        ("pins == h32.PINS", "pins == expected_pins"),
        ("        os.environ['TRITON_CACHE_DIR'] = str(args.out/'triton-cache')",
         "        c.need(os.environ.get('TRITON_KERNEL_OVERRIDE','0') == '0', 'external compiler override forbidden')\n        os.environ['TRITON_CACHE_DIR'] = str(args.out/'triton-cache')"),
        ("source = folder/'candidate_q64.py'", "source = folder/'candidate_q32_s2.py'"),
        ("(folder/'q64_hoist_proposal.py').write_text(c.proposal_source())",
         "(folder/'private_stage2_snapshot.py').write_text(c.snapshot_source())"),
        ("modules = {32:c.load_file('cjj_q64_screen_control', patch/'biv150_konly_encoded_q32.py'),\n                   64:c.load_file('cjj_q64_screen_candidate', source)}",
         "modules = {32:c.load_file('cjj_private_q32_s2_scout', source)}"),
        ("adapter = c.load_file('cjj_q64_screen_snapshot', patch/'biv150_konly_encoded_q32_adapter.py')",
         "adapter = c.load_file('cjj_private_q32_s2_snapshot', folder/'private_stage2_snapshot.py')"),
        ("        # Only use the old SSA-normalization helper. Its Q32-specific native\n        # instruction/count audit is deliberately NOT used to qualify Q64.\n        native = c.module_from_source('cjj_q64_screen_alpha', c.original('biv150_konly_encoded_hoist_native.py'))",
         '        # No hoist proposal, override, native proof or qualification is used.'),
        ('for bq in (32,64):', 'for bq in (32,):'),
        ("fn = getattr(modules[bq], 'kernel_cjj_konly_q32' if bq == 32 else c.NAME)", 'fn = getattr(modules[bq], c.NAME)'),
        ('launch.update(num_warps=16, num_stages=1)', 'launch.update(num_warps=16, num_stages=2)'),
        ("role = 'q' + str(bq)", "role = 'raw-q32-stage2'"),
        ('            h = c.proposal(bq)', "            c.need('ir_override' not in launch, 'raw scout forbids an override argument')\n            compiler_knobs = getattr(fn.compile, '__globals__', {}).get('knobs')\n            c.need(compiler_knobs is not None and compiler_knobs.compilation.override is False,\n                'actual compiler override manager must be disabled')"),
        ("before = preserve(first,'unhoisted')", "before = preserve(first,'raw-stage2')"),
        (removed, "            c.need(before['function_name'] == c.NAME and before['source_digest'] == c.FUNCTION_SHA,\n                'actual original Q32 function identity')\n            c.need(hasattr(first.metadata,'ir_override') and first.metadata.ir_override is None,\n                'actual compiled metadata must explicitly contain no override')\n            record['actual_metadata_ir_override'] = first.metadata.ir_override\n"),
        ("for label, kernel in (('unhoisted',first),('hoisted',second)):", "for label, kernel in (('raw-stage2',first),):"),
        ('actual_backend_hoist_and_resource_cost_review_pending', 'actual_backend_pipeline_and_resource_cost_review_pending'),
    ]


def driver_source():
    text = parent('probe_biv150_konly_q64_compile.py')
    return transform(text, driver_edits(text))


def driver(): return module_from_source('cjj_private_q32_s2_scout_driver', driver_source())


def plan():
    for name in FILES: original(name)
    for name in PARENT_FILES: parent(name)
    source = candidate_source(); tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == NAME)
    need(sha(ast.get_source_segment(source, fn) + '\n') == FUNCTION_SHA, 'unchanged Q32 JIT source')
    driver_source(); snapshot_source()
    config, grid = geometry(2048, 32)
    return dict(schema='cjj-private-q32-stage2-raw-compile-scout-v1',base=BASE,dependency=DEPENDENCY,
        source_pins=FILES,helper_pins=PARENT_FILES,SDK_pins=expected_sdk_pins(),
        candidate_source_sha256=sha(source),driver_sha256=sha(driver_source()),snapshot_helper_sha256=sha(snapshot_source()),
        query_rows=2048,num_seqs=1,pages=773,logical_page_bytes=16384,config=config,grid=list(grid),
        num_warps=16,num_stages=2,planned_warmups=1,source_math_byte_identical=True,
        override_allowed=False,hoist_proposal_used=False,hoist_qualification_inherited=False,
        default_enabled=False,production_modified=False,GPU_executed=False,JIT_executed=False,
        attention_launches=0,writer_launches=0,encoder_launches=0,initialized_tensor_elements=0,
        pipeline_verified=False,actual_resources_verified=False,numerical_verified=False,
        Graph_verified=False,performance_verified=False,service_verified=False,
        scope='raw S2 artifacts only; not a hoisted S2 candidate or a stage1/stage2 performance comparison')
