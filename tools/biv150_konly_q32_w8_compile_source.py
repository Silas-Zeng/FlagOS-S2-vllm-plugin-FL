"""Private raw Q32/w8/s1 compile scout, reusing the reviewed zero-launch driver."""
import ast
import importlib.util
from pathlib import Path
import re
import subprocess
import types
import hashlib

ROOT = Path(__file__).resolve().parents[1]
BASE = '899ce77470b0b539aebc108fd122f668c2d5894d'
PARENT = '2e0929793d9e744a964c719e27d253927bcb52f3'
PARENT_FILE = 'tools/biv150_konly_q32_s2_compile_source.py'
PARENT_SHA = 'ddeb922f473e54d446d915c8429ac1595f191d34641853b8f3088f4c61ee1445'


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def frozen_parent():
    data = subprocess.check_output(['git', '-C', str(ROOT), 'show', PARENT + ':' + PARENT_FILE])
    need(sha(data) == PARENT_SHA, 'reviewed compiler-helper blob changed')
    module = types.ModuleType('cjj_w8_frozen_compiler_helper')
    module.__file__ = __file__  # Its ROOT must describe this checkout, not an old debug root.
    exec(compile(data, __file__ + ':frozen-parent', 'exec'), module.__dict__)
    need(module.BASE == BASE, 'frozen helper main899 baseline')
    return module


_parent = frozen_parent()
NAME, FUNCTION_SHA = _parent.NAME, _parent.FUNCTION_SHA
FILES, PARENT_FILES, DEPENDENCY = _parent.FILES, _parent.PARENT_FILES, _parent.DEPENDENCY
original, candidate_source = _parent.original, _parent.candidate_source
geometry, kwargs = _parent.geometry, _parent.kwargs
expected_sdk_pins, load_file = _parent.expected_sdk_pins, _parent.load_file
module_from_source, transform = _parent.module_from_source, _parent.transform


def snapshot_source():
    # Only the recorder's geometry predicate changes; no production Adapter is installed.
    return transform(original('biv150_konly_encoded_q32_adapter.py'),
                     [('    md = kernel.metadata\n    need((md.num_warps,',
                       "    md = kernel.metadata\n    need(all(type(getattr(md,k)) is int for k in ('num_warps','num_stages','warp_size')), 'actual integer options')\n    need((md.num_warps,"),
                      ('== (16, 1, 64)', '== (8, 1, 64)')])


def memory_observation(text):
    matches = re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB', text)
    need(len(matches) == 1, 'unknown GPU memory observation')
    used, total = map(int, matches[0])
    need(0 <= used <= 1024 and total > used, 'high/unknown preCUDA GPU memory')
    return dict(used_MiB=used, total_MiB=total,
                empty_PID_table_is_not_exclusivity_proof=True,
                exclusion_authority='Root explicit serial coordination, not namespace PID visibility')


def observe_memory(out):
    result = subprocess.run(['/usr/local/corex/bin/ixsmi'], capture_output=True,
                            text=True, timeout=15, check=True)
    (out / 'preCUDA-memory.txt').write_text(result.stdout + '\nSTDERR:\n' + result.stderr)
    return memory_observation(result.stdout)


def preserve_unqualified(kernel, fn, folder):
    """Export actual bytes before any recorder grammar/metadata can reject them."""
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'actual.source').write_text(fn.src)
    (folder / 'metadata.txt').write_text(repr(kernel.metadata) + '\n')
    for ext in ('ttir', 'ttgir', 'llir', 'asm', 'cubin'):
        data = kernel.asm.get(ext)
        if isinstance(data, str): (folder / ('actual.' + ext)).write_text(data)
        elif isinstance(data, bytes): (folder / ('actual.' + ext)).write_bytes(data)
    (folder / 'actual-object.txt').write_text(repr(dict(
        hash=getattr(kernel, 'hash', None), signature=getattr(kernel.src, 'signature', None),
        constants=getattr(kernel.src, 'constants', None), attrs=getattr(kernel.src, 'attrs', None))) + '\n')


def driver_edits():
    return [
        ('raw Q32/stage2 single warm compile', 'raw Q32/w8/stage1 single warm compile'),
        ('import biv150_konly_q32_s2_compile_source as c', 'import biv150_konly_q32_w8_compile_source as c'),
        ('from biv150_konly_q32_s2_compile_source import kwargs', 'from biv150_konly_q32_w8_compile_source import kwargs'),
        ("source = folder/'candidate_q32_s2.py'", "source = folder/'candidate_q32_w8.py'"),
        ("(folder/'private_stage2_snapshot.py').write_text(c.snapshot_source())",
         "(folder/'private_w8_snapshot.py').write_text(c.snapshot_source())"),
        ("adapter = c.load_file('cjj_private_q32_s2_snapshot', folder/'private_stage2_snapshot.py')",
         "adapter = c.load_file('cjj_private_q32_s2_snapshot', folder/'private_w8_snapshot.py')"),
        ('cjj_private_q32_s2_scout', 'cjj_private_q32_w8_scout'),
        ('cjj_private_q32_s2_snapshot', 'cjj_private_q32_w8_snapshot'),
        ('launch.update(num_warps=16, num_stages=2)', 'launch.update(num_warps=8, num_stages=1)'),
        ("role = 'raw-q32-stage2'", "role = 'raw-q32-w8-stage1'"),
        ("before = preserve(first,'raw-stage2')", "before = preserve(first,'raw-w8-stage1')"),
        ("for label, kernel in (('raw-stage2',first),):", "for label, kernel in (('raw-w8-stage1',first),):"),
        ('threads=16*64', 'threads=8*64'),
        ("            def preserve(kernel, label):\n                snapshot = adapter.kernel_snapshot(kernel,fn)",
         "            def preserve(kernel, label):\n                c.preserve_unqualified(kernel, fn, out/(label+'-unqualified'))\n                snapshot = adapter.kernel_snapshot(kernel,fn)"),
        ("        c.need(port == errno.ECONNREFUSED, 'service 9031 not confirmed closed')",
         "        c.need(port == errno.ECONNREFUSED, 'service 9031 not confirmed closed')\n        result['preCUDA_memory'] = c.observe_memory(args.out)\n        checkpoint()"),
        ("and type(resource['n_spills']) is int and resource['n_spills'] == 0,\n                    'unknown or spilled actual handle; stop before numeric')",
         "and type(resource['n_spills']) is int and resource['n_spills'] >= 0,\n                    'unknown actual handle resource fields')"),
    ]


def driver_source():
    return transform(_parent.driver_source(), driver_edits())


def driver():
    return module_from_source('cjj_private_q32_w8_compiler_driver', driver_source())


def plan():
    # Validate every old helper/source pin and its complete inverse, without importing SDKs.
    _parent.plan()
    source = candidate_source()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == NAME)
    need(sha(ast.get_source_segment(source, fn) + '\n') == FUNCTION_SHA, 'unchanged Q32 function')
    driver_source(); snapshot_source()
    config, grid = geometry(2048, 32)
    return dict(schema='cjj-private-I8-q32-w8-stage1-compile-v1', base=BASE,
        dependency_helpers={PARENT: {PARENT_FILE: PARENT_SHA}, DEPENDENCY: PARENT_FILES},
        source_pins=FILES, SDK_pins=expected_sdk_pins(),
        candidate_source_sha256=sha(source), driver_sha256=sha(driver_source()),
        snapshot_helper_sha256=sha(snapshot_source()), function_sha256=FUNCTION_SHA,
        query_rows=2048, num_seqs=1, pages=773, physical_page_bytes=16384,
        config=config, grid=list(grid), num_warps=8, num_stages=1,
        planned_JIT_warmups=1, source_math_byte_identical=True,
        kernel_change_from_accepted_Q32_S1='launch num_warps16 to8 only; fresh raw lowering, no inherited override',
        actual_handle_spills='record nonnegative actual count, including nonzero; not a qualification gate',
        override_allowed=False, hoist_grammar_applied=False, hoist_qualification_inherited=False,
        default_enabled=False, production_modified=False, GPU_executed=False, JIT_executed=False,
        attention_launches=0, writer_launches=0, encoder_launches=0, initialized_tensor_elements=0,
        actual_I8_operand_hoist_verified=False, actual_resources_verified=False,
        numerical_verified=False, Graph_verified=False, performance_verified=False,
        model_verified=False, service_verified=False,
        scope='one raw w8/s1 actual compilation for independent hoist/layout/resources analysis; no launches or performance comparison')
