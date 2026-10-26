"""Private single-sequence 2D prefill launch-id reversal; no production install."""
import ast
import hashlib
import importlib.util
import inspect
import linecache
from pathlib import Path

MAIN = '241220723a09061c182764f8f738e5ec01190367'
SDK_SHA = '54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f'
HELPER_SHA = '59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1'
NORMALIZATION_SHA = '957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1'
OLD = '    q_block_global_idx = tl.program_id(0)\n'
NEW = '    q_block_global_idx = tl.num_programs(0) - 1 - tl.program_id(0)\n'
CASES = {
    'fresh1024': ([1024], [1024]),
    'fresh1025': ([1025], [1025]),
    'fresh2048': ([2048], [2048]),
    'fresh4096': ([4096], [4096]),
    'fresh16384': ([16384], [16384]),
    'minimum4k': ([1024], [4096]),
    'minimum16k': ([1024], [16384]),
    'partial4k': ([1025], [4096]),
    'chunk4k': ([2048], [4096]),
    'chunk16k': ([2048], [16384]),
    'suffix4k': ([1985], [4096]),
    'suffix16k': ([2017], [16384]),
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def function_source(text, name):
    functions = [n for n in ast.parse(text).body
                 if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        raise ValueError('Unique audited function required: ' + name)
    fn = functions[0]
    begin = min([fn.lineno] + [d.lineno for d in fn.decorator_list])-1
    return ''.join(text.splitlines(keepends=True)[begin:fn.end_lineno])


def load_python(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_factory(sdk_text, source_root):
    if sha(sdk_text) != SDK_SHA:
        raise ValueError('Actual audited SDK source is required')
    path = source_root/'vllm_fl/patches/biv150_prefill_normalized.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != NORMALIZATION_SHA:
        raise ValueError('Main normalization transform changed')
    normalize = load_python(path, '_reverse_cpu_normalization')
    raw = function_source(sdk_text, 'kernel_unified_attention')
    anchor = 'L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)'
    if raw.count(anchor) != 1:
        raise ValueError('Main normalization anchor changed')
    raw = raw.replace(anchor, 'L = tl.zeros([BLOCK_M], dtype=tl.float32)')
    raw = raw.replace('def kernel_unified_attention(', 'def kernel_biv150_prefill_normalized(', 1)
    baseline = normalize._specialize_strides(raw)
    if baseline.count(OLD) != 1 or baseline.count('tl.program_id(0)') != 1:
        raise ValueError('Unique q-block program-id definition required')
    candidate = baseline.replace(OLD, NEW, 1)
    if candidate.replace(NEW, OLD, 1) != baseline:
        raise ValueError('Exact byte inverse failed')
    # The one-line substitution does not change Q debug/source line numbers.
    if len(candidate.splitlines()) != len(baseline.splitlines()):
        raise ValueError('Source line count changed')
    if ast.dump(ast.parse(candidate.replace(NEW, OLD, 1))) != ast.dump(ast.parse(baseline)):
        raise ValueError('AST inverse failed')
    return baseline, candidate, dict(sdk_source_sha256=SDK_SHA,
        normalization_source_sha256=NORMALIZATION_SHA,
        normalized_source_sha256=sha(baseline), reversed_source_sha256=sha(candidate),
        exact_byte_inverse=True, source_line_count_preserved=True,
        source_edit=NEW.strip(), grid_unchanged=True, parent_dispatch_unchanged=True,
        block_m=256, block_q=32, tile_size=64, num_warps=16, num_stages=1,
        minimum_query_rows=1024, num_sequences=1, is_3d=False,
        floating_math_and_output_address_expressions_unchanged=True)


def eligible(values, grid, qfragment):
    """Use actual parent descriptor gate and require its exact Q32 launch grid."""
    try:
        return (values.get('num_seqs') == 1 and values['query_ptr'].shape[0] >= 1024
                and tuple(grid) == (values['query_ptr'].shape[0]//32+1, 2)
                and qfragment._eligible(values, 256, num_warps=16, num_stages=1, tile_size=64))
    except (KeyError, TypeError, AttributeError):
        return False


def private_manager(attention, backend, parent_manager, q32module, source_root, out):
    """Clone only the actual normalized raw JIT; keep main Q-hoist compiler proof."""
    sdk_text = Path(attention.__file__).read_text()
    baseline, candidate, proof = source_factory(sdk_text, source_root)
    if inspect.getsource(parent_manager.raw.fn) != baseline:
        raise ValueError('Actual Q32 raw JIT differs from CPU-audited main chain')
    name = '<biv150-prefill-reverse-' + proof['reversed_source_sha256'] + '.py>'
    linecache.cache[name] = (len(candidate), None, candidate.splitlines(keepends=True), name)
    namespace = dict(parent_manager.raw.fn.__globals__)
    exec(compile(candidate, name, 'exec'), namespace)
    raw = namespace['kernel_biv150_prefill_stride']
    jit = __import__('triton.runtime.jit', fromlist=['JITFunction'])
    clone = jit.JITFunction(raw.fn, version=raw.version,
        do_not_specialize=list(raw.do_not_specialize),
        do_not_specialize_on_alignment=list(raw.do_not_specialize_on_alignment),
        debug=raw.debug, noinline=raw.noinline, repr=raw._repr,
        launch_metadata=raw.launch_metadata)
    manager = q32module._VerifiedManager(attention, backend, {'kernel': raw},
        'kernel', raw, clone, block_m=256, num_warps=16, num_stages=1, tile_size=64)
    manager.install()
    (out/'baseline-normalized-kernel.py').write_text(baseline)
    (out/'candidate-reversed-kernel.py').write_text(candidate)
    return manager, proof
