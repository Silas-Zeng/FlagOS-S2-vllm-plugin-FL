"""Default-off stock-cache decode accuracy diagnostic. No cache writer/layout changes."""
import atexit
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

ENABLE = 'VLLM_FL_BIV150_STOCK_QK8_DIAGNOSTIC'
CERTIFICATE = 'VLLM_FL_BIV150_STOCK_QK8_MICRO_CERTIFICATE'
SCHEMA = 'cjj-stock-qk8-service-micro-v1'
GQA_SHA = '2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
KERNEL_SHA = 'af0d6f40e18d8e0dc6e114cf74603834e850e828401dd7600465dad206d44a44'
CODEC_SHA = '28ada2dbcc2205891f4e268ad87e679d137e621c91680299a4da7e059773303f'
COMPONENT_SOURCE = '5abf5119595474dc2f2a8b08df4835b34d8c0350'
REQUIRED_CASES = {(b, 129) for b in (1, 8, 16, 32, 40, 64)} | {(1, 4096)}
_MODULE = 'vllm_fl.patches.biv150_gqa_decode'
_STATE = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_contract():
    directory = Path(__file__).parent
    kernel = (directory / 'biv150_stock_qk8_kernel.py').read_bytes()
    codec = (directory / 'biv150_stock_qk8_codec.py').read_bytes()
    if sha(kernel) != KERNEL_SHA or sha(codec) != CODEC_SHA:
        raise RuntimeError('Stock diagnostic kernel/codec source changed')
    return dict(kernel_sha256=KERNEL_SHA, codec_sha256=CODEC_SHA, component_source=COMPONENT_SOURCE,
                hook_sha256=sha(Path(__file__).read_bytes()))


def certificate_contract(certificate):
    expected = source_contract()
    if certificate.get('schema') != SCHEMA or certificate.get('target') != ['corex', 71, 64]:
        raise RuntimeError('Missing or mismatched diagnostic micro target/schema')
    if any(certificate.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Diagnostic micro source differs from packaged source')
    if certificate.get('gpu_executed') is not True or certificate.get('passed') is not True:
        raise RuntimeError('Actual packaged-kernel micro validation is required before install')
    rows = certificate.get('cases', [])
    if len(rows) != len(REQUIRED_CASES) or {(r.get('batch'), r.get('kv')) for r in rows} != REQUIRED_CASES:
        raise RuntimeError('Micro coverage must include B1/8/16/32/40/64 and B1 long context')
    for row in rows:
        batch = row['batch']
        expected = 16 if batch == 40 else min(16, 1 << (max(1, (256 + batch * 2 - 1) // (batch * 2)) - 1).bit_length())
        if row.get('segments') != expected:
            raise RuntimeError('Micro split policy differs from main, including the B40 exception')
    required = ('quantized_operand_gate_passed', 'baseline_gate_passed', 'source_cache_unchanged',
                'canaries', 'actual_ir_passed', 'typed_abi_match', 'eager_warm_passed')
    if not rows or any(any(r.get(k) is not True for k in required) for r in rows):
        raise RuntimeError('Incomplete packaged-kernel micro hard gates')
    if certificate.get('m16_codec_bits_exact') is not True:
        raise RuntimeError('M16 codec bit evidence is required')
    return expected


def _argument_schema(source, name):
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(function.args)


def prepare_candidate(gqa, check_target=True):
    """Prepare for an explicit micro or install; does not change any live binding."""
    source_contract()
    original_source = Path(gqa.__file__).read_bytes()
    if sha(original_source) != GQA_SHA:
        raise RuntimeError('Unaudited parent GQA source')
    raw = gqa._gqa_decode_attention
    if not inspect.isfunction(getattr(raw, 'fn', None)) or raw.fn.__globals__ is not vars(gqa):
        raise RuntimeError('Parent GQA is not the audited JIT function')
    if raw.fn.__name__ != '_gqa_decode_attention' or Path(raw.fn.__code__.co_filename).resolve() != Path(gqa.__file__).resolve():
        raise RuntimeError('Parent GQA function identity changed')
    own = importlib.import_module('vllm_fl.patches.biv150_stock_qk8_kernel')
    candidate = own._gqa_decode_stock_konly_diagnostic
    path = Path(__file__).parent / 'biv150_stock_qk8_kernel.py'
    if Path(own.__file__).resolve() != path.resolve() or candidate.fn.__globals__ is not vars(own):
        raise RuntimeError('Diagnostic import does not resolve to this checkout')
    codec = importlib.import_module('vllm_fl.patches.biv150_stock_qk8_codec')
    codec_path = Path(__file__).parent / 'biv150_stock_qk8_codec.py'
    helper = codec.quantize_bf16_rne
    if Path(codec.__file__).resolve() != codec_path.resolve() or sha(Path(codec.__file__).read_bytes()) != CODEC_SHA:
        raise RuntimeError('Actual imported codec is outside the audited checkout')
    if own.quantize_bf16_rne is not helper or not inspect.isfunction(getattr(helper, 'fn', None)) or helper.fn.__globals__ is not vars(codec):
        raise RuntimeError('Actual diagnostic helper binding changed')
    if Path(helper.fn.__code__.co_filename).resolve() != codec_path.resolve():
        raise RuntimeError('Codec function code does not belong to the audited source')
    if _argument_schema(original_source, '_gqa_decode_attention') != _argument_schema(path.read_text(), candidate.fn.__name__):
        raise RuntimeError('Raw source parameter schema changed')
    if raw.arg_names != candidate.arg_names or raw.constexprs != candidate.constexprs:
        raise RuntimeError('Actual JIT argument names/constexpr positions changed')
    if check_target:
        import triton
        target = triton.runtime.driver.active.get_current_target()
        if (target.backend, target.arch, target.warp_size) != ('corex', 71, 64):
            raise RuntimeError('Stock diagnostic target mismatch')
    return raw, candidate


def _live_parent():
    names = {
        _MODULE: GQA_SHA,
        'vllm.v1.attention.ops.triton_unified_attention': '54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f',
        'vllm.v1.attention.ops.triton_attention_helpers': '59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1',
        'vllm.v1.attention.backends.triton_attn': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    }
    modules = {}
    for name, pin in names.items():
        module = sys.modules.get(name)
        if module is None or sha(Path(module.__file__).read_bytes()) != pin:
            raise RuntimeError('Missing/unaudited live dependency: ' + name)
        modules[name] = module
    gqa = modules[_MODULE]
    attention = modules['vllm.v1.attention.ops.triton_unified_attention']
    backend = modules['vllm.v1.attention.backends.triton_attn']
    if not getattr(attention, gqa._INSTALLED_ATTR, False) or backend.unified_attention is not attention.unified_attention:
        raise RuntimeError('Original GQA wrapper is not installed on both attention entry points')
    if attention.current_platform.vendor_name != 'iluvatar':
        raise RuntimeError('Unsupported vendor')
    # Reach the exact original GQA wrapper through main's optional normalization.
    current = attention.unified_attention
    for _ in range(3):
        if inspect.isfunction(current) and current.__globals__ is vars(gqa):
            cells = dict(zip(current.__code__.co_freevars, current.__closure__ or ()))
            if current.__code__.co_qualname != '_make_wrapper.<locals>.patched' or set(cells) != {'original'}:
                raise RuntimeError('Unknown GQA wrapper closure')
            if Path(current.__code__.co_filename).resolve() != Path(gqa.__file__).resolve():
                raise RuntimeError('GQA wrapper code is outside the audited source')
            original = cells['original'].cell_contents
            if not inspect.isfunction(original) or getattr(current, '__wrapped__', None) is not original or original.__globals__ is not vars(attention):
                raise RuntimeError('Original generic fallback binding changed')
            if original.__code__.co_name != 'unified_attention' or Path(original.__code__.co_filename).resolve() != Path(attention.__file__).resolve():
                raise RuntimeError('Generic fallback code changed')
            return gqa
        normalization = sys.modules.get('vllm_fl.patches.biv150_prefill_normalized')
        if normalization is None or not inspect.isfunction(current) or current.__globals__ is not vars(normalization):
            break
        if sha(Path(normalization.__file__).read_bytes()) != '957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1':
            raise RuntimeError('Unknown normalization source')
        cells = dict(zip(current.__code__.co_freevars, current.__closure__ or ()))
        if current.__code__.co_qualname != 'install.<locals>.wrapper' or set(cells) != {'normalized', 'original', 'signature'}:
            raise RuntimeError('Unknown normalization wrapper closure')
        if Path(current.__code__.co_filename).resolve() != Path(normalization.__file__).resolve():
            raise RuntimeError('Normalization wrapper code changed')
        current = getattr(current, '__wrapped__', None)
        if current is not cells['original'].cell_contents:
            raise RuntimeError('Normalization original binding changed')
    raise RuntimeError('Audited GQA wrapper not found; no diagnostic binding changed')


class Selected:
    def __init__(self, candidate, gqa):
        self.candidate, self.gqa = candidate, gqa
        self.calls = 0
        self.initial_calls = gqa._state['calls']
        self.initial_fallbacks = gqa._state['fallbacks']

    def __getitem__(self, grid):
        launch = self.candidate[grid]
        def selected(**kwargs):
            compiled = launch(**kwargs)  # Never silently fall back after a failure.
            self.calls += 1
            if self.calls & (self.calls - 1) == 0:
                print('BIV150_STOCK_QK8_SELECTED', dict(host_submissions=self.calls, grid=grid,
                      actual_kernel_hash=str(compiled.hash), original_fallbacks=self.gqa._state['fallbacks'] - self.initial_fallbacks,
                      diagnostic_only=True, graph_replays_not_counted=True), flush=True)
            return compiled
        return selected

    def snapshot(self):
        return dict(installed=True, selected_host_submissions=self.calls,
                    original_eligible_calls=self.gqa._state['calls'] - self.initial_calls,
                    original_fallback_calls=self.gqa._state['fallbacks'] - self.initial_fallbacks,
                    graph_replays_not_counted=True, diagnostic_only=True, full_prefill_quantization=False,
                    throughput_candidate=False)


def install():
    global _STATE
    if os.environ.get(ENABLE, '0') != '1':
        return dict(installed=False, reason='default off')
    if _STATE is not None:
        if _STATE.gqa._gqa_decode_attention is not _STATE:
            raise RuntimeError('Diagnostic producer binding changed after install')
        return _STATE.snapshot()
    certificate_path = os.environ.get(CERTIFICATE)
    if not certificate_path:
        raise RuntimeError('Explicit stock diagnostic requires packaged-kernel micro certificate')
    certificate_contract(json.loads(Path(certificate_path).read_text()))
    gqa = _live_parent()
    original, candidate = prepare_candidate(gqa)
    selected = Selected(candidate, gqa)
    if gqa._gqa_decode_attention is not original:
        raise RuntimeError('Parent producer changed during diagnostic preparation')
    gqa._gqa_decode_attention = selected  # The only live binding this module changes.
    _STATE = selected
    atexit.register(lambda: print('BIV150_STOCK_QK8_FINAL', selected.snapshot(), flush=True))
    return selected.snapshot()


def stats():
    return dict(installed=False) if _STATE is None else _STATE.snapshot()
