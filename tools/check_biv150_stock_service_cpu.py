#!/usr/bin/env python3
"""Source/default-off/certificate contracts only; synthetic units are not device evidence."""
import ast
import copy
import hashlib
import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = '56f1838192abc940b3fb9accef490022ac125000'
ADDITION = '''
# Decode-only accuracy diagnostic; all stock BF16 cache paths remain unchanged.
if os.environ.get("VLLM_FL_BIV150_STOCK_QK8_DIAGNOSTIC", "0") == "1":
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        raise RuntimeError("Stock QK8 diagnostic requires the audited iluvatar target")
    from vllm_fl.patches.biv150_stock_qk8_diagnostic import install as _install_stock_qk8_diagnostic

    print("BIV150_STOCK_QK8_INSTALL", _install_stock_qk8_diagnostic(), flush=True)
'''
FEATURE = '    features["VLLM_FL_BIV150_STOCK_QK8_DIAGNOSTIC"] = env.get("VLLM_FL_BIV150_STOCK_QK8_DIAGNOSTIC", "0") == "1"\n'


def original_runner(text):
    assert text.count(ADDITION) == 1
    restored = text.replace(ADDITION, '')
    assert hashlib.sha256(restored.encode()).hexdigest() == '0161c595ff2f1b115039f03980fbb6de0447f1edf0d55fba1793b6a8055c230c'
    return restored


def original_semantics(text):
    assert text.count(FEATURE) == 1
    return text.replace(FEATURE, '')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def reject(call):
    try:
        call()
    except (RuntimeError, AssertionError, ValueError, KeyError):
        return
    raise AssertionError('Expected fail-closed rejection')


def run():
    hook = module('stock_qk8_cpu_only', ROOT / 'vllm_fl/patches/biv150_stock_qk8_diagnostic.py')
    provenance = json.loads((ROOT / 'docs/evidence/stock-qk8-service-diagnostic-20261027/source-provenance.json').read_text())
    facts = hook.source_contract()
    assert facts['kernel_sha256'] == provenance['packaged_kernel_sha256']
    assert facts['codec_sha256'] == provenance['packaged_codec_sha256']
    kernel = (ROOT / 'vllm_fl/patches/biv150_stock_qk8_kernel.py').read_text()
    component = kernel.replace('from vllm_fl.patches.biv150_stock_qk8_codec import', 'from cjj_konly_decode_support import', 1)
    assert hashlib.sha256(component.encode()).hexdigest() == provenance['original_generated_sha256']
    codec = (ROOT / 'vllm_fl/patches/biv150_stock_qk8_codec.py').read_text()
    fn = next(n for n in ast.parse(codec).body if isinstance(n, ast.FunctionDef))
    fragment = ''.join(codec.splitlines(keepends=True)[min(fn.lineno, *(n.lineno for n in fn.decorator_list)) - 1:fn.end_lineno])
    assert hashlib.sha256(fragment.encode()).hexdigest() == provenance['codec_function_sha256']
    before = set(sys.modules)
    previous = os.environ.pop(hook.ENABLE, None)
    try:
        assert hook.install() == dict(installed=False, reason='default off')
        assert hook.stats() == dict(installed=False)
        assert set(sys.modules) == before
        os.environ[hook.ENABLE] = '1'
        certificate = os.environ.pop(hook.CERTIFICATE, None)
        try:
            reject(hook.install)
        finally:
            if certificate is not None:
                os.environ[hook.CERTIFICATE] = certificate
        assert hook.stats() == dict(installed=False)
    finally:
        if previous is None:
            os.environ.pop(hook.ENABLE, None)
        else:
            os.environ[hook.ENABLE] = previous
    # Synthetic schema fixture only. Never written as a device certificate.
    fixture = dict(schema=hook.SCHEMA, target=['corex', 71, 64], **facts,
                   gpu_executed=True, passed=True, m16_codec_bits_exact=True, cases=[])
    for batch, kv in sorted(hook.REQUIRED_CASES):
        segment = 16 if batch in (1, 8, 40) else {16: 8, 32: 4, 64: 2}[batch]
        fixture['cases'].append(dict(batch=batch, kv=kv, segments=segment,
            quantized_operand_gate_passed=True, baseline_gate_passed=True, source_cache_unchanged=True,
            canaries=True, actual_ir_passed=True, typed_abi_match=True, eager_warm_passed=True))
    hook.certificate_contract(fixture)
    negatives = []
    for key, value in [('kernel_sha256', '0' * 64), ('codec_sha256', '0' * 64), ('target', ['cuda', 71, 64]),
                       ('gpu_executed', False), ('m16_codec_bits_exact', False), ('passed', False), ('cases', fixture['cases'][1:])]:
        mutant = copy.deepcopy(fixture); mutant[key] = value; negatives.append(mutant)
    for field in ('quantized_operand_gate_passed', 'actual_ir_passed', 'typed_abi_match', 'segments'):
        mutant = copy.deepcopy(fixture); mutant['cases'][0][field] = False; negatives.append(mutant)
    for mutant in negatives:
        reject(lambda mutant=mutant: hook.certificate_contract(mutant))
    # Dispatch counter unit only: no fake result is ever used as a device pass.
    class Kernel:
        def __getitem__(self, grid):
            return lambda **kwargs: SimpleNamespace(hash='CPU-counter-unit-only')
    gqa = SimpleNamespace(_state=dict(calls=10, fallbacks=5))
    selected = hook.Selected(Kernel(), gqa)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(2):
            gqa._state['calls'] += 1
            selected[(1, 2, 16)](no_tensor_read=True)
    gqa._state['fallbacks'] += 3
    assert selected.snapshot()['selected_host_submissions'] == 2
    assert selected.snapshot()['original_eligible_calls'] == 2
    assert selected.snapshot()['original_fallback_calls'] == 3
    assert selected.snapshot()['graph_replays_not_counted']
    runner = (ROOT / 'vllm_fl/worker/model_runner.py').read_text()
    assert original_runner(runner) == subprocess.check_output(['git', '-C', str(ROOT), 'show', MAIN + ':vllm_fl/worker/model_runner.py'], text=True)
    semantics = (ROOT / 'vllm_fl/compile_semantics.py').read_text()
    assert original_semantics(semantics) == subprocess.check_output(['git', '-C', str(ROOT), 'show', MAIN + ':vllm_fl/compile_semantics.py'], text=True)
    order = (ROOT / 'vllm_fl/patches/biv150_compute_order.py').read_text()
    pin = hashlib.sha256(runner.encode()).hexdigest()
    restored = order.replace(pin, '0161c595ff2f1b115039f03980fbb6de0447f1edf0d55fba1793b6a8055c230c')
    assert restored == subprocess.check_output(['git', '-C', str(ROOT), 'show', MAIN + ':vllm_fl/patches/biv150_compute_order.py'], text=True)
    semantic_module = module('stock_semantics_cpu_only', ROOT / 'vllm_fl/compile_semantics.py')
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**semantic_module._GEOMETRY)))
    with tempfile.TemporaryDirectory() as folder:
        library = Path(folder) / 'never-loaded.so'; library.write_bytes(b'CPU hash fixture, never loaded')
        env = dict(BIV150_NATIVE_W8A8_LIBRARY=str(library))
        def fingerprint(extra):
            return semantic_module.build_fingerprint(config, 'iluvatar', package_root=ROOT / 'vllm_fl', environ=env | extra, abi=dict(torch='CPU contract', cxx11_abi=False))
        assert fingerprint({}) == fingerprint({hook.ENABLE: '0'})
        assert fingerprint({}) != fingerprint({hook.ENABLE: '1'})
        assert fingerprint({})['native_enabled']
    assert not {'torch', 'triton', 'numpy'}.intersection(sys.modules)
    return dict(passed=True, cpu_only=True, source_import_only_inverse=True, codec_exact_extraction=True,
                default_off_zero_hook_actions=True, missing_certificate_refused_before_binding=True,
                synthetic_certificate_schema_only=True, negative_contracts=len(negatives),
                runner_exact_inverse=True, unique_runner_pin_updated=True, semantics_exact_inverse=True,
                cache_key_default_equals_off=True, cache_key_on_separate=True, native_default_preserved=True,
                counter_unit_distinguishes_selected_and_generic_fallback=True,
                device_micro_verified=False, service_installed=False, gpu_executed=False)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
