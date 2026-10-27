#!/usr/bin/env python3
"""Materialize pinned private GPU source using stdlib factories; never import JIT."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MIXED = '4a3fc7773b37e745082d0d98f8d53713130384ea'
PREFILL = 'ef4034b23d194b8163062826447ee2b4aa950c36'
WRITER = 'ec4262557282e6c95d29e69784bb16a2c6654091'
MAIN = 'b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd'
GQA_MATH_BASE = 'dac1b8fb48e7983e48283032efeee9b592c5fbe5'
HOIST = 'e4dd64e453fbe1bdeba4d8b2a1b551167e9446c4'


def sha(data): return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()
def git(ref, path): return subprocess.check_output(['git', '-C', str(ROOT), 'show', ref + ':' + path])


def run(write=False):
    # Factory/support bytes in4a are the documented unmodified f287 dependency;
    # use that precise known Git object, not an abbreviated/unverified ref.
    entries = {name: MIXED for name in ('tools/biv150_konly_decode_factory.py', 'tools/biv150_konly_mixed_factory.py', 'tools/kernels/biv150_konly_decode_support.py')}
    entries.update({name: PREFILL for name in ('tools/biv150_konly_prefill_factory.py', 'tools/biv150_konly_decode_quant_reference.py', 'tools/biv150_konly_i8_qk_reference.py', 'docs/evidence/konly-prefill-reader-20261027/source-inputs.tar.gz', 'docs/evidence/konly-prefill-reader-20261027/source-inputs.json')})
    inputs = {}; payload = {}; proofs = {}
    with tempfile.TemporaryDirectory(prefix='cjj-encoded-source-') as temp:
        base = Path(temp)
        for name, ref in entries.items():
            data = git(ref, name); p = base / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
            inputs[name] = dict(commit=ref, bytes=len(data), sha256=sha(data))
        sys.path.insert(0, str(base / 'tools'))
        try:
            decode = importlib.import_module('biv150_konly_decode_factory')
            mixed = importlib.import_module('biv150_konly_mixed_factory')
            prefill = importlib.import_module('biv150_konly_prefill_factory')
            current = git(MAIN, 'vllm_fl/patches/biv150_gqa_decode.py').decode()
            main_source = git(GQA_MATH_BASE, 'vllm_fl/patches/biv150_gqa_decode.py').decode()
            def span(text,name):
                node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
                start = min([node.lineno]+[n.lineno for n in node.decorator_list])-1
                return ''.join(text.splitlines(True)[start:node.end_lineno])
            names = ('_gqa_decode_attention','_reduce_partials')
            assert all(span(current,n)==span(main_source,n) for n in names)
            proofs['current_main_GQA'] = dict(current_main=MAIN,current_source_sha256=sha(current),
                certified_factory_source_base=GQA_MATH_BASE,factory_source_sha256=sha(main_source),
                decorated_kernel_and_reducer_byte_identical=list(names),
                new_native_host_entry_not_covered_by_math_proof=True,
                new_host_format_boundary_checked_separately=True)
            source, proofs['decode'] = decode.build(main_source, 'inline')
            old = 'from cjj_konly_decode_support import quantize_bf16_rne\n'
            new = 'from vllm_fl.patches.biv150_konly_encoded_writer import quantize_bf16_rne\n'
            assert source.count(old) == 1
            payload['decode'] = source.replace(old, new, 1)
            modules, proofs['mixed'] = mixed.build(main_source)
            assert modules['k8'].count(old) == 1
            payload['mixed'] = modules['k8'].replace(old, new, 1)
            sources = prefill.inputs()
            for profile in ('normalized2d128', 'q32', 'multi2', 'raw3d'):
                payload[profile], proofs[profile] = prefill.build(profile, sources)
            raw = payload['raw3d']
            old_guard = 'IS_3D == True and BLOCK_M == 16 and BLOCK_Q == 2 and TILE_SIZE == 16'
            assert raw.count(old_guard) == 1
            raw2d = raw.replace(old_guard, old_guard.replace('True', 'False'), 1).replace('def kernel_cjj_konly_raw3d(', 'def kernel_cjj_konly_raw2d(', 1)
            assert raw2d.replace(old_guard.replace('True', 'False'), old_guard, 1).replace('def kernel_cjj_konly_raw2d(', 'def kernel_cjj_konly_raw3d(', 1) == raw
            payload['raw2d'] = raw2d
            proofs['raw2d'] = dict(derived_from='raw3d', only_profile_guard_and_name_changed=True, original_sdk_math_unchanged=True, actual_device_verified=False)
            original = payload['normalized2d128']
            anchor = '    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:\n        return\n'
            addition = anchor + '\n    if cur_batch_query_len == 1:\n        return\n'
            assert original.count(anchor) == 1
            skip = original.replace(anchor, addition, 1).replace('def kernel_cjj_konly_normalized2d128(', 'def kernel_cjj_konly_mixedprefill(', 1)
            assert skip.replace(addition, anchor, 1).replace('def kernel_cjj_konly_mixedprefill(', 'def kernel_cjj_konly_normalized2d128(', 1) == original
            payload['mixedprefill'] = skip
            proofs['mixedprefill'] = dict(derived_from='normalized2d128', exact_added_decode_guard_inverse=True, numerical_body_unchanged=True, actual_device_verified=False)
        finally:
            sys.path.remove(str(base / 'tools'))
            for name in ('biv150_konly_decode_factory', 'biv150_konly_mixed_factory', 'biv150_konly_prefill_factory'): sys.modules.pop(name, None)
    for unused in ('mixed', 'raw2d', 'mixedprefill'):
        payload.pop(unused, None); proofs.pop(unused, None)
    payload['views'] = git('89a12565ed50931a6e5536313a00b1e768ace1ad', 'tools/biv150_konly_view_broker.py').decode()
    payload['writer'] = git(WRITER, 'tools/kernels/biv150_konly_strided_writer.py').decode()
    original_codec = ast.unparse(next(n for n in ast.parse(git(MIXED, 'tools/kernels/biv150_konly_decode_support.py')) .body if isinstance(n, ast.FunctionDef) and n.name == 'quantize_bf16_rne'))
    writer_codec = ast.unparse(next(n for n in ast.parse(payload['writer']).body if isinstance(n, ast.FunctionDef) and n.name == 'quantize_bf16_rne'))
    assert writer_codec == original_codec
    for suffix, name in (('hoist_proposal','biv150_konly_service_q32_hoist.py'), ('hoist_native','biv150_konly_service_q32_native.py')):
        source = git(HOIST, 'tools/' + name)
        payload[suffix] = source.decode()
        inputs['tools/' + name] = dict(commit=HOIST, bytes=len(source), sha256=sha(source))
    output = {}
    for name, source in payload.items():
        compile(source, '<syntax-only-no-execution>', 'exec')
        path = 'vllm_fl/patches/biv150_konly_encoded_' + name + '.py'
        data = source.encode(); output[name] = dict(path=path, bytes=len(data), sha256=sha(data))
        if write: (ROOT / path).write_bytes(data)
        else: assert (ROOT / path).read_bytes() == data, name
    result = dict(main=MAIN, dependencies=dict(mixed85=MIXED, prefill80_87=PREFILL, writer83=WRITER, service_q32_hoist90=HOIST), input_files=inputs,
        generated=output, source_proofs=proofs, codec_function_AST_identical=True, GPU_imports_executed=False,
        service_activated=False, dynamic_I8_fragment_hoist_proved=False, all_service_profiles_device_verified=False,
        specialized_mixed85_pair_not_installed=True)
    target = ROOT / 'docs/evidence/konly-encoded-service-20261027/source-provenance.json'
    if write: target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(result, indent=2) + '\n')
    else: assert json.loads(target.read_text()) == result
    packaged = ROOT / 'vllm_fl/patches/biv150_konly_encoded_sources.json'
    if write: packaged.write_text(json.dumps(result, indent=2) + '\n')
    else: assert json.loads(packaged.read_text()) == result
    assert not {'torch', 'triton', 'vllm', 'numpy'}.intersection(sys.modules)
    return dict(passed=True, generated_modules=len(output), cpu_only=True, imported_GPU_framework=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--write', action='store_true'); a = p.parse_args()
    print(json.dumps(run(a.write), indent=2))
