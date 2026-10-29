#!/usr/bin/env python3
"""Narrow main117 merge check: existing mixed CPU routing with identity adapters."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
FROZEN = '01dfe84028560b1a982dac33bc788e875ce9cf1b'
MAIN = '56f88737fcf3b940760b9ca3027a925cd5a7d77d'
OLD = '8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
NEW = '5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def original(ref, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', ref + ':' + path])


def adapt(name, edits):
    path = 'tools/' + name + '.py'
    old = (ROOT / path).read_text()
    assert old.encode() == original(FROZEN, path)
    source, applied = old, []
    for before, after in edits:
        count = source.count(before)
        assert count > 0
        source = source.replace(before, after)
        applied.append(dict(before=before, after=after, count=count))
    inverse = source
    for edit in reversed(applied):
        assert inverse.count(edit['after']) == edit['count']
        inverse = inverse.replace(edit['after'], edit['before'])
    assert inverse == old
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__ + ':identity-only', 'exec'), vars(module))
    return module, dict(original_sha256=sha(old.encode()), adapted_sha256=sha(source.encode()),
                        inverse_byte_exact=True, edits=applied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--old-compile-summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.library.read_bytes()) == NEW
    # Only the two accepted identity substitutions may differ from frozen118.
    semantic = (ROOT / 'vllm_fl/compile_semantics.py').read_bytes()
    inverse = semantic.replace(NEW.encode(), OLD.encode()).replace(
        b'cjj_native_k8_klookahead_probe::produce', b'cjj_native_k8_v_overlap_probe::produce')
    assert inverse == original(FROZEN, 'vllm_fl/compile_semantics.py')
    identities = {}
    for path in ('vllm_fl/patches/biv150_konly_encoded_native.py',
                 'vllm_fl/patches/biv150_konly_encoded_sources.json'):
        b = (ROOT / path).read_bytes()
        assert b == original(MAIN, path)
        identities[path] = sha(b)
    for path in ('vllm_fl/patches/biv150_compute_order.py',
                 'vllm_fl/patches/biv150_mixed_decode.py',
                 'vllm_fl/patches/biv150_konly_encoded_mixed_native.py',
                 'vllm_fl/patches/biv150_konly_encoded_service.py',
                 'vllm_fl/patches/biv150_konly_encoded_runtime.py'):
        assert (ROOT / path).read_bytes() == original(FROZEN, path)
    runner = (ROOT / 'vllm_fl/worker/model_runner.py').read_bytes()
    assert runner == original(FROZEN, 'vllm_fl/worker/model_runner.py') == original(MAIN, 'vllm_fl/worker/model_runner.py')
    order_tree = ast.parse((ROOT / 'vllm_fl/patches/biv150_compute_order.py').read_bytes())
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in order_tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in ('EXPECTED_RUNNER', 'SOURCES')}
    assert constants['EXPECTED_RUNNER'] == sha(runner)
    assert constants['SOURCES']['vllm_fl.patches.biv150_mixed_decode'] == sha(
        (ROOT / 'vllm_fl/patches/biv150_mixed_decode.py').read_bytes())
    import biv150_native_k8_mixed_factory as factory
    import compile_biv150_native_k8_mixed as build
    proof = factory.verify()
    plan = build.cpu_plan()
    assert factory.BASE == 'a455a3de364051fbea80226870c239de3a48e716' and factory.HOST_BASE == MAIN
    for role in factory.FILES:
        p = factory.path(role)
        assert p.read_bytes() == original(FROZEN, str(p.relative_to(ROOT)))
    assert (ROOT / 'tools/compile_biv150_native_k8_mixed.py').read_bytes() == original(FROZEN, 'tools/compile_biv150_native_k8_mixed.py')
    old_compile = json.loads(args.old_compile_summary.read_bytes())
    assert old_compile['completed'] is old_compile['compile_passed'] is True and not old_compile['errors']
    assert old_compile['source_commit'] == FROZEN
    same_compiled_fields = ('source_files', 'explicit_host_flags', 'explicit_cuda_flags',
                            'symbol_renames', 'frontend_sha256')
    assert all(old_compile[key] == plan[key] for key in same_compiled_fields)
    assert old_compile['factory_sha256'] != plan['factory_sha256']
    delta = sorted(key for key in set(proof) | set(old_compile['source_transform'])
                   if proof.get(key) != old_compile['source_transform'].get(key))
    assert delta == ['host_reverse_base']
    for role in factory.FILES:
        p = factory.path(role)
        assert (args.old_compile_summary.parent / p.name).read_bytes() == p.read_bytes()
    # Execute the actual original admission guard in isolation on real saved
    # compile metadata. No fabricated successful receipt and no array replay.
    import biv150_native_k8_mixed_numeric_contract as contract
    contract_text = Path(contract.__file__).read_text()
    admission = next(n for n in ast.parse(contract_text).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'admission')
    guards = [n for n in admission.body if isinstance(n, ast.Expr)
              and "j['source_transform']" in ast.unparse(n)]
    assert len(guards) == 1
    guard_source = ast.get_source_segment(contract_text, guards[0])
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=guards, type_ignores=[])),
                     '<unchanged compile admission>', 'exec'),
             dict(need=contract.need, j=old_compile, factory=factory, build=build))
    except ValueError as error:
        assert str(error) == 'PR118 component: actual exact source/options'
    else:
        raise AssertionError('old compile proof unexpectedly admitted by merged host source')
    old_admission = dict(old_summary_sha256=sha(args.old_compile_summary.read_bytes()),
        old_source_commit=old_compile['source_commit'], equal_fields=list(same_compiled_fields),
        raw_compile_inputs_byte_equal=True, source_transform_differences=delta,
        factory_sha_changed=True, actual_original_guard=guard_source,
        old_compile_refused_before_GPU=True, new_qualification_inferred=False)
    # Existing opaque CPU Tensor/dispatcher tests; only their old parent identity
    # is adapted in memory. The old checker sources remain unchanged on disk.
    service, service_proof = adapt('check_biv150_native_k8_service_cpu', [
        ('cjj_native_k8_v_overlap_probe', 'cjj_native_k8_klookahead_probe'),
        ('launch_cjj_native_k8_v_overlap_', 'launch_cjj_native_k8_klookahead_')])
    service.SO = args.library
    fingerprint, fingerprint_proof = adapt('check_biv150_native_k8_fingerprint_cpu', [
        ('cjj_native_k8_v_overlap_probe', 'cjj_native_k8_klookahead_probe'), (OLD, NEW),
        ('22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd', OLD)])
    fingerprint.LIBRARY, fingerprint.HELPER = args.library, args.helper
    import check_biv150_native_k8_mixed_cpu as mixed
    with contextlib.redirect_stdout(io.StringIO()):
        results = mixed.run()
    native = mixed.load('native')
    elf = service.actual_elf(args.library, native)
    assert results['passed'] and results['default_OFF_and_pending_rejection']
    pending = mixed.load('mixed_native').QUALIFICATION
    assert all(value is None for name, value in pending.items() if name.endswith('sha256'))
    assert pending['service_verified'] is False
    assert not {'torch', 'triton', 'vllm', 'vllm_fl'}.intersection(sys.modules)
    report = dict(completed=True, errors=[], CPU_only=True, frozen118=FROZEN, accepted_main=MAIN,
        script_sha256=sha(Path(__file__).read_bytes()), new_parent_library_sha256=NEW,
        accepted117_module_manifest_exact=identities, compile_semantics_two_identity_inverse_exact=True,
        runner_and_mixed_source_pins_exact=True, mixed_production_unchanged=True,
        GPU_compile_inputs_and_frontend_unchanged=True, source_transform=proof,
        old_compile_admission=old_admission,
        compile_plan=plan, fixture_identity_adapters=dict(service=service_proof, fingerprint=fingerprint_proof),
        actual_parent_library_ELF_only=elf, routing_and_fingerprint=results, qualification=pending,
        GPU_executed=False, new_mixed_numeric_qualification=False, mixed_service_verified=False)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(sha(args.out.read_bytes()))


if __name__ == '__main__':
    main()
