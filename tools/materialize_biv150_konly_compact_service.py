"""Materialize private service sources from immutable team commits; no imports/JIT."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import biv150_konly_compact_kv_bridge as bridge

ROOT = Path(__file__).resolve().parents[1]
PARENT = '2487e75aaf367b0924b12a5d53d15c1dff971be9'
ENV = 'VLLM_FL_BIV150_KONLY_COMPACT_SERVICE'


def raw(path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', PARENT + ':' + path])


def replace(text, old, new, count=1):
    assert text.count(old) == count, old
    return text.replace(old, new)


def run():
    directory = ROOT / 'vllm_fl/patches'
    parent = json.loads(raw('vllm_fl/patches/biv150_konly_encoded_sources.json'))
    generated = {}; proofs = {}
    for role in ('writer', 'decode', 'normalized2d128', 'q32', 'multi2', 'raw3d', 'views', 'runtime'):
        entry = parent['generated'].get(role)
        path = entry['path'] if entry else 'vllm_fl/patches/biv150_konly_encoded_runtime.py'
        data = raw(path)
        if entry:
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
        source = data.decode()
        if role in ('views', 'runtime'):
            output, proof = bridge.parent_host_source(source, role)
        else:
            output, proof = bridge.kernel_source(source, role)
        if role == 'runtime':
            output = replace(output, "        result['q32_hoist'] = dict(enabled=False) if self.q32_hoist is None else self.q32_hoist.stats()",
                "        result['q32_hoist'] = dict(enabled=False) if self.q32_hoist is None else self.q32_hoist.stats()\n"
                "        result['compact_host'] = self.compact_host.stats()")
            proof = dict(geometry_transform=proof, metadata_only_allocation_export_added=True,
                         complete_service_binding=False)
        target = directory / Path(path).name.replace('_encoded_', '_compact_')
        target.write_text(output)
        generated[role] = dict(path=str(target.relative_to(ROOT)), sha256=hashlib.sha256(output.encode()).hexdigest())
        proofs[role] = proof

    source = raw('vllm_fl/patches/biv150_konly_encoded_service.py').decode()
    assert hashlib.sha256(source.encode()).hexdigest() == 'd21c1268b69b3ec1736cd8f196e89371db689be1b664590c7913da16b1bc1f81'
    # Remove the explicitly unsupported 16KiB Q32 hoist before renaming modules.
    tree = ast.parse(source); matches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
        and "env.get('VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'" in ast.unparse(n.test)]
    assert len(matches) == 1
    node = matches[0]; lines = source.splitlines(keepends=True)
    source = ''.join(lines[:node.lineno-1] + lines[node.end_lineno:])
    source = source.replace('biv150_konly_encoded_', 'biv150_konly_compact_')
    source = source.replace('VLLM_FL_BIV150_KONLY_ENCODED_STATS_PATH', 'VLLM_FL_BIV150_KONLY_COMPACT_STATS_PATH')
    source = source.replace('konly-encoded-final-', 'konly-compact-final-')
    source = replace(source, "ENV = 'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE'", "ENV = '" + ENV + "'")
    source = replace(source, "    need(_STATE is None, 'install once in a fresh worker')",
        "    need(env.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE', '0') != '1', 'two cache formats enabled')\n"
        "    need(env.get('VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST', '0') != '1', 'compact Q32 hoist unsupported')\n"
        "    from .biv150_konly_compact_identity import verify_package\n"
        "    verify_package()\n"
        "    need(_STATE is None, 'install once in a fresh worker')")
    source = replace(source, '    runtime = runtime_module.Runtime(torch, broker, gpu, gqa)',
        '    runtime = runtime_module.Runtime(torch, broker, gpu, gqa)\n'
        '    from .biv150_konly_compact_host import Host\n'
        '    host = Host(runner_class, torch, runtime, attention.current_platform)\n'
        '    runtime.compact_host = host')
    source = replace(source, "runtime.initialize(runner_class.initialize_kv_cache)",
                     "host.initialize(runtime.initialize(runner_class.initialize_kv_cache))")
    source = replace(source, '    runtime.verified_installed_sources = dict(PINS)',
                     '    runtime.verified_installed_sources = {**PINS, **host.verified_core_pins}')
    source = replace(source, "    seen = set(); old = []; attrs = []",
        "    jobs.append((vars(host.module), '_reshape_attention_kv_cache', host.reshape_hook))\n"
        "    attributes.append((runner_class, 'get_kv_cache_spec', host.get_spec_hook))\n"
        "    seen = set(); old = []; attrs = []")
    source = replace(source, '    parents = {}',
        "    host = state.compact_host\n"
        "    need(host.initialized, 'compact host epoch not initialized')\n"
        "    host.receipt(host.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec)\n"
        '    parents = {}')
    source += '''

def compact_spec_is_owned(spec):
    need(_STATE is not None, 'compact runtime missing')
    _STATE.compact_host.receipt(spec)
    return True
'''
    (directory / 'biv150_konly_compact_service.py').write_text(source)

    worker = raw('vllm_fl/worker/model_runner.py').decode()
    marker = '# Private encoded-cache diagnostic: install once before allocation/capture; default off.'
    guard = '''# Reject two cache formats or an unlicensed old hoist before either installer.
if __import__("os").environ.get("VLLM_FL_BIV150_KONLY_COMPACT_SERVICE", "0") == "1":
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        raise RuntimeError("Compact cache supports only the qualified Iluvatar source path")
    if (__import__("os").environ.get("VLLM_FL_BIV150_KONLY_ENCODED_SERVICE", "0") == "1"
            or __import__("os").environ.get("VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST", "0") == "1"):
        raise RuntimeError("Compact cache conflicts with old format/hoist")

'''
    worker = replace(worker, marker, guard + marker)
    worker += '''
# Private compact allocation: fixed-module metadata and encoded routes, default off.
if (getattr(current_platform, "vendor_name", None) == "iluvatar"
        and __import__("os").environ.get("VLLM_FL_BIV150_KONLY_COMPACT_SERVICE", "0") == "1"):
    from vllm_fl.patches.biv150_konly_compact_service import install as _install_konly_compact
    print("BIV150_KONLY_COMPACT_INSTALL", _install_konly_compact(ModelRunnerFL), flush=True)
'''
    # Preserve the target runner's CRLF convention.
    worker = worker.replace('\r\n', '\n').replace('\n', '\r\n')
    (ROOT / 'vllm_fl/worker/model_runner.py').write_bytes(worker.encode())
    worker_sha = hashlib.sha256(worker.encode()).hexdigest()

    c3 = raw('vllm_fl/patches/biv150_compute_order.py').decode()
    c3 = replace(c3, 'EXPECTED_RUNNER = "b4bf3ec5cb7b46392452c37e2ac91292d7c40522ad711ce3a8be3eb55a418d7d"',
                 'EXPECTED_RUNNER = "' + worker_sha + '"')
    c3 = replace(c3, '    if os.environ.get("VLLM_FL_BIV150_KONLY_ENCODED_SERVICE", "0") == "1":',
        '    if os.environ.get("VLLM_FL_BIV150_KONLY_COMPACT_SERVICE", "0") == "1":\r\n'
        '        compact = sys.modules.get("vllm_fl.patches.biv150_konly_compact_service")\r\n'
        '        if compact is None:\r\n'
        '            raise ValueError("compact dispatch installed source is missing")\r\n'
        '        receipt = compact.compute_order_receipt(attention, normalized_kernel)\r\n'
        '        raw_kernel, normalized_kernel = receipt["parent_raw"], receipt["parent_normalized"]\r\n'
        '    elif os.environ.get("VLLM_FL_BIV150_KONLY_ENCODED_SERVICE", "0") == "1":')
    c3 = replace(c3, '        or type(spec) is not spec_module.FullAttentionSpec',
        '        or not _compact_or_stock_spec(spec, spec_module)')
    c3 += '''

def _compact_or_stock_spec(spec, spec_module):
    if os.environ.get("VLLM_FL_BIV150_KONLY_COMPACT_SERVICE", "0") != "1":
        return type(spec) is spec_module.FullAttentionSpec
    compact = sys.modules.get("vllm_fl.patches.biv150_konly_compact_service")
    if compact is None:
        raise ValueError("compact spec receipt unavailable")
    return compact.compact_spec_is_owned(spec)
'''
    c3 = c3.replace('\r\n', '\n').replace('\n', '\r\n')
    (directory / 'biv150_compute_order.py').write_bytes(c3.encode())

    sem = raw('vllm_fl/compile_semantics.py').decode()
    marker = '    encoded = None'
    sem = replace(sem, marker,
        '    compact = None\r\n'
        '    if env.get("VLLM_FL_BIV150_KONLY_COMPACT_SERVICE", "0") == "1":\r\n'
        '        if features["VLLM_FL_BIV150_KONLY_ENCODED_SERVICE"] or env.get("VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST", "0") == "1":\r\n'
        '            raise ValueError("compact format/old-hoist conflict")\r\n'
        '        directory = package / "patches"\r\n'
        '        compact_paths = sorted(directory.glob("biv150_konly_compact_*.py")) + [directory / "biv150_konly_compact_sources.json"]\r\n'
        '        compact = {"format": "' + bridge.FORMAT + '", "page_bytes": 12544, "q32_hoist_enabled": False,\r\n'
        '                   "implementation_sha256": _digest({path.name: _sha(path) for path in compact_paths})}\r\n'
        '        facts["konly_compact"] = compact\r\n' + marker)
    sem = replace(sem, '    return result\r\n\r\n\r\nclass _WithCompileSemantics:',
        '    if compact is not None:\r\n        result["konly_compact"] = compact\r\n'
        '    return result\r\n\r\n\r\nclass _WithCompileSemantics:')
    (ROOT / 'vllm_fl/compile_semantics.py').write_bytes(sem.encode())

    for role in ('identity', 'spec', 'manager', 'host', 'service'):
        path = directory / ('biv150_konly_compact_' + role + '.py')
        generated[role] = dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(main=bridge.BASE, parent89=PARENT, contract88=bridge.CONTRACT88,
        format=bridge.FORMAT, page_bytes=12544, generated=generated, address_source_proofs=proofs,
        binding_files={name:hashlib.sha256((ROOT/'vllm_fl'/name).read_bytes()).hexdigest()
                       for name in ('worker/model_runner.py','patches/biv150_compute_order.py','compile_semantics.py')},
        q32_hoist_supported=False, service_numeric_verified=False, capacity_reclaimed=False,
        actual_IR_verified=False, performance_verified=False, precision_gate=False)
    (directory / 'biv150_konly_compact_sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for path in list(directory.glob('biv150_konly_compact_*.py')) + [ROOT/'vllm_fl/compile_semantics.py']:
        ast.parse(path.read_bytes())
    print(json.dumps(dict(materialized=True, cpu_source_only=True, parent=PARENT,
                         modules=len(generated), worker_sha256=worker_sha), indent=2))


if __name__ == '__main__':
    run()
