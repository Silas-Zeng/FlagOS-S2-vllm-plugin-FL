"""Cold source/format identity checks. Stdlib only; no registration on import."""
import hashlib
import importlib
import json
from pathlib import Path

ENV = 'VLLM_FL_BIV150_KONLY_COMPACT_SERVICE'
FORMAT = 'cjj-konly-compact12544-rn32-rne-v1'
PAGE_BYTES = 12544
CORE_PINS = {
    'vllm.v1.kv_cache_interface': '7e3c220998267190a622e216b2b125306807cf8be3ec1e9a8e84fe6e7a293003',
    'vllm.v1.kv_cache_spec_registry': '16b3f6458538910f72cdac48554c20432c7ac0e5122e91c8a6219a1ec25f44fe',
    'vllm.v1.core.single_type_kv_cache_manager': '22e3677acf0e769461bdbf278df031d6e362c97aa1ba7b862c071b76d47d8914',
    'vllm.v1.core.kv_cache_utils': '3ca561818019cef84c43c6d3839627145fd049f9479d40fc733ae142d6d71df3',
    'vllm.v1.worker.utils': '36cae980896d5607ce238606ec6c053eff17b7597a0c9185549f94583c1b9de6',
    'vllm.v1.worker.gpu.attn_utils': 'a351178df66fb5364cbe663ba7f25346c6317927fcc29a8c00f8972a23440ace',
    'vllm.v1.worker.kv_connector_model_runner_mixin': 'd140e79e93b8670a2100b8b365606d3af402430a59c990ef65820da987ef4974',
    'vllm.v1.executor.uniproc_executor': '4625c8e59aef1482600b02204701d348d7f716427581d3cf23b06bff9b056abe',
    'vllm.v1.executor.abstract': '419002195d13cf050c3ea9705d18a12456c890c27c470a1efadce7de27ded0b1',
    'vllm.v1.engine.core': '27b23827a86fd488f1f7cc9a2722fe76d284087ca7cfefef5a30c5e74f21bc71',
    'vllm.v1.worker.gpu_worker': '271c3e2eaab7f1ca36fcd7c8e1620db9db4705f46ba5b3257dd6a3937f56b7df',
    'vllm.v1.serial_utils': 'd75c17a098231b2398cbdb7da6f9983e5b9229c0ba138b2874934d45fafd1f28',
    'vllm.v1.worker.worker_base': '7da44338c2645ebf03d23394e452b31a8e3da1011fd1b42fcfcccfe99551b3fe',
    'vllm.config.cache': '0c71c5c04d8e87e88377cb8b4c2966d956be2744afef72acdbd5458a9322d5a2',
    'vllm.config.offload': '6f2db3d9a2730588473d5155811690c6241b167a21dcb62fbbd9ebb6386de42c',
}


def need(value, message):
    if not value:
        raise RuntimeError('K-only compact: ' + message)


def verify_module(name, pin):
    module = importlib.import_module(name)
    need(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == pin,
         'installed source mismatch: ' + name)
    return module


def verify_core(names=None):
    return {name: verify_module(name, CORE_PINS[name]) for name in (CORE_PINS if names is None else names)}


def verify_package():
    directory = Path(__file__).resolve().parent
    record = json.loads((directory / 'biv150_konly_compact_sources.json').read_text())
    need(record['format'] == FORMAT and record['page_bytes'] == PAGE_BYTES, 'source manifest format')
    need(set(record['generated']) == {'writer','decode','normalized2d128','q32','multi2','raw3d',
         'views','runtime','identity','spec','manager','host','service','hoist_proposal','q32_adapter'}, 'complete compact source manifest')
    for name, entry in record['generated'].items():
        path = directory / Path(entry['path']).name
        need(path.parent == directory and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'],
             'private source mismatch: ' + name)
    need(set(record['hoist_dependencies']) == {'biv150_konly_encoded_q32_adapter.py',
         'biv150_konly_encoded_hoist_native.py'}, 'complete reused hoist dependencies')
    for name, pin in record['hoist_dependencies'].items():
        need(hashlib.sha256((directory/name).read_bytes()).hexdigest() == pin,
             'reused hoist dependency source mismatch: ' + name)
    need(set(record['binding_files']) == {'worker/model_runner.py', 'patches/biv150_compute_order.py',
         'compile_semantics.py'}, 'complete host binding source manifest')
    for relative, pin in record['binding_files'].items():
        need(hashlib.sha256((directory.parent/relative).read_bytes()).hexdigest() == pin,
             'binding source mismatch: ' + relative)
    return record
