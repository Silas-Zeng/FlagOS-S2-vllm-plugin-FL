"""Replay address/copy contracts and the pinned installed KV spec's byte arithmetic."""
import ast
import copy
from dataclasses import dataclass, fields, replace
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

from biv150_konly_compact_page_contract import CompactPage, FORMAT, INSTALLED_SPEC_SHA, repack_cpu_pages

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/konly-compact-page-20261027'


def pinned_specs():
    raw = (EVIDENCE / 'target-kv-cache-interface.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == INSTALLED_SPEC_SHA
    tree = ast.parse(raw)
    names = ('KVCacheSpec', 'AttentionSpec', 'FullAttentionSpec')
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert [n.name for n in nodes] == list(names)
    @dataclass(frozen=True)
    class Mode:
        is_nvfp4: bool = False
        is_per_token_head: bool = False
    mode = Mode()
    module = ModuleType('_cjj_compact_spec_cpu_snapshot')
    scope = module.__dict__
    scope.update(dataclass=dataclass, fields=fields, replace=replace, copy=copy,
                 KVQuantMode=SimpleNamespace(NONE=mode), MLAAttentionSpec=type('MLAAttentionSpec', (), {}),
                 torch=SimpleNamespace(float32='float32'), get_dtype_size=lambda _: 2,
                 cdiv=lambda a, b: (a + b - 1) // b)
    sys.modules[module.__name__] = module
    selected = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(selected), '<pinned-installed-spec-classes>', 'exec'), scope)
    return scope['FullAttentionSpec']


def run():
    assert not {'torch', 'triton', 'vllm', 'numpy'}.intersection(sys.modules)
    p = CompactPage()
    assert (p.code_bytes, p.scale_bytes, p.value_offset, p.page_bytes) == (4096, 128, 4352, 12544)
    assert p.typed_views()['scale_full']['page_stride_elements'] == 3136
    assert p.typed_views()['value']['storage_offset_elements'] == 2176
    addresses = set()
    for page in range(3):
        for slot in range(16):
            for head in range(2):
                for dim in range(128):
                    for role, width in [('code', 1), ('value', 2)]:
                        address = p.address(role, page, slot, head, dim)
                        assert page * p.page_bytes <= address < (page + 1) * p.page_bytes
                        for byte in range(address, address + width):
                            assert byte not in addresses
                            addresses.add(byte)
                address = p.address('scale', page, slot, head)
                assert address % 4 == 0
                for byte in range(address, address + 4):
                    assert byte not in addresses
                    addresses.add(byte)
    assert len(addresses) == 3 * 12416

    # Exercise all 65,536 BF16 bit patterns as raw V payload, twice, over 32 pages.
    legacy = bytearray([0xA5]) * (32 * 16384)
    vbits = b''.join(struct.pack('<H', n) for n in range(65536)) * 2
    codes = bytes(((-127 + n % 255) & 255) for n in range(4096))
    scales = b''.join(struct.pack('<f', (n + 1) / 127) for n in range(32))
    for page in range(32):
        base = page * 16384
        legacy[base:base + 4096] = codes
        legacy[base + 4096:base + 4224] = scales
        legacy[base + 8192:base + 16384] = vbits[page * 8192:(page + 1) * 8192]
    packed = repack_cpu_pages(legacy)
    assert len(packed) == 401408
    for page in range(32):
        base = page * p.page_bytes
        assert packed[base:base + 4096] == codes
        assert packed[base + 4096:base + 4224] == scales
        assert packed[base + 4224:base + 4352] == bytes(128)
        assert packed[base + 4352:base + 12544] == vbits[page * 8192:(page + 1) * 8192]

    FullSpec = pinned_specs()

    class ProposedSpec(FullSpec):
        def __post_init__(self):
            super().__post_init__()
            if (self.block_size, self.num_kv_heads, self.head_size, self.head_size_v, self.dtype) != (16, 2, 128, 128, 'bf16'):
                raise ValueError('CPU proposal supports only this exact geometry')

        @property
        def real_page_size_bytes(self):
            return p.page_bytes

        @classmethod
        def merge(cls, specs):
            if not specs or any(type(spec) is not cls for spec in specs):
                raise ValueError('Never merge stock and private encoded formats')
            return super().merge(specs)

        def is_uniform_with_collection(self, specs):
            return bool(specs) and all(type(spec) is type(self) for spec in specs.values())

    args = dict(block_size=16, num_kv_heads=2, head_size=128, head_size_v=128, dtype='bf16')
    stock = FullSpec(**args)
    compact = ProposedSpec(**args)
    assert stock.page_size_bytes == 16384 and compact.page_size_bytes == 12544
    assert ProposedSpec.merge([compact, compact]).page_size_bytes == 12544
    assert type(compact.copy_with_new_block_size(16)) is ProposedSpec
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=131072), parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1))
    assert compact.max_memory_usage_bytes(config) == 8192 * 12544
    rejected = 0
    for call in [lambda: ProposedSpec.merge([compact, stock]), lambda: ProposedSpec.merge([]),
                 lambda: compact.copy_with_new_block_size(32),
                 lambda: ProposedSpec(**dict(args, page_size_padded=12000)).page_size_bytes,
                 lambda: CompactPage(alignment=128), lambda: p.address('code', 0, 16, 0),
                 lambda: p.address('code', 0.5, 0, 0), lambda: p.address('value', True, 0, 0),
                 lambda: p.address('scale', 0, 0, 0, 1), lambda: repack_cpu_pages(bytes(16385))]:
        try:
            call()
        except (ValueError, AssertionError):
            rejected += 1
        else:
            raise AssertionError('Unsupported contract was admitted')
    assert not compact.is_uniform_with_collection({'a': compact, 'b': stock})
    assert compact.is_uniform_with_collection({'a': compact, 'b': compact})
    assert not {'torch', 'triton', 'vllm', 'numpy'}.intersection(sys.modules)
    return dict(cpu_contract_passed=True, format_id=FORMAT, source_sha256=INSTALLED_SPEC_SHA,
                page_bytes=p.page_bytes, live_payload_bytes=12416, padding_bytes=128,
                typed_views=p.typed_views(), address_bytes_checked=len(addresses),
                raw_v_bf16_patterns=65536, repacked_pages=32, rejected_contracts=rejected,
                preserved_semantic_head_size=compact.head_size, pinned_spec_property_and_merge_replayed=True,
                stock_private_spec_mix_rejected=True,
                budget_example=p.capacity(1024**3), framework_registry_verified=False,
                backend_reshape_verified=False, allocator_verified=False, zeroer_verified=False,
                lifecycle_verified=False, gpu_executed=False, measured_capacity=False,
                model_precision_verified=False, performance_measured=False, service_activated=False)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
