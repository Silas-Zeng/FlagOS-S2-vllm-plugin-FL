"""Fixed-path compact spec. Import defines metadata only; registration is explicit."""
from dataclasses import dataclass
import importlib
import os
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from .biv150_konly_compact_identity import ENV, FORMAT, need, verify_core, verify_package

@dataclass(frozen=True, kw_only=True)
class CompactFullAttentionSpec(FullAttentionSpec):
    format_id: str = FORMAT

    def __post_init__(self):
        super().__post_init__()
        if self.format_id != FORMAT:
            raise ValueError("Unknown compact format")
        if (self.block_size, self.num_kv_heads, self.head_size, self.head_size_v, self.dtype) != (16, 2, 128, 128, torch.bfloat16):
            raise ValueError("Only BF16 MiniCPM page16/head128 compact storage")
        if self.kv_quant_mode != KVQuantMode.NONE or self.page_size_padded is not None or self.sliding_window is not None or self.attention_chunk_size is not None or self.non_causal:
            raise ValueError("No stock quantization, padding, hybrid or noncausal compact scope")

    @property
    def real_page_size_bytes(self):
        return 12544

    @classmethod
    def merge(cls, specs):
        if not specs or any(type(spec) is not cls for spec in specs):
            raise ValueError("Compact and stock formats must never merge")
        return super().merge(specs)

    def is_uniform_with_collection(self, specs):
        return bool(specs) and all(type(spec) is type(self) for spec in specs.values())


def ensure_registered(config=None):
    need(os.environ.get(ENV, '0') == '1', 'registration while compact is disabled')
    need(os.environ.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE', '0') != '1', 'two encoded formats enabled')
    verify_package()
    modules = verify_core(('vllm.v1.kv_cache_interface', 'vllm.v1.kv_cache_spec_registry',
                           'vllm.v1.core.single_type_kv_cache_manager'))
    registry = modules['vllm.v1.kv_cache_spec_registry']
    api = registry.KVCacheSpecRegistry
    api._ensure_registered(config)
    builtins = registry._REGISTRY_KVCACHESPEC_LIST
    full_manager = modules['vllm.v1.core.single_type_kv_cache_manager'].FullAttentionManager
    need(FullAttentionSpec in builtins and builtins[FullAttentionSpec].manager_class is full_manager
         and builtins[FullAttentionSpec].uniform_type_base_spec is FullAttentionSpec,
         'built-in full-attention registration missing or replaced')
    manager = importlib.import_module('vllm_fl.patches.biv150_konly_compact_manager').CompactFullAttentionManager
    api.register(CompactFullAttentionSpec, manager, CompactFullAttentionSpec)
    entry = builtins[CompactFullAttentionSpec]
    need(entry.manager_class is manager and entry.uniform_type_base_spec is CompactFullAttentionSpec,
         'compact registry identity changed')
    return manager
