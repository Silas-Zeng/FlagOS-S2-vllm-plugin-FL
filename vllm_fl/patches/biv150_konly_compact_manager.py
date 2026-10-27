"""Fixed-path full-attention manager with both compact zero notifications."""
from __future__ import annotations
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, TQFullAttentionSpec, MLAAttentionSpec, HiddenStateCacheSpec
from vllm.utils.math_utils import cdiv
from .biv150_konly_compact_spec import CompactFullAttentionSpec

class CompactFullAttentionManager(FullAttentionManager):

    def allocate_external_computed_blocks(self, request_id: str, num_local_computed_tokens: int, num_external_computed_tokens: int) -> None:
        """
        Allocate new blocks for external (KV-connector) computed tokens.

        Must run only after every group's local blocks have been touched via
        `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
        evict another group's cache-hit blocks (issue #33775).

        Args:
            request_id: The request ID.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            num_external_computed_tokens = min(num_total_computed_tokens - num_skipped_tokens, num_external_computed_tokens)
        if num_external_computed_tokens <= 0:
            return
        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks))
        req_blocks.extend(allocated_blocks)
        if type(self.kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec, MLAAttentionSpec, HiddenStateCacheSpec, CompactFullAttentionSpec):
            self.new_block_ids.extend((b.block_id for b in allocated_blocks))

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
        Returns:
            The new allocated blocks.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            if type(self.kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec, MLAAttentionSpec, HiddenStateCacheSpec, CompactFullAttentionSpec):
                self.new_block_ids.extend((b.block_id for b in new_blocks))
            return new_blocks
