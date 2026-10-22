# MetaX Decode Optimizations

This branch carries the MetaX-specific top-p optimization and documents the
small-M GEMM change maintained in the companion FlagGems repository.

| Optimization | Default | Rollback |
|---|---|---|
| GPU-only top-p sampling kernel | on | `VLLM_FL_TOP_P_KERNEL=0` |
| M=32..64 Decode GEMM fast path | FlagGems dependency | revert companion FlagGems change |

## GPU-only top-p

`impl/top_p_sample.py` assigns one Triton program per logits row. It scans the
row to obtain the maximum, argmax, minimum, and sum-exp. If the maximum token
already covers `top_p`, that token is retained immediately. Otherwise, 20
threshold-search passes identify the nucleus and write masked logits. Sampling
then uses vLLM's standard exponential-noise path. The common path has no
GPU-to-CPU synchronization.

The sampler patch only selects this path for top-p-only, batch-at-least-8,
FP32 logits without per-request generators or processed logprob modes. Other
requests use vLLM's original sampler.

## Small-M GEMM

MiniCPM5-2B decode on MetaX C500 uses four exact NT GEMM shapes. The companion
FlagGems change selects the previously validated configurations directly from
the MetaX `mm` operator when `32 <= M <= 64`:

| Projection | N | K | `BLOCK_M, BLOCK_N, BLOCK_K, warps, stages` |
|---|---:|---:|---|
| QKV | 2560 | 2048 | `64, 16, 64, 4, 2` |
| Gate/Up | 12288 | 2048 | `64, 64, 128, 4, 2` |
| Down | 2048 | 6144 | `64, 64, 128, 4, 4` |
| O-Proj | 2048 | 2048 | `32, 64, 128, 4, 4` |

The change lives in
`flag_gems.runtime.backend._metax.ops.mm.mm/mm_out`; this plugin repository
does not override FlagGems at runtime. Unsupported shapes and layouts continue
through the original FlagGems implementation.
