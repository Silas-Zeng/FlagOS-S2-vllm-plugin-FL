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
already covers `top_p`, that token is written directly and the row does not
enter the sampling kernel. Otherwise, 20 threshold-search passes identify the
nucleus and write masked logits. A second Triton kernel branches per row:
direct rows emit the recorded token, while broader rows use vLLM's
exponential-noise race without a full softmax or a separate div/argmax. The
kernels use a 32768-element tile and eight warps on MetaX C500. The common path
has no GPU-to-CPU synchronization.

The sampler patch only selects this path for top-p-only, batch-at-least-8,
FP32 logits without per-request generators or processed logprob modes. Other
requests use vLLM's original sampler.

## Small-M GEMM

MiniCPM5-2B decode on MetaX C500 uses four exact NT GEMM shapes. The companion
FlagGems change selects the previously validated configurations directly from
the MetaX `mm` operator when `32 <= M <= 64`:

| Projection | N | K | `BLOCK_M, BLOCK_N, BLOCK_K, warps, stages` |
|---|---:|---:|---|
| QKV | 2560 | 2048 | `64, 32, 64, 4, 4` |
| Gate/Up | 12288 | 2048 | `64, 64, 64, 4, 2` |
| Down | 2048 | 6144 | `32, 64, 64, 4, 4` |
| O-Proj | 2048 | 2048 | `32, 64, 64, 4, 4` |

The change lives in
`flag_gems.runtime.backend._metax.ops.mm.mm/mm_out`; this plugin repository
does not override FlagGems at runtime. Unsupported shapes and layouts continue
through the original FlagGems implementation.

## Local C500 validation

- Top-p microbenchmark, batch 64, vocab 130560: about `4.85 ms` before the
  direct-row path and about `0.96 ms` after both the direct-row and wider-tile
  changes.
- Stable 64-request decode with a 128-token prompt: median ITL
  `15.6 ms -> 11.2 ms`; the wider GEMM configurations account for about
  `0.5 ms` of that reduction.
- Official single warmup plus single measured run, 4096 input / 1024 output /
  64 concurrency / 256 requests: `8473.34 total tok/s`,
  `1694.67 output tok/s`, mean TPOT `34.40 ms`, 256/256 successful.
- Math500 Level 3, 105 questions: `98.1%` accuracy (`103/105`), above the
  required 95% threshold. The first pass completed 104 predictions before an
  EngineCore cgroup OOM; the final cached resume completed the missing sample.
- The GEMM table above is a tuned Triton path, not a switch to the vendor
  `aten::mm`. Native redispatch through `addmm(beta=0)` was rejected because it
  degraded CUDA Graph capture and microbenchmark latency.

## Next bottleneck

A current 4K decode trace shows that the top-p path is no longer dominant:

| GPU work | Approximate share |
|---|---:|
| `mm_kernel_nt` | 40% |
| FlashAttention decode/combine kernels | 23% |
| fused RMSNorm/SiLU kernels | 9% |
| top-p mask and sample kernels | <1% |

The strongest measured follow-up is shape-aware routing of unquantized BF16
GEMMs to vendor mcBLAS. With `VLLM_FL_FLAGOS_BLACKLIST=mm,mm_out`, the official
4K/1K measured run reached `10014.16 total tok/s`, versus `8473.34` with the
current Triton path. A short-context control also improved median ITL from
`11.2 ms` to `9.1 ms`. This is currently an experiment, not a retained code
change: a direct in-kernel redispatch to `aten::mm` recurses, while
`addmm(beta=0)` is slower and disrupts CUDA Graph capture. A production version
needs a shape-aware vendor entry point rather than a global mm blacklist.
