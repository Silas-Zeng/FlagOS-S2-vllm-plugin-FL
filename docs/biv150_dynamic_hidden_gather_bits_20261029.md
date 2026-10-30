# PR122 isolated direct bit-copy entry

This child of `8d1959a443964b8d0789f9dac23bcb4945729dc1` changes only tools and evidence. Production/kernel bytes remain frozen. Default execution is a stdlib CPU plan. No device work has been performed by this child.

The entry loads the one reviewed `8a67dfc8…` CompiledKernel from the original compile-out. It reconstructs the original typed ASTSource solely for the launch ABI, using the actual installed `compiler.py` API pinned at `a97f0670…`. It does not invoke warmup or an IR compiler. The original summary, source/debug root and all six artifact bytes remain unchanged. Original support SOs seed a separate writable launcher cache; the old compile-out is read-only. A host launcher builder may still run on a cache miss. The loaded launch callable must identify its owning extension module and match the reviewed `512414b6…` SO, or the worker stops before attention/data-copy launch.

Actual packed metadata must be the three-int tuple `(4,1,0)` (bools and unknown values rejected). The block dimensions `[256,1,1]` are **derived** from this tuple and the reviewed launcher's multiplication by64. They are not a hardware-observed launch property. Loaded resources must match the reviewed 6 registers/0 spills/shared0/SME0; the actual metadata and launcher identity are exported before any direct kernel call.

There are exactly four direct calls: M65/N9 A/B and M2018/N61 A/B, width2048. B retains the exact GPU storage/pointers and replaces hidden/indices/output poison first. Index fixtures include zero, M−1, repeated and permuted rows. BF16 is treated as raw uint16, including signed zero, infinities and distinct quiet/signaling NaN payloads. The oracle is complete CPU row indexing, not a GPU result. Every output word is poisoned with oracle XOR65535, so a skipped write cannot match. Full hidden/index bytes are checked read-only and output prefix/suffix guards are checked. Complete CPU inputs, expected payload and full actual backings are gzip content-addressed; equal before/after data can share a blob. Hard failures retain arrays and stop.

Local CPU commands:

```sh
python3 -B tools/check_biv150_dynamic_hidden_gather_bits_cpu.py
python3 -B tools/run_biv150_dynamic_hidden_gather_bits.py
```

Only Root may explicitly schedule the following after reviewing/closing its previous GPU owner and checking idle resources/9031. The worker's closed-port check and CLI flag do not independently prove exclusivity. Transfer the exact independent review JSON (SHA `bdb0fb8908506630c01e45323a921c4aa1d2adcb522405d28c2ce30e0dfb6a05`) and retain the original compile-out at its real path. The old compile launcher still has its original 168-warmup/0-launch contract and must not be reused unchanged. This command is a future interface, not an authorized queued job:

```sh
python -B tools/run_biv150_dynamic_hidden_gather_bits.py \
  --run-bits --root-owned-exclusive --expected-source-commit EXACT_CHILD_HEAD \
  --compile-out /workspace/experiments/minicpm-s2/cjj-dynamic-hidden-gather-compile-v1-20261029 \
  --actual-review /ABSOLUTE/cjj-agent-dynamic-hidden-gather-actual-compile-review-final-20261029.json \
  --out /ABSOLUTE/FRESH_BITS_OUTPUT
```

This scope has no timing, Graph, side-stream, real `_cold` helper/caller-frame/CPU receipt, dispatch integration, or service test. `precision_gate`, performance and service fields stay false even if direct bit-copy passes. The new private-loader connection is CPU prepared; actual SDK behavior is still pending. Unknown actual module ownership, typed ABI, metadata, resources or binary identity rejects before direct launch and preserves diagnostics. The old compile mask parser's false and the independent SSA review remain separate historical records.
