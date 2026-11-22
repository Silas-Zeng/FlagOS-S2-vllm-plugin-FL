# Private native K8 QK / original BF16 PV: first compile-only prototype

This extends the PR93 CPU layout contract on the same fresh-confirmed main b3b87d2. All production files remain byte-identical. The generated CUDA source and its exact inverse are reviewable under tools; no service installation or activation is added. Neither this author nor the CPU checks invoked a native compiler, Torch, Triton, SSH or GPU.

`biv150_native_k8_source_factory.py` SHA-checks b3's complete original CUDA, applies counted literal transformations and reverses them to the exact original bytes. Generated `tools/kernels/biv150_native_k8_qk.cu` retains the original BF16 PV, V SME loads/layout/+4096B region, FP32 online max/sum/FMA, probability BF16 conversion/shuffles, partial output mapping, final barrier and S2/S4 page-loop geometry. The reducer remains external and unmodified. New symbols isolate the prototype; OverlapV is fixed false. Inverse checks reject mutations to PV FMA/rounding/output, Q-scale ownership, signed-score interpretation and the internal scale offset.

Changes are explicit BF16 Q RN32/RNE quantization, pair-shuffle I8 packing, signed-I8 K ABI, ordinary aligned K global-load/shared scatter, four K32 native I8 QK MMA calls, and signed-I32→F32 Q/K scales before the unchanged softmax. K reads are masked before dereference for tail slots. Q reduces each original input head over 128 dimensions, then obtains scales for C rows g/g+4 from their actual owners. K/scale/V retain the full16KiB physical cache format; compact12544 is rejected. Stage remains8KiB, including the unused gap between K2KiB and V+4096B. New K barrier, Q quantization/repacking, scale loads and loss of SME-K efficiency are real costs, not performance claims.

The shared codec header explicitly requests `__fdiv_rn(peak,127)` and `__fdiv_rn(value,scale)` plus `__float2int_rn`, with zero-vector scale1, FLT_MIN scale floor and [-127,127] clipping. Ordinary `/`, reciprocal and half-away rounding are forbidden by the CPU source contract. Current locally collected headers do not yet establish these intrinsic declarations/target FTZ semantics. Compatibility and actual optimized lowering are **unknown** until Root's target compiler/LLVM inspection; no expression-level claim of RN32 conformance is made. Failure must be preserved, with no codec fallback. The separate codec diagnostic kernel is registered for later exact conformance tests but is never invoked by compile-only.

The private Torch Library host ABI accepts one-query-per-sequence BF16 batch64/S2 orbatch32/S4 with original strides, signed-I8 K [P,16,2,128] stride16384/256/128/1, F32 scale vector[P*4096] at the exact same full layer-page0 address, and BF16 V with stride8192/256/128/1 already offset8192B. It verifies complete backing extent/alignment, same cache backing, no readonly/cache alias and separate nonoverlapping output backings. Nonzero layer storage offset is permitted. Already-offset scale pointers and compact views fail. Calls require component-owned CPU-preupload finite Q/K/scales and valid length/table content, independently compared in the future device fixture; tensor dtype/shape does not certify those contents. There is no live metadata or service fallback admission here. Host and device exceptions propagate.

Append remains the actual PR83 `encode_append`, with frozen source identities in `writer83-identity.json`. Its arguments preserve independent K/V source row pitches and extents, slot_count and cache_pages; all K-code/scale/V pointers are FULL page0 bases (different from this consumer's already-offset V pointer). Negative/invalid slots mask before source loads; duplicate/unknown/alias mappings require refusal. Initial encode, incremental append, attention producer and reducer must be measured separately and then together. This compile-only frontend executes none of them and does not substitute a new writer or reencode a live cache.

CPU checks/default plan:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_k8_compile_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/compile_biv150_native_k8_qk.py
```

Root-only future target command, new empty output directory, using the confirmed environment/compiler:

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/compile_biv150_native_k8_qk.py \
  --compile-only --compiler /usr/local/corex/bin/clang++ \
  --out /workspace/experiments/minicpm-s2/NEW_NATIVE_K8_COMPILE_OUT
```

The explicit mode preserves source copies, compiler version/SHA, exact standalone device LLVM/ASM commands and stdout/stderr, plus the same Torch cpp-extension load mechanism/ivcore11 flags as b3. The standalone device artifacts are labelled separately from the extension build objects; there is no claim that their compiler-added flags or debug identity are byte-identical. Build outputs/library and every artifact receive a SHA manifest. Load registers `torch.ops.cjj_native_k8_qk_probe`; no Python `_C` module is fabricated. A successful compile only establishes compilation, not RN32 lowering, actual I8 lane execution, C++ ABI invocation, resources, correctness or speed. Errors are saved in a failed summary with exit1.

Compile-only counters are strictly 0 reader/encoder/codec/reducer launches and0 input initializations. Default mode is stdlib-only and never opens a compiler/framework. Next gate is actual codec boundary conformance and I8 QK lane markers, then eager partial/max/sum/full output versus quantized FP64 and original BF16 separately, complete buffer/readonly canaries, changed-input Graph/side and actual resources/ISA. Only then can the full-cost native producer+append+reducer be compared with current b3 on a matched pair. No model105, full-service or throughput acceptance is claimed; precision_gate remains false.
