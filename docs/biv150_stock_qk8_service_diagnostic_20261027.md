# Default-off stock-cache QK8 decode accuracy bridge

Depends on [PR79](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/79), frozen component5abf and immutable device evidencebf94. New independent branch starts from fresh-fetched/API-verified team main56f; no official SDK upgrade. This is for **pure-decode quantization105-question diagnosis**, not final stored-K8/full-prefill quantization acceptance or a throughput candidate. Prefill, cache contents, all writers, allocator/layout, original GQA eligibility/segment policy/reducer and generic fallback remain stock BF16.

PR79's root-serialized B64/4K stock experiment passed M16 codec bits, quantized-operand FP64 implementation gates, readonly cache/source/canaries, Graph/side and actual I8/PV/C-chain IR. Stock and packed output bits match; original BF16 proxy remains false and no timing was measured. This is a reason to test decode model accuracy, not claim it passed. The packaged kernel changes only the codec import path; source function/decorator bytes are identical to the tested generated producer. Only the exact codec helper function is extracted; no K8 writer is bundled. Source hashes and byte provenance are in `docs/evidence/stock-qk8-service-diagnostic-20261027/source-provenance.json`.

`VLLM_FL_BIV150_STOCK_QK8_DIAGNOSTIC` defaults to0. Off: the worker does not import the diagnostic module, no hook object is created, no target is queried, and no live binding changes. The appended startup block can be deleted to recover main's worker bytes exactly. The only runner runtime pin is `biv150_compute_order.EXPECTED_RUNNER`; it is updated to the new worker SHA so main compute-order remains enabled. Complete rg found q32/multi2 tools read that shared pin. The existing CPU identity proof first reverses only this new block and cache-feature line, then performs all original main inverse checks.

Explicit on requires a real packaged-kernel micro certificate via `VLLM_FL_BIV150_STOCK_QK8_MICRO_CERTIFICATE`. Missing/incorrect source, target, ABI or certificate raises before the producer binding changes. Main's existing GQA wrapper must already be bound on both backend/top attention entry points, with the audited original generic fallback and optional normalized wrapper. Only `gqa._gqa_decode_attention` is replaced, after all checks. It does not wrap attention forward or change any cache writer. Compile/candidate failures propagate; they never silently run BF16 and report a Q8 accuracy result.

Installation reports `BIV150_STOCK_QK8_INSTALL`. Actual selected launches print `BIV150_STOCK_QK8_SELECTED` at powers of two with actual compiled hash and host-submission count; separate main-GQA eligible and generic-fallback deltas are exposed by `stats()`/final log. These are host/capture submissions, **not Graph replay or token counts**. An accuracy result is only diagnostic evidence if the worker reports installed and selected>0; fallback counts must be interpreted separately.

The new flag is explicitly included, default false, in the compile-semantics fingerprint. No existing feature defaults or native library selection change. Off equals explicit0, on has a distinct cache key, preventing decode precision arms from sharing compiled semantics. Certificate filesystem paths are not collected into the key; actual source bytes are hashed already. Native5c/library and tuning snapshots must remain the same between diagnostic/control arms.

CPU commands:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_stock_service_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_compute_order_identity_cpu.py
```

CPU tests prove source inverses, no-action off path, refusal without a micro certificate, schema negatives, unchanged native defaults and separated cache keys. A **synthetic schema fixture exists only in CPU unit memory** and is never written as device evidence. It cannot be used to claim the packaged kernel passed. Actual packaged compilation, numeric warm validation, live hook and model105 are still pending.

The prepared explicit micro entry covers B1/8/16/32/40/64 at KV129 (full/tail/empty/ragged controls) plus B1/KV4096, including main's B40→16-split exception and S16/8/4/2. It first checks M16 code/scale bits, then actual packaged ABI/IR and independent quantized-operand references with canaries/readonly stock cache. It has not been executed on device. No author SSH/JIT/GPU work, deployment, dependency installation or new GPU queue occurs in this draft.

Keep the official service CLI and accuracy parameters unchanged: EvalScope math_500 Level3,105 questions, batch8, temperature1, top_p.95, max_tokens32768. The old normalization controller clears `VLLM_FL_BIV150_`/`BIV150_NATIVE_`; a future root diagnostic controller must explicitly set the diagnostic flags **after** that cleanup and retain main defaults/native5c and the same tuning snapshot. Do not reuse the old normalization feature environment. Model accuracy here leaves prefill attention BF16 and does not qualify final encoded-cache/all-reader K8 accuracy or service gain. Keep Draft; no deployment/merge yet.

## Packaged micro entry

`tools/bench_biv150_stock_service_micro.py` defaults to CPU plan JSON, without Torch/Triton/NumPy imports. Device flags require a frozen clean HEAD, fresh output, actual JIT hash, target corex71/warp64 and idle-device/service guards. `--compile-only` allocates empty buffers and uses `.warmup` for the packaged M16 probe, actual main producer, packaged candidate and original reducer; it never launches those kernels and never emits a usable certificate. Original pre-launch resource unknowns stay unknown.

`--run-gpu` first requires exact M16 codes/scales for141 finite vectors. It then runs each listed shape on unmodified BF16 cache with four controls: original/full, changed Q/table/ragged tail, zero Q, and empty first row. Original input factory/seed, independent CPU INT64/quantized-operand FP64 and original-BF16 references, old precision limits, source/cache hashes, canaries, inactive poison, actual I8/PV/native C-chain IR and original one-warp/reducer4/split policy remain. The TTIR public16-parameter signature and LLVM18-parameter entry (including two backend pointers) must match the actual baseline ABI. The parser is CPU-replayed against exact PR79 target header lines and rejects six changed-type/name/order mutants; it does not infer native MMA component independence.

Changed-Q/table/length poisoned Graph and side stream must match eager. There is no append writer/cache mutation test and no timing. A certificate is written only after all implementation/memory/IR/ABI/Graph hard gates pass for all seven cases; source hashes bind it to the actual packaged kernel, codec and hook. The old BF16 proxy remains separately reported and can remain false, in which case result `passed=false` even though `implementation_hard_gates_passed=true` and an explicitly diagnostic micro certificate is issued. This does not relax the proxy, claim model105 or qualify full-prefill quantization. An aborted/partial run emits no usable certificate.

Six independent reference modules are copied byte-for-byte from frozen PR79 source5abf; selected host IO/artifact functions are exact extractions, recorded in `micro-dependency-provenance.json`. They are CPU/reference code, not new production kernels. No PR79 runtime is edited. Default CPU replay:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_stock_service_micro_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/bench_biv150_stock_service_micro.py
```

Future root-serialized commands only (replace FROZEN_HEAD with the delivered exact commit and use fresh directories):

```sh
python3 tools/bench_biv150_stock_service_micro.py --compile-only --commit FROZEN_HEAD --out FRESH_COMPILE_OUT
python3 tools/bench_biv150_stock_service_micro.py --run-gpu --commit FROZEN_HEAD --out FRESH_NUMERIC_OUT
```

Only after root reviews the real `micro-certificate.json` may a future diagnostic controller set both the default-off flag and its certificate path after environment cleanup. The service kernel is not changed by the micro, and no service was launched by this author.
