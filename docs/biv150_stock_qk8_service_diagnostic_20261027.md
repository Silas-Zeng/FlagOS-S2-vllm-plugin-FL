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

The next explicit micro entry will cover B1/8/16/32/40/64 at KV129 (full/tail/empty/ragged controls) plus B1/KV4096, including main's B40→16-split exception and S16/8/4/2. It must first check M16 code/scale bits, then actual packaged ABI/IR and independent quantized-operand references with canaries/readonly stock cache. No author SSH/JIT/GPU work, deployment, dependency installation or new GPU queue is authorized by this draft.

Keep the official service CLI and accuracy parameters unchanged: EvalScope math_500 Level3,105 questions, batch8, temperature1, top_p.95, max_tokens32768. The old normalization controller clears `VLLM_FL_BIV150_`/`BIV150_NATIVE_`; a future root diagnostic controller must explicitly set the diagnostic flags **after** that cleanup and retain main defaults/native5c and the same tuning snapshot. Do not reuse the old normalization feature environment. Model accuracy here leaves prefill attention BF16 and does not qualify final encoded-cache/all-reader K8 accuracy or service gain. Keep Draft; no deployment/merge yet.
