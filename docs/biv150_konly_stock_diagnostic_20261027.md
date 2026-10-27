# Stock-BF16-cache decode quantization diagnostic bridge

This CPU-only private source factory is an early bridge for testing **decode quantization accuracy**, not another throughput candidate. It leaves the stock BF16 KV allocation, writer and cache contents intact. There is no service hook, dispatcher mutation, fallback change or device runner in this update. The existing ae00/f287 kernels/support/driver/gates and production files remain byte-for-byte unchanged. Team main was freshly checked by GitHub API as `56f1838192abc940b3fb9accef490022ac125000`; the simultaneous `git fetch origin main` failed to connect, so it is not reported as a successful synchronization. CLI identity is CJJ1008.

`tools/biv150_konly_stock_diagnostic_factory.py` accepts only audited GQA source SHA `2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1`. It extracts the original producer and makes a narrow reversible transform:

1. Keep every raw parameter/annotation, original grid assumptions, while/page16 loop, I64 addressing, full/tail Q/K/V load, mask and store unchanged.
2. After the original BF16 Q hoist load, call the exact frozen ae00 `quantize_bf16_rne` helper once per active split CTA.
3. After each original BF16 K load, transpose `[128,16]` to `[16,128]`, call the same helper per token/head row, then transpose the I8 codes back. Invalid tail columns were loaded as all-zero rows by the original mask and therefore obtain zero codes/scale1; the original `-inf` score mask remains.
4. Use the same explicit I32 `tl.dot(..., out_dtype=tl.int32)` and FP32 order `I32→F32`, multiply sQ, multiply sK, multiply original softmax scale. Keep original new-max, exp2, BF16 P, BF16 V/PV, FP32 states and reducer untouched.

The inverse removes only these named transforms and must recover the original kernel bytes SHA `b505e712ea21e10563ffb4a14f01a13a9c77fa32308095a80ebbc417b96e69ce`. CPU also verifies every `tl.load` and `tl.store` AST call is identical to the original. No key-code/scale pointer is added, and no cache store is introduced. Same raw parameter schema does **not** prove a compiled typed ABI.

The helper is codec `cjj-konly-rn32-rne-v2`, support SHA `409ca094e324dea321a9dec8286c19d066ad303113ad879af5a94eaba038a018`: fixed RN32 scale, RN32 element quotient, integer RNE/clamp. Source and mathematical protocol are identical to stored-K8, but the new `[16,128]` helper invocation, transposes, native lowering, actual IR, loaded resources, FP32/FTZ behavior and complete output remain unverified on device. Prior M1/M8 codec and stored-K8 decoder evidence cannot prove this new producer. Finite input codec semantics do not guarantee finite attention scores for every BF16 extreme.

`tools/biv150_konly_stock_diagnostic_reference.py` defines independent Python-integer dot products, ordered FP32 targets and reconstructed-operand FP64 targets for a loaded stock tile. It is not a simulator of GPU softmax/PV/reduction or FTZ. Its masked tail ignores poisoned invalid rows. The CPU checker compares 204 geometry/pattern/tail cases and 13,056 valid score cells with the independently packed v2 code/scale reference, including all valid lengths0..16, both KV heads, zero rows, half boundaries and tiny BF16 operands. It checks seven closed original/quantized FP64 attention geometries and rejects 15 source/format/ABI/mask/PV mutations. Those are CPU targets, not passed device numerical or model gates.

The future caller must positively establish **stock-BF16 provenance**. A BF16 wrapper can also alias encoded K8 bytes; `dtype=bfloat16` alone must never select this bridge or an original BF16 fallback for such storage. The CPU descriptor validator rejects the encoded format and unknown format. It does not inspect a real tensor or create a trusted runtime format registry.

This bridge only quantizes decode queries and the BF16 keys read by decode. Prefill attention still uses stock BF16 when this bridge is eventually hooked; prefill-written keys are quantized on later decode reads. A model result would therefore assess this mixed diagnostic path, not the final all-reader compressed-cache/full-prefill quantization scheme. Any claimed service accuracy or speed requires a separate, reviewed opt-in hook after direct device falsifiers. Repeated on-load K quantization may be slow; no throughput benefit is proposed or queued.

Reproduce without Torch/Triton/NumPy/JIT/GPU/SSH:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_stock_diagnostic_cpu.py --out /tmp/stock-diagnostic-cpu.json
```

Optionally emit private Python source text for inspection, without importing or compiling it:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_stock_diagnostic_cpu.py --out /tmp/stock-diagnostic-cpu.json --emit-source /tmp/stock-diagnostic.py
```

Default source SHA is `fb28165b3426bd81fe2e4a27952da9aa98625e0ce383200878b1930df0711e7d`. CPU report is `docs/evidence/konly-i8-qk-20261027/stock-diagnostic-source-cpu-v1.json`. Device compilation/numeric, micro cost, service hook, model105, full-prefill quantization and service performance are all pending. PR79 remains Draft and no existing precision gate is relaxed.
