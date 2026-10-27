# Pure-decode Math105 diagnostic controller

`tools/run_biv150_stock_decode_105.py` is a default-CPU plan with one explicit `--run-service` mode. It has no throughput path. This tools/docs-only update leaves **every production byte and every micro/hook/kernel/codec byte from867b725 unchanged**, so the real micro certificate remains applicable. Root's completed867 micro supplies the actual certificate; no CPU fixture is a substitute. Certificate metadata SHA checked locally is `ee20e9c7879b18ee4a6e28f85878cd71677aa9b75b05566060d425580e466d8b`.

Serve argv is fixed: port9031, `FULL_DECODE_ONLY`, utilization0.85, max-model-len131072, original model and served name. The exact original `accuracy()` function is SHA-pinned (`27937c1566e4b8680fc4feb2e18d6f12e0a4c157416ac9840fab6cc19b221bd2`) and executed with an owned-process adapter. EvalScope argv, dataset/subset, batch8, temperature1, top_p.95, max_tokens32768, request timeout3600, complete105/unique-question/error/score≥.95 checks and the original client timeout5400 are unchanged. `--ignore-errors` is preserved but errors still invalidate the diagnostic. No extra service or evaluation parameter is accepted through this controller.

Before any service spawn it checks exact clean source HEAD and zero production diff from867; original protocol; actual micro certificate against unchanged source/target/schema; native SO SHA5cbb; and the same accepted initial SQLite snapshot from main66. Both databases are checked by bytes/SHA, standalone-backup status and readonly `PRAGMA quick_check`, then copied into a fresh **private** `flaggems-cache/config_cache/`. Source databases are not modified. Accepted database hashes are3_2 `5764f125...` (1,605,632B),3_6 `2c5aa448...` (15,712,256B); full values are in the script. Triton/Inductor caches are separate private paths. Port9031 or GPU busy causes refusal without stopping any process.

All inherited `VLLM_FL_`/`BIV150_` switches and stale tuning/backend overrides are removed; only then the diagnostic flag, private actual certificate and native5c path are explicitly set. **M2/Q32/multi2/GQA/native enable flags remain unset**, retaining shipping defaults. It does not use the old normalization baseline that disabled M2/multi2. The unchanged native5c and initial snapshot must match the control environment. No dependency installation, SDK upgrade or source mutation is performed.

The manifest records original caller package resolution/PYTHONPATH, installed package resolution without PYTHONPATH, requested source resolution, actual owned-worker initial PYTHONPATH, complete production source hashes, packages/ABI, native/snapshot/certificate/controller identity and fixed argv. Package discovery uses `find_spec` without executing the package. An isolated CPU-only Torch metadata child reads ABI; it does not create a model or run GPU work. Actual worker compile-semantic hashes must equal expected diagnostic-on and differ from off. Main native/M2/Q32/multi2/prefill/GQA and diagnostic installers must report installed, with no C3/multi2 hard fallback.

Kernel selection logs preserve actual compiled hashes, selected host-submission lower bounds and original generic-fallback lower bounds; final host counts are saved when workers emit them. Overall installed+selected>0 is required. Evaluation-window events are recorded separately: `FULL_DECODE_ONLY` Graph replays do not execute the Python counter, so the controller does not fabricate per-token or replay counts or require a new host event for every question. Actual on-bit/source identity, captured candidate selections, complete105 requests and post-evaluation state provide the scoped evidence. No overall selection means invalid diagnosis.

Both service and EvalScope run in newly owned sessions, but cleanup never sends group signals. It records PID/start identity, remembers descendants while a matching owner is live, checks identity again before each signal, uses pidfd when available, and never signals a reused/unowned PID. Remaining/unknown same-session births are reported and prevent a valid result. On failure all raw reports, rows and errors are kept. `prediction-inventory.json` explicitly records105 indices/question IDs, failed choices/errors and parse errors. Cleanup must close before validity is reported.

Even a successful run keeps `precision_gate=false`, `full_prefill_precision_passed=false` and `encoded_cache_precision_passed=false`. `pure_decode_math105_gate_passed`/`valid_pure_decode_diagnostic` apply only to this BF16-prefill/quantized-decode path. A failed/incomplete/error/no-selection run has no valid qualified conclusion and no automatic merge or close. This is not final stored-K8 accuracy or a service speed claim.

CPU replay and plan (no Torch/Triton/NumPy/vLLM/EvalScope import, no service or GPU):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_stock_decode_105_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/run_biv150_stock_decode_105.py
```

The checker uses explicit synthetic SQLite/log/105-row/process fixtures only, including wrong fingerprint, no selection, disabled main installer, failed/duplicate/incomplete predictions and PID reuse. None are device/model evidence. Source production inverse and the original protocol are separately verified.

Future root-owned execution only, after reviewing the real micro and input paths:

```sh
/usr/local/bin/python "$SOURCE/tools/run_biv150_stock_decode_105.py" --run-service \
  --source "$SOURCE" --commit "$FROZEN_CONTROLLER_HEAD" --out "$FRESH_105_OUT" \
  --gate-library /workspace/experiments/minicpm-s2/code/baseline-profiler-v2-20261025/build-native-w8a8/biv150_native_w8a8.so \
  --tuning-snapshot /workspace/experiments/minicpm-s2/exports/small-bf16-fullwarm-tuning-snapshot-v6-20261025 \
  --micro-certificate /workspace/experiments/minicpm-s2/cjj-stock-qk8-service-micro-v1-20261027/micro-certificate.json
```

Use an independent frozen source checkout and fresh output. The historical absolute native/snapshot paths above are revalidated before use, never assumed unchanged. This author has run only CPU checks; no service, Math105, SSH or GPU work was started.
