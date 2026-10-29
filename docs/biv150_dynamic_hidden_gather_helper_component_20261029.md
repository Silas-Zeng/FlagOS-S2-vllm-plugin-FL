# PR122 actual helper component preparation

Normal child of `22ba8b7cdc3be4e9619f4fa3f4e91c9cca6cfdfc`, normally merged with accepted main `899ce77470b0b539aebc108fd122f668c2d5894d`. No additional production edit: runner's original three reversible122 seams, helper/kernel and caller/seal/scope guards are retained; main5898 plus118 mixed paths/C3 remain. Default invocation imports no Torch/Triton/NumPy and only prints a plan. Actual SDK construction, helper entry and timing remain **untested**.

The frontend uses the actual installed `ModelRunnerFL` class, original `execute_model` frame, `_prepare_inputs` and `_get_cumsum_and_arange`, installed CpuGpuBuffer CPU/NumPy aliases, original index subtraction/Plan upload, and original input-prep event. It does not replace `_cold`, `_codes`, receipts/seal, helper functions or protected methods. Model output is a controlled BF16 fixture, not a model. The surrounding synthetic scheduler/config/input batch and no-op model/head plus listed outer methods/PP/context stand-ins are explicit. The exact Plan is component-built and uses its real staging/copy methods; this does **not** establish original `make_plan`/C3 or service eligibility.

M65/N9 and M2018/N61 each cover ordinary and Plan producer formulas. Their positive scheduled lengths imply unique request-end indices; the prior direct-kernel test's repeated arbitrary indices cannot qualify these real producers. Numeric checks use the same nonperiodic BF16 bit fixtures, full output/CPU row oracle and `torch.index_select` on the actual same hidden/index tensors, including NaN payloads. Hidden storage guards/readonly and CPU-proven GPU index contents are checked. The helper's new output is left as its ordinary exact allocation; output canaries are not injected into this allocation. The existing direct-kernel guarded proof is referenced separately, never called helper/service proof.

Correctness-only profiling observes the original entry and cold/launch returns without inserting caller frames. It is disabled for all timings. Original Graph-FULL ordinary dispatch must skip receipt/candidate; Plan's incompatible FULL dispatch must raise without retry. A separate actual-capture early-refusal test intentionally uses an external caller, so it only verifies the pre-caller capture predicate, not a captured runner/replay. Spec/OFF early guards are likewise separate. Graph numeric/replay and side-stream semantics remain unqualified.

An optional single warmed ABBA compares the **whole controlled original execute_model frame**, including common original preparation/H2D/event wait as well as receipt, allocation, host and launch. The OFF control's original fallback callsite uses a prebuilt control-only proxy to `torch.index_select`; this is the requested API control, not a claim to measure the current production `hidden[indices]` implementation. Index tensors are freshly produced by the same original producer on each arm/call. Full postblock output and hidden bytes are checked. No noisy prelude subtraction, profiled latency, model throughput, or gather-only claim is made. Costs of the controlled full frame can mask small gather differences and cannot be projected onto service.

```bash
python3 -B -S tools/check_biv150_dynamic_hidden_gather_helper_cpu.py
python3 -B -S tools/run_biv150_dynamic_hidden_gather_helper.py
# Root's later isolated/serial admission ONLY, after reading the frontend:
python tools/run_biv150_dynamic_hidden_gather_helper.py --run-helper --root-owned-exclusive \
  --expected-source-commit EXACT_NEW_HEAD --compile-out ORIGINAL_8D_COMPILE_OUT \
  --actual-review BDB0_REPORT --bits-review ROOT_ACCEPTED_9FE_REPORT --out FRESH_ABSOLUTE_OUT
# Add --timing only when explicitly requested; default device mode is numeric.
```

The original compile directory is read-only and copied into an independent cache. Four bounded production warmup calls (two geometries × two producer fixtures) export all actual artifacts, ABI/options/resources and loaded launcher before any helper numeric launch. Exact actual8a/e40c/5124 and typed ABI/resources must match the reviewed artifacts; unknown specialization is saved and stops before launch. This does not repeat the84-shape sweep. The new worker source itself has no GPU, SSH, queue or publication authorization.
