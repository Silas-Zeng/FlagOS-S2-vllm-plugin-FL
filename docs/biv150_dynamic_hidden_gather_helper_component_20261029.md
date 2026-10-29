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

## Reviewed production cache after preserved v1 refusal

The04fa actual run179729/start362922962 stopped after one production warmup, with **zero candidate launches**, on unknown specialization. It was not a numeric failure. Natural production metadata wasSME5/hash `afcbb980637dd36ccf51c124652c41efc8de8f99f2bff21f7904beb586a51b58`/cubin `805a6fa835b60270ddbcd0c059afad9d44d553cfe80b1600e17c3cdbb5da0cce`. Full typed ABI remained identical to8a/e40c; all loadable ELF sections, symbols and `.text` machine bytes were identical. Only nonloadable DWARF source-root text/unit length changed. Exact path normalization restores the complete four IR forms; ASM instructions are unchanged. Static comparison39efd376 was independently replayed byRoot. This is a new explicit805a static binding; helper cold/receipt/numeric still have no device result.

SDK's first-miss `get_corex_sme` naturally permits BF16 contiguous hidden/output operand bits0+2=5. It does not add any SME instructions in this bitcast-to-u16 kernel. The old0 run did not save `torch.corex`/`TRITON_DISABLE_SME`; its branch cause cannot be asserted. The new private driver records those values before/after runner import and predicts the natural5 from actual binder metadata. It never forces either0 or5, and does not change production options.

The new private `biv150_dynamic_hidden_gather_reviewed_cache.py` follows the already-used116 cache loader: original JIT/binder constructs ASTSource and actual ABI, all7 original afc cache children are verified and copied byte-exact, and only the index paths relocate. Original helperv1 summary29ab and static report39ef are required. The actual compiler/cache implementation, standard cache manager/knobs and absolute cache root are checked. Backend `add_stages` is prohibited during the initial cache load. Afterwards only this JITFunction's compiler entry is guarded so a candidate memory-cache miss stops rather than compiling; it restores the compiler on exit. Other framework/control kernels may compile, so this is not a claim of globally zeroJIT. Actual options/typed ABI/resources/805a and actual5124 handler/packed metadata remain strict prelaunch gates. Cached DWARF continues to name the original04fa/v1 source root, separately recorded from the new validator checkout. Thev1 summary has no `source_root` field; provenance is pinned from its actual TTIR path instead of inventing it.

Add the following readonly inputs to the previously documentedRoot-only numeric command:

```bash
--production-out ORIGINAL_HELPER_V1_OUT --production-review ROOT_REPLAYED_39EF_JSON
```

Default remains CPU; no helper/production bytes, receipt guards or numeric gates changed. Do not pass `--timing` for the first revised numeric run. The actual new cache-hit path and SDK constructor are pending. Reproduce cache byte/refusal CPU checks with:

```bash
python3 -B -S tools/check_biv150_dynamic_hidden_gather_reviewed_cache_cpu.py
python3 -B -S tools/check_biv150_dynamic_hidden_gather_helper_cpu.py
```
