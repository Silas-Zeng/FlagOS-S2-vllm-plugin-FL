# Dynamic hidden-state gather: CPU candidate

Started from team main `8ba3bc18de50220943ab15d7269f59e24a59d013`, frozen
as `4a303031b0961a20c5c94353f48d84d3b49ebec8`, then normally merged accepted
PR117 main `56f88737fcf3b940760b9ca3027a925cd5a7d77d`. The current inverse
proof is relative to that new main and preserves its 5898 Native identities.
Historical `*-8ba.json` reports are retained with their original commit context.
The opt-in is `VLLM_FL_BIV150_DYNAMIC_HIDDEN_GATHER=1`; it defaults to OFF.
There is no device result, service eligibility result, or measured performance
gain for this candidate yet. It does not include the unmerged 118/120/121 work.

The hypothesis is that a fixed kernel with runtime row extents can avoid the
per-input-row generated index source observed in the existing FlagGems cache.
One actual `input_rows=1782`, H2048, int32-index generated source has SHA
`41fcc814ad7519c15972dd7fb67278daa51f0ed8a9f3e324d0ce9b8d0f4f8653`.
Its address/range expressions embed 1782. This artifact alone does not prove
call frequency, compilation time, or exact service-call attribution.

Only the common `execute_model` gather before the LM head changes. The PP
broadcast, Medusa, dummy, speculative and Graph paths retain their original
expressions. The original `_prepare_inputs` return ABI and all index creation,
H2D copies, event ordering and model-forward code remain byte-identical.
`runner-transform.json` contains three reversible CRLF edits. The compute-order
runner SHA changes with the runner; the compile fingerprint explicitly records
the default-off feature and already includes both new Python sources.

## CPU provenance and semantics

The admitted service scope is the existing exact MiniCPM non-spec text scope,
BI-V150, BF16 hidden states with H2048, contiguous rows, 1–64 selected rows and
at most 2048 input rows. The original CPU scheduled lengths must be positive
int32 and sum to the scheduler total. Zero lengths, unknown tensors, invalid
metadata and other scopes take the original expression before any allocation
or launch. Negative/general advanced indices are not admitted or clamped.

The two already-existing producers are checked inside `synchronize_input_prep`:

* Ordinary: original `query_start_loc.np` equals `[0,cumsum(scheduled)]`, aliases
  the original int32 CPU tensor, and the unchanged runner copies that buffer
  before producing `query_start_loc.gpu[1:] - 1`.
* Compute order: the original `Plan` permutation is a complete int64 request
  permutation; its int32 scheduled lengths and original-request endpoints
  agree with the CPU sum. The existing pinned `logits_cpu/logits_np` aliases
  contain those endpoints minus one and the unchanged `Plan.logits_indices`
  performs the original nonblocking copy. Output order remains request order.

An execute-local, one-use receipt retains the exact GPU index tensor, backing
description, current stream, runner code and input-prep event identity. It is
consumed only by the same reviewed `execute_model` code before allocation.
Pointer/stride/offset/storage, scope, event, request-count or stream changes
reject it. No GPU index readback, `.item()`, synchronization, new stream, cross-
call index cache or replacement staging allocation is introduced. The real
`managed_inference_mode` implementation uses `functools.wraps` in
`vllm_fl/dispatch/io_common.py`; actual installed eligibility still needs a
device smoke test. Falling back without a selected launch is not success.

The kernel copies uint16 bits without BF16 arithmetic. Both row extents are
runtime int32 with `do_not_specialize`; H2048 and tile 8×128 are fixed, four
warps, one stage, no autotuning. The original indices are read as int32 and
address arithmetic widens to int64. Explicit selected-row/index-range masks
protect tails; they do not authorize arbitrary invalid GPU index contents.
The output is always a new tensor allocated on the current stream. Allocation,
JIT and launch errors propagate; no fallback/retry surrounds those operations.

## Reproducible CPU checks

```bash
python3 -B -S tools/check_biv150_dynamic_hidden_gather_cpu.py --out /tmp/gather-cpu.json
python3 -B tools/check_biv150_dynamic_hidden_gather_cpu.py --arrays --out /tmp/gather-arrays.json
```

The second command uses NumPy and explicitly fake Tensor metadata; neither
command imports Torch/Triton, compiles a kernel, or establishes device eligibility.
Checks cover exact source inverse, CRLF and both pins, default-off fingerprint,
both positive CPU producers, invalid integer/permutation boundaries, fixed-tile
complete CPU bit mapping/canaries, new-output identity, changed metadata between
preparation and consumption, external-caller rejection, allocation/launch error
propagation and one-use receipts. All 65,536 BF16 bit patterns occur in the CPU
input fixture. CPU bit copying is not a substitute for actual GPU bit comparison.

## Smallest next device screen (Root schedules)

1. Compile the private kernel with the real installed SDK. Bind final cubin,
   TTIR/TTGIR/LLVM, metadata/resources and launcher ABI. Check actual int32
   runtime extents, masked index/source/output accesses and uint16 copying.
   Change input rows 1/63/64/65/1782/2018/2048 and selected rows
   1/7/8/9/31/41/42/43/59/60/61/62/63/64 at fixed pointer alignment; inspect
   cache keys/compiled-object counts instead of assuming `do_not_specialize`
   guarantees the desired installed-backend behavior.
2. Compare full arrays bitwise to `hidden[indices]`, including both real CPU
   producer forms, shuffled/repeated valid component indices, signed zeros,
   infinities/NaN payloads, valid offset views and full input/output canaries.
   Check current/side stream ordering and lifetimes. Arbitrary negative or
   out-of-range service indices must reject prelaunch to the original path.
   Graph capture must remain on the original service expression.
3. Measure cold and warmed full-call latency, including receipt CPU work,
   allocation and kernel launch, against the original gather for the same
   inputs. A kernel-only microsecond result omits the intended cost. Stop if
   source/cache reuse is not realized or complete-call cost is unfavorable.
4. Only after actual eligibility/hits and bitwise checks, run the existing
   service precision and fixed-CLI A/B sequence. No CLI, scheduler, KV layout,
   attention math, head math or acceptance threshold changes are proposed.
