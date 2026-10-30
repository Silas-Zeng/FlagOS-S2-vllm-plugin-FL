# Equivalent pre-cold sizes gate (CPU only)

Parent `bfbbe8c318f474777386df617e538bbc4136d372`; main dependency899ce774.
Only the host helper changes: `prepare`'s discarded `cpu_indices(sizes,total)`
call becomes `_validate_sizes(sizes,total)`. The new function has exactly the
first three checks, same order and messages: strict Python-int total1..2048,
N1..64, strict positive Python-int lengths, exact sum. It builds no permutation,
prefix list or output indices. Default feature remains OFF.

The real `cpu_indices` function is byte-for-byte unchanged. The later actual
permutation/prefix, CPU staging/alias/value, caller/source, stream and post-model
descriptor/lifetime checks remain unchanged. There is no cross-call cache,
Plan refusal, M/N threshold, GPU file or CUDA/math/ABI change. Removing the
new function and restoring the one call recovers the whole parent helper bytes.

The original preflight performed all those checks **and** constructed a default
permutation/prefix/index tuple which was never consumed. It then constructed
the genuine permutation/prefix/values later. Only the discarded construction
is removed; no existing GPU-index producer or H2D is removed or added.

## CPU evidence

```bash
/opt/homebrew/bin/python3 -B -S tools/check_biv150_dynamic_hidden_gather_sizes_gate_cpu.py
/opt/homebrew/bin/python3 -B -S tools/check_biv150_dynamic_hidden_gather_tensor_cost_cpu.py
/opt/homebrew/bin/python3 -B -S tools/run_biv150_dynamic_hidden_gather_tensor_cost.py
```

The first checker proves exact helper inversion, identical AST of the strict
checks, all remaining function bytes, 543 bounded acceptance/rejection and
exception-order comparisons (including bool/int subclasses/float/zero/negative/
N65/total2049), and exact full-prefix/permutation outputs. Twelve old/new
pre-cold control-flow cases use a sentinel that **always throws and never
returns a cold state**: valid sizes reach it once, invalid sizes never reach
it. This does not claim SDK, real cold, caller or device qualification.
All28 inherited true-Tensor cost checks also pass. Evidence JSON is under
`docs/evidence/dynamic-hidden-gather-sizes-gate-20261030/`.

## Reuse the same actual four-case frontend

`run_biv150_dynamic_hidden_gather_tensor_cost.py` changes only its parent pin and
honest default-plan host-diff description. Its exact private worker derivation,
four ordinary/Plan fixtures, true Tensor OFF/ON, actual FlagGems dispatch/source
export, original numeric guards, cache/compile barriers, ABBA order, eight
warmups and32 measured calls per block remain unchanged. The inherited CPU
checker now expects this one explicit helper-host delta, rather than reporting
production zero-diff. The original helper worker is unmodified.

Future Root-only CLI is the same `--run-helper-cost --root-owned-exclusive
--expected-source-commit NEW_HEAD ... --timing` with the existing readonly
8d/v1 directories and bdb/9fe/39ef proofs. Candidate compiled afc/805a/5124
must still match exactly; host receipt/bitwise/readonly/dispatch and timing
must run anew. The old bfb launcher source/tree/helper pins must be updated
for this explicit host delta; its old byte pins correctly reject this child.

The parent actual component had three positive points and a large Plan negative
(OFF/ON .9670); all results stay preserved. This change makes no new cost claim.
Retest the unified gate first. If large Plan remains slower, ordinary-only is a
separate semantic-routing decision, not part of this commit. Plan real service
activation/C3, Graph numeric, model quality and service throughput remain pending
for this new helper. Existing118's actual Plan selection remains proven and is
not disabled by this change.
