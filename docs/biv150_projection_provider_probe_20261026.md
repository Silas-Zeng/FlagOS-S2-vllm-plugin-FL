# Projection providers: measured full working set and next crossover probe

Dependency: PR58, includes actual team main34942f7 (fresh fetch2026-10-26).
No production installation, row policy or default change. The helper refuses
SDK rows1–8, retaining the existing independent tiny-row BF16 protection.

PR58 exact-source candidate/control scores are4K5760.266264/5624.274341
(+2.41795%) and16K4049.275300/4110.860080 (-1.49810%). Both precision gates
pass102/105 and103/105, with complete requests/tokens/working TTFT and pre/post
warm semantic checks. Hold the Draft; preserve4K while fixing16K.

Frozen component6e0505789a8812af0788e3cc4129f4c9bc224652 completed successfully.
It uses all42real O/QKV/down weights and8uncompressed edge-gate weights.
Representative first/middle/last layers, rows32/64/2048 and awkward rows
9/16/24/31/33/40/48 pass2880 CPU FP64/default+side/changed-poisoned Graph checks.
Every timed chain also verifies its last real output against independent FP64.
Three alternating provider orders replay warmed full weight working sets.

|Projection|Real weights|32 rows provided-linear speedup vs MM|64 rows|2048 rows|
|---|---:|---:|---:|---:|
|O|42|0.727|0.717|1.249|
|QKV|42|0.751|0.833|1.069|
|Down|42|0.491|0.661|1.286|
|Edge gate|8|0.964|0.911|1.213|

Ratios are MM chain time / provided-linear chain time: values above1favor
linear; below1favor MM. Functional and supplied-output linear are close.
These measured endpoints justify probing a row-dependent policy. They do not
establish a model gain, a crossover threshold, or the cause of16K regression.
Inputs are synthetic; attention/native middle-gate/consumers/host overhead
are omitted. Do not sum component time into an actual Transformer-step claim.

Raw archive24995bytes SHA256
b14582f637cf6d29b07a564d7f804b69f767d18c3207ad332bf2b3a727af38aa
and parsed summary/launch identity are in docs/evidence/projection-provider-20261026/.
Autotuning rejected an oversized shared-memory candidate and selected a valid
configuration; overall result.complete/passed=True and errors=[] are retained.

The pinned helper uses the unchanged SDK14argument bias-free ABI, GROUP_M8,
SPLIT_K1 and supplied output. Exact Iluvatar linear/LibEntry/MM source guards
remain active; no SDK symbol is replaced or new arithmetic introduced.

The next component-only probe will measure64(anchor),128/256/512/1024 and real
prefill tail rows1739/1798/1985/2017. Explicit --timed-row arguments are recorded
in the artifact. --skip-awkward-checks avoids duplicating already qualified
awkward cases; every newly timed row still receives all8numerical/stream/Graph
checks on each representative weight and provider before timing. Defaults
retain the original complete probe. No production threshold is selected yet.

The first job has exited and GPU release was verified. Subsequent GPU work is
serial on the personal instance, never the teammate instance. Any chosen
policy still needs actual service selection, full105, sufficient warmup, both
fixed benchmarks/TTFT and post-warm correctness before a production merge.
