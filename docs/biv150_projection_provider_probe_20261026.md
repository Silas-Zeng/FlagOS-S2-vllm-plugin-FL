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

## Completed crossover,2026-10-26 09:01 Shanghai

Frozen f3087151e40ef3c859d89b0cfd740a30e1db490a completed4groups,324cases,
2592FP64/default+side/changed-poisoned Graph checks,36full-working-set timings;
complete=True,passed=True,errors=[]. Child487639/start332341967 was verifiedZ
and GPU idle. No subsequent GPU job is queued.

|Rows|O linear speedup vs MM|QKV|Down|Edge gate|
|---:|---:|---:|---:|---:|
|64|0.720|0.832|0.659|0.913|
|128|1.168|1.419|1.029|1.256|
|256|1.626|1.428|1.713|1.215|
|512|1.421|1.085|1.793|1.380|
|1024|1.746|1.275|2.009|1.206|
|1739|1.178|1.076|1.240|1.193|
|1798|1.250|1.081|1.294|1.186|
|1985|1.252|1.139|1.283|1.212|
|2017|1.249|1.067|1.286|1.211|

All sampled128+ endpoints favor linear;64anchor still favors MM and agrees
with the first job. A general row policy retaining tiny1–8 protection and MM
for small rows is now supported for integration exploration. Threshold and
untested boundaries still need actual compiler/output alias validation; do
not present interpolated row performance or model gains as measured. No
production policy/default has yet been selected, implemented or merged.

Archive32707bytes SHA256
749b38d23eb5ad966d1af790328cf675d28f6b5eb7e6b2fee2837819e617767e
and parsed summary/launch are retained beside the first full raw artifact.

## Opt-in integration under validation

VLLM_FL_BIV150_LARGE_LINEAR=1 is effective only with existing small-row
protection and compiler-output lowering enabled. The scoped callback retains
M1–8 own BF16, M9–127 MM, M128–2048 supplied-output pinned linear, and original
MM above2048. Unsupported descriptors/aliases/autograd retain MM. This is a
general operator geometry/row policy; no request or benchmark identity is read.
Default remains0. All promoted helper function ASTs exactly equal the measured
component helper;44ABI CPU and228routing/fallback/cache cases pass. Active
policy joins the compilation identity; unused/off flags canonicalize equally,
logging stays identity-neutral, and live installed-policy changes are rejected.

Expanded GPU compile oracle covers127/128/129,2047/2048/2049 and intermediate
bounds/real tails, preserving independent FP64, two streams and changed-input
poisoned Graph. It verifies actual callback provider counts, not installer
status alone. Service harness has a separate post-ready1739-token one-output
request and verifies all four actual projection selections from only its new
log segment. Both arms will use the same diagnostic; official105/CLI/fixed
benchmark counts and three short+one full warmup/single score are unchanged.
The owned cleanup finally block still matches main exactly. No GPU compile or
service acceptance for this new integration has completed yet; do not merge.

## Integrated compiler validation completed; service acceptance dispatched

Frozen a48cc8b68cbb68e421ead6e2d9e6fda0156c2354 passes108real-weight/row
cases and1728independent FP64/stream/changed-poisoned Graph checks, including
127/128/129,2047/2048/2049, intermediate bounds and real prefill tails. All
expected runtime providers were positively selected;8actual generated out
modules were retained. GPU child489843/start332486039 endedZ with
complete=True/passed=True/errors=[] and GPU release verified. Raw23907bytes
SHA3e8ccecc8f82c29d8e2a95f31772b0298aa69656d571db41640f6bb78b70a440
is shared with parsed summary. TargetPython228routing/44ABI and original
lowering CPU contracts also pass. Existing own tiny CUDA math AST is unchanged.

Actual model driver491667/start332519497 launched2026-10-26T01:20:43UTC
on the same frozena48source, with native-on, small protection1, output lowering1,
large-linear1, post-ready provider probe, full105, unchanged official serving
and benchmark arguments, three short+one full warmup and one scored round
per4K/16K, plus post-score semantics. Output
/workspace/experiments/minicpm-s2/projection-integrated-model-v3-20261026.
No model precision or throughput result is yet available for the integrated
policy. A separate CPU waiter will run the exact same source/initial cache/
SO/probe/full105/warmups with only both experimental flags disabled after
candidate complete+cleanup, precision/semantics/working gates and GPU idle.
Control remains deferred on a near99precision result for later diagnostic
review; candidate itself still measures throughput per the established rule.
Production defaults remain off, PR stays Draft, no main merge is authorized
by component results alone.

At09:28:32Shanghai the exact model driver491667/server491671/Engine491819
and CPU waiter496536 were verified live with matching start ticks. Initial
mixed semantics pass (4652-token long arithmetic returns85, concurrent count
1..128 correct, HTTP overlap). Separate post-ready1739-token/one-output probe
positively selects supplied linear on all4geometries, with recorded call
counters up to32forO/QKV/down and8for edge gate. This is actual service use,
not startup installation metadata. KV capacity is548320tokens, equal to the
previous native-on control; original PR58out-only was547632. This observation
does not isolate which memory-budget term changed or prove16K throughput.
Full105 is currently running; scores and final/post-warm semantic results are
still pending. Exact selected paths/identities/probe response/installations
and private cache source are in projection-integrated-model-v3-initial-positive.json.
The actual CPU-only control waiter496536/start332536002 waits for driver
491667/start332519497; state exports/projection-integrated-control-v1-state.json.

## Complete integrated precision pass

The same frozen a48model completed all105unique Level3 predictions with no
failed predictions. Reported accuracy is0.981; independent reading of all105
review scores confirms103binary-correct/105, exact fraction0.98095238, all
review statuses successful. Initial mixed and post-ready provider proof remain
passed. This is the integrated row-policy's full accuracy result, not a
component inference. Original command remains temperature1/top_p.95/max32768,
batch8, no shortened tail. The last long question was allowed to finish.

Full raw105/command/reports/reviews/predictions/initial model state/probe/log
archive785717bytes SHAe1f7c05a44f156848b9ba77f8059b8777edddde15ba4299abce3142df7ac8c5d
is shared alongside integrated-model-v3-full-accuracy.json. Performance/TTFT
and post-score semantics remain pending; short/fullwarm results are not scores.
No production merge or default change is implied by this precision pass.
