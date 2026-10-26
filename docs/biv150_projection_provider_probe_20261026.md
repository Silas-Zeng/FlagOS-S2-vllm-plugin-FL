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

## Integrated4K score: no recovered gain established

After3short+onecomplete256-request unscored warmup, the single scored4K is
5603.546050280927 total tok/s, meanTTFT3456.967683712719ms,256/256success,
zero failure,1048576input+262144output tokens. Fullwarm5744.875167 is preserved
as unscored, not substituted for this lower score. It does not recover earlier
PR58out-only5760.266264 or historical PR285765.810177; those are different
runs/sources, not matched isolated effects. The new same-source control and
16K remain pending. Keep Draft and do not merge this observed regression.

Score added4index-linearized metadata files and0MM/linear files. File counts
alone do not bound compile/tuning wall time, but there is no evidence yet for
a large new-matmul compilation confound. Actual2048-row provider log counters
reach65536for O/QKV/down and8192for edgegate, all supplied-linear, so the
main row was actually integrated; this is not merely a1739-only installation.

The existing provider experiment times CUDA Graph chains. It excludes eager
Python/metadata/launch costs and attention interleaving; therefore its1.07–2x
component ratios cannot justify a service gain. After current serial service
and matched control finish, a targeted complete eager-vs-Graph callback probe
or current-model timeline can distinguish entry/overlap costs from tuning/
compiled-kernel changes. No such cost attribution is measured yet. Do not
repeat a large configuration scan or discard the whole output-planning win
on this one combined variant. Raw partial score/new-metadata JSON is shared.

## Complete integrated candidate; quantify noise before attribution

Runtimea48completed103/105 with all initial/post-score semantics, request/token,
working throughput/TTFT and owned cleanup gates. Scored4K5603.546050/TTFT
3456.967684ms;16K4174.742799/120888.913239ms. The16K observation exceeds earlier
4049–4111separate observations, while4K is below prior5760. No matched gain or
causal attribution is established yet.16Kscore has0new metadata. Full performance
archive77514bytes SHA1505161f9304c594264dd718e5e285b2567e783ba34c17fda582fd2e78b581fe
and final-summary retain all warmups/commands/probes/generation/cleanup. Candidate
driver491667isZ,Engine491819gone. Serial same-source control started automatically;
CPUwaiter496536/start332536002,control driver502095/start332901041 and server
502099/start332901063 were verified live; output projection-integrated-control-v1-20261026.

User now prioritizes checking whether small differences are noise. Suspend new
operator micro-optimizations until this is screened. After the existing control
cleans, a single same-config4K A/A service will reuse the qualified a48runtime,
flags and fixed copied final tuning DB, with fresh private Triton cache,3short+
1complete unscored warmup then3predeclared identical full4K repetitions. Full105
is not rerun because production code/flags are unchanged. No sampler/frame/CLI
parameter changes. New/changed metadata or logical non-benchmark tuning-table
changes mark cold/confounded rounds; only a mature suffix of>=2samples describes
steady spread. All raw rounds retained; no best-score picking. Report range/CV,
not a confidence interval. This checks within-process4K variation only; it does
not prove cross-process or16K noise. If a1–3%difference is within observed spread,
it stays unconfirmed and does not justify detailed operator work or merging.
Otherwise use minimal further repeat/cross-process evidence as needed before
attributing a small effect. The A/A runner derives only the4K workload check
and diagnostic path from the qualified owned harness; cleanup AST and serving
command stay exact. It uses original benchmark helper/seed0/256requests/64concurrency/
4096input/1024output unchanged. No A/A GPU job has been launched yet.

A/A scheduling is now actual: CPU-only owner511292/start333129429/SID511292,
launch2026-10-26T03:02:22UTC, state exports/provider-noise-v1-state.json,
log exports/provider-noise-v1-queue.log; it waits for exact control owner
496536/start332536002 and driver502095/start332901041. TargetPython derive/
cleanup/statistics/logical SQLite WAL contracts pass without importing Torch.
Uploaded runner SHA2f9e5e2fd4a7c7782c90d6405eeadd0fa964c28ab6d0196977e809f79f8f7a88,
queue SHA325115b024e6d5a75fca6e1d81dd142c6099608a7ae45fab60d7803f3539ba99.
No A/A score exists yet; no noise conclusion is made. A transientSSH publickey
failure resolved on immediate same-key retry; no model job was restarted.

## Published repeat context and new team baseline (noise check remains frozen)

Fresh fetch at2026-10-26T03:24UTC found team main6971ef48615af97a1637405c71f96dd89969563e,
which merges PR60's prefill stride specialization from34942f7. Do not change the
running a48control/A/A source or flags: these measurements remain scoped to their
frozen34942f7-based runtime. A future production proposal must first incorporate
the actual new team baseline and revalidate the resulting combination.

The teammate's published main34942-prefill-stride-pair.json (GitHub blob
c84a66a759f73244756589fdfad66e0fa394c19a) is retained in
peer-published-stride-repeat-context.json. No teammate instance was accessed.
After discarding the first of four full rounds, the unchanged baseline4K still
increases5504.07→5661.71→5750.80tok/s: observed range/mean4.3755%,first-to-last
+4.4827%. Candidate4K similarly increases5533.03→5677.28→5760.00(+4.1021%).
These are visibly nonstationary sequences, not estimates of random steady noise.
Baseline16K repeats4100.21/4100.36/4099.66 have range/mean0.01707%;candidate
4119.46/4119.07/4119.19 have0.00947%. Do not transfer these finite-sample spreads
across machines, processes, native libraries, DBs or workloads; they motivate
testing maturity separately from variance. PR60's own report explicitly preserves
its small-gain/qualification limitations. Do not infer from4K drift that16K
regressions are noise, or claim all small changes are false. Personal-instance
A/A remains pending and must retain dirty rounds as well as the mature suffix.

The frozen same-source control completed and cleaned at03:33UTC: independent
105successful binary reviews sum102correct/105unique(no failed predictions);
4K5484.242423640023tok/s,TTFT3444.88904705031ms,256/256;16K4102.5176415718315tok/s,
TTFT122508.46646928039ms,128/128. Both initial/post-score semantic checks and
cleanup pass, KV capacity548320. Scored4K adds the same4index kernels as the
candidate;16Kadds0. Apparent candidate/control differences+2.175%/+1.761% remain
single-observation comparisons, not established stable benefits. Full control
raw evidence972655bytes SHAe6b141eb8fa9905bb7961f7187965192eb7efe341ee4e36ad55cefa82319ad0a
and final-summary/independent-review are retained beside candidate evidence.

The CPU waiter actually handed off at2026-10-26T03:33:47UTC, after both control
owners terminated and cleanup/port/GPUidle checks passed. A/A driver512713/
start333317954/SID512713 is running, outputprojection-provider-noise4k-v1-20261026.
This is no longer merely queued;3repetitions remain to be observed. Do not queue
a duplicate from stale control state or an SSH timeout. No new operator work
or PR47/53/55GPU test has been started during the noise screen.

## A/A v1 completed: measurement-state drift, no steady-noise estimate yet

Same productiona48/flags, fixed copied final tuning DB and fresh private Triton
cache, no production changes. Three predeclared full4K scores are:

|Round|Total tok/s|Mean TTFT ms|New kernel metadata|Config changed|
|---|---:|---:|---|---|
|0|5776.982623|3456.093968|14linear+4index|yes|
|1|5879.490541|3433.471060|4index|yes|
|2|5925.396060|3475.200666|1index|no|

All3complete256/256 with exact token counts; pre/post-score semantics and owned
cleanup pass. Complete at03:55:47UTC, driver512713and waiter511292 exited. Full
unscored warmup5899.241267is retained, not substituted for a score. All-round
mean5860.623075/median5879.490541,observed range/mean2.53238%,sample CV1.29652%.
Every round still adds metadata, so valid_mature_suffix=False: there is NO
steady-noise estimate, no confidence interval and no new stable-best claim.
This directly demonstrates that same-code measurement state can create changes
on the scale of previous1–3%observations; stop operator attribution/merging based
on those single observations. It does not prove all small changes false or
explain16K. Before/after device snapshots all1500/1600MHz and38–39C; they do not
continuously observe clocks during scoring.

Full raw80023B SHA256df6f9018deec0404b8ecbc437c28043c4f4bb8905f612edbc5f248acbdeab9ff,
final-summary includes all three raw results, new-metadata lists, DB fingerprints,
semantics, launch and cleanup. Actual SQLite tables use benchmark-named timing
tables and separate selected-config tables, consistent with the existing filter.
Both final DBs have no WAL sidecars. No benchmark data were pruned or best picked.

A bounded v2 diagnostic will reuse this completed identical runtime's selected
config DB and Triton artifacts in a new private directory. Only cache-group file
paths are rebased; all kernel artifact bytes are verified unchanged. Completed
matching source/flags/CLI/native-library/semantic/cleanup are required. SQLite
backup includes committed WAL pages, with equal logical config fingerprints;
configs are never merged. Cleanup AST and official CLI remain exact. CPU fixture
contracts exercise byte preservation/rebase, mismatched seeds, unfinished cleanup,
symlink/path escapes and an open committed WAL. Host and targetPython3.12 pass
without importingTorch. v1exported scripts remain untouched. Planned:3short+1full
warmup then only2predeclared full4K repetitions, no repeated105. This reduces cold
state before asking whether a mature suffix exists; no claim until actual run.
