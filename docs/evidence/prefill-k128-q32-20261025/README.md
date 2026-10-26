# Prefill tile/pipeline component exploration

Current status: BM256/K64/w16/stage1 won the completed component comparison below.
A shipping-default service candidate is being integrated and remains unqualified
for model accuracy/throughput until the full acceptance completes.

The existing PR46 Q32 lineage remains experimental. This branch now includes its strict-binding corrections and current team main34942f7, including the BF16 small-row correction. The only merge conflict was EXPECTED_RUNNER: it now hashes the actual combined runner268804cb; removing only the Q32 installer exactly reconstructs the team runnercbbbabce. All local strict module pins and inverse Q-fragment source reconstruction pass. Q32 controller CPU gates pass. Old partial/confounded model results remain unchanged in their original evidence directories.

The old whole-round v2 diagnostic on the earlier baseline attributes209.50/555.89 retained seconds to non-Graph prefill attention in16K (about37.7%). This is a prioritization clue, not the current candidate's measured hotspot share or an uninstrumented score. Q-fragment hoisting is already main; this probe keeps that mechanism and tests loop/data reuse, rather than claiming it again.

Bounded next configurations: BM256/K64/w16/stage1, BM256/K128/w16/stage1, then BM128/K64/w8/stage2 and BM256/K64/w8/stage2. Compare each against BM128/K64/w8/stage1. Existing BM256/Q32 uses202 registers and64KiB shared; wider KV tiles or extra stages may exceed resources. Record actual registers/spills/shared and lowering proof. All numerical/changed-input poisoned Graph gates must pass before timing. CPU CLI/source checks do not establish stage2 lowering correctness, hardware support or speed.

Production defaults and service/benchmark parameters are unchanged. No new GPU task is queued: PR58 owns the GPU and its exact-source full105 matched control has priority. After that qualification, select the next component from the measured remaining bottleneck; this Draft is not merge-ready.

## Q64 bounded resource hypothesis,2026-10-26

Actual target autotuning rejects shared196608>131072, establishing a131072B
per-kernel launch limit; it does not establish shared capacity perSM or resident
block count. Pinned actual Iluvatar compiler a97f0670 creates CompiledKernel
without loading handles. The shared check is in _init_handles, triggered by
run/launch metadata hook/__getitem__. Q-fragment Manager compiles normal IR,
inspects hash/metadata/asm (no handle init), then transforms/compiles before
launch. Thus a potentially oversized normal BM512 module can in principle be
transformed before launch; hardware/transform capability is still unproven.

Add only BM512/Q64,K64,w16,stage1 to the bounded component CLI and actual adapted
scope/compiler-option CPU gates. Q buffer alone is128KiB; if its lifetime can
reuse K/V shared after hoisting, the final module may reach the observed limit.
This is an estimate, not compiled resources; register pressure/spills and
lowered proof remain mandatory. Existing BM256w8 uses202registers but this is
not a measured BM512/w16 count and must not be extrapolated into occupancy.
No production file/default/service argument changed. All five adapted CPU
configurations pass; no new GPU task is queued while PR59 owns service
acceptance and its serial matched control. After those finish, compare the
bounded prefill candidates against actual main BM128/K64/w8/stage1, including
real2017/1985tails, independent CPU FP64 and changed-input Graph checks before
timing. Component timing is not a whole-model speedup.

## Actual main594 component results,2026-10-26

PR59is now merged as594ae2a9d17aadc4587ad54ade2b34e17f4af990.
This Draft incorporated it throughc854f91;fcc193786fc0146dc0b7d9e0f0ed5d3a652ecdfa
adds only failure/IR evidence retention. TargetPython3.12CPU gates and actual
source/closure off/on checks passed before exclusive component runs.

|Case|BM256/K64/w16/s1 speedup|BM512/K64/w16/s1 speedup|
|---|---:|---:|
|Q2048/KV4096|1.222926x|1.126764x|
|Q2048/KV16384|1.269554x|1.194696x|
|Q1985/KV4096|1.220749x|1.100908x|
|Q2017/KV16384|1.266236x|1.187795x|

Reference is the actual mainBM128/K64/w8/s1Q-hoist path. Both configurations pass
all2patterns/full-output baseline comparisons/independent FP64/random and uniform
causal reference/nondefault-stream changed-input poisoned Graph checks before
interleaved warmedAB/BA/ABtiming. Both finished and GPUidle confirmed. SelectBM256.
No model throughput/accuracy qualification is implied. Actual shared metadata is
64KiB/128KiB; register/spill snapshots precede handle loading and can be null, so
no occupancy attribution. TransformedIR and lowering proof are retained.

Rawprefill-tiles-main594-v1-evidence-20261026.tar.gz117084B,
SHAe54a8c4ec96649da500ab86e490b59bef8d1dc8e701eca174608505cb4a7bc1a,
plus summary in this folder. Both source commits, launches, CPU/stack checks,
rounds, correctness gates and compilerIR are included; private caches excluded.

Next integrate256/w16 through the actual service controller. Existing optional
Q32controller is still256/w8and OFF; turning it on alone would not test this win.
Preserve default main manager behavior, explicitly audit the selected launch
configuration and fallback-before-launch, then cover fresh/minimumQcausal
boundaries before complete105and fixed-parameter4K/16Kmodel acceptance.

## Actual service integration and boundary gates, 2026-10-26 15:47 UTC

Production commit 1a0bb96 selects BM256/K64/w16/stage1 through the existing
verified controller for audited single-sequence BF16 prefill with >=1024 query
rows. Its proposed shipping default is ON; subsequent diagnostic commits through
a65aa43 do not alter vllm_fl or csrc. Earlier OFF/w8 status above is historical.
Explicit launch fields retain BM128/w8/stage1 defaults for existing manager callers.
The effective Q32 default and parent enablement are included in compilation identity.

Frozen target d207eeb passed four CPU checks, 12 actual-controller GPU cases
(two patterns each), and fresh1024/fresh2048/minimum16k nondefault-stream,
changed-input poisoned Graph checks. Actual installed/selected metadata is w16,
stage1, shared65536, and the feature environment variable is absent.
Additional warmed component medians are 1.108029x / 1.219792x / 1.203558x.
These remain component timings, not model scores.

The fresh1024 random case retains a raw CPU FP64 oracle failure shared by the
accepted baseline and candidate: one sampled element differs by one BF16 ULP.
All 2,097,152 baseline/candidate outputs in this case are bitwise identical.
The added diagnostic guard requires finite output, the original element tolerance,
or a pre-existing <=1-ULP baseline violation with no increased error. It rejects
new/worse violations and baseline exceptions above one ULP. Raw metrics remain
visible. Full 105-question competition precision >=0.95 is unchanged.
The original first-failure tensor is retained on the personal instance under
prefill-w16-preflight-v1-20261026/actual-policy-gpu.

Raw boundary archive: prefill-w16-preflight-v2-evidence-20261026.tar.gz,
59821 bytes, SHA256 e6347024f8bb4ba15022a28f027c863c2da0e687ac66bba8497dd7c6a1994b7f.
Its 13 members include actual-controller results/log, three CPU logs, Graph
results/log, transformed IR, launch identities and preflight summary.
The separately retained cache CPU log supplements the archive (the archive's
cache-name filter excluded that log as well as private caches).
The adjacent JSON summary and immutable pair tool proof are included.

Actual model pair started at 15:47:25 UTC on the personal instance only.
Root: /workspace/experiments/minicpm-s2/prefill-w16-main594-pair-v1-20261026.
Controller PID546274/start337719739/SID546274; baseline driver546278 and
server546282. Source main594-qualified-20261026@594ae2a versus
prefill-service-w16-v2-20261026@d207eeb. The fresh Git fetch and GitHub main ref
both confirmed 594ae2a immediately before launch.

BOTH arms use merged projection providers by their production defaults:
--baseline-out-lowering 1 --baseline-large-linear 1 --baseline-production-defaults.
Neither arm has manual provider enabling variables or a Q32 feature variable.
Both run complete 105-question precision, fixed service CLI/official benchmark,
the same initial SQLite snapshot/native library, three short plus one full-workload
warmup and one scored round each of 4K/16K, with post-score semantics and owned cleanup.
Selected configuration fingerprints are read outside benchmark calls.
Harness SHA256 10d11ba60ff85bf2781628d48a664420fee9a4c4ed6e033dbcd996484eb09cc9;
controller 99eda2ee325f2aed84e213d80a75796d511b53ad36b1d07c8565976ab13b12ea.
Their owned cleanup and expected-source ASTs match main594.
Do not overwrite those active exports scripts while the controller is running.
This Draft has no new end-to-end qualification yet. Merge requires fresh-main
throughput superiority plus precision/TTFT, actual default w16 selection and semantics.

## Post-launch resource audit and observation gap, 2026-10-27 CST

The earlier resource summary copied candidate_resources before CUDA handle load;
its null register/spill values remain visible. It omitted loaded_resources already
recorded after successful launch. CPU-only reanalysis of the unchanged archives,
with candidate hash/metadata/all assembly hashes cross-checked, now recovers:

| Configuration | shared bytes | loaded n_regs | loaded n_spills |
|---|---:|---:|---:|
| Baseline BM128/w8 | 32768 | 114 | 0 |
| Selected BM256/w16 | 65536 | 113 | 0 |
| BM512/w16 | 131072 | 201 | 0 |

The actual service-controller and fresh Graph records independently report the
same 113/0 for BM256. These driver fields do not establish occupancy, register-file
capacity or absence of all local-memory traffic. They reinforce retaining BM256
over the slower BM512; no kernel retest or production change was needed.
See prefill-loaded-resources-20261027.json and the adjacent original summary,
which now distinguishes pre-load null fields from post-launch counts.
Reproduce with tools/summarize_biv150_prefill_loaded_resources.py on the two archives.

The model pair was last verified live at 15:55:03 UTC with exact /proc start ticks:
controller546274, driver546278, service546282; baseline phase accuracy, mixed
semantics passed. Loaded package identity/defaults verified and KV548320.
SSH attempts from 15:56 through 15:58 returned Permission denied (publickey);
verbose handshake reached the Go SSH endpoint at43.99.48.148 and offered the
same explicit Ed25519 key. This is an observation gap, NOT evidence the model
process terminated. Do not restart or schedule another GPU workload until checking
the same controller/start ticks and owned state after access returns.
The user was asked to verify reservation/login information; no teammate instance
was accessed. The full model comparison and merge gate remain incomplete.

## Access restored and baseline precision, 2026-10-27 00:30 CST

A new key restored access at 16:24:25 UTC. The exact original controller546274,
driver546278 and service546282 PID/start identities match; no restart occurred.
The original pair is healthy, baseline phase throughput. The prior observation gap
does not establish that the compute task stopped or that a benchmark was interrupted.

Independent raw prediction/review/report audit confirms baseline main594:
101/105 =0.9619047619,105 unique question IDs, binary scores, all review statuses
success, no failed predictions; actual source fingerprint and provider production
defaults verified with all manual provider/Q32 enable variables absent.
See prefill-w16-pair-v1-baseline-independent-accuracy-20261027.json for cleaned
per-question scores and original file hashes. This is baseline precision only.

Baseline official4K score5778.243627140731tok/s,meanTTFT3502.0497929726844ms,
256/256/no failures and exact1048576input/262144outputtokens. Fullwarm5804.0409746
is unscored. Selected-config fingerprints changed over the complete score command
(including client preflight), and four new index metadata files were recorded;
do not call this completely compilation-free steady state. 16K and the candidate
are pending. Compare the candidate against this matched main, not an older peak.
No service, benchmark, cache policy or production file changed during recovery.

## Completed-pair CPU audit prepared, 2026-10-27 00:46 CST

Published tool tools/audit_biv150_prefill_pair.py is available on the personal
instance as R/exports/audit_biv150_prefill_pair-v1-20261027.py, SHA256
bf12b10669c6a3670b5f27db183852c2838f9b12bd42cbce3efce147349570c8.
It reads existing files only and never merges or starts GPU work.
It requires complete cleanup, independently scored105/105, three short plus one
full workload warmup per length, exact request/token geometry, source/default/
native-library/driver/CLI parity, actual w16 installation AND post-launch selection
joined to compiler proof, and no Q32 preparation fallback. It reports TTFT and
observed throughput changes, retaining cache-change caveats.
Fresh team-main synchronization and the final merge judgment remain external.

CPU validation parsed actual archived GPU selection/compiler records (both JSON
and Python-literal event formats), matched their hashes and metadata, and did not
fabricate missing model-install evidence from component logs. The real still-active
pair was rejected as incomplete, as required; target proof is adjacent JSON.
No end-to-end candidate qualification is implied. After the controller finishes,
run this tool with baseline594ae2a9d17aadc4587ad54ade2b34e17f4af990 and
candidated207eeb7df463d6149b1823fa22ed8ebc8b03d59. Preserve the active
exports/run_biv150_latest_main_pair.py and run_biv150_small_out_acceptance.py.

## Complete matched baseline and actual candidate path, 2026-10-27 01:02 CST

The main594 arm completed all workloads, both semantic checks and owned cleanup.
Independent CPU arm audit also passes on the actual complete model files.

| Main594 score | total tok/s | mean TTFT ms | requests |
|---|---:|---:|---:|
| 4K | 5778.243627140731 | 3502.0497929726844 | 256/256 |
| 16K | 4187.31786930508 | 120478.06613695138 | 128/128 |

Precision101/105, exact input/output token totals, no failed requests and TTFT
working limits pass. 4K has changed selected configs over the command interval
and four new index metadata files. 16K has unchanged selected configs and zero
new metadata. The unscored16K fullwarm3760.478274 remains a warmup.
See prefill-w16-pair-v1-baseline-final-summary-20261027.json.

The same controller automatically started candidate driver556703/start338126553,
service556707/start338126577 (EngineCore556855). It is running complete105 accuracy.
The actual default-ON service selected BM256/Q32/w16/K64/stage1 at2047rows:
hash9ed319192da32eb73ac0932b793abc52b9fbfdfff4f3d69bf226e12cd0b0c87b.
Matching actual compiler metadata reports shared65536 and verifies zero loop Q
memory reads. Installation, post-launch selected hash and compiler proof are joined
in prefill-w16-pair-v1-candidate-live-path-20261027.json. No Q32 fallback observed
at that snapshot. Loaded source fingerprint74b1413eec50ddfa479def5d5507bd5a850f82fdce1b2bc168b4ef313447d00b
was independently verified by the serving harness. Manual enable variables are
absent. Both source-default provider paths, fixed CLI and initial snapshot match;
both have548320KV tokens. This confirms real service integration only.
Candidate full precision/4K/16K and post-score semantics remain pending.

## Candidate complete105 precision passed, 2026-10-27 01:21 CST

Independent raw report/prediction/review audit confirms103/105=0.98095238095,
all105unique, binary and all-success, no failed predictions. Actual default/source
fingerprint checks pass. Compared with this run's main101/105, two question scores
changed0→1; under stochastic evaluation this is not a causal accuracy improvement
claim for the kernel. Cleaned per-question scores, changes and original SHA256 are
in prefill-w16-pair-v1-candidate-independent-accuracy-20261027.json.

The same service has moved automatically to throughput. Three short4K warmups
completed, full4K warmup is active. The existing compute-order path has now emitted
FIRST_SELECTED, with no SCOPE_REJECTED or Q32_FALLBACK in the observed log.
Candidate scored4K/16K, post-score semantics/cleanup and final merge judgment
remain pending. No extra GPU workload or benchmark-parameter change was made.
