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
