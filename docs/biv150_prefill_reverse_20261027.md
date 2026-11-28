# Single-sequence prefill CTA routing by causal work skew

Status: Draft, the refined route's two-component/13-state device screen completed;
the full numerical matrix and service/model acceptance remain pending.
`precision_gate=false`, `official_score=false`. Measured team baseline:
`241220723a09061c182764f8f738e5ec01190367`.
Production, SDK, dependencies, official launch parameters and benchmark are unchanged.

The frozen refined source `140cecc07ba199d199eeb620debe3082195ffacb` completed
strict-resource AB/BA/AB timing with median speedup **1.0345287859985568** at
query2048/KV4096 and **0.9925539958554824** at query2048/KV16384. All13 dynamic
metadata states passed their recorded guards. Both actual postlaunch binaries
have113 registers/0 spills/shared65536 with unchanged B256/Q32/w16/K64/s1.
The [raw archive and CPU audit](evidence/prefill-reverse-20261027/refined-first-falsifier/README.md)
preserve all26 originals, recompute timing/guards/resource phases and replay the
actual Manager's Q-hoist proof against saved actual IR. The useful short-context
direction remains open; long-context residual cost is still about0.75%, so this
screen does not satisfy the zero-loss goal. No service gain or model precision
acceptance is claimed.

A [minimal explicit-CPU-metadata host component](biv150_prefill_host_component_20261027.md)
now executes the conservative ratio decision at the already-parent-selected Q32
boundary. Its CPU protocol/source checks pass13 boundary and45 rejection cases,
including real main56f Q32/M2 Python-policy AST. It preserves long-context original
Call identity and introduces no long-context device predicate. This is no service
hook or device result: the new host component's numerical/performance/model gates
remain unverified.

The first unconditional-reversal falsifier at
`fcaec8af5e5bb674c16920134c5101a22e853e92` passed its numerical and Q-hoist
guards for both tested components. Known launch/shared metadata matched. The
subsequent CPU phase audit confirmed matching post-launch SDK counts of
113 registers/0 spills for both exact kernels. Timing outcomes were opposite:

| Original component | Query rows | KV rows | Median baseline/candidate speedup |
| --- | --- | --- | --- |
| chunk4k | 2048 | 4096 | 1.0382605009776722x |
| chunk16k | 2048 | 16384 | 0.9671327473679263x |

These observations preserve the useful short-prefix result while motivating a
general workload-based route. They do not establish statistical significance,
full-matrix acceptance or service gain. The original
[result/owner/log and SHA verification](evidence/prefill-reverse-20261027/first-falsifier/summary.json)
retain both results. The experiment stays Draft; no model or official service
acceptance has been performed. No GPU run was started for this refinement.

## Why this experiment

The primary [FlashAttention-4 scheduling discussion](https://tridao.me/blog/2026/flash4/)
identifies variable causal-loop lengths and proposes longest-processing-time-first
query tiles to reduce last-wave tail, including use beyond Blackwell. The novelty
audit found no earlier implementation or measurement of this exact candidate in
main docs/tools/patches, historical reports, all-ref commit titles or
`git log --all -S 'tl.num_programs(0) - 1'`. Older C/D worklists discuss planning,
head scheduling and query tiling, without a result for this permutation.

A long prefix makes the first and last causal tiles have nearly equal work;
reversal has less useful skew to address. The new first-principles candidate uses
`context <= 4 * actual_query_len`. Ignoring tile rounding, this asks for at least
about 25% growth from first to last CTA when context is nonzero. Four is an
unvalidated conservative candidate, not a measured boundary or a parameter sweep.
No branch refers to 4K/16K, a case name, a score or a benchmark environment.
Hardware launch ordering, metadata-load CSE and any refined-route gain are unproved.

## Actual device-metadata route

Host admission retains the actual parent's single-sequence 2D causal Q32 path,
with identical Q32/BM256/K64/w16/s1 geometry and grid. The candidate loads only
three CTA-uniform scalar metadata locations:

```python
q_start = cu_q[0]
actual_query_len = cu_q[1] - q_start
context = seqused_k[0] - actual_query_len
reverse = (q_start == 0 and actual_query_len >= 1024
           and context >= 0 and context <= 4 * actual_query_len)
logical_id = grid_x - 1 - physical_id if reverse else physical_id
```

The generated source uses `tl.where` and signed int64 arithmetic for the condition,
avoiding threshold overflow and any host `.item()` or device-to-host readback.
The metadata must be immutable during a launch, as required by the original
kernel. Every CTA then selects the same permutation: full reversal or identity.
The complete original grid, including overspecified early-return CTAs and storage
padding, is retained. Every logical work item keeps its original math, masks,
query rows, head and output addresses. Native causal KV loop order, Q-hoist proof,
normalization, reducer and parent dispatch are unchanged.

The original normalization source is derived from audited SDK SHA256
`54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f`
and helper SHA256
`59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1`.
Normalized source SHA256 remains
`fa6ddc420a95102d116714fcbea41bc81aa9c09379d68518af65755c015b8744`;
the conditional source is
`572c3a765b8206d0debd7617999c0c1e26ade387abf72493b1bf0b53f42408a3`.
Exact reverse substitution restores every original byte. The six inserted source
lines move Q debug anchors from [185,186,187] to [191,192,193]; the unchanged
Manager inspects actual `candidate.fn` plus the original SDK file, so its Q-load
proof follows the real line map. The old unconditional source SHA256
`38198704c37479e4b5c26778d7dd1da2a894409d1c81ff2626e1d988facca104`
remains reconstructible with the factory's historical CPU mode.

Three entry loads may overlap the original helper reads, but compiler CSE is a
hypothesis to inspect in actual IR. Added loads/predicate/int64 arithmetic may
cost time or registers even on the identity route. Default strict comparison
requires known post-launch SDK counts and known matching launch/shared metadata
before timing; unknown fields never pass by comparing None to None. The refined route must
retain the short-prefix gain and avoid long-prefix loss in device measurements.

## C3 rebasing, nonzero starts and padding

Actual main C3 code constructs
`metadata.query_start_loc[prefix:count + 1] - prefix` and slices Q/output at the
same natural prefix boundary, yielding a zero-start single suffix. CPU execution
of that exact assignment confirms prefix sizes 1,31,63,100.

Route selection uses the logical `cu_q[1]-cu_q[0]`, not tensor capacity.
Thus a padded q.shape[0] of 2048 with actual query length1023 keeps identity.
Nonzero starts also keep identity conservatively. Actual helper address proofs
cover valid nonzero starts1/31. Starts>=32 can make the original SDK resolve a
low global ID to sequence-1; this existing unsupported metadata is not made into
a new fast path or sent to a device numerical test. Keeping identity preserves
parent behavior without claiming to repair such metadata.

## CPU proof and next device checks

The [new CPU evidence](evidence/prefill-reverse-20261027/cpu-causal-skew-route.json)
executes actual SDK helper/address AST and generated route-prefix AST across
24 fresh/prefix/suffix/partial/padded cases. Threshold cases include query2048
with prefixes6144,8191,8192,8193,10240 and query1024 at prefixes4096/4097.
Both outcomes are complete bijections and involutions for every grid size1..2049.
Every checked logical output row-head has one owner; padded rows have none.
The decision's dependency graph has no program-ID input, and all CTAs load the
same three metadata addresses. Actual parent-policy/Tap controller and scope
checks still run on CPU without Torch or Triton imports.
The original unconditional [CPU evidence](evidence/prefill-reverse-20261027/cpu-result.json)
is retained separately.

```bash
python3 tools/check_biv150_prefill_reverse_cpu.py --source "$PWD" \
  --sdk-source /path/to/audited/triton_unified_attention.py \
  --helpers-source /path/to/audited/triton_attention_helpers.py \
  --out /tmp/prefill-causal-skew-cpu-result.json
```

## Device validation entry and completed partial screen

The dynamic gate keeps all pointers and the 65x2 grid fixed. It captures baseline,
candidate and an exact generated route-ID probe once each, then updates only
`cu_q` and `seqused_k` across thirteen states: short/long prefix, prefix8192±1,
actual qlen1023/1024/1025, supported nonzero starts1/31, then back to short prefix.
CPU checks confirm the probe and real candidate have identical route-prefix AST.
The device gate verifies both heads' complete logical-ID map and
bit-exact direct/graph output, CPU-double reference guards, inactive output
canaries, input bytes and descriptor/ABI identity. This precedes component timing.

The completed first screen used the two original components and the dynamic gate:

```bash
python3 tools/bench_biv150_prefill_reverse.py --source "$PWD" \
  --commit FROZEN_PR_COMMIT --run-gpu --dynamic-metadata \
  --case chunk4k --case chunk16k --out /tmp/prefill-causal-skew-first
```

Omit the two `--case` flags for all nineteen ordinary fresh/prefix/suffix and
threshold cases. The result distinguishes completion of requested work from
`refined_route_numerical_matrix_complete`, which requires both the entire matrix
and dynamic gate. That field is not precision, performance, CSE or service
acceptance. Passing a two-case screen remains partial coverage.

Both routes use the unchanged actual main Q32 policy and Q-hoist verified manager.
Before timing, the harness requires exact numerical identity/reference guards,
changed-input graphs, side streams, input/output canaries, identical ABI/launch
descriptors, known equal shared/launch metadata and, by default, known equal post-launch SDK
register/spill counts.
It records the actual kernel source, all IR/ASM/cubin and manager proofs so root
can inspect added scalar-load CSE and codegen. A live port9031 service, busy GPU,
dirty tracked source, production differences or a wrong import checkout stop the
run without stopping anything else. Component graphs replay warmed private JITs
with the parent's actual arguments; this does not claim the serving wrapper's
capture path has been integrated.

The partial screen establishes its recorded numerical/route guards, actual
Q-hoist retention and observed resources. The full matrix, metadata-load CSE,
service throughput and model precision remain unproven. Three extra source-level
loads may not CSE through a dynamic sequence index; no reuse is inferred from
equal resource counts. A [read-only host-routing feasibility study](biv150_prefill_host_route_feasibility_20261027.md)
examines using existing CPU metadata to select unconditional reversal or the
original Q32 binary. It implements no hook/kernel and starts no queue.
Root coordinates all GPU work serially.
Full model precision and unchanged official throughput/TTFT acceptance remain
mandatory before integration.


## Resource evidence phases and optional diagnostic recording

The first compile records contain `candidate_resources.n_regs/n_spills=null`
before handle loading. Those null values establish no register/spill equality.
The same records later contain explicit `loaded_resources` with 113/0, and the
final case resource snapshots also contain 113/0. The CPU
[phase audit](evidence/prefill-reverse-20261027/first-falsifier/resource-phase-audit.json)
checks hash, metadata and all ASM hashes across pre/post snapshots and final
cases. It also cross-checks the unchanged main
`tools/summarize_biv150_prefill_loaded_resources.py`, whose PR47 evidence likewise
distinguishes compiled null from observed loaded counts. This uses existing
artifacts only and does not relaunch a frozen job.

The future harness now records `post_launch_resource_evidence` with explicit
phase and source. It reads SDK attributes only after an actual launch and may
reuse the Manager's existing loaded record after proving binary identity.
A pre-launch record alone, or `None==None`, never establishes observed equality.
`--resource-mode strict` remains the default and stops timing if required counts
are unknown or known tested fields differ. Optional `--resource-mode record`
permits known register/spill changes while retaining matching known launch/shared
metadata; it labels `resource_changed` and attributes timing to the full candidate
including metadata/index work and resource costs. Unknown counts stop both modes.
No unsafe compiler parameter is added to force 113 registers.

This CPU-only audit added no `cuFuncGetAttribute` binding: no pinned COREX function
attribute ABI or live handle was available to inspect locally. It never calls
`_init_handles`, initializes a driver, or launches a kernel to obtain metrics.
Direct driver `LOCAL_SIZE_BYTES` and `SHARED_SIZE_BYTES` remain explicitly unknown.
Compiler shared metadata 65536B and SDK `n_spills=0` do not prove complete driver
resource equivalence, occupancy, absence of all local-memory traffic, or CSE.
The refined route now has observed postlaunch113/0 counts in its completed
partial screen. Metadata-load reuse remains unverified.

```bash
python3 tools/audit_biv150_prefill_resource_phases_cpu.py \
  --result docs/evidence/prefill-reverse-20261027/first-falsifier/result.json \
  --out /tmp/prefill-resource-phase-audit.json
```

## CPU AST bridge correction after the 1b4 preflight failure

The first host-binding attempt at frozen `1b4eae8fb922c9e5cf6537d9845b43023ebf3577`
stopped in `run_cpu -> source_chain -> exec(actual NONE wrapper)` on target
Python3.12: `NameError: name 'Any' is not defined` at wrapper line310. It never
entered `gpu()` and produced no GPU, resource, numerical or performance data.
Owner624317/start342842999 is recorded same-start Z. The two original files
(2923B), original manifest and traceback are retained unchanged in
`host-binding-annotation-v3/pre-gpu-v1-raw.tar.gz`: 1430B, SHA256
`8220b1350510f6a43cc2372e6c78be9cd5b10e1523fb2b10dbbd20c5110795ff`.

The actual wrapper has `from __future__ import annotations`, but the CPU bridge
extracts its method without that module import. The old transformer removed
ordinary/keyword-only argument annotations while missing `*args: Any` and
`**kwargs: Any`. Python3.14 lazy annotations hid this missing-name dependency.
The fix is confined to the CPU checker: `visit_arg` removes every argument kind;
return, local, nested and async annotations are also removed. Partial-AST
compilation sets `flags=0, dont_inherit=True` and checks the future flag is absent.
Unsupported future-import or generic-type-parameter scopes reject explicitly.
No typing names, actual metadata or framework modules are fabricated.

Five valid calls preserve signatures, parameter kinds, defaults and object
identity, including positional-only names forwarded through `**kwargs`. Eight
invalid calls retain the original TypeError text; async/nested star-argument
calls also match. Tests inspect `__annotations__` explicitly, so lazy3.14 cannot
hide a leaked annotation. The local macOS CPython3.12.15 reproduces the original
failure and passes the fixed default CPU entry; local3.14 also passes. Both new
CPU reports are byte-identical, SHA256
`22b439563b2599a309f6c20ae441a149c589a10e53b2fda727b02c69bd675477`.
The original local3.12 reproduction trace and provenance are committed beside
the report. Old 140/fca/67/e4/1b4 evidence remains unchanged.

Binding, benchmark, kernel factory and production bytes are unchanged from1b4;
all existing numerical, SDK/source, typed-ABI, Q-hoist and resource gates remain
required. This is CPU bridge compatibility work, not host-component/device or
model acceptance. The actual Linux target preflight remains for root, before
any GPU attempt; default invocation below imports no Torch/Triton or GPU driver:

```bash
/usr/local/bin/python -B tools/bench_biv150_prefill_host_binding.py \
  --source "$PWD" \
  --sdk-source /usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py \
  --helpers-source /usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py \
  --target-sources docs/evidence/prefill-reverse-20261027/host-binding-v1/actual-target-sources.tar.gz \
  --out /tmp/NEW_TARGET_CPU_PREFLIGHT_DIRECTORY
```

## Completed e684 host-binding component screen

Root's actual target3.12 default CPU preflight passed before the e684 component
attempt. Owner624624/start342942814 is recorded same-start Z. The requested two
cases completed with errors=[], parent restored and requested component gates
passed. The original `full_numerical_matrix_verified` remains **false**; this is
not the complete shape matrix, model105 or service throughput/TTFT acceptance.

| Requested component | Wall AB/BA/AB ratios | Wall median | Event median |
|---|---|---|---|
| Q2048/KV4096, reverse | 1.0512968721960483 / .9788574640839791 / .9715321071846305 | .9788574640839791 | 1.0401160131016387 |
| Q2048/KV16384, original parent | .9966925738197613 / .9948812711721242 / 1.0054336550269167 | .9966925738197613 | .9967260885414839 |

Two changed-data patterns and changed actual KV4095/16383 passed exact direct,
selected-kernel Graph, actual SDK generic capture fallback, stale-contract,
side-stream, input/canary and sampled CPU reference guards. Actual typed ABI,
target/options and Q-hoist records passed. Baseline and selected binary have
observed113regs/0spills, compiler shared65536; direct driver local/shared/occupancy
remain unqueried. Long context retained baseline binary93f15053… and did not
touch the reverse manager. Source381987… and all e684 guards remain unchanged.

All75 original files5902115B plus manifest are in
`host-binding-completed-v2/v2-raw.tar.gz`, 730748B, SHA256
`5d8a00a8adda24d5658a391627457f15d51ba4661bd3c692b92cc72e82c14f7e`.
Summary SHA256 `b067baf3d1b31ea30412ad46756120a192591e1aff82b6b6f5531d9370e20d17`.
Although the run manifest lists private cache as excluded, the downloaded75
originals include it; this archive preserves every downloaded file, including
opaque launcher SO/cubin/cache bytes, without loading them. CPU replay verifies
every SHA, recomputes all rounds/medians and resource comparison, and checks the
recorded numerical/Graph/ABI/Q-hoist gates. It does not rerun device comparisons
or claim a new independent IR-transform proof. Six malformed-result faults
(matrix, bitwise, reference, Graph, ratio, cleanup) reject. Unpack then replay:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/archive_biv150_prefill_host_binding_cpu.py \
  --snapshot SNAPSHOT_DIRECTORY --out NEW_DIRECTORY
```

4K candidate wall-minus-event residual is .2094/.3364/.3365ms per call versus
baseline .2361/.2328/.2312ms. These residuals are not a CPU phase profile: event
and wall ranges include different synchronization/host intervals. The raw run
does not identify GC, scheduler or individual-call costs. The private adapter
adds repeated descriptor checks, dataclass/context allocations and per-call
diagnostic records; those are concrete optimization targets, not a proven
causal explanation of the late-round difference. The event result keeps this
direction worth investigating, but there is no demonstrated wall/service gain.
16K's small differences are not pursued as a stable effect.

## Private host adaptation after the e684 screen

This separately committed CPU adaptation retains the old381987 unconditional
reverse kernel and the original M2/Q32/verified-Manager chain. It adds no service
hook. Source installation checks pin main56f's M2, Q32 and descriptor files, then
compile their complete modules on CPU and extract the selected code without
executing the reference modules, requiring equality with loaded function code.
Qualification also checks the current policy links,
predicate identity/code/defaults. Unknown or changed qualification fails closed.
These source/hash/AST operations occur once at setup, never on the hot path.

The unchanged Q32 policy validates all current Tensor descriptors and launch
flags before applying its fixed BM128->BM256/Q32/w16 edit. The new boundary also
requires the real caller frame's code to be that exact parent `invoke` code;
an arbitrary direct caller retains its original Call identity. Only within that
certified parent-selected boundary can a successful private duplicate descriptor
qualification be reused for the fixed geometry/mode ABI. It caches neither Tensor
metadata nor Q/U decisions. Mutable shape/stride/dtype/device/storage alias and
metadata descriptors are still checked by the original parent on every call;
the original reverse Manager also retains its live checks and Q-hoist compiler
audit. Core Router.select bytes remain unchanged, including fresh Q/U, zero-start
packing, one-shot lease/stamp, current query identity, exact grid/scalar types,
scope and late-decision checks. No benchmark labels enter routing or caching.

Without installation qualification, the original full private descriptor check
runs on every call. The optimization is intentionally not a general direct
launcher API. It requires serial coherent caller metadata and the pinned parent
source. Parent method/predicate/code/default changes revoke the private scope
before another launch; there is no automatic re-enable. The component remains
uninstalled from production and cannot be activated by missing scope assertions.

The per-launch Boundary closure becomes a small slots object owning that exact
grid reference. The admitted metadata extent uses the same one-shot Frame and
ContextVar set/reset directly, avoiding the generator/contextmanager allocation.
Nested/unknown/capture/long calls retain their existing parent path. Both context
entry and exit failures still restore the wrapper, propagate the same exception
and never retry. All per-call audit dictionaries/last records remain generated
inside the launch path; no reporting materialization or mandatory check is moved
out of the charged interval.

The benchmark changes only binding-constructor configuration and a final adapter
snapshot. A CPU exact inverse reconstructs the entire e684 benchmark and checks
SHA `cd1b4b3b8382d790d630d42cd661cbf28203547de0d607d52986171d3e0a3b58`.
Warmup counts, untapped baseline, repeats, AB/BA/AB order, event/wall ranges,
numerical/reference/Graph/side/canary/typed-ABI/Q-hoist and resource gates are
byte-identical. All new host work stays inside the same candidate call charge.

The initial0cc CPython3.12.15 and3.14 default CPU entries passed with byte-identical report
`host-binding-qualified-v1/cpu.json`, SHA256
`5e407f15822cedfe3567dd1952b1dd2d92838d03086d17a3abc35d61db86c0dd`.
That fixture used partial classes and did not establish actual module-import
code identity; the complete-module fixture and actual diagnosis below supersede
that part of its proof. The original report is retained unchanged.
CPU profiling of the actual pinned predicate sees two parent BM128 checks for
two cache-hit SDK calls, and zero duplicate private BM256 checks; without a
certificate it sees two of each. Five post-cache stride/dtype/alias/metadata
mutations still miss into the original parent. A valid padded row stride stays
eligible. Foreign direct callers, changed code/defaults, stale contracts,
nested/M2/capture/long scopes and context/launch errors are covered without Torch.

This proves a bounded CPU protocol and removal of repeated work, not a host or
device speedup. The late e684 wall residual has no measured phase attribution.
Root must first repeat the default CPU entry on the actual target, then decide
whether to run the unchanged bounded GPU component with this new source freeze.
Actual typed ABI/Q-hoist/resources/numerical/Graph gates and wall/event ratios
remain pending for this adaptation. Model105/service precision and official
throughput/TTFT are unverified; no gain or merge acceptance is claimed.

## 0cc parent-certificate failure and complete-module identity reconstruction

The actual0cc attempt owner625949/start343310889 is recorded same-start Z. Target
default CPU preflight passed, but LocalSdkBinding construction rejected loaded
parent code before the attention case loop. There is no result.json, attention
timing, numerical or new attention-resource record. Device discovery had occurred
and created an opaque cuda_utils SO; this is not the earlier pre-GPU Any failure.
All7 original files71633B plus download manifest remain unchanged in
`host-binding-parent-certificate-v4/v3-failure-raw.tar.gz`, 16883B, SHA256
`c179b1436e510b1fbd24f2df8d72e6c349515c925fc13c5f56e32b6a29b7c307`.
The original pending manifest stays pending, and its original CPU report stays
5e407f…; no synthetic result replaces the missing device result.

Root's independent actual-loaded diagnosis is preserved as
`target-loaded-diagnostic.json`, SHA256
`5d14d6da6b0744fe95c25b4738d05dd5eacb1f65d08f7530303eeabd3e6a4027`.
All three actual module files come from the original v3/0cc worktree and have
the expected source SHA; there is no site-packages import mismatch. Actual code
equals complete-module expected code for all three functions, while all three
old partial-AST expected codes differ. On actual CPython3.12:

- M2 and Q32 differ only in nested `invoke.co_code`: offset156 is full-module
  LOAD_GLOBAL9(NULL+qfragment) versus partial LOAD_GLOBAL8(qfragment); offset166
  is full LOAD_ATTR10(_eligible) versus partial LOAD_ATTR11(NULL|self+_eligible).
  The partial class lost the module import symbol table and its call optimization.
- Fragment `_eligible` and its generators lose source future.annotations in the
  partial reconstruction: main flags16777219 versus3, generator16777267 versus51.
  Filename/qualname/closure/optimization mismatch was not the cause.

The correction compiles the entire pinned original byte string with flags0,
dont_inherit=True and current sys.flags.optimize, matching SourceFileLoader.
It only extracts `_Policy.__getitem__`/`_eligible` code from co_consts; it never
executes the reference module. CodeType equality plus explicit recursive field
comparison remain strict, including nested bytecode, flags, filename/qualname,
closures, line/exception tables and constants. Loaded code/global source paths
must resolve to the pinned source root. A mismatch names the module and exact
field paths, aborting preflight without alternative compilation or fake approval.
All dynamic qualification/intact, Router, fresh metadata, cache, timing and
numerical/resource gates are unchanged.

The revised CPU fixture executes complete pinned parent module definitions with
only their package imports stubbed, preserving their real compiler context and
future flags. It imports no Torch/Triton and never instantiates/installs Managers.
It reproduces all three old partial mismatches; five code/location/future/nested
import-call mutants reject. Local3.12.15 and3.14 strict default entries pass with
identical CPU report SHA256
`30dc23a450c62fa8fd9aaa80e8929a6067f565a58a682f044366f8d80576a397`.
Existing live-parent profile, post-cache descriptor mutations, foreign-call
identity, stale/nested/capture/M2/long and error-restoration checks still pass.

`tools/diagnose_biv150_prefill_parent_code.py` defaults to stdlib compile-only;
its explicit --loaded mode imports parent modules but calls no install, CUDA,
JIT or launcher. Root used the standalone equivalent for the actual diagnosis.
The actual code-identity issue is explained and corrected; new target default
CPU and bounded device acceptance remain for root. The old kernel benefit and
0cc host adaptation are retained. No new host/device performance, model105 or
service/official-score result is claimed by this identity-tool correction.

## Completed5967 host screen: actual certificate passed, wall attribution pending

Owner626303/start343560965 is recorded same-start Z. The requested two cases
completed with errors=[], parent restored and requested component gates true;
full_numerical_matrix_verified remains false. The actual whole-module/code/path
certificate passed, with1 private qualification miss and61 hits. Original typed
ABI/Q-hoist/resource, exact direct/reference, changed-data/metadata Graph,
side-stream/stale/input-canary guards passed without changing their thresholds.

| Component | Wall AB/BA/AB | Wall median | Event median |
|---|---|---|---|
| Q2048/KV4096 | .9800652007574565 / 1.1485986345973271 / .8698464399611556 | .9800652007574565 | 1.039341118669025 |
| Q2048/KV16384 original parent | .9958128626440447 / .9949167173590074 / .9661089244611075 | .9949167173590074 | .9995017779780487 |

4K baseline wall1.686874/1.684085/1.494356ms and candidate wall
1.721186/1.466209/1.717954ms show discrete gaps around.2ms, while event ratios
remain1.038730/1.039341/1.041175. No CPU phase/GC/scheduling data identifies their
cause. Further timing-round sweeps are not justified; the next work is a bounded
diagnostic, without interpreting profiled timings as performance. The old kernel
benefit remains of interest, but neither this wall median nor the small16K
difference establishes a host/service gain. No model105/service/official score.

All75 originals5912292B plus original manifest are preserved in
`host-binding-completed-v4/v4-raw.tar.gz`, 733124B, SHA256
`3fd8d1ba36054181c8cd7887f7f53f1c097c07188fe6af554b32146c50f28672`.
Summary SHA256 `ea08f5888cb2911ce3d74ad2a6cc866d053521681b526663ea73fb6aa8651abe`.
Every downloaded file, including opaque private cache/SO/cubin bytes, is retained.
Raw SHAs, recorded gates, resource/identity records and all ratios/medians are
checked on CPU; six faulty-record gates reject. Unpacked replay is byte-identical,
and the earlier default e684 replay still reproduces its exact prior outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/archive_biv150_prefill_host_binding_cpu.py \
  --run 5967 --snapshot SNAPSHOT_DIRECTORY --out NEW_DIRECTORY
```
