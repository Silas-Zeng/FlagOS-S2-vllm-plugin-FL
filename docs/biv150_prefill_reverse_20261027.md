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
