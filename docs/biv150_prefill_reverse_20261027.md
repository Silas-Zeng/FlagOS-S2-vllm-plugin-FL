# Single-sequence prefill CTA routing by causal work skew

Status: Draft, refined shape route has CPU proof only. `precision_gate=false`,
`official_score=false`. Team baseline:
`241220723a09061c182764f8f738e5ec01190367`.
Production, SDK, dependencies, official launch parameters and benchmark are unchanged.

The first unconditional-reversal falsifier at
`fcaec8af5e5bb674c16920134c5101a22e853e92` passed its numerical, Q-hoist and
resource guards for both tested components, with opposite timing outcomes:

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
cost time or registers even on the identity route. Launch/shared metadata and
loaded registers/spills remain guarded before timing. The refined route must
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

Device tests must cover metadata crossing both route outcomes in one captured
graph with fixed pointers/grid, the qlen1024 boundary, partial query tiles and
supported nonzero/padded metadata. A route-ID probe generated from the exact
predicate is prepared; the dynamic graph entry is being completed. Actual IR
must establish scalar-load reuse, selected order, Q-hoist retention and unchanged
resources. Root coordinates every GPU run serially. Full model precision and
unchanged official throughput/TTFT acceptance remain mandatory before integration.
