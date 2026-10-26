# Single-sequence prefill reverse query-CTA component experiment

Status: Draft, CPU proof only. `precision_gate=false`, `official_score=false`.
Team baseline: `241220723a09061c182764f8f738e5ec01190367`.
No production, SDK, dependency, launch parameter or benchmark change is made.
No GPU or service execution has been performed for this experiment.

## Why this experiment

The primary [FlashAttention-4 scheduling discussion](https://tridao.me/blog/2026/flash4/)
identifies variable causal-loop lengths and proposes longest-processing-time-first
query tiles to reduce the last-wave tail. It explicitly discusses use beyond
Blackwell. This suggests a small, architecture-neutral scheduling hypothesis for
BI-V150; the hardware's actual launch ordering and any benefit remain unproved.
There is no assumed HBM or SFU bottleneck and no adopted NVIDIA-specific opcode.

The novelty audit searched existing main docs/tools/patches, top-level historical
BI-V150 reports, all-ref commit titles, and `git log --all -S 'tl.num_programs(0) - 1'`.
No implementation or measurement of this exact single-sequence prefill reversal
was found. The older C/D worklist discusses work planning, head scheduling and
query tiling; it does not supply a result for this experiment. Existing Q32,
K128 and Q-hoist measurements are retained as the accepted baseline mechanisms.

## Exact scope and transformation

Only actual parent-selected 2D causal single-sequence prefill with query rows
at least 1024 is eligible. The original Q32/BM256, K64, w16, s1 geometry, main
normalization, FP math, Q-hoist proof, original grid and parent dispatch remain.
The private source factory replaces exactly one entry assignment:

```python
q_block_global_idx = tl.program_id(0)
# becomes
q_block_global_idx = tl.num_programs(0) - 1 - tl.program_id(0)
```

The audited SDK source is pinned to SHA256 `54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f`;
its helper is pinned to `59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1`.
The existing main normalization/stride transformation is reused byte for byte.
The resulting normalized kernel SHA256 is
`fa6ddc420a95102d116714fcbea41bc81aa9c09379d68518af65755c015b8744`;
the reversed source is
`38198704c37479e4b5c26778d7dd1da2a894409d1c81ff2626e1d988facca104`.
Reverse substitution restores every original byte and source line numbers stay
unchanged, preserving Q-hoist source/debug lookup.

The full parent grid is `query_rows//32+1` by 2. All launch IDs, including the
extra early-return CTA when rows divide 32, are permuted. No CTA is deleted.
The map i -> N-1-i is a bijection and its own inverse. A physical launch ID now
owns a different logical tile; that logical tile retains exactly its original
query rows, masks, head and output addresses. This does not reorder requests,
change the 3D decode path, widen multi-sequence scope or split extra work.

The causal loop order within each logical CTA is untouched. Compiler resources
are pending device inspection: launch geometry is fixed, while the added integer
index computation may affect registers. A scheduling improvement cannot be
claimed from the CPU work-order proof alone.

## CPU validation

`tools/check_biv150_prefill_reverse_cpu.py` extracts and executes the actual pinned
SDK helper AST with scalar CPU load/min/max stubs. It checks the original sequence
resolution, loop bounds, and original AST expressions for query/output addresses.
Twelve fresh/prefix/suffix/partial fixtures cover 1K/2K/4K/16K fresh queries and
4K/16K total-KV contexts. All logical output row-heads have exactly one owner,
including partial query tiles. Original and reversed logical work sets agree,
including invalid CTAs, and valid reversed work costs are nonincreasing.
A separate complete permutation/involution check covers every grid size 1..2049.
Actual parent descriptor gates reject 3D, multiple sequences, short queries,
changed geometry, aliasing and mismatched grids. Modified SDK text is rejected.
The CPU controller check also executes the unchanged main `_Policy` plus the
actual experiment Tap using fake launchers: both routes receive identical
selected grids/kwargs, the baseline does not recurse through its temporary Tap,
and excluded scopes retain the parent's original fallback. No Torch or Triton
is imported by this CPU checker.

Reproduce against the original audited SDK **files**, without importing them:

```bash
python3 tools/check_biv150_prefill_reverse_cpu.py --source "$PWD" \
  --sdk-source /path/to/audited/triton_unified_attention.py \
  --helpers-source /path/to/audited/triton_attention_helpers.py \
  --out /tmp/prefill-reverse-cpu-result.json
```

The stored [CPU result](evidence/prefill-reverse-20261027/cpu-result.json) records
source/helper AST hashes, scope checks and per-fixture ownership/work summaries.

## Coordinated component command (not executed)

```bash
python3 tools/bench_biv150_prefill_reverse.py --source "$PWD" \
  --commit FROZEN_PR_COMMIT --run-gpu \
  --out /tmp/prefill-reverse-component
```

The default matrix includes all twelve cases. `--case chunk4k --case chunk16k`
selects a smaller diagnostic; partial coverage does not pass the full matrix.
The harness refuses a live port9031 service, a busy GPU, dirty tracked sources,
a changed production tree, or imports from another plugin checkout. It installs
the existing in-process fixture baseline and main default Q32 policy. Both
routes traverse that same policy object; only its private kernel-launch target
changes temporarily. The candidate uses the original main Q-hoist verified
manager and rejects a raw JIT source unlike the CPU-audited normalized source.

Uniform and random inputs must be bit-identical to current Q32 output, pass the
existing CPU-double reference guard, preserve input bytes/output canaries and
match changed-input graph replay and side-stream results. Both routes must
compile identical ABI/constants/attrs and receive identical launch descriptors.
Q-hoist proof is required before graph capture. Graphs replay the warmed private
JITs with the captured parent-selected arguments; this tests component graph
behavior and is not the production wrapper's capture fallback path.

The harness records loaded registers/spills, shared memory, use_sme, warp count,
all code artifacts and Q-hoist manager snapshots. Current prefill w16 resources
are 65536B shared memory, warp_size64 and use_sme114691 (the GQA use_sme14 setting
is unrelated). Changed metadata or registers/spills stop this scheduling-only
experiment before timing. Passing cases run three alternating graph-timing
pairs. Failure evidence is saved and the original parent policy target restored.

Device numerical identity, actual codegen/Q-hoist retention, resources, graphs,
streams and alternating timings remain pending. Root coordinates GPU work
serially. Component timing is not a service score. Full model precision and
unchanged official throughput/TTFT acceptance are required before any integration.
This candidate is independent of the GQA software-exp/rescale experiments.
