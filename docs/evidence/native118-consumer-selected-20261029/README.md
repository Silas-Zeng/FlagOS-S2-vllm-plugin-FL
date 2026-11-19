# Frozen PR118 consumer: selected results

Root's independent review passed for source
`01dfe84028560b1a982dac33bc788e875ce9cf1b` and actual mixed library
`ebbc35d028bb7c719895c8b3569f3120387380888f23c4bb772b7176f0856bba`.
The full private review SHA is
`458cd7638ad5b9df8efb3d3f72f626a3414ee947da5cc12f0524c7c1fc97d653`.
Only selected scalar results and identities are recorded in [results.json](results.json).

The scope is twelve cases: N29/30/31 with S4 and N60/61/63 with S2, each at
Q pitch 2048/2560; 29 physical 16KiB pages, maximum context 193, D128, scratch
width 16 and table pitch 8192. The same-math reference is the explicitly bound
old 8b library, not the current main117 performance baseline.

* All 528 hard array metrics and 960 cross-arm real-row field comparisons passed.
* BF16 proxy: 96/96. The ideal-partial diagnostic remains 0/144; it is not the
  original partial hard gate and was neither dropped nor relabeled as passing.
* All 24 arm records preserve complete readonly-input/guard checks, changed
  Graph-B versus fresh Eager-B equality, and changed-input side-stream results.
  These conclusions use exact frozen execution order and complete snapshots,
  not a profiler trace or a general memory-sanitizer guarantee.
* Saved FP64 references were recomputed locally with maximum delta
  `2.842170943040401e-14`. The saved-reference comparison tolerance `2e-12`
  applies only to that CPU recomputation; original numerical thresholds remain
  unchanged.

This is consumer component evidence. It grants no writer/full-step, 42-layer
capacity, performance, model 105, official-score or service result. In particular,
the subsequently merged `bc0faf1d628b0cf6da0df28ca3a92f8b6bd8d2fe` does not
automatically inherit this qualification. Its main117 parent identity and new
host source proof require separate review; production mixed pins remain `None`
and the feature remains OFF. See the [merge boundary](../../biv150_native118_main117_merge_20261029.md).

The complete private archive has 1732 files/1,194,892,621 uncompressed bytes;
its gzip SHA is retained only as an identity. This commit includes no raw
arrays, cache contents, execution logs, binaries, or full audit JSON, and does
not rerun any GPU workload or numerical audit.
