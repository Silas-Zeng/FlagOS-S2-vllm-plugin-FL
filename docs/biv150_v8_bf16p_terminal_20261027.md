# V8 BF16-P: terminal component result, frozen4a

Retire/close PR73's baseline V8 dataflow: it is slower in every measured full
writer+producer+reducer component case. Do not merge it into serving code. This
does not measure service throughput, model105 accuracy or TTFT. Precision_gate
is false;16KiB/page is still allocated and no KV capacity was reclaimed.

Owner run: `cjj-v8-bf16p-numeric-all-v5-20261027`, PID621444/start341491339,
same start and terminal Z in the downloaded owner record. Source:
`4a1f7ddd38e4bcade30e702faba2e9a7cdb5441d`, operator SHA256
`a7fc539dccbd2e2910538665abf42d0dd6204939d39f1ebeb7a6cac1f992c3ca`.
This evidence-only commit changes no kernel, reference, validation or production
source. Prior b8/c8/4a records are retained separately without overwrite.

| Case | Baseline ms | V8 ms | baseline/V8 | Numeric | Hard guards |
| --- | ---: | ---: | ---: | --- | --- |
| ragged B8 |0.01147211529314518|0.013063942082226276|0.8781511140311311|false|all true|
|4K/B64|0.5275581479072571|0.7300740480422974|0.7226090960525317|true|all true|
|16K/B32|1.0312336683273315|1.5115258693695068|0.6822467873192826|true|all true|

Each row is **one paired round, baseline first then V8**, with graph_time's
internal warmup/replay/event medians. It is not an ABBA service pair or a formal
competition throughput score. The timed call includes the token writer (V
quantization for candidate), producer and original reducer. Initial whole-cache
encoding, CPU references and artifact saving are outside the measured graph.
All cases completed, errors=[], hard_checks_passed=true. The finite candidate
precision failure was explicitly retained by numeric-diagnostic; it was not
turned into a passed gate. The independent baseline numeric guard passed both
patterns for every case.

Ragged pattern0 reproduces the original one pointwise violation at
`[seq2,head7,dim99]`, length15: actual0.875 versus canonical V8 FP64
0.864741108604122, error0.01025889139587799, tolerance0.00932370554302061.
Its max_abs0.010334571823477745 is at `[1,8,26]`, a different coordinate.
Pattern1 adds two violations, `[2,7,99]` and `[3,1,34]`; max_abs is
0.019193127751350403. Complete coordinate JSON and numeric-evidence PT now exist,
including full failed rows with original/overlaid BF16 inputs and independently
validated canonical/actual codes/scales. No cause is inferred from aggregate
max_abs or from an explanatory FP64 weighted-P model alone.

Actual resources: BF16 baseline93regs/0spill/8192B shared; V8
143regs/0spill/4096B shared. Ragged use_sme is15/79, steady is14/78.
These resource fields are observations, not an independent explanation of time.
The separate PR75 scheduling candidate is an independent, still-unaccepted
hypothesis; its resources or source order do not reverse this result.

[result.json](evidence/v8-bf16p-20261027/numeric-all-v5/result.json) is an exact
copy of the raw result, and [summary.json](evidence/v8-bf16p-20261027/numeric-all-v5/summary.json)
is a derived reading aid. All42 raw files/15428324B, including PT inputs,
outputs/oracles, coordinates, compiler IR/ASM/cubin, log and owner, are preserved
byte-for-byte in six tar.gz parts, each below2MiB. The original download manifest
is preserved as an additional archive member. See
[archive-manifest.json](evidence/v8-bf16p-20261027/numeric-all-v5/archive-manifest.json)
for original per-file SHA256 and each part's SHA/member list. Archive roundtrip
was checked against every original byte. Extract all parts into one new empty
directory; members retain their original `out/` and `exports/` paths.

No additional SSH, JIT, GPU, model run or service operation was performed while
archiving and reviewing these local owner artifacts.
