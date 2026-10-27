# PR78 A/bias byte-stage: equivalent to old V8, slower in one complete-cost round

The owner-run steady4K/B64 diagnostic completed successfully, including both
old/A numeric phases, all original hard guards and both patterns' four-field
bitwise checks. Its only complete writer+producer+reducer timing round measured:

| Path | Milliseconds |
| --- | ---: |
| main BF16 | 0.5275394320487976 |
| frozen V8 | 0.7301254868507385 |
| A/bias byte-stage | 0.8034855723381042 |

Main/A is **0.6565636648753795**; old-V8/A is **0.9086976941304727**.
A did not beat either control in this fixed-order baseline-first triplet.
There is no measured benefit that supports continuing this A variant. PR78
stays Draft/open for root's independent archive replay before closure.
This single round is a component diagnostic, not a service throughput estimate
or a statistical conclusion about every input. Old numeric-phase timing was
disabled, and attention-only timing remains null/absent as appropriate.

## Three source identities, preserved separately

The actual operator/baseline/helpers came from the original sourceRoot at
`a6980ddf1984422b93438d53429bddf36434ccde`. The separate tested validator was
`11ed97dc0800f195bb98688a43ae670cc1ab569b`. This new evidence-only commit is
neither of those tested roots. Original old/A modules kept their original
out/sources/debug paths; the generated A SHA remains
`dba93c46e32f1761c15df73494185122e8784ecaa769ff2c5b7e078b3dd24fbd`.
No factory, driver, gate, kernel or production bytes change in this archive.

The original compile proof remains SHA
`d27751146b16823166a2ed8f080c83255a7434b739fcc6c618ab58374cac93f2`,
completed=false/errors retained/v1 mechanism=false. The independently named
v2 byte-stage contract is true only for the audited steady4K original files.
The existing67-file archive and all60 compiled files remain untouched, and
the new checker replays their original static contracts again.

Runtime GQA exports match the old compiled artifacts: BF16 main93 registers/
8192B shared, frozen V8 143/4096B, A124/6144B; all zero spill. The A pipeline
removed numeric I8→float conversions and moved the layout conversion to raw
byte staging, but fewer registers did not imply a complete-cost gain. No new
profile attributes the slowdown to a particular instruction or memory stage.

## Actual admission and numeric evidence

Root recorded owner623978/start342792903 as same-start terminal Z. Driver host
PID was1410043, distinct from the container PID. Actual native observations
were empty→[1410043] after the tiny initialization→the same singleton before
old→the same singleton before A. The receipt is completed/errors=[] and
records the one tiny CUDA setup outside every numeric/timing path. These are
historical owner observations, not queries performed by this CPU archive job.

Both phases completed two changed-input patterns with numeric_passed=true,
all six hard guards true, writer/codec/unmapped-byte checks and Graph/eager
coherence. Independent FP64 checks sampled rows0/1/63 and relative-RMS
consumption covered64 rows. Both complete CPU-input sets and logical narrowed
scratch/max/sum/output bytes matched between old/A. Corresponding complete PT
files are also byte-identical. This establishes the recorded case equivalence;
it does not repair the old4a ragged failure or establish general precision.

Precision_gate=false and official_score=false remain unchanged. A16K,
model105 and TTFT were not tested; no service integration was exercised.
Each KV allocation remains268435456B,16KiB/page; capacity was not recovered.

## Lossless local CPU archive

All33 main raw files/18111262B, two auxiliary originals (the actual sidecar and
outside admission receipt), and both download manifests are retained:37 files,
18134789 original bytes. The seven tar.gz parts total5552084B; the largest is
1122764B, below2MiB. Identical byte chunks are stored once and reassembled for
every original path; no PT interpretation or numerical transformation occurs.

[archive-manifest.json](evidence/v8-bias-decode-20261027/numeric-steady4k-v2/archive-manifest.json)
contains every original file/chunk/part SHA and length. Empty original logs,
all JSON, PT inputs/outputs/oracles, runtime IR/ASM/cubin and both original
auxiliary files are preserved verbatim. Sidecar and outside admission match
their embedded runtime-binding copies, with exact sidecar SHA linkage.

From the worktree root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_bias_result_archive_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_bias_result_archive_cpu.py --extract /tmp/NEW_PR78_RAW
```

The checker verifies all bytes and identities, replays the old60-file contract,
matches16 runtime GQA files to compile evidence, checks recorded numeric and
bitwise gates, admission transitions and exact three-way timing arithmetic.
The other roles retain their original pre-launch six-file checks in the frozen
validator; they do not have separate post-run exports in this33-file dataset.
PT remains opaque: the checker never imports Torch/Triton or calls torch.load,
recomputes an FP64 oracle, compiles a target kernel or accesses a remote host.

[archive-check.json](evidence/v8-bias-decode-20261027/numeric-steady4k-v2/archive-check.json)
is reproducible stdout. Extraction was compared byte-for-byte to both root
download directories. Further directions must first be deduplicated against
the existing4a/PV and P8/V8/native-PV sources; no new quantization branch is
implemented here. No SSH, GPU, JIT, queue, PR closure or merge occurred during
this evidence update.
