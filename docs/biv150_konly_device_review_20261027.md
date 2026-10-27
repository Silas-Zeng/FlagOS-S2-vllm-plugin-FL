# Independent CPU review of five actual reader diagnostics

This evidence-only update changes no e256/a432/7db driver, kernel, codec, gate
or production file. All482 raw files (294,086,418B) and their five manifests
remain intact in Root's local downloads. The small fe1 review first shared every
original manifest/SHA and independently replayed reports. A separate subsequent
archive update now includes all original bytes in166 deterministic archives,
each<2MiB, under device-originals-v1. All487 reconstructed files, including the
five manifests, were compared directly byte-for-byte against Root's downloads.
No device or SSH work was performed here. Device frontends stay frozen at their
original e256/a432/7db commits; archive delivery is not a service dependency.

| Runtime | Actual scope | Owner/start, same-start Z | Raw files / bytes |
|---|---|---|---|
| e256 | Q2048, KV4096, prefix2048 |636781/345148988|95 /55,836,603|
| e256 | Q2048, KV16384, prefix14336 |637166/345166569|95 /101,394,796|
| a432 | M2 totalQ2048, Q=[2047,1]/[2046,2] |638590/345496241|80 /77,477,630|
| a432 | raw64×q1, short ragged KV |638859/345524964|105 /27,174,404|
|7db | raw64seq mixedq1/2, totalQ96 |639428/345607034|107 /32,202,985|

All completed/errors[]/implementation hard checks are true. CPU reconstruction
from the original BF16 inputs regenerated every K code/FP32 scale with RN32/RNE
and exact half-boundary repair, then matched expected and actual full16KiB/page
bytes. All full BF16 output arrays are finite. Actual source/typed ABI/options/
TTIR/TTGIR/LLVM and current ASM/native instruction/control audits replay exactly.
No I8-Q operand hoist is licensed for M2/raw3D by this evidence.

For M2/raw q1/q2, all twelve case/pattern records reproduce the saved three-way
FP64 metrics exactly. Output/workspace guard arrays and read-only before/after
digests independently match. All q1 nonempty workspaces are finite with positive
sums and inactive regions remain NaN. For q2, all8/24 causal-empty early-query
token/segment rows, across every head/channel and both arms/patterns, are exactly
M=0/L=0/O=0 as derived from actual59d9 helper behavior. Token-indexed scratch and
the original reducer preserve prefix mapping. The saved private host/proof SHA
and exact metadata inverse also replay. Pre-device CPU sidecar false labels
remain unchanged; the actual four hard-gated device records establish this
specific q2 implementation scope, not general raw3D coverage.

| Random-input original BF16 proxy RRMSE | Case0 | Case1 |
|---|---|---|
| M2 |0.008435738870|0.007482617415|
| raw q1 |0.008156464777|0.005062169950|
| raw q2 |0.008357659741|0.005207555377|

The unchanged0.005 original proxy gate remains false. Quantized-reference
implementation and BF16 baseline gates pass; this is not a formal model105
accuracy verdict. Closed references are independent FP64 global softmax with
INT64 quantized QK, not a model of BF16 probability/online-rounding. This CPU
review recomputes metrics against the saved references; it does not regenerate
global softmax a second time.

The actual e256 control imports clean DAC1 and selects the main Manager entry
with its BF16 lowered proof. CPU review confirms two selected calls, zero
fallbacks, actual hash/options/override SHA and32 actual Q shared reads outside
the KV loop. Full4,194,304-element outputs per pattern match main/raw BF16 and
I8 hoisted/unhoisted bitwise. The raw BF16 semantic control is not substituted
for the accepted-main timing control.

| Single warm AB/BA64 observation | Main mean ms | I8 mean ms | Main/I8 |
|---|---|---|---|
| KV4096 |1.449863493443|1.431780695915|1.012629586066|
| KV16384 |6.556466102600|6.185547828674|1.059965306906|

Each is the only recorded round. Cold whole-cache encoding, incremental writer,
SDK dispatch, startup and compilation are excluded; writer_ms/fullcost_ms=null.
Geometry/cache allocations and logical payload are logged, with no cache flush.
Resident repeated fixtures do not establish HBM, stable speed or service gain.
The perf artifacts save four eager arrays; Graph/post-timing equality, canary
and read-only checks are recorded frozen-runtime guards, without separate
Graph-mode arrays in these two packages. The earlier c7 Graph evidence has its
own saved full modes and should not be conflated with these perf packages.

Loaded observations: accepted-main113regs/0spill/65536shared; raw BF16 control
101/0/98304; I8 unhoisted135/0/57344; I8 Q32-hoisted122/0/33792. Raw3D BF16/I8
are96/121regs,0 spills,12416/8448 shared, original reducer22/0/1024. Resources
are observations, not evidence of performance or a new zero-spill requirement.
All runs retain precision_gate=false, performance_verified=false, no service
acceptance and capacity_reclaimed=false.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_allreader_device_evidence.py \
  --profile multi2 --raw ORIGINAL_MULTI2_DOWNLOAD
# --profile raw3d or rawq2 for the respective complete downloads
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_attention_device_evidence.py \
  --raw ORIGINAL_ATTENTION_4K_OR_16K_DOWNLOAD
# Complete shared raw-byte reconstruction, optionally all CPU array audits:
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_device_archives.py \
  --arrays --extract FRESH_EXTRACT
```

These CPU checkers use installed NumPy and the restricted CPU-storage reader,
never Torch or an arbitrary pickle loader. Reports, all manifest hashes and
original local paths are in docs/evidence/konly-prefill-reader-20261027/
device-review-cpu-v1/index.json. Its pending-upload labels describe the earlier
fe1 review snapshot; complete payloads are indexed independently in
device-originals-v1/index.json. No original failed gate or raw file is rewritten.
