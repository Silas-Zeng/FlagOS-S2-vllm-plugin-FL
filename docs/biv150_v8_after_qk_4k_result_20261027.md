# PR75 after-QK 4K diagnostic: evidence retained, no baseline gain

The owner-run steady4K/B64 complete writer+producer+reducer diagnostic finished
with numeric_passed=true and all original hard guards true, but main BF16/late
V8 was **0.6228189806905853**. This is one baseline-first paired diagnostic,
not a service, model105, TTFT or capacity result. The current late-read variant
has no demonstrated benefit over main BF16 and is retained as negative evidence.
PR75 stays Draft/open pending root's independent archive replay before closure.

The tested source is ac6f7c2348a4e05b468cbed03635f36ba8c43917. Kernel SHA256
remains b562d2cdbabf0dba9b7e495fd9b6a48b7dc70f7ca64220fe441efcbdd3cb66c8.
This evidence update changes no production, kernel, old tool, numeric threshold
or previous artifact bytes. PR73 remains closed and PR78 is separate.

## Exact owner evidence

The original controller download manifests recorded terminal state Z with the
same start identity: compile owner622278/start341714398, numeric
owner622409/start342361705. These are retained historical observations; this
archive work did not contact the device or restart/rerun either owner.

- Compile raw:21files/1422848B, including the empty original compile log.
- Numeric raw:17files/6385974B, including logs, result JSON, inputs, both numeric
  evidence PT files and coordinates, runtime IR/ASM/cubin and mechanism contract.
- Both original download manifests and root's independent actual-IR replay are
  also preserved:41 files in total, split into eight tar.gz parts, each below
  2MiB. Large PT files use byte chunks rejoined to their original SHA/length.

[Archive manifest](evidence/v8-late-read-20261027/after-qk4k-v1/archive-manifest.json)
records every original file/chunk and every tar SHA256. Root download metadata,
remote paths, owner commands and original compile proof are retained verbatim.
Compile proof SHA256 is
75b8057bda19bfe8d979ad788586980441aa08068f42f61fcb200d729b3a77aa.

## Mechanism and resources

Actual 4K ASM and LLVM replay confirms the separate after-QK contract, while
the original strict after-all-softmax contract **remains false**. Both strict
reasons remain: packed V expansion/conversion begins before softmax finishes.
The source movement did not achieve the original strict scheduling target.

Full-path ASM last QK MMA is line555; scale557, packed V loads572–839,
expansion590–932, casts608–965, first PV1040. Tail last QK1429; V loads1576–1715,
scale1732, expansion1738–1828, casts1833–1896, first PV1979. Every full/tail V
operation is after last QK and before PV; original BF16 MMA, shared layouts,
store/barrier signatures and actual packed/narrowed descriptors agree.
These are static template locations and counts, not runtime duration shares.

| Compiled path | Registers | Spill | Shared bytes | use_sme |
| --- | ---: | ---: | ---: | ---: |
| main BF16 | 93 | 0 | 8192 | 14 |
| frozen4a V8 | 143 | 0 | 4096 | 78 |
| late V8 | 137 | 0 | 4096 | 78 |

The six-register decline is real metadata evidence, not a speed inference.
Frozen V8 was compiled for the mechanism comparison; this run did not time it.
Older PR73 ratios cannot be divided into this result to infer late/old speed.

## Numerical and timing scope

Both changed-input patterns passed the unchanged codec, target writer,
slot mapping, unmapped-byte, finite, baseline CPU-numeric and Graph/eager
guards. Independent FP64 pointwise checks sampled rows0/1/63; relative-RMS
consumption covered all64 rows. Codec coverage was sampled pages plus all
writer targets,2144 vectors at initial/final checks. This is the original
steady-range coverage, not full model precision; old PR73 ragged failures are
not superseded by this passing 4K case. Precision_gate and official_score stay
false throughout.

The single complete-cost pair measured BF16 baseline0.5277456641197205ms and
late V8 0.847350001335144ms, ratio0.6228189806905853. Attention-only timing was
disabled. The benchmark retains both caches; each physical allocation is
268435456B and remains16KiB/page. Payload reduction is not capacity recovery.
No16K, model105, TTFT or service acceptance is asserted for this variant.

## CPU archive replay

From this worktree root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_after_qk_evidence_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_after_qk_evidence_cpu.py --extract /tmp/NEW_PR75_RAW
```

The checker validates every archived byte, both download manifests and owner
identities, replays the original ac6 strict/after-QK inspectors against actual
IR with temporary local path rebinding, checks full proof equality, and links
numeric runtime IR back to the compiled hashes/options/resources. It checks
recorded gates, coverage and timing arithmetic without loading PT via Torch,
recomputing FP64 or running a target kernel. Steady inputs retain the original
sampled-page coverage, so this is not an independent numerical recomputation.

[archive-check.json](evidence/v8-late-read-20261027/after-qk4k-v1/archive-check.json)
is reproducible stdout. Extraction was compared byte-for-byte against both
download directories, including their manifests. This archival work used only
local CPU and GitHub; no SSH, target JIT, GPU, queue or runtime changes.
