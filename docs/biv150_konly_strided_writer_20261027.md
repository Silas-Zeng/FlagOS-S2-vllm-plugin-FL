# Private stride-aware K-only encode_append

Independent CPU-prepared component depending on PR79(codec/decoder), PR80(readers)
and PR82(lifecycle). Team base is actual main56f1838. Production vllm_fl/csrc,
the old79 branch and existing M2/Q32 chain are unchanged. No service hook,
installer, lifecycle activation, SDK upgrade or GPU work is included.

The private kernel reuses79's `quantize_bf16_rne` function byte-for-byte: fixed
RN32 FP32 scale/division, nearest/even code rounding, signedI8[-127,127]. Both
stdlib reference files and old141 finite-BF16 test vectors are unchanged copies
with complete archive/SHA provenance. This is reuse of the existing codec, not
a new quantization algorithm or proof of target compiler/FTZ equivalence.

`encode_append` accepts BF16 key/value sources independently shaped[rows,2,128],
inner strides(128,1), positive row pitch>=256. K/V pitches and row extents are
separateI64 arguments. Slot count must not exceed either source's rows; it is
not inferred from query rows. Explicit row extent guards precede the slot load;
negative slots return before either K/V load. Additional kernel positive-OOB
defence is not a successful admission of invalid data.

All three typed cache pointers refer to the **same cache-page0 address**:
I8 code offset`page*16384+slot*256+head*128+dim`, FP32 scale offset
`page*4096+1024+slot*2+head`, BF16 V offset
`page*8192+4096+slot*256+head*128+dim`. Do not pass an already-offset scale/V view.
The16KiB physical page and reserved4224..8191 remain unchanged. The old BF16
writer never touches live encoded K first. V is an unmodified BF16 load/store,
including source bit patterns for signed zero/subnormal/extreme/Inf/NaN probes.
Valid K retains the old finite-BF16 precondition; host metadata alone does not
inspect GPU numerical values or certify that precondition in a live service.

The stdlib descriptor contract rejects insufficient row/storage/slot extents,
wrong dtype/head/dim/stride, scalar overflow, unknown slot provenance, invalid
positive slots and duplicate writable slots. Repeated negative padding slots
are allowed. Inputs cannot overlap writable cache; slot storage cannot overlap
K/V. K and V may share read-only QKV backing with different offsets. Full
backing-range checks are deliberately conservative. Empty slot count has no
kernel launch. None of these checks reads a GPU scalar.

For this standalone component, an explicit CPU-owned pre-upload slot snapshot
is required. The future device entry itself constructs and byte-checks its GPU
slot tensor from that snapshot. An epoch number or caller-provided Boolean is
not proof of arbitrary live GPU-derived mapping freshness/uniqueness. GPU-only
unknown mappings, concurrent writers, stale graph/storage/format and actual
service ownership remain PR82 admission gaps; this component does not install
itself there or silently fall back to the old BF16 writer. Kernel bounds checks
do not detect duplicate writers and cannot replace the external contract.

Initial CPU cases compare a pitched-source address emulator with an independent
logical-field cache reconstruction. Nine geometries each use two changed
patterns: empty, extra K/V rows,15/16/17/18-slot tails, unequal256/384/640/2560
pitches, shared QKV rows, negative-only NaN-poisoned source rows and141 codec
vectors. Complete destination bytes include canaries/reserved/untouched slots
and original V bits. Initial21 rejections include malformed descriptors, alias,
unknown/OOB/duplicate slots and source mutations for shared pitch, scale/V double
offset and missing negative guard. These are CPU address/byte/source gates.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_strided_writer_cpu.py --out NEW_JSON
```

Python3.12.15/3.14 outputs match `cpu-v1.json`, SHA
`158f725aadafed31f0694bf6aef96acdf0cd371a81d76269c1b6dfbe6fadd33d`.
Old codec code/scale witness SHA is
`be6129bb54bb510053344d8c65a84b9e8289eec26870b5226f80fd2c0a979a4d`.
The upstream archive is5666B SHA
`278cc3be8228a1eff50a6af4a55d8e86564dff6b711850a4f9d3de1cfa5641ea`;
device helper SHA`d73af8368193e1cab4ba9e74132d056d5b61d0c0950d600d494d75c3151f4cf7`.

An explicit minimal device falsifier is being prepared separately. It will use
only the existing M1/head geometry, num_warps1/stages1, with no tile scan/timing
score. Actual typed ABI/I64 lowering, exact codec/cache/source/canary checks,
changed-input Graph and side stream, compiled source/IR/ASM/cubin and loaded
resources are still required. There is no device result, model precision,
throughput or TTFT acceptance. Root alone schedules any later device execution.

Initial normal fetch timed out; authenticated API confirmed actual main56f and
the matching local object before creating this independent branch. No successful
fetch or upstream change is claimed from that failed attempt.
