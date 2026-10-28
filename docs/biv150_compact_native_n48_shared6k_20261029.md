# Private N48/S4 shared-stage6144 candidate

This is a CPU-only source/address experiment based on shared112
`e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8`. It does not modify112,111,
production, the accepted2ad library, cache format, host shape qualification,
RN32/RNE codec, math, wait placement or compiler flags. Its separate namespace
is `cjj_native_k8_compact_n48_shared6k_probe`; there is no service activation.
Actual compile, resource, numerical, Graph, performance and model qualification
are all pending. The existing112 fullstep numbers cannot qualify this candidate.

## Shared address argument

|region|old source byte range|candidate source byte range|
|---|---:|---:|
|packed K I8|0…2047|0…2047|
|unused middle|2048…4095|none|
|V BF16|4096…8191|2048…6143|

`issue_k_i8` has512 aligned uint32 stores. `k_i8_index` is a bijection over2048
bytes; every actual I8 TCU K read also lies in words0…511. The shared probability
fragments are registers, not a further stage allocation. The private fork makes
exactly four CUDA edits plus private symbol renames:

- V SME destinations4096/5120/6144/7168 become2048/3072/4096/5120;
- V tail clear's base word1024 becomes512;
- PV TCU base `stage+1024` becomes `stage+512`;
- the aligned16 array has1536 uint32 words instead of2048.

Global compact page12544, global V offset4352, FULL F32 scale base and its
internal+1024 element offset remain unchanged. Reversing the four declared
edits and private names recovers **every original source byte**. The C++ and
codec differ only in private symbols. The source factory rejects a partial
relocation, a changed global scale offset or an added legacy K-tail call.

The complete CPU model enumerates all K stores, actual SDK I8 K-read word
addresses, PV BF16 word addresses, every slot/dimension and valid count0…16.
Each V word is visited once by the tail clearer; odd cuts preserve the valid
low BF16 half. Its payload test includes zeros, signed zeros, infinities and
NaN bit patterns without floating conversion. All17408 PV word comparisons
match the old layout. K is unchanged after all V writes/clears. This is an
address/bit-image proof conditional on the same SME tile mapping, not device
numerical validation.

## Actual SDK and alignment limits

The local actual SDK header `IX/crt-ixsdk/iluvatar_mma.h` SHA
`36a0b4f3e5d518780f4549f7a42c9938c982b493c26f0696daa22091c635e17d`
defines the lane offsets. Its actual implementation `.hpp` SHA
`ddb5cd4a1d53aa2cacad501a1b2d138715c045bbf837a162d6c20a144ca45094`
implements the relevant `FragmentRow`/`FragmentCol` TCU loads as uint32 pointer
arithmetic relative to the supplied base. For PV, Nloop8/Kloop1 and BF16 give
relative words `n*128 + (lane xor ((n&1)*32)) + component*64`.
Thus all1024 PV words occupy4096 bytes and translate uniformly by−2048.
The4-byte C++ accesses remain aligned, and the source stage alignment16 is
unchanged. No proprietary SDK headers are copied into this branch.

The existing actual2ad compile archive is rehashed using its original strict
compile/SO contract. Both its standalone LLVM and SO-extracted LLVM declare
`[2048 x i32]` shared stage aligned16. The distinction between those artifacts
is retained. Old LLVM shows the four SME destination offsets above; the new
offsets preserve all low10 destination bits for the **same** stage base.
They retain relative1024 alignment, including the old non-4096 panel offsets.
This is source/old-artifact evidence, not a complete documented hardware SME
alignment contract. No header here specifies all restrictions on a new2048 V
base, and the future shared allocator may relocate the stage. Actual new
compile/IR/ISA, shared metadata, numerical/tail/Graph/side checks are necessary.

The source array is25% smaller. Neither physical shared allocation granularity,
per-SM resource limits, register allocation, occupancy, nor a throughput benefit
is established. A compiler may still allocate8192 bytes. No claim of improved
capacity or service throughput follows.

## Legacy BF16 helper hazard

`k_half_index` and `clear_k_tail` are retained solely for exact source inverse.
They are **not compatible with this packed K/V stage if called**: their old
BF16 K writes can extend to byte4095 and overlap the new V region. In the
unchanged reachable source path, K is handled only by `issue_k_i8`; the legacy
clear helper occurs only at its definition and its index helper only in that
dead body. The source factory makes this unreachable condition mandatory.
Neither helper appears in the old actual generated LLVM. Fresh candidate IR
must independently retain that absence. A future call requires redesign and
cannot be accepted under this source contract.

## CPU replay and a future compile-only entry

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/biv150_native_k8_compact_n48_shared6k_factory.py
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/check_biv150_native_k8_compact_n48_shared6k_cpu.py --out /private/tmp/shared6k-cpu-NEW.json
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/compile_biv150_native_k8_compact_n48_shared6k.py
```

For the checked local SDK/old artifacts, append the following arguments to the
CPU checker: `--sdk-header /private/tmp/cjj-corex-native-qk-header-definitions-20261028/IX/crt-ixsdk/iluvatar_mma.h --sdk-implementation /private/tmp/cjj-corex-iluvatar-mma-implementation-20261028.hpp --old-compile-out /private/tmp/cjj-native112-n48-compile-v1-raw-20261029/out`.
Both SDK files must match the exact SHAs. Default execution imports no Torch,
Triton or NumPy. The compile frontend is byte-inverse to the old entry except
factory import, private load name and the requested-experiment label.

Only Root may schedule a later explicit zero-launch compile:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/compile_biv150_native_k8_compact_n48_shared6k.py \
  --compile-only --root-owned-exclusive \
  --out /workspace/experiments/minicpm-s2/cjj-native-shared6k-compile-v1-20261029
```

No compile or device work was performed for this candidate. A successful compile
alone would not approve hardware alignment, numerical equivalence or resources.
All future results must use the new namespace, actual library and artifacts.
