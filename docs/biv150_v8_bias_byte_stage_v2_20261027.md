# Exact steady4K byte-stage v2: static contract, no numeric or speed result

Owner622613/start342383793 completed the a698 compile-only job with zero
kernel launches. The original proof SHA256 is
`d27751146b16823166a2ed8f080c83255a7434b739fcc6c618ab58374cac93f2`.
Its completed=false, original errors and mechanism_passed=false remain
unchanged. The v1 gate stopped on exactly four staging/layout differences,
which are unknown to that contract, not a mathematical or performance failure.

This independent v2 accepts only this exact original steady4K proof and all
ten roles'60 actual compiler files. No other case, new compiler output,
substituted descriptor, unknown resource or additional v1 rejection is licensed.
Original67 raw files/2141528B, download manifest, logs, source, IR/ASM/cubin and
Python caches are preserved verbatim in a309350B archive with SHA metadata in
`docs/evidence/v8-bias-decode-20261027/byte-stage-v2/`.

## Reviewed actual chain

Full TTGIR203–205 loads raw i8 V and converts its layout through i8 shared
memory before any numeric decoding. Lines234–239 and247–249 then follow
unsigned extension, xor128, OR0x4b000000, FP32 bitcast, subtract8388736, BF16
conversion and the original BF16 PV. Tail293–294 and325–340 repeat the chain
with the original masked-load zero. No numeric integer-to-float conversion
remains. Both paths retain32 bias operations/eight packed V loads; four typed
BF16 dots and32 BF16 MMA template instructions remain.

The CPU validator follows actual LLVM SSA through packed loads, scalar byte
stores, shared addresses, loads, bias operations, BF16 conversions and every
PV B fragment. All64 lanes write2048 unique bytes in[4096,6143]. Old BF16
store addresses obey `old_address = 2*(new_address-4096)`.
Full and tail each have2048/2048 matching old/A PV coordinates. Q/K BF16 stores
remain within[0,4095]; the byte bank is disjoint. The tail's conditional load
merges zeroinitializer, not poison. All256 signed-byte reconstructions are exact.

Actual LLVM barriers162/435/549/1339/1564 provide initial Q synchronization and
pre-store/post-store synchronization for each path. The separate byte bank
avoids the previous K→BF16-V shared-buffer reuse. ASM correspondingly changes
from64 b16 stores to64 byte stores plus64 byte loads, with64 xor/bias/conversion
chains, and barriers9→5. Q/K layouts and surviving shared operation counts are
checked independently. The compiler's OR→integer-add lowering is exact because
the low8 bits of0x4b000000 are zero. Exact ASM SHA binds the manually reviewed
instruction chain; this is not a general target-ISA simulator.

Frozen V8 has143 registers/4096B shared; A has124/6144B, both zero spill and
use_sme78. These resources are known and recorded; their change is not a speed
inference. Static tail verification does not supply actual ragged/16K numeric
coverage. The existing old ragged precision failure remains relevant.

## Two distinct roots and an untouched v1 proof

The current checkout supplies only the new validator/driver. Its committed
version, file hashes and absolute root are recorded separately in a new
sidecar. Its HEAD and actual validator bytes must agree and its tree be clean.

The original source root is fixed to
`/workspace/experiments/minicpm-s2/code/cjj-v8-bias-compile4k-v1-20261027`
and must still be clean at a6980ddf1984422b93438d53429bddf36434ccde with all
original baseline/helpers/factory/fixture bytes. Baseline and reducer are
loaded from that root. Both old/A modules continue to load the original
`/workspace/experiments/minicpm-s2/cjj-v8-bias-compile4k-v1-20261027/sources/`
files, preserving debug paths and actual specialization hashes.

The original v1 runner has three reversible host edits: explicit proof
selection, whole-job native-PID admission, and injection of that phase callback.
The frozen probe's single full-table busy assertion selects the native callback;
its original assertion is independently restored by the inverse proof. It receives
the unchanged failed original proof, never a fabricated successful v1 JSON.
The entire numeric/bitwise/three-way full-cost body is otherwise AST-identical.
An additional pre-launch wrapper requires all six runtime artifact hashes for
each role, then calls the unchanged original ABI/options/resources/IR guard.
A failed guard cannot cache a successful six-file check.

The two input patterns, four-field byte equivalence, independent baseline
numeric gate and every codec/writer/slot/unmapped/Graph/finite guard remain.
Only explicit numeric-diagnostic may retain finite candidate precision failure;
it cannot change numeric_passed/precision_gate or bypass hard/bitwise failures.
Full cost still compares current-main BF16, old V8 and A, in the original order.

## CPU replay and future owner commands

From the new validator worktree root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_bias_byte_stage_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_v8_bias_byte_stage.py
```

Default CPU replay validates archived actual IR; it explicitly does not claim
that the original remote root has been inspected live. The following commands
are for the owner after exclusive device use is established. They were not
executed or queued here. Use the same new clean validator checkout for both;
sidecar and output paths must be new and outside both source roots.

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/bench_biv150_v8_bias_byte_stage.py --make-sidecar --out /tmp/NEW_BYTE_STAGE_SIDECAR.json
PYTHONDONTWRITEBYTECODE=1 python tools/bench_biv150_v8_bias_byte_stage.py --run-gpu --numeric-diagnostic --case steady4k --sidecar /tmp/NEW_BYTE_STAGE_SIDECAR.json --out /tmp/NEW_BYTE_STAGE_NUMERIC
```

Sidecar creation uses only CPU and checks the live old source/out paths; it
does not compile or launch. GPU mode rechecks both roots, the sidecar and all
original artifacts before importing baseline/helpers from the old root.
Actual warmup must independently match the proven specialization before its
first launch. `byte-stage-runtime-binding.json` records both versions alongside
the unchanged-source result and the original failed proof SHA.
Existing outputs are refused before root/device access; Python bytecode writes
are disabled before old-root imports so the original cache files also remain.

No target JIT, GPU, SSH, queue or new numeric/throughput/service result was
performed in this CPU implementation. Future explicit GPU runs do one recorded
tiny CUDA initialization for admission, outside all numerics and timing; they
do not claim zero GPU work. See the native-PID admission note. No capacity
claim is made. Physical cache remains16KiB/page.
