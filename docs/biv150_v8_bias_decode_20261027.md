# A/bias V8 decode primitive: CPU source contract only

Fresh `git fetch origin main` succeeded. Local main, origin/main and live remote
ref agreed at `56f1838192abc940b3fb9accef490022ac125000` (team PR66 merged).
This independent branch/worktree does not reuse closed PR73 or modify PR75.
Its reference is the exact frozen4a V8 kernel, SHA256
`a7fc539dccbd2e2910538665abf42d0dd6204939d39f1ebeb7a6cac1f992c3ca`, retained as
[an inert text fixture](../tools/fixtures/biv150_v8_bf16p_4a.py.txt). No production
source changes, serving registration, GPU runner, runtime installation, SSH,
JIT or device work is included.

## Scope and exact arithmetic

Only factory A is implemented. It inserts the following V-only SSA reassignment
after running_sum and immediately before the original PV in both full/tail paths:

```python
V = (((V.to(tl.uint8, bitcast=True).to(tl.uint32) ^ 128) | 0x4B000000).to(tl.float32, bitcast=True) - 8388736.0)
```

For raw unsigned byte b, u=b xor128 lies in[0,255]. IEEE FP32 bits
`0x4B000000|u` denote exactly2^23+u. Subtracting2^23+128 yields u−128, the exact
signed-byte integer, including−128,0,127. Both the operands and result are exact
FP32 values; each signed I8 integer is also exactly representable in BF16.
This does not widen the codec's valid range: its original[-127,127] gate remains.
The masked tail's original other=0 becomes positive FP32/BF16 zero.

The original PV expression stays verbatim:

```python
tl.dot((probability * value_scale[None, :]).to(tl.bfloat16), V.to(tl.bfloat16))
```

No scale is multiplied into wide V. QK, P, PAGE16, softmax, segmentation,
natural-log handoff/reducer, load masks, cache addresses, encoder/writer,
codes/scales and ABI are unchanged. Removing exactly the two inserted blocks
restores every original byte and the complete module AST. The immutable fixture
cannot be silently replaced by an unpinned template. Current-main BF16 GQA source
still hashes to2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1.

The payload remains I8 plus FP32 per-token/head scales in16KiB/page allocation.
This is a register conversion experiment, not BF16 compressed cache, different
quantization, reclaimed capacity, or a precision recovery scheme. It does not
change the BF16(P*scale) approximation that failed the original ragged gate.

## B/LUT deferred; API and target limits

B would require an additional512B immutable BF16[256] reader parameter, exact
unsigned-byte mapping, positive-zero entry, device dtype/length/contiguity,
alias and lifetime/Graph checks. That is a new ABI and a separate full-consumer
contract; it is not implemented here. The A factory rejects every table argument
and the lookup variant before emission. These rejections are not a claim that
B's device table validation or immutable ownership has been implemented.

Upstream documents `bitcast=True` on [cast](https://triton-lang.org/main/python-api/generated/triton.language.cast.html)
and [tensor.to as its alias](https://triton-lang.org/main/python-api/generated/triton.language.tensor.html).
Those upstream API semantics do not establish support/lowering quality in the
installed Corex Triton fork. The uint8→uint32 extension, uint32/FP32 bitcast,
FP32 subtraction and FP32→BF16 conversion have not been compiled on the target.
Any unsupported API or unfamiliar lowering must be reported; do not substitute
unverified inline assembly or assume the desired code generation.

## CPU evidence and two device falsifiers still outstanding

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_decode_factory_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/biv150_v8_decode_factory.py
# Optional source output only; fails if the destination already exists:
PYTHONDONTWRITEBYTECODE=1 python3 tools/biv150_v8_decode_factory.py --emit NEW_SOURCE.py
```

[cpu.json](evidence/v8-bias-decode-20261027/cpu.json) records exhaustive256-byte
FP32/BF16 bit-pattern equality against an independent signed-byte oracle,
positive-zero mask behavior, exact source/ABI inverse,12,288 disjoint code-byte
addresses covering full slots/heads/dimensions at pages0,2^17,2^31−1,64-bit
address boundaries, and invalid source/byte/table/address rejection. Address
rejection belongs to this CPU contract; no new device bounds branch was added.
Torch/Triton are never imported and generated source is only syntax-compiled,
never executed by these tools.

The first future target question is whether actual LLVM/ASM avoids the old
per-path32 I8→I16 plus32 I16→BF16 instructions. A may require more integer/FP32
operations or be folded straight back to the old CVT sequence. Count the actual
replacement instructions, register/live ranges and options; different source or
pipeline choice is not evidence of speed. No fake dependence forces scheduling.

The second question is whether the original32 scalar b16 shared V stores per
path remain. This factory does not change store/layout source, but dtype changes
can affect inferred layout. Record actual convert_layout/shared/barrier/store
mapping; a changed store pattern requires its own correctness evidence. No paired
store optimization is bundled. The old kernel already had eight packed-word V
loads per path: uint32 arithmetic here does not establish fewer LDG operations.

The full writer+producer+reducer consumer is **not connected** in this Draft.
Any later wiring must retain all frozen4a baseline independent numerics, codec,
slot/unmapped bytes, changed-input Graph/eager, finite and original precision
gates; no new tolerance is introduced. Real-shape/narrowed descriptors and actual
hash/options/IR identity remain necessary before device timing. The final
comparison must be against current-main BF16 as well as the slow V8 reference:
PR73 measured only0.7226/0.6822 baseline/candidate on4K/16K. Beating that retired
V8 path alone would not establish a useful optimization. No new actual IR,
component throughput, service result, model105/TTFT or formal precision is claimed.
