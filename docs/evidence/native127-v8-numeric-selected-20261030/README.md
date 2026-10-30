# PR127 packed V8: selected numerical component result

Root independently replayed the complete saved-array numerical screen for
sourcecd2f4ab846b79e4ab32a545d8c276bd802ed66d5. All190originals/175022287B were
SHA-verified. Exact identities, counts and limits are in [results.json](results.json).
Only these selected texts are shared here: arrays, binaries, logs and the
full113107B independent audit stay private.

The actual SO138cac22 packed-V8 path matches the Native117 SO5898a00a control
whose V is explicitly BF16_RNE(F32(I8code*F32scale)). K/Q codec and original
QK/softmax/BF16-P/PV mathematics are unchanged. N64/S2, both Q pitches2048/2560,
389physical16KiB pages, mixed empty/short rows and lengths6143/6144 were tested.
This is compact S2 consumer scratch; width16/fullstep and a packed V writer
were not tested. V8 currently changes neither physical allocation nor capacity.

The full audit passed40snapshots,88FP64 hard comparisons,80cross-arm guarded
O/M/L/final bitwise comparisons,160readonly comparisons and48CPU-oracle array
reconstructions. Changed-input GraphB was checked before a fresh eagerB;
sideC and its complete snapshots also passed. Stream use is source-bound plus
snapshot evidence, without an independent stream timeline.

All16original-BF16 Q/K/V proxy checks passed the unchanged.005 gates; maximum
relative RMSE was0.004840202200510765. Worst absolute proxy error was0.00390625
on a length1row (pointwise threshold ratio0.6756756756756757). This is a
component proxy result, not model105 precision or service acceptance.
The24ideal-partial diagnostics still failed and remain visible: the hard
partial reference uses the original pagewise BF16-P operand, while ideal
partial does not. No numerical gate or rounding convention was relaxed.

Static actual-ELF resources remain candidate134VGPR/66SGPR vs reference85/70,
8192B shared and zero reported vector/scalar spills for both. Static counts
are not speed evidence. Root issued a separately hashed same-scope timing
receipt, requiring review of the new timing source and an explicit serial
launch. No timing, writer, model, service or capacity benefit is claimed here.
