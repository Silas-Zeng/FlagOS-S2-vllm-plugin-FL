# PR127: isolated packed-V8 numerical screen (device pending)

This child prepares an executable consumer, with no writer, encoder, timing,
service hook or capacity qualification. It normally merges main5f0086be after
shared compile sourcefcf11f11. Main5f only changes gather here; the actual CUDA
compile remains the immutable fcf source and SO138cac22. The control is the
actual Native117 SO5898a00a, not a newly compiled or renamed library. Neither
library, original CUDA file, original3eb fixture nor production file changes.

Root independently replayed static report
`1c9bf67eaa92a25865c2ffdf62270f8bb91c40391e959f541ec86560c7e0f352`.
This exact static pin is now bound, following Root's later authorization; its
numeric/performance/service labels remain false. The original pending-None
state still refuses in the CPU fault test. No automatic slot or device run
follows from a static report. Root must first read this numerical child and
explicitly confirm a serial slot; `--gpu --root-owned-exclusive` is required.

The old3eb/117 consumer run function is retained through a finite inverse
adapter: substitute the two exact library/namespace identities, the private
V view, extra CPU-oracle blobs and actual view metadata. Numerical checks,
reducer, poison/snapshot, capture/Graph-before-fresh, side-stream and readonly
logic are unchanged. Two N64/S2 cases use Q pitches2048/2560. The scratch is
the original consumer's compact S2 layout; this does not qualify width16
service/fullstep scratch. Each A/B/C phase has short boundary lengths,
zero-length rows, one6143 row and one6144 row. A unique tail page's invalid
slot15 has I8/NaN poisons; no other request references that physical page.
All other unused table entries are-1. Shared pages are readonly throughout.

CPU V preparation reuses the original rational RN32/RNE Q/K codec, per
128-D token/head vector (zero→scale1, clamp to2^-126, codes[-127,127]). It is
not a new GPU writer. The private16KiB page stores Kcode[0,4096),
Kscale[4096,4224), Vcode[8192,12288), Vscale[12288,12416). The control instead
stores `BF16_RNE(F32(signedI8code * F32scale))` as original BF16 V at+8192.
The multiplication is explicitly rounded to FP32 before BF16 RNE. K/Q inputs
and codec are shared. Vscale uses full-base FP32 page stride4096; Vcode uses
full-byte page stride16384 with+8192 applied once. No compact layout or
physical capacity recovery is claimed.

The independent3eb ideal final/M/L and pagewise BF16-P partial references
are reused, with only the page-table decoder extent changed16→384. The
original pointwise `.005+.005*abs(reference)` and relative-RMSE`.005` gates
stay hard. The partial reference models each page's BF16 P; the ideal final
is still a different reference. Entire guarded O/M/L/final arrays must also
match control/candidate bitwise at all ten saved stages per pitch. Each arm
separately checks every input byte, full cache, output guards, empty-zero
output, inactive scratch poison, GraphB before fresh eagerB and sideC.

The original BF16 proxy uses the original unquantized BF16 Q/K/**V**, rather
than silently replacing its V with dequantized data. It is retained as the
original separate diagnostic; `precision_gate`, model and service remain
false even if the implementation screen passes. Original BF16 K/V bits,
private code/scale cache, control dequantized cache, complete CPU inputs,
full golden/output/scratch arrays and before/after snapshots are saved as
content-addressed blobs. A dequantV/original-QK diagnostic is separate too.
This can establish operand implementation; it does not establish model
accuracy or writer cost. No timing option exists.

CPU replay (stdlib; no framework or GPU):

```sh
python3 -B -S tools/check_biv150_native127_v8_numeric_cpu.py \
  --parent-source /private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028 \
  --actual-raw /private/tmp/cjj-native127-packed-v8-compile-raw-20261030 \
  --static-review /private/tmp/cjj-root-native127-actual-so-review-20261030.json \
  --control-library /private/tmp/cjj-native117-klookahead-compile-v1-raw-20261029/out/compile/build/cjj_native_k8_klookahead_probe.so \
  --codec-summary /private/tmp/cjj-native-k8-codec-v1-raw-20261028/out/codec-summary.json
python3 -B -S tools/probe_biv150_native127_v8_numeric.py
```

The CPU admission exercise uses a synthetic true scheduling flag solely to
validate file binding/rejection; it grants no real device permission. The
separate one-phase NumPy smoke decoded every nonpoison V operand bit and
ran the original CPU reference over6143/6144; it is not GPU evidence.

After Root review and an actual closed predecessor, the worker argv is:

```sh
python3 -B tools/probe_biv150_native127_v8_numeric.py --gpu --root-owned-exclusive \
  --parent-source "$FROZEN_3EB_ROOT" --compile-summary "$PACKED127_COMPILE_SUMMARY" \
  --static-review "$ROOT127_STATIC_REPORT" \
  --static-review-sha256 1c9bf67eaa92a25865c2ffdf62270f8bb91c40391e959f541ec86560c7e0f352 \
  --control-library "$NATIVE117_5898_SO" --library "$PACKED127_138CAC_SO" \
  --codec-summary "$ORIGINAL_CODEC_SUMMARY" \
  --codec-proof-sha256 3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263 \
  --out "$FRESH_COMPONENT_OUTPUT"
```

These variables require Root's real paths; this document does not assert
that a new remote checkout/proof/output already exists. No SSH, GPU, JIT,
network or GitHub action was performed in preparing this child.
