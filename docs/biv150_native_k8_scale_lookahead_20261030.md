# Private NativeK8 next-page scale lookahead

Based on main `5f0086be50f1d6d115c1ebb25b9d3c7250470372` and the exact
source of accepted SO `5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b`.
This is an uncompiled component prototype with no production hook.

5898 prefetches eight packed next-page K words per thread after the current
V-ready barrier, before current PV. Its actual full-loop SASS loads the current
K scale after the first QK MMA and waits for that load before score scaling;
the load already overlaps the remaining QK MMAs. Its actual stall cost is
unknown. This candidate carries the next complete page's one F32 K scale per
lane beside the existing K lookahead. It does not claim a performance gain.

`biv150_native_k8_scale_lookahead_factory.py` applies seven bounded source edits
and an independent symbol rename. Every generated file has an exact inverse
to pinned main5f bytes. C++ host guards, public argument order/types, codec,
QK/scale multiplication order, softmax, BF16 P, PV, reducer, 16KiB page and
8KiB shared layout remain unchanged. Both N64/S2 and N32/S4 are instantiated;
neither inherits device qualification from 5898.

- The first scale is loaded only inside the existing first-full-page guard.
- `KReady=true` full pages read a private per-thread carry. Next scale load is
  inside the original `has_next` block, using that block's returned page ID,
  after V readiness beside the eight next-K loads. The carry is updated after
  current PV. There is no new shared-memory slot or explicit barrier.
- `KReady=false` tail retains the original masked scale expression and exact
  original call. Its default null carry is never dereferenced. Empty segments
  still return before any cache access. The last complete page never fetches
  the tail, next segment or an extra page.

CPU checks validate the finite inverse, tail call, source ordering, all
lane/head scale addresses including maximum I32 page ID, bounded S2/S4
segments and short/empty/full/tail boundaries. Stale-carry, last-page read and
six source mutations are rejected. This address model is not a device
execution, synchronization, numerical or compiler-scheduling proof.

```sh
python3 -B -W error -S tools/check_biv150_native_k8_scale_lookahead_cpu.py
python3 -B -W error -S tools/compile_biv150_native_k8_scale_lookahead.py
```

The default frontend emits only a stdlib CPU plan. The finite compiler adapter
reuses 117's compiler flags, clean-source check, standalone diagnostics, final
SO IR/SASS export and before/after CUDA-context checks. Explicit compile-only
also requires Root's exclusive-schedule acknowledgement, an exact full source
commit and a fresh output directory outside the checkout. It does not launch
attention/codec/reducer or initialize input tensors. No remote launcher or
numeric/timing framework is added here.

After a separately authorized device compile, final SO/embedded ELF evidence
must establish the actual first/full/last/tail carry identity, no duplicate
current-page full-loop scale load, guarded next-scale scheduling after V-ready
and before PV, dependencies across the backedge, unchanged arithmetic and
actual resources. A new spill or scheduling cost must be reported, not inferred
away from source. All assembly review template approvals remain false.
Then reuse the original 117 consumer/fullstep numerical and timing rules;
this CPU draft grants no numerical, performance, model or service admission.

No SSH, GPU test, network access or publication occurred during preparation.
