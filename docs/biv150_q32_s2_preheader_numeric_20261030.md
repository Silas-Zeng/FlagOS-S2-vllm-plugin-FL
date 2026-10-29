# N1 Q32 S1/S2 numerical-only successor

This is a tools-only child of `2e0929793d9e744a964c719e27d253927bcb52f3`, with production byte-identical to main `899ce77470b0b539aebc108fd122f668c2d5894d`. No device, numerical, timing, model or service qualification is granted by this commit. No network or device work was performed by the author.

The control is the actual main899 A2 service N1 S1-hoisted object, not an N2 pair or raw S2. Its archived binding is `out/konly-q32-hoist/206782/6a927892585b09d9dfd2635c7e188ca35920f9f62e2de430954b32ce19cc12a7/binding.json`. The service's final `actual_runtime_after_throughput.selected` explicitly records this hash, Q2048, N1 and I8-fragment hoist. The full 3219-file archive and separate 27-file S2 archive are checked by SHA; the S1 native audit is reexecuted. The original proposal and numerical/service pending flags stay false.

| Actual object | S1 production control | S2 preheader candidate |
|---|---|---|
| Hash | `53ef08b799eddb1170100ca6a7fca2171a54443acbf5a5e88549b35ceffd1603` | `2d8938635879f904f3d844b27cb8919ec3d6d4e71cf9a5b990362059065d90e1` |
| Final cubin SHA | `7eadbdbf854be149083065cb30f0f8c1bdc593aea6d2b9baf454502adb6fbb43` | `dab726c48cabe17bd795e2b35f094c5fefcb40db0e802f655bff51470b8715ce` |
| Runtime regs/spills/shared | 122 / 0 / 33792 | 124 / 0 / 41216 |
| Actual ELF SGPR count | 104 | 104 |
| ELF SGPR spills | 96 | 93 |
| Stages/warps/warp size | 1 / 16 / 64 | 2 / 16 / 64 |

The complete source digest and typed ABI, including constexpr strides and pointer attributes, match. Actual metadata differs only in hash, stages, shared size and the fixed override path. K uses I8 page/slot/head/channel strides `[16384,256,128,1]`; V is BF16 with strides `[8192,256,128,1]` at BF16 offset4096; F32 scales share the full raw base. The fixture has773 physical16KiB pages, Q2048×16×128, query pitch2048, table pitch8192 and grid65×2. Original source/module/debug/override paths remain distinct and fixed for each arm. Neither the S2 compiler ASM nor reduced resource counts prove final-cubin scheduling or latency overlap.

The numerical worker is a reversible adaptation of frozen `522c0b0617a8d2d862f60ebb03f05f6e08ccfcfe`. Its entire timing region and timing warmups are removed. It reuses the original allocation/kwargs, writer, CPU RN32/RNE codec, e4dd INT64-QK/FP64 global-softmax oracle, output/cache archives, explicit side stream and Graph-B-before-fresh-eager-B checks. The reference remains ideal FP64; it does not model online-softmax or tilewise BF16 P rounding. The pointwise `.005+.005*abs(reference)` and original BF16 relative-RMSE `.005` gates are unchanged. Explicit `--numeric-diagnostic` may retain a BF16-proxy quality failure; finite, codec, complete two-arm output bitwise, quantized reference, guards, readonly and actual-IR failures remain fatal.

Pattern0 uses Q2048/KV6144/prefix4096 and a full2048-row uniform oracle. Pattern1 uses Q2048/KV6143/prefix4095: last physical page has15 valid tokens and the K64 tile has63. The original fixture poisons the unused lane. Reference changes have exact inverse proof and are limited to the extra rows63/64, dynamic length, its matching causal prefix and the fixed-scope assertion. The writer slots cover `[4095,6143)` for this case, so the unused final lane is not written. B changes Q/K/V, lengths and writer slots, and is replayed before fresh eager B. This is one deliberately small aligned-plus-tail geometry, not16K coverage.

CPU replay, with no Torch/Triton import:

```sh
python3 -B -S tools/check_biv150_konly_q32_s2_numeric_cpu.py \
  --s1-raw /private/tmp/cjj-native120-a-4096-throughput-v2-raw-20261030 \
  --s2-raw /private/tmp/cjj-q32-s2-preheader-compile-v1-raw-20261030 \
  --static-review /private/tmp/cjj-agent-q32-s2-preheader-actual-review-20261030.json
python3 -B -S tools/probe_biv150_konly_q32_s2_numeric.py
```

Prepare one small dual-root proof locally; only this JSON and a separate Root decision need transport, not the service's84MiB archive:

```sh
python3 -B -S tools/probe_biv150_konly_q32_s2_numeric.py --prepare \
  --s1-raw /private/tmp/cjj-native120-a-4096-throughput-v2-raw-20261030 \
  --s2-raw /private/tmp/cjj-q32-s2-preheader-compile-v1-raw-20261030 \
  --static-review /private/tmp/cjj-agent-q32-s2-preheader-actual-review-20261030.json \
  --out /private/tmp/NEW-q32-S1-S2-preflight.json
```

The prepared proof contains the real remote roots and every selected seven-file-cache SHA. It preserves the original source paths. A fresh output/cache gets byte-exact copies of those files, relocating only group-index paths. The unchanged522c reviewed-cache warmup disables backend compilation: a new key, missing child, changed metadata, source, override, final cubin or ABI stops for review. Actual reader warmups and launches are separately observed. The unchanged writer can compile during its explicit warmup, outside numerical stages.

Root must review this frontend, retain the proof SHA, create a separate decision using `admission_template(FRONTEND_HEAD, PREFLIGHT_SHA)` (default `Root_numeric_approved=false`), and explicitly approve only this numeric scope. The remote source and both original output/cache directories must still exist; the script refuses missing/changed roots. It also requires an exact clean frontend, unchanged main899 production, pinned installed SDK/loader, Root-exclusive acknowledgement, no visible preexisting GPU process and closed9031. Root is responsible for serial scheduling and the fresh device-idle check. No launcher or automatic queue is included.

Only after that decision, the exact entry is:

```sh
/usr/local/bin/python tools/probe_biv150_konly_q32_s2_numeric.py --gpu \
  --root-owned-exclusive --commit FRONTEND_HEAD --numeric-diagnostic \
  --preflight ROOT_PREPARED_JSON --preflight-sha256 ROOT_PREPARED_SHA \
  --admission ROOT_NUMERIC_RECEIPT --admission-sha256 ROOT_RECEIPT_SHA \
  --out NEW_ABSOLUTE_OUTPUT
```

The bundle must include immutable helper commit522c and oracle commit e4dd (already present locally); source loading has no fetch fallback. The optional full-archive CPU checker additionally selects only the existing ELF/msgpack decoder definitions from `6b33cec889d9462dc3f167eff4be42a3c5108713` (source SHA09ac1c98); device execution does not depend on this decoder commit. It independently checks the two actual cubins' complete28-argument descriptors/216B kernarg and resource notes. The candidate source/override remains the actual2e compile root, while the baseline uses the actual main899 A2 root. A new checkout is not relabelled as either old compilation.

Full reader timing remains pending. If numerical/Graph/side gates pass, a subsequent explicit frontend may reuse the original single ABBA protocol for S1 versus S2 complete-reader and unchanged-writer-plus-reader separately, with full post-timing cache/output checks. It must report fixture-residency limits and cannot infer service or paired-prefill cost from this N1 component. No16K, new descriptor, model, capacity or performance claim is made here.
