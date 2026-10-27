# Native K8 V-first source experiment (CPU preparation)

Depends on PR93 frozen `3ebdbebf61e0e90735cd8ad83eb1605627b6ef46`, containing authenticated main `4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b`. CJJ verified main by fresh fetch and GitHub API on2026-10-28. No official framework update. This independent branch changes no production or existing PR93 file/library.

The sole dataflow edit issues the original V SME loads before the original K uint32 global-to-shared packing. The K barrier, V wait8/barrier, tail clearing, codec, QK, FP32 online softmax, page16 BF16 P, BF16 PV, C accumulator, stores, S2/S4 and ABI remain unchanged. K already uses aligned4-byte words; this does not reduce load count. Shared K covers exactly bytes[0,2048); V covers[4096,8192). This nonoverlap is a source safety argument, not proof of the actual backend schedule or useful overlap.

CUDA entry symbols, kernel names, codec export names and Torch namespace are isolated as `cjj_native_k8_vfirst_probe`; codec changes are names only. Exact inverses restore all three parent K8 files byte for byte, then the original parent inverse restores b3 native BF16 source. `COREX_NATIVE_GQA_V_OVERLAP=0` and its static rejection of the wait-moving variant remain.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_k8_vfirst_cpu.py
```

The private frontend preserves the original build/consumer/fullstep bodies by a narrow inverse-proved host adapter: independent output SO and actual LLVM/ASM first; Root CPU review of real V issue/K packing/barrier/wait ordering; then fresh consumer and complete BF16-vs-variant fullstep FP64/Graph/guards/true-width16/ABBA. Numeric adapters replace only library identity/namespace and result provenance; consumer keeps both pre/post library checks. The consumer replay receipt similarly changes only the frozen run/library identities, retaining every array/guard/input/Graph/FP64 test. The fullstep additionally requires the fresh variant consumer, while preserving the original c2e parent consumer precondition.

Parent22e and BF16aace library identities remain historical prerequisites/controls, not qualifications of this new binary. No compiler, Torch, Triton or GPU was invoked here. Actual binding, scheduling, numerical, performance, model, service and capacity qualification are false; no performance claim follows from the source order.

Root-only phases, each with a new output directory, under Root's externally coordinated exclusive device schedule:

```sh
python3 -B tools/probe_biv150_native_k8_vfirst.py --compile-only --root-owned-exclusive --out /ABS/NEW_COMPILE
python3 -B tools/probe_biv150_native_k8_vfirst.py --review-template --compile-summary /ABS/NEW_COMPILE/compile-summary.json
```

The second command prints an **unreviewed** sidecar template with three false flags. It cannot admit any numerical work. Root must inspect both full/tail actual LLVM and ASM, bind the exact summary/SO/LLVM/ASM/source hashes and record evidence in a separate review. If V issue does not precede K packing, or the original wait/tail/math cannot be established, leave the flags false and stop. This is a manual Root review receipt, not an automatic actual-IR auditor or proof conferred by setting flags. New bindings, library mismatch, missing/false review and unknown scope reject before Torch. `--root-owned-exclusive` records Root coordination; it does not infer ownership from invisible PID tables.

```sh
python3 -B tools/probe_biv150_native_k8_vfirst.py --numeric --root-owned-exclusive \
  --compile-summary /ABS/NEW_COMPILE/compile-summary.json \
  --assembly-review /ABS/REVIEW.json --assembly-review-sha256 REVIEW_SHA \
  --variant-library /ABS/NEW_COMPILE/build/cjj_native_k8_vfirst_probe.so \
  --codec-summary /ABS/PARENT_CODEC/codec-summary.json --codec-proof-sha256 CODEC_SHA \
  --out /ABS/NEW_VARIANT_CONSUMER
python3 -B tools/probe_biv150_native_k8_vfirst.py --fullstep --root-owned-exclusive \
  --compile-summary /ABS/NEW_COMPILE/compile-summary.json \
  --assembly-review /ABS/REVIEW.json --assembly-review-sha256 REVIEW_SHA \
  --variant-library /ABS/NEW_COMPILE/build/cjj_native_k8_vfirst_probe.so \
  --variant-consumer-summary /ABS/NEW_VARIANT_CONSUMER/consumer-summary.json --variant-consumer-sha256 NEW_CONSUMER_SHA \
  --consumer-summary /ABS/PARENT_CONSUMER/consumer-summary.json \
  --consumer-proof-sha256 c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128 \
  --bf16-library /ABS/B3_NATIVE/corex_native_gqa_s2_probe.so --out /ABS/NEW_FULLSTEP
```

Old codec receipt/consumer snapshots must remain present to replay their data, not only JSON labels. The new consumer uses all four original N64/S2 and N32/S4, pitch2048/2560 cases and three changed-input controls, with no timings. Fullstep keeps the immutable4K/16K protocol: both appends, BF16aace versus new K8 producer, original reducer, real scratch width16, complete cache/FP64/Graph/side/canary checks, then one128-warm/64-replay/8-unroll ABBA. Service new-KV pitch2560, model and all-reader dispatch remain out of scope. No threshold changes or capacity recovery.
