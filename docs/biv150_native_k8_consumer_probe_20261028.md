# Small native K8 consumer + original b3 reducer numerical gate

Current branch normally merges fresh main4229b48 (PR94). Production is byte-identical to that main; b3 native/reducer source and all tested377 prototype files are unchanged. Existing library22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd is reused without native compilation. No service hook, writer, timing or model gate is added.

Actual codec gate now passed: owner696818/start347452708, same-start Z;33 rows×4 observations had exact complete I8 code bytes and F32 scale bits, six host refusals, changed-input GraphB before fresh eagerB, sideC and full guarded storage/readonly. All39 raw496437B and collector hashes were independently checked; full snapshots were replayed against the rational PR83 formula. `codec-v1/original-codec.tar.gz` retains all39 originals+manifest (52698B, SHAa5dbfd2ebb2131e43ae76a8685d9bb275905b329d21367a2843e5a9cf0feda0b). Actual summary SHA3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263; Root's independent review SHAda3e9cf727d0f0c9b4ad4c7fd534e0a60ee47ce4e7f120990e991f26f3356d44 is retained separately. This qualification covers codec boundary fixtures, not native attention, service or model precision.

The new consumer entry refuses before importing Torch unless the actual codec summary matches an explicit supplied SHA and indicates all strict gates passed. It revalidates every artifact SHA and the original four phase outputs, admission before/after storage, changed B input/poison and independent code/scale bits. It never runs codec to manufacture a prerequisite.

Consumer cases are batch64/S2 andbatch32/S4, both Q row pitches2048/2560, D128/16 query heads/2 KV heads/page16. Contexts include0/1/15/16/17/31/32/33/63/64/65/127/128/129/191/192/193. A small shared29-page readonly arena avoids a bandwidth test. Its encoded layer starts at nonzero byte offset128; K-code and F32 scale views point to the same FULL layer-page0 base, V already points to+8192B, with canonical16KiB page pitch. All CPU bytes for codes/scales/BF16 V/reserved fields are built before upload; no GPU encoder is invoked.

Q/K/V use full128-D nonperiodic integer-counter patterns, independent by head. Q has an exact distinct peak/scale for each head, at changing dimensions/amplitudes across phases; input pair2g/2g+1 differs from outputg/g+4. K scales also vary by token/head. Full CPU NumPy INT64 Qcode×Kcode matrices are dequantized into FP64 scores, then complete FP64 softmax/PV outputs and per-segment O/M/L are formed. This uses every dimension, not a periodic identity or an output sample. CPU checks cover89518080 integer products across12 full cases and reject a wrong K D32-chunk permutation. Original BF16 FP64 output is saved as a separate proxy; it grants no model qualification.

The hard implementation gate remains finite values, every absdiff<=.005+.005*abs(reference), and relative RMSE<=.005 on the entire active O/M/L and final output arrays. Inactive partial/max/sum slots must retain the exact NaN poison bits. Zero-length rows must produce positive-zero output bits. Before/after whole backing storage includes scratch/output prefix/suffix guards, Q row padding, encoded cache, table and lengths. Producer-only snapshots precede reduction; O/M/L must remain byte-identical after the reducer.

The reducer is actual b3 `vllm_fl/patches/biv150_gqa_decode.py::_reduce_partials`, whole-module SHAf1877ad06e8be4cef5960ef2d66d59045b4ac1ac0d7f2fe448618bbdec0d7f40. It is imported privately without installing the package. This is the stride-aware reducer used after native production, including seq_len0 handling, rather than SDK reduce_segments. Its normal Triton specialization may occur on first device use; no new reducer algorithm or native build is introduced. Actual reducer source/compiled handle/IR are recorded.

Each case runs eagerA, captureA, changes Q/cache/table/lengths and Q/K scales toB, poisons every O/M/L/output, replays GraphB before new eagerB, then runs changedC on a side stream. Complete arrays are saved as content-addressed raw/NumPy blobs referenced by every observation, preserving before/after bytes without duplicating identical readonly buffers. Whole-storage CPU copies are authoritative; no GPU advanced negative indexing is used. All postlaunch errors propagate into a failed record with raw snapshots. There is no fallback or tolerance adjustment.

Default plan is stdlib only. Full CPU reference checks require NumPy and import no Torch/Triton:

```sh
python3 -B -W error tools/probe_biv150_native_k8_consumer.py
python3 -B -W error tools/check_biv150_native_k8_consumer_cpu.py
```

Root-only future command, serialized after current throughput and with a new output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/probe_biv150_native_k8_consumer.py --gpu \
  --library /workspace/experiments/minicpm-s2/cjj-native-k8-compile-v1-20261028/build/cjj_native_k8_qk_probe.so \
  --codec-summary /workspace/experiments/minicpm-s2/cjj-native-k8-codec-v1-20261028/codec-summary.json \
  --codec-proof-sha256 3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263 \
  --out /workspace/experiments/minicpm-s2/NEW_CONSUMER_NUMERIC_OUT
```

No device execution was performed by this author. Native consumer/Graph/side numerical qualification remains pending. No append/full-cost performance, service lifecycle or model105 evidence is claimed; precision_gate=false. The separate d8ef model102/105 result does not qualify this22e native implementation. A large-arena complete append/consumer/reducer experiment is deferred until these small gates pass.
