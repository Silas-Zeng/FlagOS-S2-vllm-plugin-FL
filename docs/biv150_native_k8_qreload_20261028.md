# Native K8 Q reload experiment (CPU-ready, default-off)

Dependency: PR93 frozen `3ebdbebf61e0e90735cd8ad83eb1605627b6ef46`, whose kernel/codec are the original 377/22e implementation. Fresh team main `4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b` is its ancestor; normal fast-forward into a new CJJ branch preserves both. No PR98/PR96 changes and no V-first issue order. All production `vllm_fl/` and `csrc/` bytes match team main.

The original producer retains `float values0[8], values1[8]` across peak reduction. This candidate first loads BF16 Q only for peak, preserving each `k`, peak0/peak1 update, lane XOR reduction, scale, shuffle and packing order. The existing `i=2*k+component` packing loop loads the same BF16 Q locations again. It adds no volatile operation, memory barrier or fast-math option. All original K/V issue, waits, MMA/C chains, softmax, output stores, ABI, S2/S4 and threshold expressions remain byte-identical after symbol inversion. Kernel/launcher/codec symbols and `cjj_native_k8_qreload_probe` are isolated from 22e and PR96.

Each entered CTA logically reads 8×128 BF16 Q values on each pass. The source requests an additional 2048 bytes per entered CTA, 32 bytes per lane; full 64×2KV×S2 and 32×2KV×S4 launches each request an additional 524288 bytes. Empty CTAs still return before either pass. These are source-level requested bytes, not measured DRAM traffic: the compiler may common-subexpression-eliminate the second load and retain the original float live range. Even a real reload may cost more than it saves. No register/occupancy/performance improvement is claimed.

Both Q passes use the original INT64 sequence address, head stride128 and identical dimension; Q is finite, readonly and nonaliasing with outputs for the full launch. The CPU proof uses nonzero base offset, Q pitches2048/2560, first/last sequence, all lanes/heads/dimensions, existing finite BF16 A/B/C fixture rows, the original peak reduction order and independent RN32/RNE row oracle. It checks exact FP32 scale bits and packed I8 words. It does not simulate device FTZ, division lowering or allocation.

```sh
python3 -Werror tools/check_biv150_native_k8_qreload_cpu.py
python3 -Werror tools/compile_biv150_native_k8_qreload.py
```

Only Root may execute the explicit target compile-only gate (0 input initializations/attention/codec/encoder/reducer launches):

```sh
python3 tools/compile_biv150_native_k8_qreload.py --compile-only \
  --compiler /usr/local/corex/bin/clang++ --out NEW_EMPTY_DIRECTORY
```

The output keeps all four exact generated sources, compiler identity, actual LLVM/ASM, effective build.ninja, objects and new SO SHA. Inspect each S2/S4 actual producer's Q global loads and float liveness across reduction, all native MMA/issue/wait chains, VGPR/SGPR/spills and actual loaded metadata. Unknown resources remain unknown. The parent reported100VGPR/68SGPR/0spill is background, never this candidate's result. If compiler behavior is unchanged or worse, retain the negative evidence and stop rather than forcing volatile/flags.

This first freeze provides source/CPU/compile entry only. No new consumer or timing qualification is inherited. If actual resources/mechanism justify proceeding, the next independent gate must bind this variant's own SO/IR receipt and rerun the frozen9695 consumer FP64 **pagewise BF16 P** reference plus complete final ideal gate, whole-input/cache/guard checks, changed-input Graph/side and narrowed width16 scratch. Only its own passing consumer may admit reuse of the full3eb append+producer+reducer ABBA logic; initial encoding remains separate, all old numerical thresholds stay unchanged. No service hook, model claim or GPU queue is introduced here.
