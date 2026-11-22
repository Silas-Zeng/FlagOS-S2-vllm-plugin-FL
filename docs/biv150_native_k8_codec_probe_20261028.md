# Actual compile evidence and next codec-only gate

Actual source37713909 compile-only completed: owner688550/start347221367, same-start Z; 28 originals /906281B, all download and compile manifests rechecked. Four standalone LLVM/ASM commands exited0 and the actual Torch Library22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd loaded without CUDA initialization. Counters remain zero reader/encoder/codec/reducer/input initialization. All28 originals plus their collector manifest are in deterministic `compile-v1/original-compile.tar.gz` (222400B, SHA5c831857413e0c1f2a4f199f64cdb50c85997318cead2ece84dd5016dfb553af). No producer, host ABI, codec, Graph, model or throughput numerical pass follows from compilation.

Independent stdlib audit verifies exact tested source/inverse, every original SHA, both manifests, actual library, native symbols and the extension's embedded GPU ELF. The SO contains exactly the same GPU ELF bytes as its two saved CUDA objects. Their actual binary metadata reports producerS2/S4 SGPR68/VGPR100/spills0/shared8192, codec SGPR24/VGPR8/spills0/shared0, warp64. These are binary metadata, not loaded-handle/postlaunch observations or occupancy measurements.

Standalone LLVM retains unflagged F32 fdiv and explicit llvm.nvvm.f2i.rn. The codec has three static divides and two conversions; producer profiles have18 divides/16 conversions. Each producer has4 signed-I8 QK MMA and8 BF16 PV MMA on each full/tail page path, with signed sitofp for the C score. Actual ASM includes I8 m16n16k32 and BF16 m16n16k16. Codec ASM uses reciprocal plus quotient/error FMA correction and exceptional-class selection; it is not just an unchecked reciprocal. Nonetheless exact RN32/FTZ device behavior is still unverified. The extension build adds Torch definitions/includes, ABI flags, -cl-single-precision-constant, bonus-inst-threshold/skip-double and fclang-abi-compat; standalone LLVM is not claimed to be extension-object IR. The hard-SHA actual SO is therefore used for the next numerical gate.

New `probe_biv150_native_k8_codec.py` defaults to a stdlib-only CPU plan. Its explicit device mode only loads that existing library, never compiles. The independent rational PR83 RN32/RNE formula generates complete code bytes and F32 scale bits from exact BF16 input bits. Vectors cover signed zeros, subnormals/minimum normal, maximum finite, mantissa boundaries, half-integer families across exponents and mixed finite BF16 exponents. Input conversion/upload preserves bits through integer views. Strict equality uses no epsilon or alternate reference.

Device phases are eagerA, captureA, changedB+poison+replay before new eagerB, then changedC on a side stream. Both codes and scales change fromA toB. All before/after input/code/scale backing bytes are saved; entire storage is copied to CPU before any guard selection, avoiding the previously observed negative GPU advanced-index issue. Graph output must match both the independent B reference and subsequent new eagerB. Six invalid dtype/extent/alias calls must throw before launch and leave storage unchanged. Failure stops and preserves raw snapshots/trace; no fallback or tolerance widening exists. This probes only codec execution, not K-cache append, I8 QK layout or attention output.

CPU replay commands:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_k8_archive_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_k8_codec_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/probe_biv150_native_k8_codec.py
```

Root-only future device command after its serialized queue, with a new empty output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/probe_biv150_native_k8_codec.py --gpu \
  --library /workspace/experiments/minicpm-s2/cjj-native-k8-compile-v1-20261028/build/cjj_native_k8_qk_probe.so \
  --out /workspace/experiments/minicpm-s2/NEW_CODEC_NUMERIC_OUT
```

Current author performed only CPU reads/checks and archive construction, no Torch/JIT/native compile/SSH/GPU. Production and tested377 sources remain unchanged. Native attention numerical/Graph/resource/performance and model105/service gates remain pending; precision_gate=false. Root retains sole scheduling authority; this commit requests no concurrent device work.
