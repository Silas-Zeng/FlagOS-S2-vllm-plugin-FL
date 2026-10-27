# Compact K8 Q32 operand-hoist prototype

Default OFF, CPU preparation only. Base is actual main4229 plus an explicit normal merge of PR91 at5ca6a5464f9e8d11f30428c46508ff0b76e204de. PR89/98's encoded adapter/native auditor are source dependencies, not transferred device or model certificates. PR91's pending service/model qualifications remain pending. No SSH, GPU, compiler, installation or timing run is performed here.

The six compact GPU source files are byte-identical to5ca. Compact Q32 differs from the encoded16KiB source in only the scale helper's page stride4096→3136. Its top-level JIT fn.src can therefore remain equal while the independently jitted scale helper changes. Cold admission checks the complete compact module SHA and all full-module CodeType fields (including nested constants, filename/flags/line/exception tables) of the kernel, scale helper and RN32/RNE codec, exact module globals and no closure. Hot calls retain these exact Python/JIT/helper/code identities. Source/helper replacement is a hard prelaunch rejection; an old top-level source digest alone never authenticates this format.

## Physical and lifecycle contract

For layer-local page p, slot t, KV head h and dimension d:

- K I8 full-layer base: byte p*12544+t*256+h*128+d; tensor strides12544/256/128/1.
- Scale F32 full-layer page-zero base: element p*3136+1024+t*2+h. The pointer is not pre-offset4096B; the helper adds that offset exactly once.
- V BF16 already-offset view: pointer layer+4352B, element stride6272/256/128/1; final byte p*12544+4352+t*512+h*256+2*d.

The layer's storage_offset may be nonzero, aligned to4B. Code0..4095 and scales4096..4223 leave128B padding before V4352..12543. All views share the registered storage and remain in the full layer extent. Neither V nor scale receives a second offset. The existing mode0 I8-QK/I32 score/dequant and BF16 PV remain unchanged; there is no K decompression allocation and no stock BF16 reader for an owned compact cache.

Before Q32 preparation, the current registry must be armed and own the exact Entry. Its generation is positive/current; the owner descriptor must equal the claimed descriptor; current role Tensor identities/scalars and their complete broker metadata must match the compact prefill ABI. The first admitted entry's owner/generation/role binding is retained and checked on later calls. Reallocation, a changed generation or role view, double offset, stale owner and foreign helper fail before any reader. K/V read-only sources may share each other, as before; cache/source alias rejection remains the existing registry/broker rule.

Compiler proofs are keyed by the original actual SHA-pinned JIT binder/compute_cache_key, actual CompiledKernel identity, typed ABI and geometry. They are not cache-content, pointer-address or generation certificates. Reusing native code for a new layer/pointer/query geometry still requires that call's current compact owner/roles and complete Q32 scope. These metadata checks have real host cost; zero cost is not claimed. No per-call source hashing, inspect.bind or file reads are added.

## Narrow opt-in and failure behavior

The existing flag now selects the compact-specific adapter when compact is the sole active cache format:

```sh
VLLM_FL_BIV150_KONLY_COMPACT_SERVICE=1
VLLM_FL_BIV150_KONLY_ENCODED_SERVICE=0
VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST=1
TRITON_CACHE_DIR=ABSOLUTE_PRIVATE_CACHE_DIRECTORY
```

The hoist flag defaults to0. Fixed serve/benchmark CLI and model parameters stay unchanged. This branch does not integrate the separate native compact-decode candidate. Only the existing eager single-sequence Q32 selection (query rows>=1024, B256/Q32/K64/w16/s1) can prepare a hoist. M2 uses the full qrows//32+num_seqs grid, normalized128/raw3D/decode/writer retain their original routes and bytes. Capture retains the existing private normalized route; this is not a new Graph qualification.

Each actual cold specialization warms the original private Q32, proposes one immutable QCode local_load move, freshly compiles its private override, compares complete typed ABI/source TTIR, audits actual TTGIR/LLVM/native synchronization and I8/BF16 matrix grammar, and checks loaded register/spill/shared bounds. No old hash or overlay is an admission predicate. Unknown compile/IR/resources before launch retain the same correct compact unhoisted function and arguments/grid. Address/source/epoch corruption cannot take that fallback. The single reader launch is outside the catch; postlaunch errors propagate without retry. All old audit thresholds stay exact.

Manifest and implementation fingerprint include the new compact files and actual bytes of the reused encoded adapter/native helper. Cold source verification checks every dependency. The worker still rejects simultaneous compact+encoded16KiB activation; compile_semantics records the explicit compact hoist flag. The manifest distinguishes adapter availability from device qualification: q32_hoist_opt_in_adapter_available=true, q32_hoist_supported=false and actual_compact_device_verified=false until fresh device evidence exists.

## Reproduction and evidence

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_q32_hoist_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_q32_hoist_cpu.py --installer-on
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_konly_compact_install_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_hoist_standalone_cpu.py \
  --helper /private/tmp/cjj-actual-owned-service-helper-94db-20261028.py
```

CPU evidence covers12288 physical addresses including all dimensions/tails/last page/nonzero layer offset, real full-module CodeTypes, eleven source/role/epoch negative cases, actual facade arguments/parent grids, inherited preparation/audit methods, one-launch postlaunch exception identity, and installer/fingerprint OFF/ON routes. Frameworks/JIT handles for installer tests are explicit stand-ins; actual binder realization remains unknown.

The separate source-only subprocess uses -I -S, loads the exact original94db helper, and invokes its unmodified expected_fingerprint, including the original spec_from_file_location loader with no parent package. Compact ON with hoist OFF and ON both generate fingerprints without GPU/framework imports; a finder actively rejects torch/triton/vllm/vllm_fl/numpy imports. Explicit CPU ABI and disabled native-library flags avoid claiming actual loaded-library admission. No normal-package/fakeJIT same-process test substitutes for this gate.

There is no fresh compact Q32 TTGIR/native/resource/numeric/Graph/105/service/performance qualification yet. Inherited encoded proof is not relabelled compact. Next Root-only gates are actual fresh compact specialization and operand/synchronization evidence, bitwise original-vs-hoisted compact output with full guards, changed-input/stream behavior, then appropriate service/model and matched full-cost performance. A failed/unknown native grammar must retain unhoisted execution, never relax its auditor.
