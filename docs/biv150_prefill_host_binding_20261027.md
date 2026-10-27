# Private SDK-scope host-binding diagnostic

This update closes the earlier **source** gap using the actual target files and
prepares a private component entry. Default execution remains stdlib-only CPU.
The explicit GPU branch has **not been run**, and no service, SSH, JIT, device
queue or production installation was performed while preparing it. All old140,
67 and e4 evidence, the old unconditional kernel/factory/bench and production
files remain unchanged. The historical missing-source statements in the earlier
feasibility/protocol evidence describe those earlier snapshots.

## Actual target freshness and capture chain

The target export identifies vLLM `0.24.0+empty`. The two actual files are:

| File | SHA256 |
| --- | --- |
| model_executor/layers/attention/attention.py | 3414faca4cc04c699dbb880ba1761230e1ed97035f2934604cad9a5490f4dba1 |
| forward_context.py | 1c6fa51e9a02a120fed5e7357afb1190516cbdfefd1c644dfcd860603f1527c5 |

Their exact bytes and original manifest are in the11,897B deterministic
[actual-target source archive](evidence/prefill-reverse-20261027/host-binding-v1/actual-target-sources.tar.gz),
SHA256 `eac530fbc7a1d10453144c01764283d848e41d762566dd3883b76e26f7130781`.
The new proof reads that verified actual export, not an unverified official package.

Target attention.py692 reads `get_forward_context().attn_metadata` on each
`get_attention_context` call. The real `unified_attention_with_output` at772–784
passes that metadata to `self.impl.forward`;800–805 registers this function as
the real custom-op implementation and the separate787–797 function as its fake.
Both the direct-call and torch.ops paths in `Attention.forward` reach this op.
The output mutation declaration and KV dummy dependency preserve the intended
opaque-op ordering; the fake implementation is not runtime attention evidence.

Target forward_context.py221–233 stores the supplied current metadata in a new
ForwardContext. Its235–246 override restores the previous context in `finally`;
the307–323 set-forward-context path constructs/yields that context for the call.
Together with pinned main56f runner/C3, dispatcher51f and wrapper722 source,
ordinary nonuniform prefill under the fixed FULL_DECODE_ONLY/key invariants takes
NONE and calls the runnable each time. Forced dummy/capture metadata is not claimed
to be ordinary prefill. The CPU proof executes the actual getter/custom-op/context,
NONE wrapper, runner uniform predicate and dispatcher AST; it observes metadata
4096→16384→4096→16384 with exact object identities and nested restoration.
Runtime registration/compiled-graph behavior still needs its device/service gates.

## Prefer explicit SDK arguments over an outer context hook

The SDK54c495 entry already receives Python `max_seqlen_q` and `max_seqlen_k`.
`LocalSdkBinding.call` reads these **on this call**, within the SDK call's local
extent. It installs no outer forward-context/custom-op/provider hook and does not
replace either module's high-level attention callable. CPU cu_q/pointer/packing/
coherence provenance is an explicit one-shot component contract. A service adapter
would still need to establish this provenance from its audited caller; arbitrary
raw SDK calls are not automatically admitted by shapes or package version.

Only an eligible short call temporarily installs a diagnostic tap on the
already-selected Q32 manager's wrapper. The same top-level M2/Q32 objects remain.
Baseline calls have **no new tap or wrapper**. Long/unknown calls go straight to
the original SDK callable with **no new metadata Frame, boundary tap or capture
query**. Their CPU checks/contract handling still cost time. A short call restores
the Q32 wrapper in `finally`, with no overwrite on unknown ownership or launch retry.
The tap reuses the e4 router and exact old fca source381987; it does not add a device
predicate. Long context returns the original selected Q32 call/binary. An oversized
batch U can conservatively miss a short prefill. Extra host cost is not assumed0.

The CPU proof runs the actual main56f Q32/M2 class AST: reverse→original Q32→reverse,
reused/missing/wrong/nested contract fallbacks, exception cleanup, and M2's513+513
grid34x2 bypassing the single-sequence tap. Source/setup checks do not establish
GPU typed ABI, numerical correctness, actual Q-hoist retention, resources or speed.

## Default CPU entry

```bash
python3 -B tools/bench_biv150_prefill_host_binding.py \
  --source "$PWD" \
  --sdk-source /path/to/pinned/triton_unified_attention.py \
  --helpers-source /path/to/pinned/triton_attention_helpers.py \
  --target-sources docs/evidence/prefill-reverse-20261027/host-binding-v1/actual-target-sources.tar.gz \
  --out /tmp/cjj-host-binding-new-cpu-output
```

The output directory must not exist. The initial source/scope
[v1 CPU JSON](evidence/prefill-reverse-20261027/host-binding-v1/cpu.json), SHA256
`c36dba6d966ae116d8c0f5ff0781479540e993e10972c7d173f90afff712f49c`, is retained
unchanged. It did not exercise the erroneous native GPU admission corrected below.
The current [v2 CPU evidence](evidence/prefill-reverse-20261027/host-binding-native-v2/cpu.json)
adds the actual native source contract; archive/directory replay remains identical.
No Torch/Triton import, library load or JIT occurs in this branch.

## Native identity correction before any GPU run

The initial3af diagnostic incorrectly forced `import vllm_fl._C` and equated its
file with the accepted5c SO. That admission was wrong and was caught before GPU
execution. The5c file is `biv150_native_w8a8.so`: native_w8a8.cpp163/171 registers
`TORCH_LIBRARY/IMPL(biv150_native,...)`, and the build script explicitly sets
`is_python_module=False`. The existing W8 patch loads it through
`torch.ops.load_library`, then registers fake schemas. The separate package `_C`
import at __init__.py111 is optional and catches ImportError/OSError.

`--native-library` now means an identity file only: verify its5c disk SHA and set
the **actually recognized** `BIV150_NATIVE_W8A8_LIBRARY` environment variable.
`compile_semantics.native_library_path()` must resolve to that same path. The
similarly named `VLLM_FL_BIV150_NATIVE_LIB` is not read by main56f. The pure-attention
diagnostic does not call the W8 loader, force an optional extension import, rename
the library or install a fake module. No5c-as-`_C` materialization is required.

Before/after records distinguish the W8 identity file, whether its path is mapped
in `/proc/self/maps`, its presence in `torch.ops.loaded_libraries`, and the existing
W8 loader's `_LOADED` flag. Unavailable evidence is unknown. If optional Python
`vllm_fl._C` was naturally imported, its real path/hash is recorded independently;
it is never used as the5c identity. CPU tests use marker files and isolated stub
arguments only, execute the real resolver/loader AST, and import/load no Torch/SO.

## Explicit future GPU entry: prepared, not executed

After root verifies instance identity and schedules an idle device, the same
entry requires `--run-gpu`, exact `--commit`, a new output directory and the accepted
W8 identity-file SHA5cbb8853 (loading that operator is unnecessary for attention).
It rejects a live9031 service, busy GPU, dirty/wrong source,
production differences from main56f, wrong imported package/target sources/native
library or disabled M2/Q32. It uses a private Triton cache and never edits installed
packages. Existing accepted parent setup is in-memory in the diagnostic process.

```bash
python3 -B tools/bench_biv150_prefill_host_binding.py \
  --source "$PWD" --commit FROZEN_HEAD --run-gpu \
  --sdk-source /path/to/actual/pinned/triton_unified_attention.py \
  --helpers-source /path/to/actual/pinned/triton_attention_helpers.py \
  --target-sources docs/evidence/prefill-reverse-20261027/host-binding-v1/actual-target-sources.tar.gz \
  --native-library /path/to/confirmed/accepted_native_library.so \
  --out /path/to/new-component-output
```

The default bounded screen is query2048/KV4096 and query2048/KV16384; optional
`--case upper-bound-miss` uses own KV4096/U16384. Cases cannot repeat and repeats
are bounded1..32. Policy selection never reads these names. It prepares these gates:

- Exact direct baseline/candidate output, unchanged parent binary for misses,
  CPU-double/reference guards without tolerance changes, NaN poisoning/canaries,
  unchanged inputs, stale contracts, side streams and fresh KV metadata.
- Actual selected typed ABI/compiler-options/target equality, verified Q-hoist,
  observed postlaunch strict resource comparison, actual source/IR/ASM/cubin and
  Manager override-IR preservation. A long miss cannot touch private reverse counts.
- **Two distinct Graph checks.** Warmed selected-kernel graphs bypass the host scope
  and replay after changed input/KV metadata. Actual SDK-scope capture follows the
  existing parent policies' BM128 generic fallback (prewarmed and independently
  compared); it is not claimed to capture Q32/reverse. Its exact graph binary is
  not independently traced, and this limitation is explicit in the result.
- Eager SDK AB/BA/AB timing reports wall-clock and CUDA-event ratios. Baseline has
  no new diagnostic tap; candidate includes contract allocation and actual host
  checks, plus local scope/routing/diagnostic recording for eligible short calls.
  Parent observers used for numerical leave-evidence calls are removed for timing.
  Raw selected Graph timings are not
  substituted for this full component cost.

Completion only means the requested component work finished. The result keeps
`model_precision_gate=false`, `official_score=false`, no service throughput claim,
and no full numerical-matrix/statistical-stability claim. All those gates, as well
as this new binding's device gates, remain pending until actually executed.
