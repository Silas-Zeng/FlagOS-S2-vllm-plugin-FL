# Private compact12544 Native N48/S4 host-scope candidate

This branch starts at shared PR110 **4d1166792a5247b774d6b1977e60466574e6a34c**. It adds a separately named component, not a production installer or service route. Main a455, PR109 fc25 and the original PR110 branch/sources remain unchanged. Root independently confirmed main a455/PR109 shared74ee48 via CJJ1008 API; this worktree's fetch failed DNS, so no successful fetch is claimed.

The motivating observation is bounded: saved PR10916K server samples include many zero-prompt/nonzero-generation points with Running41/42, compared with31/32 in the older PR107 samples. Under the saved SDK's FULL padding policy this makes N48/S4 a candidate route. Samples are not actual graph-bucket frequencies and do not establish a causal explanation for the throughput regression. Parent N40/S16 and all non-selected routes stay unchanged.

## Concrete source change

New namespace **`cjj_native_k8_compact_n48_s4_probe`** has the same `produce` parameter schema and only N48/S4 host admission. `tools/biv150_native_k8_compact_n48_factory.py` pins the exact PR110 source and emits the three new files under `tools/kernels/`.

- CUDA and codec: private symbol renames only. After inverse renaming, their bytes exactly equal PR110. The RN32/RNE codec, I8-QK lane/C mapping, BF16-PV, online softmax, page loop, masks, 8KiB shared stage, V issue order and deferred-wait placement are unchanged.
- C++: the same symbol renames plus exactly two lines: the S2/S4 gate becomes S4-only, and `rows=segments==2?64:32` becomes `rows=48`. Every original dtype, stride, alias, scale/full-base, extent, stream and launch-error guard is retained byte-for-byte. CUDA already receives `batch` and derives `seq=blockIdx.x`; outputs use seq and passed strides. There is no N32 device-math constant to change.
- The unused private S2 CUDA symbol is retained solely to keep the source inverse exact. Its host gate rejects S2; no N48/S2 scope is claimed.
- Existing32/64 libraries retain only their own existing scopes. Candidate SO SHA is **null**. No qualification is inherited from40f9,22e,7892,PR95 or PR110's future/actual binary.

Compact roles remain K I8 with byte page stride12544, FULL layer-page0 F32 scale base (internal `page*3136+1024+slot*2+head`), and BF16 V already offset+4352 with element page stride6272. The common codec header remains unchanged. This is parameterized addressing, not decompression or BF16 reinterpretation fallback.

## Completed CPU checks

`check_biv150_native_k8_compact_n48_cpu.py` verifies source inverse, both Q pitches2048/2560,39 refusal cases, and six full-D128 nonperiodic fixtures (A/B/C × two pitches). It uses only the standard library. The CPU integer descriptor model is explicitly not execution of real C++ TORCH_CHECKs or a real Tensor/storage binding.

The29-page fixtures reuse the exact counter/codec functions from the original consumer, allowing N48 solely through a private function-global row whitelist; parent fixture files remain unchanged. They preserve nonzero layer offset128, all Kcode/scale/V bytes, reserved128-byte padding, guards, distinct head scales and complete global-row identity across the prospective split control. This check does not execute FP64 golden, Torch, GPU, Graph or side streams.

## Executable defaults and next device boundary

All following defaults are CPU-only:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_native_k8_compact_n48_cpu.py --out /private/tmp/cjj-n48-cpu-NEW.json
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/biv150_native_k8_compact_n48_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/compile_biv150_native_k8_compact_n48.py
```

Only Root may later compile with the same PR110 flags and an isolated fresh output. This command is prepared, **not run here**:

```sh
python tools/compile_biv150_native_k8_compact_n48.py --compile-only --root-owned-exclusive --out /workspace/experiments/minicpm-s2/cjj-compact-native-n48-compile-v1-20261029
```

The compile frontend is byte-inverse to PR110 after exactly two identity changes (factory import and TorchLibrary load namespace). It emits standalone LLVM/ASM, the actual SO, commands and artifact manifest;0 attention/codec/writer/reducer calls and0 input initializations. Torch's actual CUDA-initialization state is recorded rather than presumed. Actual SO/ISA/resources/schedule review is still required. It must bind this new SO; no old32/64 binary may be called with N48.

## Small consumer protocol before fullstep

The device numeric frontend is the next bounded work **after** candidate compile and actual SO review; no nonfunctional `--gpu` stub is supplied. Reuse the original compact consumer machinery and correct v2 pagewise-BF16-P reference, with `small_fixture()` here providing exact prospective N48 inputs. Keep both Q pitches,29 read-only pages, maxctx193 including0/1/15/16/17/tails, nonzero layer offset, max16 scratch backing, all O/M/L/final arrays, guards/poison/read-only snapshots, complete128D INT64/FP64 reference, unchanged `.005+.005|ref|` pointwise and `.005` RRMSE gates. Ideal-partial and original-BF16 proxies remain separate; failed diagnostics are not deleted or used to loosen final gates.

For an additional **reference-only** bitwise control, use two real qualified N32/S4 calls:

1. A controlled64-row Q/length/table backing has actual48 rows followed by16 zero-length padded rows. First call uses rows0..31; second uses rows32..63 (the last16 are padding). Preserve the original global rows32..47 counter vectors; do not generate new local0..15 vectors.
2. Candidate receives Q/length/table prefix48. Its producer uses max16-stride scratch narrowed toS4. The reference calls use the corresponding32-row slices of a separate max16-stride scratch, matching each original API's exact32-row shape and full-base/offset roles.
3. Compare all O/M/L/final fields for the actual first48 rows only; check each arm's remaining rows/guards under its own contract. Candidate-only backing rows48..63 remain poison; control's zero-length reducer rows48..63 become exact zero. No cross-arm full-buffer assertion is made for these different unused-row semantics.
4. Whole-cache/read-only checks remain complete. Changed B updates Q/cache/table/lengths and poisons outputs **before Graph replay**, then compares with fresh eagerB. SideC uses the same fixed pointers/epoch. A correct control is an independent numerical aid, not a substitute for INT64/FP64 or real Graph/side evidence.

The old22e library expects16KiB roles, so it cannot silently consume compact12544.40f9's actual N32/S4 compact scope is a possible explicitly pinned reference; its source/math similarity grants no N48 qualification. Any PR110 new SO reference likewise needs its own actual qualification first.

**The two-N32 split is never a performance baseline.** Its two launches and padding/control costs would make a misleading speedup denominator.

## Later same-N48 fullstep comparison

`fullstep_plan()` enforces44769 physical pages,48 query rows with41 or42 effective rows and explicit zero padding, full table width8192, disjoint physical pages and actual max16 scratch. It reserves three successive append boundaries:

| Effective rows | Initial context | After3 appends | Reserved pages | Spare pages |
|---:|---:|---:|---:|---:|
| 41 | 17408 | 17411 | 44649 | 120 |
| 42 | 16896 | 16899 | 44394 | 375 |

42 independent17408 contexts already need45696 pages (45738 including three append steps), and48 need52224; both are refused. No modulo/wrap mapping, hidden shared pages or fictitious48 independent17K contexts may replace the real capacity constraint. Actual future sampled lengths may choose a different explicitly checked page sum.

Performance must compare **new Native48 producer + matching S4 reducer + actual compact writer** against the **current real compact TL48 producer + matching reducer + the same writer**, at identical48 rows,41/42 effective+padding, contexts/table/page map/cache format/streams/source pitch. Keep append in the timed region; report full initial encode separately. ExistingBF16 PR95/aace or two-N32 controls are auxiliary numerical references only. Preserve full FP64 O/M/L/final, complete cache bytes, canaries/read-only, changedGraphB-before-eagerB and side gates before timing. A contiguous256 append source remains a component scope, not proof of the service's2560 source stride.

Current status: source and CPU protocol checks pass; actual compile/SO SHA/resource/wait/consumer/Graph/side/fullstep/timing/Math105/service activation are pending. No model/CLI/scheduler/threshold or production route changed. Root owns future GPU serialization and Draft creation/publication.
