# Compact12544 writer / split GQA device frontier (prepared, not device-tested)

This independent tools revision starts at PR91 `8babed1957f3477bfa158ecdadc8b7d020b1147c` and normally merges authenticated team main `4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b`. The merge adds PR94 tools/docs. Every production/compact source remains byte-identical to8bab; no algorithm, codec, thresholds, dispatch, installer or physical layout changes. All13 embedded PR89 implementation/manifest files match `099f38aa382581645420699c3f6291604e7bc147` (whose production is d8ef). The receipt records that comparison; the new runner reads reachable8bab Git blobs, so an unrelated99 object is not a runtime dependency. PR89's 16KiB-page Math105 result does **not** qualify compact12544.

The existing real default-off spec, manager and allocator/reshape hook are unchanged. Their two old CPU checkers still pass with framework/Tensor substitutes. The ten physical-address literal edits across six generated GPU roles remain exact byte inverses to the parent. No currently known additional bridge defect was found. This new component entry does not enable the service flag or install/activate any hook.

The first device scope is the compact **decode79 producer's 3D grid**: `[64,2,2]` at4K and `[32,2,4]` at16K, with actual `_pick_segments`, one warp/stage1, original reducer4 and scratch narrowed from capacity16. It is not the SDK raw3D profile. 2D/BM128, Q32, multi2, SDK raw3D, scattered mixed, actual service configuration/allocator lifetime, model and capacity/performance qualification remain separate follow-ups. The old16KiB Q32 hoist is rejected before device imports and stays unsupported in the real compact installer.

The explicit device path first executes the **actual frozen `_allocate_kv_cache_tensors` method** with real Torch and an explicitly one-layer/two-page configuration, instantiates the actual fixed compact spec, and calls the exact reshape bridge. It must observe25088 backing bytes and carrier strides `(6272,2176,256,128,1)`. This is a real Tensor/component probe, not42-layer service/registry/RPC qualification.

Guarded reader/writer fixtures use a256-byte nonzero storage offset. K code and scale point at the layer's full page0 base, with byte/F32 page strides12544/3136. The reader's BF16 V pointer is already4352 bytes past that base and has page stride6272. The writer's `value_full` is fullbase and adds2176 BF16 elements exactly once. Actual Tensor observations are checked against the independent role broker. No nominal BF16 K carrier is read as BF16 keys.

The numerical entry performs M1/M8 code+scale bit tests over141 vectors each; a strided/nonzero-offset writer edge case including invalid I64 slots, poisoned invalid K/V rows and token-count bounds; complete initial encoding into a distinct destination; full cache-byte checks including reserved/unmapped bytes and all cache guards; and original BF16 versus independent INT64/quantized-operand FP64 attention references. It retains original `.005+.005*abs(ref)` pointwise and original `max(.005,1.2*baseline_RRMSE)` quality policy from the unchanged79 reference helpers. The compact byte oracle is new12544 layout; old16KiB packing/append helpers are not called.

Every actual producer is freshly compiled and audited for logical8, K32 I8 MMA, complete integer C chains, original BF16 PV/native accumulator chains and exact score-cast sites. Typed ABI, actual JIT source and all actual source/TTIR/TTGIR/LLVM/ASM/cubin are exported. Post-launch loaded resource fields are recorded separately; pre-launch nulls never prove equality. Cold/source hashes, CPU generator seed/version, complete outputs/partials and FP64 witnesses are retained; half-GB cache payloads are not archived. All attention rows receive independent references, not only output samples.

Three controls include dense, ragged/empty/tail and zero-Q cases. The captured closure includes compact append+producer+reducer. Replay changes Q/K/V/slots/table/lengths after poisoning cache/scratch and compares every output/partial field bitwise with eager. A side-stream run must match too. The BF16 numerical control is the exact current main Triton producer/reducer; its post-append BF16 cache is prepared independently outside capture. This is **not** a matched full-cost benchmark, and no timings are measured. Original BF16 quality failure remains `passed=false` even if all implementation gates pass; implementation, model, capacity and performance are separate fields.

Default commands (stdlib only, no Torch/Triton/NumPy import):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_compact_device_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/bench_biv150_konly_compact_device.py
```

Root-only first4K compile frontier, using the actual frozen source commit and a fresh absolute output directory:

```sh
python tools/bench_biv150_konly_compact_device.py --run-gpu --compile-only --case 4k --commit COMMIT --out /ABS/FRESH_OUT
```

`--compile-only` skips all Triton attention/codec/writer launches; Torch allocation/fixture initialization still uses the device. It exports newly compiled IR and never counts that as numerical evidence. After Root reviews it, numerical/Graph/side execution is explicit:

```sh
python tools/bench_biv150_konly_compact_device.py --run-gpu --case 4k --commit COMMIT --out /ABS/ANOTHER_FRESH_OUT
python tools/bench_biv150_konly_compact_device.py --run-gpu --case 16k --commit COMMIT --out /ABS/THIRD_FRESH_OUT
```

A busy9031 or GPU-memory check refuses; it is not a replacement for Root's exclusive scheduling. Source HEAD/clean tree, exact8bab production, current pinned JIT/frontend and target ABI must match. Runtime errors write their real result/traceback and propagate. Output directories cannot be reused. No SSH/JIT/GPU work was performed by the author; all device/IR/numeric/Graph/side/resource/model/service/capacity/performance gates remain pending in this freeze.

## Frontend correction v2 (no kernel/numerical changes)

Root separate import probe showed normal platform bootstrap already loads the current-source `vllm_fl` and initializes CUDA. The354 candidate's package-absence assertion would incorrectly refuse; no91 GPU run was made with354. The preserved observation is labeled with its separate source path. The next tools revision allows normal platform bootstrap without disabling plugins, authenticates actual package/patch/private module paths and frozen-source SHA, requires all private cache/hoist flags off and both service `_STATE` values absent/None, and reuses canonical already-loaded modules. Five CPU rejection tests cover foreign paths, activation switches and active service state. The original complete numeric NPZ/checks are now saved before the unchanged final-FP64 assertion, so a failing first numeric record remains reviewable. CPU v2 is `cpu-bootstrap-v2.json`; the original v1 CPU report,354 source and absence of a354 device run remain explicit.
