# Actual359 service specialization: fresh native mechanism

Root performed the6dcd3d8 compile-only job, owner652461/start346116491, same-start terminalZ. All39 raw files/3610628B and the unchanged download manifest were independently rehashed, then archived unchanged as40 regular members. Archive476440B/SHAaae141e17f58233df0ca18f92c998d18fb3052a2e0a53de5673b79bb68e1340e. The original compile summary retains its own native/hoist/sync/resources/numerical=false flags; no raw result was edited.

Fresh unhoisted c0a523d9d0d4d5dde57b241df9d22c9c38fa71e480878d18e690b4eb002e24f7 reconstructs the real selected service cache source/TTGIR/LLVM exactly. The new override produces597cf3bd3f7fda00c986b8af314f37fe59615a30b713e36dc6e40b804b3cfd05. Its actual TTGIRb7e4fe65d131892ed59a9d492dd3b39e255d5ee305d4fe752c87be101127143a is scoped-SSA-equivalent to the new d2671c… proposal; actual LLVM8d5ba289b3712aace744c7b29c7b3f449613f5d9d02f08f6d9de0de2f007367a and ASMce6bd638127529e46f93410d87abd37541c2537a81c34ad4947a91ad5e8436dd were freshly audited. These hashes are evidence, not admission predicates.

The new CPU audit JSON SHAa6adbe981060efada6724aef4cd039a6fe1dbc27a24852fae6d619d70464cc9e independently proves the mechanism on the actual new artifact: LLVM16 two-byte Q operand reads dominate KV with zero inside-loop Q reads, original16 BF16 global Q loads and48 RN32 divisions remain outside/dominating KV;16 native I8 QK and32 BF16 PV. ASM independently discovers KV headerBB0_39/Qv32:39,32 Q byte roots/24 packs, and no Q register write in any loop block. Q publish/K reuse/acquire barriers and their store/read CFG relations pass. Names/register numbers here are reported discoveries, never preapproved constants.

Replay without framework/device:

```sh
mkdir FRESH_CPU_EXTRACTION
 tar -xzf docs/evidence/konly-service-q32-hoist-20261027/fresh-native-v1/original-compile.tar.gz -C FRESH_CPU_EXTRACTION
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/audit_biv150_konly_service_q32_fresh_cpu.py --compile-out FRESH_CPU_EXTRACTION/out --out FRESH_AUDIT_JSON
```

The report's `compile_out` records the supplied local extraction path; source/runtime provenance and all original files remain strict. Actual live source signature/constants/attrs are now present in the original records. This is a0attention/0encoder/0initialized-input compile, not numerical or service validation. Loaded registers/spills are still unknown prelaunch; no loaded resource/occupancy claim is made. New bitwise numerical tool is prepared, not executed by this agent. Precision/performance/model/Graph/side/service gates remain false/pending. Earlier unit fixture and old overlay never license this new service.
