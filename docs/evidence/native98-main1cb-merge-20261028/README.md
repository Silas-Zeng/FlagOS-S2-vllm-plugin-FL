# PR98 normal merge of team main 1cb: CPU compatibility

This local merge starts from PR98 shared head `c30cf4e2f56b58efaca100ffa54157dd53744944` and merges team main `1cbac531ca9afaa0e9a68624d5d10cb1d0412055`, with no textual conflicts. It does not update PR98 or any frozen a975 checkout. The target main object was previously fetched and verified; a final re-fetch attempt failed first at DNS and then at the 443 connection timeout. These failed attempts are not recorded as a fresh successful fetch. Root must recheck team main before publishing this local merge.

The incoming main CUDA source, native builder, and stock NativeGQA Python module remain byte-identical to 1cb. Both the CUDA fallback macro and explicit build flag enable `COREX_NATIVE_GQA_V_OVERLAP=1`; the builder receipt and stock runtime events retain `overlap_v=True`. No private encoded-cache math, source factory, hoist adapter, native K8 module, or worker hook changed.

The only additional production edit updates the strict installer pin for the actual new stock NativeGQA Python file from `86f8ca199ce632ccb3bbc251bb5c632e57da3373d8b87bd5f53334ef818a26c4` to `2f5e730a428548a235b7c1cee7f5222176cce6d840fb406079c042052a49b0e1`. The existing CPU producer-boundary checker now compares against exact main 1cb. Neither change weakens a guard.

## CPU verification

Python 3.12, `PYTHONDONTWRITEBYTECODE=1`, `-B -W error`; no framework import, native load, JIT, CUDA, device execution, or SSH was performed. Existing checker commands (run from this worktree):

```sh
python3 -B -W error tools/check_biv150_native_k8_service_cpu.py
python3 -B -W error tools/check_biv150_konly_native_boundary_cpu.py
python3 -B -W error tools/check_biv150_konly_encoded_hoist_cpu.py
```

All three passed; their full JSON outputs are adjacent. The native checker reads the real immutable 22e ELF and uses CPU metadata mocks. The hoist checker replays the existing actual compiler artifact audit and CPU adapter protocol; it grants no new device or service qualification.

The actual immutable 94db owned-service helper was independently loaded via `importlib.util.spec_from_file_location` without a parent package, in an isolated `-I -S` subprocess that rejected framework imports. Its actual `expected_fingerprint` succeeded for encoded OFF, encoded ON/private native OFF, encoded ON/private native ON, and native ON/Q32 hoist ON. All fingerprints differ; its original official-105 protocol source is unchanged (`27937c1566e4b8680fc4feb2e18d6f12e0a4c157416ac9840fab6cc19b221bd2`). Stock W8A8 and stock NativeGQA were explicitly disabled for this source-only test because the new stock library is not built here. The real private K8 22e file was only hashed. `standalone-helper94db-cpu.json` records this narrower scope.

The older PR89 `check_biv150_konly_encoded_cpu.py` is **not a passing PR98 check**: it fails at line 178's original package-import-tail assertion both on untouched c30 and on this merge. The two original stderr files preserve the same failure; the test and guard are unchanged. The PR98 worker preflight adds an import absent from that legacy fixture. This inherited failure is not presented as a passing suite or a new merge regression.

## Dependencies and later acceptance

`runtime-pins-and-compatibility.json` records the complete installed module/helper pins, hoist SDK/helper/JIT/codec pins, private source hashes, the three incoming main file hashes, and all eight diagnostic Graph bootstrap pins. All eight still match the unchanged frozen bootstrap and the most recent captured actual vLLM sources. This source comparison preserves bootstrap compatibility; actual installation/capture/replay remains to be checked on the renewed target. Diagnostic wrappers stay OFF for formal throughput.

The private K8 operator remains `cjj_native_k8_qk_probe::produce`, using SO SHA `22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd`. It is distinct from the stock NativeGQA library. The old stock `aace693bdb302d2685ec1b5e11e859c8a4a77d7f45c0f4dacddc6b5e49b015b8` is **not** the new overlap-enabled library. Root must independently build the new stock library from 1cb and pass its actual absolute path and full SHA through the existing controller pair options. No new stock library SHA is claimed here. The controller's existing W8A8 gate library remains the separately pinned 5c library; this task does not replace it.

For an existing v2/v3 controller run, use this new exact source commit and explicitly set `--base 1cbac531ca9afaa0e9a68624d5d10cb1d0412055`, because its `git diff --exit-code BASE -- csrc` must compare against the new main, not old 4229. Keep the original fixed service and benchmark CLI, owner/cleanup gates, same tuning snapshot and runtime path/SHA checks. The original controller/helper files were not edited.

The old a975 throughput (6512.18/5012.05) and its separate 105 evaluation apply to a975. They do not qualify this new merged source. Official 105 and two-configuration throughput acceptance need this exact source, its new implementation fingerprint, and the actual new stock library. Official parallel-8 quality may execute private Triton fallbacks; it does not alone prove private native bucket coverage. Existing supplemental parallel-64/Graph coverage diagnostics remain separate, with quality coverage pending its independent evidence. No new precision, throughput, or model pass is asserted by this CPU merge.
