# Default-off compact/native service composition — qualification pending

This records the historical104 freeze. The later reviewed-component service
candidate is documented in `biv150_compact_native_service_a455_20261028.md`;
its model and real capacity gates remain pending.

Normal two-parent CPU integration of PR91 `5ca6a5464f9e8d11f30428c46508ff0b76e204de`
and PR98 `c30cf4e2f56b58efaca100ffa54157dd53744944`, both based on fresh team main
`4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b`. Depends on PR100
`51da6fd73e7fff1513081fb330323ce72c00856c` for the new compact decoder namespace,
compile/actual-SO review and consumer evidence, all still device pending. Fresh
compact Q32-hoist work is independent and not included or licensed here.

The six shared-file resolutions keep PR98's encoded runtime/service/source
manifest, PR91's compact spec/C3 selector and allocator installation, both worker
preflight/installation branches, and the union of compile-semantic identities.
Both sets of generated writer/decode/Q32/M2/raw3D/normalized2D GPU sources remain
byte identical to their respective parents. No CUDA, codec, math, sampler, batch
reorder, service parameter, official dependency or shared parent branch changes.

`VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_DECODE` is a new, default0 switch. The original
`VLLM_FL_BIV150_KONLY_NATIVE_DECODE` still means only the old16KiB/22e path and
rejects compact. The compact format and encoded format remain mutually exclusive;
old16KiB Q32-hoist remains forbidden for compact. Default OFF retains the current
compact-Triton and encoded-native behavior; source fingerprints appropriately
change for this newly composed checkout and do not inherit old device licenses.

The new stdlib `compact_native_qualification.py` has an explicit **pending** source
record. There is no invented SO SHA. Compact-native ON always raises a clear
pending error before any worker, Torch, target import or library load. The unique
conditional package guard precedes the first legacy Torch import; independent
compile-semantics loading, direct worker loading and compact installer also guard
the entry. This is not a None-equals-missing admission: pending rejects first;
future reviewed pins require a complete strict key set, four known64hex SHAs,
the exact PR100 CUDA source, namespace and four N64/S2 or N32/S4 pitch scopes,
then actual absolute-library bytes and the explicit env SHA pair. Enabling later
requires a new reviewed pin commit and refreshed source manifest, not an env-only
bypass. Even that component pin will not grant model/service/capacity acceptance.

The prepared compact adapter follows the existing98 host validation and current
stream dispatcher. It admits registered compact carrier `(6272,2176,256,128,1)`,
then uses the broker's real I8/page12544 and F32/page3136 fullbase roles and BF16
V+4352B/page6272 view. Its namespace is `cjj_native_k8_compact_probe::produce`.
It is inserted only after exact ownership/format selection, before private compact
Triton. Unsupported known compact geometry retains that private reader; unknown
or old-format aliases cannot reach stock native. Postlaunch exceptions never
retry a second producer. Adapter/op/library/code/qualification identity joins C3,
manifest and compile fingerprints. Since qualification is pending, this adapter's
enabled branch has not been loaded or executed. No service hook is activated by
the new switch in this freeze.

The original91 host already checks all42 specs, allocation plans, actual raw
buffers, reshape objects and forward bindings. This merge preserves those hooks
without replacing allocator math. CPU mocks do not prove real42-layer allocation,
new scheduler token capacity, originalBF16 precision, service throughput or memory
benefit. The service registry retains many valid per-layer entry generations;
this does not replace them with the small consumer's single-allocation lease.
N40 remains S16/private Triton and is not a newly qualified native scope.

CPU reproduction (stdlib; no Torch/Triton/NumPy import or target operation):

```sh
python3 -B tools/check_biv150_compact_native_service_cpu.py
```

It runs the original91 install/42-layer mock checker and original98 service/22e
ELF checker **without editing either**. Four fresh subprocesses independently
load `compile_semantics.py` with `spec_from_file_location` and no parent package:
stock, old encoded-native ON and compact TL ON generate fingerprints; pending
compact-native ON refuses. A separate full package-prefix subprocess proves ON
rejects before Torch/worker. Executing the actual unique guard with OFF imports
no new module and changes no binding; deleting that guard restores the entire91
`__init__.py` bytes exactly.

The unchanged original91 device CPU checker correctly refuses its literal old
bootstrap source pin after the new package guard. Its original stdout/stderr is
saved under `docs/evidence/compact-native-service-20261028/old91-device-cpu-refusal.*.txt`.
That refusal is preserved, not changed into a pass or a transferred device license.
The combined CPU report and all generated-source SHA checks are adjacent.

No SSH, target compilation, GPU, queue, service start or new library load was
performed. Actual new SO/SASS/IR/resources, compact consumer full O/M/L/final bits,
real42-layer startup, all-reader numerics/Graph, model105, capacity and throughput
remain unqualified. There is deliberately no new GPU launcher in this PR.


Root independently replayed the full CPU entry with bundled Python3.12 on tested source10da126. Exit0; report `docs/evidence/compact-native-service-20261028/root-python312-cpu.json` SHA4100668a5f5c98ec61e08f6215f2969713958e87e375258e28270723def24f55. All structured fields match the original report except the preserved old91 rejection traceback renderer; both end with `AssertionError: bootstrap source hash mismatch: vllm_fl`. Original traceback and Root traceback remain intact, not normalized over the originals.

Root also checked that encoded_native.py, encoded_runtime.py and encoded_service.py are byte-exact to parentc30cf4e, the stdlib preflight checks explicit pending before reading a missing SHA/path, and GPU sources remain exactly the two parents. This evidence-only commit does not enable the pending path, load an SO, claim a device run, borrow a model score or change any production file.
