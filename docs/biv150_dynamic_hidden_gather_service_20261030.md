# PR122 natural service admission

This child starts at `a10145e173e27fd27c798452701b101c8cdcf69d`, which includes
main `899ce77470b0b539aebc108fd122f668c2d5894d`. It retains 5898 Native decode,
ebbc mixed Native, Q32, 5c W8A8 and 0c66 GQA. It does not include PR120 storage
or PR123/124 changes. No service, model-accuracy or throughput result is claimed.

The source bridge binds the actual a101 component summary `5919c98c…` and
independent review `dc59a27e…`. That component used real Tensor OFF/ON dispatch
and complete controlled-frame costs, including both ordinary and constructed
Plan cases. Its four event ratios were 1.13135, 1.12870, 1.08327 and 1.00364;
the large Plan case was approximately equal. It did not prove natural Plan,
model C3 or service behavior. No ordinary-only route is introduced here.

The GPU kernel, runner CRLF bytes and caller/event/scope/receipt checks are
unchanged from a101. Four reversible helper edits add an optional cold-state
observation dictionary, return the already launched CompiledKernel from the
private `_launch`, and accumulate two counters only after that call returns.
The counters distinguish the original prefix-minus-one index producer and
original compute-order Plan logits copy. At most four compiled objects are
retained; an explicit overflow invalidates complete observation. No histogram,
per-call serialization, profiler, tensor readback, new stream or synchronization
is added. Allocation/JIT/launch errors still propagate. These are **host submit
returns**, not GPU completion, token counts, request attribution or precision.

`CJJ_HIDDEN_GATHER_DIAGNOSTICS` defaults OFF and is fixed at helper cold setup.
The controller requires `--gather-diagnostics` for plain smoke, and prohibits it
for Math105 and throughput. Feature OFF is still the original gather fallback.
All diagnostic private keys and bootstrap paths are removed from probes/clients;
the feature ON environment is injected after the original controller clearing.
A separate gather-OFF fingerprint keeps both Native libraries ON and proves
C3 flag separation. No token enters manifests, command arguments or logs.

The new controller is an exact reversible copy of 118 controller `637d29e3…`:
five import/path identity sites plus five explicit observation seams. Original
67+96 smoke requests, service/benchmark CLI, e236 benchmark, 105 functions,
warmup, scoring and owner cleanup are untouched. Root approval for old c14 does
not qualify the new source; the new bridge starts with both approval flags false.
Its production proof permits only the exact five a101 gather files, with the
new helper reduced byte-for-byte to a101 by its diagnostic inverse.

Two fixed JSON snapshots surround the existing plain smoke. They use the normal
API-to-worker RPC path with a private in-memory token and no general RPC,
pickle, counter reset or device API. The actual WorkerFL/ModelRunnerFL process,
PID/start epoch, owned session, helper/runner/kernel bytes, both Native library
identities and encoded implementation are bound. The before snapshot can
legitimately have zero counts and no initialized gather runtime. The after
snapshot reads only compiled objects returned by real submissions and records
actual hash, ABI, options/resources, artifact SHA and loaded launcher SHA.
Unknown actual compilation is preserved in the ACK before rejection. Missing
ordinary or Plan submissions remain a coverage gap, not invented success.

Original PR119 files and contracts remain unchanged. The new narrow transport
has only `smoke-before` and `smoke-after`, a 128 KiB JSON limit and its own route,
method, flags and bootstrap. It cannot be enabled for performance. No finalizer
or FINAL_STATS is needed to close the observed interval; old missing lifetime
statistics are not rewritten.

CPU review uses `check_biv150_hidden_gather_service_cpu.py` for exact inverses,
actual post-launch statements, fixed RPC transport, bounded compiled identities,
epoch/reset/refusal cases and explicitly fake worker metadata over real saved
artifact bytes. `check_biv150_hidden_gather_run_preflight_cpu.py` invokes the
actual frozen controller using real 94db/four SO/v6 files and a temporary
CPU-only approval copy, stopping before the first `idle` implementation. These
checks do not start a server, inspect a GPU or grant Root approval. Root prepares
a fresh receipt with `biv150_hidden_gather_service_bridge.py`, then independently
approves and schedules smoke. Performance remains pending that natural entry.
