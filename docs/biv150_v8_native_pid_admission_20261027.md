# Strict native-PID transition for one diagnostic job

Container PID is not the driver's native host PID. Root's supplied controlled
calibration recorded container623393/NSpid623393, but the sole native GPU PID
after torch.zeros(1)+sync was1362794. The native query was empty before that
initialization. Full ixsmi showed No running while idle, but only an empty
process-table frame once that context existed. No ixsmi text is fabricated or
PID inferred from os.getpid/NSpid. The original calibration JSON is retained
verbatim under `docs/evidence/v8-bias-decode-20261027/native-pid-admission-v1/`.

The new explicit GPU runner performs this protocol:

1. Before importing Torch/Triton or initializing CUDA, run the original strict
   preflight, require port9031 closed and the actual native PID query empty.
2. Capture this process's container PID/start ticks and the committed validator
   plus original a698 root identities. Mint a fresh in-memory transition token.
3. This process creates one tiny torch.zeros(1,device=cuda) allocation and
   synchronizes. Its allocation remains alive for the whole job. Require the
   same identity/roots and exactly one native PID; bind that observed host PID.
4. Before each old/A phase, requery and require exactly that same singleton,
   the same PID/start/roots/token, and the expected phase order. A token alone
   never admits a phase. Missing, changed, duplicate or extra native PID,
   unknown output, command error or stale identity permanently rejects the job.

The query is the actual root-verified CLI:

```sh
/usr/local/corex/bin/ixsmi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits
```

The original full-idle check remains in the initial preflight. Only the frozen
probe's repeated NoRunning assertion is replaced by this stricter owner check,
with an exact AST inverse. Its service-port check and all numerical, codec,
writer, unmapped-byte, finite, Graph/eager, byte-equivalence and timing code
stay unchanged. No process is killed or stopped on a mismatch.

`NEW_OUTPUT.admission.json` is written even when admission fails before numeric
output exists. It records attempted/confirmed initialization, process/root
identity, token digest, observed native PID lists, phases and errors. The same
receipt is included in byte-stage-runtime-binding.json when that output exists.
The tiny initialization is diagnostic setup outside every numeric/timing path;
a future GPU run does not pretend to have performed zero GPU work. Sidecar
creation and default CPU replay perform no CUDA initialization.

CPU replay:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_native_pid_admission_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_bias_byte_stage_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_v8_bias_byte_stage.py
```

The fixtures cover foreign/multiple/missing PIDs, namespace misidentification,
stale PID/start/root/token, unknown or failed query, port/preflight rejection,
phase reorder/reuse and permanent failure after a foreign PID appears. Their
initializer is a CPU object fixture; no Torch/Triton is imported or executed.

Future owner commands remain the two explicit sidecar/GPU commands in the v2
note. Regenerate the sidecar from this new clean validator commit: its recorded
file hashes now also include biv150_native_pid_admission.py. The original a698
sourceRoot, original out/sources/debug paths and failed d2775114 proof stay
unchanged; only steady4K is licensed. No SSH/GPU/JIT or queue was used here.
