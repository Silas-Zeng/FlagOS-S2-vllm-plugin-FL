# N48/S4 compact consumer: CPU preparation

This adds tools only on parent `83b6c8c9282445219782ee1377680778a2aa8d3a`.
The N48 C++/CUDA/codec, production, original consumers and thresholds are unchanged.
No device execution or service qualification is supplied by this commit.

The default frontend prints a stdlib-only plan. `--preflight-cpu` executes the
**same** actual-file/approval/codec prerequisites as `--gpu`, without importing
NumPy, Torch or Triton. The explicit device mode also requires Root's external
exclusive-owner acknowledgement and a new output directory. It does not infer
GPU idleness from an empty process table.

## Actual prerequisites

Candidate actual compile SO is
`2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1`.
Root's independent approval is the original JSON
`cjj-root-native112-n48-consumer-review-approved-20261029.json`, SHA
`14c8f26fa0d7afb7d00d480aad9ec5cf84f490d11cca8da8cb5f6a959a114a50`.
Candidate admission requires all six actual-SO flags, `manual_Root_approval=true`,
exact N48/S4 consumer scope, only S4 resources, and rehashes the compile/SO files.
Static S4 resources are 100 VGPR / 66 SGPR / 8192 shared / zero metadata spills
and private bytes. These are static metadata, not occupancy measurements.
The pending-root review remains rejected. The compiled unused S2 symbol provides
no S2 qualification.

The numeric control is the actual PR110 library
`7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4`,
its separately approved compile/SO review (`90f6e18e142813e732694f453779b876048daaa8c122fd3e2d985dc2969cdb17`),
and complete four-case consumer summary
`6f719fe84e83e547d0e42f7c2be5bdb408e64cdb107cc455994a681705e34c7c`.
The latter's complete artifact manifest is rehashed; completion, whole-field
bitwise, Graph/side, all six-phase hard gates and live-epoch closure are required.
The original actual codec receipt
`3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263`
is independently replayed in full by the existing codec gate. No receipts are
manufactured or approvals inferred from the compile success.

## Four cases and six checked phases

Cases have 41 or 42 live rows among 48, Q pitch 2048 or 2560, S4, 29 shared read-only
pages and lengths through 193 including 1/15/16/17/tails. Zero-length row positions
rotate with changed inputs. Queries, K, V and heads retain the original complete
nonperiodic D128 counter pattern. First48 row slices of a common64-row input
backing are identical; the control's second N32 call reads original rows32..63,
with rows48..63 zero-length, rather than regenerating row0 counter values.

The candidate dispatches one N48 producer and reducer. The control dispatches two
N32/S4 producers and two original reducers, stitching their original row slices.
**This double-launch control is solely a bitwise/numeric reference and cannot be
a performance denominator.** Later fullstep performance needs real compact TL48
with the same 41/42 rows, padding, page budget and writer.

Both arms use full64-row, width16 scratch narrowed to S4, preserving token/head
strides. All firstS4 regions start poisoned; unused segments4..15 have distinct
guard bits. The producer's early return for `lower>=length` must leave inactive
segments poisoned. Candidate rows48..63 remain untouched, including final output;
the control reducer's extra16 zero-length rows must return positive zero.
All comparisons across arms cover only the common first48 rows; each arm checks
its own complete backing, outside guards, unused segments and zero-length rows.

Checked phases are A-producer, A-after-reducer, Graph-B, fresh-eager-B-producer,
fresh-eager-B-after and side-stream-C. B changes Q, full cache bytes, table and
lengths at the same pointers. Poisoned Graph-B runs before fresh eager-B and must
equal every complete fresh-B input/output backing. Side C changes the same inputs.
OML is compared with the complete I64-QK/FP64 reference, using the already corrected
pagewise BF16-P partial algorithm. Ideal quantized final retains the original
`.005` pointwise/relative/RRMSE hard gates. Ideal partial and original BF16 proxy
remain separate diagnostics; a failing proxy is preserved. No writer/encoder or
timing is included.

The live cache is byte-validated before each process-local epoch lease. Its
128-byte nonzero layer offset is retained: K I8 has12544-byte page pitch, scale
F32 points to the **full page0 base**, and V BF16 is already offset by4352 bytes.
Native role binding never adds either offset twice. All full inputs/cache are
checked read-only. Two actual host rejection attempts (N48/S2 and N32/S4) require
their precise C++ errors and unchanged complete input/output bytes. Their pinned
guards precede `CUDAGuard` and native launch.

## CPU replay

From this worktree, with Python3.12 (`$PY` below):

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/probe_biv150_native_k8_compact_n48_consumer.py
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/check_biv150_native_k8_compact_n48_consumer_cpu.py --golden-cpu --out /private/tmp/n48-golden-NEW.json
```

The committed evidence contains the default plan, complete CPU golden report
(12 phase-fixtures, 48 scalar I64 controls, all first48 golden fields equal to
the64-row reference), and the real-file complete preflight. The preflight uses
the actual local112/110 compile directories, Root approval, all110 consumer
snapshots and the actual codec, and imports no framework.

## Root-only future device command

Use the actual installed Python environment and this exact commit's source.
All prerequisite directories, inspection files and snapshots must remain available.
The following command assumes Root exports its approved112 JSON to the shown path.
First substitute `--preflight-cpu` for `--gpu --root-owned-exclusive` and omit
`--out` to replay the identical prerequisites without touching the device.

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/probe_biv150_native_k8_compact_n48_consumer.py \
  --gpu --root-owned-exclusive \
  --candidate-compile-summary /workspace/experiments/minicpm-s2/cjj-compact-native-n48-compile-v1-20261029/compile-summary.json \
  --candidate-review /workspace/experiments/minicpm-s2/exports/cjj-root-native112-n48-consumer-review-approved-20261029.json \
  --candidate-review-sha256 14c8f26fa0d7afb7d00d480aad9ec5cf84f490d11cca8da8cb5f6a959a114a50 \
  --candidate-library /workspace/experiments/minicpm-s2/cjj-compact-native-n48-compile-v1-20261029/build/cjj_native_k8_compact_n48_s4_probe.so \
  --reference-compile-summary /workspace/experiments/minicpm-s2/cjj-native-k8-compact-v-overlap-compile-v1-20261029/compile-summary.json \
  --reference-review /workspace/experiments/minicpm-s2/exports/cjj-root-native110-consumer-review-approved-20261029.json \
  --reference-review-sha256 90f6e18e142813e732694f453779b876048daaa8c122fd3e2d985dc2969cdb17 \
  --reference-library /workspace/experiments/minicpm-s2/cjj-native-k8-compact-v-overlap-compile-v1-20261029/build/cjj_native_k8_compact_v_overlap_probe.so \
  --reference-library-sha256 7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4 \
  --reference-consumer-summary /workspace/experiments/minicpm-s2/cjj-native-k8-compact-v-overlap-consumer-v1-20261029/consumer-summary.json \
  --reference-consumer-summary-sha256 6f719fe84e83e547d0e42f7c2be5bdb408e64cdb107cc455994a681705e34c7c \
  --codec-summary /workspace/experiments/minicpm-s2/cjj-native-k8-codec-v1-20261028/codec-summary.json \
  --codec-proof-sha256 3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263 \
  --out /workspace/experiments/minicpm-s2/cjj-native112-n48-consumer-v1-20261029
```

The original approved control review's bytes and SHA must be retained. A fresh
output directory is required. Failures keep the original summary, traceback,
full before/after buffers and manifests and do not relax any gate. The next
device result alone can qualify this small consumer; it cannot establish
44769-page fullstep performance, actual service row frequency, capacity or model
quality. No GPU result is reported here.
