# Compact12544 service wiring on native-GQA main: CPU stage

This source stage merges main `b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd`
and reads PR89's immutable `d8ef055422d9d5c45049f85468b486b47b993406`.
It does not reuse the DAC/2487 device results. The old `cpu.json`,
`install-cpu-dac2487.json`, and DAC wiring report remain historical originals.

The fixed spec/manager modules, builtins-first registry, get-spec and reshape
hooks, unchanged allocator, exact C3 private receipt, six compact reader/writer
sources, and compile fingerprint are now wired into the default-off worker
switch. This is actual service-hook code exercised with explicit CPU framework,
Tensor and JIT substitutes. Actual allocation, device binding, IR, numerics,
model accuracy, capacity and performance are all pending.

## Native producer boundary

Main's `_launch_producer` tries native BF16 before the Triton producer. A nominal
BF16 K carrier contains private INT8 K bytes. Its compact stride6272 is not a
safety mechanism. The installed compact launcher checks strong storage ownership
before calling main's launcher, retaining the new PR89 guard method byte-for-byte
at AST level:

- Owned storage reaches the private facade. Exact key/value pair, dtype, shape,
  stride, offset and original profile guards apply there; an owned alias rejects.
- Unowned storage before the allocation epoch reaches the original main launcher,
  preserving its native/Triton decision and native runtime object.
- Unknown storage after the private epoch rejects; private cache cannot fall back
  into stock BF16 interpretation.

C3's private receipt checks the installed launcher/code, its original parent,
retained native object, private facades, host hook and allocation identities.
Replacing the launcher or native object invalidates the receipt. Original main
native Python/C++/CUDA sources, M2 and Q-fragment files are byte-exact b3; their
hashes are in `parent-d8-native-production.json`. Native feature/library-content
fingerprinting remains alongside the compact format fingerprint. The compact
Q32 I8-hoist switch remains unsupported and rejects before either installer.

## CPU replay

From this worktree, using the previously collected target source archives:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_compact_install_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_compact_bridge_cpu.py
```

The first output matches `install-cpu-b3-native.json` (SHA256
`2b3476cee487b6ca9114aa49bd25175d76edcb2ec8330ccd1ffcc6fecd3e1570`).
It replays actual target method bodies with mocks: 42 allocations ×2 pages ask
for1053696 bytes, private carriers use stride6272 and a4352-byte V offset, and the
UniProc string-method path preserves spec object identity without serialization.
The child-process test checks module export only; it is not an RPC serializer
claim. Pure FullAttention does not activate the Mamba zeroer.

A poison native stub accepts every dtype/stride. Cold unowned tensors still reach
it; owned private tensors, wrong owned strides/offsets and armed unknown tensors
never do. Native/launcher receipt mutants reject. Native library fingerprinting
uses a labeled non-executable synthetic file and detects content changes; no
native library is loaded. The second command retains the original phase-one
result exactly. Materialization uses `git show d8ef055:<path>` and retains the
six compact kernel expressions; only physical format addresses differ from
PR89 and new-device proof is required for that difference.

## Device boundary

No device/JIT command is authorized by this report. A future Root-owned validation
must first record actual 12544-byte raw allocation metadata, registered class
identity, all newly compiled source/options/IR and the format-safe producer
binding. Codec/canary/FP64/Graph/side and model/service checks remain required.
Neither the old 16KiB Q32 hoist license nor any DAC1 scores qualify this format.
No capacity or service-throughput benefit is claimed.
