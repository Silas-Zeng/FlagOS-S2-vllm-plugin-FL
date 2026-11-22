# PR127: one dense4K reader falsifier (Root scope bound; device not run)

Parent8393 and its two-pitch negative short-fixture result remain unchanged. This
is the single additional bandwidth screen requested by Root: N64/S2, Q pitch2048,
all64 lengths4096,16384 distinct active physical pages in the original34276-page
arena, table pitch8192. It is producer+original reducer, compactS2 scratch, with
neither append nor service/width16/16K/second-pitch coverage. A large working set
does not itself prove HBM traffic or bandwidth limitation.

Native libraries are the existing5898 BF16-V control and138cac packed-V8 variant;
no native compilation or GPU encoder is added. Actual sourcefcf, Root static1c9,
and the earlier independently reviewed numeric260414 evidence stay immutable.
Root independently read ead4107 and replayed CPU report d790, then issued the new
dense-only receipt SHA5824ff9fffa96fd0b8442f7b16f9deb5b60b1a87eb994c864a74ce27082bb8c9.
This receipt permits only a future single component falsifier after Root again
confirms a fresh serial launch. This round was **not run**: no new GPU job was
started after11:50 before the scheduled12:00 shutdown. Default execution remains
CPU-only with Root/device flags false. The old389-page permission cannot
authorize this fixture. Production and GPU source bytes are unchanged.

The never-launched ead version compared Python tuple-valued `cases` directly
against JSON lists and would refuse this valid receipt. The bound follow-up
normalizes only the expected scope through JSON, retaining complete strict dict
equality. Actual5824 CPU preflight passes; changed pitch, old arena and writer
scope still refuse even with CPU-test fault pins. The original CPU refusal is
retained locally; it was not a numeric failure or a GPU run.

The generator reuses original3eb `chunk_sources`, `codebook`, `encode_chunk` and
`fixed_table` seed20261028. Its finite V alphabet uses the original rational
RN32/RNE codec; control V is explicitly BF16_RNE(F32(I8code*F32scale)). New V
packing is CPU initialization outside all timing. The original fullstep golden
is used without editing its code: its last-token replacement is made idempotent
by setting reference new-K/V to the existing last mapped token, and verifying
existing encoded K/scales. Its page16 BF16-P partial, ideal quantized-QK final/M/L
and unchanged .005 metrics remain. The original unquantized BF16 Q/K/V proxy is
saved separately and may fail; it is not a model precision qualification.

CaptureA and eight initial steps are preparation only. A→B changes Q and rotates
each request's256-page order by1; physical cache bytes and every length4096 stay
unchanged. B is copied into the same GPU input allocations, O/M/L/final are
poisoned, and the8-step Graph is replayed/read back **before** a new eagerB step.
Both are checked over complete arrays against the original references and each
other; both libraries' full guarded O/M/L/final must be bitwise equal. This adds
no separate A or side-stream matrix; previous source-level tail/empty and side
evidence is inherited, not newly executed by this all-full-page case.

Timing reuses exactly the8393 capture/pair helpers and original3eb event/host
clock:128 warm Graph replays per arm,8 complete steps per replay,64 replays per
block, one control/packed/packed/control sequence (four blocks,512steps each).
Every block has complete post-timing same-arm snapshot equality, cache readonly
and guards outside the timed interval. No cache flush or tuning is introduced.

## Storage and transfer budget

Each guarded cache is561,578,240B; both GPU cache allocations total1,123,156,480B
(~1.046GiB), plus small input/scratch/output allocations. CPU generation/readback
temporaries can peak at several GiB; no Python list of all GPU cache bytes is
constructed. CPU initialization/copy/reference costs remain outside timing.

Successful raw output does **not** contain two562MB cache blobs. Like the original
fullstep protocol, each actual complete GPU readback is compared byte-for-byte
to the CPU cache, including both128B guards, then recorded as whole-cache SHA
and128-page chunk SHAs with the exact counter/codec/fixture source recipe.
An independent reviewer can regenerate the original bytes chunkwise. The first
bad chunk saves actual+expected bytes before refusing (at most4MiB extra).
All actual guarded O/M/L/final arrays, B query/table/length inputs and complete
FP64 arrays remain raw content-addressed files. Estimated successful output is
under40MiB, not a collection truncation limit. Root must inspect the final
manifest and choose a sufficient collector cap; if data exceeds it, retrieve
the same job with a larger scoped transport, never rerun or omit evidence.

The existing native SOs,3eb parent and small prior proof files are reused on the
personal instance; only this Git child and the future small Root dense receipt
need transport. Neither the old175MB numeric raw nor the reconstructed1.123GB
cache data needs re-uploading.

## CPU replay and default plan

```sh
/opt/homebrew/bin/python3 -B tools/check_biv150_native127_v8_dense_cpu.py \
  --parent-source /private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028 \
  --numpy-oracles
python3 -B -S tools/probe_biv150_native127_v8_dense_timing.py
```

To replay actual bound5824 preflight as well, add these CPU-check arguments:

```sh
  --dense-admission /private/tmp/cjj-root-native127-dense-reader-admission-20261030.json \
  --numeric-summary /private/tmp/cjj-native127-packed-v8-numeric-raw-20261030/out/v8-consumer-summary.json \
  --numeric-review /private/tmp/cjj-root-native127-v8-numeric-independent-review-20261030.json \
  --timing-admission /private/tmp/cjj-root-native127-timing-admission-20261030.json
```

The original tuple/list preflight refusal is preserved at
`/private/tmp/cjj-native127-dense-reader-ead-json-scope-refusal-20261030.txt`.

The CPU test allocates only two source pages, checks the finite V conversion
alphabet, exact scalar-source and original codec bytes, neutral last-KV
reference behavior, whole-cache mismatch rejection, source inverse and original
numeric/clock bodies. It does not generate the full arena or execute a device.
Future run uses the previous short timing CLI/evidence plus `--dense-admission`
and `--dense-admission-sha256`, explicit `--gpu --timing --root-owned-exclusive
--root-timing-approved`, and a fresh output directory. Scope binding is not
automatic launch permission, a numeric pass or a performance observation.

If this screen is nonpositive, stop this exact prototype. A positive result
only justifies considering a later complete component; capacity remains16KiB
per page and model/service/performance qualification remain false.
